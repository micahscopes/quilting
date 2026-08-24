//! Durable, causal replication for Hyperscape's protocol-v0.1 authored lane.
//!
//! This crate intentionally has no dependency on `hyperscope-app`,
//! `quilting-wasm`, browser APIs, cameras, frame clocks, renderer state, or GPU
//! resources. Its only admitted application value is [`AuthoredEnvelope`].
//! Ephemeral presence is therefore not representable at this boundary.

#![forbid(unsafe_code)]

use bincode::Options;
use futures::Stream;
use futures_signals::signal_vec::SignalVec;
use hhhs::{DagRead, DagSnapshot, Digest, EntryHash, LazyReach, Position, Reach};
use hhhs_reactive::{signal_vec_view, stream_view, Revision};
use hhhs_replica::{
    AdmissionPolicy, AdmittedAuthority, AsyncTransactionSink, DurableReplicaHost, Replica,
    ReplicaRecord,
};
use hhhs_store::{
    append_storage_transaction_log, decode_storage_transaction_log, empty_storage_transaction_log,
    history_root, MemoryStorage, ReplicaStorage, StorageError, StorageTransaction,
};
use hhhs_sync::{Refusal, RepairHost};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, MessageHeader,
    ProtocolVersion, WireTransform, CURRENT_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use uuid::Uuid;

/// Frozen application-domain prefix for durable payload schema 0.1.
pub const PAYLOAD_DOMAIN: &[u8] = b"hyperscape authored operation v0.1\0";
/// The only payload schema admitted by this crate.
pub const PAYLOAD_VERSION: ProtocolVersion = ProtocolVersion { major: 0, minor: 1 };
/// Strict upper bound for the complete application payload inside one HHHS
/// entry. This is substantially below HHHS's transport ceiling on purpose.
pub const MAX_AUTHORED_PAYLOAD_BYTES: usize = 1024 * 1024;
const PAYLOAD_HEADER_BYTES: usize = PAYLOAD_DOMAIN.len() + 2 + 2 + 2 + 2 + 16;
const MAX_AUTHORED_BODY_BYTES: u64 = (MAX_AUTHORED_PAYLOAD_BYTES - PAYLOAD_HEADER_BYTES) as u64;
const STATE_ROOT_DOMAIN: &[u8] = b"hyperscape materialized project state v0.1";

/// Stable identity of one replicated Hyperscape project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjectId(Uuid);

impl ProjectId {
    pub fn new(value: Uuid) -> Result<Self, AdapterError> {
        if value.is_nil() {
            Err(AdapterError::NilProjectId)
        } else {
            Ok(Self(value))
        }
    }

    pub fn from_u128(value: u128) -> Result<Self, AdapterError> {
        Self::new(Uuid::from_u128(value))
    }

    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("project ID must not be nil")]
    NilProjectId,
    #[error("durable payload domain is unsupported")]
    WrongDomain,
    #[error("durable payload version {0:?} is unsupported")]
    WrongPayloadVersion(ProtocolVersion),
    #[error("embedded protocol version {0:?} is unsupported")]
    WrongProtocolVersion(ProtocolVersion),
    #[error("durable payload belongs to project {actual}, expected {expected}")]
    WrongProject { expected: Uuid, actual: Uuid },
    #[error("durable payload is truncated or malformed")]
    MalformedPayload,
    #[error("durable payload is {actual} bytes, exceeding the {max}-byte limit")]
    PayloadTooLarge { actual: usize, max: usize },
    #[error("authored envelope is invalid: {0}")]
    InvalidEnvelope(String),
    #[error("replica admission failed: {0}")]
    Replica(#[from] hhhs_replica::ReplicaError),
    #[error("replica repair failed: {0}")]
    Repair(#[from] hhhs_replica::ReplicaRepairError),
    #[error("durable transaction log failed: {0}")]
    Storage(#[from] StorageError),
}

fn canonical_codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
}

fn payload_codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(MAX_AUTHORED_BODY_BYTES)
        .reject_trailing_bytes()
}

/// Codec-only mirror. The protocol enum is deliberately JSON-tagged, while a
/// durable binary schema needs an explicit, frozen discriminant layout.
#[derive(Serialize, Deserialize)]
struct FrozenEnvelope {
    header: MessageHeader,
    command: FrozenCommand,
}

#[derive(Serialize, Deserialize)]
enum FrozenCommand {
    UpsertAsset(FrozenAssetDescriptor),
    SetEntityTransform(EntityId, WireTransform),
    RemoveEntity(EntityId),
}

/// Binary-schema mirror without the protocol type's JSON-only
/// `skip_serializing_if` field policy. Positional codecs must encode both
/// option discriminants even when their values are absent.
#[derive(Serialize, Deserialize)]
struct FrozenAssetDescriptor {
    id: AssetId,
    uri: String,
    media_type: Option<String>,
    content_digest: Option<[u8; 32]>,
}

impl From<&AssetDescriptor> for FrozenAssetDescriptor {
    fn from(asset: &AssetDescriptor) -> Self {
        Self {
            id: asset.id,
            uri: asset.uri.clone(),
            media_type: asset.media_type.clone(),
            content_digest: asset.content_digest,
        }
    }
}

impl From<FrozenAssetDescriptor> for AssetDescriptor {
    fn from(asset: FrozenAssetDescriptor) -> Self {
        Self {
            id: asset.id,
            uri: asset.uri,
            media_type: asset.media_type,
            content_digest: asset.content_digest,
        }
    }
}

impl From<&AuthoredEnvelope> for FrozenEnvelope {
    fn from(envelope: &AuthoredEnvelope) -> Self {
        let command = match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => FrozenCommand::UpsertAsset(asset.into()),
            AuthoredCommand::SetEntityTransform { entity, transform } => {
                FrozenCommand::SetEntityTransform(*entity, *transform)
            }
            AuthoredCommand::RemoveEntity { entity } => FrozenCommand::RemoveEntity(*entity),
        };
        Self {
            header: envelope.header,
            command,
        }
    }
}

impl From<FrozenEnvelope> for AuthoredEnvelope {
    fn from(envelope: FrozenEnvelope) -> Self {
        let command = match envelope.command {
            FrozenCommand::UpsertAsset(asset) => AuthoredCommand::UpsertAsset {
                asset: asset.into(),
            },
            FrozenCommand::SetEntityTransform(entity, transform) => {
                AuthoredCommand::SetEntityTransform { entity, transform }
            }
            FrozenCommand::RemoveEntity(entity) => AuthoredCommand::RemoveEntity { entity },
        };
        Self {
            header: envelope.header,
            command,
        }
    }
}

/// Encode the frozen durable payload. The project and both version coordinates
/// are part of the authenticated HHHS entry payload, rather than ambient state.
pub fn encode_authored(
    project_id: ProjectId,
    envelope: &AuthoredEnvelope,
) -> Result<Vec<u8>, AdapterError> {
    envelope
        .validate()
        .map_err(|error| AdapterError::InvalidEnvelope(error.to_string()))?;
    let frozen = FrozenEnvelope::from(envelope);
    let body_len = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialized_size(&frozen)
        .map_err(|_| AdapterError::MalformedPayload)? as usize;
    let total_len =
        PAYLOAD_HEADER_BYTES
            .checked_add(body_len)
            .ok_or(AdapterError::PayloadTooLarge {
                actual: usize::MAX,
                max: MAX_AUTHORED_PAYLOAD_BYTES,
            })?;
    if total_len > MAX_AUTHORED_PAYLOAD_BYTES {
        return Err(AdapterError::PayloadTooLarge {
            actual: total_len,
            max: MAX_AUTHORED_PAYLOAD_BYTES,
        });
    }
    let body = payload_codec()
        .serialize(&frozen)
        .map_err(|_| AdapterError::MalformedPayload)?;
    let mut bytes = Vec::with_capacity(PAYLOAD_HEADER_BYTES + body.len());
    bytes.extend_from_slice(PAYLOAD_DOMAIN);
    bytes.extend_from_slice(&PAYLOAD_VERSION.major.to_le_bytes());
    bytes.extend_from_slice(&PAYLOAD_VERSION.minor.to_le_bytes());
    bytes.extend_from_slice(&CURRENT_PROTOCOL_VERSION.major.to_le_bytes());
    bytes.extend_from_slice(&CURRENT_PROTOCOL_VERSION.minor.to_le_bytes());
    bytes.extend_from_slice(project_id.as_uuid().as_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

/// Decode and validate one durable authored payload for an expected project.
pub fn decode_authored(
    expected_project: ProjectId,
    bytes: &[u8],
) -> Result<AuthoredEnvelope, AdapterError> {
    if bytes.len() > MAX_AUTHORED_PAYLOAD_BYTES {
        return Err(AdapterError::PayloadTooLarge {
            actual: bytes.len(),
            max: MAX_AUTHORED_PAYLOAD_BYTES,
        });
    }
    if bytes.len() < PAYLOAD_HEADER_BYTES {
        return Err(AdapterError::MalformedPayload);
    }
    if &bytes[..PAYLOAD_DOMAIN.len()] != PAYLOAD_DOMAIN {
        return Err(AdapterError::WrongDomain);
    }
    let mut cursor = PAYLOAD_DOMAIN.len();
    let payload_version = read_version(bytes, &mut cursor)?;
    if payload_version != PAYLOAD_VERSION {
        return Err(AdapterError::WrongPayloadVersion(payload_version));
    }
    let protocol_version = read_version(bytes, &mut cursor)?;
    if protocol_version != CURRENT_PROTOCOL_VERSION {
        return Err(AdapterError::WrongProtocolVersion(protocol_version));
    }
    let project_bytes: [u8; 16] = bytes
        .get(cursor..cursor + 16)
        .ok_or(AdapterError::MalformedPayload)?
        .try_into()
        .map_err(|_| AdapterError::MalformedPayload)?;
    cursor += 16;
    let actual_project = Uuid::from_bytes(project_bytes);
    if actual_project != expected_project.as_uuid() {
        return Err(AdapterError::WrongProject {
            expected: expected_project.as_uuid(),
            actual: actual_project,
        });
    }
    let envelope: FrozenEnvelope = payload_codec()
        .deserialize(&bytes[cursor..])
        .map_err(|_| AdapterError::MalformedPayload)?;
    let envelope = AuthoredEnvelope::from(envelope);
    envelope
        .validate()
        .map_err(|error| AdapterError::InvalidEnvelope(error.to_string()))?;
    Ok(envelope)
}

fn read_version(bytes: &[u8], cursor: &mut usize) -> Result<ProtocolVersion, AdapterError> {
    let major = u16::from_le_bytes(
        bytes
            .get(*cursor..*cursor + 2)
            .ok_or(AdapterError::MalformedPayload)?
            .try_into()
            .map_err(|_| AdapterError::MalformedPayload)?,
    );
    *cursor += 2;
    let minor = u16::from_le_bytes(
        bytes
            .get(*cursor..*cursor + 2)
            .ok_or(AdapterError::MalformedPayload)?
            .try_into()
            .map_err(|_| AdapterError::MalformedPayload)?,
    );
    *cursor += 2;
    Ok(ProtocolVersion { major, minor })
}

#[derive(Clone)]
struct AuthoredPolicy {
    project_id: ProjectId,
}

impl AdmissionPolicy for AuthoredPolicy {
    fn validate(
        &self,
        entry: &hhhs::Entry,
        _history: &DagSnapshot,
        _authority: &AdmittedAuthority,
    ) -> Result<(), String> {
        decode_authored(self.project_id, &entry.payload)
            .map(drop)
            .map_err(|error| error.to_string())
    }
}

/// One live value in the materialized authored scene.
#[derive(Debug, Clone, PartialEq)]
pub enum StateRow {
    Asset(AssetDescriptor),
    EntityTransform {
        entity: EntityId,
        transform: WireTransform,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateKey {
    Asset(AssetId),
    Entity(EntityId),
}

/// Deterministic authored state at one immutable HHHS horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectState {
    pub project_id: ProjectId,
    pub assets: BTreeMap<AssetId, AssetDescriptor>,
    pub entity_transforms: BTreeMap<EntityId, WireTransform>,
    pub history_root: [u8; 32],
    pub state_root: [u8; 32],
}

impl ProjectState {
    pub fn rows(&self) -> BTreeMap<StateKey, StateRow> {
        self.assets
            .iter()
            .map(|(id, asset)| (StateKey::Asset(*id), StateRow::Asset(asset.clone())))
            .chain(self.entity_transforms.iter().map(|(entity, transform)| {
                (
                    StateKey::Entity(*entity),
                    StateRow::EntityTransform {
                        entity: *entity,
                        transform: *transform,
                    },
                )
            }))
            .collect()
    }
}

#[derive(Clone)]
enum EntityValue {
    Transform(WireTransform),
    Removed,
}

/// Materialize authored state with HHHS register semantics: causal successors
/// replace observed writes; concurrent maxima use the maximum raw entry hash.
pub fn materialize_project(
    project_id: ProjectId,
    history: &DagSnapshot,
) -> Result<ProjectState, AdapterError> {
    // Production materialization must not retain ReachIndex's quadratic
    // transitive closure. LazyReach keeps O(N+E) adjacency and memoizes only
    // ancestry actually requested by these registers.
    let reach = LazyReach::new(history);
    let mut values = BTreeMap::<EntryHash, AuthoredCommand>::new();
    let mut asset_candidates = BTreeMap::<AssetId, BTreeSet<EntryHash>>::new();
    let mut entity_candidates = BTreeMap::<EntityId, BTreeSet<EntryHash>>::new();

    for entry in history.entries_topo() {
        let envelope = decode_authored(project_id, &entry.payload)?;
        let hash = entry.hash();
        match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => {
                asset_candidates.entry(asset.id).or_default().insert(hash);
            }
            AuthoredCommand::SetEntityTransform { entity, .. }
            | AuthoredCommand::RemoveEntity { entity } => {
                entity_candidates.entry(*entity).or_default().insert(hash);
            }
        }
        values.insert(hash, envelope.command);
    }

    let mut assets = BTreeMap::new();
    for (asset_id, candidates) in asset_candidates {
        let winner = reach
            .resolve(&candidates)
            .expect("a nonempty register has a causal maximum");
        let Some(AuthoredCommand::UpsertAsset { asset }) = values.get(&winner) else {
            unreachable!("asset registers contain only asset writes")
        };
        assets.insert(asset_id, asset.clone());
    }

    let mut entity_transforms = BTreeMap::new();
    for (entity, candidates) in entity_candidates {
        let winner = reach
            .resolve(&candidates)
            .expect("a nonempty register has a causal maximum");
        let entity_value = match values.get(&winner) {
            Some(AuthoredCommand::SetEntityTransform { transform, .. }) => {
                EntityValue::Transform(*transform)
            }
            Some(AuthoredCommand::RemoveEntity { .. }) => EntityValue::Removed,
            _ => unreachable!("entity registers contain only entity writes"),
        };
        if let EntityValue::Transform(transform) = entity_value {
            entity_transforms.insert(entity, transform);
        }
    }

    let history_digest = history_root(history);
    let state_root = state_root(project_id, &assets, &entity_transforms)?;
    Ok(ProjectState {
        project_id,
        assets,
        entity_transforms,
        history_root: *history_digest.as_bytes(),
        state_root,
    })
}

fn state_root(
    project_id: ProjectId,
    assets: &BTreeMap<AssetId, AssetDescriptor>,
    entities: &BTreeMap<EntityId, WireTransform>,
) -> Result<[u8; 32], AdapterError> {
    let canonical = canonical_codec()
        .serialize(&(project_id.as_uuid(), assets, entities))
        .map_err(|_| AdapterError::MalformedPayload)?;
    let mut bytes = Vec::with_capacity(STATE_ROOT_DOMAIN.len() + canonical.len());
    bytes.extend_from_slice(STATE_ROOT_DOMAIN);
    bytes.extend_from_slice(&canonical);
    Ok(*Digest::of(&bytes).as_bytes())
}

fn namespace(project_id: ProjectId) -> Digest {
    let mut bytes = Vec::with_capacity(PAYLOAD_DOMAIN.len() + 16);
    bytes.extend_from_slice(PAYLOAD_DOMAIN);
    bytes.extend_from_slice(project_id.as_uuid().as_bytes());
    Digest::of(&bytes)
}

/// In-memory durable transaction log used by the behavior-disconnected host.
/// Browser/IndexedDB storage can implement the same HHHS sink trait later.
#[derive(Debug, Clone)]
pub struct MemoryDurability {
    bytes: Vec<u8>,
    fail_next: bool,
}

impl Default for MemoryDurability {
    fn default() -> Self {
        Self {
            bytes: empty_storage_transaction_log(),
            fail_next: false,
        }
    }
}

impl MemoryDurability {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, AdapterError> {
        decode_storage_transaction_log(&bytes)?;
        Ok(Self {
            bytes,
            fail_next: false,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn fail_next_persist(&mut self) {
        self.fail_next = true;
    }
}

impl AsyncTransactionSink for MemoryDurability {
    type Error = String;

    async fn persist(&mut self, transaction: &StorageTransaction) -> Result<(), Self::Error> {
        if self.fail_next {
            self.fail_next = false;
            return Err("injected persistence failure".into());
        }
        let next = append_storage_transaction_log(&self.bytes, transaction)
            .map_err(|error| error.to_string())?;
        self.bytes = next;
        Ok(())
    }
}

/// Durable, single-writer project host. It owns authored history only; no
/// renderer or per-frame authority is hidden in this type. `D` is the host's
/// asynchronous persistence seam (for example an IndexedDB adapter); the
/// supplied sink must be the sole writer from prepare through finalization.
pub struct DurableProject<D = MemoryDurability> {
    project_id: ProjectId,
    storage: Arc<MemoryStorage>,
    host: DurableReplicaHost<MemoryStorage, AuthoredPolicy, D>,
}

/// Result of offering an unordered record batch. `deferred` names records whose
/// causal predecessors were absent; callers retain and resend those records
/// after obtaining their missing closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyRecordsReport {
    pub admitted: Vec<EntryHash>,
    pub refused: Vec<(EntryHash, Refusal)>,
    pub deferred: Vec<EntryHash>,
    pub lifted: usize,
}

impl DurableProject<MemoryDurability> {
    pub fn new(project_id: ProjectId) -> Result<Self, AdapterError> {
        Self::with_sink(project_id, MemoryDurability::default())
    }

    /// Recover from a locally trusted log previously returned by
    /// [`MemoryDurability::bytes`]. Untrusted peer input must enter through
    /// [`DurableProject::apply_records`] so normal policy admission runs.
    pub fn recover(project_id: ProjectId, bytes: Vec<u8>) -> Result<Self, AdapterError> {
        let durability = MemoryDurability::from_bytes(bytes)?;
        let transactions = decode_storage_transaction_log(durability.bytes())?;
        Self::with_replayed_sink(project_id, durability, transactions)
    }
}

impl<D> DurableProject<D> {
    /// Construct an empty durable project around an application-provided sink.
    /// A browser host can supply IndexedDB here without adding any browser API
    /// to this crate.
    pub fn with_sink(project_id: ProjectId, durability: D) -> Result<Self, AdapterError> {
        Self::with_replayed_sink(project_id, durability, Vec::new())
    }

    /// Recover a project from storage transactions read from this exact local
    /// durability sink.
    ///
    /// # Trust boundary
    ///
    /// `transactions` must come from locally trusted storage previously
    /// written by `durability`. This constructor validates the embedded
    /// Hyperscape payload schema and project identity, but deliberately replays
    /// storage transactions instead of re-running peer authority admission.
    /// Network, imported, or otherwise untrusted records must enter through
    /// [`DurableProject::apply_records`].
    ///
    /// The sink must already contain the supplied transactions in the same
    /// order. Supplying an empty or unrelated sink would make subsequent
    /// persistence unable to recover the complete project after another
    /// restart.
    pub fn recover_trusted_transactions(
        project_id: ProjectId,
        durability: D,
        transactions: Vec<StorageTransaction>,
    ) -> Result<Self, AdapterError> {
        Self::with_replayed_sink(project_id, durability, transactions)
    }

    fn with_replayed_sink(
        project_id: ProjectId,
        durability: D,
        transactions: Vec<StorageTransaction>,
    ) -> Result<Self, AdapterError> {
        let storage = Arc::new(MemoryStorage::new());
        for transaction in transactions {
            for entry in transaction.entries() {
                decode_authored(project_id, &entry.payload)?;
            }
            storage.commit(transaction)?;
        }
        let replica = Replica::builder(
            storage.as_ref().clone(),
            AuthoredPolicy { project_id },
            namespace(project_id),
        )
        .open()
        .build()?;
        let host = DurableReplicaHost::new(replica, durability);
        Ok(Self {
            project_id,
            storage,
            host,
        })
    }

    pub fn project_id(&self) -> ProjectId {
        self.project_id
    }

    pub fn durability(&self) -> &D {
        self.host.durability()
    }

    pub fn durability_mut(&mut self) -> &mut D {
        self.host.durability_mut()
    }

    pub fn state(&self) -> Result<ProjectState, AdapterError> {
        materialize_project(self.project_id, &self.storage.snapshot())
    }

    pub fn history_len(&self) -> usize {
        self.storage.snapshot().len()
    }

    pub fn state_stream(&self) -> impl Stream<Item = Revision<StateRow>> + 'static {
        stream_view(
            Arc::clone(&self.storage),
            trusted_state_view(self.project_id),
        )
    }

    pub fn state_signal_vec(&self) -> impl SignalVec<Item = StateRow> + 'static {
        signal_vec_view(
            Arc::clone(&self.storage),
            trusted_state_view(self.project_id),
        )
    }
}

impl<D> DurableProject<D>
where
    D: AsyncTransactionSink,
{
    pub async fn admit(
        &mut self,
        envelope: &AuthoredEnvelope,
    ) -> Result<ReplicaRecord, AdapterError> {
        let payload = encode_authored(self.project_id, envelope)?;
        let prepared = self.host.replica().prepare_open(payload)?;
        let commit = self.host.commit_prepared(prepared).await?;
        Ok(commit.replica_record().clone())
    }

    pub async fn apply_records(
        &mut self,
        records: &[ReplicaRecord],
    ) -> Result<ApplyRecordsReport, AdapterError> {
        // Carriers may deliver a child before its parent. RepairHost correctly
        // leaves that child pending, but this convenience boundary has the
        // complete batch and can deterministically lift its causal closure.
        let mut pending: BTreeMap<_, _> = records
            .iter()
            .cloned()
            .map(|record| (record.entry_hash(), record))
            .collect();
        let mut known: BTreeSet<_> = self.storage.all_hashes().into_iter().collect();
        let mut admitted = BTreeSet::new();
        let mut refused = Vec::new();
        let mut lifted = 0;

        loop {
            let ready: Vec<_> = pending
                .iter()
                .filter(|(_, record)| record.entry().header.prevs.0.is_subset(&known))
                .map(|(hash, record)| (*hash, record.encode()))
                .collect();
            if ready.is_empty() {
                break;
            }
            for (hash, _) in &ready {
                pending.remove(hash);
            }
            let report = RepairHost::apply(&mut self.host, &ready).await?;
            known.extend(report.admitted.iter().copied());
            admitted.extend(report.admitted);
            refused.extend(report.refused);
            lifted += report.lifted;
        }

        // Missing causal predecessors remain intentionally unannounced and are
        // returned explicitly so the carrier cannot accidentally discard them.
        Ok(ApplyRecordsReport {
            admitted: admitted.into_iter().collect(),
            refused,
            deferred: pending.into_keys().collect(),
            lifted,
        })
    }
}

fn trusted_state_view(
    project_id: ProjectId,
) -> impl Fn(&DagSnapshot, &Position) -> BTreeMap<StateKey, StateRow> + Clone {
    move |history, _at| {
        materialize_project(project_id, history)
            .expect("DurableProject admits only validated authored payloads")
            .rows()
    }
}
