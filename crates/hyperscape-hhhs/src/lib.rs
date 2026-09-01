//! Durable, causal replication for Hyperscape's versioned authored lane.
//!
//! This crate intentionally has no dependency on `hyperscope-app`,
//! `quilting-wasm`, browser APIs, cameras, frame clocks, renderer state, or GPU
//! resources. Its only admitted application value is [`AuthoredEnvelope`].
//! Ephemeral presence is therefore not representable at this boundary.

#![forbid(unsafe_code)]

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use bincode::Options;
use hhhs::{DagRead, DagSnapshot, Digest, EntryHash, LazyReach, Reach, SlicedDag};
use hhhs_replica::{
    AdmissionPolicy, AdmittedAuthority, AsyncTransactionSink, AuthorityInput, DurableReplicaHost,
    DurableReplicaHostError, Replica, ReplicaError, ReplicaRecord, ReplicaRepairError,
    ReplicaWireError, MAX_REPLICA_RECORD_BYTES,
};
use hhhs_store::{
    append_storage_transaction_log, decode_storage_transaction_log, empty_storage_transaction_log,
    history_root, MemoryStorage, ReplicaStorage, StorageError, StorageRecoveryState,
    StorageTransaction,
};
pub use hhhs_store::{ProjectionCheckpoint, ProjectionKey};
use hhhs_sync::{Refusal, RepairHost};
pub use hyperscape_protocol::ProjectId;
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, ConformalFrameId, EntityId,
    MessageHeader, ProtocolVersion, WireConformalGenerator, WireTransform,
    CURRENT_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Frozen application-domain prefix for durable payload schema 0.1.
pub const LEGACY_PAYLOAD_DOMAIN: &[u8] = b"hyperscape authored operation v0.1\0";
/// Current application-domain prefix for durable payload schema 0.2.
pub const PAYLOAD_DOMAIN: &[u8] = b"hyperscape authored operation v0.2\0";
pub const LEGACY_PAYLOAD_VERSION: ProtocolVersion = LEGACY_PROTOCOL_VERSION;
pub const PAYLOAD_VERSION: ProtocolVersion = CURRENT_PROTOCOL_VERSION;
/// Strict upper bound for the complete application payload inside one HHHS
/// entry. This is substantially below HHHS's transport ceiling on purpose.
pub const MAX_AUTHORED_PAYLOAD_BYTES: usize = 1024 * 1024;
const PAYLOAD_HEADER_BYTES: usize = PAYLOAD_DOMAIN.len() + 2 + 2 + 2 + 2 + 16;
const MAX_AUTHORED_BODY_BYTES: u64 = (MAX_AUTHORED_PAYLOAD_BYTES - PAYLOAD_HEADER_BYTES) as u64;
const LEGACY_STATE_ROOT_DOMAIN: &[u8] = b"hyperscape materialized project state v0.1";
const STATE_ROOT_DOMAIN: &[u8] = b"hyperscape materialized project state v0.2";
/// Frozen domain prefix for a portable, authority-rechecked project archive.
pub const PROJECT_ARCHIVE_DOMAIN: &[u8] = b"hyperscape hhhs project archive v0.1\0";
/// The only project archive schema understood by this crate.
pub const PROJECT_ARCHIVE_VERSION: ProtocolVersion = ProtocolVersion { major: 0, minor: 1 };
/// Strict aggregate bound applied before an imported archive is parsed.
pub const MAX_PROJECT_ARCHIVE_BYTES: usize = 512 * 1024 * 1024;
/// Defensive bound on the number of records declared by one archive.
pub const MAX_PROJECT_ARCHIVE_RECORDS: usize = 1_000_000;
/// Frozen JSON carrier schema for one public authored replica record.
pub const AUTHORED_RECORD_FRAME_VERSION: ProtocolVersion = ProtocolVersion { major: 0, minor: 1 };
pub const AUTHORED_RECORD_FRAME_LANE: &str = "authored_record";
/// Defensive bound including base64url expansion and JSON framing.
pub const MAX_AUTHORED_RECORD_FRAME_JSON_BYTES: usize =
    MAX_REPLICA_RECORD_BYTES.div_ceil(3) * 4 + 512;
const ARCHIVE_CHECKSUM_BYTES: usize = 32;
const ARCHIVE_FIXED_BODY_BYTES: usize = 4 + 16 + 8 + 32 + 32;
const MIN_PROJECT_ARCHIVE_BYTES: usize =
    PROJECT_ARCHIVE_DOMAIN.len() + ARCHIVE_FIXED_BODY_BYTES + ARCHIVE_CHECKSUM_BYTES;

/// Transport-neutral announcement of one already-authorized HHHS record.
///
/// This is distinct from a raw authored envelope proposal and from ephemeral
/// presence. Relays may carry the JSON opaquely; receivers must still run the
/// decoded [`ReplicaRecord`] through normal HHHS admission. The project field
/// is routing metadata and is rechecked against the record's authored payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredRecordFrame {
    project_id: ProjectId,
    record: ReplicaRecord,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredRecordFrameWire {
    lane: String,
    version: ProtocolVersion,
    project_id: Uuid,
    record_base64: String,
}

impl AuthoredRecordFrame {
    pub fn new(project_id: ProjectId, record: ReplicaRecord) -> Result<Self, RecordFrameError> {
        decode_authored(project_id, &record.entry().payload)?;
        Ok(Self { project_id, record })
    }

    pub fn project_id(&self) -> ProjectId {
        self.project_id
    }

    pub fn record(&self) -> &ReplicaRecord {
        &self.record
    }

    pub fn into_record(self) -> ReplicaRecord {
        self.record
    }

    pub fn encode_json(&self) -> Result<String, RecordFrameError> {
        let record = self.record.encode();
        let wire = AuthoredRecordFrameWire {
            lane: AUTHORED_RECORD_FRAME_LANE.to_owned(),
            version: AUTHORED_RECORD_FRAME_VERSION,
            project_id: self.project_id.as_uuid(),
            record_base64: URL_SAFE_NO_PAD.encode(record),
        };
        let json = serde_json::to_string(&wire)?;
        if json.len() > MAX_AUTHORED_RECORD_FRAME_JSON_BYTES {
            return Err(RecordFrameError::FrameTooLarge {
                actual: json.len(),
                max: MAX_AUTHORED_RECORD_FRAME_JSON_BYTES,
            });
        }
        Ok(json)
    }

    pub fn decode_json(json: &str) -> Result<Self, RecordFrameError> {
        if json.len() > MAX_AUTHORED_RECORD_FRAME_JSON_BYTES {
            return Err(RecordFrameError::FrameTooLarge {
                actual: json.len(),
                max: MAX_AUTHORED_RECORD_FRAME_JSON_BYTES,
            });
        }
        let wire: AuthoredRecordFrameWire = serde_json::from_str(json)?;
        if wire.lane != AUTHORED_RECORD_FRAME_LANE {
            return Err(RecordFrameError::WrongLane(wire.lane));
        }
        if wire.version != AUTHORED_RECORD_FRAME_VERSION {
            return Err(RecordFrameError::WrongVersion(wire.version));
        }
        let project_id = ProjectId::new(wire.project_id)
            .map_err(|_| RecordFrameError::Adapter(AdapterError::NilProjectId))?;
        let bytes = URL_SAFE_NO_PAD
            .decode(wire.record_base64.as_bytes())
            .map_err(|_| RecordFrameError::InvalidBase64)?;
        if URL_SAFE_NO_PAD.encode(&bytes) != wire.record_base64 {
            return Err(RecordFrameError::NonCanonicalBase64);
        }
        if bytes.len() > MAX_REPLICA_RECORD_BYTES {
            return Err(RecordFrameError::RecordTooLarge {
                actual: bytes.len(),
                max: MAX_REPLICA_RECORD_BYTES,
            });
        }
        let record = ReplicaRecord::decode(&bytes)?;
        Self::new(project_id, record)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecordFrameError {
    #[error("authored record frame is invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("authored record frame lane {0:?} is unsupported")]
    WrongLane(String),
    #[error("authored record frame version {0:?} is unsupported")]
    WrongVersion(ProtocolVersion),
    #[error("authored record frame uses invalid base64url")]
    InvalidBase64,
    #[error("authored record frame base64url is not canonical unpadded encoding")]
    NonCanonicalBase64,
    #[error("authored record is {actual} bytes, exceeding the {max}-byte limit")]
    RecordTooLarge { actual: usize, max: usize },
    #[error("authored record frame is {actual} bytes, exceeding the {max}-byte limit")]
    FrameTooLarge { actual: usize, max: usize },
    #[error(transparent)]
    Record(#[from] ReplicaWireError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
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
    #[error("durable replica host initialization failed: {0}")]
    DurableHost(#[from] DurableReplicaHostError),
    #[error("durable transaction log failed: {0}")]
    Storage(#[from] StorageError),
    #[error("project archive domain is unsupported")]
    WrongArchiveDomain,
    #[error("project archive version {0:?} is unsupported")]
    WrongArchiveVersion(ProtocolVersion),
    #[error("project archive is truncated or malformed")]
    MalformedArchive,
    #[error("project archive is {actual} bytes, exceeding the {max}-byte limit")]
    ArchiveTooLarge { actual: usize, max: usize },
    #[error("project archive declares {actual} records, exceeding the {max}-record limit")]
    TooManyArchiveRecords { actual: usize, max: usize },
    #[error("project archive record is {actual} bytes, exceeding the {max}-byte limit")]
    ArchiveRecordTooLarge { actual: usize, max: usize },
    #[error("project archive checksum does not match its contents")]
    ArchiveChecksumMismatch,
    #[error("project archive contains an invalid HHHS replica record: {0}")]
    ArchiveRecord(#[from] ReplicaWireError),
    #[error("project archive record {0:?} does not use the project's open authority profile")]
    UnsupportedArchiveAuthority(EntryHash),
    #[error("project archive repeats record {0:?}")]
    DuplicateArchiveRecord(EntryHash),
    #[error("project archive record {entry:?} is missing predecessor {predecessor:?}")]
    MissingArchivePredecessor {
        entry: EntryHash,
        predecessor: EntryHash,
    },
    #[error("project archive records are not in canonical topological order")]
    NonCanonicalArchiveOrder,
    #[error("project archive belongs to project {actual}, expected {expected}")]
    WrongArchiveProject { expected: Uuid, actual: Uuid },
    #[error("local project has {local_only} history records absent from the imported archive")]
    ArchiveLocalDivergence { local_only: usize },
    #[error("project archive import refused {refused} records and deferred {deferred}")]
    IncompleteArchiveAdmission { refused: usize, deferred: usize },
    #[error("project archive expected {expected} history entries but imported {actual}")]
    ArchiveHistoryLengthMismatch { expected: usize, actual: usize },
    #[error("project archive history root does not match the admitted history")]
    ArchiveHistoryRootMismatch,
    #[error("project archive state root does not match the materialized project")]
    ArchiveStateRootMismatch,
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

fn encode_frozen_body(value: &impl Serialize) -> Result<Vec<u8>, AdapterError> {
    let body_len = canonical_codec()
        .serialized_size(value)
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
    payload_codec()
        .serialize(value)
        .map_err(|_| AdapterError::MalformedPayload)
}

/// Codec-only mirror. The protocol enum is deliberately JSON-tagged, while a
/// durable binary schema needs an explicit, frozen discriminant layout.
#[derive(Serialize, Deserialize)]
struct FrozenEnvelopeV01 {
    header: MessageHeader,
    command: FrozenCommandV01,
}

#[derive(Serialize, Deserialize)]
enum FrozenCommandV01 {
    UpsertAsset(FrozenAssetDescriptor),
    SetEntityTransform(EntityId, WireTransform),
    RemoveEntity(EntityId),
}

#[derive(Serialize, Deserialize)]
struct FrozenEnvelopeV02 {
    header: MessageHeader,
    command: FrozenCommandV02,
}

#[derive(Serialize, Deserialize)]
enum FrozenCommandV02 {
    UpsertAsset(FrozenAssetDescriptor),
    SetEntityTransform(EntityId, WireTransform),
    RemoveEntity(EntityId),
    SetConformalFrameTransform(ConformalFrameId, Vec<FrozenConformalGenerator>),
}

/// Positional-codec mirror of Quilting's JSON-tagged generator enum. Variant
/// order is frozen independently of serde's self-describing JSON tags.
#[derive(Serialize, Deserialize)]
enum FrozenConformalGenerator {
    Translation([f64; 3]),
    Rotation([f64; 4]),
    UniformScale(f64),
    SphereReflection([f64; 3], f64),
}

impl From<&WireConformalGenerator> for FrozenConformalGenerator {
    fn from(generator: &WireConformalGenerator) -> Self {
        match *generator {
            WireConformalGenerator::Translation { offset } => Self::Translation(offset),
            WireConformalGenerator::Rotation { quaternion_wxyz } => Self::Rotation(quaternion_wxyz),
            WireConformalGenerator::UniformScale { factor } => Self::UniformScale(factor),
            WireConformalGenerator::SphereReflection { center, radius } => {
                Self::SphereReflection(center, radius)
            }
        }
    }
}

impl From<FrozenConformalGenerator> for WireConformalGenerator {
    fn from(generator: FrozenConformalGenerator) -> Self {
        match generator {
            FrozenConformalGenerator::Translation(offset) => Self::Translation { offset },
            FrozenConformalGenerator::Rotation(quaternion_wxyz) => {
                Self::Rotation { quaternion_wxyz }
            }
            FrozenConformalGenerator::UniformScale(factor) => Self::UniformScale { factor },
            FrozenConformalGenerator::SphereReflection(center, radius) => {
                Self::SphereReflection { center, radius }
            }
        }
    }
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

impl TryFrom<&AuthoredEnvelope> for FrozenEnvelopeV01 {
    type Error = AdapterError;

    fn try_from(envelope: &AuthoredEnvelope) -> Result<Self, Self::Error> {
        let command = match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => FrozenCommandV01::UpsertAsset(asset.into()),
            AuthoredCommand::SetEntityTransform { entity, transform } => {
                FrozenCommandV01::SetEntityTransform(*entity, *transform)
            }
            AuthoredCommand::RemoveEntity { entity } => FrozenCommandV01::RemoveEntity(*entity),
            AuthoredCommand::SetConformalFrameTransform { .. } => {
                return Err(AdapterError::WrongProtocolVersion(envelope.header.version));
            }
        };
        Ok(Self {
            header: envelope.header,
            command,
        })
    }
}

impl From<FrozenEnvelopeV01> for AuthoredEnvelope {
    fn from(envelope: FrozenEnvelopeV01) -> Self {
        let command = match envelope.command {
            FrozenCommandV01::UpsertAsset(asset) => AuthoredCommand::UpsertAsset {
                asset: asset.into(),
            },
            FrozenCommandV01::SetEntityTransform(entity, transform) => {
                AuthoredCommand::SetEntityTransform { entity, transform }
            }
            FrozenCommandV01::RemoveEntity(entity) => AuthoredCommand::RemoveEntity { entity },
        };
        Self {
            header: envelope.header,
            command,
        }
    }
}

impl From<&AuthoredEnvelope> for FrozenEnvelopeV02 {
    fn from(envelope: &AuthoredEnvelope) -> Self {
        let command = match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => FrozenCommandV02::UpsertAsset(asset.into()),
            AuthoredCommand::SetEntityTransform { entity, transform } => {
                FrozenCommandV02::SetEntityTransform(*entity, *transform)
            }
            AuthoredCommand::RemoveEntity { entity } => FrozenCommandV02::RemoveEntity(*entity),
            AuthoredCommand::SetConformalFrameTransform { frame, generators } => {
                FrozenCommandV02::SetConformalFrameTransform(
                    *frame,
                    generators.iter().map(Into::into).collect(),
                )
            }
        };
        Self {
            header: envelope.header,
            command,
        }
    }
}

impl From<FrozenEnvelopeV02> for AuthoredEnvelope {
    fn from(envelope: FrozenEnvelopeV02) -> Self {
        let command = match envelope.command {
            FrozenCommandV02::UpsertAsset(asset) => AuthoredCommand::UpsertAsset {
                asset: asset.into(),
            },
            FrozenCommandV02::SetEntityTransform(entity, transform) => {
                AuthoredCommand::SetEntityTransform { entity, transform }
            }
            FrozenCommandV02::RemoveEntity(entity) => AuthoredCommand::RemoveEntity { entity },
            FrozenCommandV02::SetConformalFrameTransform(frame, generators) => {
                AuthoredCommand::SetConformalFrameTransform {
                    frame,
                    generators: generators.into_iter().map(Into::into).collect(),
                }
            }
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
    let (domain, payload_version, body) = match envelope.header.version {
        LEGACY_PROTOCOL_VERSION => (
            LEGACY_PAYLOAD_DOMAIN,
            LEGACY_PAYLOAD_VERSION,
            encode_frozen_body(&FrozenEnvelopeV01::try_from(envelope)?)?,
        ),
        CURRENT_PROTOCOL_VERSION => (
            PAYLOAD_DOMAIN,
            PAYLOAD_VERSION,
            encode_frozen_body(&FrozenEnvelopeV02::from(envelope))?,
        ),
        version => return Err(AdapterError::WrongProtocolVersion(version)),
    };
    let mut bytes = Vec::with_capacity(PAYLOAD_HEADER_BYTES + body.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&payload_version.major.to_le_bytes());
    bytes.extend_from_slice(&payload_version.minor.to_le_bytes());
    bytes.extend_from_slice(&envelope.header.version.major.to_le_bytes());
    bytes.extend_from_slice(&envelope.header.version.minor.to_le_bytes());
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
    let domain = &bytes[..PAYLOAD_DOMAIN.len()];
    let expected_version = if domain == PAYLOAD_DOMAIN {
        PAYLOAD_VERSION
    } else if domain == LEGACY_PAYLOAD_DOMAIN {
        LEGACY_PAYLOAD_VERSION
    } else {
        return Err(AdapterError::WrongDomain);
    };
    let mut cursor = PAYLOAD_DOMAIN.len();
    let payload_version = read_version(bytes, &mut cursor)?;
    if payload_version != expected_version {
        return Err(AdapterError::WrongPayloadVersion(payload_version));
    }
    let protocol_version = read_version(bytes, &mut cursor)?;
    if protocol_version != expected_version {
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
    let envelope = if protocol_version == LEGACY_PROTOCOL_VERSION {
        AuthoredEnvelope::from(
            payload_codec()
                .deserialize::<FrozenEnvelopeV01>(&bytes[cursor..])
                .map_err(|_| AdapterError::MalformedPayload)?,
        )
    } else {
        AuthoredEnvelope::from(
            payload_codec()
                .deserialize::<FrozenEnvelopeV02>(&bytes[cursor..])
                .map_err(|_| AdapterError::MalformedPayload)?,
        )
    };
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

/// A complete, portable authored-history snapshot.
///
/// Unlike [`MemoryDurability`], this is an interchange format. It carries
/// public [`ReplicaRecord`] values rather than storage transactions, so import
/// can run the ordinary authority, payload, and causal-repair path again. The
/// checksum detects accidental corruption; it is not a signature and conveys
/// no author identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectArchive {
    project_id: ProjectId,
    history_root: [u8; 32],
    state_root: [u8; 32],
    records: Vec<ReplicaRecord>,
}

impl ProjectArchive {
    pub fn project_id(&self) -> ProjectId {
        self.project_id
    }

    pub fn history_root(&self) -> [u8; 32] {
        self.history_root
    }

    pub fn state_root(&self) -> [u8; 32] {
        self.state_root
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Decode and structurally validate an untrusted archive without admitting
    /// it. Call [`DurableProject::import_archive`] to re-run HHHS admission.
    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        if bytes.len() > MAX_PROJECT_ARCHIVE_BYTES {
            return Err(AdapterError::ArchiveTooLarge {
                actual: bytes.len(),
                max: MAX_PROJECT_ARCHIVE_BYTES,
            });
        }
        if bytes.len() < MIN_PROJECT_ARCHIVE_BYTES {
            return Err(AdapterError::MalformedArchive);
        }
        if &bytes[..PROJECT_ARCHIVE_DOMAIN.len()] != PROJECT_ARCHIVE_DOMAIN {
            return Err(AdapterError::WrongArchiveDomain);
        }

        let checksum_offset = bytes.len() - ARCHIVE_CHECKSUM_BYTES;
        let expected_checksum = Digest::of(&bytes[..checksum_offset]);
        if expected_checksum.as_bytes() != &bytes[checksum_offset..] {
            return Err(AdapterError::ArchiveChecksumMismatch);
        }

        let mut reader = ArchiveReader::new(&bytes[PROJECT_ARCHIVE_DOMAIN.len()..checksum_offset]);
        let version = ProtocolVersion {
            major: reader.u16()?,
            minor: reader.u16()?,
        };
        if version != PROJECT_ARCHIVE_VERSION {
            return Err(AdapterError::WrongArchiveVersion(version));
        }
        let project_id = ProjectId::new(Uuid::from_bytes(reader.array::<16>()?))
            .map_err(|_| AdapterError::NilProjectId)?;
        let record_count =
            usize::try_from(reader.u64()?).map_err(|_| AdapterError::TooManyArchiveRecords {
                actual: usize::MAX,
                max: MAX_PROJECT_ARCHIVE_RECORDS,
            })?;
        if record_count > MAX_PROJECT_ARCHIVE_RECORDS {
            return Err(AdapterError::TooManyArchiveRecords {
                actual: record_count,
                max: MAX_PROJECT_ARCHIVE_RECORDS,
            });
        }
        let history_root = reader.array::<32>()?;
        let state_root = reader.array::<32>()?;
        // Every record needs at least its u64 length prefix. Reject impossible
        // declarations before reserving attacker-controlled capacity.
        if record_count > reader.remaining() / 8 {
            return Err(AdapterError::MalformedArchive);
        }
        let mut records = Vec::with_capacity(record_count);
        let mut hashes = BTreeSet::new();
        for _ in 0..record_count {
            let record_len = usize::try_from(reader.u64()?).map_err(|_| {
                AdapterError::ArchiveRecordTooLarge {
                    actual: usize::MAX,
                    max: MAX_REPLICA_RECORD_BYTES,
                }
            })?;
            if record_len > MAX_REPLICA_RECORD_BYTES {
                return Err(AdapterError::ArchiveRecordTooLarge {
                    actual: record_len,
                    max: MAX_REPLICA_RECORD_BYTES,
                });
            }
            let record = ReplicaRecord::decode(reader.take(record_len)?)?;
            let hash = record.entry_hash();
            if record.authority() != &AuthorityInput::Open {
                return Err(AdapterError::UnsupportedArchiveAuthority(hash));
            }
            if !hashes.insert(hash) {
                return Err(AdapterError::DuplicateArchiveRecord(hash));
            }
            decode_authored(project_id, &record.entry().payload)?;
            records.push(record);
        }
        if !reader.is_empty() {
            return Err(AdapterError::MalformedArchive);
        }
        for record in &records {
            for predecessor in &record.entry().header.prevs.0 {
                if !hashes.contains(predecessor) {
                    return Err(AdapterError::MissingArchivePredecessor {
                        entry: record.entry_hash(),
                        predecessor: *predecessor,
                    });
                }
            }
        }
        let canonical_hashes: Vec<_> =
            DagSnapshot::from_entries(records.iter().map(|record| record.entry().clone()))
                .entries_topo()
                .into_iter()
                .map(|entry| entry.hash())
                .collect();
        let encoded_hashes: Vec<_> = records.iter().map(ReplicaRecord::entry_hash).collect();
        if canonical_hashes != encoded_hashes {
            return Err(AdapterError::NonCanonicalArchiveOrder);
        }

        Ok(Self {
            project_id,
            history_root,
            state_root,
            records,
        })
    }

    /// Re-encode a validated archive in its canonical deterministic order.
    pub fn encode(&self) -> Result<Vec<u8>, AdapterError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PROJECT_ARCHIVE_DOMAIN);
        bytes.extend_from_slice(&PROJECT_ARCHIVE_VERSION.major.to_le_bytes());
        bytes.extend_from_slice(&PROJECT_ARCHIVE_VERSION.minor.to_le_bytes());
        bytes.extend_from_slice(self.project_id.as_uuid().as_bytes());
        bytes.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.history_root);
        bytes.extend_from_slice(&self.state_root);
        for record in &self.records {
            let encoded = record.encode();
            if encoded.len() > MAX_REPLICA_RECORD_BYTES {
                return Err(AdapterError::ArchiveRecordTooLarge {
                    actual: encoded.len(),
                    max: MAX_REPLICA_RECORD_BYTES,
                });
            }
            let next_len = bytes
                .len()
                .checked_add(8)
                .and_then(|length| length.checked_add(encoded.len()))
                .and_then(|length| length.checked_add(ARCHIVE_CHECKSUM_BYTES))
                .ok_or(AdapterError::ArchiveTooLarge {
                    actual: usize::MAX,
                    max: MAX_PROJECT_ARCHIVE_BYTES,
                })?;
            if next_len > MAX_PROJECT_ARCHIVE_BYTES {
                return Err(AdapterError::ArchiveTooLarge {
                    actual: next_len,
                    max: MAX_PROJECT_ARCHIVE_BYTES,
                });
            }
            bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&encoded);
        }
        let final_len = bytes.len().checked_add(ARCHIVE_CHECKSUM_BYTES).ok_or(
            AdapterError::ArchiveTooLarge {
                actual: usize::MAX,
                max: MAX_PROJECT_ARCHIVE_BYTES,
            },
        )?;
        if final_len > MAX_PROJECT_ARCHIVE_BYTES {
            return Err(AdapterError::ArchiveTooLarge {
                actual: final_len,
                max: MAX_PROJECT_ARCHIVE_BYTES,
            });
        }
        let checksum = Digest::of(&bytes);
        bytes.extend_from_slice(checksum.as_bytes());
        Ok(bytes)
    }
}

struct ArchiveReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ArchiveReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], AdapterError> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(AdapterError::MalformedArchive)?;
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], AdapterError> {
        self.take(N)?
            .try_into()
            .map_err(|_| AdapterError::MalformedArchive)
    }

    fn u16(&mut self) -> Result<u16, AdapterError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, AdapterError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
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
    ConformalFrameTransform {
        frame: ConformalFrameId,
        generators: Vec<WireConformalGenerator>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateKey {
    Asset(AssetId),
    Entity(EntityId),
    ConformalFrame(ConformalFrameId),
}

/// Deterministic authored state at one immutable HHHS horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectState {
    pub project_id: ProjectId,
    pub assets: BTreeMap<AssetId, AssetDescriptor>,
    pub entity_transforms: BTreeMap<EntityId, WireTransform>,
    pub conformal_frame_transforms: BTreeMap<ConformalFrameId, Vec<WireConformalGenerator>>,
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
            .chain(
                self.conformal_frame_transforms
                    .iter()
                    .map(|(frame, generators)| {
                        (
                            StateKey::ConformalFrame(*frame),
                            StateRow::ConformalFrameTransform {
                                frame: *frame,
                                generators: generators.clone(),
                            },
                        )
                    }),
            )
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
    let mut frame_candidates = BTreeMap::<ConformalFrameId, BTreeSet<EntryHash>>::new();
    let mut uses_protocol_v02 = false;

    for entry in history.entries_topo() {
        let envelope = decode_authored(project_id, &entry.payload)?;
        uses_protocol_v02 |= envelope.header.version == CURRENT_PROTOCOL_VERSION;
        let hash = entry.hash();
        match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => {
                asset_candidates.entry(asset.id).or_default().insert(hash);
            }
            AuthoredCommand::SetEntityTransform { entity, .. }
            | AuthoredCommand::RemoveEntity { entity } => {
                entity_candidates.entry(*entity).or_default().insert(hash);
            }
            AuthoredCommand::SetConformalFrameTransform { frame, .. } => {
                frame_candidates.entry(*frame).or_default().insert(hash);
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

    let mut conformal_frame_transforms = BTreeMap::new();
    for (frame, candidates) in frame_candidates {
        let winner = reach
            .resolve(&candidates)
            .expect("a nonempty register has a causal maximum");
        let Some(AuthoredCommand::SetConformalFrameTransform { generators, .. }) =
            values.get(&winner)
        else {
            unreachable!("conformal-frame registers contain only frame writes")
        };
        conformal_frame_transforms.insert(frame, generators.clone());
    }

    let history_digest = history_root(history);
    let state_root = state_root(
        project_id,
        &assets,
        &entity_transforms,
        &conformal_frame_transforms,
        uses_protocol_v02,
    )?;
    Ok(ProjectState {
        project_id,
        assets,
        entity_transforms,
        conformal_frame_transforms,
        history_root: *history_digest.as_bytes(),
        state_root,
    })
}

fn state_root(
    project_id: ProjectId,
    assets: &BTreeMap<AssetId, AssetDescriptor>,
    entities: &BTreeMap<EntityId, WireTransform>,
    frames: &BTreeMap<ConformalFrameId, Vec<WireConformalGenerator>>,
    uses_protocol_v02: bool,
) -> Result<[u8; 32], AdapterError> {
    let (domain, canonical) = if uses_protocol_v02 {
        (
            STATE_ROOT_DOMAIN,
            canonical_codec()
                .serialize(&(project_id.as_uuid(), assets, entities, frames))
                .map_err(|_| AdapterError::MalformedPayload)?,
        )
    } else {
        (
            LEGACY_STATE_ROOT_DOMAIN,
            canonical_codec()
                .serialize(&(project_id.as_uuid(), assets, entities))
                .map_err(|_| AdapterError::MalformedPayload)?,
        )
    };
    let mut bytes = Vec::with_capacity(domain.len() + canonical.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&canonical);
    Ok(*Digest::of(&bytes).as_bytes())
}

fn namespace(project_id: ProjectId) -> Digest {
    // Replica identity is stable across authored payload-schema revisions.
    // The original 0.1 domain is therefore a permanent namespace salt, not a
    // claim that every entry body uses schema 0.1.
    let mut bytes = Vec::with_capacity(LEGACY_PAYLOAD_DOMAIN.len() + 16);
    bytes.extend_from_slice(LEGACY_PAYLOAD_DOMAIN);
    bytes.extend_from_slice(project_id.as_uuid().as_bytes());
    Digest::of(&bytes)
}

#[derive(Debug)]
struct MemoryDurabilityState {
    bytes: Vec<u8>,
    recovered_state: StorageRecoveryState,
    fail_next: bool,
}

/// Non-writing observation and failure-injection handle for the in-memory
/// durability sink. Cloning this value cannot persist a transaction or advance
/// HHHS recovery state, so it does not create a second writer around the
/// exclusive durable host.
#[derive(Debug, Clone)]
pub struct MemoryDurabilityControl {
    state: Arc<Mutex<MemoryDurabilityState>>,
}

impl MemoryDurabilityControl {
    pub fn bytes(&self) -> Vec<u8> {
        self.state
            .lock()
            .expect("memory durability lock poisoned")
            .bytes
            .clone()
    }

    pub fn fail_next_persist(&self) {
        self.state
            .lock()
            .expect("memory durability lock poisoned")
            .fail_next = true;
    }

    pub fn recovered_state(&self) -> StorageRecoveryState {
        self.state
            .lock()
            .expect("memory durability lock poisoned")
            .recovered_state
    }
}

/// In-memory durable transaction log used by the behavior-disconnected host.
/// The sink itself is deliberately not cloneable: only its non-writing
/// [`MemoryDurabilityControl`] may be shared outside [`DurableReplicaHost`].
#[derive(Debug)]
pub struct MemoryDurability {
    state: Arc<Mutex<MemoryDurabilityState>>,
}

impl Default for MemoryDurability {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(MemoryDurabilityState {
                bytes: empty_storage_transaction_log(),
                recovered_state: StorageRecoveryState::default(),
                fail_next: false,
            })),
        }
    }
}

impl MemoryDurability {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, AdapterError> {
        let transactions = decode_storage_transaction_log(&bytes)?;
        let mut recovered_state = StorageRecoveryState::default();
        for transaction in &transactions {
            recovered_state = recovered_state.advance(
                transaction,
                recovered_state
                    .sequence()
                    .checked_add(1)
                    .ok_or(StorageError::SequenceOverflow)?,
            )?;
        }
        Ok(Self {
            state: Arc::new(Mutex::new(MemoryDurabilityState {
                bytes,
                recovered_state,
                fail_next: false,
            })),
        })
    }

    pub fn control(&self) -> MemoryDurabilityControl {
        MemoryDurabilityControl {
            state: Arc::clone(&self.state),
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.control().bytes()
    }

    pub fn fail_next_persist(&self) {
        self.control().fail_next_persist();
    }
}

impl AsyncTransactionSink for MemoryDurability {
    type Error = String;

    fn writer_lease_active(&self) -> bool {
        true
    }

    fn recovered_state(&self) -> StorageRecoveryState {
        self.state
            .lock()
            .expect("memory durability lock poisoned")
            .recovered_state
    }

    async fn persist(
        &mut self,
        transaction: &StorageTransaction,
        expected: StorageRecoveryState,
    ) -> Result<(), Self::Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "memory durability lock poisoned".to_owned())?;
        if state.fail_next {
            state.fail_next = false;
            return Err("injected persistence failure".into());
        }
        let actual = state
            .recovered_state
            .advance(
                transaction,
                state
                    .recovered_state
                    .sequence()
                    .checked_add(1)
                    .ok_or_else(|| "memory durability sequence overflow".to_owned())?,
            )
            .map_err(|error| error.to_string())?;
        if actual != expected {
            return Err(format!(
                "memory durability recovery preview mismatch: expected {expected:?}, actual {actual:?}"
            ));
        }
        let next = append_storage_transaction_log(&state.bytes, transaction)
            .map_err(|error| error.to_string())?;
        state.bytes = next;
        state.recovered_state = expected;
        Ok(())
    }
}

/// Durable, single-writer project host. It owns authored history only; no
/// renderer or per-frame authority is hidden in this type. `D` is the host's
/// asynchronous persistence seam (for example an IndexedDB adapter); the
/// supplied sink must be the sole writer from prepare through finalization.
pub struct DurableProject<D = MemoryDurability> {
    project_id: ProjectId,
    host: DurableReplicaHost<MemoryStorage, AuthoredPolicy, D>,
    memory_durability: Option<MemoryDurabilityControl>,
    published_history_len: usize,
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

/// Stable refusal reasons for one typed inbound replica record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordRefusal {
    HashMismatch,
    Unauthorized,
}

/// Result of offering one typed carrier record with a receiver-local
/// checkpoint attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyRecordWithCheckpointReport {
    Applied {
        entry: EntryHash,
    },
    AlreadyPresent {
        entry: EntryHash,
    },
    Deferred {
        entry: EntryHash,
        missing: Vec<EntryHash>,
    },
    Refused {
        entry: EntryHash,
        reason: RecordRefusal,
    },
}

impl DurableProject<MemoryDurability> {
    pub fn new(project_id: ProjectId) -> Result<Self, AdapterError> {
        let durability = MemoryDurability::default();
        let control = durability.control();
        Self::with_replayed_sink(project_id, durability, Vec::new(), Some(control))
    }

    /// Recover from a locally trusted log previously returned by
    /// [`MemoryDurability::bytes`]. Untrusted peer input must enter through
    /// [`DurableProject::apply_records`] so normal policy admission runs.
    pub fn recover(project_id: ProjectId, bytes: Vec<u8>) -> Result<Self, AdapterError> {
        let transactions = decode_storage_transaction_log(&bytes)?;
        let durability = MemoryDurability::from_bytes(bytes)?;
        let control = durability.control();
        Self::with_replayed_sink(project_id, durability, transactions, Some(control))
    }

    /// Import a portable archive through ordinary HHHS repair and application
    /// policy. A failure drops the temporary in-memory host, so callers never
    /// receive a partially admitted project.
    pub async fn import_archive(bytes: &[u8]) -> Result<Self, AdapterError> {
        let archive = ProjectArchive::decode(bytes)?;
        let mut project = Self::new(archive.project_id())?;
        project.apply_archive(&archive).await?;
        Ok(project)
    }

    /// Detached bytes from the exclusively owned in-memory durability sink.
    pub fn durable_bytes(&self) -> Vec<u8> {
        self.memory_durability
            .as_ref()
            .expect("memory durability control installed")
            .bytes()
    }

    /// Deterministically fail the next external persistence attempt in tests.
    pub fn fail_next_persist(&self) {
        self.memory_durability
            .as_ref()
            .expect("memory durability control installed")
            .fail_next_persist();
    }

    pub fn durability_control(&self) -> MemoryDurabilityControl {
        self.memory_durability
            .as_ref()
            .expect("memory durability control installed")
            .clone()
    }

    /// Recover an in-memory sink from the exact transactions decoded from its
    /// own durable bytes while retaining only a non-writing control handle.
    pub fn recover_trusted_transactions(
        project_id: ProjectId,
        durability: MemoryDurability,
        transactions: Vec<StorageTransaction>,
    ) -> Result<Self, AdapterError> {
        let control = durability.control();
        Self::with_replayed_sink(project_id, durability, transactions, Some(control))
    }
}

impl<D> DurableProject<D>
where
    D: AsyncTransactionSink,
{
    /// Construct an empty durable project around an application-provided sink.
    /// A browser host can supply IndexedDB here without adding any browser API
    /// to this crate.
    pub fn with_sink(project_id: ProjectId, durability: D) -> Result<Self, AdapterError> {
        Self::with_replayed_sink(project_id, durability, Vec::new(), None)
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
    pub fn recover_trusted_transactions_with_sink(
        project_id: ProjectId,
        durability: D,
        transactions: Vec<StorageTransaction>,
    ) -> Result<Self, AdapterError> {
        Self::with_replayed_sink(project_id, durability, transactions, None)
    }

    fn with_replayed_sink(
        project_id: ProjectId,
        durability: D,
        transactions: Vec<StorageTransaction>,
        memory_durability: Option<MemoryDurabilityControl>,
    ) -> Result<Self, AdapterError> {
        project_id
            .validate()
            .map_err(|_| AdapterError::NilProjectId)?;
        let storage = MemoryStorage::new();
        for transaction in &transactions {
            for entry in transaction.entries() {
                decode_authored(project_id, &entry.payload)?;
            }
            storage.commit(transaction.clone())?;
        }
        let published_history_len = storage.snapshot().len();
        let replica = Replica::builder(
            storage,
            AuthoredPolicy { project_id },
            namespace(project_id),
        )
        .open()
        .build()?;
        let host = DurableReplicaHost::new(replica, durability)?;
        Ok(Self {
            project_id,
            host,
            memory_durability,
            published_history_len,
        })
    }

    pub fn project_id(&self) -> ProjectId {
        self.project_id
    }

    pub fn state(&self) -> Result<ProjectState, AdapterError> {
        let snapshot = self.host.snapshot()?;
        materialize_project(self.project_id, &snapshot.history)
    }

    pub fn history_len(&self) -> usize {
        self.published_history_len
    }

    /// Decode the complete authored lane in canonical HHHS topological order.
    ///
    /// This is a domain-level restart view, not a carrier/archive view: entry
    /// hashes, authority evidence, receiver-local checkpoints, and storage
    /// framing remain encapsulated. Every returned envelope passed the same
    /// frozen payload validation used during admission and trusted recovery.
    pub fn authored_history(&self) -> Result<Vec<AuthoredEnvelope>, AdapterError> {
        self.host
            .snapshot()?
            .history
            .entries_topo()
            .into_iter()
            .map(|entry| decode_authored(self.project_id, &entry.payload))
            .collect()
    }

    /// Read one receiver-local, rebuildable projection checkpoint.
    ///
    /// Checkpoints never enter public replica records, project archives, or
    /// repair streams. They are local acceleration/source-cursor state tied to
    /// an exact admitted history horizon.
    pub fn projection_checkpoint(
        &self,
        key: &ProjectionKey,
    ) -> Result<Option<ProjectionCheckpoint>, AdapterError> {
        Ok(self.host.snapshot()?.checkpoint(key).cloned())
    }

    /// Return whether a receiver-local checkpoint is intact and anchors an
    /// exact prefix of this project's current canonical history.
    ///
    /// A checkpoint at the current horizon is also a valid prefix. This does
    /// not interpret the checkpoint bytes; it only proves their history
    /// attachment before an application-specific import lifecycle advances
    /// them.
    pub fn projection_checkpoint_matches_history_prefix(
        &self,
        checkpoint: &ProjectionCheckpoint,
    ) -> Result<bool, AdapterError> {
        if !checkpoint.is_intact() {
            return Ok(false);
        }
        let snapshot = self.host.snapshot()?;
        Ok(SlicedDag::new(&snapshot.history, &checkpoint.at)
            .is_ok_and(|prefix| history_root(&prefix) == checkpoint.history_root))
    }

    /// Capture the complete authored history as a deterministic portable
    /// archive value. The live durability sink and local transaction framing
    /// are deliberately excluded.
    pub fn project_archive(&self) -> Result<ProjectArchive, AdapterError> {
        let history = self.host.snapshot()?.history;
        let state = materialize_project(self.project_id, &history)?;
        let entries = history.entries_topo();
        if entries.len() > MAX_PROJECT_ARCHIVE_RECORDS {
            return Err(AdapterError::TooManyArchiveRecords {
                actual: entries.len(),
                max: MAX_PROJECT_ARCHIVE_RECORDS,
            });
        }
        let records = entries
            .into_iter()
            .map(|entry| ReplicaRecord::new(entry, AuthorityInput::Open))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ProjectArchive {
            project_id: self.project_id,
            history_root: state.history_root,
            state_root: state.state_root,
            records,
        })
    }

    /// Encode [`Self::project_archive`] for file, removable-media, or other
    /// offline transport.
    pub fn export_archive(&self) -> Result<Vec<u8>, AdapterError> {
        self.project_archive()?.encode()
    }
}

impl<D> DurableProject<D>
where
    D: AsyncTransactionSink,
{
    /// Admit one typed carrier record and a receiver-local checkpoint in the
    /// same persist-before-publish transaction.
    ///
    /// A duplicate is a zero-write result and deliberately ignores the
    /// supplied checkpoint bytes. Missing predecessors are returned for
    /// carrier repair without changing history or local metadata. Callers can
    /// therefore advance a projection revision only for `Applied` records and
    /// retry safely after persistence failure.
    pub async fn apply_record_with_projection_checkpoint(
        &mut self,
        record: ReplicaRecord,
        key: ProjectionKey,
        checkpoint_bytes: Vec<u8>,
    ) -> Result<ApplyRecordWithCheckpointReport, AdapterError> {
        let entry = record.entry_hash();
        if self.host.snapshot()?.history.contains(&entry) {
            return Ok(ApplyRecordWithCheckpointReport::AlreadyPresent { entry });
        }
        let prepared = match self.host.prepare(record.into_admission_request()) {
            Ok(prepared) => prepared,
            Err(ReplicaRepairError::Replica(ReplicaError::MissingPrevs(missing))) => {
                return Ok(ApplyRecordWithCheckpointReport::Deferred { entry, missing });
            }
            Err(ReplicaRepairError::Replica(ReplicaError::BadDigest(_))) => {
                return Ok(ApplyRecordWithCheckpointReport::Refused {
                    entry,
                    reason: RecordRefusal::HashMismatch,
                });
            }
            Err(ReplicaRepairError::Replica(
                ReplicaError::AuthorityProfileRequired
                | ReplicaError::AuthorityProfileMismatch
                | ReplicaError::ApplicationRejected(_)
                | ReplicaError::Presentation(_)
                | ReplicaError::CapabilityDenied(_)
                | ReplicaError::InvalidTrustedRoot(_),
            )) => {
                return Ok(ApplyRecordWithCheckpointReport::Refused {
                    entry,
                    reason: RecordRefusal::Unauthorized,
                });
            }
            Err(error) => return Err(error.into()),
        };
        let prepared = prepared.with_local_attachments(move |view, local| {
            local.save_checkpoint(view.checkpoint(key, checkpoint_bytes)?);
            Ok(())
        })?;
        self.host.commit_prepared(prepared).await?;
        self.published_history_len = self.host.snapshot()?.history.len();
        Ok(ApplyRecordWithCheckpointReport::Applied { entry })
    }

    /// Persist one receiver-local projection checkpoint at the exact current
    /// public-history horizon without adding an authored entry.
    ///
    /// This is intended for explicit local lifecycle transitions such as
    /// adopting a validated portable archive. External durability completes
    /// before the in-memory replica publishes the checkpoint. An exact retry
    /// at the same history/bytes horizon performs no write.
    pub async fn persist_projection_checkpoint(
        &mut self,
        key: ProjectionKey,
        bytes: Vec<u8>,
    ) -> Result<ProjectionCheckpoint, AdapterError> {
        let mut checkpoint = None;
        let prepared = self.host.prepare_local_transaction(|view, local| {
            let value = view.checkpoint(key, bytes)?;
            local.save_checkpoint(value.clone());
            checkpoint = Some(value);
            Ok(())
        })?;
        self.host
            .commit_prepared_local_transaction(prepared)
            .await?;
        let checkpoint = checkpoint.expect("local checkpoint preparation produced a value");
        Ok(checkpoint)
    }

    /// Re-admit a complete portable archive into this durability sink.
    ///
    /// Exact retries and resuming a prefix already present locally are safe.
    /// A local branch absent from the archive is rejected before any write, so
    /// this operation never silently replaces or discards divergent history.
    pub async fn apply_archive(
        &mut self,
        archive: &ProjectArchive,
    ) -> Result<ApplyRecordsReport, AdapterError> {
        if self.project_id != archive.project_id {
            return Err(AdapterError::WrongArchiveProject {
                expected: self.project_id.as_uuid(),
                actual: archive.project_id.as_uuid(),
            });
        }
        let archive_hashes: BTreeSet<_> = archive
            .records
            .iter()
            .map(ReplicaRecord::entry_hash)
            .collect();
        let local_only = self
            .host
            .snapshot()?
            .history
            .all_hashes()
            .into_iter()
            .filter(|hash| !archive_hashes.contains(hash))
            .count();
        if local_only != 0 {
            return Err(AdapterError::ArchiveLocalDivergence { local_only });
        }

        let report = self.apply_records(&archive.records).await?;
        if !report.refused.is_empty() || !report.deferred.is_empty() {
            return Err(AdapterError::IncompleteArchiveAdmission {
                refused: report.refused.len(),
                deferred: report.deferred.len(),
            });
        }
        if self.history_len() != archive.record_count() {
            return Err(AdapterError::ArchiveHistoryLengthMismatch {
                expected: archive.record_count(),
                actual: self.history_len(),
            });
        }
        let state = self.state()?;
        if state.history_root != archive.history_root {
            return Err(AdapterError::ArchiveHistoryRootMismatch);
        }
        if state.state_root != archive.state_root {
            return Err(AdapterError::ArchiveStateRootMismatch);
        }
        Ok(report)
    }

    pub async fn admit(
        &mut self,
        envelope: &AuthoredEnvelope,
    ) -> Result<ReplicaRecord, AdapterError> {
        let payload = encode_authored(self.project_id, envelope)?;
        let prepared = self.host.prepare_open(payload)?;
        let commit = self.host.commit_prepared(prepared).await?;
        self.published_history_len = self.host.snapshot()?.history.len();
        Ok(commit.replica_record().clone())
    }

    /// Persist one authored envelope and one receiver-local projection
    /// checkpoint in the same HHHS storage transaction.
    ///
    /// The checkpoint is anchored by HHHS to the staged post-entry frontier
    /// and history root. Neither the public entry nor the local checkpoint is
    /// published when external durability fails. The checkpoint is excluded
    /// from public [`ReplicaRecord`] values, project archives, and repair.
    pub async fn admit_with_projection_checkpoint(
        &mut self,
        envelope: &AuthoredEnvelope,
        key: ProjectionKey,
        checkpoint_bytes: Vec<u8>,
    ) -> Result<ReplicaRecord, AdapterError> {
        let payload = encode_authored(self.project_id, envelope)?;
        let prepared = self.host.prepare_open_with(payload, move |view, local| {
            local.save_checkpoint(view.checkpoint(key, checkpoint_bytes)?);
            Ok(())
        })?;
        let commit = self.host.commit_prepared(prepared).await?;
        self.published_history_len = self.host.snapshot()?.history.len();
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
        let mut known: BTreeSet<_> = self
            .host
            .snapshot()?
            .history
            .all_hashes()
            .into_iter()
            .collect();
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
            self.published_history_len = self.host.snapshot()?.history.len();
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
