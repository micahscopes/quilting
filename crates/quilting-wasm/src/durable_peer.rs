//! Thin WASM lifecycle for the optional IndexedDB-backed authored peer.
//!
//! The Rust application store and durable session retain all semantic and
//! persistence authority. This module only parses browser values, owns the
//! single-writer lease, and exposes carrier-ready replica-record bytes.

use crate::app_shadow::{peer_receipt_to_js, HyperscopeAppShadow};
use hhhs::EntryHash;
use hhhs_replica::ReplicaRecord;
use hyperscape_hhhs::ProjectId;
use hyperscape_protocol::{LocalPeerEnvelope, PresenceEnvelope};
use hyperscope_app::AppStore;
use hyperscope_hhhs_shadow::{
    DurableAuthoredObservation, DurableAuthoredSession, DurableCarrierDispatch,
    DurableCarrierObservation, DurableLocalPeerDispatch,
};
use hyperscope_web::durable_history::{
    import_durable_authored_session, open_durable_authored_session, IndexedDbDurability,
};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::fmt::{Display, Write as _};
use std::rc::Rc;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct HyperscopeDurablePeer {
    store: AppStore,
    session: RefCell<Option<DurableAuthoredSession<IndexedDbDurability>>>,
    restored_projection_on_open: bool,
    writer_lease: Rc<Cell<bool>>,
}

impl Drop for HyperscopeDurablePeer {
    fn drop(&mut self) {
        self.writer_lease.set(false);
    }
}

#[wasm_bindgen]
impl HyperscopeAppShadow {
    /// Open one IndexedDB-backed authored peer sharing this application's Rust
    /// store. Only one such peer may be live per application object.
    #[wasm_bindgen(js_name = openDurableAuthoredPeer)]
    pub async fn open_durable_authored_peer(
        &self,
        project_id: &str,
    ) -> Result<HyperscopeDurablePeer, JsValue> {
        let project_id = ProjectId::new(
            Uuid::parse_str(project_id)
                .map_err(|error| js_error(format!("project ID is invalid: {error}")))?,
        )
        .map_err(js_error)?;
        let writer_reservation = WriterLeaseReservation::acquire(self.durable_peer_lease())?;

        let store = self.store_clone();
        let opened = open_durable_authored_session(project_id, &store)
            .await
            .map_err(js_error)?;
        let restored_projection_on_open = opened.restored_projection().is_some();
        let (session, _) = opened.into_parts();
        Ok(HyperscopeDurablePeer {
            store,
            session: RefCell::new(Some(session)),
            restored_projection_on_open,
            writer_lease: writer_reservation.commit(),
        })
    }

    /// Validate portable authored history, persist it to IndexedDB, adopt its
    /// receiver-local cursor, and return the sole durable peer for this app.
    #[wasm_bindgen(js_name = importDurableAuthoredPeer)]
    pub async fn import_durable_authored_peer(
        &self,
        archive_bytes: js_sys::Uint8Array,
    ) -> Result<HyperscopeDurablePeer, JsValue> {
        let writer_reservation = WriterLeaseReservation::acquire(self.durable_peer_lease())?;
        let store = self.store_clone();
        let opened = import_durable_authored_session(&archive_bytes.to_vec(), &store)
            .await
            .map_err(js_error)?;
        let restored_projection_on_open = opened.restored_projection().is_some();
        let (session, _) = opened.into_parts();
        Ok(HyperscopeDurablePeer {
            store,
            session: RefCell::new(Some(session)),
            restored_projection_on_open,
            writer_lease: writer_reservation.commit(),
        })
    }
}

#[wasm_bindgen]
impl HyperscopeDurablePeer {
    /// Report durable authority without exposing IndexedDB or HHHS internals.
    #[wasm_bindgen(js_name = status)]
    pub fn status(&self) -> Result<JsValue, JsValue> {
        let session = self
            .session
            .try_borrow()
            .map_err(|_| JsValue::from_str("durable authored peer is already active"))?;
        let session = session
            .as_ref()
            .ok_or_else(|| JsValue::from_str("durable authored peer is already active"))?;
        serde_wasm_bindgen::to_value(&DurablePeerStatus {
            project_id: session.project_id().as_uuid().to_string(),
            history_len: session.history_len().to_string(),
            projection_revision: session
                .observed_projection_revision()
                .map(|revision| revision.to_string()),
            restored_projection_on_open: self.restored_projection_on_open,
            fault: session.fault().map(|fault| fault.reason.clone()),
        })
        .map_err(js_error)
    }

    /// Persist authored envelopes before publication while keeping presence on
    /// the direct ephemeral lane. Applied authored results include encoded
    /// `ReplicaRecord` bytes suitable for a carrier announcement.
    #[wasm_bindgen(js_name = receiveLocalPeerEnvelope)]
    pub async fn receive_local_peer_envelope(
        &self,
        received_at_seconds: f64,
        frame_json: &str,
    ) -> Result<JsValue, JsValue> {
        let envelope = serde_json::from_str::<LocalPeerEnvelope>(frame_json)
            .map_err(|error| js_error(format!("local peer frame is invalid JSON: {error}")))?;
        let mut session = DurableSessionLease::take(&self.session)?;
        let dispatch = session
            .session_mut()
            .accept_local_peer(&self.store, envelope, received_at_seconds)
            .await
            .map_err(js_error)?;
        durable_dispatch_to_js(&dispatch, session.session().history_len())
    }

    /// Persist one encoded HHHS carrier record and atomically advance the
    /// receiver-local projection cursor. The returned disposition exposes
    /// missing causal predecessors for transport repair without interpreting
    /// remote commands in arrival order.
    #[wasm_bindgen(js_name = receiveReplicaRecord)]
    pub async fn receive_replica_record(
        &self,
        replica_record: js_sys::Uint8Array,
    ) -> Result<JsValue, JsValue> {
        let record = ReplicaRecord::decode(&replica_record.to_vec())
            .map_err(|error| js_error(format!("replica record is invalid: {error}")))?;
        let mut session = DurableSessionLease::take(&self.session)?;
        let dispatch = session
            .session_mut()
            .accept_replica_record(&self.store, record)
            .await
            .map_err(js_error)?;
        carrier_dispatch_to_js(&dispatch)
    }

    /// Remember an outbound local presence echo without admitting presence to
    /// HHHS or projecting this process as its own remote peer.
    #[wasm_bindgen(js_name = recordLocalPresenceEnvelope)]
    pub fn record_local_presence_envelope(&self, envelope_json: &str) -> Result<(), JsValue> {
        let envelope =
            serde_json::from_str::<PresenceEnvelope>(envelope_json).map_err(|error| {
                js_error(format!("local presence envelope is invalid JSON: {error}"))
            })?;
        self.session
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("durable authored peer is already active"))?
            .as_mut()
            .ok_or_else(|| JsValue::from_str("durable authored peer is already active"))?
            .record_local_presence(&envelope)
            .map_err(js_error)
    }
}

/// Own the session across an asynchronous IndexedDB write without retaining a
/// `RefCell` guard. Cancellation drops this lease and restores the session.
struct DurableSessionLease<'a> {
    slot: &'a RefCell<Option<DurableAuthoredSession<IndexedDbDurability>>>,
    session: Option<DurableAuthoredSession<IndexedDbDurability>>,
}

impl<'a> DurableSessionLease<'a> {
    fn take(
        slot: &'a RefCell<Option<DurableAuthoredSession<IndexedDbDurability>>>,
    ) -> Result<Self, JsValue> {
        let session = slot
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("durable authored peer is already active"))?
            .take()
            .ok_or_else(|| JsValue::from_str("durable authored peer is already active"))?;
        Ok(Self {
            slot,
            session: Some(session),
        })
    }

    fn session(&self) -> &DurableAuthoredSession<IndexedDbDurability> {
        self.session
            .as_ref()
            .expect("a live durable session lease owns its session")
    }

    fn session_mut(&mut self) -> &mut DurableAuthoredSession<IndexedDbDurability> {
        self.session
            .as_mut()
            .expect("a live durable session lease owns its session")
    }
}

impl Drop for DurableSessionLease<'_> {
    fn drop(&mut self) {
        let session = self
            .session
            .take()
            .expect("a live durable session lease owns its session");
        let previous = self.slot.borrow_mut().replace(session);
        debug_assert!(
            previous.is_none(),
            "durable session slot changed while leased"
        );
    }
}

/// Reserve the one-writer application slot while IndexedDB is opening. If the
/// future errors or is cancelled, dropping this value releases the slot.
struct WriterLeaseReservation {
    lease: Rc<Cell<bool>>,
    committed: bool,
}

impl WriterLeaseReservation {
    fn acquire(lease: Rc<Cell<bool>>) -> Result<Self, JsValue> {
        if lease.replace(true) {
            return Err(JsValue::from_str(
                "an IndexedDB-backed authored peer is already open for this application",
            ));
        }
        Ok(Self {
            lease,
            committed: false,
        })
    }

    fn commit(mut self) -> Rc<Cell<bool>> {
        self.committed = true;
        Rc::clone(&self.lease)
    }
}

impl Drop for WriterLeaseReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.lease.set(false);
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DurablePeerStatus {
    project_id: String,
    history_len: String,
    projection_revision: Option<String>,
    restored_projection_on_open: bool,
    fault: Option<String>,
}

fn durable_dispatch_to_js(
    dispatch: &DurableLocalPeerDispatch,
    history_len: usize,
) -> Result<JsValue, JsValue> {
    let object = js_sys::Object::new();
    set_property(&object, "peer", &peer_receipt_to_js(&dispatch.peer)?)?;
    set_property(
        &object,
        "historyLen",
        &JsValue::from_str(&history_len.to_string()),
    )?;

    let mut durable_disposition = "none";
    let mut record = JsValue::NULL;
    let mut app_projection_fault = JsValue::NULL;
    if let Some(durable) = &dispatch.durable {
        if let Err(fault) = &durable.app {
            app_projection_fault = JsValue::from_str(&fault.reason);
        }
        match &durable.durable {
            DurableAuthoredObservation::Applied { record: value, .. } => {
                durable_disposition = "applied";
                let encoded = value.encode();
                record = js_sys::Uint8Array::from(encoded.as_slice()).into();
            }
            DurableAuthoredObservation::IgnoredStale { .. } => {
                durable_disposition = "ignored_stale";
            }
        }
    }
    set_property(
        &object,
        "durableDisposition",
        &JsValue::from_str(durable_disposition),
    )?;
    set_property(&object, "replicaRecord", &record)?;
    set_property(&object, "appProjectionFault", &app_projection_fault)?;
    Ok(object.into())
}

fn carrier_dispatch_to_js(dispatch: &DurableCarrierDispatch) -> Result<JsValue, JsValue> {
    let object = js_sys::Object::new();
    let (disposition, entry, projection_revision, history_len, missing, refusal) =
        match &dispatch.durable {
            DurableCarrierObservation::Applied {
                projection_revision,
                entry,
                history_len,
            } => (
                "applied",
                entry,
                Some(*projection_revision),
                *history_len,
                &[][..],
                None,
            ),
            DurableCarrierObservation::AlreadyPresent { entry, history_len } => (
                "already_present",
                entry,
                None,
                *history_len,
                &[][..],
                None,
            ),
            DurableCarrierObservation::Deferred {
                entry,
                missing,
                history_len,
            } => (
                "deferred",
                entry,
                None,
                *history_len,
                missing.as_slice(),
                None,
            ),
            DurableCarrierObservation::Refused {
                entry,
                reason,
                history_len,
            } => (
                "refused",
                entry,
                None,
                *history_len,
                &[][..],
                Some(match reason {
                    hyperscape_hhhs::RecordRefusal::HashMismatch => "hash_mismatch",
                    hyperscape_hhhs::RecordRefusal::Unauthorized => "unauthorized",
                }),
            ),
        };
    set_property(
        &object,
        "durableDisposition",
        &JsValue::from_str(disposition),
    )?;
    set_property(
        &object,
        "entryHash",
        &JsValue::from_str(&entry_hash_hex(entry)),
    )?;
    set_property(
        &object,
        "projectionRevision",
        &projection_revision
            .map(|revision| JsValue::from_str(&revision.to_string()))
            .unwrap_or(JsValue::NULL),
    )?;
    set_property(
        &object,
        "historyLen",
        &JsValue::from_str(&history_len.to_string()),
    )?;
    let missing_hashes = js_sys::Array::new();
    for hash in missing {
        missing_hashes.push(&JsValue::from_str(&entry_hash_hex(hash)));
    }
    set_property(&object, "missingEntryHashes", missing_hashes.as_ref())?;
    set_property(
        &object,
        "refusal",
        &refusal.map(JsValue::from_str).unwrap_or(JsValue::NULL),
    )?;
    let app_projection_fault = dispatch
        .app
        .as_ref()
        .and_then(|result| result.as_ref().err())
        .map(|fault| JsValue::from_str(&fault.reason))
        .unwrap_or(JsValue::NULL);
    set_property(&object, "appProjectionFault", &app_projection_fault)?;
    Ok(object.into())
}

fn entry_hash_hex(hash: &EntryHash) -> String {
    let mut encoded = String::with_capacity(hash.as_bytes().len() * 2);
    for byte in hash.as_bytes() {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn set_property(object: &js_sys::Object, name: &str, value: &JsValue) -> Result<(), JsValue> {
    let written = js_sys::Reflect::set(object.as_ref(), &JsValue::from_str(name), value)?;
    if !written {
        return Err(js_error(format!("could not set durable peer field {name}")));
    }
    Ok(())
}

fn js_error(error: impl Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape_hhhs::DurableProject;
    use hyperscape_protocol::{
        AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, MessageHeader, MessageId,
        PeerId, CURRENT_PROTOCOL_VERSION,
    };
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn opening_reservation_releases_on_drop_and_transfers_on_commit() {
        let lease = Rc::new(Cell::new(false));
        {
            let reservation = WriterLeaseReservation::acquire(Rc::clone(&lease)).unwrap();
            assert!(lease.get());
            drop(reservation);
        }
        assert!(!lease.get());

        let reservation = WriterLeaseReservation::acquire(Rc::clone(&lease)).unwrap();
        let committed = reservation.commit();
        assert!(lease.get());
        committed.set(false);
    }

    #[wasm_bindgen_test(async)]
    async fn application_writer_lease_is_exclusive_and_released_on_drop() {
        let time = js_sys::Date::now() as u128;
        let random = (js_sys::Math::random() * u64::MAX as f64) as u128;
        let project_id = Uuid::from_u128(((time + 1) << 64) | (random + 1)).to_string();
        let app = HyperscopeAppShadow::new();

        let first = app.open_durable_authored_peer(&project_id).await.unwrap();
        assert!(app.open_durable_authored_peer(&project_id).await.is_err());
        drop(first);

        let reopened = app.open_durable_authored_peer(&project_id).await.unwrap();
        assert_eq!(reopened.session.borrow().as_ref().unwrap().history_len(), 0);
    }

    #[wasm_bindgen_test(async)]
    async fn imported_peer_restarts_through_the_strict_open_path() {
        let time = js_sys::Date::now() as u128;
        let random = (js_sys::Math::random() * u64::MAX as f64) as u128;
        let project_id = ProjectId::from_u128(((time + 1) << 64) | (random + 1)).unwrap();
        let mut source = DurableProject::new(project_id).unwrap();
        source
            .admit(&AuthoredEnvelope {
                header: MessageHeader {
                    version: CURRENT_PROTOCOL_VERSION,
                    message_id: MessageId::from_u128(0xd001).unwrap(),
                    sender: PeerId::from_u128(0xd002).unwrap(),
                    sequence: 1,
                },
                command: AuthoredCommand::UpsertAsset {
                    asset: AssetDescriptor {
                        id: AssetId::from_u128(0xd003).unwrap(),
                        uri: "portable.glb".into(),
                        media_type: Some("model/gltf-binary".into()),
                        content_digest: None,
                    },
                },
            })
            .await
            .unwrap();
        let archive = source.export_archive().unwrap();
        let app = HyperscopeAppShadow::new();

        let imported = app
            .import_durable_authored_peer(js_sys::Uint8Array::from(archive.as_slice()))
            .await
            .unwrap();
        assert_eq!(
            imported
                .session
                .borrow()
                .as_ref()
                .unwrap()
                .observed_projection_revision(),
            Some(0)
        );
        drop(imported);

        let reopened = app
            .open_durable_authored_peer(&project_id.as_uuid().to_string())
            .await
            .unwrap();
        assert_eq!(
            reopened
                .session
                .borrow()
                .as_ref()
                .unwrap()
                .observed_projection_revision(),
            Some(0)
        );
    }

    #[wasm_bindgen_test(async)]
    async fn durable_peer_consumes_encoded_carrier_records_idempotently() {
        let time = js_sys::Date::now() as u128;
        let random = (js_sys::Math::random() * u64::MAX as f64) as u128;
        let project_id = ProjectId::from_u128(((time + 1) << 64) | (random + 1)).unwrap();
        let mut source = DurableProject::new(project_id).unwrap();
        let record = source
            .admit(&AuthoredEnvelope {
                header: MessageHeader {
                    version: CURRENT_PROTOCOL_VERSION,
                    message_id: MessageId::from_u128(0xe001).unwrap(),
                    sender: PeerId::from_u128(0xe002).unwrap(),
                    sequence: 1,
                },
                command: AuthoredCommand::UpsertAsset {
                    asset: AssetDescriptor {
                        id: AssetId::from_u128(0xe003).unwrap(),
                        uri: "carrier.glb".into(),
                        media_type: Some("model/gltf-binary".into()),
                        content_digest: None,
                    },
                },
            })
            .await
            .unwrap();
        let encoded = record.encode();
        let app = HyperscopeAppShadow::new();
        let peer = app
            .open_durable_authored_peer(&project_id.as_uuid().to_string())
            .await
            .unwrap();

        let applied = peer
            .receive_replica_record(js_sys::Uint8Array::from(encoded.as_slice()))
            .await
            .unwrap();
        assert_eq!(
            js_sys::Reflect::get(&applied, &JsValue::from_str("durableDisposition"))
                .unwrap()
                .as_string()
                .as_deref(),
            Some("applied")
        );
        assert_eq!(
            js_sys::Reflect::get(&applied, &JsValue::from_str("projectionRevision"))
                .unwrap()
                .as_string()
                .as_deref(),
            Some("0")
        );

        let duplicate = peer
            .receive_replica_record(js_sys::Uint8Array::from(encoded.as_slice()))
            .await
            .unwrap();
        assert_eq!(
            js_sys::Reflect::get(&duplicate, &JsValue::from_str("durableDisposition"))
                .unwrap()
                .as_string()
                .as_deref(),
            Some("already_present")
        );
        assert_eq!(
            peer.session
                .borrow()
                .as_ref()
                .unwrap()
                .observed_projection_revision(),
            Some(0)
        );
    }
}
