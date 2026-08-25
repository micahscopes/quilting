//! Versioned wire values shared by Hyperscape, Hyperscope, Blender, and
//! optional replication adapters.
//!
//! Durable authored commands and short-lived presence deliberately use
//! different envelope types. An HHHS adapter can therefore accept
//! [`AuthoredEnvelope`] without gaining an API that accepts camera frames,
//! hover, selection presence, or animation clocks.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

pub const CURRENT_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion { major: 0, minor: 1 };
pub const MAX_PRESENCE_TTL_MILLIS: u32 = 60_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub fn ensure_supported(self) -> Result<(), WireError> {
        if self == CURRENT_PROTOCOL_VERSION {
            Ok(())
        } else {
            Err(WireError::UnsupportedVersion(self))
        }
    }
}

macro_rules! stable_id {
    ($name:ident, $label:literal) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            pub fn new(value: Uuid) -> Result<Self, WireError> {
                if value.is_nil() {
                    Err(WireError::NilId($label))
                } else {
                    Ok(Self(value))
                }
            }

            pub fn from_u128(value: u128) -> Result<Self, WireError> {
                Self::new(Uuid::from_u128(value))
            }

            pub fn as_uuid(self) -> Uuid {
                self.0
            }

            pub fn validate(self) -> Result<(), WireError> {
                Self::new(self.0).map(drop)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

stable_id!(MessageId, "message");
stable_id!(PeerId, "peer");
stable_id!(RequestId, "request");
stable_id!(AssetId, "asset");
stable_id!(EntityId, "entity");

/// Stable identity of one entity within a particular source asset.
///
/// A glTF node index and a renderer node offset are container/runtime handles,
/// not identities. Carrying the asset explicitly prevents composed scenes
/// from aliasing nodes that happen to use the same local index, while the
/// entity UUID remains stable across Blender, Hyperscape, and authored edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AssetEntityId {
    pub asset: AssetId,
    pub entity: EntityId,
}

impl AssetEntityId {
    pub fn new(asset: AssetId, entity: EntityId) -> Result<Self, WireError> {
        let identity = Self { asset, entity };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(self) -> Result<(), WireError> {
        self.asset.validate()?;
        self.entity.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageHeader {
    pub version: ProtocolVersion,
    pub message_id: MessageId,
    pub sender: PeerId,
    /// Sender-local ordering only. Causal identity belongs to HHHS when the
    /// authored lane is replicated.
    pub sequence: u64,
}

impl MessageHeader {
    pub fn validate(self) -> Result<(), WireError> {
        self.version.ensure_supported()?;
        self.message_id.validate()?;
        self.sender.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetDescriptor {
    pub id: AssetId,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_digest: Option<[u8; 32]>,
}

impl AssetDescriptor {
    pub fn validate(&self) -> Result<(), WireError> {
        self.id.validate()?;
        if self.uri.trim().is_empty() {
            return Err(WireError::InvalidValue("asset URI must not be empty"));
        }
        if self
            .media_type
            .as_ref()
            .is_some_and(|media_type| media_type.trim().is_empty())
        {
            return Err(WireError::InvalidValue(
                "asset media type must not be empty when present",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WireTransform {
    pub translation: [f64; 3],
    /// `(w, x, y, z)` convention.
    pub rotation_wxyz: [f64; 4],
    pub scale: [f64; 3],
}

impl WireTransform {
    pub fn validate(self) -> Result<(), WireError> {
        if !finite3(self.translation)
            || !finite4(self.rotation_wxyz)
            || !finite3(self.scale)
            || self
                .rotation_wxyz
                .into_iter()
                .map(|component| component * component)
                .sum::<f64>()
                <= 1.0e-24
            || self.scale.into_iter().any(|component| component == 0.0)
        {
            return Err(WireError::InvalidValue(
                "transform must be finite with nonzero rotation and scale",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthoredCommand {
    UpsertAsset {
        asset: AssetDescriptor,
    },
    SetEntityTransform {
        entity: EntityId,
        /// Absolute ordinary TRS in the source asset's world chart. Render
        /// extraction replaces the node's flattened glTF world transform with
        /// this value before applying any presentation-layer transform. An
        /// authoring adapter is responsible for basis conversion into the
        /// source asset chart.
        transform: WireTransform,
    },
    RemoveEntity {
        entity: EntityId,
    },
}

impl AuthoredCommand {
    pub fn validate(&self) -> Result<(), WireError> {
        match self {
            Self::UpsertAsset { asset } => asset.validate(),
            Self::SetEntityTransform { entity, transform } => {
                entity.validate()?;
                transform.validate()
            }
            Self::RemoveEntity { entity } => entity.validate(),
        }
    }
}

/// The only protocol value intended for durable history admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthoredEnvelope {
    pub header: MessageHeader,
    pub command: AuthoredCommand,
}

impl AuthoredEnvelope {
    pub fn validate(&self) -> Result<(), WireError> {
        self.header.validate()?;
        self.command.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraPresence {
    pub eye: [f64; 3],
    pub forward: [f64; 3],
    pub up: [f64; 3],
}

impl CameraPresence {
    pub fn validate(self) -> Result<(), WireError> {
        if !finite3(self.eye)
            || normalized_length(self.forward).is_none()
            || normalized_length(self.up).is_none()
            || cross_length(self.forward, self.up) <= 1.0e-12
        {
            return Err(WireError::InvalidValue(
                "presence camera needs a finite eye and independent directions",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FocusPresence {
    pub center: [f64; 3],
    pub radius: f64,
    pub inversion_enabled: bool,
}

impl FocusPresence {
    pub fn validate(self) -> Result<(), WireError> {
        if !finite3(self.center) || !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(WireError::InvalidValue(
                "presence focus sphere must be finite and positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EphemeralPresence {
    pub ttl_millis: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraPresence>,
    #[serde(default)]
    pub selection: Vec<EntityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<FocusPresence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_cue: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_seconds: Option<f64>,
}

impl EphemeralPresence {
    pub fn validate(&self) -> Result<(), WireError> {
        if self.ttl_millis == 0 || self.ttl_millis > MAX_PRESENCE_TTL_MILLIS {
            return Err(WireError::InvalidValue("presence TTL is out of range"));
        }
        if let Some(camera) = self.camera {
            camera.validate()?;
        }
        for entity in &self.selection {
            entity.validate()?;
        }
        if let Some(focus) = self.focus {
            focus.validate()?;
        }
        if let Some(cue) = self.active_cue {
            cue.validate()?;
        }
        if self
            .animation_seconds
            .is_some_and(|seconds| !seconds.is_finite() || seconds < 0.0)
        {
            return Err(WireError::InvalidValue(
                "presence animation time must be finite and nonnegative",
            ));
        }
        Ok(())
    }

    pub fn expires_at_seconds(&self, received_at_seconds: f64) -> Result<f64, WireError> {
        self.validate()?;
        if !received_at_seconds.is_finite() || received_at_seconds < 0.0 {
            return Err(WireError::InvalidValue(
                "presence receipt time must be finite and nonnegative",
            ));
        }
        Ok(received_at_seconds + f64::from(self.ttl_millis) / 1_000.0)
    }
}

/// Short-lived state. This type is intentionally not convertible to
/// [`AuthoredEnvelope`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresenceEnvelope {
    pub header: MessageHeader,
    pub presence: EphemeralPresence,
}

impl PresenceEnvelope {
    pub fn validate(&self) -> Result<(), WireError> {
        self.header.validate()?;
        self.presence.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    UnsupportedVersion(ProtocolVersion),
    NilId(&'static str),
    InvalidValue(&'static str),
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => write!(
                formatter,
                "unsupported protocol version {}.{}",
                version.major, version.minor
            ),
            Self::NilId(label) => write!(formatter, "{label} ID must not be nil"),
            Self::InvalidValue(message) => formatter.write_str(message),
        }
    }
}

impl Error for WireError {}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn finite4(value: [f64; 4]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn normalized_length(value: [f64; 3]) -> Option<f64> {
    if !finite3(value) {
        return None;
    }
    let length = value.into_iter().map(|x| x * x).sum::<f64>().sqrt();
    (length > 1.0e-12).then_some(length)
}

fn cross_length(left: [f64; 3], right: [f64; 3]) -> f64 {
    let cross = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
    cross.into_iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const AUTHORED_FIXTURE: &str =
        include_str!("../../../fixtures/protocol/authored-set-transform-v0.1.json");
    const PRESENCE_FIXTURE: &str =
        include_str!("../../../fixtures/protocol/presence-camera-v0.1.json");

    fn header() -> MessageHeader {
        MessageHeader {
            version: CURRENT_PROTOCOL_VERSION,
            message_id: MessageId::from_u128(1).unwrap(),
            sender: PeerId::from_u128(2).unwrap(),
            sequence: 3,
        }
    }

    #[test]
    fn authored_wire_roundtrip_is_versioned_and_validated() {
        let envelope = AuthoredEnvelope {
            header: header(),
            command: AuthoredCommand::SetEntityTransform {
                entity: EntityId::from_u128(4).unwrap(),
                transform: WireTransform {
                    translation: [1.0, 2.0, 3.0],
                    rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                    scale: [1.0; 3],
                },
            },
        };
        envelope.validate().unwrap();
        let encoded = serde_json::to_string(&envelope).unwrap();
        let decoded: AuthoredEnvelope = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, envelope);
        decoded.validate().unwrap();
    }

    #[test]
    fn checked_in_authored_fixture_is_the_canonical_pretty_json() {
        let envelope: AuthoredEnvelope = serde_json::from_str(AUTHORED_FIXTURE).unwrap();
        envelope.validate().unwrap();
        assert_eq!(
            format!("{}\n", serde_json::to_string_pretty(&envelope).unwrap()),
            AUTHORED_FIXTURE
        );
    }

    #[test]
    fn presence_has_local_ttl_and_cannot_smuggle_invalid_camera_state() {
        let mut envelope = PresenceEnvelope {
            header: header(),
            presence: EphemeralPresence {
                ttl_millis: 1_500,
                camera: Some(CameraPresence {
                    eye: [0.0, 0.0, 3.0],
                    forward: [0.0, 0.0, -1.0],
                    up: [0.0, 1.0, 0.0],
                }),
                selection: vec![EntityId::from_u128(5).unwrap()],
                focus: None,
                active_cue: None,
                animation_seconds: Some(2.0),
            },
        };
        envelope.validate().unwrap();
        assert_eq!(envelope.presence.expires_at_seconds(10.0).unwrap(), 11.5);

        envelope.presence.camera.as_mut().unwrap().up = [0.0, 0.0, -2.0];
        assert!(envelope.validate().is_err());
    }

    #[test]
    fn checked_in_presence_fixture_is_valid_but_remains_a_separate_lane() {
        let envelope: PresenceEnvelope = serde_json::from_str(PRESENCE_FIXTURE).unwrap();
        envelope.validate().unwrap();
        assert_eq!(
            format!("{}\n", serde_json::to_string_pretty(&envelope).unwrap()),
            PRESENCE_FIXTURE
        );
    }

    #[test]
    fn future_or_nil_wire_identity_is_rejected_after_deserialization() {
        let nil = Uuid::nil().to_string();
        let encoded = format!(
            r#"{{"header":{{"version":{{"major":0,"minor":2}},"message_id":"{nil}","sender":"{nil}","sequence":0}},"command":{{"type":"remove_entity","entity":"{nil}"}}}}"#
        );
        let envelope: AuthoredEnvelope = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            envelope.validate(),
            Err(WireError::UnsupportedVersion(ProtocolVersion {
                major: 0,
                minor: 2
            }))
        );
        let mut envelope = envelope;
        envelope.header.version = CURRENT_PROTOCOL_VERSION;
        assert_eq!(envelope.validate(), Err(WireError::NilId("message")));
    }

    #[test]
    fn asset_entity_identity_is_explicit_and_validated() {
        let identity = AssetEntityId::new(
            AssetId::from_u128(4).unwrap(),
            EntityId::from_u128(5).unwrap(),
        )
        .unwrap();
        let encoded = serde_json::to_string(&identity).unwrap();
        assert_eq!(
            serde_json::from_str::<AssetEntityId>(&encoded).unwrap(),
            identity,
        );
        assert_ne!(
            identity,
            AssetEntityId::new(
                AssetId::from_u128(6).unwrap(),
                EntityId::from_u128(5).unwrap(),
            )
            .unwrap(),
            "the same entity UUID in another asset is a different selection",
        );

        let nil_entity = format!(
            r#"{{"asset":"{}","entity":"{}"}}"#,
            AssetId::from_u128(4).unwrap(),
            Uuid::nil(),
        );
        assert_eq!(
            serde_json::from_str::<AssetEntityId>(&nil_entity)
                .unwrap()
                .validate(),
            Err(WireError::NilId("entity")),
        );
        let nil_asset = format!(
            r#"{{"asset":"{}","entity":"{}"}}"#,
            Uuid::nil(),
            EntityId::from_u128(5).unwrap(),
        );
        assert_eq!(
            serde_json::from_str::<AssetEntityId>(&nil_asset)
                .unwrap()
                .validate(),
            Err(WireError::NilId("asset")),
        );
    }
}
