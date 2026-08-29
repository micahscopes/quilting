//! Resolution of authored presentation animation intent against one exact
//! renderer residency.
//!
//! Presentation asset identity and process-local renderer identity are kept
//! deliberately distinct. A browser may satisfy a presentation layer from an
//! IndexedDB or dropped-byte load whose session asset ID is not the durable
//! presentation asset UUID. The explicit binding below records that ephemeral
//! fact without promoting it into authored state or HHHS history.

use crate::{AnimationClipDescriptor, AnimationClock, InstalledPrimarySceneReadModel};
use hyperscape::PresentationSnapshot;
use hyperscape_protocol::{AssetId, RequestId};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

/// Ephemeral evidence that one presentation asset is currently backed by an
/// exact installed renderer scene.
///
/// `presentation_asset_id` names the authored presentation resource.
/// `resident_asset_id` and `scene_request_id` fence the process-local upload.
/// They may intentionally differ from the presentation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationAnimationResidencyBinding {
    pub presentation_asset_id: AssetId,
    pub scene_request_id: RequestId,
    pub resident_asset_id: AssetId,
}

/// A cue directive resolved to a canonical renderer clip and exact residency.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPresentationAnimation {
    pub layer_id: Uuid,
    pub presentation_asset_id: AssetId,
    pub scene_request_id: RequestId,
    pub resident_asset_id: AssetId,
    pub clip: AnimationClipDescriptor,
    /// Clip-relative authored transport state. Sampling adds the clip's
    /// authored minimum and applies the application's loop policy.
    pub clock: AnimationClock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationAnimationResolutionError {
    ResidentSceneMismatch {
        expected_request: RequestId,
        actual_request: RequestId,
        expected_asset: AssetId,
        actual_asset: AssetId,
    },
    AmbiguousAssetDirective {
        presentation_asset_id: AssetId,
    },
    UnknownClip {
        layer_id: Uuid,
        clip_name: String,
    },
    AmbiguousClip {
        layer_id: Uuid,
        clip_name: String,
    },
    TimeOutsideClip {
        layer_id: Uuid,
        clip_name: String,
    },
}

impl fmt::Display for PresentationAnimationResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResidentSceneMismatch {
                expected_request,
                actual_request,
                expected_asset,
                actual_asset,
            } => write!(
                formatter,
                "animation residency expects scene {expected_request}/{expected_asset}, but {actual_request}/{actual_asset} is installed",
            ),
            Self::AmbiguousAssetDirective {
                presentation_asset_id,
            } => write!(
                formatter,
                "presentation cue has more than one animation directive for resident asset {presentation_asset_id}",
            ),
            Self::UnknownClip {
                layer_id,
                clip_name,
            } => write!(
                formatter,
                "presentation layer {layer_id} requests unknown animation clip {clip_name:?}",
            ),
            Self::AmbiguousClip {
                layer_id,
                clip_name,
            } => write!(
                formatter,
                "presentation layer {layer_id} requests ambiguous animation clip {clip_name:?}",
            ),
            Self::TimeOutsideClip {
                layer_id,
                clip_name,
            } => write!(
                formatter,
                "presentation layer {layer_id} requests time outside animation clip {clip_name:?}",
            ),
        }
    }
}

impl Error for PresentationAnimationResolutionError {}

/// Resolve the active cue's directive for one explicitly bound asset.
///
/// A binding for an asset outside the active cue returns `Ok(None)`: it can be
/// retained across cue changes without becoming false identity. An active
/// asset with no directive likewise leaves the resident clip undisturbed.
/// Multiple animated instances of one asset are rejected at this single-scene
/// adapter boundary; the presentation model itself remains fully multi-layer.
pub fn resolve_presentation_animation(
    snapshot: &PresentationSnapshot,
    binding: PresentationAnimationResidencyBinding,
    installed: &InstalledPrimarySceneReadModel,
) -> Result<Option<ResolvedPresentationAnimation>, PresentationAnimationResolutionError> {
    let presentation_asset_uuid = binding.presentation_asset_id.as_uuid();
    if !snapshot
        .layers
        .iter()
        .any(|layer| layer.asset == presentation_asset_uuid)
    {
        return Ok(None);
    }
    if binding.scene_request_id != installed.asset.request_id
        || binding.resident_asset_id != installed.asset.descriptor.id
    {
        return Err(
            PresentationAnimationResolutionError::ResidentSceneMismatch {
                expected_request: binding.scene_request_id,
                actual_request: installed.asset.request_id,
                expected_asset: binding.resident_asset_id,
                actual_asset: installed.asset.descriptor.id,
            },
        );
    }

    let mut directives = snapshot.animations.iter().filter(|animation| {
        snapshot
            .layers
            .iter()
            .any(|layer| layer.id == animation.layer && layer.asset == presentation_asset_uuid)
    });
    let Some(directive) = directives.next() else {
        return Ok(None);
    };
    if directives.next().is_some() {
        return Err(
            PresentationAnimationResolutionError::AmbiguousAssetDirective {
                presentation_asset_id: binding.presentation_asset_id,
            },
        );
    }

    let mut clips = installed
        .install
        .animation_clips
        .iter()
        .filter(|clip| clip.name == directive.clip);
    let Some(clip) = clips.next().cloned() else {
        return Err(PresentationAnimationResolutionError::UnknownClip {
            layer_id: directive.layer,
            clip_name: directive.clip.clone(),
        });
    };
    if clips.next().is_some() {
        return Err(PresentationAnimationResolutionError::AmbiguousClip {
            layer_id: directive.layer,
            clip_name: directive.clip.clone(),
        });
    }
    if directive.time_seconds > clip.duration_seconds() {
        return Err(PresentationAnimationResolutionError::TimeOutsideClip {
            layer_id: directive.layer,
            clip_name: directive.clip.clone(),
        });
    }

    Ok(Some(ResolvedPresentationAnimation {
        layer_id: directive.layer,
        presentation_asset_id: binding.presentation_asset_id,
        scene_request_id: binding.scene_request_id,
        resident_asset_id: binding.resident_asset_id,
        clip,
        clock: AnimationClock {
            playing: directive.playing,
            time_seconds: directive.time_seconds,
            speed: directive.speed,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetMetadata, PrimaryAssetReadModel, PrimarySceneInstallMetadata};
    use hyperscape::{
        CueAnimation, LayerTransform, PresentationLayerState, PresentationTessellation, RenderStyle,
    };
    use hyperscape_protocol::AssetDescriptor;

    fn uuid(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn asset(value: u128) -> AssetId {
        AssetId::from_u128(value).unwrap()
    }

    fn request(value: u128) -> RequestId {
        RequestId::from_u128(value).unwrap()
    }

    fn snapshot(layer_id: Uuid, presentation_asset: AssetId) -> PresentationSnapshot {
        PresentationSnapshot {
            cue_index: 0,
            cue_id: uuid(11),
            scene_id: uuid(12),
            view_id: uuid(13),
            text: None,
            required_assets: Vec::new(),
            layers: vec![PresentationLayerState {
                id: layer_id,
                name: "Animated layer".to_owned(),
                asset: presentation_asset.as_uuid(),
                transform: LayerTransform::default(),
                visible: true,
                opacity: 1.0,
            }],
            animations: vec![CueAnimation {
                layer: layer_id,
                clip: "Canter".to_owned(),
                time_seconds: 1.25,
                playing: false,
                speed: -0.5,
            }],
            render_style: RenderStyle::Pbr,
            overlays: Vec::new(),
            tessellation: PresentationTessellation::default(),
        }
    }

    fn installed(
        scene_request: RequestId,
        resident_asset: AssetId,
    ) -> InstalledPrimarySceneReadModel {
        InstalledPrimarySceneReadModel {
            asset: PrimaryAssetReadModel {
                request_id: scene_request,
                descriptor: AssetDescriptor {
                    id: resident_asset,
                    uri: "cached-horse.glb".to_owned(),
                    media_type: Some("model/gltf-binary".to_owned()),
                    content_digest: None,
                },
                byte_length: 42,
                content_digest: None,
                metadata: AssetMetadata::default(),
            },
            install: PrimarySceneInstallMetadata {
                num_vertices: 8,
                num_faces: 12,
                animation_clips: vec![
                    AnimationClipDescriptor {
                        index: 0,
                        name: "Idle".to_owned(),
                        time_min_seconds: 2.0,
                        time_max_seconds: 3.0,
                    },
                    AnimationClipDescriptor {
                        index: 1,
                        name: "Canter".to_owned(),
                        time_min_seconds: 4.0,
                        time_max_seconds: 6.0,
                    },
                ],
            },
        }
    }

    #[test]
    fn resolves_distinct_presentation_and_session_asset_identity() {
        let layer_id = uuid(20);
        let presentation_asset = asset(21);
        let resident_asset = asset(22);
        let scene_request = request(23);
        let resolved = resolve_presentation_animation(
            &snapshot(layer_id, presentation_asset),
            PresentationAnimationResidencyBinding {
                presentation_asset_id: presentation_asset,
                scene_request_id: scene_request,
                resident_asset_id: resident_asset,
            },
            &installed(scene_request, resident_asset),
        )
        .unwrap()
        .unwrap();

        assert_eq!(resolved.clip.index, 1);
        assert_eq!(resolved.presentation_asset_id, presentation_asset);
        assert_eq!(resolved.resident_asset_id, resident_asset);
        assert_eq!(
            resolved.clock,
            AnimationClock {
                playing: false,
                time_seconds: 1.25,
                speed: -0.5,
            },
        );
    }

    #[test]
    fn inactive_binding_is_deferred_without_relabeling_identity() {
        let presentation_asset = asset(31);
        let resident_asset = asset(32);
        let scene_request = request(33);
        assert_eq!(
            resolve_presentation_animation(
                &snapshot(uuid(30), presentation_asset),
                PresentationAnimationResidencyBinding {
                    presentation_asset_id: asset(98),
                    scene_request_id: scene_request,
                    resident_asset_id: resident_asset,
                },
                &installed(scene_request, resident_asset),
            )
            .unwrap(),
            None,
        );
    }

    #[test]
    fn rejects_wrong_renderer_residency() {
        let layer_id = uuid(40);
        let presentation_asset = asset(41);
        let resident_asset = asset(42);
        let error = resolve_presentation_animation(
            &snapshot(layer_id, presentation_asset),
            PresentationAnimationResidencyBinding {
                presentation_asset_id: presentation_asset,
                scene_request_id: request(43),
                resident_asset_id: resident_asset,
            },
            &installed(request(44), resident_asset),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PresentationAnimationResolutionError::ResidentSceneMismatch { .. }
        ));
    }

    #[test]
    fn rejects_ambiguous_clip_names() {
        let layer_id = uuid(50);
        let presentation_asset = asset(51);
        let resident_asset = asset(52);
        let scene_request = request(53);
        let mut installed = installed(scene_request, resident_asset);
        installed
            .install
            .animation_clips
            .push(AnimationClipDescriptor {
                index: 2,
                name: "Canter".to_owned(),
                time_min_seconds: 0.0,
                time_max_seconds: 2.0,
            });
        assert!(matches!(
            resolve_presentation_animation(
                &snapshot(layer_id, presentation_asset),
                PresentationAnimationResidencyBinding {
                    presentation_asset_id: presentation_asset,
                    scene_request_id: scene_request,
                    resident_asset_id: resident_asset,
                },
                &installed,
            ),
            Err(PresentationAnimationResolutionError::AmbiguousClip { .. })
        ));
    }

    #[test]
    fn rejects_clip_relative_time_outside_authored_range() {
        let layer_id = uuid(60);
        let presentation_asset = asset(61);
        let resident_asset = asset(62);
        let scene_request = request(63);
        let mut snapshot = snapshot(layer_id, presentation_asset);
        snapshot.animations[0].time_seconds = 2.5;
        assert!(matches!(
            resolve_presentation_animation(
                &snapshot,
                PresentationAnimationResidencyBinding {
                    presentation_asset_id: presentation_asset,
                    scene_request_id: scene_request,
                    resident_asset_id: resident_asset,
                },
                &installed(scene_request, resident_asset),
            ),
            Err(PresentationAnimationResolutionError::TimeOutsideClip { .. })
        ));
    }
}
