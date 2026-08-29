//! Read-only presentation-card projection for browser views.
//!
//! Cue activation remains an application event and renderer adaptation remains
//! a platform effect. This module only derives stable display state from the
//! committed low-rate [`hyperscope_app::PresentationReadModel`].

use hyperscope_app::{
    AppEffect, AppStore, PresentationAction, PresentationReadModel, ReduceError, SemanticAction,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_presentation_card;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationCardViewModel {
    pub cue_id: String,
    pub eyebrow: String,
    pub heading: String,
    pub body: String,
    pub progress: String,
    pub can_reverse: bool,
    pub can_advance: bool,
    pub desired_assets: usize,
    pub layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationCardAction {
    Reverse,
    Advance,
}

impl PresentationCardAction {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Reverse => "reverse",
            Self::Advance => "advance",
        }
    }

    const fn semantic(self) -> PresentationAction {
        match self {
            Self::Reverse => PresentationAction::Reverse,
            Self::Advance => PresentationAction::Advance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationCardCommit {
    pub sequence: u64,
    pub revision: u64,
    pub selection: Option<PresentationAnimationClipEffect>,
    pub cancellations: Vec<PresentationAnimationClipEffect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationAnimationClipEffect {
    pub job_id: u64,
    pub scene_request_id: String,
    pub asset_id: String,
    pub clip_index: u32,
}

/// Commit one card navigation action through the application reducer after a
/// platform adapter has synchronized its incumbent camera/focus state.
pub fn activate_presentation_card(
    store: &AppStore,
    action: PresentationCardAction,
) -> Result<PresentationCardCommit, ReduceError> {
    let (sequence, commit) = store.dispatch_semantic(SemanticAction::Present(action.semantic()))?;
    let mut selection = None;
    let mut cancellations = Vec::new();
    for effect in commit.effects {
        let effect = match effect {
            AppEffect::SelectAnimationClip {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => {
                let effect = PresentationAnimationClipEffect {
                    job_id,
                    scene_request_id: scene_request_id.to_string(),
                    asset_id: asset_id.to_string(),
                    clip_index,
                };
                selection = Some(effect);
                continue;
            }
            AppEffect::CancelAnimationClipSelection {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => PresentationAnimationClipEffect {
                job_id,
                scene_request_id: scene_request_id.to_string(),
                asset_id: asset_id.to_string(),
                clip_index,
            },
            _ => continue,
        };
        cancellations.push(effect);
    }
    Ok(PresentationCardCommit {
        sequence,
        revision: commit.revision,
        selection,
        cancellations,
    })
}

impl PresentationCardViewModel {
    pub fn adapter_status(&self) -> String {
        let asset_suffix = if self.desired_assets == 1 { "" } else { "s" };
        let layer_suffix = if self.layers == 1 { "" } else { "s" };
        format!(
            "{} desired asset{} · {} layer{}",
            self.desired_assets, asset_suffix, self.layers, layer_suffix
        )
    }
}

/// Return no card until a presentation and active cue have committed. The
/// application remains the sole cue-count and active-index authority.
pub fn project_presentation_card(
    presentation: Option<&PresentationReadModel>,
) -> Option<PresentationCardViewModel> {
    let presentation = presentation?;
    let active = presentation.active.as_ref()?;
    let text = active.text.as_ref();
    Some(PresentationCardViewModel {
        cue_id: active.cue_id.to_string(),
        eyebrow: text
            .and_then(|text| text.eyebrow.clone())
            .unwrap_or_default(),
        heading: text.map(|text| text.heading.clone()).unwrap_or_default(),
        body: text.map(|text| text.body.clone()).unwrap_or_default(),
        progress: format!("{} / {}", active.cue_index + 1, presentation.cue_count),
        can_reverse: active.cue_index > 0,
        can_advance: active.cue_index + 1 < presentation.cue_count,
        desired_assets: active.required_assets.len(),
        layers: active.layers.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{
        CueText, LayerTransform, PresentationLayerState, PresentationSnapshot,
        PresentationTessellation,
    };
    use hyperscape_protocol::{AssetDescriptor, AssetId, RequestId};
    use hyperscope_app::{
        AnimationClipDescriptor, AssetLoadCompletion, AssetLoadOutcome, AssetLoadScope,
        AssetMetadata, EffectCompletion, PresentationAnimationResidencyBinding,
        PrimarySceneInstallCompletion, PrimarySceneInstallMetadata, PrimarySceneInstallOutcome,
    };

    fn id(value: u128) -> uuid::Uuid {
        uuid::Uuid::from_u128(value)
    }

    #[test]
    fn projection_waits_for_an_active_application_cue() {
        assert_eq!(project_presentation_card(None), None);
        let presentation = PresentationReadModel {
            presentation_id: id(1),
            title: "Demo".to_owned(),
            cue_count: 3,
            assets: Vec::new(),
            active: None,
            animation_residency: None,
        };
        assert_eq!(project_presentation_card(Some(&presentation)), None);
    }

    #[test]
    fn projection_derives_navigation_and_composition_labels() {
        let layer = |value| PresentationLayerState {
            id: id(value),
            name: format!("Layer {value}"),
            asset: id(value + 10),
            transform: LayerTransform::default(),
            visible: true,
            opacity: 1.0,
        };
        let presentation = PresentationReadModel {
            presentation_id: id(1),
            title: "Demo".to_owned(),
            cue_count: 3,
            assets: Vec::new(),
            active: Some(PresentationSnapshot {
                cue_index: 1,
                cue_id: id(2),
                scene_id: id(3),
                view_id: id(4),
                text: Some(CueText {
                    eyebrow: Some("Geometry".to_owned()),
                    heading: "One scene, distinct assets".to_owned(),
                    body: "Identity survives composition.".to_owned(),
                }),
                required_assets: Vec::new(),
                layers: vec![layer(20), layer(21)],
                animations: Vec::new(),
                render_style: hyperscape::RenderStyle::Pbr,
                overlays: Vec::new(),
                tessellation: PresentationTessellation::default(),
            }),
            animation_residency: None,
        };

        let card = project_presentation_card(Some(&presentation)).unwrap();
        assert_eq!(card.cue_id, id(2).to_string());
        assert_eq!(card.eyebrow, "Geometry");
        assert_eq!(card.heading, "One scene, distinct assets");
        assert_eq!(card.body, "Identity survives composition.");
        assert_eq!(card.progress, "2 / 3");
        assert!(card.can_reverse);
        assert!(card.can_advance);
        assert_eq!(card.desired_assets, 0);
        assert_eq!(card.layers, 2);
        assert_eq!(card.adapter_status(), "0 desired assets · 2 layers");
    }

    #[test]
    fn card_navigation_commits_through_the_application_sequence_authority() {
        let store = AppStore::default();
        let presentation =
            hyperscape::Presentation::from_json(hyperscape::HACKER_NIGHT_PRESENTATION_JSON)
                .unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::PresentationLoaded(presentation))
            .unwrap();
        store
            .dispatch_semantic(SemanticAction::Present(PresentationAction::Start))
            .unwrap();

        assert_eq!(
            activate_presentation_card(&store, PresentationCardAction::Advance).unwrap(),
            PresentationCardCommit {
                sequence: 1,
                revision: 3,
                selection: None,
                cancellations: Vec::new(),
            },
        );
        assert_eq!(
            store
                .presentation_snapshot()
                .unwrap()
                .active
                .unwrap()
                .cue_index,
            1,
        );
        assert_eq!(
            activate_presentation_card(&store, PresentationCardAction::Reverse).unwrap(),
            PresentationCardCommit {
                sequence: 2,
                revision: 4,
                selection: None,
                cancellations: Vec::new(),
            },
        );
        assert_eq!(
            store
                .presentation_snapshot()
                .unwrap()
                .active
                .unwrap()
                .cue_index,
            0,
        );
    }

    #[test]
    fn card_navigation_forwards_exact_presentation_clip_effects() {
        let store = AppStore::default();
        let mut presentation =
            hyperscape::Presentation::from_json(hyperscape::HACKER_NIGHT_PRESENTATION_JSON)
                .unwrap();
        presentation.cues[1].animations[0].clip = "turn".to_owned();
        let presentation_asset_id = AssetId::new(presentation.assets[0].id).unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::PresentationLoaded(presentation))
            .unwrap();
        store
            .dispatch_semantic(SemanticAction::Present(PresentationAction::Start))
            .unwrap();

        let request_id = RequestId::from_u128(10).unwrap();
        let resident_asset_id = AssetId::from_u128(20).unwrap();
        store
            .dispatch_semantic(SemanticAction::RequestAsset {
                request_id,
                asset: AssetDescriptor {
                    id: resident_asset_id,
                    uri: "cached-horse.glb".to_owned(),
                    media_type: Some("model/gltf-binary".to_owned()),
                    content_digest: None,
                },
                scope: AssetLoadScope::PrimaryScene,
            })
            .unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::EffectCompleted(
                EffectCompletion::AssetLoad(AssetLoadCompletion {
                    request_id,
                    asset_id: resident_asset_id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 42,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                }),
            ))
            .unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id,
                    asset_id: resident_asset_id,
                    outcome: PrimarySceneInstallOutcome::Installed(PrimarySceneInstallMetadata {
                        num_vertices: 8,
                        num_faces: 12,
                        animation_clips: vec![
                            AnimationClipDescriptor {
                                index: 0,
                                name: "horse_A_".to_owned(),
                                time_min_seconds: 0.0,
                                time_max_seconds: 1.5,
                            },
                            AnimationClipDescriptor {
                                index: 1,
                                name: "turn".to_owned(),
                                time_min_seconds: 2.0,
                                time_max_seconds: 3.0,
                            },
                        ],
                    }),
                }),
            ))
            .unwrap();
        let binding = store
            .dispatch(
                hyperscope_app::AppEvent::PresentationAnimationResidencyChanged(Some(
                    PresentationAnimationResidencyBinding {
                        presentation_asset_id,
                        scene_request_id: request_id,
                        resident_asset_id,
                    },
                )),
            )
            .unwrap();
        assert!(binding.effects.is_empty());

        assert_eq!(
            activate_presentation_card(&store, PresentationCardAction::Advance).unwrap(),
            PresentationCardCommit {
                sequence: 2,
                revision: 7,
                selection: Some(PresentationAnimationClipEffect {
                    job_id: 0,
                    scene_request_id: request_id.to_string(),
                    asset_id: resident_asset_id.to_string(),
                    clip_index: 1,
                }),
                cancellations: Vec::new(),
            },
        );
    }
}
