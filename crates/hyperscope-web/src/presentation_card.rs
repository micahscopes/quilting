//! Read-only presentation-card projection for browser views.
//!
//! Cue activation remains an application event and renderer adaptation remains
//! a platform effect. This module only derives stable display state from the
//! committed low-rate [`hyperscope_app::PresentationReadModel`].

use hyperscope_app::PresentationReadModel;

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
}
