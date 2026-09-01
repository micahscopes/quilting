//! Read-only selection projection for browser views.
//!
//! Stable identity, source geometry, and exact picked surface come from the
//! committed application frame. Renderer-local packed nodes and transient
//! platform notices deliberately stay outside this view model.

use hyperscope_app::AppFrameSnapshot;

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_interaction_status;

#[derive(Debug, Clone, PartialEq)]
pub struct InteractionSelectionViewModel {
    pub asset_id: String,
    pub entity_id: String,
    pub source_bound_radius: f64,
    pub source_pivot: [f64; 3],
    pub source_face: Option<u32>,
}

impl InteractionSelectionViewModel {
    pub fn identity_label(&self) -> String {
        format!("Selected entity {}", compact_identity(&self.entity_id))
    }

    pub fn geometry_label(&self) -> String {
        match self.source_face {
            Some(face) => format!(
                "source radius {:.3} · face {face}",
                self.source_bound_radius
            ),
            None => format!("source radius {:.3}", self.source_bound_radius),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InteractionStatusViewModel {
    pub revision: u64,
    pub selection: Option<InteractionSelectionViewModel>,
}

impl InteractionStatusViewModel {
    pub fn status_label(&self) -> String {
        self.selection.as_ref().map_or_else(
            || "No object selected".to_owned(),
            |selection| selection.identity_label(),
        )
    }
}

pub fn project_interaction_status(frame: &AppFrameSnapshot) -> InteractionStatusViewModel {
    let selection = frame.selected_focus.map(|selected| {
        let source_face = frame
            .interaction
            .hovered
            .filter(|hovered| hovered.identity == selected.identity)
            .and_then(|hovered| hovered.surface.map(|surface| surface.face));
        InteractionSelectionViewModel {
            asset_id: selected.identity.asset.to_string(),
            entity_id: selected.identity.entity.to_string(),
            source_bound_radius: selected.source_bound.radius,
            source_pivot: selected.source_pivot,
            source_face,
        }
    });
    InteractionStatusViewModel {
        revision: frame.revision,
        selection,
    }
}

fn compact_identity(identity: &str) -> &str {
    identity.split('-').next().unwrap_or(identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{FocusSphere, InteractionAction, InteractionHit};
    use hyperscape_protocol::{AssetEntityId, AssetId, EntityId};
    use hyperscope_app::{AppEvent, AppStore, FrameTick, SemanticAction};

    fn identity() -> AssetEntityId {
        AssetEntityId::new(
            AssetId::from_u128(0xa000_0000_0000_4000_8000_0000_0000_0001).unwrap(),
            EntityId::from_u128(0xe000_0000_0000_4000_8000_0000_0000_0002).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn projection_reads_stable_selection_and_exact_surface_from_one_frame() {
        let store = AppStore::default();
        let hit = InteractionHit::new(
            identity(),
            FocusSphere::new([1.0, 2.0, 3.0], 0.75).unwrap(),
            [1.25, 2.0, 3.0],
            4.0,
        )
        .unwrap()
        .with_surface(17, [0.5, 0.25, 0.25])
        .unwrap();
        store
            .dispatch_semantic(SemanticAction::Interact(
                InteractionAction::ActivatePrimary(hit),
            ))
            .unwrap();
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();

        let projected = project_interaction_status(&store.frame_snapshot());
        let selection = projected.selection.unwrap();
        assert_eq!(selection.asset_id, identity().asset.to_string());
        assert_eq!(selection.entity_id, identity().entity.to_string());
        assert_eq!(selection.source_bound_radius, 0.75);
        assert_eq!(selection.source_pivot, [1.25, 2.0, 3.0]);
        assert_eq!(selection.source_face, Some(17));
        assert_eq!(selection.identity_label(), "Selected entity e0000000");
        assert_eq!(selection.geometry_label(), "source radius 0.750 · face 17");
    }

    #[test]
    fn detached_projection_has_no_browser_synthesized_selection() {
        let projected = project_interaction_status(&AppStore::default().frame_snapshot());
        assert_eq!(projected.selection, None);
        assert_eq!(projected.status_label(), "No object selected");
    }
}
