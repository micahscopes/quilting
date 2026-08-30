//! Rust-owned Patch Lab projection for browser views.
//!
//! The reducer owns asynchronous generations and stale-completion policy. This
//! module owns the complete control intent and derives UI domains from the
//! canonical route registry, leaving a web component or Leptos island to do
//! only event and element adaptation.

use crate::controls::numeric_control_domain;
pub use crate::controls::NumericControlViewDomain;
use hyperscope_app::{
    AppRenderSnapshot, PatchLabControls, PatchLabField, PatchLabReadModel, PatchLabSessionIntent,
    PatchLabShape, PATCH_LAB_PHASE_TURN_MICRORADIANS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchLabControlIntent {
    pub active: bool,
    pub controls: PatchLabControls,
}

impl From<PatchLabControlIntent> for PatchLabSessionIntent {
    fn from(intent: PatchLabControlIntent) -> Self {
        Self {
            active: intent.active,
            controls: intent.controls,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchLabControlsViewModel {
    pub revision: u64,
    pub value: PatchLabControlIntent,
    pub state: PatchLabReadModel,
    pub exponent: NumericControlViewDomain,
    pub phase: NumericControlViewDomain,
    pub bend: NumericControlViewDomain,
    pub grid: NumericControlViewDomain,
    pub resident_atlas_exponent: u8,
    pub max_face_edge_ratio: u8,
}

impl PatchLabControlsViewModel {
    pub fn manual_edges_visible(&self) -> bool {
        self.value.controls.field == PatchLabField::ManualEdges
    }

    pub fn field_controls_visible(&self) -> bool {
        !self.manual_edges_visible()
    }

    pub fn bend_visible(&self) -> bool {
        self.value.controls.shape == PatchLabShape::Triangle
    }

    pub fn grid_visible(&self) -> bool {
        self.value.controls.shape == PatchLabShape::Plane
    }

    pub fn activate_shape(self, shape: PatchLabShape) -> PatchLabSessionIntent {
        self.session(
            true,
            PatchLabControls {
                shape,
                ..self.value.controls
            },
        )
    }

    pub fn deactivate(self) -> PatchLabSessionIntent {
        self.session(false, self.value.controls)
    }

    pub fn with_field(self, field: PatchLabField) -> PatchLabSessionIntent {
        self.session(
            true,
            PatchLabControls {
                field,
                ..self.value.controls
            },
        )
    }

    pub fn with_manual_edge(
        self,
        edge: usize,
        exponent: u8,
    ) -> Result<PatchLabSessionIntent, &'static str> {
        let mut controls = self.value.controls;
        let Some(value) = controls.manual_edge_exponents.get_mut(edge) else {
            return Err("Patch Lab edge index must be 0, 1, or 2");
        };
        *value = exponent;
        Ok(self.session(true, controls))
    }

    pub fn with_min_exponent(self, exponent: u8) -> PatchLabSessionIntent {
        let mut controls = self.value.controls;
        controls.min_exponent = exponent;
        controls.max_exponent = controls.max_exponent.max(exponent);
        self.session(true, controls)
    }

    pub fn with_max_exponent(self, exponent: u8) -> PatchLabSessionIntent {
        let mut controls = self.value.controls;
        controls.max_exponent = exponent;
        controls.min_exponent = controls.min_exponent.min(exponent);
        self.session(true, controls)
    }

    pub fn with_phase_radians(
        self,
        phase_radians: f64,
    ) -> Result<PatchLabSessionIntent, &'static str> {
        if !phase_radians.is_finite() {
            return Err("Patch Lab phase must be finite");
        }
        let phase_microradians = ((phase_radians.rem_euclid(std::f64::consts::TAU) * 1_000_000.0)
            .round() as u32)
            % PATCH_LAB_PHASE_TURN_MICRORADIANS;
        Ok(self.session(
            true,
            PatchLabControls {
                phase_microradians,
                ..self.value.controls
            },
        ))
    }

    pub fn with_bend_percent(self, bend_percent: u8) -> PatchLabSessionIntent {
        self.session(
            true,
            PatchLabControls {
                bend_percent,
                ..self.value.controls
            },
        )
    }

    pub fn with_grid(self, grid: u8) -> PatchLabSessionIntent {
        self.session(
            true,
            PatchLabControls {
                grid,
                ..self.value.controls
            },
        )
    }

    pub fn with_animation(self, animate: bool) -> PatchLabSessionIntent {
        self.session(
            true,
            PatchLabControls {
                animate,
                ..self.value.controls
            },
        )
    }

    fn session(&self, active: bool, controls: PatchLabControls) -> PatchLabSessionIntent {
        PatchLabSessionIntent {
            active,
            controls: controls.normalized(self.resident_atlas_exponent),
        }
    }
}

pub fn project_patch_lab_controls(
    state: &PatchLabReadModel,
    render: &AppRenderSnapshot,
) -> PatchLabControlsViewModel {
    let mut exponent = numeric_control_domain("laba");
    exponent.maximum = f64::from(render.settings.atlas_exponent);
    PatchLabControlsViewModel {
        revision: render.revision,
        value: PatchLabControlIntent {
            active: state.active,
            controls: state.controls,
        },
        state: state.clone(),
        exponent,
        phase: numeric_control_domain("labphase"),
        bend: numeric_control_domain("labbend"),
        grid: numeric_control_domain("labgrid"),
        resident_atlas_exponent: render.settings.atlas_exponent,
        max_face_edge_ratio: render.settings.max_face_edge_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{PresentationTessellation, RenderStyle};
    use hyperscope_app::{PatchLabHistogramBin, PatchLabLodSummary, RenderSettings};

    fn render_snapshot() -> AppRenderSnapshot {
        AppRenderSnapshot {
            revision: 17,
            settings: RenderSettings {
                style: RenderStyle::Lod,
                resolution_level: 0,
                tessellation: PresentationTessellation::default(),
                atlas_exponent: 8,
                max_face_edge_ratio: 4,
            },
        }
    }

    #[test]
    fn projection_uses_one_registry_and_preserves_structured_results() {
        let mut state = PatchLabReadModel::default();
        state.active = true;
        state.latest_lod = Some(PatchLabLodSummary {
            requested_first_face: Some([8, 64, 64]),
            resident_first_face: Some([16, 64, 64]),
            promoted_faces: 1,
            promoted_edges: 1,
            shared_edges: 3,
            shared_edge_mismatches: 0,
            max_face_edge_ratio: 4,
            rendered_triangles: 3_072,
            histogram: vec![PatchLabHistogramBin {
                edge_subdivisions: [16, 64, 64],
                face_count: 1,
            }],
        });
        let view = project_patch_lab_controls(&state, &render_snapshot());
        assert_eq!(view.revision, 17);
        assert_eq!(view.exponent.minimum, 0.0);
        assert_eq!(view.exponent.maximum, 8.0);
        assert_eq!(view.exponent.step, 1.0);
        assert_eq!(view.phase.step, 0.001);
        assert_eq!(view.max_face_edge_ratio, 4);
        assert!(view.manual_edges_visible());
        assert_eq!(
            view.state.latest_lod.as_ref().unwrap().rendered_triangles,
            3_072
        );
    }

    #[test]
    fn complete_intents_normalize_shape_ranges_phase_and_atlas() {
        let state = PatchLabReadModel::default();
        let view = project_patch_lab_controls(&state, &render_snapshot());
        let plane = view.clone().activate_shape(PatchLabShape::Plane);
        assert!(plane.active);
        assert_eq!(plane.controls.shape, PatchLabShape::Plane);
        assert_eq!(plane.controls.field, PatchLabField::Wave);

        let minimum = view.clone().with_min_exponent(9);
        assert_eq!(minimum.controls.min_exponent, 8);
        assert_eq!(minimum.controls.max_exponent, 8);
        let maximum = view.clone().with_max_exponent(0);
        assert_eq!(maximum.controls.min_exponent, 0);
        assert_eq!(maximum.controls.max_exponent, 0);
        let phase = view
            .with_phase_radians(std::f64::consts::TAU + 0.25)
            .unwrap();
        assert_eq!(phase.controls.phase_microradians, 250_000);
    }
}
