//! Rust-owned render-control projection for browser views.
//!
//! The application reducer remains authoritative. This module derives one
//! complete control value from the committed render signal and emits complete
//! replacement intent, so a view never performs a split read/modify/write over
//! independently ordered browser signals.

use crate::controls::numeric_control_domain;
pub use crate::controls::NumericControlViewDomain;
use hyperscope_app::{
    AppEffect, AppRenderSnapshot, AppStore, FocusPostprocessSettings, PatchLabEffect, ReduceError,
    RenderSettings, SemanticAction,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_render_controls;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderControlIntent {
    pub style: &'static str,
    pub resolution_level: u8,
    pub density: f64,
    pub screen_attenuation: bool,
    pub min_pixels_per_subdivision: f64,
    pub atlas_exponent: u8,
    pub max_face_edge_ratio: u8,
    pub focus_postprocess: FocusPostprocessSettings,
}

impl RenderControlIntent {
    /// Convert a complete view intent through the application's canonical
    /// parser and validation boundary before it reaches the reducer.
    pub fn into_settings(self) -> Result<RenderSettings, &'static str> {
        RenderSettings::from_wire_values(
            self.style,
            self.resolution_level,
            self.density,
            self.screen_attenuation,
            self.min_pixels_per_subdivision,
            self.atlas_exponent,
            self.max_face_edge_ratio,
        )
        .and_then(|settings| settings.with_focus_postprocess(self.focus_postprocess))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderControlCommit {
    pub sequence: u64,
    pub revision: u64,
    pub value: RenderControlIntent,
    pub patch_lab_effects: Vec<PatchLabEffect>,
}

pub fn set_render_controls(
    store: &AppStore,
    intent: RenderControlIntent,
) -> Result<RenderControlCommit, RenderControlError> {
    let settings = intent
        .into_settings()
        .map_err(RenderControlError::InvalidSettings)?;
    let (sequence, commit) = store
        .dispatch_semantic(SemanticAction::SetRenderSettings(settings))
        .map_err(RenderControlError::Reduce)?;
    let patch_lab_effects = commit
        .effects
        .into_iter()
        .filter_map(|effect| match effect {
            AppEffect::PatchLab(effect) => Some(effect),
            _ => None,
        })
        .collect();
    Ok(RenderControlCommit {
        sequence,
        revision: commit.revision,
        value: project_render_controls(&store.render_snapshot()).value,
        patch_lab_effects,
    })
}

#[derive(Debug)]
pub enum RenderControlError {
    InvalidSettings(&'static str),
    Reduce(ReduceError),
}

impl std::fmt::Display for RenderControlError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSettings(error) => formatter.write_str(error),
            Self::Reduce(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RenderControlError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderControlsViewModel {
    pub revision: u64,
    pub value: RenderControlIntent,
    pub resolution: NumericControlViewDomain,
    pub density: NumericControlViewDomain,
    pub pixel_floor: NumericControlViewDomain,
    pub atlas: NumericControlViewDomain,
}

impl RenderControlsViewModel {
    pub fn with_style(self, style: &'static str) -> RenderControlIntent {
        RenderControlIntent {
            style,
            ..self.value
        }
    }

    pub fn with_resolution(self, resolution_level: u8) -> RenderControlIntent {
        RenderControlIntent {
            resolution_level,
            ..self.value
        }
    }

    pub fn with_density(self, density: f64) -> RenderControlIntent {
        RenderControlIntent {
            density,
            ..self.value
        }
    }

    pub fn with_screen_attenuation(self, screen_attenuation: bool) -> RenderControlIntent {
        RenderControlIntent {
            screen_attenuation,
            ..self.value
        }
    }

    pub fn with_pixel_floor(self, min_pixels_per_subdivision: f64) -> RenderControlIntent {
        RenderControlIntent {
            min_pixels_per_subdivision,
            ..self.value
        }
    }

    pub fn with_atlas(self, atlas_exponent: u8) -> RenderControlIntent {
        RenderControlIntent {
            atlas_exponent,
            ..self.value
        }
    }

    pub fn with_grading(self, max_face_edge_ratio: u8) -> RenderControlIntent {
        RenderControlIntent {
            max_face_edge_ratio,
            ..self.value
        }
    }
}

pub fn project_render_controls(snapshot: &AppRenderSnapshot) -> RenderControlsViewModel {
    let settings = snapshot.settings;
    RenderControlsViewModel {
        revision: snapshot.revision,
        value: RenderControlIntent {
            style: settings.style.wire_name(),
            resolution_level: settings.resolution_level,
            density: settings.tessellation.density,
            screen_attenuation: settings.tessellation.screen_attenuation,
            min_pixels_per_subdivision: settings.tessellation.min_pixels_per_subdivision,
            atlas_exponent: settings.atlas_exponent,
            max_face_edge_ratio: settings.max_face_edge_ratio,
            focus_postprocess: settings.focus_postprocess,
        },
        resolution: numeric_control_domain("res"),
        density: numeric_control_domain("density"),
        pixel_floor: numeric_control_domain("minpx"),
        atlas: numeric_control_domain("atlas"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{PresentationTessellation, RenderStyle};
    use hyperscope_app::{
        AppEvent, EffectCompletion, PatchLabCompletion, PatchLabGeometryCompletion,
        PatchLabGeometryOutcome, PatchLabLodCompletion, PatchLabLodOutcome, PatchLabLodSummary,
        PatchLabSessionIntent, RenderSettings,
    };

    #[test]
    fn projection_uses_committed_settings_and_canonical_domains() {
        let snapshot = AppRenderSnapshot {
            revision: 42,
            settings: RenderSettings {
                style: RenderStyle::MatcapWire,
                resolution_level: 4,
                tessellation: PresentationTessellation {
                    density: 237.0,
                    screen_attenuation: false,
                    min_pixels_per_subdivision: 48.25,
                },
                atlas_exponent: 9,
                max_face_edge_ratio: 4,
                ..RenderSettings::default()
            },
        };
        let view = project_render_controls(&snapshot);
        assert_eq!(view.revision, 42);
        assert_eq!(view.value.style, "matcap_wire");
        assert_eq!(view.resolution.minimum, 0.0);
        assert_eq!(view.resolution.maximum, 6.0);
        assert!(view.resolution.integral);
        assert_eq!(view.resolution.step, 1.0);
        assert_eq!(view.pixel_floor.minimum, 1.0);
        assert_eq!(view.pixel_floor.maximum, 64.0);
        assert!(!view.pixel_floor.integral);
        assert_eq!(view.pixel_floor.step, 0.1);
        assert_eq!(view.with_density(125.0).density, 125.0);
        assert_eq!(view.with_style("lod").style, "lod");
        assert_eq!(view.with_grading(2).max_face_edge_ratio, 2);
        assert_eq!(view.value.into_settings().unwrap(), snapshot.settings);
        assert_eq!(
            RenderControlIntent {
                style: "browser_magic",
                ..view.value
            }
            .into_settings(),
            Err("unknown backend-neutral render style"),
        );
    }

    #[test]
    fn control_dispatch_returns_committed_value_and_adapter_effects() {
        let store = AppStore::default();
        let intent = project_render_controls(&store.render_snapshot()).with_atlas(8);
        let committed = set_render_controls(&store, intent).unwrap();
        assert_eq!(committed.sequence, 0);
        assert_eq!(committed.revision, 1);
        assert_eq!(committed.value.atlas_exponent, 8);
        assert!(committed.patch_lab_effects.is_empty());
        assert_eq!(store.render_snapshot().settings.atlas_exponent, 8);
    }

    #[test]
    fn atlas_edit_returns_the_cross_domain_patch_lab_job() {
        let store = AppStore::default();
        store
            .dispatch_semantic(SemanticAction::SetPatchLab(PatchLabSessionIntent {
                active: true,
                controls: Default::default(),
            }))
            .unwrap();
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::PatchLab(
                PatchLabCompletion::Geometry(PatchLabGeometryCompletion {
                    job_id: 0,
                    outcome: PatchLabGeometryOutcome::Built {
                        vertex_count: 3,
                        face_count: 1,
                    },
                }),
            )))
            .unwrap();
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::PatchLab(
                PatchLabCompletion::Lod(PatchLabLodCompletion {
                    job_id: 1,
                    geometry_job_id: 0,
                    outcome: PatchLabLodOutcome::Evaluated(PatchLabLodSummary {
                        requested_first_face: Some([8, 16, 16]),
                        resident_first_face: Some([8, 16, 16]),
                        promoted_faces: 0,
                        promoted_edges: 0,
                        shared_edges: 0,
                        shared_edge_mismatches: 0,
                        max_face_edge_ratio: 2,
                        rendered_triangles: 256,
                        histogram: Vec::new(),
                    }),
                }),
            )))
            .unwrap();

        let intent = project_render_controls(&store.render_snapshot()).with_atlas(8);
        let committed = set_render_controls(&store, intent).unwrap();
        assert_eq!(committed.patch_lab_effects.len(), 1);
        assert!(matches!(
            committed.patch_lab_effects[0],
            PatchLabEffect::EvaluateLod {
                job_id: 2,
                geometry_job_id: 0,
                ..
            }
        ));
    }
}
