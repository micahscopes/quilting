//! Backend-neutral policy for selecting the visible graphics presenter.
//!
//! Renderer adapters report facts about their current residency. This module
//! turns those facts into one deterministic presentation decision without
//! knowing about canvases, DOM classes, GPU handles, or browser lifecycle.

use quilting_core::render::RenderStyle;

/// Facts observed from one optional WebGPU presentation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuPresentationEvidence {
    /// The user explicitly requested live WebGPU presentation. Shadow-only
    /// backends leave this false because they must never replace the incumbent.
    pub live_presentation_requested: bool,
    /// `None` represents a browser-only or otherwise unsupported debug style.
    pub requested_style: Option<RenderStyle>,
    /// The adapter has rendered after the latest style change and therefore
    /// permits a matching retained frame to become visible.
    pub presentation_armed: bool,
    pub backend_ready: bool,
    pub backend_failed: bool,
    pub surface_ready: bool,
    /// PBR support depends on coherent materials/textures and is consequently
    /// dynamic; diagnostic styles are a static renderer capability.
    pub pbr_presentation_ready: bool,
    /// The current frame requests the retained focus post-process. The present
    /// WebGPU implementation composes that exact pass only with PBR.
    pub focus_postprocess_requested: bool,
    /// Exact PBR scene, environment, focus pipelines, and resident-root focus
    /// resources are all available for the requested post-process.
    pub focus_presentation_ready: bool,
    /// The renderer admitted its latest logical source frame to the live
    /// presentation target. Offscreen evidence renders do not satisfy this.
    pub frame_admitted: bool,
    pub has_presented_frame: bool,
    pub surface_lost: bool,
    pub presented_style: Option<RenderStyle>,
}

/// Stable reason/state exposed to adapters and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsPresentationPhase {
    Disabled,
    AwaitingBackend,
    Fallback,
    UnsupportedMode,
    PresentationReady,
    Presenting,
}

impl GraphicsPresentationPhase {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::AwaitingBackend => "awaiting-presentation",
            Self::Fallback => "fallback",
            Self::UnsupportedMode => "unsupported-mode",
            Self::PresentationReady => "presentation-ready",
            Self::Presenting => "presenting",
        }
    }
}

/// One coherent decision shared by DOM presentation, fallback diagnostics,
/// and WebGPU LOD authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsPresentationDecision {
    pub phase: GraphicsPresentationPhase,
    pub supports_requested_style: bool,
    pub failed: bool,
    pub present_webgpu: bool,
    /// A complete device LOD epoch may be prepared while WebGL2 remains
    /// visible, allowing recovery without a circular dependency.
    pub device_lod_recovery_eligible: bool,
    /// Device LOD may replace the incumbent only for the same admitted frame
    /// that makes WebGPU the visible presenter.
    pub device_lod_authority_eligible: bool,
}

/// Resolve presentation facts without mutating application or renderer state.
pub fn resolve_graphics_presentation(
    evidence: WebGpuPresentationEvidence,
) -> GraphicsPresentationDecision {
    let supports_requested_style = evidence.requested_style.is_some_and(|style| {
        if evidence.focus_postprocess_requested {
            style == RenderStyle::Pbr
                && evidence.pbr_presentation_ready
                && evidence.focus_presentation_ready
        } else {
            style != RenderStyle::Pbr || evidence.pbr_presentation_ready
        }
    });
    let failed = evidence.backend_failed || evidence.surface_lost;
    let device_lod_recovery_eligible = evidence.live_presentation_requested
        && evidence.backend_ready
        && evidence.surface_ready
        && supports_requested_style
        && !evidence.surface_lost;
    let present_webgpu = device_lod_recovery_eligible
        && evidence.presentation_armed
        && evidence.frame_admitted
        && evidence.has_presented_frame
        && evidence.presented_style == evidence.requested_style
        && !failed;
    let phase = if !evidence.live_presentation_requested {
        GraphicsPresentationPhase::Disabled
    } else if failed {
        GraphicsPresentationPhase::Fallback
    } else if present_webgpu {
        GraphicsPresentationPhase::Presenting
    } else if evidence.surface_ready && !supports_requested_style {
        GraphicsPresentationPhase::UnsupportedMode
    } else if evidence.surface_ready {
        GraphicsPresentationPhase::PresentationReady
    } else {
        GraphicsPresentationPhase::AwaitingBackend
    };

    GraphicsPresentationDecision {
        phase,
        supports_requested_style,
        failed,
        present_webgpu,
        device_lod_recovery_eligible,
        device_lod_authority_eligible: present_webgpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready(style: RenderStyle) -> WebGpuPresentationEvidence {
        WebGpuPresentationEvidence {
            live_presentation_requested: true,
            requested_style: Some(style),
            presentation_armed: true,
            backend_ready: true,
            backend_failed: false,
            surface_ready: true,
            pbr_presentation_ready: true,
            focus_postprocess_requested: false,
            focus_presentation_ready: false,
            frame_admitted: true,
            has_presented_frame: true,
            surface_lost: false,
            presented_style: Some(style),
        }
    }

    #[test]
    fn matching_admitted_frame_presents_and_owns_device_lod() {
        let decision = resolve_graphics_presentation(ready(RenderStyle::Wire));
        assert_eq!(decision.phase, GraphicsPresentationPhase::Presenting);
        assert!(decision.present_webgpu);
        assert!(decision.device_lod_recovery_eligible);
        assert!(decision.device_lod_authority_eligible);
    }

    #[test]
    fn stale_style_cannot_flash_during_a_mode_change() {
        let mut evidence = ready(RenderStyle::Wire);
        evidence.requested_style = Some(RenderStyle::Normals);
        evidence.presentation_armed = false;
        let decision = resolve_graphics_presentation(evidence);
        assert_eq!(decision.phase, GraphicsPresentationPhase::PresentationReady);
        assert!(!decision.present_webgpu);
        assert!(decision.device_lod_recovery_eligible);
        assert!(!decision.device_lod_authority_eligible);
    }

    #[test]
    fn recovery_can_preheat_before_the_first_admitted_frame() {
        let mut evidence = ready(RenderStyle::Lod);
        evidence.frame_admitted = false;
        evidence.has_presented_frame = false;
        let decision = resolve_graphics_presentation(evidence);
        assert!(decision.device_lod_recovery_eligible);
        assert!(!decision.present_webgpu);
        assert!(!decision.device_lod_authority_eligible);
    }

    #[test]
    fn pbr_requires_dynamic_scene_capability() {
        let mut evidence = ready(RenderStyle::Pbr);
        evidence.pbr_presentation_ready = false;
        let decision = resolve_graphics_presentation(evidence);
        assert_eq!(decision.phase, GraphicsPresentationPhase::UnsupportedMode);
        assert!(!decision.supports_requested_style);
        assert!(!decision.device_lod_recovery_eligible);
    }

    #[test]
    fn focus_composition_requires_ready_pbr_focus_residency() {
        let mut evidence = ready(RenderStyle::Pbr);
        evidence.focus_postprocess_requested = true;
        let unavailable = resolve_graphics_presentation(evidence);
        assert_eq!(unavailable.phase, GraphicsPresentationPhase::UnsupportedMode);
        assert!(!unavailable.supports_requested_style);

        evidence.focus_presentation_ready = true;
        let ready = resolve_graphics_presentation(evidence);
        assert!(ready.present_webgpu);

        evidence.requested_style = Some(RenderStyle::Matcap);
        evidence.presented_style = Some(RenderStyle::Matcap);
        let diagnostic = resolve_graphics_presentation(evidence);
        assert_eq!(diagnostic.phase, GraphicsPresentationPhase::UnsupportedMode);
        assert!(!diagnostic.supports_requested_style);
    }

    #[test]
    fn browser_only_debug_style_is_an_explicit_fallback() {
        let mut evidence = ready(RenderStyle::Pbr);
        evidence.requested_style = None;
        evidence.presented_style = None;
        let decision = resolve_graphics_presentation(evidence);
        assert_eq!(decision.phase, GraphicsPresentationPhase::UnsupportedMode);
        assert!(!decision.present_webgpu);
    }

    #[test]
    fn historical_surface_loss_prevents_recovery_and_presentation() {
        let mut evidence = ready(RenderStyle::Matcap);
        evidence.surface_lost = true;
        let decision = resolve_graphics_presentation(evidence);
        assert_eq!(decision.phase, GraphicsPresentationPhase::Fallback);
        assert!(decision.failed);
        assert!(!decision.device_lod_recovery_eligible);
        assert!(!decision.present_webgpu);
    }

    #[test]
    fn shadow_backend_never_presents_even_with_complete_residency() {
        let mut evidence = ready(RenderStyle::Stretch);
        evidence.live_presentation_requested = false;
        let decision = resolve_graphics_presentation(evidence);
        assert_eq!(decision.phase, GraphicsPresentationPhase::Disabled);
        assert!(!decision.present_webgpu);
        assert!(!decision.device_lod_recovery_eligible);
    }
}
