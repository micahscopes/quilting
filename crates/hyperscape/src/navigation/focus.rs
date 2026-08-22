use super::TransitionEasing;
use bevy_ecs::prelude::{Entity, Resource};

/// A positive ordinary-space sphere used by selection focus and inversion.
///
/// The sphere lives before the entity/view Möbius map. This keeps object
/// selection, the sharp focus field, and spherical inversion on one exact
/// geometric boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusSphere {
    pub center: [f64; 3],
    pub radius: f64,
}

impl FocusSphere {
    pub fn new(center: [f64; 3], radius: f64) -> Result<Self, &'static str> {
        if center.into_iter().any(|value| !value.is_finite()) {
            return Err("focus sphere center must be finite");
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err("focus sphere radius must be finite and positive");
        }
        Ok(Self { center, radius })
    }

    /// Exact normalized geodesic radius in the round compactification induced
    /// by this sphere: center=0, sphere=1/2, infinity=1.
    pub fn compactified_radial_coordinate(&self, point: [f64; 3]) -> Result<f64, &'static str> {
        if point.into_iter().any(|value| !value.is_finite()) {
            return Err("focus query point must be finite");
        }
        let distance = point
            .into_iter()
            .zip(self.center)
            .map(|(coordinate, center)| (coordinate - center).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(std::f64::consts::FRAC_2_PI * (distance / self.radius).atan())
    }
}

/// Optional object ownership of the shared focus/inversion sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusAnchor {
    pub entity: Entity,
    pub source_bound: FocusSphere,
    pub margin: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusSphereTransition {
    pub start: FocusSphere,
    pub target: FocusSphere,
    pub elapsed_seconds: f64,
    pub duration_seconds: f64,
    pub easing: TransitionEasing,
}

/// Deterministic interaction state shared by selection, navigation, focus,
/// and spherical inversion. Device-specific mappings emit edits into this
/// resource rather than owning another camera or sphere representation.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct FocusNavigation {
    pub sphere: FocusSphere,
    pub anchor: Option<FocusAnchor>,
    pub transition: Option<FocusSphereTransition>,
    pub focus_enabled: bool,
    pub inversion_enabled: bool,
    /// Focal shell in normalized compactified geodesic radius:
    /// origin=0, inversion sphere=1/2, infinity=1.
    pub focus_coordinate: f64,
    /// Spheroidal depth-of-field aperture in the same angular coordinate.
    pub angular_aperture: f64,
}

impl Default for FocusNavigation {
    fn default() -> Self {
        Self {
            sphere: FocusSphere {
                center: [0.5, 0.0, 0.0],
                radius: 2.0,
            },
            anchor: None,
            transition: None,
            focus_enabled: false,
            inversion_enabled: false,
            focus_coordinate: 0.5,
            angular_aperture: 0.1,
        }
    }
}

impl FocusNavigation {
    pub const MIN_RADIUS: f64 = 0.011;
    pub const MIN_ANCHORED_MARGIN: f64 = 1.0;
    pub const MAX_ANCHORED_MARGIN: f64 = 4.0;

    /// Smooth circle-of-confusion response around the configured spheroidal
    /// focal shell. One aperture away from the shell has 50% defocus.
    pub fn defocus_at(&self, point: [f64; 3]) -> Result<f64, &'static str> {
        if !self.focus_coordinate.is_finite()
            || !self.angular_aperture.is_finite()
            || self.angular_aperture <= 0.0
        {
            return Err("focus coordinate must be finite and aperture must be positive");
        }
        let coordinate = self.sphere.compactified_radial_coordinate(point)?;
        let coc = (coordinate - self.focus_coordinate).abs() / self.angular_aperture;
        Ok(coc / (1.0 + coc))
    }

    /// Select and smoothly fit around an entity while preserving one sphere.
    pub fn anchor_to(
        &mut self,
        entity: Entity,
        source_bound: FocusSphere,
        margin: f64,
        duration_seconds: f64,
    ) -> Result<(), &'static str> {
        self.anchor_to_with_easing(
            entity,
            source_bound,
            margin,
            duration_seconds,
            TransitionEasing::SmootherStep,
        )
    }

    pub fn anchor_to_with_easing(
        &mut self,
        entity: Entity,
        source_bound: FocusSphere,
        margin: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    ) -> Result<(), &'static str> {
        FocusSphere::new(source_bound.center, source_bound.radius)?;
        if !margin.is_finite() || !duration_seconds.is_finite() || duration_seconds < 0.0 {
            return Err("focus margin and transition duration must be finite and nonnegative");
        }
        let margin = margin.clamp(Self::MIN_ANCHORED_MARGIN, Self::MAX_ANCHORED_MARGIN);
        let target = FocusSphere {
            center: source_bound.center,
            radius: (source_bound.radius * margin).max(Self::MIN_RADIUS),
        };
        self.anchor = Some(FocusAnchor {
            entity,
            source_bound,
            margin,
        });
        self.focus_enabled = true;
        if duration_seconds == 0.0 {
            self.sphere = target;
            self.transition = None;
        } else {
            self.transition = Some(FocusSphereTransition {
                start: self.sphere,
                target,
                elapsed_seconds: 0.0,
                duration_seconds,
                easing,
            });
        }
        Ok(())
    }

    /// Remove object ownership without destroying or resetting the sphere.
    pub fn detach(&mut self) {
        self.anchor = None;
        self.transition = None;
    }

    /// Advance the selection fit using linear center and logarithmic radius.
    pub fn advance(&mut self, delta_seconds: f64) -> bool {
        let Some(mut transition) = self.transition else {
            return false;
        };
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return false;
        }
        transition.elapsed_seconds += delta_seconds;
        let linear = (transition.elapsed_seconds / transition.duration_seconds).clamp(0.0, 1.0);
        let t = transition.easing.sample(linear);
        self.sphere.center = std::array::from_fn(|axis| {
            transition.start.center[axis]
                + (transition.target.center[axis] - transition.start.center[axis]) * t
        });
        self.sphere.radius = (transition.start.radius.ln()
            + (transition.target.radius.ln() - transition.start.radius.ln()) * t)
            .exp();
        if linear >= 1.0 {
            self.transition = None;
        } else {
            self.transition = Some(transition);
        }
        true
    }

    /// Translate only a detached sphere. Anchored controls cannot drift it
    /// away from the selected object's bound.
    pub fn translate_free(&mut self, delta: [f64; 3]) -> bool {
        if self.anchor.is_some() || delta.into_iter().any(|value| !value.is_finite()) {
            return false;
        }
        for (coordinate, delta) in self.sphere.center.iter_mut().zip(delta) {
            *coordinate += delta;
        }
        self.transition = None;
        true
    }

    /// Scale by a ratio. An anchor converts this into a bounded object margin.
    pub fn scale_radius(&mut self, ratio: f64) -> bool {
        if !ratio.is_finite() || ratio <= 0.0 {
            return false;
        }
        self.transition = None;
        if let Some(mut anchor) = self.anchor {
            self.sphere.center = anchor.source_bound.center;
            self.sphere.radius = (self.sphere.radius * ratio).clamp(
                anchor.source_bound.radius * Self::MIN_ANCHORED_MARGIN,
                anchor.source_bound.radius * Self::MAX_ANCHORED_MARGIN,
            );
            anchor.margin = self.sphere.radius / anchor.source_bound.radius;
            self.anchor = Some(anchor);
        } else {
            self.sphere.radius = (self.sphere.radius * ratio).clamp(Self::MIN_RADIUS, 5.0);
        }
        true
    }

    pub fn toggle_inversion(&mut self) -> bool {
        self.inversion_enabled = !self.inversion_enabled;
        self.inversion_enabled
    }
}
