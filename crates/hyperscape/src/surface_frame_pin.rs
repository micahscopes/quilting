//! Authored conformal frames pinned to animated source-surface addresses.
//!
//! The surface constraint updates only a frame's local-to-parent mapping. Its
//! descendants continue through the ordinary conformal-frame composition, so
//! a pinned object can itself contain reflected, inverted, or further pinned
//! coordinate frames without introducing a second transform hierarchy.

use crate::{
    ConformalScene, HyperscapeDiagnostics, SurfaceAnchorError, SurfaceAttachment, SurfaceSample,
    SurfaceTangentFrame,
};
use bevy_ecs::prelude::{Local, Res, ResMut, Resource};
pub use hyperscape_protocol::SurfaceFrameOrientation;
use quilting_core::{
    ConformalError, ConformalFrameForest, ConformalGenerator, ConformalTransformChain, FrameId,
};
use std::error::Error;
use std::fmt;

/// A durable surface constraint for one conformal frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceFramePin {
    pub attachment: SurfaceAttachment,
    /// Material-frame yaw, measured from the posed QB `u` differential.
    pub heading_radians: f64,
    /// Explicit conformal scale. The generally anisotropic surface metric is
    /// used for orientation only and is never smuggled in as nonconformal
    /// object deformation.
    pub uniform_scale: f64,
    pub orientation: SurfaceFrameOrientation,
    /// Authored child-local conformal operations applied before the surface
    /// similarity. This is where nested reflections and inversions live.
    pub local_offset: ConformalTransformChain,
}

impl SurfaceFramePin {
    pub fn new(attachment: SurfaceAttachment) -> Self {
        Self {
            attachment,
            heading_radians: 0.0,
            uniform_scale: 1.0,
            orientation: SurfaceFrameOrientation::Inherit,
            local_offset: ConformalTransformChain::identity(),
        }
    }

    /// Resolve a posed QB sample expressed in the frame parent's coordinates.
    pub fn resolve(
        &self,
        sample: SurfaceSample,
        parent_world_orientation: i8,
    ) -> Result<ResolvedSurfaceFramePin, SurfaceFramePinError> {
        if !matches!(parent_world_orientation, -1 | 1) {
            return Err(SurfaceFramePinError::InvalidParentOrientation);
        }
        if !self.heading_radians.is_finite() {
            return Err(SurfaceFramePinError::InvalidHeading);
        }
        if !self.uniform_scale.is_finite() || self.uniform_scale <= 0.0 {
            return Err(SurfaceFramePinError::InvalidScale);
        }
        self.local_offset.validate()?;

        let surface_frame = SurfaceTangentFrame::from_sample(self.attachment, sample)?
            .with_heading(self.heading_radians)?;
        let orientation = surface_frame
            .basis()
            .orientation()
            .map_err(SurfaceAnchorError::from)?;

        let mut generators = self.local_offset.generators.clone();
        let authored_world_orientation =
            parent_world_orientation * self.local_offset.orientation_sign();
        let requested_world_orientation = match self.orientation {
            SurfaceFrameOrientation::Inherit => authored_world_orientation,
            SurfaceFrameOrientation::RightSideIn => 1,
            SurfaceFrameOrientation::InsideOut => -1,
        };
        if authored_world_orientation != requested_world_orientation {
            // Reflection in local Z preserves local X/right and Y/up. It is
            // represented by a proper half-turn followed by negative scale,
            // using only the conformal generator vocabulary.
            generators.push(ConformalGenerator::rotation_axis_angle(
                [0.0, 0.0, 1.0],
                std::f64::consts::PI,
            )?);
            generators.push(ConformalGenerator::uniform_scale(-1.0));
        }
        generators.push(ConformalGenerator::uniform_scale(self.uniform_scale));
        generators.push(ConformalGenerator::Rotation {
            quaternion_wxyz: [orientation.w, orientation.x, orientation.y, orientation.z],
        });
        generators.push(ConformalGenerator::translation(surface_frame.origin));
        let local_to_parent = ConformalTransformChain::new(generators)?;
        debug_assert_eq!(
            parent_world_orientation * local_to_parent.orientation_sign(),
            requested_world_orientation
        );

        Ok(ResolvedSurfaceFramePin {
            attachment: self.attachment,
            surface_frame,
            local_to_parent,
            world_orientation_sign: requested_world_orientation,
        })
    }

    /// Resolve and atomically install the latest posed sample. The target
    /// frame keeps its existing spatial parent; descendants immediately
    /// inherit the updated material frame on their next world-chain query.
    pub fn apply_sample(
        &self,
        frames: &mut ConformalFrameForest,
        frame: FrameId,
        sample: SurfaceSample,
    ) -> Result<ResolvedSurfaceFramePin, SurfaceFramePinError> {
        let parent = frames.frame(frame)?.parent;
        let parent_world_orientation = match parent {
            Some(parent) => frames.world_chain(parent)?.orientation_sign(),
            None => 1,
        };
        let resolved = self.resolve(sample, parent_world_orientation)?;
        frames.set_local_to_parent(frame, resolved.local_to_parent.clone())?;
        Ok(resolved)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSurfaceFramePin {
    pub attachment: SurfaceAttachment,
    pub surface_frame: SurfaceTangentFrame,
    pub local_to_parent: ConformalTransformChain,
    pub world_orientation_sign: i8,
}

/// One constraint-graph edge from an authored conformal frame to a stable
/// material point on another entity.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceFramePinBinding {
    pub frame: FrameId,
    pub pin: SurfaceFramePin,
}

/// Authored surface-pin constraints kept separate from the spatial frame
/// forest. A pose/geometry adapter supplies samples after animation and before
/// coordinate extraction.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct SurfaceFramePinSet(pub Vec<SurfaceFramePinBinding>);

impl SurfaceFramePinSet {
    pub fn apply_sample(
        &self,
        index: usize,
        frames: &mut ConformalFrameForest,
        sample: SurfaceSample,
    ) -> Result<ResolvedSurfaceFramePin, SurfaceFramePinError> {
        let binding = self
            .0
            .get(index)
            .ok_or(SurfaceFramePinError::UnknownBinding(index))?;
        binding.pin.apply_sample(frames, binding.frame, sample)
    }

    /// Apply one coherent external pose sample across the constraint graph.
    ///
    /// Parent frames are resolved before pinned descendants even when the
    /// authored constraint array is in another order. All mappings are staged
    /// on a clone, so one invalid differential publishes none of the frame
    /// changes from that pose revision.
    pub fn apply_samples_atomically(
        &self,
        frames: &mut ConformalFrameForest,
        samples: &[Option<SurfaceSample>],
    ) -> Result<Vec<(usize, ResolvedSurfaceFramePin)>, SurfaceFramePinError> {
        let mut order = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(index, binding)| {
                samples
                    .get(index)
                    .and_then(Option::as_ref)
                    .map(|_| (index, binding.frame))
            })
            .map(|(index, frame)| Ok((frame_depth(frames, frame)?, frame.0, index)))
            .collect::<Result<Vec<_>, SurfaceFramePinError>>()?;
        order.sort_unstable();

        let mut staged = frames.clone();
        let mut resolved = Vec::with_capacity(order.len());
        for (_, _, index) in order {
            let sample = samples[index].expect("sampled pin was filtered above");
            let binding = &self.0[index];
            resolved.push((
                index,
                binding
                    .pin
                    .apply_sample(&mut staged, binding.frame, sample)?,
            ));
        }
        *frames = staged;
        Ok(resolved)
    }
}

fn frame_depth(
    frames: &ConformalFrameForest,
    frame: FrameId,
) -> Result<usize, SurfaceFramePinError> {
    let mut depth = 0;
    let mut cursor = Some(frame);
    while let Some(current) = cursor {
        depth += 1;
        if depth > frames.frames().len() {
            return Err(ConformalError::Cycle(current).into());
        }
        cursor = frames.frame(current)?.parent;
    }
    Ok(depth)
}

/// Latest renderer/geometry samples aligned with [`SurfaceFramePinSet`].
///
/// Geometry adapters replace this resource only after a complete pose has
/// been accepted. The monotonic revision lets the ECS consume each coherent
/// pose once without polling or replaying stale asynchronous results.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct SurfaceFramePinSamples {
    revision: u64,
    samples: Vec<Option<SurfaceSample>>,
}

impl SurfaceFramePinSamples {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn samples(&self) -> &[Option<SurfaceSample>] {
        &self.samples
    }

    pub fn replace(&mut self, samples: Vec<Option<SurfaceSample>>) -> u64 {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.samples = samples;
        self.revision
    }
}

/// Resolve an externally supplied posed-surface revision before paths,
/// coordinates, constraints, and render extraction observe the frame graph.
pub(crate) fn apply_surface_frame_pin_samples(
    mut scene: ResMut<ConformalScene>,
    pins: Res<SurfaceFramePinSet>,
    samples: Res<SurfaceFramePinSamples>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    mut applied_revision: Local<u64>,
) {
    if samples.revision() == 0 || samples.revision() == *applied_revision {
        return;
    }
    if let Err(error) = pins.apply_samples_atomically(&mut scene.frames, samples.samples()) {
        diagnostics.0.push(format!(
            "could not apply surface-pin pose revision {}: {error}",
            samples.revision()
        ));
    }
    *applied_revision = samples.revision();
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceFramePinError {
    InvalidHeading,
    InvalidScale,
    InvalidParentOrientation,
    UnknownBinding(usize),
    Surface(SurfaceAnchorError),
    Conformal(ConformalError),
}

impl fmt::Display for SurfaceFramePinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeading => formatter.write_str("surface pin heading must be finite"),
            Self::InvalidScale => {
                formatter.write_str("surface pin scale must be finite and positive")
            }
            Self::InvalidParentOrientation => {
                formatter.write_str("parent world orientation must be +1 or -1")
            }
            Self::UnknownBinding(index) => write!(formatter, "unknown surface pin {index}"),
            Self::Surface(error) => write!(formatter, "surface pin sample: {error}"),
            Self::Conformal(error) => write!(formatter, "surface pin transform: {error}"),
        }
    }
}

impl Error for SurfaceFramePinError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Surface(error) => Some(error),
            Self::Conformal(error) => Some(error),
            Self::InvalidHeading
            | Self::InvalidScale
            | Self::InvalidParentOrientation
            | Self::UnknownBinding(_) => None,
        }
    }
}

impl From<SurfaceAnchorError> for SurfaceFramePinError {
    fn from(value: SurfaceAnchorError) -> Self {
        Self::Surface(value)
    }
}

impl From<ConformalError> for SurfaceFramePinError {
    fn from(value: ConformalError) -> Self {
        Self::Conformal(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StableEntityId, SurfaceAddress};
    use uuid::Uuid;

    const EPSILON: f64 = 1.0e-9;

    fn attachment() -> SurfaceAttachment {
        SurfaceAttachment::new(
            SurfaceAddress::new(StableEntityId(Uuid::from_u128(1)), 3, [0.5, 0.25, 0.25]).unwrap(),
        )
        .unwrap()
    }

    fn sample(origin: [f64; 3]) -> SurfaceSample {
        SurfaceSample {
            output_position: origin,
            tangent_u: [0.0, 0.0, -2.0],
            tangent_v: [-3.0, 0.0, 0.0],
            surface_velocity: [0.0; 3],
        }
    }

    fn assert_point_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < EPSILON,
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn posed_surface_differential_becomes_a_conformal_similarity() {
        let mut pin = SurfaceFramePin::new(attachment());
        pin.uniform_scale = 2.0;
        let resolved = pin.resolve(sample([10.0, 2.0, 3.0]), 1).unwrap();
        assert_eq!(resolved.local_to_parent.orientation_sign(), 1);
        assert_point_close(
            resolved
                .local_to_parent
                .apply_point([1.0, 0.0, 0.0])
                .unwrap(),
            [12.0, 2.0, 3.0],
        );
        assert_point_close(
            resolved
                .local_to_parent
                .apply_point([0.0, 1.0, 0.0])
                .unwrap(),
            [10.0, 4.0, 3.0],
        );
        assert_point_close(
            resolved
                .local_to_parent
                .apply_point([0.0, 0.0, 1.0])
                .unwrap(),
            [10.0, 2.0, 5.0],
        );
    }

    #[test]
    fn heading_rotates_in_the_material_tangent_plane() {
        let mut pin = SurfaceFramePin::new(attachment());
        pin.heading_radians = std::f64::consts::FRAC_PI_2;
        let resolved = pin.resolve(sample([0.0; 3]), 1).unwrap();
        assert_point_close(
            resolved
                .local_to_parent
                .apply_point([1.0, 0.0, 0.0])
                .unwrap(),
            [0.0, 0.0, 1.0],
        );
        assert_point_close(
            resolved
                .local_to_parent
                .apply_point([0.0, 0.0, 1.0])
                .unwrap(),
            [-1.0, 0.0, 0.0],
        );
    }

    #[test]
    fn right_side_in_child_corrects_an_inside_out_parent_without_changing_side() {
        let mut frames = ConformalFrameForest::new();
        let reflected = frames
            .add_frame(
                "reflected parent",
                None,
                ConformalTransformChain::new(vec![ConformalGenerator::sphere_reflection(
                    [0.0; 3], 5.0,
                )])
                .unwrap(),
            )
            .unwrap();
        let child = frames
            .add_frame(
                "right-side-in child",
                Some(reflected),
                ConformalTransformChain::identity(),
            )
            .unwrap();
        let mut pin = SurfaceFramePin::new(attachment());
        pin.orientation = SurfaceFrameOrientation::RightSideIn;
        let resolved = pin
            .apply_sample(&mut frames, child, sample([2.0, 0.0, 0.0]))
            .unwrap();

        assert_eq!(resolved.surface_frame.normal, [0.0, 1.0, 0.0]);
        assert_eq!(resolved.world_orientation_sign, 1);
        assert_eq!(frames.world_chain(child).unwrap().orientation_sign(), 1);
    }

    #[test]
    fn animated_sample_moves_the_frame_and_all_descendants() {
        let mut frames = ConformalFrameForest::new();
        let pinned = frames
            .add_frame("pinned", None, ConformalTransformChain::identity())
            .unwrap();
        let descendant = frames
            .add_frame(
                "nested reflection",
                Some(pinned),
                ConformalTransformChain::new(vec![ConformalGenerator::sphere_reflection(
                    [0.0; 3], 1.0,
                )])
                .unwrap(),
            )
            .unwrap();
        let pin = SurfaceFramePin::new(attachment());
        pin.apply_sample(&mut frames, pinned, sample([2.0, 0.0, 0.0]))
            .unwrap();
        let first = frames
            .world_chain(descendant)
            .unwrap()
            .apply_point([2.0, 0.0, 0.0])
            .unwrap();
        pin.apply_sample(&mut frames, pinned, sample([5.0, 0.0, 0.0]))
            .unwrap();
        let second = frames
            .world_chain(descendant)
            .unwrap()
            .apply_point([2.0, 0.0, 0.0])
            .unwrap();
        assert_point_close(first, [2.5, 0.0, 0.0]);
        assert_point_close(second, [5.5, 0.0, 0.0]);
    }

    #[test]
    fn coherent_samples_sort_parent_pins_first() {
        let mut frames = ConformalFrameForest::new();
        let parent = frames
            .add_frame("parent", None, ConformalTransformChain::identity())
            .unwrap();
        let child = frames
            .add_frame("child", Some(parent), ConformalTransformChain::identity())
            .unwrap();

        let mut parent_pin = SurfaceFramePin::new(attachment());
        parent_pin.orientation = SurfaceFrameOrientation::InsideOut;
        let mut child_pin = SurfaceFramePin::new(attachment());
        child_pin.orientation = SurfaceFrameOrientation::RightSideIn;
        let pins = SurfaceFramePinSet(vec![
            SurfaceFramePinBinding {
                frame: child,
                pin: child_pin,
            },
            SurfaceFramePinBinding {
                frame: parent,
                pin: parent_pin,
            },
        ]);

        let resolved = pins
            .apply_samples_atomically(
                &mut frames,
                &[Some(sample([3.0, 0.0, 0.0])), Some(sample([1.0, 0.0, 0.0]))],
            )
            .unwrap();

        assert_eq!(
            resolved.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [1, 0]
        );
        assert_eq!(frames.world_chain(parent).unwrap().orientation_sign(), -1);
        assert_eq!(frames.world_chain(child).unwrap().orientation_sign(), 1);
    }

    #[test]
    fn invalid_coherent_sample_publishes_no_partial_frame_updates() {
        let mut frames = ConformalFrameForest::new();
        let first = frames
            .add_frame("first", None, ConformalTransformChain::identity())
            .unwrap();
        let second = frames
            .add_frame("second", None, ConformalTransformChain::identity())
            .unwrap();
        let pins = SurfaceFramePinSet(vec![
            SurfaceFramePinBinding {
                frame: first,
                pin: SurfaceFramePin::new(attachment()),
            },
            SurfaceFramePinBinding {
                frame: second,
                pin: SurfaceFramePin::new(attachment()),
            },
        ]);
        let before = frames.clone();
        let invalid = SurfaceSample {
            output_position: [4.0, 0.0, 0.0],
            tangent_u: [0.0; 3],
            tangent_v: [0.0; 3],
            surface_velocity: [0.0; 3],
        };

        assert!(pins
            .apply_samples_atomically(&mut frames, &[Some(sample([2.0, 0.0, 0.0])), Some(invalid)],)
            .is_err());
        assert_eq!(frames, before);
    }

    #[test]
    fn invalid_pose_rejects_without_mutating_the_frame() {
        let mut frames = ConformalFrameForest::new();
        let frame = frames
            .add_frame(
                "pinned",
                None,
                ConformalTransformChain::new(vec![ConformalGenerator::translation([
                    1.0, 2.0, 3.0,
                ])])
                .unwrap(),
            )
            .unwrap();
        let before = frames.frame(frame).unwrap().local_to_parent.clone();
        let pin = SurfaceFramePin::new(attachment());
        let mut invalid = sample([0.0; 3]);
        invalid.tangent_u = [0.0; 3];
        assert!(pin.apply_sample(&mut frames, frame, invalid).is_err());
        assert_eq!(frames.frame(frame).unwrap().local_to_parent, before);
    }
}
