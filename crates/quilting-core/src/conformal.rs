//! Serializable conformal generator words and coordinate frames.
//!
//! The renderer consumes a collapsed [`Mobius`] matrix, but authoring and
//! scene semantics need more structure.  In particular, the inverse of a word
//! of known generators is constructive even though a generic inverse formula
//! for a noncommutative 2×2 quaternion matrix is subtle.  Keeping the word also
//! preserves the operation boundaries that Blender and glTF need to animate.
//!
//! A frame has at most one parent.  This is intentionally a forest rather than
//! an unrestricted DAG: there is one unambiguous path to ambient Euclidean
//! space.  Multiple paths require an explicit path-consistency or holonomy
//! policy and are outside this module's contract.

use crate::quaternion::{Mobius, Quat};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

const MIN_ROTATION_NORM_SQ: f64 = 1.0e-24;

fn finite3(v: [f64; 3]) -> bool {
    v.into_iter().all(f64::is_finite)
}

fn finite4(v: [f64; 4]) -> bool {
    v.into_iter().all(f64::is_finite)
}

/// An authorable conformal generator.
///
/// Quaternion arrays use the project-wide `(w, x, y, z)` convention.  The
/// generator list is stored in application order: element zero acts first.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConformalGenerator {
    Translation {
        offset: [f64; 3],
    },
    Rotation {
        quaternion_wxyz: [f64; 4],
    },
    /// Nonzero scale. A negative factor reverses orientation in 3D.
    UniformScale {
        factor: f64,
    },
    /// Euclidean inversion/reflection in a sphere.  In three dimensions this
    /// reverses orientation and is an involution.
    SphereReflection {
        center: [f64; 3],
        radius: f64,
    },
}

impl ConformalGenerator {
    pub fn translation(offset: [f64; 3]) -> Self {
        Self::Translation { offset }
    }

    pub fn rotation_axis_angle(axis: [f64; 3], angle: f64) -> Result<Self, ConformalError> {
        if !finite3(axis) || !angle.is_finite() {
            return Err(ConformalError::InvalidGenerator(
                "rotation axis and angle must be finite".into(),
            ));
        }
        let norm_sq = axis.into_iter().map(|x| x * x).sum::<f64>();
        if norm_sq < MIN_ROTATION_NORM_SQ {
            return Err(ConformalError::InvalidGenerator(
                "rotation axis must be nonzero".into(),
            ));
        }
        let inv_norm = norm_sq.sqrt().recip();
        let half = angle * 0.5;
        let s = half.sin();
        Ok(Self::Rotation {
            quaternion_wxyz: [
                half.cos(),
                axis[0] * inv_norm * s,
                axis[1] * inv_norm * s,
                axis[2] * inv_norm * s,
            ],
        })
    }

    pub fn uniform_scale(factor: f64) -> Self {
        Self::UniformScale { factor }
    }

    pub fn sphere_reflection(center: [f64; 3], radius: f64) -> Self {
        Self::SphereReflection { center, radius }
    }

    pub fn validate(&self) -> Result<(), ConformalError> {
        match *self {
            Self::Translation { offset } if !finite3(offset) => Err(
                ConformalError::InvalidGenerator("translation must be finite".into()),
            ),
            Self::Rotation { quaternion_wxyz }
                if !finite4(quaternion_wxyz)
                    || quaternion_wxyz.into_iter().map(|x| x * x).sum::<f64>()
                        < MIN_ROTATION_NORM_SQ =>
            {
                Err(ConformalError::InvalidGenerator(
                    "rotation quaternion must be finite and nonzero".into(),
                ))
            }
            Self::UniformScale { factor } if !factor.is_finite() || factor == 0.0 => Err(
                ConformalError::InvalidGenerator("uniform scale must be finite and nonzero".into()),
            ),
            Self::SphereReflection { center, radius }
                if !finite3(center) || !radius.is_finite() || radius <= 0.0 =>
            {
                Err(ConformalError::InvalidGenerator(
                    "sphere center must be finite and radius must be positive".into(),
                ))
            }
            _ => Ok(()),
        }
    }

    /// Collapse this authoring operation to the renderer's quaternionic
    /// fractional-linear representation.
    pub fn to_mobius(&self) -> Result<Mobius, ConformalError> {
        self.validate()?;
        Ok(match *self {
            Self::Translation { offset: [x, y, z] } => {
                Mobius::translation(Quat::from_point(x, y, z))
            }
            Self::Rotation {
                quaternion_wxyz: [w, x, y, z],
            } => {
                let q = Quat::new(w, x, y, z).normalize();
                // For pure-imaginary points, (q*x)*(q)^-1 = q*x*q-bar.
                Mobius::new(q, Quat::ZERO, Quat::ZERO, q)
            }
            Self::UniformScale { factor } => Mobius::scale(factor),
            Self::SphereReflection {
                center: [x, y, z],
                radius,
            } => Mobius::sphere_reflection(Quat::from_point(x, y, z), radius),
        })
    }

    /// The inverse operation, still represented as an authorable generator.
    pub fn inverse(&self) -> Result<Self, ConformalError> {
        self.validate()?;
        Ok(match *self {
            Self::Translation { offset: [x, y, z] } => Self::translation([-x, -y, -z]),
            Self::Rotation {
                quaternion_wxyz: [w, x, y, z],
            } => {
                let q = Quat::new(w, x, y, z).normalize().conj();
                Self::Rotation {
                    quaternion_wxyz: [q.w, q.x, q.y, q.z],
                }
            }
            Self::UniformScale { factor } => Self::uniform_scale(factor.recip()),
            // Sphere reflection is an involution.
            Self::SphereReflection { center, radius } => Self::SphereReflection { center, radius },
        })
    }

    pub fn orientation_sign(&self) -> i8 {
        match self {
            Self::UniformScale { factor } if *factor < 0.0 => -1,
            Self::SphereReflection { .. } => -1,
            _ => 1,
        }
    }
}

/// A sequence of generators in application order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConformalTransformChain {
    pub generators: Vec<ConformalGenerator>,
}

impl ConformalTransformChain {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn new(generators: Vec<ConformalGenerator>) -> Result<Self, ConformalError> {
        let chain = Self { generators };
        chain.validate()?;
        Ok(chain)
    }

    pub fn validate(&self) -> Result<(), ConformalError> {
        for (index, generator) in self.generators.iter().enumerate() {
            generator
                .validate()
                .map_err(|source| ConformalError::InvalidChain {
                    index,
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }

    /// A chain that applies `self`, followed by `next`.
    pub fn followed_by(&self, next: &Self) -> Self {
        let mut generators = Vec::with_capacity(self.generators.len() + next.generators.len());
        generators.extend_from_slice(&self.generators);
        generators.extend_from_slice(&next.generators);
        Self { generators }
    }

    pub fn inverse(&self) -> Result<Self, ConformalError> {
        let generators = self
            .generators
            .iter()
            .rev()
            .map(ConformalGenerator::inverse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { generators })
    }

    pub fn to_mobius(&self) -> Result<Mobius, ConformalError> {
        self.validate()?;
        self.generators
            .iter()
            .try_fold(Mobius::identity(), |acc, generator| {
                // `compose` means self-after-other; generator zero acts first.
                Ok(generator.to_mobius()?.compose(&acc))
            })
    }

    pub fn apply_point(&self, point: [f64; 3]) -> Result<[f64; 3], ConformalError> {
        if !finite3(point) {
            return Err(ConformalError::NonFinitePoint);
        }
        Ok(self
            .to_mobius()?
            .apply(Quat::from_point(point[0], point[1], point[2]))
            .to_point())
    }

    pub fn orientation_sign(&self) -> i8 {
        self.generators
            .iter()
            .map(ConformalGenerator::orientation_sign)
            .product()
    }
}

/// Stable index of a conformal coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrameId(pub usize);

/// A local-to-parent conformal coordinate frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalFrame {
    pub name: String,
    pub parent: Option<FrameId>,
    pub local_to_parent: ConformalTransformChain,
}

/// A collection of conformal frames with at most one parent per frame.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConformalFrameForest {
    frames: Vec<ConformalFrame>,
}

impl ConformalFrameForest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frames(&self) -> &[ConformalFrame] {
        &self.frames
    }

    pub fn frame(&self, id: FrameId) -> Result<&ConformalFrame, ConformalError> {
        self.frames
            .get(id.0)
            .ok_or(ConformalError::UnknownFrame(id))
    }

    pub fn add_frame(
        &mut self,
        name: impl Into<String>,
        parent: Option<FrameId>,
        local_to_parent: ConformalTransformChain,
    ) -> Result<FrameId, ConformalError> {
        local_to_parent.validate()?;
        if let Some(parent) = parent {
            self.frame(parent)?;
        }
        let id = FrameId(self.frames.len());
        self.frames.push(ConformalFrame {
            name: name.into(),
            parent,
            local_to_parent,
        });
        Ok(id)
    }

    /// Validate references, generator parameters, and the forest invariant.
    pub fn validate(&self) -> Result<(), ConformalError> {
        for (index, frame) in self.frames.iter().enumerate() {
            frame.local_to_parent.validate()?;
            if let Some(parent) = frame.parent {
                self.frame(parent)?;
                if parent.0 == index {
                    return Err(ConformalError::Cycle(FrameId(index)));
                }
            }
            self.walk_to_root(FrameId(index))?;
        }
        Ok(())
    }

    fn walk_to_root(&self, start: FrameId) -> Result<Vec<FrameId>, ConformalError> {
        self.frame(start)?;
        let mut path = Vec::new();
        let mut seen = vec![false; self.frames.len()];
        let mut cursor = Some(start);
        while let Some(id) = cursor {
            if seen[id.0] {
                return Err(ConformalError::Cycle(id));
            }
            seen[id.0] = true;
            path.push(id);
            cursor = self.frame(id)?.parent;
        }
        Ok(path)
    }

    /// The chain mapping coordinates in `frame` to ambient Euclidean space.
    pub fn world_chain(&self, frame: FrameId) -> Result<ConformalTransformChain, ConformalError> {
        let path = self.walk_to_root(frame)?;
        let mut world = ConformalTransformChain::identity();
        for id in path {
            world = world.followed_by(&self.frame(id)?.local_to_parent);
        }
        world.validate()?;
        Ok(world)
    }

    /// The chain mapping coordinates expressed in `from` into coordinates
    /// expressed in `to`.
    pub fn relative_chain(
        &self,
        from: FrameId,
        to: FrameId,
    ) -> Result<ConformalTransformChain, ConformalError> {
        let from_world = self.world_chain(from)?;
        let world_to_to = self.world_chain(to)?.inverse()?;
        Ok(from_world.followed_by(&world_to_to))
    }

    pub fn convert_point(
        &self,
        point: [f64; 3],
        from: FrameId,
        to: FrameId,
    ) -> Result<[f64; 3], ConformalError> {
        self.relative_chain(from, to)?.apply_point(point)
    }

    /// Change a frame's parent while retaining its local-coordinate mapping to
    /// ambient Euclidean space.  Descendants therefore retain their world
    /// mappings as well.
    pub fn reparent_preserve_world(
        &mut self,
        frame: FrameId,
        new_parent: Option<FrameId>,
    ) -> Result<(), ConformalError> {
        self.frame(frame)?;
        if let Some(parent) = new_parent {
            self.frame(parent)?;
            if self.walk_to_root(parent)?.contains(&frame) {
                return Err(ConformalError::Cycle(frame));
            }
        }

        let old_world = self.world_chain(frame)?;
        let new_local = match new_parent {
            Some(parent) => old_world.followed_by(&self.world_chain(parent)?.inverse()?),
            None => old_world,
        };
        let target = self
            .frames
            .get_mut(frame.0)
            .ok_or(ConformalError::UnknownFrame(frame))?;
        target.parent = new_parent;
        target.local_to_parent = new_local;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformalError {
    InvalidGenerator(String),
    InvalidChain {
        index: usize,
        source: Box<ConformalError>,
    },
    UnknownFrame(FrameId),
    Cycle(FrameId),
    NonFinitePoint,
}

impl fmt::Display for ConformalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGenerator(message) => write!(f, "invalid conformal generator: {message}"),
            Self::InvalidChain { index, source } => {
                write!(f, "invalid generator at chain index {index}: {source}")
            }
            Self::UnknownFrame(id) => write!(f, "unknown conformal frame {}", id.0),
            Self::Cycle(id) => write!(f, "conformal frame cycle through {}", id.0),
            Self::NonFinitePoint => write!(f, "conformal point coordinates must be finite"),
        }
    }
}

impl Error for ConformalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidChain { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-8;

    fn assert_point_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < EPS,
                "axis {axis}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    fn chain(generators: Vec<ConformalGenerator>) -> ConformalTransformChain {
        ConformalTransformChain::new(generators).unwrap()
    }

    #[test]
    fn generator_words_use_application_order() {
        let transform = chain(vec![
            ConformalGenerator::translation([1.0, 0.0, 0.0]),
            ConformalGenerator::uniform_scale(2.0),
        ]);
        assert_point_close(
            transform.apply_point([1.0, 0.0, 0.0]).unwrap(),
            [4.0, 0.0, 0.0],
        );
    }

    #[test]
    fn inverse_word_round_trips_a_point() {
        let transform = chain(vec![
            ConformalGenerator::translation([1.5, -0.5, 2.0]),
            ConformalGenerator::rotation_axis_angle([0.0, 0.0, 1.0], 0.7).unwrap(),
            ConformalGenerator::uniform_scale(3.0),
            ConformalGenerator::sphere_reflection([0.25, 0.5, -0.25], 2.0),
        ]);
        let point = [0.4, -1.2, 2.5];
        let transformed = transform.apply_point(point).unwrap();
        assert_point_close(
            transform
                .inverse()
                .unwrap()
                .apply_point(transformed)
                .unwrap(),
            point,
        );
    }

    #[test]
    fn orientation_parity_counts_all_reversing_generators() {
        let one = chain(vec![ConformalGenerator::sphere_reflection([0.0; 3], 1.0)]);
        let two = one.followed_by(&one);
        let negative_scale = chain(vec![ConformalGenerator::uniform_scale(-2.0)]);
        let proper_composition = one.followed_by(&negative_scale);
        assert_eq!(one.orientation_sign(), -1);
        assert_eq!(two.orientation_sign(), 1);
        assert_eq!(negative_scale.orientation_sign(), -1);
        assert_eq!(proper_composition.orientation_sign(), 1);
    }

    #[test]
    fn frame_world_and_relative_coordinates_are_unambiguous() {
        let mut frames = ConformalFrameForest::new();
        let parent = frames
            .add_frame(
                "parent",
                None,
                chain(vec![ConformalGenerator::translation([10.0, 0.0, 0.0])]),
            )
            .unwrap();
        let child = frames
            .add_frame(
                "child",
                Some(parent),
                chain(vec![ConformalGenerator::uniform_scale(2.0)]),
            )
            .unwrap();
        assert_point_close(
            frames
                .world_chain(child)
                .unwrap()
                .apply_point([1.0, 0.0, 0.0])
                .unwrap(),
            [12.0, 0.0, 0.0],
        );
        assert_point_close(
            frames
                .convert_point([1.0, 0.0, 0.0], child, parent)
                .unwrap(),
            [2.0, 0.0, 0.0],
        );
    }

    #[test]
    fn preserve_world_reparent_keeps_frame_and_descendant_mappings() {
        let mut frames = ConformalFrameForest::new();
        let left = frames
            .add_frame(
                "left",
                None,
                chain(vec![ConformalGenerator::translation([10.0, 0.0, 0.0])]),
            )
            .unwrap();
        let right = frames
            .add_frame(
                "right",
                None,
                chain(vec![ConformalGenerator::translation([-5.0, 2.0, 0.0])]),
            )
            .unwrap();
        let moving = frames
            .add_frame(
                "moving",
                Some(left),
                chain(vec![ConformalGenerator::uniform_scale(2.0)]),
            )
            .unwrap();
        let descendant = frames
            .add_frame(
                "descendant",
                Some(moving),
                chain(vec![ConformalGenerator::translation([0.0, 3.0, 0.0])]),
            )
            .unwrap();

        let sample = [0.7, -0.2, 0.4];
        let before = frames
            .world_chain(moving)
            .unwrap()
            .apply_point(sample)
            .unwrap();
        let descendant_before = frames
            .world_chain(descendant)
            .unwrap()
            .apply_point(sample)
            .unwrap();
        frames.reparent_preserve_world(moving, Some(right)).unwrap();
        let after = frames
            .world_chain(moving)
            .unwrap()
            .apply_point(sample)
            .unwrap();
        let descendant_after = frames
            .world_chain(descendant)
            .unwrap()
            .apply_point(sample)
            .unwrap();
        assert_point_close(after, before);
        assert_point_close(descendant_after, descendant_before);
    }

    #[test]
    fn reparent_rejects_cycles() {
        let mut frames = ConformalFrameForest::new();
        let root = frames
            .add_frame("root", None, ConformalTransformChain::identity())
            .unwrap();
        let child = frames
            .add_frame("child", Some(root), ConformalTransformChain::identity())
            .unwrap();
        assert_eq!(
            frames.reparent_preserve_world(root, Some(child)),
            Err(ConformalError::Cycle(root))
        );
    }

    #[test]
    fn frame_forest_serialization_preserves_authoring_words() {
        let mut frames = ConformalFrameForest::new();
        frames
            .add_frame(
                "animated",
                None,
                chain(vec![
                    ConformalGenerator::translation([1.0, 2.0, 3.0]),
                    ConformalGenerator::sphere_reflection([0.0, 1.0, 0.0], 2.5),
                ]),
            )
            .unwrap();
        let json = serde_json::to_string(&frames).unwrap();
        let restored: ConformalFrameForest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, frames);
        restored.validate().unwrap();
    }
}
