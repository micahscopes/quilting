//! Framed round walls and their anchor-dependent open sides.
//!
//! A wall is an unoriented sphere or plane.  Contact and crossing belong to
//! that anchor-invariant wall skeleton.  Choosing one of its two strict sides
//! is separate state, and re-anchoring can update that state by sparse XOR.

use crate::conformal::{ConformalError, ConformalFrameForest, FrameId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

fn finite3(v: [f64; 3]) -> bool {
    v.into_iter().all(f64::is_finite)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WallId(pub usize);

/// An unoriented round wall in a Euclidean chart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoundWallGeometry {
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    /// The plane `dot(unit_normal, x) = offset`.
    Plane {
        unit_normal: [f64; 3],
        offset: f64,
    },
}

impl RoundWallGeometry {
    pub fn sphere(center: [f64; 3], radius: f64) -> Result<Self, MereologyError> {
        let geometry = Self::Sphere { center, radius };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn plane(normal: [f64; 3], offset: f64) -> Result<Self, MereologyError> {
        if !finite3(normal) || !offset.is_finite() {
            return Err(MereologyError::InvalidWall(
                "plane normal and offset must be finite".into(),
            ));
        }
        let length = norm(normal);
        if length <= 1.0e-12 {
            return Err(MereologyError::InvalidWall(
                "plane normal must be nonzero".into(),
            ));
        }
        Ok(Self::Plane {
            unit_normal: [normal[0] / length, normal[1] / length, normal[2] / length],
            offset: offset / length,
        })
    }

    pub fn validate(&self) -> Result<(), MereologyError> {
        match *self {
            Self::Sphere { center, radius }
                if !finite3(center) || !radius.is_finite() || radius <= 0.0 =>
            {
                Err(MereologyError::InvalidWall(
                    "sphere center must be finite and radius must be positive".into(),
                ))
            }
            Self::Plane {
                unit_normal,
                offset,
            } if !finite3(unit_normal)
                || !offset.is_finite()
                || (norm(unit_normal) - 1.0).abs() > 1.0e-9 =>
            {
                Err(MereologyError::InvalidWall(
                    "plane normal must be finite and normalized".into(),
                ))
            }
            _ => Ok(()),
        }
    }

    /// A signed defining function.  Its strict negative and positive loci are
    /// the two complementary open sides; zero is the wall itself.
    pub fn signed_value(&self, point: [f64; 3]) -> Result<f64, MereologyError> {
        self.validate()?;
        if !finite3(point) {
            return Err(MereologyError::NonFinitePoint);
        }
        Ok(match *self {
            Self::Sphere { center, radius } => {
                let delta = sub(point, center);
                dot(delta, delta) - radius * radius
            }
            Self::Plane {
                unit_normal,
                offset,
            } => dot(unit_normal, point) - offset,
        })
    }

    /// Signed inversive distance for two positive-radius spheres.  The
    /// absolute value is the anchor-invariant coarse classifier; the sign
    /// distinguishes external separation from nesting in this orientation.
    pub fn signed_inversive_distance(&self, other: &Self) -> Result<f64, MereologyError> {
        let (a, r, b, s) = match (*self, *other) {
            (
                Self::Sphere {
                    center: a,
                    radius: r,
                },
                Self::Sphere {
                    center: b,
                    radius: s,
                },
            ) => (a, r, b, s),
            _ => return Err(MereologyError::SphereOperationOnPlane),
        };
        self.validate()?;
        other.validate()?;
        let delta = sub(a, b);
        Ok((dot(delta, delta) - r * r - s * s) / (2.0 * r * s))
    }

    pub fn relation(
        &self,
        other: &Self,
        epsilon: f64,
    ) -> Result<RoundWallRelation, MereologyError> {
        self.validate()?;
        other.validate()?;
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(MereologyError::InvalidTolerance);
        }

        match (*self, *other) {
            (
                Self::Sphere {
                    center: a,
                    radius: r,
                },
                Self::Sphere {
                    center: b,
                    radius: s,
                },
            ) => {
                let distance = norm(sub(a, b));
                if distance <= epsilon && (r - s).abs() <= epsilon {
                    return Ok(RoundWallRelation::Coincident);
                }
                let outer_sum = r + s;
                if distance > outer_sum + epsilon {
                    return Ok(RoundWallRelation::ExternallySeparated);
                }
                if (distance - outer_sum).abs() <= epsilon {
                    return Ok(RoundWallRelation::Tangent {
                        kind: TangencyKind::External,
                    });
                }
                let radius_difference = (r - s).abs();
                if distance < radius_difference - epsilon {
                    return Ok(RoundWallRelation::Nested {
                        first_inside_second: r < s,
                    });
                }
                if (distance - radius_difference).abs() <= epsilon {
                    return Ok(RoundWallRelation::Tangent {
                        kind: TangencyKind::Internal,
                    });
                }
                Ok(RoundWallRelation::Crossing)
            }
            (
                Self::Sphere { center, radius },
                Self::Plane {
                    unit_normal,
                    offset,
                },
            )
            | (
                Self::Plane {
                    unit_normal,
                    offset,
                },
                Self::Sphere { center, radius },
            ) => {
                let distance = (dot(unit_normal, center) - offset).abs();
                if distance > radius + epsilon {
                    Ok(RoundWallRelation::ExternallySeparated)
                } else if (distance - radius).abs() <= epsilon {
                    Ok(RoundWallRelation::Tangent {
                        kind: TangencyKind::SpherePlane,
                    })
                } else {
                    Ok(RoundWallRelation::Crossing)
                }
            }
            (
                Self::Plane {
                    unit_normal: n1,
                    offset: o1,
                },
                Self::Plane {
                    unit_normal: n2,
                    offset: o2,
                },
            ) => {
                if norm(cross(n1, n2)) > epsilon {
                    return Ok(RoundWallRelation::Crossing);
                }
                let aligned_o2 = if dot(n1, n2) < 0.0 { -o2 } else { o2 };
                if (o1 - aligned_o2).abs() <= epsilon {
                    Ok(RoundWallRelation::Coincident)
                } else {
                    Ok(RoundWallRelation::ParallelSeparated)
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundWall {
    pub name: String,
    pub frame: FrameId,
    pub geometry: RoundWallGeometry,
}

impl RoundWall {
    pub fn validate(&self, frames: &ConformalFrameForest) -> Result<(), MereologyError> {
        frames.frame(self.frame)?;
        self.geometry.validate()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundWallSet {
    walls: Vec<RoundWall>,
}

impl RoundWallSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn walls(&self) -> &[RoundWall] {
        &self.walls
    }

    pub fn wall(&self, id: WallId) -> Result<&RoundWall, MereologyError> {
        self.walls.get(id.0).ok_or(MereologyError::UnknownWall(id))
    }

    pub fn add_wall(
        &mut self,
        frames: &ConformalFrameForest,
        wall: RoundWall,
    ) -> Result<WallId, MereologyError> {
        wall.validate(frames)?;
        let id = WallId(self.walls.len());
        self.walls.push(wall);
        Ok(id)
    }

    pub fn contains(
        &self,
        frames: &ConformalFrameForest,
        side: OpenRoundSide,
        point: [f64; 3],
        point_frame: FrameId,
    ) -> Result<bool, MereologyError> {
        let wall = self.wall(side.wall)?;
        let in_wall_frame = frames.convert_point(point, point_frame, wall.frame)?;
        let value = wall.geometry.signed_value(in_wall_frame)?;
        Ok(match side.orientation {
            RoundSideOrientation::Negative => value < 0.0,
            RoundSideOrientation::Positive => value > 0.0,
        })
    }
}

/// The two strict complementary sides of a wall.  For spheres, `Negative` is
/// the bounded interior in the current Euclidean chart and `Positive` is the
/// exterior containing chart infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundSideOrientation {
    Negative,
    Positive,
}

impl RoundSideOrientation {
    pub fn complement(self) -> Self {
        match self {
            Self::Negative => Self::Positive,
            Self::Positive => Self::Negative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenRoundSide {
    pub wall: WallId,
    pub orientation: RoundSideOrientation,
}

impl OpenRoundSide {
    pub fn complement(self) -> Self {
        Self {
            wall: self.wall,
            orientation: self.orientation.complement(),
        }
    }
}

/// Sparse orientation state attached to a chosen coordinate anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorState {
    pub frame: FrameId,
    flipped_walls: BTreeSet<WallId>,
}

impl AnchorState {
    pub fn new(frame: FrameId) -> Self {
        Self {
            frame,
            flipped_walls: BTreeSet::new(),
        }
    }

    pub fn flipped_walls(&self) -> &BTreeSet<WallId> {
        &self.flipped_walls
    }

    pub fn flip(&mut self, wall: WallId) {
        if !self.flipped_walls.remove(&wall) {
            self.flipped_walls.insert(wall);
        }
    }

    pub fn apply_flip_set(&mut self, changes: &BTreeSet<WallId>) {
        for &wall in changes {
            self.flip(wall);
        }
    }

    pub fn orient(&self, wall: WallId, canonical: RoundSideOrientation) -> RoundSideOrientation {
        if self.flipped_walls.contains(&wall) {
            canonical.complement()
        } else {
            canonical
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TangencyKind {
    External,
    Internal,
    SpherePlane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoundWallRelation {
    Coincident,
    Crossing,
    Tangent { kind: TangencyKind },
    ExternallySeparated,
    Nested { first_inside_second: bool },
    ParallelSeparated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MereologyError {
    InvalidWall(String),
    UnknownWall(WallId),
    InvalidTolerance,
    NonFinitePoint,
    SphereOperationOnPlane,
    Conformal(ConformalError),
}

impl From<ConformalError> for MereologyError {
    fn from(value: ConformalError) -> Self {
        Self::Conformal(value)
    }
}

impl fmt::Display for MereologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWall(message) => write!(f, "invalid round wall: {message}"),
            Self::UnknownWall(id) => write!(f, "unknown round wall {}", id.0),
            Self::InvalidTolerance => {
                write!(f, "relation tolerance must be finite and nonnegative")
            }
            Self::NonFinitePoint => write!(f, "wall query point must be finite"),
            Self::SphereOperationOnPlane => write!(f, "operation requires two sphere walls"),
            Self::Conformal(source) => source.fmt(f),
        }
    }
}

impl Error for MereologyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Conformal(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::{ConformalGenerator, ConformalTransformChain};

    const EPS: f64 = 1.0e-9;

    fn translated_frame() -> (ConformalFrameForest, FrameId, FrameId) {
        let mut frames = ConformalFrameForest::new();
        let world = frames
            .add_frame("world", None, ConformalTransformChain::identity())
            .unwrap();
        let shifted = frames
            .add_frame(
                "shifted",
                Some(world),
                ConformalTransformChain::new(vec![ConformalGenerator::translation([
                    10.0, 0.0, 0.0,
                ])])
                .unwrap(),
            )
            .unwrap();
        (frames, world, shifted)
    }

    #[test]
    fn complementary_open_sides_exclude_the_wall() {
        let sphere = RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap();
        assert!(sphere.signed_value([0.0; 3]).unwrap() < 0.0);
        assert!(sphere.signed_value([2.0, 0.0, 0.0]).unwrap() > 0.0);
        assert_eq!(sphere.signed_value([1.0, 0.0, 0.0]).unwrap(), 0.0);
        assert_eq!(
            RoundSideOrientation::Negative.complement(),
            RoundSideOrientation::Positive
        );
    }

    #[test]
    fn framed_membership_converts_coordinates_before_testing() {
        let (frames, world, shifted) = translated_frame();
        let mut walls = RoundWallSet::new();
        let wall = walls
            .add_wall(
                &frames,
                RoundWall {
                    name: "local sphere".into(),
                    frame: shifted,
                    geometry: RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap(),
                },
            )
            .unwrap();
        let inside = OpenRoundSide {
            wall,
            orientation: RoundSideOrientation::Negative,
        };
        assert!(walls
            .contains(&frames, inside, [10.0, 0.0, 0.0], world)
            .unwrap());
        assert!(!walls
            .contains(&frames, inside, [0.0, 0.0, 0.0], world)
            .unwrap());
    }

    #[test]
    fn sphere_relations_keep_external_and_nested_branches_distinct() {
        let unit = RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap();
        let crossing = RoundWallGeometry::sphere([1.0, 0.0, 0.0], 1.0).unwrap();
        let tangent = RoundWallGeometry::sphere([2.0, 0.0, 0.0], 1.0).unwrap();
        let separated = RoundWallGeometry::sphere([3.0, 0.0, 0.0], 1.0).unwrap();
        let outer = RoundWallGeometry::sphere([0.25, 0.0, 0.0], 3.0).unwrap();

        assert_eq!(
            unit.relation(&crossing, EPS).unwrap(),
            RoundWallRelation::Crossing
        );
        assert_eq!(
            unit.relation(&tangent, EPS).unwrap(),
            RoundWallRelation::Tangent {
                kind: TangencyKind::External
            }
        );
        assert_eq!(
            unit.relation(&separated, EPS).unwrap(),
            RoundWallRelation::ExternallySeparated
        );
        assert_eq!(
            unit.relation(&outer, EPS).unwrap(),
            RoundWallRelation::Nested {
                first_inside_second: true
            }
        );
        assert!(unit.signed_inversive_distance(&separated).unwrap() > 1.0);
        assert!(unit.signed_inversive_distance(&outer).unwrap() < -1.0);
    }

    #[test]
    fn sphere_plane_and_plane_plane_relations_are_supported() {
        let sphere = RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap();
        let cutting = RoundWallGeometry::plane([1.0, 0.0, 0.0], 0.0).unwrap();
        let tangent = RoundWallGeometry::plane([1.0, 0.0, 0.0], 1.0).unwrap();
        let away = RoundWallGeometry::plane([1.0, 0.0, 0.0], 2.0).unwrap();
        let transverse = RoundWallGeometry::plane([0.0, 1.0, 0.0], 0.0).unwrap();
        assert_eq!(
            sphere.relation(&cutting, EPS).unwrap(),
            RoundWallRelation::Crossing
        );
        assert_eq!(
            sphere.relation(&tangent, EPS).unwrap(),
            RoundWallRelation::Tangent {
                kind: TangencyKind::SpherePlane
            }
        );
        assert_eq!(
            sphere.relation(&away, EPS).unwrap(),
            RoundWallRelation::ExternallySeparated
        );
        assert_eq!(
            cutting.relation(&away, EPS).unwrap(),
            RoundWallRelation::ParallelSeparated
        );
        assert_eq!(
            cutting.relation(&transverse, EPS).unwrap(),
            RoundWallRelation::Crossing
        );
    }

    #[test]
    fn anchor_updates_are_sparse_xor() {
        let mut anchor = AnchorState::new(FrameId(0));
        let wall = WallId(7);
        assert_eq!(
            anchor.orient(wall, RoundSideOrientation::Negative),
            RoundSideOrientation::Negative
        );
        anchor.flip(wall);
        assert_eq!(
            anchor.orient(wall, RoundSideOrientation::Negative),
            RoundSideOrientation::Positive
        );
        anchor.flip(wall);
        assert_eq!(
            anchor.orient(wall, RoundSideOrientation::Negative),
            RoundSideOrientation::Negative
        );
        assert!(anchor.flipped_walls().is_empty());
    }
}
