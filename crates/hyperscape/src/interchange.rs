//! Conversion from `quilting-gltf`'s validated extras into ECS entities.

use crate::{
    ActiveAnchor, ChamberSignature, ConformalPath, ConformalScene, CrossFrameTarget, EntityFrame,
    EuclideanCoordinates, EuclideanModelMatrix, LocalCoordinates, PathKeyframe, ProjectionCamera,
    RenderSubject, TrackedCoordinates,
};
use bevy_ecs::prelude::*;
use quilting_gltf::hyperscape::{HyperscapeConstraint, HyperscapeGltfError};
use quilting_gltf::scene::Transform;
use quilting_gltf::GltfScene;
use std::error::Error;
use std::fmt;

/// Stable link back to the ordinary glTF node array.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfNodeIndex(pub usize);

/// Spawn one ECS entity per ordinary glTF node, then attach conformal data to
/// the nodes that opt into `extras.hyperscape`. The ordinary node hierarchy and
/// affine matrices remain available and are not repurposed as frame parents.
pub fn spawn_gltf_hyperscape(
    world: &mut World,
    scene: &GltfScene,
) -> Result<Vec<Entity>, HyperscapeImportError> {
    let asset = scene
        .hyperscape
        .as_ref()
        .ok_or(HyperscapeImportError::MissingPayload)?;
    let runtime = asset.validate()?;

    let mut entities = Vec::with_capacity(scene.nodes.len());
    for (node_index, node) in scene.nodes.iter().enumerate() {
        let matrix_f64 = node.transform.to_matrix();
        let mut matrix = [0.0_f32; 16];
        for (target, source) in matrix.iter_mut().zip(matrix_f64) {
            *target = source as f32;
        }
        let entity = world
            .spawn((GltfNodeIndex(node_index), EuclideanModelMatrix(matrix)))
            .id();
        if node.mesh.is_some() {
            world.entity_mut(entity).insert(RenderSubject);
        }

        if let Some(binding) = &asset.node_bindings[node_index] {
            let local_point = binding
                .path
                .and_then(|path| asset.payload.paths.get(path))
                .and_then(|path| path.keyframes.first())
                .map(|key| key.point)
                .unwrap_or_else(|| node_origin(&node.transform));
            world.entity_mut(entity).insert((
                EntityFrame(quilting_core::FrameId(binding.frame)),
                LocalCoordinates(local_point),
                EuclideanCoordinates([0.0; 3]),
                ChamberSignature::default(),
            ));
            if let Some(anchor) = binding.anchor {
                world
                    .entity_mut(entity)
                    .insert(ActiveAnchor(runtime.anchors[anchor].clone()));
            }
            if let Some(path) = binding.path {
                let authored = &asset.payload.paths[path];
                world.entity_mut(entity).insert(ConformalPath {
                    keyframes: authored
                        .keyframes
                        .iter()
                        .map(|key| PathKeyframe {
                            time_seconds: key.time_seconds,
                            point: key.point,
                        })
                        .collect(),
                    looping: authored.looping,
                });
            }
        }
        entities.push(entity);
    }

    for constraint in &asset.payload.constraints {
        match *constraint {
            HyperscapeConstraint::Track {
                node,
                target_node,
                local_offset,
            } => {
                world.entity_mut(entities[node]).insert((
                    CrossFrameTarget {
                        target: entities[target_node],
                        local_offset,
                    },
                    TrackedCoordinates([0.0; 3]),
                ));
            }
            HyperscapeConstraint::ProjectionCamera { node, frame } => {
                world.entity_mut(entities[node]).insert(ProjectionCamera {
                    frame: quilting_core::FrameId(frame),
                });
            }
        }
    }

    world.insert_resource(ConformalScene {
        frames: runtime.frames,
        walls: runtime.walls,
    });
    Ok(entities)
}

fn node_origin(transform: &Transform) -> [f64; 3] {
    let matrix = transform.to_matrix();
    [matrix[12], matrix[13], matrix[14]]
}

#[derive(Debug)]
pub enum HyperscapeImportError {
    MissingPayload,
    InvalidPayload(HyperscapeGltfError),
}

impl From<HyperscapeGltfError> for HyperscapeImportError {
    fn from(value: HyperscapeGltfError) -> Self {
        Self::InvalidPayload(value)
    }
}

impl fmt::Display for HyperscapeImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPayload => write!(formatter, "glTF scene has no Hyperscape payload"),
            Self::InvalidPayload(error) => error.fmt(formatter),
        }
    }
}

impl Error for HyperscapeImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidPayload(error) => Some(error),
            Self::MissingPayload => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HyperscapePlugin, HyperscopeExtraction};
    use bevy_app::App;

    #[test]
    fn checked_in_gltf_spawns_paths_anchors_constraints_and_extraction() {
        let scene =
            quilting_gltf::load_gltf(include_bytes!("../../../examples/hyperscape-track.gltf"))
                .unwrap();
        let mut app = App::new();
        app.add_plugins(HyperscapePlugin);
        let entities = spawn_gltf_hyperscape(app.world_mut(), &scene).unwrap();
        assert_eq!(entities.len(), 3);

        let horse = entities[0];
        let camera = entities[1];
        assert!(app.world().get::<ConformalPath>(horse).is_some());
        assert!(app.world().get::<ActiveAnchor>(horse).is_some());
        assert_eq!(
            app.world().get::<GltfNodeIndex>(camera),
            Some(&GltfNodeIndex(1))
        );
        assert!(app.world().get::<CrossFrameTarget>(camera).is_some());
        assert!(app.world().get::<ProjectionCamera>(camera).is_some());
        assert_eq!(
            app.world().get::<EuclideanModelMatrix>(horse).unwrap().0[12],
            -2.0
        );

        // The fixture has no mesh buffers, so opt the fallback horse into this
        // extraction test explicitly.
        app.world_mut().entity_mut(horse).insert(RenderSubject);
        app.update();
        let extraction = app.world().resource::<HyperscopeExtraction>();
        assert_eq!(extraction.0.len(), 1);
        assert_eq!(extraction.0[0].subject, horse);
        assert_eq!(extraction.0[0].camera, camera);
        assert_eq!(extraction.0[0].euclidean_model[12], -2.0);
    }
}
