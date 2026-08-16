//! Conversion from `quilting-gltf`'s validated extras into ECS entities.

use crate::{
    ActiveAnchor, ChamberSignature, ConformalPath, ConformalScene, CrossFrameTarget, EntityFrame,
    EuclideanCoordinates, EuclideanModelMatrix, LocalCoordinates, PathKeyframe, ProjectionCamera,
    RenderSubject, TrackedCoordinates,
};
use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_time::{Time, TimeUpdateStrategy, Virtual};
use quilting_gltf::hyperscape::{HyperscapeConstraint, HyperscapeGltfError};
use quilting_gltf::scene::Transform;
use quilting_gltf::GltfScene;
use std::error::Error;
use std::fmt;
use std::time::Duration;

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
    spawn_hyperscape_asset(world, &scene.nodes, asset)
}

/// Lower-level form used by the inexpensive main-thread graph loader.
pub fn spawn_hyperscape_asset(
    world: &mut World,
    nodes: &[quilting_gltf::scene::Node],
    asset: &quilting_gltf::hyperscape::HyperscapeAsset,
) -> Result<Vec<Entity>, HyperscapeImportError> {
    if nodes.len() != asset.node_bindings.len() {
        return Err(HyperscapeImportError::NodeCount {
            nodes: nodes.len(),
            bindings: asset.node_bindings.len(),
        });
    }
    let runtime = asset.validate()?;
    let world_matrices = quilting_gltf::scene::compute_all_world_transforms(nodes);

    let mut entities = Vec::with_capacity(nodes.len());
    for (node_index, node) in nodes.iter().enumerate() {
        let matrix_f64 = world_matrices[node_index];
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

/// A self-contained, main-thread-friendly Hyperscape application. It exposes
/// extracted packets without exposing Bevy to browser bindings.
pub struct HyperscapeGltfRuntime {
    app: App,
    entities: Vec<Entity>,
}

/// One extracted renderer packet with stable ordinary glTF identities for
/// both ends of the subject/view relation.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfHyperscopePacket {
    pub subject_node: usize,
    pub camera_node: usize,
    pub packet: crate::HyperscopePacket,
}

impl HyperscapeGltfRuntime {
    pub fn new(
        nodes: &[quilting_gltf::scene::Node],
        asset: &quilting_gltf::hyperscape::HyperscapeAsset,
    ) -> Result<Self, HyperscapeImportError> {
        let mut app = App::new();
        app.add_plugins(crate::HyperscapePlugin)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
            .insert_resource(Time::<Virtual>::from_max_delta(Duration::MAX));
        let entities = spawn_hyperscape_asset(app.world_mut(), nodes, asset)?;
        // Establish Bevy's real/virtual clock origin at authored time zero.
        app.update();
        Ok(Self { app, entities })
    }

    pub fn tick(&mut self, delta: Duration) {
        *self.app.world_mut().resource_mut::<TimeUpdateStrategy>() =
            TimeUpdateStrategy::ManualDuration(delta);
        self.app.update();
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn packets(&self) -> &[crate::HyperscopePacket] {
        &self.app.world().resource::<crate::HyperscopeExtraction>().0
    }

    /// Renderer packets keyed by stable ordinary glTF subject and camera node
    /// indices. Retaining the camera identity prevents multi-view extraction
    /// from silently overwriting one view with another.
    pub fn packets_by_node(&self) -> Vec<GltfHyperscopePacket> {
        self.packets()
            .iter()
            .filter_map(|packet| {
                let subject_node = self.app.world().get::<GltfNodeIndex>(packet.subject)?.0;
                let camera_node = self.app.world().get::<GltfNodeIndex>(packet.camera)?.0;
                Some(GltfHyperscopePacket {
                    subject_node,
                    camera_node,
                    packet: packet.clone(),
                })
            })
            .collect()
    }

    pub fn diagnostics(&self) -> &[String] {
        &self
            .app
            .world()
            .resource::<crate::HyperscapeDiagnostics>()
            .0
    }

    pub fn app(&self) -> &App {
        &self.app
    }

    pub fn app_mut(&mut self) -> &mut App {
        &mut self.app
    }
}

fn node_origin(transform: &Transform) -> [f64; 3] {
    let matrix = transform.to_matrix();
    [matrix[12], matrix[13], matrix[14]]
}

#[derive(Debug)]
pub enum HyperscapeImportError {
    MissingPayload,
    NodeCount { nodes: usize, bindings: usize },
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
            Self::NodeCount { nodes, bindings } => write!(
                formatter,
                "ordinary node count {nodes} does not match binding count {bindings}"
            ),
            Self::InvalidPayload(error) => error.fmt(formatter),
        }
    }
}

impl Error for HyperscapeImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidPayload(error) => Some(error),
            Self::MissingPayload | Self::NodeCount { .. } => None,
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
        assert_eq!(extraction.0[0].camera_eye, [0.0, -8.0, 3.0]);
        assert!(extraction.0[0].camera_target.is_some());
    }

    #[test]
    fn self_contained_runtime_ticks_authored_time() {
        let bytes = include_bytes!("../../../examples/hyperscape-track.gltf");
        let (nodes, asset) = quilting_gltf::load_hyperscape_graph(bytes).unwrap();
        let mut runtime = HyperscapeGltfRuntime::new(&nodes, &asset.unwrap()).unwrap();
        assert_eq!(runtime.packets().len(), 0);
        let horse = runtime.entities()[0];
        runtime
            .app_mut()
            .world_mut()
            .entity_mut(horse)
            .insert(RenderSubject);
        let second_camera = runtime.entities()[2];
        runtime
            .app_mut()
            .world_mut()
            .entity_mut(second_camera)
            .insert(ProjectionCamera {
                frame: quilting_core::FrameId(1),
            });
        runtime.tick(Duration::from_secs(1));
        assert_eq!(runtime.packets().len(), 2);
        let keyed = runtime.packets_by_node();
        assert_eq!(keyed.len(), 2);
        assert_eq!(keyed[0].subject_node, 0);
        assert_eq!(keyed[1].subject_node, 0);
        assert_eq!(
            keyed
                .iter()
                .map(|packet| packet.camera_node)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([1, 2])
        );
        assert!(runtime.diagnostics().is_empty());
        assert_eq!(
            runtime
                .app()
                .world()
                .get::<LocalCoordinates>(horse)
                .unwrap()
                .0,
            [-1.0, 0.0, 0.0]
        );
        assert_eq!(
            runtime
                .app()
                .world()
                .get::<EuclideanModelMatrix>(horse)
                .unwrap()
                .0[12],
            -1.0
        );
    }
}
