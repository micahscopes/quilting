//! Conversion from `quilting-gltf`'s validated extras into ECS entities.

use crate::{
    ActiveAnchor, ChamberAggregateState, ChamberKey, ChamberSide, ChamberSignature, ConformalPath,
    ConformalPathTimeline, ConformalScene, ContactRecord, ContactState, CrossFrameTarget,
    EntityFrame, EuclideanCoordinates, EuclideanModelMatrix, LocalCoordinates, PathKeyframe,
    PathTransition, ProjectionCamera, RenderSubject, StableEntityId, TrackedCoordinates,
    SurfaceAddress, SurfaceAttachment, SurfaceFramePin, SurfaceFramePinBinding, SurfaceFramePinSet,
    TransformHistory, TransformHistorySample,
};
use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_time::{Time, Virtual};
use quilting_gltf::hyperscape::{HyperscapeConstraint, HyperscapeGltfError};
use quilting_gltf::scene::Transform;
use quilting_gltf::GltfScene;
use std::collections::BTreeSet;
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
            if let Some(stable_id) = binding.stable_id {
                world.entity_mut(entity).insert(StableEntityId::from(stable_id));
            }
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
                let initial_anchor = binding
                    .anchor
                    .map(|anchor| runtime.anchors[anchor].clone())
                    .unwrap_or_else(|| {
                        quilting_core::AnchorState::new(quilting_core::FrameId(binding.frame))
                    });
                let transitions = authored
                    .transitions
                    .iter()
                    .map(|transition| PathTransition {
                        time_seconds: transition.time_seconds,
                        frame: quilting_core::FrameId(transition.frame),
                        anchor: transition
                            .anchor
                            .map(|anchor| runtime.anchors[anchor].clone())
                            .unwrap_or_else(|| {
                                quilting_core::AnchorState::new(quilting_core::FrameId(
                                    transition.frame,
                                ))
                            }),
                    })
                    .collect();
                world.entity_mut(entity).insert((
                    ConformalPath {
                        keyframes: authored
                            .keyframes
                            .iter()
                            .map(|key| PathKeyframe {
                                time_seconds: key.time_seconds,
                                point: key.point,
                            })
                            .collect(),
                        looping: authored.looping,
                    },
                    ConformalPathTimeline {
                        coordinate_frame: quilting_core::FrameId(
                            authored.coordinate_frame.unwrap_or(binding.frame),
                        ),
                        initial_frame: quilting_core::FrameId(binding.frame),
                        initial_anchor: initial_anchor.clone(),
                        transitions,
                    },
                    ActiveAnchor(initial_anchor),
                ));
            }
        }
        entities.push(entity);
    }

    let mut surface_frame_pins = Vec::new();
    for constraint in &asset.payload.constraints {
        match constraint {
            HyperscapeConstraint::Track {
                node,
                target_node,
                local_offset,
            } => {
                world.entity_mut(entities[*node]).insert((
                    CrossFrameTarget {
                        target: entities[*target_node],
                        local_offset: *local_offset,
                    },
                    TrackedCoordinates([0.0; 3]),
                ));
            }
            HyperscapeConstraint::ProjectionCamera { node, frame } => {
                world.entity_mut(entities[*node]).insert(ProjectionCamera {
                    frame: quilting_core::FrameId(*frame),
                });
            }
            HyperscapeConstraint::SurfacePin {
                frame,
                target_entity,
                face,
                barycentric,
                normal_sign,
                heading_radians,
                uniform_scale,
                orientation,
                local_offset,
            } => {
                let address = SurfaceAddress::new(
                    StableEntityId::from(*target_entity),
                    *face,
                    *barycentric,
                )
                .expect("validated surface address");
                let attachment = SurfaceAttachment::with_normal_sign(address, *normal_sign)
                    .expect("validated surface side");
                surface_frame_pins.push(SurfaceFramePinBinding {
                    frame: quilting_core::FrameId(*frame),
                    pin: SurfaceFramePin {
                        attachment,
                        heading_radians: *heading_radians,
                        uniform_scale: *uniform_scale,
                        orientation: *orientation,
                        local_offset: quilting_core::ConformalTransformChain::new(
                            local_offset.clone(),
                        )
                        .expect("validated surface-pin offset"),
                    },
                });
            }
        }
    }

    world.insert_resource(ConformalScene {
        frames: runtime.frames,
        walls: runtime.walls,
    });
    world.insert_resource(SurfaceFramePinSet(surface_frame_pins));
    Ok(entities)
}

/// A self-contained, main-thread-friendly Hyperscape application. It exposes
/// extracted packets without exposing Bevy to browser bindings.
pub struct HyperscapeGltfRuntime {
    app: App,
    entities: Vec<Entity>,
    node_names: Vec<Option<String>>,
}

/// One extracted renderer packet with stable ordinary glTF identities for
/// both ends of the subject/view relation.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfHyperscopePacket {
    pub subject_node: usize,
    pub camera_node: usize,
    pub packet: crate::HyperscopePacket,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDiagnostic {
    pub frame: usize,
    pub name: String,
    pub parent: Option<usize>,
    pub generator_count: usize,
    pub local_orientation_sign: i8,
    pub world_orientation_sign: i8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityDiagnostic {
    pub node: usize,
    pub name: Option<String>,
    pub frame: usize,
    pub local: [f64; 3],
    pub euclidean: [f64; 3],
    pub anchor_frame: Option<usize>,
    pub flipped_walls: Vec<usize>,
    pub chamber: Vec<(usize, ChamberSide)>,
    pub history: Vec<TransformHistorySample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChamberAggregateDiagnostic {
    pub epoch: u64,
    pub counts: Vec<(ChamberKey, usize)>,
    pub changed_nodes: Vec<usize>,
    pub changed_walls: Vec<usize>,
    pub contact_frontier: Vec<usize>,
    pub classifications_last_tick: usize,
    pub aggregate_updates_last_tick: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibilityHintDiagnostic {
    pub subject_node: usize,
    pub camera_node: usize,
    pub comparable_chambers: bool,
    pub same_chamber: bool,
    pub separating_walls: Vec<usize>,
    pub contact_frontier: Vec<usize>,
    /// Chamber/contact information is only a scheduling hint here. It never
    /// claims that geometry is occluded.
    pub can_cull: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeDiagnosticSnapshot {
    pub elapsed_seconds: f64,
    pub frames: Vec<FrameDiagnostic>,
    pub entities: Vec<EntityDiagnostic>,
    pub contacts: Vec<ContactRecord>,
    pub chamber_aggregates: ChamberAggregateDiagnostic,
    pub visibility_hints: Vec<VisibilityHintDiagnostic>,
    pub transform_history_epoch: u64,
}

impl HyperscapeGltfRuntime {
    pub fn new(
        nodes: &[quilting_gltf::scene::Node],
        asset: &quilting_gltf::hyperscape::HyperscapeAsset,
    ) -> Result<Self, HyperscapeImportError> {
        let mut app = App::new();
        app.add_plugins(crate::HyperscapePlugin)
            .insert_resource(Time::<Virtual>::from_max_delta(Duration::MAX));
        let entities = spawn_hyperscape_asset(app.world_mut(), nodes, asset)?;
        // Resolve the authored scene once at deterministic time zero.
        app.update();
        let node_names = nodes.iter().map(|node| node.name.clone()).collect();
        Ok(Self {
            app,
            entities,
            node_names,
        })
    }

    pub fn tick(&mut self, delta: Duration) {
        self.app
            .world_mut()
            .resource_mut::<Time<Virtual>>()
            .advance_by(delta);
        self.app.update();
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn entity_by_stable_id(&self, stable_id: uuid::Uuid) -> Option<Entity> {
        self.entities.iter().copied().find(|&entity| {
            self.app.world().get::<StableEntityId>(entity).copied()
                == Some(StableEntityId(stable_id))
        })
    }

    pub fn stable_id_for_node(&self, node: usize) -> Option<uuid::Uuid> {
        let entity = *self.entities.get(node)?;
        self.app
            .world()
            .get::<StableEntityId>(entity)
            .map(|stable| stable.0)
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

    pub fn diagnostic_snapshot(&self) -> RuntimeDiagnosticSnapshot {
        let world = self.app.world();
        let scene = world.resource::<ConformalScene>();
        let contacts = world.resource::<ContactState>();
        let aggregates = world.resource::<ChamberAggregateState>();
        let history = world.resource::<TransformHistory>();
        let elapsed_seconds = world.resource::<Time<Virtual>>().elapsed_secs_f64();

        let frames = scene
            .frames
            .frames()
            .iter()
            .enumerate()
            .map(|(frame, authored)| FrameDiagnostic {
                frame,
                name: authored.name.clone(),
                parent: authored.parent.map(|parent| parent.0),
                generator_count: authored.local_to_parent.generators.len(),
                local_orientation_sign: authored.local_to_parent.orientation_sign(),
                world_orientation_sign: scene
                    .frames
                    .world_chain(quilting_core::FrameId(frame))
                    .map(|chain| chain.orientation_sign())
                    .unwrap_or(1),
            })
            .collect();

        let entities = self
            .entities
            .iter()
            .enumerate()
            .filter_map(|(node, &entity)| {
                let frame = world.get::<EntityFrame>(entity)?;
                let local = world.get::<LocalCoordinates>(entity)?;
                let euclidean = world.get::<EuclideanCoordinates>(entity)?;
                let anchor = world.get::<ActiveAnchor>(entity);
                let chamber = world
                    .get::<ChamberSignature>(entity)
                    .map(|signature| {
                        signature
                            .0
                            .iter()
                            .map(|(wall, side)| (wall.0, *side))
                            .collect()
                    })
                    .unwrap_or_default();
                Some(EntityDiagnostic {
                    node,
                    name: self.node_names.get(node).cloned().flatten(),
                    frame: frame.0 .0,
                    local: local.0,
                    euclidean: euclidean.0,
                    anchor_frame: anchor.map(|anchor| anchor.0.frame.0),
                    flipped_walls: anchor
                        .map(|anchor| anchor.0.flipped_walls().iter().map(|wall| wall.0).collect())
                        .unwrap_or_default(),
                    chamber,
                    history: history
                        .samples
                        .get(&entity)
                        .map(|samples| samples.iter().cloned().collect())
                        .unwrap_or_default(),
                })
            })
            .collect();

        let entity_node = |entity: Entity| world.get::<GltfNodeIndex>(entity).map(|node| node.0);
        let chamber_aggregates = ChamberAggregateDiagnostic {
            epoch: aggregates.epoch,
            counts: aggregates
                .counts
                .iter()
                .map(|(key, count)| (key.clone(), *count))
                .collect(),
            changed_nodes: aggregates
                .changed_entities
                .iter()
                .filter_map(|&entity| entity_node(entity))
                .collect(),
            changed_walls: aggregates.changed_walls.iter().map(|wall| wall.0).collect(),
            contact_frontier: aggregates
                .contact_frontier
                .iter()
                .map(|wall| wall.0)
                .collect(),
            classifications_last_tick: aggregates.classifications_last_tick,
            aggregate_updates_last_tick: aggregates.aggregate_updates_last_tick,
        };

        let visibility_hints = self
            .packets_by_node()
            .into_iter()
            .map(|packet| {
                let subject = world.get::<ChamberSignature>(packet.packet.subject);
                let camera = world.get::<ChamberSignature>(packet.packet.camera);
                let comparable_chambers = subject.is_some() && camera.is_some();
                let mut separating = BTreeSet::new();
                if let (Some(subject), Some(camera)) = (subject, camera) {
                    for wall in subject.0.keys().chain(camera.0.keys()) {
                        if subject.0.get(wall) != camera.0.get(wall) {
                            separating.insert(*wall);
                        }
                    }
                }
                let mut frontier = separating.clone();
                for contact in &contacts.0 {
                    if separating.contains(&contact.first) || separating.contains(&contact.second) {
                        frontier.insert(contact.first);
                        frontier.insert(contact.second);
                    }
                }
                VisibilityHintDiagnostic {
                    subject_node: packet.subject_node,
                    camera_node: packet.camera_node,
                    comparable_chambers,
                    same_chamber: comparable_chambers && separating.is_empty(),
                    separating_walls: separating.iter().map(|wall| wall.0).collect(),
                    contact_frontier: frontier.iter().map(|wall| wall.0).collect(),
                    can_cull: false,
                }
            })
            .collect();

        RuntimeDiagnosticSnapshot {
            elapsed_seconds,
            frames,
            entities,
            contacts: contacts.0.clone(),
            chamber_aggregates,
            visibility_hints,
            transform_history_epoch: history.epoch,
        }
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
    use crate::{HyperscapePlugin, HyperscopeExtraction, SurfaceFrameOrientation};
    use bevy_app::App;
    use hyperscape_protocol::{ConformalFrameId, EntityId};
    use quilting_core::ConformalGenerator;

    #[test]
    fn checked_in_gltf_spawns_paths_anchors_constraints_and_extraction() {
        let scene =
            quilting_gltf::load_gltf(quilting_gltf::HYPERSCAPE_TRACK_GLTF).unwrap();
        let mut app = App::new();
        app.add_plugins(HyperscapePlugin);
        let entities = spawn_gltf_hyperscape(app.world_mut(), &scene).unwrap();
        assert_eq!(entities.len(), 3);

        let horse = entities[0];
        let camera = entities[1];
        let horse_id = uuid::Uuid::parse_str("11111111-1111-4111-8111-111111111111").unwrap();
        assert_eq!(
            app.world().get::<StableEntityId>(horse),
            Some(&StableEntityId(horse_id))
        );
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
        let bytes = quilting_gltf::HYPERSCAPE_TRACK_GLTF;
        let (nodes, asset) = quilting_gltf::load_hyperscape_graph(bytes).unwrap();
        let mut runtime = HyperscapeGltfRuntime::new(&nodes, &asset.unwrap()).unwrap();
        assert_eq!(runtime.packets().len(), 0);
        let horse = runtime.entities()[0];
        let horse_id = uuid::Uuid::parse_str("11111111-1111-4111-8111-111111111111").unwrap();
        assert_eq!(runtime.entity_by_stable_id(horse_id), Some(horse));
        assert_eq!(runtime.stable_id_for_node(0), Some(horse_id));
        assert_eq!(runtime.stable_id_for_node(99), None);
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
            [-9.0, 0.0, -1.0]
        );
        assert_eq!(
            runtime
                .app()
                .world()
                .get::<EuclideanCoordinates>(horse)
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
            -9.0
        );
    }

    #[test]
    fn gltf_surface_pin_imports_as_a_typed_constraint_edge() {
        let (nodes, asset) =
            quilting_gltf::load_hyperscape_graph(quilting_gltf::HYPERSCAPE_TRACK_GLTF).unwrap();
        let mut asset = asset.unwrap();
        asset.payload.frames[1].stable_id = Some(ConformalFrameId::from_u128(42).unwrap());
        asset.payload.constraints.push(HyperscapeConstraint::SurfacePin {
            frame: 1,
            target_entity: EntityId::from_u128(0x1111_1111_1111_4111_8111_1111_1111_1111)
                .unwrap(),
            face: 5,
            barycentric: [0.2, 0.3, 0.5],
            normal_sign: -1,
            heading_radians: 0.25,
            uniform_scale: 0.75,
            orientation: SurfaceFrameOrientation::RightSideIn,
            local_offset: vec![ConformalGenerator::sphere_reflection([0.0; 3], 2.0)],
        });

        let mut world = World::new();
        spawn_hyperscape_asset(&mut world, &nodes, &asset).unwrap();
        let pins = world.resource::<SurfaceFramePinSet>();
        assert_eq!(pins.0.len(), 1);
        assert_eq!(pins.0[0].frame, quilting_core::FrameId(1));
        assert_eq!(pins.0[0].pin.attachment.address.face, 5);
        assert_eq!(pins.0[0].pin.attachment.normal_sign, -1);
        assert_eq!(pins.0[0].pin.orientation, SurfaceFrameOrientation::RightSideIn);
        assert_eq!(pins.0[0].pin.local_offset.orientation_sign(), -1);
    }

}
