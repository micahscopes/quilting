//! Deterministic ordinary-affine extraction for packed renderer nodes.
//!
//! This boundary joins durable entity identity to runtime node handles without
//! giving either a browser or a transport scene authority. Matrices are
//! column-major. A live [`WireTransform`] is an absolute transform in the
//! asset's ordinary source-world chart: it replaces the flattened glTF world
//! matrix for that entity, then the presentation layer remains the outermost
//! affine transform. Conformal frame extraction is deliberately separate.

use crate::LayerTransform;
use hyperscape_protocol::{AssetEntityId, AssetId, EntityId, WireTransform};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

/// One source node in a packed renderer instance.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedNodeSource {
    /// Scene-wide renderer handle. It is never durable identity.
    pub packed_node: u32,
    /// Node index in the source asset.
    pub source_node: u32,
    /// Durable identity when the glTF node opted into Hyperscape metadata.
    pub entity: Option<EntityId>,
    /// Flattened glTF world matrix in the asset's ordinary source chart.
    pub source_world: [f32; 16],
}

/// One presentation-layer instance of an ordinary source asset.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedAssetInstance {
    pub layer: Uuid,
    pub asset: AssetId,
    pub layer_transform: LayerTransform,
    pub nodes: Vec<PackedNodeSource>,
}

/// Which ordinary source transform supplied an extracted node matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedNodeTransformSource {
    GltfWorld,
    /// Absolute source-world TRS from the durable authored lane.
    AuthoredAbsolute,
}

/// One backend-neutral ordinary-affine node record.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedNodeTransform {
    pub layer: Uuid,
    pub asset: AssetId,
    pub packed_node: u32,
    pub source_node: u32,
    pub identity: Option<AssetEntityId>,
    pub source: PackedNodeTransformSource,
    pub matrix: [f32; 16],
}

/// Deterministic extraction plus edits that have no resident node yet.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PackedSceneExtraction {
    /// Sorted by scene-wide packed node handle.
    pub nodes: Vec<PackedNodeTransform>,
    /// Sorted durable entity IDs with no binding in the current packed scene.
    pub unmatched_authored_entities: Vec<EntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedSceneError {
    NilLayer,
    DuplicateLayer(Uuid),
    InvalidAsset(AssetId),
    DuplicatePackedNode(u32),
    DuplicateEntityBinding {
        layer: Uuid,
        entity: EntityId,
    },
    AmbiguousEntityBinding {
        entity: EntityId,
        first_asset: AssetId,
        second_asset: AssetId,
    },
    InvalidLayerTransform {
        layer: Uuid,
        reason: &'static str,
    },
    InvalidSourceMatrix {
        layer: Uuid,
        source_node: u32,
    },
    InvalidAuthoredTransform {
        entity: EntityId,
        reason: String,
    },
    MatrixNotRepresentable {
        packed_node: u32,
    },
}

impl fmt::Display for PackedSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NilLayer => write!(formatter, "packed scene layer ID must not be nil"),
            Self::DuplicateLayer(layer) => {
                write!(formatter, "packed scene repeats layer {layer}")
            }
            Self::InvalidAsset(asset) => write!(formatter, "packed scene asset {asset} is invalid"),
            Self::DuplicatePackedNode(node) => {
                write!(formatter, "packed scene repeats renderer node {node}")
            }
            Self::DuplicateEntityBinding { layer, entity } => write!(
                formatter,
                "packed scene layer {layer} repeats entity binding {entity}"
            ),
            Self::AmbiguousEntityBinding {
                entity,
                first_asset,
                second_asset,
            } => write!(
                formatter,
                "authored entity {entity} is ambiguous between assets {first_asset} and {second_asset}"
            ),
            Self::InvalidLayerTransform { layer, reason } => {
                write!(formatter, "packed scene layer {layer} transform is invalid: {reason}")
            }
            Self::InvalidSourceMatrix { layer, source_node } => write!(
                formatter,
                "packed scene layer {layer} source node {source_node} has a non-finite world matrix"
            ),
            Self::InvalidAuthoredTransform { entity, reason } => {
                write!(formatter, "authored entity {entity} transform is invalid: {reason}")
            }
            Self::MatrixNotRepresentable { packed_node } => write!(
                formatter,
                "packed node {packed_node} matrix is not representable by the f32 renderer boundary"
            ),
        }
    }
}

impl Error for PackedSceneError {}

/// Resolve ordinary node matrices without mutating ECS, application, or
/// renderer state.
///
/// Repeating one asset in multiple presentation layers is supported: an
/// authored entity edit is applied to every instance. Reusing one entity UUID
/// across different assets is rejected because protocol v0.1 transform
/// commands carry an entity ID but no asset ID.
pub fn extract_packed_scene(
    instances: &[PackedAssetInstance],
    authored_transforms: &BTreeMap<EntityId, WireTransform>,
) -> Result<PackedSceneExtraction, PackedSceneError> {
    for (&entity, &transform) in authored_transforms {
        entity
            .validate()
            .map_err(|error| PackedSceneError::InvalidAuthoredTransform {
                entity,
                reason: error.to_string(),
            })?;
        transform
            .validate()
            .map_err(|error| PackedSceneError::InvalidAuthoredTransform {
                entity,
                reason: error.to_string(),
            })?;
    }
    let mut layers = BTreeSet::new();
    let mut packed_nodes = BTreeSet::new();
    let mut bound_entities = BTreeMap::<EntityId, AssetId>::new();
    let mut matched_entities = BTreeSet::new();
    let mut nodes = Vec::new();

    for instance in instances {
        if instance.layer.is_nil() {
            return Err(PackedSceneError::NilLayer);
        }
        if !layers.insert(instance.layer) {
            return Err(PackedSceneError::DuplicateLayer(instance.layer));
        }
        if instance.asset.validate().is_err() {
            return Err(PackedSceneError::InvalidAsset(instance.asset));
        }
        let layer_matrix = trs_matrix(
            instance.layer_transform.translation,
            instance.layer_transform.rotation,
            instance.layer_transform.scale,
        )
        .map_err(|reason| PackedSceneError::InvalidLayerTransform {
            layer: instance.layer,
            reason,
        })?;
        let mut layer_entities = BTreeSet::new();

        for node in &instance.nodes {
            if !packed_nodes.insert(node.packed_node) {
                return Err(PackedSceneError::DuplicatePackedNode(node.packed_node));
            }
            if node.source_world.iter().any(|value| !value.is_finite()) {
                return Err(PackedSceneError::InvalidSourceMatrix {
                    layer: instance.layer,
                    source_node: node.source_node,
                });
            }

            let identity = node
                .entity
                .map(|entity| {
                    if entity.validate().is_err() {
                        return Err(PackedSceneError::InvalidAuthoredTransform {
                            entity,
                            reason: "entity ID must not be nil".to_owned(),
                        });
                    }
                    if !layer_entities.insert(entity) {
                        return Err(PackedSceneError::DuplicateEntityBinding {
                            layer: instance.layer,
                            entity,
                        });
                    }
                    if let Some(first_asset) = bound_entities.insert(entity, instance.asset) {
                        if first_asset != instance.asset {
                            return Err(PackedSceneError::AmbiguousEntityBinding {
                                entity,
                                first_asset,
                                second_asset: instance.asset,
                            });
                        }
                    }
                    AssetEntityId::new(instance.asset, entity).map_err(|error| {
                        PackedSceneError::InvalidAuthoredTransform {
                            entity,
                            reason: error.to_string(),
                        }
                    })
                })
                .transpose()?;

            let (source, ordinary_world) = if let Some(entity) = node.entity {
                if let Some(transform) = authored_transforms.get(&entity) {
                    matched_entities.insert(entity);
                    (
                        PackedNodeTransformSource::AuthoredAbsolute,
                        trs_matrix(
                            transform.translation,
                            transform.rotation_wxyz,
                            transform.scale,
                        )
                        .map_err(|reason| {
                            PackedSceneError::InvalidAuthoredTransform {
                                entity,
                                reason: reason.to_owned(),
                            }
                        })?,
                    )
                } else {
                    (
                        PackedNodeTransformSource::GltfWorld,
                        node.source_world.map(f64::from),
                    )
                }
            } else {
                (
                    PackedNodeTransformSource::GltfWorld,
                    node.source_world.map(f64::from),
                )
            };
            let matrix = checked_f32_matrix(
                multiply_matrix(layer_matrix, ordinary_world),
                node.packed_node,
            )?;
            nodes.push(PackedNodeTransform {
                layer: instance.layer,
                asset: instance.asset,
                packed_node: node.packed_node,
                source_node: node.source_node,
                identity,
                source,
                matrix,
            });
        }
    }

    nodes.sort_by_key(|node| node.packed_node);
    let unmatched_authored_entities = authored_transforms
        .keys()
        .filter(|entity| !matched_entities.contains(entity))
        .copied()
        .collect();
    Ok(PackedSceneExtraction {
        nodes,
        unmatched_authored_entities,
    })
}

fn trs_matrix(
    translation: [f64; 3],
    rotation_wxyz: [f64; 4],
    scale: [f64; 3],
) -> Result<[f64; 16], &'static str> {
    if translation.into_iter().any(|value| !value.is_finite())
        || rotation_wxyz.into_iter().any(|value| !value.is_finite())
        || scale
            .into_iter()
            .any(|value| !value.is_finite() || value == 0.0)
    {
        return Err("TRS values must be finite and scale must be nonzero");
    }
    let norm_squared = rotation_wxyz
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !norm_squared.is_finite() || norm_squared <= 1.0e-24 {
        return Err("rotation must be finite and nondegenerate");
    }
    let inverse_norm = norm_squared.sqrt().recip();
    let [w, x, y, z] = rotation_wxyz.map(|value| value * inverse_norm);
    let [sx, sy, sz] = scale;
    let [tx, ty, tz] = translation;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    Ok([
        (1.0 - 2.0 * (yy + zz)) * sx,
        2.0 * (xy + wz) * sx,
        2.0 * (xz - wy) * sx,
        0.0,
        2.0 * (xy - wz) * sy,
        (1.0 - 2.0 * (xx + zz)) * sy,
        2.0 * (yz + wx) * sy,
        0.0,
        2.0 * (xz + wy) * sz,
        2.0 * (yz - wx) * sz,
        (1.0 - 2.0 * (xx + yy)) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ])
}

fn multiply_matrix(left: [f64; 16], right: [f64; 16]) -> [f64; 16] {
    let mut output = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = left[row] * right[column * 4]
                + left[4 + row] * right[column * 4 + 1]
                + left[8 + row] * right[column * 4 + 2]
                + left[12 + row] * right[column * 4 + 3];
        }
    }
    output
}

fn checked_f32_matrix(matrix: [f64; 16], packed_node: u32) -> Result<[f32; 16], PackedSceneError> {
    let mut output = [0.0_f32; 16];
    for (target, source) in output.iter_mut().zip(matrix) {
        *target = source as f32;
        if !target.is_finite() {
            return Err(PackedSceneError::MatrixNotRepresentable { packed_node });
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(value: u128) -> AssetId {
        AssetId::from_u128(value).unwrap()
    }

    fn entity(value: u128) -> EntityId {
        EntityId::from_u128(value).unwrap()
    }

    fn layer(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn identity_matrix() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn source(packed_node: u32, source_node: u32, entity: Option<EntityId>) -> PackedNodeSource {
        PackedNodeSource {
            packed_node,
            source_node,
            entity,
            source_world: identity_matrix(),
        }
    }

    fn instance(
        layer_id: u128,
        asset_id: u128,
        nodes: Vec<PackedNodeSource>,
    ) -> PackedAssetInstance {
        PackedAssetInstance {
            layer: layer(layer_id),
            asset: asset(asset_id),
            layer_transform: LayerTransform::default(),
            nodes,
        }
    }

    fn assert_matrix_close(actual: [f32; 16], expected: [f32; 16]) {
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "matrix element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn authored_absolute_replaces_source_world_then_layer_is_outermost() {
        let edited = entity(30);
        let mut node = source(7, 3, Some(edited));
        node.source_world[12] = 99.0;
        let mut resident = instance(10, 20, vec![node]);
        resident.layer_transform.translation = [10.0, 0.0, 0.0];
        resident.layer_transform.scale = [2.0, 1.0, 1.0];
        let authored = BTreeMap::from([(
            edited,
            WireTransform {
                translation: [1.0, 2.0, 3.0],
                // Deliberately non-unit: the boundary normalizes wxyz.
                rotation_wxyz: [0.0, 0.0, 0.0, 2.0],
                scale: [-2.0, 3.0, 4.0],
            },
        )]);

        let extraction = extract_packed_scene(&[resident], &authored).unwrap();

        assert_eq!(
            extraction.nodes[0].source,
            PackedNodeTransformSource::AuthoredAbsolute
        );
        assert_matrix_close(
            extraction.nodes[0].matrix,
            [
                4.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 12.0, 2.0, 3.0, 1.0,
            ],
        );
        assert!(extraction.unmatched_authored_entities.is_empty());
    }

    #[test]
    fn extraction_is_packed_node_sorted_and_reports_unmatched_edits() {
        let bound = entity(40);
        let unmatched = entity(41);
        let mut second = source(9, 1, None);
        second.source_world[13] = 5.0;
        let instances = vec![
            instance(12, 21, vec![second]),
            instance(11, 20, vec![source(2, 0, Some(bound))]),
        ];
        let authored = BTreeMap::from([(
            unmatched,
            WireTransform {
                translation: [4.0, 5.0, 6.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                scale: [1.0; 3],
            },
        )]);

        let extraction = extract_packed_scene(&instances, &authored).unwrap();

        assert_eq!(
            extraction
                .nodes
                .iter()
                .map(|node| node.packed_node)
                .collect::<Vec<_>>(),
            vec![2, 9]
        );
        assert_eq!(
            extraction
                .nodes
                .iter()
                .map(|node| node.source)
                .collect::<Vec<_>>(),
            vec![PackedNodeTransformSource::GltfWorld; 2]
        );
        assert_eq!(extraction.unmatched_authored_entities, vec![unmatched]);
    }

    #[test]
    fn one_asset_may_repeat_across_layers_and_receives_the_same_edit() {
        let edited = entity(50);
        let instances = vec![
            instance(11, 20, vec![source(1, 0, Some(edited))]),
            instance(12, 20, vec![source(2, 0, Some(edited))]),
        ];
        let authored = BTreeMap::from([(
            edited,
            WireTransform {
                translation: [8.0, 0.0, 0.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                scale: [1.0; 3],
            },
        )]);

        let extraction = extract_packed_scene(&instances, &authored).unwrap();

        assert_eq!(extraction.nodes.len(), 2);
        assert!(extraction.nodes.iter().all(|node| {
            node.source == PackedNodeTransformSource::AuthoredAbsolute && node.matrix[12] == 8.0
        }));
    }

    #[test]
    fn cross_asset_entity_reuse_is_rejected_as_protocol_ambiguity() {
        let repeated = entity(60);
        let error = extract_packed_scene(
            &[
                instance(11, 20, vec![source(1, 0, Some(repeated))]),
                instance(12, 21, vec![source(2, 0, Some(repeated))]),
            ],
            &BTreeMap::new(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PackedSceneError::AmbiguousEntityBinding {
                entity: repeated,
                first_asset: asset(20),
                second_asset: asset(21),
            }
        );
    }

    #[test]
    fn structural_and_numeric_failures_are_atomic_errors() {
        let duplicate_node = extract_packed_scene(
            &[
                instance(11, 20, vec![source(1, 0, None)]),
                instance(12, 21, vec![source(1, 0, None)]),
            ],
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert_eq!(duplicate_node, PackedSceneError::DuplicatePackedNode(1));

        let mut invalid_source = source(2, 3, None);
        invalid_source.source_world[0] = f32::NAN;
        assert!(matches!(
            extract_packed_scene(&[instance(13, 22, vec![invalid_source])], &BTreeMap::new()),
            Err(PackedSceneError::InvalidSourceMatrix { source_node: 3, .. })
        ));

        let mut overflow = instance(14, 23, vec![source(4, 0, None)]);
        overflow.layer_transform.translation = [f32::MAX as f64 * 2.0, 0.0, 0.0];
        assert_eq!(
            extract_packed_scene(&[overflow], &BTreeMap::new()).unwrap_err(),
            PackedSceneError::MatrixNotRepresentable { packed_node: 4 }
        );

        let invalid_unmatched = BTreeMap::from([(
            entity(70),
            WireTransform {
                translation: [0.0; 3],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                scale: [0.0, 1.0, 1.0],
            },
        )]);
        assert!(matches!(
            extract_packed_scene(&[], &invalid_unmatched),
            Err(PackedSceneError::InvalidAuthoredTransform { .. })
        ));
    }
}
