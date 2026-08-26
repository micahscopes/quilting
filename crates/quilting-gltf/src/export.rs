//! Deterministic export of small, static triangle assets.
//!
//! This deliberately covers the narrow interchange subset used by generated
//! educational fixtures: one indexed triangle primitive, positions, smooth
//! normals, one PBR material, one node, and one scene. It is not a general
//! glTF writer and does not silently drop unsupported mesh semantics.

use serde_json::json;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct StaticMeshGlb<'a> {
    pub name: &'a str,
    pub positions: &'a [[f64; 3]],
    pub triangles: &'a [[usize; 3]],
    pub base_color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub enum StaticMeshGlbError {
    EmptyPositions,
    EmptyTriangles,
    InvalidName,
    NonFinitePosition { vertex: usize },
    PositionOutsideF32 { vertex: usize },
    InvalidIndex { face: usize, vertex: usize },
    DegenerateTriangle { face: usize },
    UndefinedVertexNormal { vertex: usize },
    InvalidBaseColor,
    CountOverflow,
    Json(String),
}

impl fmt::Display for StaticMeshGlbError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => formatter.write_str("static GLB requires at least one vertex"),
            Self::EmptyTriangles => {
                formatter.write_str("static GLB requires at least one triangle")
            }
            Self::InvalidName => formatter.write_str("static GLB name must be nonempty"),
            Self::NonFinitePosition { vertex } => {
                write!(formatter, "static GLB vertex {vertex} is non-finite")
            }
            Self::PositionOutsideF32 { vertex } => {
                write!(
                    formatter,
                    "static GLB vertex {vertex} is outside finite f32 range"
                )
            }
            Self::InvalidIndex { face, vertex } => write!(
                formatter,
                "static GLB face {face} references missing vertex {vertex}",
            ),
            Self::DegenerateTriangle { face } => {
                write!(formatter, "static GLB face {face} is degenerate")
            }
            Self::UndefinedVertexNormal { vertex } => {
                write!(
                    formatter,
                    "static GLB vertex {vertex} has no defined normal"
                )
            }
            Self::InvalidBaseColor => {
                formatter.write_str("static GLB base color must contain finite values in [0, 1]")
            }
            Self::CountOverflow => formatter.write_str("static GLB exceeds GLB 2.0 count limits"),
            Self::Json(error) => write!(formatter, "static GLB JSON encoding failed: {error}"),
        }
    }
}

impl std::error::Error for StaticMeshGlbError {}

/// Encode one validated static mesh as a self-contained GLB 2.0 byte stream.
pub fn encode_static_mesh_glb(input: StaticMeshGlb<'_>) -> Result<Vec<u8>, StaticMeshGlbError> {
    let name = input.name.trim();
    if name.is_empty() {
        return Err(StaticMeshGlbError::InvalidName);
    }
    if input.positions.is_empty() {
        return Err(StaticMeshGlbError::EmptyPositions);
    }
    if input.triangles.is_empty() {
        return Err(StaticMeshGlbError::EmptyTriangles);
    }
    if input
        .base_color
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(StaticMeshGlbError::InvalidBaseColor);
    }

    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    let mut positions = Vec::with_capacity(input.positions.len());
    for (vertex, position) in input.positions.iter().copied().enumerate() {
        if position.iter().any(|value| !value.is_finite()) {
            return Err(StaticMeshGlbError::NonFinitePosition { vertex });
        }
        let converted = position.map(|value| value as f32);
        if converted.iter().any(|value| !value.is_finite()) {
            return Err(StaticMeshGlbError::PositionOutsideF32 { vertex });
        }
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(converted[axis]);
            maximum[axis] = maximum[axis].max(converted[axis]);
        }
        positions.push(converted);
    }

    let mut normal_sums = vec![[0.0_f64; 3]; input.positions.len()];
    for (face, triangle) in input.triangles.iter().copied().enumerate() {
        for vertex in triangle {
            if vertex >= input.positions.len() {
                return Err(StaticMeshGlbError::InvalidIndex { face, vertex });
            }
        }
        let a = input.positions[triangle[0]];
        let b = input.positions[triangle[1]];
        let c = input.positions[triangle[2]];
        let normal = cross(subtract(b, a), subtract(c, a));
        let length_squared = dot(normal, normal);
        if !length_squared.is_finite() || length_squared <= f64::EPSILON {
            return Err(StaticMeshGlbError::DegenerateTriangle { face });
        }
        for vertex in triangle {
            for axis in 0..3 {
                normal_sums[vertex][axis] += normal[axis];
            }
        }
    }
    let normals = normal_sums
        .into_iter()
        .enumerate()
        .map(|(vertex, normal)| {
            let length = dot(normal, normal).sqrt();
            if !length.is_finite() || length <= f64::EPSILON {
                return Err(StaticMeshGlbError::UndefinedVertexNormal { vertex });
            }
            Ok(normal.map(|component| (component / length) as f32))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let vertex_count =
        u32::try_from(positions.len()).map_err(|_| StaticMeshGlbError::CountOverflow)?;
    let index_count = input
        .triangles
        .len()
        .checked_mul(3)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or(StaticMeshGlbError::CountOverflow)?;
    let maximum_index = vertex_count
        .checked_sub(1)
        .ok_or(StaticMeshGlbError::CountOverflow)?;

    let mut binary = Vec::new();
    for position in &positions {
        extend_f32x3(&mut binary, *position);
    }
    let positions_length = binary.len();
    let normals_offset = positions_length;
    for normal in &normals {
        extend_f32x3(&mut binary, *normal);
    }
    let normals_length = binary.len() - normals_offset;
    let indices_offset = binary.len();
    for triangle in input.triangles {
        for vertex in triangle {
            let index = u32::try_from(*vertex).map_err(|_| StaticMeshGlbError::CountOverflow)?;
            binary.extend_from_slice(&index.to_le_bytes());
        }
    }
    let indices_length = binary.len() - indices_offset;
    let binary_length = binary.len();
    pad_to_four(&mut binary, 0);

    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": "quilting-gltf deterministic static mesh exporter",
            "extras": { "title": name }
        },
        "scene": 0,
        "scenes": [{ "name": name, "nodes": [0] }],
        "nodes": [{ "name": name, "mesh": 0 }],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1 },
                "indices": 2,
                "material": 0,
                "mode": 4
            }]
        }],
        "materials": [{
            "name": "Quilting neutral",
            "pbrMetallicRoughness": {
                "baseColorFactor": input.base_color,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.62
            }
        }],
        "buffers": [{ "byteLength": binary_length }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": positions_length, "target": 34962 },
            { "buffer": 0, "byteOffset": normals_offset, "byteLength": normals_length, "target": 34962 },
            { "buffer": 0, "byteOffset": indices_offset, "byteLength": indices_length, "target": 34963 }
        ],
        "accessors": [
            {
                "bufferView": 0, "byteOffset": 0, "componentType": 5126,
                "count": vertex_count, "type": "VEC3", "min": minimum, "max": maximum
            },
            {
                "bufferView": 1, "byteOffset": 0, "componentType": 5126,
                "count": vertex_count, "type": "VEC3"
            },
            {
                "bufferView": 2, "byteOffset": 0, "componentType": 5125,
                "count": index_count, "type": "SCALAR", "min": [0], "max": [maximum_index]
            }
        ]
    });
    let mut json_bytes = serde_json::to_vec(&document)
        .map_err(|error| StaticMeshGlbError::Json(error.to_string()))?;
    pad_to_four(&mut json_bytes, b' ');

    let json_length =
        u32::try_from(json_bytes.len()).map_err(|_| StaticMeshGlbError::CountOverflow)?;
    let binary_chunk_length =
        u32::try_from(binary.len()).map_err(|_| StaticMeshGlbError::CountOverflow)?;
    let total_length = 12_u32
        .checked_add(8)
        .and_then(|length| length.checked_add(json_length))
        .and_then(|length| length.checked_add(8))
        .and_then(|length| length.checked_add(binary_chunk_length))
        .ok_or(StaticMeshGlbError::CountOverflow)?;
    let mut glb = Vec::with_capacity(total_length as usize);
    glb.extend_from_slice(&0x4654_6c67_u32.to_le_bytes());
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&total_length.to_le_bytes());
    glb.extend_from_slice(&json_length.to_le_bytes());
    glb.extend_from_slice(&0x4e4f_534a_u32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&binary_chunk_length.to_le_bytes());
    glb.extend_from_slice(&0x004e_4942_u32.to_le_bytes());
    glb.extend_from_slice(&binary);
    Ok(glb)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn extend_f32x3(buffer: &mut Vec<u8>, value: [f32; 3]) {
    for component in value {
        buffer.extend_from_slice(&component.to_le_bytes());
    }
}

fn pad_to_four(buffer: &mut Vec<u8>, byte: u8) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> StaticMeshGlb<'static> {
        StaticMeshGlb {
            name: "Triangle",
            positions: &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: &[[0, 1, 2]],
            base_color: [0.2, 0.4, 0.8, 1.0],
        }
    }

    #[test]
    fn static_glb_round_trips_through_the_production_loader() {
        let bytes = encode_static_mesh_glb(triangle()).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len(),
        );

        let scene = crate::load_gltf_raw(&bytes).unwrap();
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].primitives.len(), 1);
        let primitive = &scene.meshes[0].primitives[0];
        assert_eq!(primitive.positions.len(), 3);
        assert_eq!(primitive.triangles, [[0, 1, 2]]);
        assert_eq!(primitive.normals.as_ref().unwrap().len(), 3);
        assert_eq!(scene.materials.len(), 1);
        assert_eq!(scene.asset_metadata.title.as_deref(), Some("Triangle"));
        assert_eq!(
            scene.asset_metadata.generator.as_deref(),
            Some("quilting-gltf deterministic static mesh exporter"),
        );
    }

    #[test]
    fn static_glb_rejects_invalid_geometry_and_materials() {
        let mut invalid = triangle();
        invalid.triangles = &[[0, 1, 3]];
        assert_eq!(
            encode_static_mesh_glb(invalid),
            Err(StaticMeshGlbError::InvalidIndex { face: 0, vertex: 3 }),
        );

        invalid = triangle();
        invalid.base_color[0] = f32::NAN;
        assert_eq!(
            encode_static_mesh_glb(invalid),
            Err(StaticMeshGlbError::InvalidBaseColor),
        );
    }
}
