//! Checked codec and independent validator for the classic Quilting atlas ABI.
//!
//! Schema v1 is intentionally narrow. It contains canonical patch keys, packed
//! patch-local ranges, barycentric vertices, and global triangle indices. It
//! contains no native-width integers, maps, or implementation-specific padding.

use std::fmt;

pub mod fixed_raster_source;

#[cfg(feature = "quilting-export")]
pub mod quilting_export;

#[cfg(all(test, feature = "fe-oracle"))]
mod fe_oracle;

#[cfg(all(test, feature = "raster-oracle"))]
mod raster_oracle;

pub const MAGIC: [u8; 8] = *b"CQATLAS\0";
pub const SCHEMA_VERSION: u32 = 1;
pub const ENDIANNESS_MARKER: u32 = 0x0102_0304;
pub const HEADER_BYTES: usize = 128;
pub const PATCH_BYTES: usize = 32;
pub const VERTEX_BYTES: usize = 16;
pub const TRIANGLE_BYTES: usize = 16;
pub const PAYLOAD_HASH_BYTES: usize = 32;

const HEADER_BYTES_U32: u32 = 128;
const HEADER_RESERVED_WORD_OFFSET: usize = 44;
const HEADER_HASH_OFFSET: usize = 80;
const HEADER_TRAILING_RESERVED_OFFSET: usize = 112;
const BARYCENTRIC_SUM_TOLERANCE: f64 = 2.0e-6;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AtlasKey {
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

impl AtlasKey {
    #[must_use]
    pub const fn new(a: u32, b: u32, c: u32) -> Self {
        Self { a, b, c }
    }

    const fn resolutions(self) -> [u32; 3] {
        [self.a, self.b, self.c]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SourceClass {
    Interior = 0,
    Edge = 1,
    Corner = 2,
}

impl TryFrom<u32> for SourceClass {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Interior),
            1 => Ok(Self::Edge),
            2 => Ok(Self::Corner),
            _ => Err(Error::InvalidSourceClass(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasVertex {
    pub barycentric: [f32; 3],
    pub source_class: SourceClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtlasTriangle {
    /// Global indices into [`Artifact::vertices`].
    pub indices: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtlasPatch {
    pub key: AtlasKey,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_triangle: u32,
    pub triangle_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Artifact {
    pub algorithm_version: u32,
    pub master_seed: u64,
    pub patches: Vec<AtlasPatch>,
    pub vertices: Vec<AtlasVertex>,
    pub triangles: Vec<AtlasTriangle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    ArithmeticOverflow,
    TooShort {
        actual: usize,
    },
    BadMagic,
    UnsupportedSchema(u32),
    UnsupportedAlgorithmVersion(u32),
    BadEndiannessMarker(u32),
    BadHeaderSize(u32),
    NonzeroReserved {
        offset: usize,
        value: u32,
    },
    CountTooLarge {
        field: &'static str,
        value: usize,
    },
    OffsetMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    HashMismatch,
    EmptyPatchTable,
    InvalidKey {
        patch: usize,
        key: AtlasKey,
    },
    NonCanonicalKeyOrder {
        patch: usize,
    },
    NonPackedRange {
        patch: usize,
        field: &'static str,
    },
    InvalidSourceClass(u32),
    InvalidBarycentric {
        vertex: usize,
        reason: &'static str,
    },
    SourceClassMismatch {
        vertex: usize,
    },
    BoundaryCountMismatch {
        patch: usize,
        edge: usize,
    },
    BoundaryParameterMismatch {
        patch: usize,
        edge: usize,
    },
    TriangleOutOfRange {
        patch: usize,
        triangle: usize,
    },
    TriangleRepeatedIndex {
        patch: usize,
        triangle: usize,
    },
    TriangleNotMinFirst {
        patch: usize,
        triangle: usize,
    },
    TriangleNotCounterClockwise {
        patch: usize,
        triangle: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
struct Layout {
    vertex_offset: usize,
    triangle_offset: usize,
    total_bytes: usize,
}

impl Layout {
    fn new(key_count: usize, vertex_count: usize, triangle_count: usize) -> Result<Self, Error> {
        let patch_bytes = key_count
            .checked_mul(PATCH_BYTES)
            .ok_or(Error::ArithmeticOverflow)?;
        let vertex_offset = HEADER_BYTES
            .checked_add(patch_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let vertex_bytes = vertex_count
            .checked_mul(VERTEX_BYTES)
            .ok_or(Error::ArithmeticOverflow)?;
        let triangle_offset = vertex_offset
            .checked_add(vertex_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let triangle_bytes = triangle_count
            .checked_mul(TRIANGLE_BYTES)
            .ok_or(Error::ArithmeticOverflow)?;
        let total_bytes = triangle_offset
            .checked_add(triangle_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(Self {
            vertex_offset,
            triangle_offset,
            total_bytes,
        })
    }

    fn payload_bytes(self) -> usize {
        self.total_bytes - HEADER_BYTES
    }
}

/// Encode an admitted artifact into canonical schema-v1 bytes.
///
/// # Errors
///
/// Returns an [`Error`] when the artifact violates a semantic invariant or a
/// count/offset cannot be represented by schema v1.
pub fn encode(artifact: &Artifact) -> Result<Vec<u8>, Error> {
    validate_artifact(artifact)?;

    let key_count = checked_u32("key_count", artifact.patches.len())?;
    let vertex_count = checked_u32("vertex_count", artifact.vertices.len())?;
    let triangle_count = checked_u32("triangle_count", artifact.triangles.len())?;
    let layout = Layout::new(
        artifact.patches.len(),
        artifact.vertices.len(),
        artifact.triangles.len(),
    )?;
    let mut bytes = vec![0_u8; layout.total_bytes];

    bytes[0..8].copy_from_slice(&MAGIC);
    put_u32(&mut bytes, 8, SCHEMA_VERSION);
    put_u32(&mut bytes, 12, artifact.algorithm_version);
    put_u32(&mut bytes, 16, ENDIANNESS_MARKER);
    put_u32(&mut bytes, 20, HEADER_BYTES_U32);
    put_u64(&mut bytes, 24, artifact.master_seed);
    put_u32(&mut bytes, 32, key_count);
    put_u32(&mut bytes, 36, vertex_count);
    put_u32(&mut bytes, 40, triangle_count);
    put_u64(&mut bytes, 48, HEADER_BYTES as u64);
    put_u64(&mut bytes, 56, layout.vertex_offset as u64);
    put_u64(&mut bytes, 64, layout.triangle_offset as u64);
    put_u64(&mut bytes, 72, layout.payload_bytes() as u64);

    for (index, patch) in artifact.patches.iter().enumerate() {
        let offset = HEADER_BYTES + index * PATCH_BYTES;
        put_u32(&mut bytes, offset, patch.key.a);
        put_u32(&mut bytes, offset + 4, patch.key.b);
        put_u32(&mut bytes, offset + 8, patch.key.c);
        put_u32(&mut bytes, offset + 12, patch.first_vertex);
        put_u32(&mut bytes, offset + 16, patch.vertex_count);
        put_u32(&mut bytes, offset + 20, patch.first_triangle);
        put_u32(&mut bytes, offset + 24, patch.triangle_count);
    }

    for (index, vertex) in artifact.vertices.iter().enumerate() {
        let offset = layout.vertex_offset + index * VERTEX_BYTES;
        for (lane, value) in vertex.barycentric.iter().enumerate() {
            put_u32(&mut bytes, offset + lane * 4, value.to_bits());
        }
        put_u32(&mut bytes, offset + 12, vertex.source_class as u32);
    }

    for (index, triangle) in artifact.triangles.iter().enumerate() {
        let offset = layout.triangle_offset + index * TRIANGLE_BYTES;
        for (lane, value) in triangle.indices.iter().enumerate() {
            put_u32(&mut bytes, offset + lane * 4, *value);
        }
    }

    let digest = sha256(&bytes[HEADER_BYTES..]);
    bytes[HEADER_HASH_OFFSET..HEADER_HASH_OFFSET + PAYLOAD_HASH_BYTES].copy_from_slice(&digest);
    Ok(bytes)
}

/// Decode and fully validate canonical schema-v1 bytes.
///
/// # Errors
///
/// Returns an [`Error`] when the header, layout, hash, records, or semantic
/// topology invariants are invalid.
pub fn decode(bytes: &[u8]) -> Result<Artifact, Error> {
    if bytes.len() < HEADER_BYTES {
        return Err(Error::TooShort {
            actual: bytes.len(),
        });
    }
    if bytes[0..8] != MAGIC {
        return Err(Error::BadMagic);
    }

    let schema_version = get_u32(bytes, 8);
    if schema_version != SCHEMA_VERSION {
        return Err(Error::UnsupportedSchema(schema_version));
    }
    let algorithm_version = get_u32(bytes, 12);
    if algorithm_version == 0 {
        return Err(Error::UnsupportedAlgorithmVersion(algorithm_version));
    }
    let endianness = get_u32(bytes, 16);
    if endianness != ENDIANNESS_MARKER {
        return Err(Error::BadEndiannessMarker(endianness));
    }
    let header_bytes = get_u32(bytes, 20);
    if header_bytes != HEADER_BYTES_U32 {
        return Err(Error::BadHeaderSize(header_bytes));
    }
    require_zero_word(bytes, HEADER_RESERVED_WORD_OFFSET)?;
    for offset in (HEADER_TRAILING_RESERVED_OFFSET..HEADER_BYTES).step_by(4) {
        require_zero_word(bytes, offset)?;
    }

    let key_count = get_u32(bytes, 32) as usize;
    let vertex_count = get_u32(bytes, 36) as usize;
    let triangle_count = get_u32(bytes, 40) as usize;
    let layout = Layout::new(key_count, vertex_count, triangle_count)?;
    require_offset(bytes, 48, "patch_table_offset", HEADER_BYTES)?;
    require_offset(bytes, 56, "vertex_offset", layout.vertex_offset)?;
    require_offset(bytes, 64, "triangle_offset", layout.triangle_offset)?;
    require_offset(bytes, 72, "payload_bytes", layout.payload_bytes())?;
    if bytes.len() != layout.total_bytes {
        return Err(Error::LengthMismatch {
            expected: layout.total_bytes,
            actual: bytes.len(),
        });
    }
    let expected_hash = &bytes[HEADER_HASH_OFFSET..HEADER_HASH_OFFSET + PAYLOAD_HASH_BYTES];
    if sha256(&bytes[HEADER_BYTES..]) != expected_hash {
        return Err(Error::HashMismatch);
    }

    let mut patches = Vec::with_capacity(key_count);
    for index in 0..key_count {
        let offset = HEADER_BYTES + index * PATCH_BYTES;
        require_zero_word(bytes, offset + 28)?;
        patches.push(AtlasPatch {
            key: AtlasKey::new(
                get_u32(bytes, offset),
                get_u32(bytes, offset + 4),
                get_u32(bytes, offset + 8),
            ),
            first_vertex: get_u32(bytes, offset + 12),
            vertex_count: get_u32(bytes, offset + 16),
            first_triangle: get_u32(bytes, offset + 20),
            triangle_count: get_u32(bytes, offset + 24),
        });
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    for index in 0..vertex_count {
        let offset = layout.vertex_offset + index * VERTEX_BYTES;
        vertices.push(AtlasVertex {
            barycentric: [
                f32::from_bits(get_u32(bytes, offset)),
                f32::from_bits(get_u32(bytes, offset + 4)),
                f32::from_bits(get_u32(bytes, offset + 8)),
            ],
            source_class: SourceClass::try_from(get_u32(bytes, offset + 12))?,
        });
    }

    let mut triangles = Vec::with_capacity(triangle_count);
    for index in 0..triangle_count {
        let offset = layout.triangle_offset + index * TRIANGLE_BYTES;
        require_zero_word(bytes, offset + 12)?;
        triangles.push(AtlasTriangle {
            indices: [
                get_u32(bytes, offset),
                get_u32(bytes, offset + 4),
                get_u32(bytes, offset + 8),
            ],
        });
    }

    let artifact = Artifact {
        algorithm_version,
        master_seed: get_u64(bytes, 24),
        patches,
        vertices,
        triangles,
    };
    validate_artifact(&artifact)?;
    Ok(artifact)
}

/// Validate semantic invariants independently of the byte decoder.
///
/// # Errors
///
/// Returns an [`Error`] for the first non-canonical key/range, invalid
/// barycentric boundary, or invalid triangle encountered.
pub fn validate_artifact(artifact: &Artifact) -> Result<(), Error> {
    if artifact.algorithm_version == 0 {
        return Err(Error::UnsupportedAlgorithmVersion(0));
    }
    if artifact.patches.is_empty() {
        return Err(Error::EmptyPatchTable);
    }
    checked_u32("key_count", artifact.patches.len())?;
    checked_u32("vertex_count", artifact.vertices.len())?;
    checked_u32("triangle_count", artifact.triangles.len())?;

    let mut next_vertex = 0_u32;
    let mut next_triangle = 0_u32;
    let mut previous_key = None;

    for (patch_index, patch) in artifact.patches.iter().enumerate() {
        validate_key(patch_index, patch.key, previous_key)?;
        previous_key = Some(patch.key);
        if patch.first_vertex != next_vertex {
            return Err(Error::NonPackedRange {
                patch: patch_index,
                field: "first_vertex",
            });
        }
        if patch.first_triangle != next_triangle {
            return Err(Error::NonPackedRange {
                patch: patch_index,
                field: "first_triangle",
            });
        }
        next_vertex = next_vertex
            .checked_add(patch.vertex_count)
            .ok_or(Error::ArithmeticOverflow)?;
        next_triangle = next_triangle
            .checked_add(patch.triangle_count)
            .ok_or(Error::ArithmeticOverflow)?;

        let vertex_start =
            usize::try_from(patch.first_vertex).map_err(|_| Error::ArithmeticOverflow)?;
        let vertex_end = usize::try_from(next_vertex).map_err(|_| Error::ArithmeticOverflow)?;
        let triangle_start =
            usize::try_from(patch.first_triangle).map_err(|_| Error::ArithmeticOverflow)?;
        let triangle_end = usize::try_from(next_triangle).map_err(|_| Error::ArithmeticOverflow)?;
        let vertices =
            artifact
                .vertices
                .get(vertex_start..vertex_end)
                .ok_or(Error::NonPackedRange {
                    patch: patch_index,
                    field: "vertex_count",
                })?;
        let triangles =
            artifact
                .triangles
                .get(triangle_start..triangle_end)
                .ok_or(Error::NonPackedRange {
                    patch: patch_index,
                    field: "triangle_count",
                })?;

        validate_vertices(vertex_start, vertices)?;
        validate_boundary(patch_index, patch.key, vertices)?;
        validate_triangles(
            patch_index,
            triangle_start,
            patch.first_vertex,
            next_vertex,
            vertices,
            triangles,
        )?;
    }

    if usize::try_from(next_vertex).ok() != Some(artifact.vertices.len()) {
        return Err(Error::NonPackedRange {
            patch: artifact.patches.len() - 1,
            field: "vertex_count",
        });
    }
    if usize::try_from(next_triangle).ok() != Some(artifact.triangles.len()) {
        return Err(Error::NonPackedRange {
            patch: artifact.patches.len() - 1,
            field: "triangle_count",
        });
    }
    Ok(())
}

fn validate_key(index: usize, key: AtlasKey, previous: Option<AtlasKey>) -> Result<(), Error> {
    let resolutions = key.resolutions();
    if resolutions.iter().any(|value| !value.is_power_of_two()) || key.a > key.b || key.b > key.c {
        return Err(Error::InvalidKey { patch: index, key });
    }
    if previous.is_some_and(|prior| prior >= key) {
        return Err(Error::NonCanonicalKeyOrder { patch: index });
    }
    Ok(())
}

fn validate_vertices(global_start: usize, vertices: &[AtlasVertex]) -> Result<(), Error> {
    for (local_index, vertex) in vertices.iter().enumerate() {
        let global_index = global_start + local_index;
        let mut zero_count = 0;
        let mut sum = 0.0_f64;
        for value in vertex.barycentric {
            if !value.is_finite() {
                return Err(Error::InvalidBarycentric {
                    vertex: global_index,
                    reason: "nonfinite component",
                });
            }
            if value.to_bits() == (-0.0_f32).to_bits() {
                return Err(Error::InvalidBarycentric {
                    vertex: global_index,
                    reason: "negative zero",
                });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(Error::InvalidBarycentric {
                    vertex: global_index,
                    reason: "component outside [0, 1]",
                });
            }
            zero_count += usize::from(value == 0.0);
            sum += f64::from(value);
        }
        if (sum - 1.0).abs() > BARYCENTRIC_SUM_TOLERANCE {
            return Err(Error::InvalidBarycentric {
                vertex: global_index,
                reason: "components do not sum to one",
            });
        }
        let expected_class = match zero_count {
            0 => SourceClass::Interior,
            1 => SourceClass::Edge,
            2 => SourceClass::Corner,
            _ => {
                return Err(Error::InvalidBarycentric {
                    vertex: global_index,
                    reason: "more than two zero components",
                });
            }
        };
        if vertex.source_class != expected_class {
            return Err(Error::SourceClassMismatch {
                vertex: global_index,
            });
        }
    }
    Ok(())
}

fn validate_boundary(
    patch_index: usize,
    key: AtlasKey,
    vertices: &[AtlasVertex],
) -> Result<(), Error> {
    for (edge, resolution) in key.resolutions().into_iter().enumerate() {
        let expected_count = usize::try_from(resolution)
            .map_err(|_| Error::ArithmeticOverflow)?
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        if expected_count > vertices.len() {
            return Err(Error::BoundaryCountMismatch {
                patch: patch_index,
                edge,
            });
        }
        let mut parameters = Vec::with_capacity(expected_count);
        for vertex in vertices {
            let bary = vertex.barycentric;
            if bary[edge] == 0.0 {
                let parameter = match edge {
                    0 => bary[2], // BC, B -> C
                    1 => bary[0], // CA, C -> A
                    2 => bary[1], // AB, A -> B
                    _ => unreachable!(),
                };
                parameters.push(parameter);
            }
        }
        if parameters.len() != expected_count {
            return Err(Error::BoundaryCountMismatch {
                patch: patch_index,
                edge,
            });
        }
        parameters.sort_by(f32::total_cmp);
        for (step, actual) in parameters.into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let expected = step as f32 / resolution as f32;
            if actual.to_bits() != expected.to_bits() {
                return Err(Error::BoundaryParameterMismatch {
                    patch: patch_index,
                    edge,
                });
            }
        }
    }
    Ok(())
}

fn validate_triangles(
    patch_index: usize,
    global_triangle_start: usize,
    first_vertex: u32,
    vertex_end: u32,
    vertices: &[AtlasVertex],
    triangles: &[AtlasTriangle],
) -> Result<(), Error> {
    for (local_triangle, triangle) in triangles.iter().enumerate() {
        let global_triangle = global_triangle_start + local_triangle;
        let [i0, i1, i2] = triangle.indices;
        if [i0, i1, i2]
            .into_iter()
            .any(|index| index < first_vertex || index >= vertex_end)
        {
            return Err(Error::TriangleOutOfRange {
                patch: patch_index,
                triangle: global_triangle,
            });
        }
        if i0 == i1 || i1 == i2 || i2 == i0 {
            return Err(Error::TriangleRepeatedIndex {
                patch: patch_index,
                triangle: global_triangle,
            });
        }
        if i0 > i1 || i0 > i2 {
            return Err(Error::TriangleNotMinFirst {
                patch: patch_index,
                triangle: global_triangle,
            });
        }

        let local = |index: u32| -> Result<usize, Error> {
            usize::try_from(index - first_vertex).map_err(|_| Error::ArithmeticOverflow)
        };
        let p0 = vertices[local(i0)?].barycentric;
        let p1 = vertices[local(i1)?].barycentric;
        let p2 = vertices[local(i2)?].barycentric;
        let orientation = (f64::from(p1[1]) - f64::from(p0[1]))
            * (f64::from(p2[2]) - f64::from(p0[2]))
            - (f64::from(p1[2]) - f64::from(p0[2])) * (f64::from(p2[1]) - f64::from(p0[1]));
        if orientation <= 0.0 {
            return Err(Error::TriangleNotCounterClockwise {
                patch: patch_index,
                triangle: global_triangle,
            });
        }
    }
    Ok(())
}

fn checked_u32(field: &'static str, value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::CountTooLarge { field, value })
}

fn require_offset(
    bytes: &[u8],
    offset: usize,
    field: &'static str,
    expected: usize,
) -> Result<(), Error> {
    let expected = u64::try_from(expected).map_err(|_| Error::ArithmeticOverflow)?;
    let actual = get_u64(bytes, offset);
    if actual != expected {
        return Err(Error::OffsetMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_zero_word(bytes: &[u8], offset: usize) -> Result<(), Error> {
    let value = get_u32(bytes, offset);
    if value != 0 {
        return Err(Error::NonzeroReserved { offset, value });
    }
    Ok(())
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated slice"),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated slice"),
    )
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

// Dependency-free SHA-256 keeps the ABI oracle standalone. This is artifact
// identity, not a cryptographic authentication boundary.
#[allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut state = [
        0x6a09_e667_u32,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let padded_len = input
        .len()
        .checked_add(9)
        .and_then(|length| length.checked_add((64 - length % 64) % 64))
        .expect("input length addressable by usize");
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four-byte word"),
            );
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big_s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big_s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut output = [0_u8; 32];
    for (index, word) in state.into_iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    fn source_triangle() -> Artifact {
        Artifact {
            algorithm_version: 1,
            master_seed: 42,
            patches: vec![AtlasPatch {
                key: AtlasKey::new(1, 1, 1),
                first_vertex: 0,
                vertex_count: 3,
                first_triangle: 0,
                triangle_count: 1,
            }],
            vertices: vec![
                AtlasVertex {
                    barycentric: [1.0, 0.0, 0.0],
                    source_class: SourceClass::Corner,
                },
                AtlasVertex {
                    barycentric: [0.0, 1.0, 0.0],
                    source_class: SourceClass::Corner,
                },
                AtlasVertex {
                    barycentric: [0.0, 0.0, 1.0],
                    source_class: SourceClass::Corner,
                },
            ],
            triangles: vec![AtlasTriangle { indices: [0, 1, 2] }],
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().fold(String::new(), |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing into String cannot fail");
            output
        })
    }

    fn rewrite_hash(bytes: &mut [u8]) {
        let hash = sha256(&bytes[HEADER_BYTES..]);
        bytes[HEADER_HASH_OFFSET..HEADER_HASH_OFFSET + PAYLOAD_HASH_BYTES].copy_from_slice(&hash);
    }

    #[test]
    fn sha256_matches_public_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn source_triangle_round_trips_with_frozen_payload_hash() {
        let artifact = source_triangle();
        let bytes = encode(&artifact).expect("valid source triangle");
        assert_eq!(
            bytes.len(),
            HEADER_BYTES + PATCH_BYTES + 3 * VERTEX_BYTES + TRIANGLE_BYTES
        );
        assert_eq!(
            hex(&bytes[HEADER_HASH_OFFSET..HEADER_HASH_OFFSET + PAYLOAD_HASH_BYTES]),
            "9d273b78db9143626019207fbb7822135ad0d3cdd777ecbe2b7a2d01430d1500"
        );
        assert_eq!(decode(&bytes).expect("canonical bytes"), artifact);
    }

    #[test]
    fn rejects_noncanonical_key_and_range_order() {
        let mut artifact = source_triangle();
        artifact.patches[0].key = AtlasKey::new(2, 1, 1);
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::InvalidKey { .. })
        ));

        let mut artifact = source_triangle();
        artifact.patches[0].first_vertex = 1;
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::NonPackedRange { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_negative_zero_and_wrong_boundary_counts() {
        let mut artifact = source_triangle();
        artifact.vertices[0].barycentric[0] = f32::NAN;
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::InvalidBarycentric { .. })
        ));

        let mut artifact = source_triangle();
        artifact.vertices[0].barycentric[1] = -0.0;
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::InvalidBarycentric { .. })
        ));

        let mut artifact = source_triangle();
        artifact.patches[0].key = AtlasKey::new(1, 1, 2);
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::BoundaryCountMismatch { edge: 2, .. })
        ));
    }

    #[test]
    fn rejects_bad_triangle_winding_and_rotation() {
        let mut artifact = source_triangle();
        artifact.triangles[0].indices = [0, 2, 1];
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::TriangleNotCounterClockwise { .. })
        ));

        let mut artifact = source_triangle();
        artifact.triangles[0].indices = [1, 2, 0];
        assert!(matches!(
            validate_artifact(&artifact),
            Err(Error::TriangleNotMinFirst { .. })
        ));
    }

    #[test]
    fn decoder_rejects_hash_offsets_reserved_words_and_trailing_bytes() {
        let canonical = encode(&source_triangle()).expect("valid source triangle");

        let mut bytes = canonical.clone();
        bytes[HEADER_BYTES + PATCH_BYTES] ^= 1;
        assert_eq!(decode(&bytes), Err(Error::HashMismatch));

        let mut bytes = canonical.clone();
        put_u64(&mut bytes, 56, 999);
        assert!(matches!(decode(&bytes), Err(Error::OffsetMismatch { .. })));

        let mut bytes = canonical.clone();
        put_u32(&mut bytes, HEADER_BYTES + 28, 1);
        rewrite_hash(&mut bytes);
        assert!(matches!(decode(&bytes), Err(Error::NonzeroReserved { .. })));

        let mut bytes = canonical;
        bytes.push(0);
        assert!(matches!(decode(&bytes), Err(Error::LengthMismatch { .. })));
    }
}
