//! Normative definition of the per-face instance buffer layout.
//!
//! This module is the single source of truth for how face instance data is
//! packed. Rust packers, the renderer's VAO setup, and the WGSL vertex shader
//! must all agree; anything that hardcodes a stride or offset instead of using
//! the constants here is a silent-corruption hazard.
//!
//! # Layout: 40 floats / 160 bytes / 10 instanced attributes
//!
//! | floats | bytes | `@location` | content                          |
//! |--------|-------|-------------|----------------------------------|
//! | 0..4   | 0     | 1           | p0: `[vertex_idx, x, y, z]`      |
//! | 4..8   | 16    | 2           | p1                               |
//! | 8..12  | 32    | 3           | p2                               |
//! | 12..16 | 48    | 7           | edge LODs + permutation index    |
//! | 16..20 | 64    | 8           | vertex LODs + pad                |
//! | 20..24 | 80    | 9           | uv01 `(u0, v0, u1, v1)`          |
//! | 24..28 | 96    | 10          | uv2 `(u2, v2, 0, 0)`             |
//! | 28..32 | 112   | 11          | n0 `(x, y, z, 0)`                |
//! | 32..36 | 128   | 12          | n1                               |
//! | 36..40 | 144   | 13          | n2                               |
//!
//! Two details routinely surprise readers:
//!
//! - The first component of each position is a **vertex index**, not the
//!   quaternion scalar part. GPU skinning uses it to look up the deformed
//!   vertex. The rest of the codebase stores quaternions `(w, x, y, z)`, so
//!   this slot is the one deliberate exception.
//! - Locations 4, 5, 6 (the QB weight quaternions) are **not in the buffer**.
//!   They are supplied as the constant `[1, 0, 0, 0]` because the fused
//!   Möbius-QB evaluation derives conformal weights from the Möbius uniforms
//!   in the shader. See [`CONSTANT_WEIGHT_LOCATIONS`].

/// Floats per face instance.
pub const STRIDE: usize = 40;

/// Bytes per face instance.
pub const STRIDE_BYTES: usize = STRIDE * 4;

/// Float offsets of each field within one instance.
pub mod offset {
    /// Position `i` occupies `POSITIONS + i * 4`, as `[vertex_idx, x, y, z]`.
    pub const POSITIONS: usize = 0;
    /// Three edge LODs followed by the per-instance S3 permutation index.
    pub const EDGE_LODS: usize = 12;
    /// Per-instance S3 permutation index, stored in `lod_info.w`.
    pub const PERM_INDEX: usize = EDGE_LODS + 3;
    /// Three vertex LODs, fourth float is padding.
    pub const VERTEX_LODS: usize = 16;
    /// Six UV floats: `(u0, v0, u1, v1, u2, v2)`, then two floats of padding.
    pub const UVS: usize = 20;
    /// Normal `i` occupies `NORMALS + i * 4`, as `(x, y, z, 0)`.
    pub const NORMALS: usize = 28;
}

/// Instanced vertex attributes as `(location, byte_offset)`.
///
/// Every entry is a `vec4` with an attribute divisor of 1.
pub const ATTR_MAP: [(u32, i32); 10] = [
    (1, 0),    // p0
    (2, 16),   // p1
    (3, 32),   // p2
    (7, 48),   // edge LODs
    (8, 64),   // vertex LODs
    (9, 80),   // uv01
    (10, 96),  // uv2
    (11, 112), // n0
    (12, 128), // n1
    (13, 144), // n2
];

/// Weight-quaternion attribute locations fed as a constant rather than from
/// the instance buffer. Each is set to `[1, 0, 0, 0]` (identity quaternion).
pub const CONSTANT_WEIGHT_LOCATIONS: [u32; 3] = [4, 5, 6];

/// Cursor over one instance's slice, so packers name fields instead of
/// rediscovering offsets.
pub struct InstanceWriter<'a> {
    slice: &'a mut [f32],
}

impl<'a> InstanceWriter<'a> {
    /// Borrow instance `index` out of a flat buffer of `STRIDE`-float records.
    ///
    /// Panics if the buffer is too short — a missized allocation is a bug
    /// worth failing loudly on rather than silently truncating geometry.
    pub fn new(buffer: &'a mut [f32], index: usize) -> Self {
        let base = index * STRIDE;
        Self { slice: &mut buffer[base..base + STRIDE] }
    }

    /// Set corner `i` to `xyz`, tagged with the vertex index used by skinning.
    pub fn set_position(&mut self, i: usize, vertex_idx: u32, xyz: [f32; 3]) {
        let o = offset::POSITIONS + i * 4;
        self.slice[o] = vertex_idx as f32;
        self.slice[o + 1] = xyz[0];
        self.slice[o + 2] = xyz[1];
        self.slice[o + 3] = xyz[2];
    }

    pub fn set_edge_lods(&mut self, lods: [f32; 3]) {
        self.slice[offset::EDGE_LODS..offset::EDGE_LODS + 3].copy_from_slice(&lods);
    }

    /// Select how the canonical tessellation's barycentrics map back to this face.
    pub fn set_perm_index(&mut self, perm_index: u32) {
        debug_assert!(perm_index < 6);
        self.slice[offset::PERM_INDEX] = perm_index as f32;
    }

    pub fn set_vertex_lods(&mut self, lods: [f32; 3]) {
        self.slice[offset::VERTEX_LODS..offset::VERTEX_LODS + 3].copy_from_slice(&lods);
    }

    /// UVs for the three corners, in corner order.
    pub fn set_uvs(&mut self, uvs: [[f32; 2]; 3]) {
        let o = offset::UVS;
        for (i, uv) in uvs.iter().enumerate() {
            self.slice[o + i * 2] = uv[0];
            self.slice[o + i * 2 + 1] = uv[1];
        }
    }

    /// Smooth normal for corner `i`. Zeroed normals tell the shader to fall
    /// back to analytic QB normals (SPEC invariant 8).
    pub fn set_normal(&mut self, i: usize, n: [f32; 3]) {
        let o = offset::NORMALS + i * 4;
        self.slice[o] = n[0];
        self.slice[o + 1] = n[1];
        self.slice[o + 2] = n[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attr_map_tiles_the_stride_exactly() {
        let mut offsets: Vec<i32> = ATTR_MAP.iter().map(|&(_, o)| o).collect();
        offsets.sort_unstable();
        for (i, o) in offsets.iter().enumerate() {
            assert_eq!(*o as usize, i * 16, "attributes must be contiguous vec4s");
        }
        assert_eq!(
            offsets.len() * 16,
            STRIDE_BYTES,
            "attributes must cover the whole stride with no gap or overhang"
        );
    }

    #[test]
    fn attr_locations_are_unique_and_disjoint_from_constant_weights() {
        let mut locs: Vec<u32> = ATTR_MAP.iter().map(|&(l, _)| l).collect();
        locs.sort_unstable();
        locs.dedup();
        assert_eq!(locs.len(), ATTR_MAP.len(), "duplicate attribute location");
        for w in CONSTANT_WEIGHT_LOCATIONS {
            assert!(!locs.contains(&w), "location {w} is both buffered and constant");
        }
    }

    #[test]
    fn named_offsets_agree_with_attr_map() {
        let byte = |float_off: usize| (float_off * 4) as i32;
        for (i, expected) in [(0usize, 0), (1, 16), (2, 32)] {
            assert_eq!(byte(offset::POSITIONS + i * 4), expected);
        }
        assert_eq!(byte(offset::EDGE_LODS), 48);
        assert_eq!(byte(offset::VERTEX_LODS), 64);
        assert_eq!(byte(offset::UVS), 80);
        assert_eq!(byte(offset::NORMALS), 112);
    }

    #[test]
    fn writer_places_fields_at_documented_offsets() {
        let mut buf = vec![0.0f32; STRIDE * 2];
        let mut w = InstanceWriter::new(&mut buf, 1);
        w.set_position(2, 7, [1.0, 2.0, 3.0]);
        w.set_edge_lods([4.0, 8.0, 16.0]);
        w.set_perm_index(5);
        w.set_uvs([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
        w.set_normal(0, [0.0, 1.0, 0.0]);

        let b = STRIDE;
        assert_eq!(&buf[b + 8..b + 12], &[7.0, 1.0, 2.0, 3.0]);
        assert_eq!(&buf[b + 12..b + 15], &[4.0, 8.0, 16.0]);
        assert_eq!(buf[b + offset::PERM_INDEX], 5.0);
        assert_eq!(&buf[b + 20..b + 26], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(&buf[b + 28..b + 31], &[0.0, 1.0, 0.0]);
        // Instance 0 untouched.
        assert!(buf[..STRIDE].iter().all(|&f| f == 0.0));
    }
}
