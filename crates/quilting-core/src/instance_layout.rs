//! Normative definition of the per-face instance buffer layout.
//!
//! This module is the single source of truth for how face instance data is
//! packed. Rust packers, the renderer's VAO setup, and the WGSL vertex shader
//! must all agree; anything that hardcodes a stride or offset instead of using
//! the constants here is a silent-corruption hazard.
//!
//! # Prepared layout: 52 floats / 208 bytes / 13 instanced attributes
//!
//! | floats | bytes | `@location` | content                          |
//! |--------|-------|-------------|----------------------------------|
//! | 0..4   | 0     | 1           | p0: `[vertex_idx, x, y, z]`      |
//! | 4..8   | 16    | 2           | p1                               |
//! | 8..12  | 32    | 3           | p2                               |
//! | 12..16 | 48    | 4           | rational QB weight w0            |
//! | 16..20 | 64    | 5           | rational QB weight w1            |
//! | 20..24 | 80    | 6           | rational QB weight w2            |
//! | 24..28 | 96    | 7           | edge LODs + permutation index    |
//! | 28..32 | 112   | 8           | vertex LODs + source face ID     |
//! | 32..36 | 128   | 9           | uv01 `(u0, v0, u1, v1)`          |
//! | 36..40 | 144   | 10          | uv2 + preparation `(u2,v2,reserved,prepared)` |
//! | 40..44 | 160   | 11          | n0 `(x, y, z, semantic_node_id)` |
//! | 44..48 | 176   | 12          | n1                               |
//! | 48..52 | 192   | 13          | n2                               |
//!
//! Two details routinely surprise readers:
//!
//! - The first component of each position is a **vertex index**, not the
//!   quaternion scalar part. GPU skinning uses it to look up the deformed
//!   vertex. The rest of the codebase stores quaternions `(w, x, y, z)`, so
//!   this slot is the one deliberate exception.
//! - Locations 4, 5, 6 carry the source patch's rational QB weights. Ordinary
//!   triangle meshes use identity weights; fitted/remeshed patches must retain
//!   their non-identity weights through preparation and rendering. The shader
//!   combines these source weights with the current Möbius transform once.
//! - Location 14 is a separate one-float instanced visibility stream. It is
//!   intentionally absent from the 52-float prepared record because it changes
//!   with the camera while posed geometry usually does not.

/// Floats per face instance.
pub const STRIDE: usize = 52;

/// Bytes per face instance.
pub const STRIDE_BYTES: usize = STRIDE * 4;

/// One camera-dependent visibility scalar is stored beside, rather than
/// inside, each prepared patch record.
pub const VISIBILITY_STRIDE_BYTES: usize = 4;

/// Instanced input location for the separate visibility scalar. WebGL2
/// guarantees at least 16 vertex attributes, so location 14 remains portable.
pub const VISIBILITY_ATTR_LOCATION: u32 = 14;

/// Floats in the topology-only record streamed when a face changes draw
/// bucket. Static control points, UVs, and normals live in a renderer-owned
/// per-face texture and are fetched by `FACE_ID` during patch preparation.
pub const BATCH_TOPOLOGY_STRIDE: usize = 8;

/// Bytes per topology-only batch record.
pub const BATCH_TOPOLOGY_STRIDE_BYTES: usize = BATCH_TOPOLOGY_STRIDE * 4;

/// Preparation-pass attributes as `(location, byte_offset)`.
pub const BATCH_TOPOLOGY_ATTR_MAP: [(u32, i32); 2] = [
    (7, 0),  // edge LODs + permutation
    (8, 16), // source face ID + current per-vertex visualization LODs
];

/// Float offsets of each field within one instance.
pub mod offset {
    /// Position `i` occupies `POSITIONS + i * 4`, as `[vertex_idx, x, y, z]`.
    pub const POSITIONS: usize = 0;
    /// Rational QB weight `i` occupies `WEIGHTS + i * 4`, as `(w, x, y, z)`.
    pub const WEIGHTS: usize = 12;
    /// Three edge LODs followed by the per-instance S3 permutation index.
    pub const EDGE_LODS: usize = 24;
    /// Per-instance S3 permutation index, stored in `lod_info.w`.
    pub const PERM_INDEX: usize = EDGE_LODS + 3;
    /// Three vertex LODs followed by the stable source face ID.
    pub const VERTEX_LODS: usize = 28;
    /// Original source-face index, stored in `vert_lod.w` for picking.
    pub const FACE_ID: usize = VERTEX_LODS + 3;
    /// Six UV floats: `(u0, v0, u1, v1, u2, v2)`.
    pub const UVS: usize = 32;
    /// Reserved preparation lane. Camera-dependent visibility lives in its own
    /// one-float stream so camera motion does not rewrite this record.
    pub const PREPARED_RESERVED: usize = UVS + 6;
    /// Nonzero when the record's positions and visibility have been prepared.
    pub const PREPARED_FLAG: usize = UVS + 7;
    /// Normal `i` occupies `NORMALS + i * 4`; `n0.w` carries semantic node ID.
    pub const NORMALS: usize = 40;
    /// Stable semantic source node, stored in the otherwise-unused `n0.w`.
    pub const NODE_ID: usize = NORMALS + 3;
}

/// Float offsets in a topology-only batch record.
pub mod batch_offset {
    /// Three face-local edge LODs followed by the S3 permutation index.
    pub const EDGE_LODS: usize = 0;
    /// Stable source face ID followed by current per-vertex visualization LODs.
    pub const FACE_ID: usize = 4;
    pub const VERTEX_LODS: usize = 5;
}

/// Instanced vertex attributes as `(location, byte_offset)`.
///
/// Every entry is a `vec4` with an attribute divisor of 1.
pub const ATTR_MAP: [(u32, i32); 13] = [
    (1, 0),    // p0
    (2, 16),   // p1
    (3, 32),   // p2
    (4, 48),   // w0
    (5, 64),   // w1
    (6, 80),   // w2
    (7, 96),   // edge LODs
    (8, 112),  // vertex LODs
    (9, 128),  // uv01
    (10, 144), // uv2
    (11, 160), // n0
    (12, 176), // n1
    (13, 192), // n2
];

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
        let slice = &mut buffer[base..base + STRIDE];
        // Identity is the correct source weight for ordinary mesh triangles
        // and a safe default for every producer that does not fit rational QB
        // patches. Rational producers overwrite these through `set_weight`.
        for i in 0..3 {
            slice[offset::WEIGHTS + i * 4] = 1.0;
        }
        Self { slice }
    }

    /// Set corner `i` to `xyz`, tagged with the vertex index used by skinning.
    pub fn set_position(&mut self, i: usize, vertex_idx: u32, xyz: [f32; 3]) {
        let o = offset::POSITIONS + i * 4;
        self.slice[o] = vertex_idx as f32;
        self.slice[o + 1] = xyz[0];
        self.slice[o + 2] = xyz[1];
        self.slice[o + 3] = xyz[2];
    }

    /// Set source rational-QB weight `i` in quaternion `(w, x, y, z)` order.
    pub fn set_weight(&mut self, i: usize, weight: [f32; 4]) {
        let o = offset::WEIGHTS + i * 4;
        self.slice[o..o + 4].copy_from_slice(&weight);
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

    /// Preserve the source face identity after visibility filtering and batching.
    pub fn set_face_id(&mut self, face_id: u32) {
        self.slice[offset::FACE_ID] = face_id as f32;
    }

    /// UVs for the three corners, in corner order.
    pub fn set_uvs(&mut self, uvs: [[f32; 2]; 3]) {
        let o = offset::UVS;
        for (i, uv) in uvs.iter().enumerate() {
            self.slice[o + i * 2] = uv[0];
            self.slice[o + i * 2 + 1] = uv[1];
        }
    }

    /// Mark a record as GPU-prepared. The camera-dependent visibility scalar
    /// is owned by a separate stream.
    pub fn mark_prepared(&mut self) {
        self.slice[offset::PREPARED_RESERVED] = 0.0;
        self.slice[offset::PREPARED_FLAG] = 1.0;
    }

    /// Smooth normal for corner `i`. Zeroed normals tell the shader to fall
    /// back to analytic QB normals (SPEC invariant 8).
    pub fn set_normal(&mut self, i: usize, n: [f32; 3]) {
        let o = offset::NORMALS + i * 4;
        self.slice[o] = n[0];
        self.slice[o + 1] = n[1];
        self.slice[o + 2] = n[2];
    }

    /// Preserve semantic source-node identity independently of draw grouping.
    pub fn set_node_id(&mut self, node_id: u32) {
        self.slice[offset::NODE_ID] = node_id as f32;
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
    fn attr_locations_are_unique() {
        let mut locs: Vec<u32> = ATTR_MAP.iter().map(|&(l, _)| l).collect();
        locs.sort_unstable();
        locs.dedup();
        assert_eq!(locs.len(), ATTR_MAP.len(), "duplicate attribute location");
        assert_eq!(locs, (1..=13).collect::<Vec<_>>());
        assert_eq!(VISIBILITY_ATTR_LOCATION, 14);
        assert_eq!(VISIBILITY_STRIDE_BYTES, std::mem::size_of::<f32>());
    }

    #[test]
    fn topology_record_is_two_aligned_vec4s() {
        assert_eq!(BATCH_TOPOLOGY_STRIDE, 8);
        assert_eq!(BATCH_TOPOLOGY_STRIDE_BYTES, 32);
        assert_eq!(batch_offset::EDGE_LODS, 0);
        assert_eq!(batch_offset::FACE_ID, 4);
        assert_eq!(BATCH_TOPOLOGY_ATTR_MAP, [(7, 0), (8, 16)]);
    }

    #[test]
    fn named_offsets_agree_with_attr_map() {
        let byte = |float_off: usize| (float_off * 4) as i32;
        for (i, expected) in [(0usize, 0), (1, 16), (2, 32)] {
            assert_eq!(byte(offset::POSITIONS + i * 4), expected);
        }
        assert_eq!(byte(offset::WEIGHTS), 48);
        assert_eq!(byte(offset::EDGE_LODS), 96);
        assert_eq!(byte(offset::VERTEX_LODS), 112);
        assert_eq!(byte(offset::UVS), 128);
        assert_eq!(byte(offset::NORMALS), 160);
    }

    #[test]
    fn writer_places_fields_at_documented_offsets() {
        let mut buf = vec![0.0f32; STRIDE * 2];
        let mut w = InstanceWriter::new(&mut buf, 1);
        w.set_position(2, 7, [1.0, 2.0, 3.0]);
        w.set_weight(2, [0.5, 0.1, 0.2, 0.3]);
        w.set_edge_lods([4.0, 8.0, 16.0]);
        w.set_perm_index(5);
        w.set_vertex_lods([2.0, 3.0, 4.0]);
        w.set_face_id(123);
        w.set_uvs([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
        w.mark_prepared();
        w.set_normal(0, [0.0, 1.0, 0.0]);
        w.set_node_id(91);

        let b = STRIDE;
        assert_eq!(&buf[b + 8..b + 12], &[7.0, 1.0, 2.0, 3.0]);
        assert_eq!(&buf[b + 12..b + 16], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&buf[b + 20..b + 24], &[0.5, 0.1, 0.2, 0.3]);
        assert_eq!(&buf[b + 24..b + 27], &[4.0, 8.0, 16.0]);
        assert_eq!(buf[b + offset::PERM_INDEX], 5.0);
        assert_eq!(&buf[b + offset::VERTEX_LODS..b + offset::VERTEX_LODS + 3], &[2.0, 3.0, 4.0]);
        assert_eq!(buf[b + offset::FACE_ID], 123.0);
        assert_eq!(&buf[b + 32..b + 38], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(buf[b + offset::PREPARED_RESERVED], 0.0);
        assert_eq!(buf[b + offset::PREPARED_FLAG], 1.0);
        assert_eq!(&buf[b + 40..b + 43], &[0.0, 1.0, 0.0]);
        assert_eq!(buf[b + offset::NODE_ID], 91.0);
        // Instance 0 untouched.
        assert!(buf[..STRIDE].iter().all(|&f| f == 0.0));
    }
}
