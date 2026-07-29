//! Batch grouping: group faces by (atlas index, permutation, material) for instanced draw.
//!
//! Uses O(n) bucket sort with direct indexing — the key space is bounded
//! (atlas 0-255 × perm 0-5 × materials) and small enough for a flat Vec.

/// Instance data stride in floats. Re-exported from [`crate::instance_layout`],
/// which is the normative definition — do not restate the number here.
pub use crate::instance_layout::STRIDE as INSTANCE_STRIDE;

/// Per-face LOD stride: 6 floats from GPU pass 2.
pub const FACE_LOD_STRIDE: usize = 6;

/// A tessellation key identifying a unique atlas patch + permutation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TessKey {
    /// Canonical LOD triple (sorted ascending).
    pub lod: [u32; 3],
    /// S3 permutation index (0-5).
    pub perm_index: u32,
}

impl TessKey {
    pub fn as_string(&self) -> String {
        format!("{},{},{}/{}", self.lod[0], self.lod[1], self.lod[2], self.perm_index)
    }
}

/// A logical draw batch — faces grouped by (LOD, permutation, material).
/// Contains packed instance data ready for GPU upload.
#[derive(Debug)]
pub struct DrawBatch {
    /// Canonical LOD triple.
    pub lod: [u32; 3],
    /// S3 permutation index (0-5).
    pub perm_index: u32,
    /// Permutation parity (+1.0 or -1.0).
    pub parity: f32,
    /// Material index.
    pub material_index: usize,
    /// Tessellation key for atlas lookup.
    pub tess_key: TessKey,
    /// Original face indices in this batch.
    pub face_indices: Vec<u32>,
    /// Packed instance data (INSTANCE_STRIDE floats per face).
    pub instance_data: Vec<f32>,
}

/// Group faces into draw batches using O(n) bucket sort.
///
/// # Arguments
/// - `face_lods`: 6 floats per face: [canon_a, canon_b, canon_c, perm_index, parity, atlas_index]
/// - `all_instances`: flat f32 array, INSTANCE_STRIDE floats per face
/// - `face_materials`: per-face material index (len == num_faces)
/// - `num_faces`: number of faces
///
/// # Returns
/// Grouped `DrawBatch`es. Order is deterministic (sorted by bucket key).
pub fn group_into_batches(
    face_lods: &[f32],
    all_instances: &[f32],
    face_materials: &[usize],
    num_faces: usize,
) -> Vec<DrawBatch> {
    if num_faces == 0 { return Vec::new(); }
    assert!(face_lods.len() >= num_faces * FACE_LOD_STRIDE);
    assert!(all_instances.len() >= num_faces * INSTANCE_STRIDE);

    // Find max material index for bucket array sizing
    let num_materials = face_materials.iter().copied().max().unwrap_or(0) + 1;

    // Bucket array: key = atlas_idx * (6 * num_materials) + perm * num_materials + material
    // Max atlas_idx = 255, max perm = 5, so max buckets = 256 * 6 * num_materials
    let num_buckets = 256 * 6 * num_materials;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];

    // Single pass: distribute faces into buckets
    for fi in 0..num_faces {
        let lo = fi * FACE_LOD_STRIDE;
        let atlas_idx = face_lods[lo + 5] as usize;
        let perm = face_lods[lo + 3] as usize;
        let mat = if fi < face_materials.len() { face_materials[fi] } else { 0 };

        let key = atlas_idx * (6 * num_materials) + perm * num_materials + mat;
        if key < num_buckets {
            buckets[key].push(fi as u32);
        }
    }

    // Collect non-empty buckets into DrawBatches
    let mut result = Vec::new();
    for (key, faces) in buckets.into_iter().enumerate() {
        if faces.is_empty() { continue; }

        let mat = key % num_materials;
        let perm = (key / num_materials) % 6;

        // Read canonical LODs and parity from first face in bucket
        let fi0 = faces[0] as usize;
        let lo = fi0 * FACE_LOD_STRIDE;
        let ca = face_lods[lo] as u32;
        let cb = face_lods[lo + 1] as u32;
        let cc = face_lods[lo + 2] as u32;
        let parity = face_lods[lo + 4];

        // Pack instance data
        let mut instance_data = Vec::with_capacity(faces.len() * INSTANCE_STRIDE);
        for &fi in &faces {
            let start = fi as usize * INSTANCE_STRIDE;
            instance_data.extend_from_slice(&all_instances[start..start + INSTANCE_STRIDE]);
        }

        result.push(DrawBatch {
            lod: [ca, cb, cc],
            perm_index: perm as u32,
            parity,
            material_index: mat,
            tess_key: TessKey { lod: [ca, cb, cc], perm_index: perm as u32 },
            face_indices: faces,
            instance_data,
        });
    }

    result
}

/// A batch range into a shared, sorted instance buffer.
/// No instance data copy — just an offset and count.
#[derive(Debug)]
pub struct BatchRange {
    pub lod: [u32; 3],
    pub perm_index: u32,
    pub parity: f32,
    pub material_index: usize,
    pub tess_key: TessKey,
    /// Byte offset into the shared instance buffer where this batch starts.
    pub instance_offset: usize,
    /// Number of instances (faces) in this batch.
    pub instance_count: usize,
}

/// Sort instance data by batch key and return contiguous ranges.
///
/// Permutes `all_instances` in-place so faces in the same batch are contiguous.
/// Returns `BatchRange`s pointing into the sorted buffer.
///
/// This avoids copying 73MB of instance data into per-batch vecs.
pub fn sort_into_ranges(
    face_lods: &[f32],
    all_instances: &mut [f32],
    face_materials: &[usize],
    num_faces: usize,
) -> Vec<BatchRange> {
    if num_faces == 0 { return Vec::new(); }
    assert!(face_lods.len() >= num_faces * FACE_LOD_STRIDE);
    assert!(all_instances.len() >= num_faces * INSTANCE_STRIDE);

    let num_materials = face_materials.iter().copied().max().unwrap_or(0) + 1;
    let num_buckets = 256 * 6 * num_materials;

    // Phase 1: bucket sort to get face ordering
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];
    for fi in 0..num_faces {
        let lo = fi * FACE_LOD_STRIDE;
        let atlas_idx = face_lods[lo + 5] as usize;
        let perm = face_lods[lo + 3] as usize;
        let mat = if fi < face_materials.len() { face_materials[fi] } else { 0 };
        let key = atlas_idx * (6 * num_materials) + perm * num_materials + mat;
        if key < num_buckets {
            buckets[key].push(fi as u32);
        }
    }

    // Phase 2: build the sorted permutation and batch ranges
    let mut sorted_order: Vec<u32> = Vec::with_capacity(num_faces);
    let mut ranges: Vec<BatchRange> = Vec::new();

    for (key, faces) in buckets.into_iter().enumerate() {
        if faces.is_empty() { continue; }
        let mat = key % num_materials;
        let perm = (key / num_materials) % 6;

        let fi0 = faces[0] as usize;
        let lo = fi0 * FACE_LOD_STRIDE;
        let ca = face_lods[lo] as u32;
        let cb = face_lods[lo + 1] as u32;
        let cc = face_lods[lo + 2] as u32;
        let parity = face_lods[lo + 4];

        let offset = sorted_order.len();
        sorted_order.extend_from_slice(&faces);

        ranges.push(BatchRange {
            lod: [ca, cb, cc],
            perm_index: perm as u32,
            parity,
            material_index: mat,
            tess_key: TessKey { lod: [ca, cb, cc], perm_index: perm as u32 },
            instance_offset: offset * crate::instance_layout::STRIDE_BYTES,
            instance_count: faces.len(),
        });
    }

    // Phase 3: permute all_instances according to sorted_order
    // Use a temporary buffer for the permutation (unavoidable for in-place sort of strided data)
    let mut sorted = vec![0.0f32; num_faces * INSTANCE_STRIDE];
    for (dst, &src_fi) in sorted_order.iter().enumerate() {
        let src = src_fi as usize * INSTANCE_STRIDE;
        let dst_off = dst * INSTANCE_STRIDE;
        sorted[dst_off..dst_off + INSTANCE_STRIDE]
            .copy_from_slice(&all_instances[src..src + INSTANCE_STRIDE]);
    }
    all_instances[..num_faces * INSTANCE_STRIDE].copy_from_slice(&sorted[..num_faces * INSTANCE_STRIDE]);

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_face_single_batch() {
        // 6 floats: canon_a=4, canon_b=4, canon_c=4, perm=0, parity=1, atlas_idx=0
        let face_lods = vec![4.0, 4.0, 4.0, 0.0, 1.0, 0.0];
        let all_instances = vec![0.0f32; INSTANCE_STRIDE];
        let face_materials = vec![0];
        let batches = group_into_batches(&face_lods, &all_instances, &face_materials, 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].lod, [4, 4, 4]);
        assert_eq!(batches[0].face_indices, vec![0]);
        assert_eq!(batches[0].instance_data.len(), INSTANCE_STRIDE);
    }

    #[test]
    fn groups_by_lod_and_material() {
        let face_lods = vec![
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,  // face 0, atlas=0
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,  // face 1, same atlas
            8.0, 8.0, 8.0, 0.0, 1.0, 1.0,  // face 2, atlas=1
        ];
        let mut all_instances = vec![0.0f32; 3 * INSTANCE_STRIDE];
        all_instances[0] = 100.0;
        all_instances[INSTANCE_STRIDE] = 200.0;
        all_instances[2 * INSTANCE_STRIDE] = 300.0;

        let face_materials = vec![0, 0, 0];
        let batches = group_into_batches(&face_lods, &all_instances, &face_materials, 3);
        assert_eq!(batches.len(), 2);

        let batch_4 = batches.iter().find(|b| b.lod == [4, 4, 4]).unwrap();
        assert_eq!(batch_4.face_indices, vec![0, 1]);
        assert_eq!(batch_4.instance_data[0], 100.0);
        assert_eq!(batch_4.instance_data[INSTANCE_STRIDE], 200.0);

        let batch_8 = batches.iter().find(|b| b.lod == [8, 8, 8]).unwrap();
        assert_eq!(batch_8.face_indices, vec![2]);
        assert_eq!(batch_8.instance_data[0], 300.0);
    }

    #[test]
    fn groups_by_material() {
        let face_lods = vec![
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,
        ];
        let all_instances = vec![0.0f32; 2 * INSTANCE_STRIDE];
        let face_materials = vec![0, 1];
        let batches = group_into_batches(&face_lods, &all_instances, &face_materials, 2);
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn empty_input() {
        let batches = group_into_batches(&[], &[], &[], 0);
        assert!(batches.is_empty());
    }
}
