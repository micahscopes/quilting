//! Batch grouping: group faces by (canonical LOD, permutation, material) for instanced draw.
//!
//! This is pure data transformation — no GPU dependency. Replaces the duplicated
//! grouping logic that was in JS (setupGpuSkinning + rebuildBatchesFromFaceLods).

use std::collections::BTreeMap;

/// Instance data stride: 40 floats per face.
pub const INSTANCE_STRIDE: usize = 40;

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

/// Group faces into draw batches by (canonical LOD triple, permutation, material).
///
/// # Arguments
/// - `face_lods`: 5 floats per face: [canon_a, canon_b, canon_c, perm_index, parity]
/// - `all_instances`: flat f32 array, INSTANCE_STRIDE floats per face
/// - `face_materials`: per-face material index (len == num_faces)
/// - `num_faces`: number of faces
///
/// # Returns
/// Grouped `DrawBatch`es sorted by key for deterministic ordering.
pub fn group_into_batches(
    face_lods: &[f32],
    all_instances: &[f32],
    face_materials: &[usize],
    num_faces: usize,
) -> Vec<DrawBatch> {
    assert!(face_lods.len() >= num_faces * 5);
    assert!(all_instances.len() >= num_faces * INSTANCE_STRIDE);

    // BTreeMap for deterministic ordering.
    // Key: (lod_a, lod_b, lod_c, perm_index, material_index)
    let mut groups: BTreeMap<(u32, u32, u32, u32, usize), (f32, Vec<u32>)> = BTreeMap::new();

    for fi in 0..num_faces {
        let lo = fi * 5;
        let ca = face_lods[lo] as u32;
        let cb = face_lods[lo + 1] as u32;
        let cc = face_lods[lo + 2] as u32;
        let perm_idx = face_lods[lo + 3] as u32;
        let parity = face_lods[lo + 4];
        let mat_idx = if fi < face_materials.len() { face_materials[fi] } else { 0 };

        let key = (ca, cb, cc, perm_idx, mat_idx);
        groups.entry(key)
            .or_insert_with(|| (parity, Vec::new()))
            .1
            .push(fi as u32);
    }

    groups.into_iter().map(|((ca, cb, cc, perm_idx, mat_idx), (parity, faces))| {
        // Pack instance data for this group
        let mut instance_data = Vec::with_capacity(faces.len() * INSTANCE_STRIDE);
        for &fi in &faces {
            let start = fi as usize * INSTANCE_STRIDE;
            instance_data.extend_from_slice(&all_instances[start..start + INSTANCE_STRIDE]);
        }

        DrawBatch {
            lod: [ca, cb, cc],
            perm_index: perm_idx,
            parity,
            material_index: mat_idx,
            tess_key: TessKey { lod: [ca, cb, cc], perm_index: perm_idx },
            face_indices: faces,
            instance_data,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_face_single_batch() {
        let face_lods = vec![4.0, 4.0, 4.0, 0.0, 1.0]; // one face
        let all_instances = vec![0.0f32; INSTANCE_STRIDE]; // one face of data
        let face_materials = vec![0];
        let batches = group_into_batches(&face_lods, &all_instances, &face_materials, 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].lod, [4, 4, 4]);
        assert_eq!(batches[0].face_indices, vec![0]);
        assert_eq!(batches[0].instance_data.len(), INSTANCE_STRIDE);
    }

    #[test]
    fn groups_by_lod_and_material() {
        // 3 faces: two share LOD+material, one differs
        let face_lods = vec![
            4.0, 4.0, 4.0, 0.0, 1.0,  // face 0
            4.0, 4.0, 4.0, 0.0, 1.0,  // face 1 (same group as 0)
            8.0, 8.0, 8.0, 0.0, 1.0,  // face 2 (different LOD)
        ];
        let mut all_instances = vec![0.0f32; 3 * INSTANCE_STRIDE];
        // Tag each face's first float so we can verify data packing
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
            4.0, 4.0, 4.0, 0.0, 1.0,
            4.0, 4.0, 4.0, 0.0, 1.0,
        ];
        let all_instances = vec![0.0f32; 2 * INSTANCE_STRIDE];
        let face_materials = vec![0, 1]; // different materials
        let batches = group_into_batches(&face_lods, &all_instances, &face_materials, 2);
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn empty_input() {
        let batches = group_into_batches(&[], &[], &[], 0);
        assert!(batches.is_empty());
    }
}
