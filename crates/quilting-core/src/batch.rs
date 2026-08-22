//! Batch grouping: group faces by (atlas index, permutation parity, material) for instanced draw.
//!
//! Uses O(n) bucket sort with direct indexing — the key space is bounded
//! (atlas 0-255 × two winding parities × materials) and small enough for a flat Vec.

use quilting_mesh::HalfEdgeMesh;

/// Instance data stride in floats. Re-exported from [`crate::instance_layout`],
/// which is the normative definition — do not restate the number here.
pub use crate::instance_layout::STRIDE as INSTANCE_STRIDE;

/// Per-face LOD stride: 6 floats from GPU pass 2.
pub const FACE_LOD_STRIDE: usize = 6;

/// Tessellation topology kept resident for a face between asynchronous LOD
/// classifications. Invisible payloads must retain this topology: the render
/// GPU may classify a newer animated pose as visible before the worker result
/// catches up, and resurrecting it with a minimum-LOD fallback causes a coarse
/// one-frame flash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentLod {
    pub canonical: [u32; 3],
    pub perm_index: usize,
    pub parity_bucket: usize,
}

impl ResidentLod {
    pub fn uniform(lod: u32) -> Self {
        Self {
            canonical: [lod; 3],
            perm_index: 0,
            parity_bucket: 0,
        }
    }

    /// Construct resident topology from face-local edge resolutions.
    pub fn from_edge_lods(edge_lods: [u32; 3]) -> Self {
        let key = crate::permutation::canonical_form(edge_lods);
        Self {
            canonical: key.res,
            perm_index: key.perm_index,
            parity_bucket: usize::from(crate::permutation::perm_sign(key.perm_index) < 0),
        }
    }

    /// Recover face-local edge resolutions from canonical atlas order.
    pub fn edge_lods(self) -> [u32; 3] {
        let permutation = crate::permutation::S3_PERMUTATIONS[self.perm_index.min(5)];
        [
            self.canonical[permutation[0]],
            self.canonical[permutation[1]],
            self.canonical[permutation[2]],
        ]
    }

    /// Decode a valid, visible six-float GPU classification payload.
    pub fn from_visible_payload(face_lods: &[f32], face_index: usize) -> Option<Self> {
        if !face_is_visible(face_lods, face_index) {
            return None;
        }
        let offset = face_index.checked_mul(FACE_LOD_STRIDE)?;
        let payload = face_lods.get(offset..offset + FACE_LOD_STRIDE)?;
        if !payload[..5].iter().all(|value| value.is_finite())
            || payload[..3].iter().any(|lod| *lod < 1.0)
        {
            return None;
        }
        Some(Self {
            canonical: [payload[0] as u32, payload[1] as u32, payload[2] as u32],
            perm_index: payload[3].round().clamp(0.0, 5.0) as usize,
            parity_bucket: usize::from(payload[4] < 0.0),
        })
    }
}

/// Update a face's resident topology from a visible payload, or retain its
/// previous topology when the asynchronous classifier reports it invisible.
pub fn retain_face_lod(
    face_lods: &[f32],
    face_index: usize,
    resident: &mut Option<ResidentLod>,
    initial: ResidentLod,
) -> ResidentLod {
    let selected = ResidentLod::from_visible_payload(face_lods, face_index)
        .or(*resident)
        .unwrap_or(initial);
    *resident = Some(selected);
    selected
}

/// Reconcile the topology retained across asynchronous visibility results.
///
/// The worker's GPU pass guarantees agreement when both neighboring faces are
/// visible in that classification. An invisible face intentionally keeps its
/// previous resident topology, however, so a visible neighbor may otherwise
/// change one side of their shared edge alone. Taking the maximum on every
/// resident shared edge restores the same crack-free invariant without ever
/// reducing a visible face's requested resolution.
pub fn reconcile_resident_edges(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
) -> usize {
    let num_faces = residents.len().min(topology.num_faces as usize);
    let mut face_edges: Vec<Option<[u32; 3]>> = residents[..num_faces]
        .iter()
        .map(|resident| resident.map(ResidentLod::edge_lods))
        .collect();

    for half_edge in 0..topology.half_edges.len() as u32 {
        let Some(twin) = topology.twin(half_edge) else { continue };
        if half_edge > twin {
            continue;
        }
        let face = topology.half_edges[half_edge as usize].face as usize;
        let twin_face = topology.half_edges[twin as usize].face as usize;
        if face >= num_faces || twin_face >= num_faces {
            continue;
        }
        let edge_index = (half_edge as usize % 3 + 2) % 3;
        let twin_edge_index = (twin as usize % 3 + 2) % 3;
        let (Some(face_lods), Some(twin_lods)) =
            (face_edges[face], face_edges[twin_face])
        else {
            continue;
        };
        let shared = face_lods[edge_index].max(twin_lods[twin_edge_index]);
        face_edges[face].as_mut().unwrap()[edge_index] = shared;
        face_edges[twin_face].as_mut().unwrap()[twin_edge_index] = shared;
    }

    let mut changed = 0;
    for (resident, edge_lods) in residents[..num_faces].iter_mut().zip(face_edges) {
        let Some(edge_lods) = edge_lods else { continue };
        let reconciled = ResidentLod::from_edge_lods(edge_lods);
        if *resident != Some(reconciled) {
            *resident = Some(reconciled);
            changed += 1;
        }
    }
    changed
}

/// A tessellation key identifying one canonical atlas patch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TessKey {
    /// Canonical LOD triple (sorted ascending).
    pub lod: [u32; 3],
}

impl TessKey {
    pub fn as_string(&self) -> String {
        format!("{},{},{}", self.lod[0], self.lod[1], self.lod[2])
    }
}

/// A logical draw batch — faces grouped by (LOD, permutation parity, material).
/// Contains packed instance data ready for GPU upload.
#[derive(Debug)]
pub struct DrawBatch {
    /// Canonical LOD triple.
    pub lod: [u32; 3],
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
/// - `face_lods`: 6 floats per face: [canon_a, canon_b, canon_c, perm_index, parity, atlas_index].
///   A negative/non-finite atlas index is the visibility-cull sentinel.
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

    // Permutation itself is carried per instance. Hardware front-face winding is
    // draw state, so only odd/even parity has to split batches.
    let num_buckets = 256 * 2 * num_materials;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];

    // Single pass: distribute faces into buckets
    for fi in 0..num_faces {
        let lo = fi * FACE_LOD_STRIDE;
        if !face_is_visible(face_lods, fi) { continue; }
        let atlas_idx = face_lods[lo + 5] as usize;
        let parity_bucket = usize::from(face_lods[lo + 4] < 0.0);
        let mat = if fi < face_materials.len() { face_materials[fi] } else { 0 };

        let key = atlas_idx * (2 * num_materials) + parity_bucket * num_materials + mat;
        if key < num_buckets {
            buckets[key].push(fi as u32);
        }
    }

    // Collect non-empty buckets into DrawBatches
    let mut result = Vec::new();
    for (key, faces) in buckets.into_iter().enumerate() {
        if faces.is_empty() { continue; }

        let mat = key % num_materials;
        let parity_bucket = (key / num_materials) % 2;

        // Read canonical LODs and parity from first face in bucket
        let fi0 = faces[0] as usize;
        let lo = fi0 * FACE_LOD_STRIDE;
        let ca = face_lods[lo] as u32;
        let cb = face_lods[lo + 1] as u32;
        let cc = face_lods[lo + 2] as u32;
        let parity = if parity_bucket == 0 { 1.0 } else { -1.0 };

        // Pack instance data
        let mut instance_data = Vec::with_capacity(faces.len() * INSTANCE_STRIDE);
        for &fi in &faces {
            let start = fi as usize * INSTANCE_STRIDE;
            instance_data.extend_from_slice(&all_instances[start..start + INSTANCE_STRIDE]);
            let dst = instance_data.len() - INSTANCE_STRIDE;
            let lod_offset = fi as usize * FACE_LOD_STRIDE;
            instance_data[dst + crate::instance_layout::offset::PERM_INDEX] =
                face_lods[lod_offset + 3];
            instance_data[dst + crate::instance_layout::offset::FACE_ID] = fi as f32;
        }

        result.push(DrawBatch {
            lod: [ca, cb, cc],
            parity,
            material_index: mat,
            tess_key: TessKey { lod: [ca, cb, cc] },
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
    let num_buckets = 256 * 2 * num_materials;

    // Phase 1: bucket sort to get face ordering
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_buckets];
    for fi in 0..num_faces {
        let lo = fi * FACE_LOD_STRIDE;
        if !face_is_visible(face_lods, fi) { continue; }
        let atlas_idx = face_lods[lo + 5] as usize;
        let parity_bucket = usize::from(face_lods[lo + 4] < 0.0);
        let mat = if fi < face_materials.len() { face_materials[fi] } else { 0 };
        let key = atlas_idx * (2 * num_materials) + parity_bucket * num_materials + mat;
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
        let parity_bucket = (key / num_materials) % 2;

        let fi0 = faces[0] as usize;
        let lo = fi0 * FACE_LOD_STRIDE;
        let ca = face_lods[lo] as u32;
        let cb = face_lods[lo + 1] as u32;
        let cc = face_lods[lo + 2] as u32;
        let parity = if parity_bucket == 0 { 1.0 } else { -1.0 };

        let offset = sorted_order.len();
        sorted_order.extend_from_slice(&faces);

        ranges.push(BatchRange {
            lod: [ca, cb, cc],
            parity,
            material_index: mat,
            tess_key: TessKey { lod: [ca, cb, cc] },
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
        let lod_offset = src_fi as usize * FACE_LOD_STRIDE;
        sorted[dst_off + crate::instance_layout::offset::PERM_INDEX] =
            face_lods[lod_offset + 3];
        sorted[dst_off + crate::instance_layout::offset::FACE_ID] = src_fi as f32;
    }
    all_instances[..num_faces * INSTANCE_STRIDE].copy_from_slice(&sorted[..num_faces * INSTANCE_STRIDE]);

    ranges
}

/// Pass 2 encodes conservative visibility by reserving negative atlas indices.
#[inline]
pub fn face_is_visible(face_lods: &[f32], face_index: usize) -> bool {
    face_lods
        .get(face_index * FACE_LOD_STRIDE + 5)
        .is_some_and(|atlas_index| atlas_index.is_finite() && *atlas_index >= 0.0)
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
    fn invisible_payload_does_not_replace_resident_topology() {
        let visible = [16.0, 32.0, 64.0, 4.0, 1.0, 17.0];
        let invisible = [1.0, 1.0, 1.0, 0.0, 1.0, -1.0];
        let resident = ResidentLod::from_visible_payload(&visible, 0).unwrap();

        assert_eq!(resident.canonical, [16, 32, 64]);
        assert_eq!(resident.perm_index, 4);
        assert_eq!(ResidentLod::from_visible_payload(&invisible, 0), None);
        let mut stored = Some(resident);
        assert_eq!(
            retain_face_lod(&invisible, 0, &mut stored, ResidentLod::uniform(2)),
            resident,
        );
        assert_eq!(stored, Some(resident));
    }

    #[test]
    fn retained_topology_reconciles_across_welded_attribute_seams() {
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [3, 4, 5]];
        let topology = HalfEdgeMesh::from_triangles_welded_exact(&positions, &faces);
        let mut residents = [
            Some(ResidentLod::from_edge_lods([8, 2, 2])),
            Some(ResidentLod::from_edge_lods([2, 2, 32])),
        ];

        assert_eq!(reconcile_resident_edges(&mut residents, &topology), 1);
        assert_eq!(residents[0].unwrap().edge_lods(), [32, 2, 2]);
        assert_eq!(residents[1].unwrap().edge_lods(), [2, 2, 32]);
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
    fn merges_permutations_with_the_same_parity() {
        let face_lods = vec![
            4.0, 8.0, 16.0, 0.0, 1.0, 7.0,
            4.0, 8.0, 16.0, 3.0, 1.0, 7.0,
            4.0, 8.0, 16.0, 1.0, -1.0, 7.0,
        ];
        let all_instances = vec![0.0f32; 3 * INSTANCE_STRIDE];
        let batches = group_into_batches(&face_lods, &all_instances, &[0, 0, 0], 3);

        assert_eq!(batches.len(), 2, "only winding parity should split draws");
        let even = batches.iter().find(|batch| batch.parity > 0.0).unwrap();
        assert_eq!(even.face_indices, vec![0, 1]);
        assert_eq!(
            even.instance_data[crate::instance_layout::offset::PERM_INDEX],
            0.0,
        );
        assert_eq!(
            even.instance_data[INSTANCE_STRIDE + crate::instance_layout::offset::PERM_INDEX],
            3.0,
        );
        let odd = batches.iter().find(|batch| batch.parity < 0.0).unwrap();
        assert_eq!(
            odd.instance_data[crate::instance_layout::offset::PERM_INDEX],
            1.0,
        );
    }

    #[test]
    fn empty_input() {
        let batches = group_into_batches(&[], &[], &[], 0);
        assert!(batches.is_empty());
    }

    #[test]
    fn visibility_sentinel_omits_faces_and_preserves_source_ids() {
        let face_lods = vec![
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,
            4.0, 4.0, 4.0, 0.0, 1.0, -1.0,
            4.0, 4.0, 4.0, 0.0, 1.0, 0.0,
        ];
        let all_instances = vec![0.0f32; 3 * INSTANCE_STRIDE];
        let batches = group_into_batches(&face_lods, &all_instances, &[0, 0, 0], 3);

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].face_indices, vec![0, 2]);
        assert_eq!(
            batches[0].instance_data[crate::instance_layout::offset::FACE_ID],
            0.0,
        );
        assert_eq!(
            batches[0].instance_data[INSTANCE_STRIDE + crate::instance_layout::offset::FACE_ID],
            2.0,
        );
    }
}
