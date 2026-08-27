//! Batch grouping: group faces by (atlas index, permutation parity, material) for instanced draw.
//!
//! Uses O(n) bucket sort with direct indexing — the key space is bounded
//! (atlas 0-255 × two winding parities × materials) and small enough for a flat Vec.

use quilting_mesh::HalfEdgeMesh;
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

use crate::screen_leaf_lod::{
    ScreenLeafLodError, ScreenMeshLeafFrontier, ScreenMeshLeafLodScratch, ScreenMeshLeafTopology,
};
use crate::screen_partition::ScreenPatchLeafId;

/// Instance data stride in floats. Re-exported from [`crate::instance_layout`],
/// which is the normative definition — do not restate the number here.
pub use crate::instance_layout::STRIDE as INSTANCE_STRIDE;

/// Per-face LOD stride: 6 floats from GPU pass 2.
pub const FACE_LOD_STRIDE: usize = 6;

/// Structural failure while admitting one full or sparse classifier
/// publication into retained renderer topology.
///
/// Validation completes before any retained vector is changed. This lets a
/// WebGL worker, same-context WebGL classifier, or future WebGPU backend share
/// one fail-closed semantic boundary instead of partially applying malformed
/// topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceLodPublicationError {
    PayloadLength,
    FaceIndexOutOfBounds,
    FaceIndicesNotIncreasing,
}

impl std::fmt::Display for FaceLodPublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PayloadLength => write!(formatter, "face LOD publication has the wrong length"),
            Self::FaceIndexOutOfBounds => {
                write!(formatter, "face LOD publication index is out of bounds")
            }
            Self::FaceIndicesNotIncreasing => {
                write!(formatter, "face LOD publication indices are not strictly increasing")
            }
        }
    }
}

impl std::error::Error for FaceLodPublicationError {}

/// Reusable retained vectors changed by [`admit_face_lod_publication`].
pub struct FaceLodAdmissionBuffers<'a> {
    pub requested: &'a mut Vec<Option<ResidentLod>>,
    pub resident: &'a mut Vec<Option<ResidentLod>>,
    pub visible: &'a mut Vec<bool>,
    pub dirty_faces: &'a mut Vec<usize>,
}

/// Result of validating and admitting one classifier publication before
/// crack-free reconciliation and GPU resource publication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceLodAdmissionResult {
    pub full_snapshot: bool,
    pub records: usize,
    pub changed_records: usize,
    pub culled_faces: usize,
}

impl FaceLodAdmissionResult {
    pub fn topology_changed(self) -> bool {
        self.changed_records != 0
    }
}

/// Compare a completed GPU classification with the previous snapshot and
/// retain only changed face records. Returns `true` when the current payload
/// must be treated as a full snapshot (initial result or a size change).
///
/// Keeping this below the WASM boundary avoids copying and scanning a full
/// mesh-sized typed array in JavaScript before the sparse worker transfer.
pub fn encode_face_lod_delta(
    current: &[f32],
    previous: &mut Vec<f32>,
    changed_faces: &mut Vec<u32>,
    changed_lods: &mut Vec<f32>,
) -> bool {
    changed_faces.clear();
    changed_lods.clear();

    if previous.len() != current.len() || !current.len().is_multiple_of(FACE_LOD_STRIDE) {
        previous.clear();
        previous.extend_from_slice(current);
        changed_lods.extend_from_slice(current);
        return true;
    }

    for (face_index, record) in current.chunks_exact(FACE_LOD_STRIDE).enumerate() {
        let offset = face_index * FACE_LOD_STRIDE;
        let prior = &mut previous[offset..offset + FACE_LOD_STRIDE];
        if record.iter().zip(prior.iter()).any(|(next, old)| next != old) {
            changed_faces.push(face_index as u32);
            changed_lods.extend_from_slice(record);
            prior.copy_from_slice(record);
        }
    }
    false
}

/// Monotonic identity of one full or sparse face-LOD publication.
///
/// A sparse publication is only meaningful relative to `base_revision` in the
/// same epoch. Resetting the encoder advances `epoch`, and the next publication
/// is a self-contained revision-one snapshot. This makes skipped/crossed
/// worker messages detectable instead of silently applying a delta to the
/// wrong resident topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceLodDeltaSequence {
    pub epoch: u32,
    pub base_revision: u32,
    pub revision: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceLodDeltaError {
    InvalidPayload,
    TooManyFaces,
    InvalidSequence,
    MissingBase,
}

impl std::fmt::Display for FaceLodDeltaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPayload => write!(formatter, "face LOD payload has an invalid stride"),
            Self::TooManyFaces => write!(formatter, "face LOD payload exceeds u32 indexing"),
            Self::InvalidSequence => write!(formatter, "face LOD delta sequence is invalid"),
            Self::MissingBase => write!(formatter, "face LOD delta does not extend the resident base"),
        }
    }
}

impl std::error::Error for FaceLodDeltaError {}

/// Borrowed full/sparse publication produced by [`FaceLodDeltaEncoder`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodedFaceLodDelta<'a> {
    pub sequence: FaceLodDeltaSequence,
    pub full_snapshot: bool,
    pub changed_faces: &'a [u32],
    pub lods: &'a [f32],
}

/// Retained classifier baseline and reusable sparse-delta work storage.
#[derive(Clone, Debug)]
pub struct FaceLodDeltaEncoder {
    epoch: u32,
    revision: u32,
    previous: Vec<f32>,
    changed_faces: Vec<u32>,
    changed_lods: Vec<f32>,
}

impl Default for FaceLodDeltaEncoder {
    fn default() -> Self {
        Self {
            epoch: 1,
            revision: 0,
            previous: Vec::new(),
            changed_faces: Vec::new(),
            changed_lods: Vec::new(),
        }
    }
}

impl FaceLodDeltaEncoder {
    /// Start a new publication epoch. The next encoded result is necessarily a
    /// full revision-one snapshot, even when its face count is unchanged.
    pub fn reset(&mut self) {
        self.epoch = self.epoch.checked_add(1).unwrap_or(1);
        self.revision = 0;
        self.previous.clear();
        self.changed_faces.clear();
        self.changed_lods.clear();
    }

    pub fn encode(
        &mut self,
        current: &[f32],
    ) -> Result<EncodedFaceLodDelta<'_>, FaceLodDeltaError> {
        if !current.len().is_multiple_of(FACE_LOD_STRIDE) {
            return Err(FaceLodDeltaError::InvalidPayload);
        }
        if current.len() / FACE_LOD_STRIDE > u32::MAX as usize {
            return Err(FaceLodDeltaError::TooManyFaces);
        }
        if self.revision == u32::MAX {
            self.reset();
        }
        let base_revision = self.revision;
        let full_snapshot = encode_face_lod_delta(
            current,
            &mut self.previous,
            &mut self.changed_faces,
            &mut self.changed_lods,
        );
        self.revision += 1;
        Ok(EncodedFaceLodDelta {
            sequence: FaceLodDeltaSequence {
                epoch: self.epoch,
                base_revision,
                revision: self.revision,
            },
            full_snapshot,
            changed_faces: &self.changed_faces,
            lods: &self.changed_lods,
        })
    }
}

/// Receiver-side oracle for full/sparse publication continuity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceLodDeltaCursor {
    epoch: u32,
    revision: u32,
}

impl FaceLodDeltaCursor {
    pub fn sequence(self) -> Option<(u32, u32)> {
        (self.revision > 0).then_some((self.epoch, self.revision))
    }

    /// Validate and accept a publication identity. A revision-one full
    /// snapshot is an explicit stream reset; every other message must extend
    /// the exact accepted base in the same epoch.
    pub fn accept(
        &mut self,
        sequence: FaceLodDeltaSequence,
        full_snapshot: bool,
    ) -> Result<bool, FaceLodDeltaError> {
        let expected_revision = sequence.base_revision.checked_add(1);
        if sequence.epoch == 0
            || sequence.revision == 0
            || expected_revision != Some(sequence.revision) {
            return Err(FaceLodDeltaError::InvalidSequence);
        }
        let reset = full_snapshot
            && sequence.base_revision == 0
            && sequence.revision == 1;
        if !reset
            && (self.revision == 0
                || sequence.epoch != self.epoch
                || sequence.base_revision != self.revision)
        {
            return Err(FaceLodDeltaError::MissingBase);
        }
        self.epoch = sequence.epoch;
        self.revision = sequence.revision;
        Ok(reset)
    }
}

/// Canonical tessellation topology requested or kept resident for one face.
/// The renderer stores raw requests separately from their crack-free promoted
/// closure so later classifications can genuinely lower detail.
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

    /// Decode the topology fields of a six-float GPU classification payload,
    /// including the drawable standby carried beside an invisible sentinel.
    pub fn from_payload_topology(face_lods: &[f32], face_index: usize) -> Option<Self> {
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

    /// Decode a valid, visible six-float GPU classification payload.
    pub fn from_visible_payload(face_lods: &[f32], face_index: usize) -> Option<Self> {
        face_is_visible(face_lods, face_index)
            .then(|| Self::from_payload_topology(face_lods, face_index))?
    }
}

/// Backend-neutral classifier result for one source face.
///
/// WebGL worker floats, packed same-context WebGL output, and future WebGPU
/// compute output all enter retained topology through this semantic record.
/// Atlas residency is validated by the producing backend; batching needs only
/// the requested topology and conservative visibility decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceLodClassification {
    pub requested: ResidentLod,
    pub visible: bool,
}

/// Backend-neutral identity of one hardware draw/compaction bucket.
///
/// WebGL2 uses this key to retain per-bucket attribute buffers and VAOs.
/// A future WebGPU compute pass can emit the same key beside compacted face
/// indices before indirect draw generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderBatchKey {
    pub lod: [u32; 3],
    pub parity_bucket: u8,
    pub material_index: usize,
    /// Representative node whose render transform/state is shared by every
    /// member. This is deliberately distinct from each member's semantic
    /// source node so world-baked glTF primitives can share one draw bucket.
    pub render_node_index: usize,
}

impl Ord for RenderBatchKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Submission order is deliberately state-major. Batch identity still
        // contains the exact same fields, but grouping material and authored
        // node before atlas topology lets both WebGL2 and future backends retain
        // expensive pipeline/resource state across adjacent draws.
        (
            self.material_index,
            self.render_node_index,
            self.lod,
            self.parity_bucket,
        )
            .cmp(&(
                other.material_index,
                other.render_node_index,
                other.lod,
                other.parity_bucket,
            ))
    }
}

impl PartialOrd for RenderBatchKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl RenderBatchKey {
    pub fn from_resident(
        resident: ResidentLod,
        material_index: usize,
        render_node_index: usize,
    ) -> Self {
        Self {
            lod: resident.canonical,
            parity_bucket: resident.parity_bucket.min(1) as u8,
            material_index,
            render_node_index,
        }
    }

    pub fn parity(self) -> f32 {
        if self.parity_bucket == 0 { 1.0 } else { -1.0 }
    }
}

/// Physical publication role of one backend-neutral draw bucket.
///
/// `Complete` is the incumbent monolithic frontier. A retained adaptive scene
/// instead contains `RetainedRoot` buckets plus `AdaptiveOverlay` buckets and
/// an exact scene-level source-root suppression set.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderBatchLayer {
    #[default]
    Complete,
    RetainedRoot,
    AdaptiveOverlay,
}

/// Stable resource identity shared by WebGL2 maps and future WebGPU
/// extraction. Layer is ordered inside the ordinary material/node/topology
/// key so equal state remains adjacent while root precedes overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderBatchId {
    pub key: RenderBatchKey,
    pub layer: RenderBatchLayer,
}

impl RenderBatchId {
    pub const fn complete(key: RenderBatchKey) -> Self {
        Self {
            key,
            layer: RenderBatchLayer::Complete,
        }
    }
}

impl Ord for RenderBatchId {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.key, self.layer).cmp(&(other.key, other.layer))
    }
}

impl PartialOrd for RenderBatchId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Minimal per-face record needed to detect whether a retained batch's source
/// instance stream is still valid. Canonical LOD, parity, material, and node
/// are already represented by [`RenderBatchKey`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderBatchMember {
    pub face_index: u32,
    /// Ephemeral dyadic identity within the stable source face. Legacy source
    /// triangles use `ROOT`; adaptive extraction may emit several distinct
    /// leaves without pretending they are separate authored entities.
    pub leaf_id: ScreenPatchLeafId,
    /// Stable semantic glTF node identity. Unlike the render-state
    /// representative in [`RenderBatchKey`], this survives draw consolidation
    /// for picking, selection, attribution, and future scene extraction.
    pub node_index: usize,
    /// Face-local edge resolutions for this drawable patch. Adaptive leaves
    /// of one authored face may differ, so this cannot be recovered from a
    /// source-face-indexed resident array.
    pub edge_lods: [u32; 3],
    pub permutation_index: u8,
    /// Compatibility corner maxima in absolute source-domain units. The
    /// normative density visualization is derived from reconciled edge LoDs;
    /// this retained field stays populated until the prepared-record ABI can
    /// remove it behind backend-equivalence gates.
    pub vertex_lods: [u32; 3],
}

impl RenderBatchMember {
    pub fn patch_id(self) -> (u32, ScreenPatchLeafId) {
        (self.face_index, self.leaf_id)
    }
}

/// Sparse replacement layer over a complete resident root grouping.
///
/// A backend draws the retained root groups except for `suppressed_faces`, then
/// draws `groups`. Composing those two layers is exactly equivalent to grouping
/// the complete adaptive frontier, while the replacement layer contains only
/// source faces whose drawable patch membership or topology changed.
#[derive(Debug, Default)]
pub struct AdaptiveRenderOverlay {
    pub suppressed_faces: Vec<u32>,
    pub groups: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    face_leaf_counts: Vec<usize>,
    face_matches_baseline_root: Vec<bool>,
}

/// Exact scene work represented by retained roots plus one sparse adaptive
/// overlay. Counting root buckets is proportional to render-state diversity;
/// only the explicitly suppressed faces require per-face lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveRenderWork {
    pub baseline_triangles: u64,
    pub suppressed_root_triangles: u64,
    pub overlay_triangles: u64,
    pub composed_triangles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveRenderWorkError {
    MissingAtlasPatch { lod: [u32; 3] },
    BaselineCoverage { expected: usize, actual: usize },
    SuppressedFaceOutOfBounds { source_face: u32 },
    SuppressedFacesNotIncreasing { source_face: u32 },
    TriangleCountOverflow,
    SuppressedWorkExceedsBaseline,
}

impl std::fmt::Display for AdaptiveRenderWorkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAtlasPatch { lod } => {
                write!(formatter, "adaptive render work needs missing atlas patch {lod:?}")
            }
            Self::BaselineCoverage { expected, actual } => write!(
                formatter,
                "retained root groups cover {actual} faces; expected {expected}",
            ),
            Self::SuppressedFaceOutOfBounds { source_face } => write!(
                formatter,
                "suppressed source face {source_face} is outside the retained root field",
            ),
            Self::SuppressedFacesNotIncreasing { source_face } => write!(
                formatter,
                "suppressed source faces are not strictly increasing at {source_face}",
            ),
            Self::TriangleCountOverflow => {
                write!(formatter, "adaptive render triangle count overflowed")
            }
            Self::SuppressedWorkExceedsBaseline => write!(
                formatter,
                "suppressed root work exceeds the complete retained baseline",
            ),
        }
    }
}

impl std::error::Error for AdaptiveRenderWorkError {}

fn grouped_triangle_count(
    groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
) -> Result<(u64, usize), AdaptiveRenderWorkError> {
    let mut triangles = 0u64;
    let mut members = 0usize;
    for (key, group) in groups {
        let patch_triangles = atlas_triangle_counts
            .get(&key.lod)
            .copied()
            .ok_or(AdaptiveRenderWorkError::MissingAtlasPatch { lod: key.lod })?;
        let member_count = u64::try_from(group.len())
            .map_err(|_| AdaptiveRenderWorkError::TriangleCountOverflow)?;
        let group_triangles = patch_triangles
            .checked_mul(member_count)
            .ok_or(AdaptiveRenderWorkError::TriangleCountOverflow)?;
        triangles = triangles
            .checked_add(group_triangles)
            .ok_or(AdaptiveRenderWorkError::TriangleCountOverflow)?;
        members = members
            .checked_add(group.len())
            .ok_or(AdaptiveRenderWorkError::TriangleCountOverflow)?;
    }
    Ok((triangles, members))
}

/// Validate atlas residency and count the exact physical triangles drawn by a
/// retained-root/overlay composition without scanning unaffected source faces.
pub fn measure_adaptive_render_work(
    baseline_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    overlay: &AdaptiveRenderOverlay,
    baseline_residents: &[Option<ResidentLod>],
    initial: ResidentLod,
    atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
) -> Result<AdaptiveRenderWork, AdaptiveRenderWorkError> {
    let (baseline_triangles, baseline_members) =
        grouped_triangle_count(baseline_groups, atlas_triangle_counts)?;
    if baseline_members != baseline_residents.len() {
        return Err(AdaptiveRenderWorkError::BaselineCoverage {
            expected: baseline_residents.len(),
            actual: baseline_members,
        });
    }

    let mut suppressed_root_triangles = 0u64;
    let mut previous_face = None;
    for &source_face in &overlay.suppressed_faces {
        if previous_face.is_some_and(|previous| previous >= source_face) {
            return Err(AdaptiveRenderWorkError::SuppressedFacesNotIncreasing {
                source_face,
            });
        }
        let resident = baseline_residents
            .get(source_face as usize)
            .ok_or(AdaptiveRenderWorkError::SuppressedFaceOutOfBounds { source_face })?
            .unwrap_or(initial);
        let patch_triangles = atlas_triangle_counts
            .get(&resident.canonical)
            .copied()
            .ok_or(AdaptiveRenderWorkError::MissingAtlasPatch {
                lod: resident.canonical,
            })?;
        suppressed_root_triangles = suppressed_root_triangles
            .checked_add(patch_triangles)
            .ok_or(AdaptiveRenderWorkError::TriangleCountOverflow)?;
        previous_face = Some(source_face);
    }

    let retained_triangles = baseline_triangles
        .checked_sub(suppressed_root_triangles)
        .ok_or(AdaptiveRenderWorkError::SuppressedWorkExceedsBaseline)?;
    let (overlay_triangles, _) =
        grouped_triangle_count(&overlay.groups, atlas_triangle_counts)?;
    let composed_triangles = retained_triangles
        .checked_add(overlay_triangles)
        .ok_or(AdaptiveRenderWorkError::TriangleCountOverflow)?;
    Ok(AdaptiveRenderWork {
        baseline_triangles,
        suppressed_root_triangles,
        overlay_triangles,
        composed_triangles,
    })
}

/// Rebuild the C0-continuous resident LOD field used by diagnostic shading.
///
/// Each compact topology vertex receives the maximum resolution of every
/// incident edge. Faces sharing that vertex consequently upload the same value
/// at the shared corner, so barycentric log interpolation has no face seams.
/// The work vectors are retained by the caller to avoid mesh-sized allocation
/// churn after asynchronous classifications.
pub fn rebuild_resident_vertex_lods(
    residents: &[Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    initial: ResidentLod,
    vertex_max: &mut Vec<u32>,
    face_vertex_lods: &mut Vec<[u32; 3]>,
) {
    let num_faces = residents.len().min(topology.num_faces as usize);
    vertex_max.clear();
    vertex_max.resize(topology.num_vertices as usize, 1);
    face_vertex_lods.clear();
    face_vertex_lods.resize(residents.len(), [1; 3]);

    for face in 0..num_faces {
        let [v0, v1, v2] = topology.face_vertices(face as u32).map(|vertex| vertex as usize);
        let [edge_a, edge_b, edge_c] = residents[face].unwrap_or(initial).edge_lods();
        vertex_max[v0] = vertex_max[v0].max(edge_b).max(edge_c);
        vertex_max[v1] = vertex_max[v1].max(edge_a).max(edge_c);
        vertex_max[v2] = vertex_max[v2].max(edge_a).max(edge_b);
    }
    for (face, output) in face_vertex_lods.iter_mut().take(num_faces).enumerate() {
        let vertices = topology.face_vertices(face as u32).map(|vertex| vertex as usize);
        *output = vertices.map(|vertex| vertex_max[vertex]);
    }
}

/// Deterministically group resident faces into backend-neutral draw buckets.
/// Faces are visited in source order, so equal memberships compare byte-for-
/// byte across LOD updates without sorting or hashing each bucket afterward.
pub fn group_resident_faces(
    residents: &[Option<ResidentLod>],
    face_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
) -> BTreeMap<RenderBatchKey, Vec<RenderBatchMember>> {
    let mut groups = BTreeMap::<RenderBatchKey, Vec<RenderBatchMember>>::new();
    group_resident_faces_into(
        residents,
        face_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        initial,
        &mut groups,
    );
    groups
}

/// Rebuild draw-bucket memberships while retaining the map and vector
/// allocations from the previous classification.
pub fn group_resident_faces_into(
    residents: &[Option<ResidentLod>],
    face_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
    groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
) {
    for members in groups.values_mut() {
        members.clear();
    }
    for (face_index, resident) in residents.iter().enumerate() {
        let resident = resident.unwrap_or(initial);
        let key = RenderBatchKey::from_resident(
            resident,
            face_materials.get(face_index).copied().unwrap_or(0),
            face_render_nodes
                .get(face_index)
                .copied()
                .or_else(|| face_nodes.get(face_index).copied())
                .unwrap_or(0),
        );
        groups.entry(key).or_default().push(RenderBatchMember {
            face_index: face_index as u32,
            leaf_id: ScreenPatchLeafId::ROOT,
            node_index: face_nodes.get(face_index).copied().unwrap_or(0),
            edge_lods: resident.edge_lods(),
            permutation_index: resident.perm_index.min(5) as u8,
            vertex_lods: face_vertex_lods.get(face_index).copied().unwrap_or([1; 3]),
        });
    }
    groups.retain(|_, members| !members.is_empty());
}

/// Deterministically group reconciled adaptive leaves into the same
/// backend-neutral draw buckets used by source-face residency.
///
/// Authored face identity supplies material, semantic node, and render node;
/// dyadic leaf identity remains only the ephemeral drawable-patch suffix.
/// Leaf topology and LoDs are deliberately validated instead of being
/// silently truncated, because dropping or overlapping one leaf creates a
/// geometric hole or a double-rendered region.
pub fn group_resident_screen_leaves(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
) -> Result<BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>, ScreenLeafLodError> {
    let mut groups = BTreeMap::<RenderBatchKey, Vec<RenderBatchMember>>::new();
    let mut lod_scratch = ScreenMeshLeafLodScratch::default();
    group_resident_screen_leaves_into(
        frontier,
        resident_edge_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        &mut lod_scratch,
        &mut groups,
    )?;
    Ok(groups)
}

/// Retained-allocation variant of [`group_resident_screen_leaves`].
pub fn group_resident_screen_leaves_into(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    lod_scratch: &mut ScreenMeshLeafLodScratch,
    groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
) -> Result<(), ScreenLeafLodError> {
    let leaves = frontier.leaves();
    if leaves.len() != resident_edge_lods.len() {
        return Err(ScreenLeafLodError::LengthMismatch);
    }
    let leaf_vertex_lods = frontier.rebuild_vertex_lods_into(resident_edge_lods, lod_scratch)?;
    for (leaf_index, leaf) in leaves.iter().copied().enumerate() {
        let source_face = leaf.source_face as usize;
        if source_face >= face_materials.len()
            || source_face >= face_nodes.len()
            || source_face >= face_render_nodes.len()
        {
            return Err(ScreenLeafLodError::MissingFaceMetadata {
                leaf_index,
                source_face: leaf.source_face,
            });
        }
    }
    for members in groups.values_mut() {
        members.clear();
    }
    for ((leaf, edge_lods), vertex_lods) in leaves
        .iter()
        .copied()
        .zip(resident_edge_lods.iter().copied())
        .zip(leaf_vertex_lods.iter().copied())
    {
        let source_face = leaf.source_face as usize;
        let resident = ResidentLod::from_edge_lods(edge_lods);
        let material_index = face_materials[source_face];
        let node_index = face_nodes[source_face];
        let render_node_index = face_render_nodes[source_face];
        let key = RenderBatchKey::from_resident(resident, material_index, render_node_index);
        groups.entry(key).or_default().push(RenderBatchMember {
            face_index: leaf.source_face,
            leaf_id: leaf.id,
            node_index,
            edge_lods,
            permutation_index: resident.perm_index.min(5) as u8,
            vertex_lods,
        });
    }
    groups.retain(|_, members| !members.is_empty());
    Ok(())
}

/// Extract only the source-face replacements needed above retained root
/// batches. The input frontier must cover every baseline source face.
///
/// Validation and corner-density reconstruction finish before the public
/// overlay is changed, so malformed candidates cannot partially replace a
/// previously valid layer. Internal face scratch retains its allocation.
#[allow(clippy::too_many_arguments)]
pub fn group_resident_screen_overlay_into(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    baseline_residents: &[Option<ResidentLod>],
    baseline_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
    lod_scratch: &mut ScreenMeshLeafLodScratch,
    overlay: &mut AdaptiveRenderOverlay,
) -> Result<(), ScreenLeafLodError> {
    group_resident_screen_overlay_coverage_into(
        frontier,
        resident_edge_lods,
        AdaptiveOverlayCoverage::Complete(baseline_residents.len()),
        None,
        baseline_residents,
        baseline_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        initial,
        lod_scratch,
        overlay,
    )
}

/// Extract replacements for complete, independently recomputable welded-source
/// components. Faces outside `component_faces` remain represented by retained
/// baseline roots and are neither scanned nor copied into the overlay.
///
/// `component_faces` must be the globally ordered certificate produced by
/// [`ScreenMeshTopologyCache::collect_component_closure`]. The frontier must
/// cover every certified face and no others.
#[allow(clippy::too_many_arguments)]
pub fn group_resident_screen_component_overlay_into(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    component_faces: &[u32],
    baseline_residents: &[Option<ResidentLod>],
    baseline_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
    lod_scratch: &mut ScreenMeshLeafLodScratch,
    overlay: &mut AdaptiveRenderOverlay,
) -> Result<(), ScreenLeafLodError> {
    group_resident_screen_overlay_coverage_into(
        frontier,
        resident_edge_lods,
        AdaptiveOverlayCoverage::Components(component_faces),
        None,
        baseline_residents,
        baseline_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        initial,
        lod_scratch,
        overlay,
    )
}

/// Extract exact replacements for a bounded mutable neighborhood plus fixed
/// root observers. Fixed observer corners max-compose the already-reconciled
/// baseline field with local changes, preserving contributions from retained
/// faces outside the local frontier.
#[allow(clippy::too_many_arguments)]
pub fn group_resident_screen_neighborhood_overlay_into(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    observed_faces: &[u32],
    fixed_leaves: &[bool],
    baseline_residents: &[Option<ResidentLod>],
    baseline_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
    lod_scratch: &mut ScreenMeshLeafLodScratch,
    overlay: &mut AdaptiveRenderOverlay,
) -> Result<(), ScreenLeafLodError> {
    group_resident_screen_overlay_coverage_into(
        frontier,
        resident_edge_lods,
        AdaptiveOverlayCoverage::Components(observed_faces),
        Some(fixed_leaves),
        baseline_residents,
        baseline_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        initial,
        lod_scratch,
        overlay,
    )
}

#[derive(Clone, Copy)]
enum AdaptiveOverlayCoverage<'a> {
    Complete(usize),
    Components(&'a [u32]),
}

impl AdaptiveOverlayCoverage<'_> {
    fn len(self) -> usize {
        match self {
            Self::Complete(source_faces) => source_faces,
            Self::Components(component_faces) => component_faces.len(),
        }
    }

    fn source_face(self, local_face: usize) -> Option<u32> {
        match self {
            Self::Complete(source_faces) => (local_face < source_faces).then_some(local_face as u32),
            Self::Components(component_faces) => component_faces.get(local_face).copied(),
        }
    }

    fn local_face(self, source_face: u32) -> Option<usize> {
        match self {
            Self::Complete(source_faces) => {
                ((source_face as usize) < source_faces).then_some(source_face as usize)
            }
            Self::Components(component_faces) => component_faces.binary_search(&source_face).ok(),
        }
    }

    fn validate(self, source_faces: usize) -> Result<(), ScreenLeafLodError> {
        let Self::Components(component_faces) = self else {
            return Ok(());
        };
        let mut previous = None;
        for &source_face in component_faces {
            if source_face as usize >= source_faces
                || previous.is_some_and(|previous| previous >= source_face)
            {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index: 0 });
            }
            previous = Some(source_face);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn group_resident_screen_overlay_coverage_into(
    frontier: &ScreenMeshLeafFrontier,
    resident_edge_lods: &[[u32; 3]],
    coverage: AdaptiveOverlayCoverage<'_>,
    fixed_leaves: Option<&[bool]>,
    baseline_residents: &[Option<ResidentLod>],
    baseline_vertex_lods: &[[u32; 3]],
    face_materials: &[usize],
    face_nodes: &[usize],
    face_render_nodes: &[usize],
    initial: ResidentLod,
    lod_scratch: &mut ScreenMeshLeafLodScratch,
    overlay: &mut AdaptiveRenderOverlay,
) -> Result<(), ScreenLeafLodError> {
    let leaves = frontier.leaves();
    let source_faces = baseline_residents.len();
    if leaves.len() != resident_edge_lods.len()
        || fixed_leaves.is_some_and(|fixed| fixed.len() != leaves.len())
        || baseline_vertex_lods.len() != source_faces
        || face_materials.len() != source_faces
        || face_nodes.len() != source_faces
        || face_render_nodes.len() != source_faces
    {
        return Err(ScreenLeafLodError::LengthMismatch);
    }
    coverage.validate(source_faces)?;
    let leaf_vertex_lods = frontier.rebuild_vertex_lods_into(resident_edge_lods, lod_scratch)?;

    overlay.face_leaf_counts.clear();
    overlay.face_leaf_counts.resize(coverage.len(), 0);
    overlay.face_matches_baseline_root.clear();
    overlay
        .face_matches_baseline_root
        .resize(coverage.len(), false);
    for (leaf_index, ((leaf, edge_lods), vertex_lods)) in leaves
        .iter()
        .copied()
        .zip(resident_edge_lods.iter().copied())
        .zip(leaf_vertex_lods.iter().copied())
        .enumerate()
    {
        let source_face = leaf.source_face as usize;
        let vertex_lods = compose_fixed_observer_vertex_lods(
            leaf_index,
            leaf,
            vertex_lods,
            fixed_leaves,
            baseline_vertex_lods,
        )?;
        let Some(local_face) = coverage.local_face(leaf.source_face) else {
            return Err(ScreenLeafLodError::MissingFaceMetadata {
                leaf_index,
                source_face: leaf.source_face,
            });
        };
        overlay.face_leaf_counts[local_face] += 1;
        if leaf.id == ScreenPatchLeafId::ROOT {
            let resident = ResidentLod::from_edge_lods(edge_lods);
            overlay.face_matches_baseline_root[local_face] = resident
                == baseline_residents[source_face].unwrap_or(initial)
                && vertex_lods == baseline_vertex_lods[source_face];
        }
    }
    if let Some(source_face) = overlay
        .face_leaf_counts
        .iter()
        .position(|&leaf_count| leaf_count == 0)
    {
        return Err(ScreenLeafLodError::MissingSourceFace {
            source_face: coverage
                .source_face(source_face)
                .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?,
        });
    }

    overlay.suppressed_faces.clear();
    for members in overlay.groups.values_mut() {
        members.clear();
    }
    for local_face in 0..coverage.len() {
        if overlay.face_leaf_counts[local_face] != 1
            || !overlay.face_matches_baseline_root[local_face]
        {
            overlay.suppressed_faces.push(
                coverage
                    .source_face(local_face)
                    .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?,
            );
        }
    }
    for (leaf_index, ((leaf, edge_lods), vertex_lods)) in leaves
        .iter()
        .copied()
        .zip(resident_edge_lods.iter().copied())
        .zip(leaf_vertex_lods.iter().copied())
        .enumerate()
    {
        let source_face = leaf.source_face as usize;
        let vertex_lods = compose_fixed_observer_vertex_lods(
            leaf_index,
            leaf,
            vertex_lods,
            fixed_leaves,
            baseline_vertex_lods,
        )?;
        let local_face = coverage
            .local_face(leaf.source_face)
            .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
        if overlay.face_leaf_counts[local_face] == 1
            && overlay.face_matches_baseline_root[local_face]
        {
            continue;
        }
        let resident = ResidentLod::from_edge_lods(edge_lods);
        let key = RenderBatchKey::from_resident(
            resident,
            face_materials[source_face],
            face_render_nodes[source_face],
        );
        overlay.groups.entry(key).or_default().push(RenderBatchMember {
            face_index: leaf.source_face,
            leaf_id: leaf.id,
            node_index: face_nodes[source_face],
            edge_lods,
            permutation_index: resident.perm_index.min(5) as u8,
            vertex_lods,
        });
    }
    overlay.groups.retain(|_, members| !members.is_empty());
    Ok(())
}

fn compose_fixed_observer_vertex_lods(
    leaf_index: usize,
    leaf: ScreenMeshLeafTopology,
    mut vertex_lods: [u32; 3],
    fixed_leaves: Option<&[bool]>,
    baseline_vertex_lods: &[[u32; 3]],
) -> Result<[u32; 3], ScreenLeafLodError> {
    if !fixed_leaves
        .and_then(|fixed| fixed.get(leaf_index))
        .copied()
        .unwrap_or(false)
    {
        return Ok(vertex_lods);
    }
    if leaf.id != ScreenPatchLeafId::ROOT {
        return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
    }
    let baseline = baseline_vertex_lods.get(leaf.source_face as usize).ok_or(
        ScreenLeafLodError::MissingFaceMetadata {
            leaf_index,
            source_face: leaf.source_face,
        },
    )?;
    for corner in 0..3 {
        vertex_lods[corner] = vertex_lods[corner].max(baseline[corner]);
    }
    Ok(vertex_lods)
}

/// Decode a visible request or choose the bounded offscreen standby topology.
/// The render GPU independently performs a current-pose conservative cull, so
/// the standby remains drawable if asynchronous classification lags by a frame.
pub fn requested_face_lod(
    face_lods: &[f32],
    face_index: usize,
    standby: ResidentLod,
) -> ResidentLod {
    ResidentLod::from_payload_topology(face_lods, face_index).unwrap_or(standby)
}

/// Compatibility/default edge-resolution ratio inside one resident source
/// triangle. The live renderer may explicitly select another validated
/// [`FaceLodGrading`] policy.
pub const MAX_FACE_EDGE_LOD_RATIO: u32 = 2;

/// Supported within-face resident-LOD grading policies.
///
/// Shared-edge equality is the crack-free invariant. This policy separately
/// bounds anisotropy and the promotion halo, and is deliberately restricted
/// to the two measured power-of-two ratios represented in the runtime atlas.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FaceLodGrading {
    #[default]
    TwoToOne,
    FourToOne,
}

impl FaceLodGrading {
    pub const fn from_ratio(ratio: u32) -> Option<Self> {
        match ratio {
            2 => Some(Self::TwoToOne),
            4 => Some(Self::FourToOne),
            _ => None,
        }
    }

    pub const fn ratio(self) -> u32 {
        match self {
            Self::TwoToOne => 2,
            Self::FourToOne => 4,
        }
    }
}

/// Reusable work storage for sparse resident-LOD reconciliation.
///
/// Queue membership is reset as entries are consumed, while changed-face
/// flags are cleared through the sparse `changed_faces` list. Reusing this
/// storage avoids allocating and zeroing mesh-sized vectors after every
/// asynchronous classification.
#[derive(Default)]
pub struct ResidentLodBalanceScratch {
    queue: VecDeque<usize>,
    queued: Vec<bool>,
    changed: Vec<bool>,
    changed_faces: Vec<usize>,
    component_queue: VecDeque<usize>,
    component_seen: Vec<bool>,
    component_faces: Vec<usize>,
}

impl ResidentLodBalanceScratch {
    fn begin(&mut self, num_faces: usize) {
        self.queue.clear();
        self.queued.resize(num_faces, false);
        for face in self.changed_faces.drain(..) {
            if let Some(changed) = self.changed.get_mut(face) {
                *changed = false;
            }
        }
        self.changed.resize(num_faces, false);
    }

    fn enqueue(&mut self, face: usize) {
        if !self.queued[face] {
            self.queue.push_back(face);
            self.queued[face] = true;
        }
    }

    fn mark_changed(&mut self, face: usize) {
        if !self.changed[face] {
            self.changed[face] = true;
            self.changed_faces.push(face);
        }
    }

    fn begin_components(&mut self, num_faces: usize) {
        self.component_queue.clear();
        for face in self.component_faces.drain(..) {
            if let Some(seen) = self.component_seen.get_mut(face) {
                *seen = false;
            }
        }
        self.component_seen.resize(num_faces, false);
    }
}

/// Admit one complete or sparse GPU classification into retained visibility
/// and raw source-face requests. Crack-free reconciliation is the separately
/// measured next phase.
///
/// `face_indices = None` denotes an exact `num_faces` snapshot. Sparse indices
/// must be strictly increasing, matching the canonical delta encoder. The
/// shape is validated before any retained vector changes, so malformed backend
/// output cannot leave visibility, requests, and resident topology at
/// different revisions.
pub fn admit_face_lod_publication(
    face_lods: &[f32],
    face_indices: Option<&[u32]>,
    num_faces: usize,
    standby: ResidentLod,
    previous_culled_faces: usize,
    buffers: FaceLodAdmissionBuffers<'_>,
) -> Result<FaceLodAdmissionResult, FaceLodPublicationError> {
    let records = face_indices.map_or(num_faces, <[u32]>::len);
    let expected_floats = records
        .checked_mul(FACE_LOD_STRIDE)
        .ok_or(FaceLodPublicationError::PayloadLength)?;
    if face_lods.len() != expected_floats {
        return Err(FaceLodPublicationError::PayloadLength);
    }

    admit_face_lod_classification_publication(
        records,
        face_indices,
        num_faces,
        previous_culled_faces,
        buffers,
        |record_index| FaceLodClassification {
            requested: requested_face_lod(face_lods, record_index, standby),
            visible: face_is_visible(face_lods, record_index),
        },
    )
}

/// Admit already-decoded backend classifications without constructing the
/// historical six-float transport payload.
///
/// `records` must cover every face for a full snapshot, or exactly the sparse
/// index count. Structural validation completes before the callback is first
/// invoked and before retained state changes. Producing backends must validate
/// their own binary/texture ABI before calling this semantic boundary.
pub fn admit_face_lod_classification_publication(
    records: usize,
    face_indices: Option<&[u32]>,
    num_faces: usize,
    previous_culled_faces: usize,
    buffers: FaceLodAdmissionBuffers<'_>,
    mut classification_at: impl FnMut(usize) -> FaceLodClassification,
) -> Result<FaceLodAdmissionResult, FaceLodPublicationError> {
    let full_snapshot = face_indices.is_none();
    let expected_records = face_indices.map_or(num_faces, <[u32]>::len);
    if records != expected_records {
        return Err(FaceLodPublicationError::PayloadLength);
    }
    if let Some(indices) = face_indices {
        let mut previous = None;
        for &face in indices {
            let face = face as usize;
            if face >= num_faces {
                return Err(FaceLodPublicationError::FaceIndexOutOfBounds);
            }
            if previous.is_some_and(|previous| face <= previous) {
                return Err(FaceLodPublicationError::FaceIndicesNotIncreasing);
            }
            previous = Some(face);
        }
    }

    buffers.requested.resize(num_faces, None);
    buffers.resident.resize(num_faces, None);
    let visibility_shape_changed = buffers.visible.len() != num_faces;
    if visibility_shape_changed {
        buffers.visible.clear();
        buffers.visible.resize(num_faces, false);
    }
    buffers.dirty_faces.clear();

    let mut culled_faces = if full_snapshot {
        0
    } else if visibility_shape_changed {
        num_faces
    } else {
        previous_culled_faces.min(num_faces)
    };
    let mut changed_records = 0;
    for record_index in 0..records {
        let face = face_indices.map_or(record_index, |indices| indices[record_index] as usize);
        let classification = classification_at(record_index);
        let visible = classification.visible;
        if full_snapshot {
            buffers.visible[face] = visible;
            culled_faces += usize::from(!visible);
        } else {
            let previously_visible = buffers.visible[face];
            if previously_visible != visible {
                if visible {
                    culled_faces = culled_faces.saturating_sub(1);
                } else {
                    culled_faces = culled_faces.saturating_add(1).min(num_faces);
                }
                buffers.visible[face] = visible;
            }
        }

        let requested = classification.requested;
        let request_changed = buffers.requested[face] != Some(requested);
        buffers.requested[face] = Some(requested);
        if request_changed || buffers.resident[face].is_none() {
            changed_records += 1;
            buffers.dirty_faces.push(face);
        }
    }

    Ok(FaceLodAdmissionResult {
        full_snapshot,
        records,
        changed_records,
        culled_faces,
    })
}

/// Reconcile and grade topology retained across asynchronous visibility results.
///
/// The worker's GPU pass guarantees agreement when both neighboring faces are
/// visible in that classification. An invisible face intentionally keeps its
/// previous resident topology, however, so a visible neighbor may otherwise
/// change one side of their shared edge alone. Taking the maximum on every
/// resident shared edge restores the same crack-free invariant. A second 2:1
/// face-balance constraint prevents extreme anisotropic atlas patches such as
/// `[1, 1, 128]`. Promotions are iterated to a fixed point because grading one
/// face can raise an edge shared with its neighbor. No requested resolution is
/// ever reduced, and a peak decays by one LOD level per neighboring face.
pub fn balance_resident_lods(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
) -> usize {
    let num_faces = residents.len().min(topology.num_faces as usize);
    let mut scratch = ResidentLodBalanceScratch::default();
    balance_resident_lods_seeded::<MAX_FACE_EDGE_LOD_RATIO>(
        residents,
        topology,
        0..num_faces,
        &mut scratch,
    )
}

/// Reconcile resident shared edges and grade each face to an explicit maximum
/// edge-resolution ratio.
///
/// The compatibility wrapper remains fixed at [`MAX_FACE_EDGE_LOD_RATIO`].
/// Runtime policy selection and offline probes both dispatch through this same
/// fixed-point implementation without copying its topology semantics.
pub fn balance_resident_lods_with_ratio<const MAX_FACE_EDGE_RATIO: u32>(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
) -> usize {
    assert!(
        MAX_FACE_EDGE_RATIO.is_power_of_two(),
        "LOD ratio must be a positive power of two",
    );
    let num_faces = residents.len().min(topology.num_faces as usize);
    let mut scratch = ResidentLodBalanceScratch::default();
    balance_resident_lods_seeded::<MAX_FACE_EDGE_RATIO>(
        residents,
        topology,
        0..num_faces,
        &mut scratch,
    )
}

/// Reconcile a complete resident snapshot using a validated runtime policy.
pub fn balance_resident_lods_with_grading(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    grading: FaceLodGrading,
) -> usize {
    match grading {
        FaceLodGrading::TwoToOne => balance_resident_lods_with_ratio::<2>(residents, topology),
        FaceLodGrading::FourToOne => balance_resident_lods_with_ratio::<4>(residents, topology),
    }
}

/// Restore crack-free, default-graded topology from a sparse set of changed faces.
///
/// The resident topology must already satisfy the invariants before the listed
/// faces are changed. Promotions are propagated through twin adjacency until
/// the same least fixed point as [`balance_resident_lods`] is reached.
pub fn balance_resident_lods_from_faces(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    dirty_faces: &[usize],
    scratch: &mut ResidentLodBalanceScratch,
) -> usize {
    balance_resident_lods_seeded::<MAX_FACE_EDGE_LOD_RATIO>(
        residents,
        topology,
        dirty_faces.iter().copied(),
        scratch,
    )
}

/// Restore crack-free topology from a sparse frontier using a validated
/// runtime grading policy.
pub fn balance_resident_lods_from_faces_with_grading(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    dirty_faces: &[usize],
    scratch: &mut ResidentLodBalanceScratch,
    grading: FaceLodGrading,
) -> usize {
    match grading {
        FaceLodGrading::TwoToOne => balance_resident_lods_seeded::<2>(
            residents,
            topology,
            dirty_faces.iter().copied(),
            scratch,
        ),
        FaceLodGrading::FourToOne => balance_resident_lods_seeded::<4>(
            residents,
            topology,
            dirty_faces.iter().copied(),
            scratch,
        ),
    }
}

/// Rebuild the connected components touched by changed raw requests, then
/// reconcile them to the least crack-free fixed point.
///
/// Resident values are promoted closure, not source requests. Starting a
/// demotion from those promoted values makes old peaks ratchet forever: an
/// unchanged neighbor immediately promotes the changed face back. Keeping the
/// raw requests separately and resetting the affected topological components
/// before the monotone pass permits both increases and decreases without a
/// full-scene scan.
pub fn reconcile_resident_lods_from_requests_with_grading(
    requests: &[Option<ResidentLod>],
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    dirty_faces: &[usize],
    scratch: &mut ResidentLodBalanceScratch,
    grading: FaceLodGrading,
) -> usize {
    let num_faces = requests
        .len()
        .min(residents.len())
        .min(topology.num_faces as usize);
    scratch.begin_components(num_faces);

    for &seed in dirty_faces {
        if seed >= num_faces || scratch.component_seen[seed] {
            continue;
        }
        scratch.component_seen[seed] = true;
        scratch.component_queue.push_back(seed);
        while let Some(face) = scratch.component_queue.pop_front() {
            scratch.component_faces.push(face);
            for half_edge in topology.face_half_edges(face as u32) {
                let Some(twin) = topology.twin(half_edge) else {
                    continue;
                };
                let neighbor = topology.half_edges[twin as usize].face as usize;
                if neighbor < num_faces && !scratch.component_seen[neighbor] {
                    scratch.component_seen[neighbor] = true;
                    scratch.component_queue.push_back(neighbor);
                }
            }
        }
    }

    let component_faces = std::mem::take(&mut scratch.component_faces);
    for &face in &component_faces {
        residents[face] = requests[face];
    }
    let corrections = match grading {
        FaceLodGrading::TwoToOne => balance_resident_lods_seeded::<2>(
            residents,
            topology,
            component_faces.iter().copied(),
            scratch,
        ),
        FaceLodGrading::FourToOne => balance_resident_lods_seeded::<4>(
            residents,
            topology,
            component_faces.iter().copied(),
            scratch,
        ),
    };
    scratch.component_faces = component_faces;
    corrections
}

fn balance_resident_lods_seeded<const MAX_FACE_EDGE_RATIO: u32>(
    residents: &mut [Option<ResidentLod>],
    topology: &HalfEdgeMesh,
    dirty_faces: impl IntoIterator<Item = usize>,
    scratch: &mut ResidentLodBalanceScratch,
) -> usize {
    let num_faces = residents.len().min(topology.num_faces as usize);
    scratch.begin(num_faces);

    // Both constraints are monotone promotions over a small power-of-two
    // lattice: twins take their maximum, and a face's low edges rise to half
    // its maximum. A work queue therefore reaches the same least fixed point
    // while revisiting only faces touched by a promotion. The former global
    // fixed-point loop rescanned every half-edge and every face per wave; a
    // localized high-LOD chess piece could make a 95k-face scene stall for
    // nearly a second after each camera movement.
    for face in dirty_faces {
        if face < num_faces && residents[face].is_some() {
            scratch.enqueue(face);
        }
    }

    while let Some(face) = scratch.queue.pop_front() {
        scratch.queued[face] = false;
        let Some(resident) = residents[face] else { continue };
        let mut face_lods = resident.edge_lods();

        let mut face_changed = false;
        for half_edge in topology.face_half_edges(face as u32) {
            let Some(twin) = topology.twin(half_edge) else { continue };
            let twin_face = topology.half_edges[twin as usize].face as usize;
            if twin_face >= num_faces || residents[twin_face].is_none() {
                continue;
            }
            let edge_index = (half_edge as usize % 3 + 2) % 3;
            let twin_edge_index = (twin as usize % 3 + 2) % 3;
            if twin_face == face {
                let shared = face_lods[edge_index].max(face_lods[twin_edge_index]);
                if face_lods[edge_index] != shared || face_lods[twin_edge_index] != shared {
                    face_lods[edge_index] = shared;
                    face_lods[twin_edge_index] = shared;
                    face_changed = true;
                }
                continue;
            }

            let mut twin_lods = residents[twin_face].unwrap().edge_lods();
            let face_lod = face_lods[edge_index];
            let twin_lod = twin_lods[twin_edge_index];
            let shared = face_lod.max(twin_lod);
            if face_lod != shared {
                face_lods[edge_index] = shared;
                face_changed = true;
            }
            if twin_lod != shared {
                twin_lods[twin_edge_index] = shared;
                residents[twin_face] = Some(ResidentLod::from_edge_lods(twin_lods));
                scratch.mark_changed(twin_face);
                scratch.enqueue(twin_face);
            }
        }

        let maximum = *face_lods.iter().max().unwrap_or(&1);
        let minimum_allowed = (maximum / MAX_FACE_EDGE_RATIO).max(1);
        for edge_lod in &mut face_lods {
            if *edge_lod < minimum_allowed {
                *edge_lod = minimum_allowed;
                face_changed = true;
            }
        }

        // Promotions made by face grading must be copied to twins. Requeueing
        // the face is enough; any promoted twin then queues its own face.
        if face_changed {
            let reconciled = ResidentLod::from_edge_lods(face_lods);
            if residents[face] != Some(reconciled) {
                residents[face] = Some(reconciled);
                scratch.mark_changed(face);
            }
            scratch.enqueue(face);
        }
    }

    scratch.changed_faces.len()
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
    use crate::patch::QBPatchDomain;
    use crate::screen_leaf_lod::{
        ScreenMeshLeafFrontier, ScreenMeshLeafTopology, ScreenMeshTopologyCache,
    };

    fn lod_record(edge_lods: [f32; 3], visible: bool) -> [f32; FACE_LOD_STRIDE] {
        [
            edge_lods[0],
            edge_lods[1],
            edge_lods[2],
            0.0,
            1.0,
            if visible { 0.0 } else { -1.0 },
        ]
    }

    #[test]
    fn face_lod_publication_applies_full_sparse_and_demotion() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let standby = ResidentLod::uniform(2);
        let mut requested = Vec::new();
        let mut resident = Vec::new();
        let mut visible = Vec::new();
        let mut dirty_faces = Vec::new();
        let mut balance_scratch = ResidentLodBalanceScratch::default();
        let mut full = Vec::new();
        full.extend_from_slice(&lod_record([2.0; 3], true));
        full.extend_from_slice(&lod_record([2.0; 3], false));

        let initial = admit_face_lod_publication(
            &full,
            None,
            2,
            standby,
            0,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
        )
        .unwrap();
        let initial_corrections = reconcile_resident_lods_from_requests_with_grading(
            &requested,
            &mut resident,
            &topology,
            &dirty_faces,
            &mut balance_scratch,
            FaceLodGrading::TwoToOne,
        );
        assert!(initial.full_snapshot);
        assert_eq!(initial.records, 2);
        assert_eq!(initial.changed_records, 2);
        assert_eq!(initial.culled_faces, 1);
        assert_eq!(initial_corrections, 0);
        assert_eq!(resident, vec![Some(standby); 2]);

        let raised = lod_record([8.0; 3], true);
        let raised = admit_face_lod_publication(
            &raised,
            Some(&[1]),
            2,
            standby,
            initial.culled_faces,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
        )
        .unwrap();
        let raised_corrections = reconcile_resident_lods_from_requests_with_grading(
            &requested,
            &mut resident,
            &topology,
            &dirty_faces,
            &mut balance_scratch,
            FaceLodGrading::TwoToOne,
        );
        assert!(!raised.full_snapshot);
        assert!(raised.topology_changed());
        assert!(raised_corrections > 0);
        assert_eq!(raised.culled_faces, 0);
        assert_eq!(requested[1], Some(ResidentLod::uniform(8)));
        assert_eq!(resident[1], Some(ResidentLod::uniform(8)));

        let demoted = admit_face_lod_publication(
            &full,
            None,
            2,
            standby,
            raised.culled_faces,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
        )
        .unwrap();
        reconcile_resident_lods_from_requests_with_grading(
            &requested,
            &mut resident,
            &topology,
            &dirty_faces,
            &mut balance_scratch,
            FaceLodGrading::TwoToOne,
        );
        assert!(demoted.topology_changed());
        assert_eq!(resident, vec![Some(standby); 2]);
        assert_eq!(demoted.culled_faces, 1);
    }

    #[test]
    fn decoded_face_lod_publication_shares_the_atomic_admission_boundary() {
        let classifications = [
            FaceLodClassification {
                requested: ResidentLod::from_edge_lods([2, 4, 8]),
                visible: true,
            },
            FaceLodClassification {
                requested: ResidentLod::uniform(16),
                visible: false,
            },
        ];
        let mut requested = Vec::new();
        let mut resident = Vec::new();
        let mut visible = Vec::new();
        let mut dirty_faces = Vec::new();
        let publication = admit_face_lod_classification_publication(
            classifications.len(),
            None,
            classifications.len(),
            0,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
            |index| classifications[index],
        )
        .unwrap();
        assert_eq!(publication.changed_records, 2);
        assert_eq!(publication.culled_faces, 1);
        assert_eq!(
            requested,
            classifications.map(|classification| Some(classification.requested)),
        );
        assert_eq!(visible, vec![true, false]);

        let retained_requested = requested.clone();
        let retained_resident = resident.clone();
        let retained_visible = visible.clone();
        let retained_dirty = dirty_faces.clone();
        let mut callback_calls = 0;
        let error = admit_face_lod_classification_publication(
            1,
            None,
            classifications.len(),
            publication.culled_faces,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
            |_| {
                callback_calls += 1;
                classifications[0]
            },
        )
        .unwrap_err();
        assert_eq!(error, FaceLodPublicationError::PayloadLength);
        assert_eq!(callback_calls, 0);
        assert_eq!(requested, retained_requested);
        assert_eq!(resident, retained_resident);
        assert_eq!(visible, retained_visible);
        assert_eq!(dirty_faces, retained_dirty);
    }

    #[test]
    fn malformed_face_lod_publication_is_atomic() {
        let standby = ResidentLod::uniform(2);
        let original = vec![Some(standby)];
        let mut requested = original.clone();
        let mut resident = original.clone();
        let mut visible = vec![true];
        let mut dirty_faces = vec![91];
        let mut duplicate_payload = Vec::new();
        duplicate_payload.extend_from_slice(&lod_record([4.0; 3], true));
        duplicate_payload.extend_from_slice(&lod_record([8.0; 3], false));

        let error = admit_face_lod_publication(
            &duplicate_payload,
            Some(&[0, 0]),
            1,
            standby,
            0,
            FaceLodAdmissionBuffers {
                requested: &mut requested,
                resident: &mut resident,
                visible: &mut visible,
                dirty_faces: &mut dirty_faces,
            },
        )
        .unwrap_err();
        assert_eq!(error, FaceLodPublicationError::FaceIndicesNotIncreasing);
        assert_eq!(requested, original);
        assert_eq!(resident, original);
        assert_eq!(visible, vec![true]);
        assert_eq!(dirty_faces, vec![91]);
    }

    #[test]
    fn face_lod_delta_emits_full_then_only_changed_records() {
        let initial = [
            2.0, 2.0, 2.0, 0.0, 1.0, 3.0,
            4.0, 4.0, 8.0, 2.0, -1.0, 7.0,
        ];
        let mut previous = Vec::new();
        let mut faces = Vec::new();
        let mut lods = Vec::new();

        assert!(encode_face_lod_delta(
            &initial, &mut previous, &mut faces, &mut lods,
        ));
        assert!(faces.is_empty());
        assert_eq!(lods, initial);

        assert!(!encode_face_lod_delta(
            &initial, &mut previous, &mut faces, &mut lods,
        ));
        assert!(faces.is_empty());
        assert!(lods.is_empty());

        let mut changed = initial;
        changed[6..12].copy_from_slice(&[8.0, 8.0, 8.0, 0.0, 1.0, 9.0]);
        assert!(!encode_face_lod_delta(
            &changed, &mut previous, &mut faces, &mut lods,
        ));
        assert_eq!(faces, [1]);
        assert_eq!(lods, changed[6..12]);
        assert_eq!(previous, changed);
    }

    #[test]
    fn sequenced_face_lod_delta_advances_across_empty_and_sparse_publications() {
        let initial = [
            2.0, 2.0, 2.0, 0.0, 1.0, 3.0,
            4.0, 4.0, 8.0, 2.0, -1.0, 7.0,
        ];
        let mut encoder = FaceLodDeltaEncoder::default();
        let mut cursor = FaceLodDeltaCursor::default();

        let full = encoder.encode(&initial).unwrap();
        assert!(full.full_snapshot);
        assert_eq!(full.sequence, FaceLodDeltaSequence {
            epoch: 1, base_revision: 0, revision: 1,
        });
        assert!(cursor.accept(full.sequence, full.full_snapshot).unwrap());

        let empty = encoder.encode(&initial).unwrap();
        assert!(!empty.full_snapshot);
        assert!(empty.changed_faces.is_empty());
        assert!(empty.lods.is_empty());
        assert_eq!(empty.sequence.base_revision, 1);
        assert!(!cursor.accept(empty.sequence, empty.full_snapshot).unwrap());

        let mut changed = initial;
        changed[6..12].copy_from_slice(&[8.0, 8.0, 8.0, 0.0, 1.0, 9.0]);
        let sparse = encoder.encode(&changed).unwrap();
        assert_eq!(sparse.changed_faces, [1]);
        assert_eq!(sparse.lods, &changed[6..12]);
        assert_eq!(sparse.sequence.base_revision, 2);
        assert!(!cursor.accept(sparse.sequence, sparse.full_snapshot).unwrap());
        assert_eq!(cursor.sequence(), Some((1, 3)));
    }

    #[test]
    fn face_lod_delta_cursor_rejects_gaps_and_recovers_from_explicit_reset() {
        let mut cursor = FaceLodDeltaCursor::default();
        assert_eq!(
            cursor.accept(
                FaceLodDeltaSequence { epoch: 4, base_revision: 1, revision: 2 },
                false,
            ),
            Err(FaceLodDeltaError::MissingBase),
        );
        assert!(cursor.accept(
            FaceLodDeltaSequence { epoch: 4, base_revision: 0, revision: 1 },
            true,
        ).unwrap());
        assert_eq!(
            cursor.accept(
                FaceLodDeltaSequence { epoch: 4, base_revision: 2, revision: 3 },
                false,
            ),
            Err(FaceLodDeltaError::MissingBase),
        );
        assert!(cursor.accept(
            FaceLodDeltaSequence { epoch: 5, base_revision: 0, revision: 1 },
            true,
        ).unwrap());
        assert_eq!(cursor.sequence(), Some((5, 1)));
    }

    #[test]
    fn face_lod_delta_encoder_rejects_partial_records() {
        let error = FaceLodDeltaEncoder::default().encode(&[1.0; 5]).unwrap_err();
        assert_eq!(error, FaceLodDeltaError::InvalidPayload);
    }

    fn balance_resident_lods_reference(
        residents: &mut [Option<ResidentLod>],
        topology: &HalfEdgeMesh,
    ) -> usize {
        let num_faces = residents.len().min(topology.num_faces as usize);
        let mut face_edges: Vec<Option<[u32; 3]>> = residents[..num_faces]
            .iter()
            .map(|resident| resident.map(ResidentLod::edge_lods))
            .collect();

        loop {
            let mut pass_changed = false;
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
                pass_changed |= face_lods[edge_index] != shared
                    || twin_lods[twin_edge_index] != shared;
                face_edges[face].as_mut().unwrap()[edge_index] = shared;
                face_edges[twin_face].as_mut().unwrap()[twin_edge_index] = shared;
            }

            for edge_lods in face_edges.iter_mut().flatten() {
                let maximum = *edge_lods.iter().max().unwrap_or(&1);
                let minimum_allowed = (maximum / MAX_FACE_EDGE_LOD_RATIO).max(1);
                for edge_lod in edge_lods {
                    if *edge_lod < minimum_allowed {
                        *edge_lod = minimum_allowed;
                        pass_changed = true;
                    }
                }
            }

            if !pass_changed {
                break;
            }
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
    fn invisible_payload_selects_bounded_standby_topology() {
        let visible = [16.0, 32.0, 64.0, 4.0, 1.0, 17.0];
        let invisible = [2.0, 2.0, 2.0, 0.0, 1.0, -1.0];
        let resident = ResidentLod::from_visible_payload(&visible, 0).unwrap();

        assert_eq!(resident.canonical, [16, 32, 64]);
        assert_eq!(resident.perm_index, 4);
        assert_eq!(ResidentLod::from_visible_payload(&invisible, 0), None);
        assert_eq!(
            ResidentLod::from_payload_topology(&invisible, 0),
            Some(ResidentLod::uniform(2)),
        );
        assert_eq!(
            requested_face_lod(&invisible, 0, ResidentLod::uniform(2)),
            ResidentLod::uniform(2),
        );
    }

    #[test]
    fn resident_groups_have_stable_backend_neutral_membership() {
        let residents = [
            Some(ResidentLod::from_edge_lods([2, 4, 8])),
            Some(ResidentLod::from_edge_lods([8, 2, 4])),
            Some(ResidentLod::uniform(4)),
        ];
        let groups = group_resident_faces(
            &residents,
            &[[8, 8, 8], [8, 8, 8], [4, 4, 4]],
            &[3, 3, 7],
            &[11, 11, 12],
            &[11, 11, 12],
            ResidentLod::uniform(1),
        );

        assert_eq!(groups.len(), 2);
        let anisotropic = groups.iter()
            .find(|(key, _)| key.lod == [2, 4, 8])
            .unwrap();
        assert_eq!(anisotropic.0.material_index, 3);
        assert_eq!(anisotropic.0.render_node_index, 11);
        assert_eq!(anisotropic.1[0].node_index, 11);
        assert_eq!(anisotropic.1.iter().map(|member| member.face_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_ne!(anisotropic.1[0].permutation_index, anisotropic.1[1].permutation_index);
    }

    #[test]
    fn all_root_adaptive_grouping_is_exactly_legacy_grouping() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let residents = [
            Some(ResidentLod::from_edge_lods([2, 4, 8])),
            Some(ResidentLod::from_edge_lods([8, 4, 2])),
        ];
        let initial = ResidentLod::uniform(1);
        let mut vertex_max = Vec::new();
        let mut face_vertex_lods = Vec::new();
        rebuild_resident_vertex_lods(
            &residents,
            &topology,
            initial,
            &mut vertex_max,
            &mut face_vertex_lods,
        );
        let face_materials = [3, 7];
        let face_nodes = [11, 13];
        let face_render_nodes = [17, 19];
        let legacy = group_resident_faces(
            &residents,
            &face_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            initial,
        );

        let leaves = [0, 1].map(|source_face| ScreenMeshLeafTopology {
            source_face,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology_cache).unwrap();
        let adaptive = group_resident_screen_leaves(
            &frontier,
            &residents.map(|resident| resident.unwrap().edge_lods()),
            &face_materials,
            &face_nodes,
            &face_render_nodes,
        )
        .unwrap();

        assert_eq!(adaptive, legacy);
    }

    #[test]
    fn adaptive_leaves_keep_source_identity_inside_draw_buckets() {
        let first_id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let second_id = ScreenPatchLeafId::ROOT.child(1).unwrap();
        let leaves = [
            ScreenMeshLeafTopology {
                source_face: 1,
                id: first_id,
                domain: first_id.domain().unwrap(),
            },
            ScreenMeshLeafTopology {
                source_face: 1,
                id: second_id,
                domain: second_id.domain().unwrap(),
            },
        ];
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology_cache).unwrap();
        let groups = group_resident_screen_leaves(
            &frontier,
            &[[2, 4, 8]; 2],
            &[3, 7],
            &[11, 13],
            &[17, 19],
        )
        .unwrap();

        assert_eq!(groups.len(), 1);
        let (key, members) = groups.first_key_value().unwrap();
        assert_eq!(key.material_index, 7);
        assert_eq!(key.render_node_index, 19);
        assert_eq!(members.len(), 2);
        assert!(members.iter().all(|member| member.face_index == 1));
        assert!(members.iter().all(|member| member.node_index == 13));
        assert_eq!(members[0].patch_id(), (1, first_id));
        assert_eq!(members[1].patch_id(), (1, second_id));
    }

    fn compose_root_groups_with_overlay(
        mut roots: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        overlay: &AdaptiveRenderOverlay,
    ) -> BTreeMap<RenderBatchKey, Vec<RenderBatchMember>> {
        for members in roots.values_mut() {
            members.retain(|member| {
                overlay
                    .suppressed_faces
                    .binary_search(&member.face_index)
                    .is_err()
            });
        }
        roots.retain(|_, members| !members.is_empty());
        for (&key, replacements) in &overlay.groups {
            roots
                .entry(key)
                .or_default()
                .extend_from_slice(replacements);
        }
        for members in roots.values_mut() {
            members.sort_unstable_by_key(|member| member.patch_id());
        }
        roots
    }

    #[test]
    fn adaptive_overlay_composes_to_the_complete_frontier_grouping() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let baseline_residents = vec![Some(ResidentLod::uniform(2)); 2];
        let mut baseline_vertex_lods = Vec::new();
        let mut vertex_scratch = Vec::new();
        rebuild_resident_vertex_lods(
            &baseline_residents,
            &topology,
            ResidentLod::uniform(1),
            &mut vertex_scratch,
            &mut baseline_vertex_lods,
        );
        let face_materials = [3, 7];
        let face_nodes = [11, 13];
        let face_render_nodes = [17, 19];
        let roots = group_resident_faces(
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
        );

        let mut leaves = (0..4)
            .map(|child| {
                let id = ScreenPatchLeafId::ROOT.child(child).unwrap();
                ScreenMeshLeafTopology {
                    source_face: 0,
                    id,
                    domain: id.domain().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        leaves.push(ScreenMeshLeafTopology {
            source_face: 1,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology_cache).unwrap();
        let requested = vec![[1; 3], [1; 3], [1; 3], [1; 3], [2; 3]];
        let reconciled = frontier.reconcile_lods(&requested, 2, 64).unwrap();
        let complete = group_resident_screen_leaves(
            &frontier,
            &reconciled.resident,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
        )
        .unwrap();

        let mut overlay = AdaptiveRenderOverlay::default();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        group_resident_screen_overlay_into(
            &frontier,
            &reconciled.resident,
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut overlay,
        )
        .unwrap();

        assert!(overlay.suppressed_faces.binary_search(&0).is_ok());
        assert_eq!(compose_root_groups_with_overlay(roots, &overlay), complete);
    }

    #[test]
    fn component_overlay_composes_without_scanning_unaffected_components() {
        let topology = HalfEdgeMesh::from_triangles(
            8,
            &[[0, 1, 2], [2, 3, 4], [5, 6, 7]],
        );
        let baseline_residents = vec![Some(ResidentLod::uniform(2)); 3];
        let mut baseline_vertex_lods = Vec::new();
        let mut vertex_scratch = Vec::new();
        rebuild_resident_vertex_lods(
            &baseline_residents,
            &topology,
            ResidentLod::uniform(1),
            &mut vertex_scratch,
            &mut baseline_vertex_lods,
        );
        let face_materials = [3, 7, 11];
        let face_nodes = [13, 17, 19];
        let face_render_nodes = [23, 29, 31];
        let roots = group_resident_faces(
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
        );

        let mut component_leaves = (0..4)
            .map(|child| {
                let id = ScreenPatchLeafId::ROOT.child(child).unwrap();
                ScreenMeshLeafTopology {
                    source_face: 0,
                    id,
                    domain: id.domain().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        component_leaves.push(ScreenMeshLeafTopology {
            source_face: 1,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let mut complete_leaves = component_leaves.clone();
        complete_leaves.push(ScreenMeshLeafTopology {
            source_face: 2,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let component_frontier =
            ScreenMeshLeafFrontier::build(&component_leaves, &topology_cache).unwrap();
        let complete_frontier =
            ScreenMeshLeafFrontier::build(&complete_leaves, &topology_cache).unwrap();
        let component_requested = vec![[1; 3], [1; 3], [1; 3], [1; 3], [2; 3]];
        let mut complete_requested = component_requested.clone();
        complete_requested.push([2; 3]);
        let component_resident = component_frontier
            .reconcile_lods(&component_requested, 2, 64)
            .unwrap();
        let complete_resident = complete_frontier
            .reconcile_lods(&complete_requested, 2, 64)
            .unwrap();
        let complete = group_resident_screen_leaves(
            &complete_frontier,
            &complete_resident.resident,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
        )
        .unwrap();

        let mut overlay = AdaptiveRenderOverlay::default();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        group_resident_screen_component_overlay_into(
            &component_frontier,
            &component_resident.resident,
            &[0, 1],
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut overlay,
        )
        .unwrap();

        assert_eq!(overlay.face_leaf_counts.len(), 2);
        assert!(overlay.suppressed_faces.binary_search(&0).is_ok());
        assert!(overlay.suppressed_faces.binary_search(&2).is_err());
        assert_eq!(compose_root_groups_with_overlay(roots, &overlay), complete);
    }

    #[test]
    fn neighborhood_overlay_composes_fixed_observer_corner_baselines() {
        // Face 0 is replaced, face 1 observes its shared vertex, and retained
        // face 2 contributes a larger baseline maximum at a different corner
        // of face 1. A local rebuild alone cannot see that outer contribution.
        let topology = HalfEdgeMesh::from_triangles(
            7,
            &[[0, 1, 2], [2, 3, 4], [4, 5, 6]],
        );
        let baseline_residents = [
            Some(ResidentLod::uniform(1)),
            Some(ResidentLod::uniform(1)),
            Some(ResidentLod::uniform(8)),
        ];
        let mut baseline_vertex_lods = Vec::new();
        let mut vertex_scratch = Vec::new();
        rebuild_resident_vertex_lods(
            &baseline_residents,
            &topology,
            ResidentLod::uniform(1),
            &mut vertex_scratch,
            &mut baseline_vertex_lods,
        );
        assert!(baseline_vertex_lods[1].contains(&8));
        let face_materials = [3, 7, 11];
        let face_nodes = [13, 17, 19];
        let face_render_nodes = [23, 29, 31];
        let roots = group_resident_faces(
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
        );

        let mut neighborhood_leaves = (0..4)
            .map(|child| {
                let id = ScreenPatchLeafId::ROOT.child(child).unwrap();
                ScreenMeshLeafTopology {
                    source_face: 0,
                    id,
                    domain: id.domain().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        neighborhood_leaves.push(ScreenMeshLeafTopology {
            source_face: 1,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let mut complete_leaves = neighborhood_leaves.clone();
        complete_leaves.push(ScreenMeshLeafTopology {
            source_face: 2,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let neighborhood_frontier =
            ScreenMeshLeafFrontier::build(&neighborhood_leaves, &topology_cache).unwrap();
        let complete_frontier =
            ScreenMeshLeafFrontier::build(&complete_leaves, &topology_cache).unwrap();
        let neighborhood_requested = vec![[1; 3]; 5];
        let mut complete_requested = neighborhood_requested.clone();
        complete_requested.push([8; 3]);
        let fixed_leaves = [false, false, false, false, true];
        let neighborhood_resident = neighborhood_frontier
            .reconcile_lods_with_fixed_boundary(
                &neighborhood_requested,
                &fixed_leaves,
                2,
                64,
            )
            .unwrap();
        let complete_resident = complete_frontier
            .reconcile_lods(&complete_requested, 2, 64)
            .unwrap();
        let complete = group_resident_screen_leaves(
            &complete_frontier,
            &complete_resident.resident,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
        )
        .unwrap();

        let mut raw_overlay = AdaptiveRenderOverlay::default();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        group_resident_screen_component_overlay_into(
            &neighborhood_frontier,
            &neighborhood_resident.resident,
            &[0, 1],
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut raw_overlay,
        )
        .unwrap();
        assert_ne!(
            compose_root_groups_with_overlay(roots.clone(), &raw_overlay),
            complete,
        );

        let mut neighborhood_overlay = AdaptiveRenderOverlay::default();
        group_resident_screen_neighborhood_overlay_into(
            &neighborhood_frontier,
            &neighborhood_resident.resident,
            &[0, 1],
            &fixed_leaves,
            &baseline_residents,
            &baseline_vertex_lods,
            &face_materials,
            &face_nodes,
            &face_render_nodes,
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut neighborhood_overlay,
        )
        .unwrap();
        assert!(neighborhood_overlay.suppressed_faces.contains(&0));
        assert!(neighborhood_overlay.suppressed_faces.contains(&1));
        assert_eq!(
            compose_root_groups_with_overlay(roots, &neighborhood_overlay),
            complete,
        );
    }

    #[test]
    fn sparse_overlay_work_matches_the_composed_scene_without_source_scan() {
        let baseline_residents = [
            Some(ResidentLod::uniform(2)),
            Some(ResidentLod::uniform(4)),
        ];
        let roots = group_resident_faces(
            &baseline_residents,
            &[[2; 3], [4; 3]],
            &[0; 2],
            &[0; 2],
            &[0; 2],
            ResidentLod::uniform(1),
        );
        let replacement = ResidentLod::uniform(1);
        let replacement_key = RenderBatchKey::from_resident(replacement, 0, 0);
        let mut overlay = AdaptiveRenderOverlay::default();
        overlay.suppressed_faces.push(1);
        overlay.groups.insert(
            replacement_key,
            [0, 1]
                .map(|child| {
                    let leaf_id = ScreenPatchLeafId::ROOT.child(child).unwrap();
                    RenderBatchMember {
                        face_index: 1,
                        leaf_id,
                        node_index: 0,
                        edge_lods: [1; 3],
                        permutation_index: 0,
                        vertex_lods: [1; 3],
                    }
                })
                .to_vec(),
        );
        let atlas = BTreeMap::from([([1; 3], 1), ([2; 3], 4), ([4; 3], 16)]);

        let work = measure_adaptive_render_work(
            &roots,
            &overlay,
            &baseline_residents,
            ResidentLod::uniform(1),
            &atlas,
        )
        .unwrap();

        assert_eq!(work.baseline_triangles, 20);
        assert_eq!(work.suppressed_root_triangles, 16);
        assert_eq!(work.overlay_triangles, 2);
        assert_eq!(work.composed_triangles, 6);
    }

    #[test]
    fn sparse_overlay_work_fails_closed_on_invalid_residency_or_identity() {
        let resident = ResidentLod::uniform(2);
        let roots = group_resident_faces(
            &[Some(resident)],
            &[[2; 3]],
            &[0],
            &[0],
            &[0],
            resident,
        );
        let mut overlay = AdaptiveRenderOverlay::default();
        overlay.suppressed_faces.extend([0, 0]);
        let atlas = BTreeMap::from([([2; 3], 4)]);

        assert_eq!(
            measure_adaptive_render_work(
                &roots,
                &overlay,
                &[Some(resident)],
                resident,
                &atlas,
            ),
            Err(AdaptiveRenderWorkError::SuppressedFacesNotIncreasing {
                source_face: 0,
            }),
        );

        overlay.suppressed_faces.truncate(1);
        assert_eq!(
            measure_adaptive_render_work(
                &roots,
                &overlay,
                &[Some(resident)],
                resident,
                &BTreeMap::new(),
            ),
            Err(AdaptiveRenderWorkError::MissingAtlasPatch { lod: [2; 3] }),
        );
    }

    #[test]
    fn component_overlay_rejects_an_incomplete_certificate_atomically() {
        let topology = HalfEdgeMesh::from_triangles(5, &[[0, 1, 2], [2, 3, 4]]);
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(
            &[ScreenMeshLeafTopology {
                source_face: 0,
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            }],
            &topology_cache,
        )
        .unwrap();
        let mut overlay = AdaptiveRenderOverlay::default();
        overlay.suppressed_faces.push(99);
        let before = overlay.suppressed_faces.clone();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        let error = group_resident_screen_component_overlay_into(
            &frontier,
            &[[1; 3]],
            &[0, 1],
            &[Some(ResidentLod::uniform(1)); 2],
            &[[1; 3]; 2],
            &[0; 2],
            &[0; 2],
            &[0; 2],
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut overlay,
        )
        .unwrap_err();

        assert_eq!(error, ScreenLeafLodError::MissingSourceFace { source_face: 1 });
        assert_eq!(overlay.suppressed_faces, before);
    }

    #[test]
    fn all_root_frontier_needs_no_adaptive_overlay() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let residents = vec![Some(ResidentLod::uniform(4)); 2];
        let mut baseline_vertex_lods = Vec::new();
        let mut vertex_scratch = Vec::new();
        rebuild_resident_vertex_lods(
            &residents,
            &topology,
            ResidentLod::uniform(1),
            &mut vertex_scratch,
            &mut baseline_vertex_lods,
        );
        let leaves = [0, 1].map(|source_face| ScreenMeshLeafTopology {
            source_face,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology_cache).unwrap();
        let mut overlay = AdaptiveRenderOverlay::default();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        group_resident_screen_overlay_into(
            &frontier,
            &[[4; 3]; 2],
            &residents,
            &baseline_vertex_lods,
            &[0; 2],
            &[0; 2],
            &[0; 2],
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut overlay,
        )
        .unwrap();

        assert!(overlay.suppressed_faces.is_empty());
        assert!(overlay.groups.is_empty());
    }

    #[test]
    fn adaptive_overlay_rejects_an_incomplete_frontier_atomically() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(
            &[ScreenMeshLeafTopology {
                source_face: 0,
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            }],
            &topology_cache,
        )
        .unwrap();
        let mut overlay = AdaptiveRenderOverlay::default();
        overlay.suppressed_faces.push(99);
        let before = overlay.suppressed_faces.clone();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        let error = group_resident_screen_overlay_into(
            &frontier,
            &[[1; 3]],
            &[Some(ResidentLod::uniform(1)); 2],
            &[[1; 3]; 2],
            &[0; 2],
            &[0; 2],
            &[0; 2],
            ResidentLod::uniform(1),
            &mut lod_scratch,
            &mut overlay,
        )
        .unwrap_err();

        assert_eq!(error, ScreenLeafLodError::MissingSourceFace { source_face: 1 });
        assert_eq!(overlay.suppressed_faces, before);
    }

    #[test]
    fn adaptive_leaf_grouping_rejects_parallel_array_truncation() {
        let id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let leaves = [ScreenMeshLeafTopology {
            source_face: 0,
            id,
            domain: id.domain().unwrap(),
        }];
        let topology = HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        let topology_cache = ScreenMeshTopologyCache::from_half_edge_mesh(&topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology_cache).unwrap();
        let error = group_resident_screen_leaves(
            &frontier,
            &[],
            &[0],
            &[0],
            &[0],
        )
        .unwrap_err();
        assert_eq!(error, ScreenLeafLodError::LengthMismatch);

        let mut retained = group_resident_screen_leaves(
            &frontier,
            &[[1; 3]],
            &[0],
            &[0],
            &[0],
        )
        .unwrap();
        let before = retained.clone();
        let mut lod_scratch = ScreenMeshLeafLodScratch::default();
        let error = group_resident_screen_leaves_into(
            &frontier,
            &[[3, 1, 1]],
            &[0],
            &[0],
            &[0],
            &mut lod_scratch,
            &mut retained,
        )
        .unwrap_err();
        assert!(matches!(error, ScreenLeafLodError::InvalidLod { .. }));
        assert_eq!(retained, before);

        let error = group_resident_screen_leaves_into(
            &frontier,
            &[[1; 3]],
            &[],
            &[],
            &[],
            &mut lod_scratch,
            &mut retained,
        )
        .unwrap_err();
        assert!(matches!(error, ScreenLeafLodError::MissingFaceMetadata { .. }));
        assert_eq!(retained, before);
    }

    #[test]
    fn render_state_grouping_preserves_semantic_node_identity() {
        let groups = group_resident_faces(
            &[Some(ResidentLod::uniform(1)); 3],
            &[[1; 3]; 3],
            &[4; 3],
            &[101, 202, 303],
            &[usize::MAX; 3],
            ResidentLod::uniform(1),
        );

        assert_eq!(groups.len(), 1);
        let (key, members) = groups.first_key_value().unwrap();
        assert_eq!(key.render_node_index, usize::MAX);
        assert_eq!(
            members.iter().map(|member| member.node_index).collect::<Vec<_>>(),
            vec![101, 202, 303],
        );
    }

    #[test]
    fn resident_batch_submission_order_is_material_then_node_major() {
        let high_lod_first_material = RenderBatchKey {
            lod: [64, 64, 64],
            parity_bucket: 1,
            material_index: 2,
            render_node_index: 9,
        };
        let low_lod_later_material = RenderBatchKey {
            lod: [1, 1, 1],
            parity_bucket: 0,
            material_index: 3,
            render_node_index: 0,
        };
        let same_material_later_node = RenderBatchKey {
            lod: [1, 1, 1],
            parity_bucket: 0,
            material_index: 2,
            render_node_index: 10,
        };

        assert!(high_lod_first_material < low_lod_later_material);
        assert!(high_lod_first_material < same_material_later_node);
        let retained_root = RenderBatchId {
            key: high_lod_first_material,
            layer: RenderBatchLayer::RetainedRoot,
        };
        let adaptive_overlay = RenderBatchId {
            key: high_lod_first_material,
            layer: RenderBatchLayer::AdaptiveOverlay,
        };
        assert!(retained_root < adaptive_overlay);
        assert!(adaptive_overlay
            < RenderBatchId::complete(same_material_later_node));
    }

    #[test]
    fn resident_group_membership_detects_even_permutation_changes() {
        let a = ResidentLod::from_edge_lods([2, 4, 8]);
        let b = ResidentLod::from_edge_lods([8, 2, 4]);
        assert_eq!(a.canonical, b.canonical);
        assert_eq!(a.parity_bucket, b.parity_bucket);

        let first = group_resident_faces(
            &[Some(a)],
            &[[8, 8, 8]],
            &[0],
            &[0],
            &[0],
            ResidentLod::uniform(1),
        );
        let second = group_resident_faces(
            &[Some(b)],
            &[[8, 8, 8]],
            &[0],
            &[0],
            &[0],
            ResidentLod::uniform(1),
        );
        assert_eq!(first.keys().next(), second.keys().next());
        assert_ne!(first.values().next(), second.values().next());
    }

    #[test]
    fn resident_group_scratch_reuses_member_capacity() {
        let resident = ResidentLod::uniform(4);
        let mut groups = BTreeMap::new();
        group_resident_faces_into(
            &[Some(resident); 8],
            &[[4, 4, 4]; 8],
            &[0; 8],
            &[0; 8],
            &[0; 8],
            ResidentLod::uniform(1),
            &mut groups,
        );
        let capacity = groups.values().next().unwrap().capacity();

        group_resident_faces_into(
            &[Some(resident); 2],
            &[[4, 4, 4]; 2],
            &[0; 2],
            &[0; 2],
            &[0; 2],
            ResidentLod::uniform(1),
            &mut groups,
        );
        assert_eq!(groups.values().next().unwrap().len(), 2);
        assert!(groups.values().next().unwrap().capacity() >= capacity);
    }

    #[test]
    fn resident_vertex_lods_are_shared_and_detect_visualization_changes() {
        let topology = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let residents = [
            Some(ResidentLod::from_edge_lods([2, 4, 8])),
            Some(ResidentLod::from_edge_lods([16, 2, 4])),
        ];
        let mut vertex_max = Vec::new();
        let mut face_vertex_lods = Vec::new();
        rebuild_resident_vertex_lods(
            &residents,
            &topology,
            ResidentLod::uniform(1),
            &mut vertex_max,
            &mut face_vertex_lods,
        );

        assert_eq!(face_vertex_lods[0][1], face_vertex_lods[1][1]);
        assert_eq!(face_vertex_lods[0][2], face_vertex_lods[1][0]);

        let first = group_resident_faces(
            &residents,
            &face_vertex_lods,
            &[0, 0],
            &[0, 0],
            &[0, 0],
            ResidentLod::uniform(1),
        );
        face_vertex_lods[0][0] *= 2;
        let second = group_resident_faces(
            &residents,
            &face_vertex_lods,
            &[0, 0],
            &[0, 0],
            &[0, 0],
            ResidentLod::uniform(1),
        );
        assert_ne!(first, second);
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

        assert_eq!(balance_resident_lods(&mut residents, &topology), 2);
        assert_eq!(residents[0].unwrap().edge_lods(), [32, 16, 16]);
        assert_eq!(residents[1].unwrap().edge_lods(), [16, 16, 32]);
    }

    #[test]
    fn resident_topology_limits_face_anisotropy_to_two_to_one() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let topology = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces);
        let mut residents = [Some(ResidentLod::from_edge_lods([1, 1, 128]))];

        assert_eq!(balance_resident_lods(&mut residents, &topology), 1);
        assert_eq!(residents[0].unwrap().edge_lods(), [64, 64, 128]);
    }

    #[test]
    fn explicit_four_to_one_policy_uses_the_same_reconciler() {
        let faces = [[0, 1, 2]];
        let topology = HalfEdgeMesh::from_triangles(3, &faces);
        let mut residents = [Some(ResidentLod::from_edge_lods([1, 1, 128]))];

        assert_eq!(
            balance_resident_lods_with_ratio::<4>(&mut residents, &topology),
            1,
        );
        assert_eq!(residents[0].unwrap().edge_lods(), [32, 32, 128]);
    }

    #[test]
    fn runtime_grading_policy_admits_only_measured_atlas_ratios() {
        assert_eq!(FaceLodGrading::from_ratio(2), Some(FaceLodGrading::TwoToOne));
        assert_eq!(FaceLodGrading::from_ratio(4), Some(FaceLodGrading::FourToOne));
        for unsupported in [0, 1, 3, 8] {
            assert_eq!(FaceLodGrading::from_ratio(unsupported), None);
        }

        let topology = HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        let mut residents = [Some(ResidentLod::from_edge_lods([1, 1, 128]))];
        let mut scratch = ResidentLodBalanceScratch::default();
        assert_eq!(
            balance_resident_lods_from_faces_with_grading(
                &mut residents,
                &topology,
                &[0],
                &mut scratch,
                FaceLodGrading::FourToOne,
            ),
            1,
        );
        assert_eq!(residents[0].unwrap().edge_lods(), [32, 32, 128]);
    }

    #[test]
    #[should_panic(expected = "positive power of two")]
    fn grading_policy_rejects_ratios_outside_the_atlas_lattice() {
        let topology = HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        let mut residents = [Some(ResidentLod::uniform(1))];
        balance_resident_lods_with_ratio::<3>(&mut residents, &topology);
    }

    #[test]
    fn resident_work_queue_matches_global_fixed_point_on_a_connected_grid() {
        const SIDE: u32 = 24;
        let mut faces = Vec::with_capacity((SIDE * SIDE * 2) as usize);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let row = SIDE + 1;
                let a = y * row + x;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                faces.push([a, b, c]);
                faces.push([b, d, c]);
            }
        }
        let topology = HalfEdgeMesh::from_triangles((SIDE + 1) * (SIDE + 1), &faces);
        let residents: Vec<Option<ResidentLod>> = (0..faces.len())
            .map(|face| {
                if face % 29 == 0 {
                    None
                } else {
                    Some(ResidentLod::from_edge_lods([
                        1 << ((face * 3) % 8),
                        1 << ((face * 5 + 1) % 8),
                        1 << ((face * 7 + 2) % 8),
                    ]))
                }
            })
            .collect();
        let mut expected = residents.clone();
        let mut actual = residents;

        let expected_changed = balance_resident_lods_reference(&mut expected, &topology);
        let actual_changed = balance_resident_lods(&mut actual, &topology);

        assert_eq!(actual_changed, expected_changed);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sparse_resident_frontier_matches_global_reconciliation() {
        const SIDE: u32 = 12;
        let mut faces = Vec::with_capacity((SIDE * SIDE * 2) as usize);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let row = SIDE + 1;
                let a = y * row + x;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                faces.push([a, b, c]);
                faces.push([b, d, c]);
            }
        }
        let topology = HalfEdgeMesh::from_triangles((SIDE + 1) * (SIDE + 1), &faces);
        let mut expected = vec![Some(ResidentLod::uniform(2)); faces.len()];
        let mut actual = expected.clone();
        let dirty_faces = [0, faces.len() / 2, faces.len() - 1];
        let replacements = [
            ResidentLod::from_edge_lods([128, 1, 1]),
            ResidentLod::from_edge_lods([1, 64, 2]),
            ResidentLod::from_edge_lods([2, 1, 32]),
        ];
        for (&face, &resident) in dirty_faces.iter().zip(&replacements) {
            expected[face] = Some(resident);
            actual[face] = Some(resident);
        }

        let expected_changed = balance_resident_lods_reference(&mut expected, &topology);
        let mut scratch = ResidentLodBalanceScratch::default();
        let actual_changed = balance_resident_lods_from_faces(
            &mut actual,
            &topology,
            &dirty_faces,
            &mut scratch,
        );
        assert_eq!(actual_changed, expected_changed);
        assert_eq!(actual, expected);

        // A later classifier may lower a face that was previously promoted by
        // its neighborhood. Sparse reconciliation must restore that edge from
        // the unchanged twin just as a new global pass would.
        let demoted_face = faces.len() / 2;
        expected[demoted_face] = Some(ResidentLod::uniform(1));
        actual[demoted_face] = Some(ResidentLod::uniform(1));
        let expected_changed = balance_resident_lods_reference(&mut expected, &topology);
        let actual_changed = balance_resident_lods_from_faces(
            &mut actual,
            &topology,
            &[demoted_face],
            &mut scratch,
        );
        assert_eq!(actual_changed, expected_changed);
        assert_eq!(actual, expected);
    }

    #[test]
    fn raw_requests_allow_a_promoted_component_to_demote() {
        const SIDE: u32 = 4;
        let mut faces = Vec::with_capacity((SIDE * SIDE * 2) as usize);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let row = SIDE + 1;
                let a = y * row + x;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                faces.push([a, b, c]);
                faces.push([b, d, c]);
            }
        }
        let topology = HalfEdgeMesh::from_triangles((SIDE + 1) * (SIDE + 1), &faces);
        let low = Some(ResidentLod::uniform(2));
        for grading in [FaceLodGrading::TwoToOne, FaceLodGrading::FourToOne] {
            let mut requests = vec![low; faces.len()];
            requests[0] = Some(ResidentLod::from_edge_lods([128, 1, 1]));
            let mut residents = requests.clone();
            let mut scratch = ResidentLodBalanceScratch::default();

            reconcile_resident_lods_from_requests_with_grading(
                &requests,
                &mut residents,
                &topology,
                &[0],
                &mut scratch,
                grading,
            );
            assert!(residents.iter().any(|resident| *resident != low));

            requests[0] = low;
            reconcile_resident_lods_from_requests_with_grading(
                &requests,
                &mut residents,
                &topology,
                &[0],
                &mut scratch,
                grading,
            );
            assert!(residents.iter().all(|resident| *resident == low));
        }
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
