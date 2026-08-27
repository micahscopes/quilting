//! Crack-free LoD reconciliation across adaptive dyadic QB leaves.
//!
//! A coarse leaf edge can meet two or more finer leaf edges. Their local edge
//! LoDs are not directly comparable: the invariant is the absolute dyadic
//! resolution `leaf_depth + log2(local_lod)`. This module groups overlapping
//! collinear spans, promotes each group to one absolute resolution, applies the
//! selected within-leaf grading ratio, and iterates to a fixed point.

use crate::atlas::TessellationAtlas;
use crate::patch::QBPatchDomain;
use crate::permutation::canonical_form;
use crate::screen_partition::{ScreenPatchLeaf, ScreenPatchLeafId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenLeafTopology {
    pub id: ScreenPatchLeafId,
    pub domain: QBPatchDomain,
}

impl From<&ScreenPatchLeaf> for ScreenLeafTopology {
    fn from(leaf: &ScreenPatchLeaf) -> Self {
        Self {
            id: leaf.id,
            domain: leaf.restricted.domain,
        }
    }
}

/// One adaptive leaf located in the welded authored-mesh topology. Source
/// identity remains separate from the ephemeral dyadic path so two faces may
/// choose different refinement trees while still negotiating their common
/// physical boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenMeshLeafTopology {
    pub source_face: u32,
    pub id: ScreenPatchLeafId,
    pub domain: QBPatchDomain,
}

impl ScreenMeshLeafTopology {
    pub fn from_leaf(source_face: u32, leaf: &ScreenPatchLeaf) -> Self {
        Self {
            source_face,
            id: leaf.id,
            domain: leaf.restricted.domain,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenLeafLodResult {
    pub resident: Vec<[u32; 3]>,
    pub iterations: usize,
    pub shared_edge_promotions: usize,
    pub grading_promotions: usize,
    pub max_absolute_exponent: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScreenLeafAtlasWork {
    pub instances: u64,
    pub vertices: u64,
    pub triangles: u64,
}

impl ScreenLeafLodResult {
    /// Exact atlas work implied by the reconciled leaves, before material/pass
    /// multiplication. This gives WebGL2 and WebGPU the same workload oracle.
    pub fn atlas_work(
        &self,
        atlas: &TessellationAtlas,
    ) -> Result<ScreenLeafAtlasWork, ScreenLeafLodError> {
        let mut work = ScreenLeafAtlasWork::default();
        for (leaf_index, edge_lods) in self.resident.iter().copied().enumerate() {
            let key = canonical_form(edge_lods).res;
            let Some(entry) = atlas.patches.get(&key) else {
                return Err(ScreenLeafLodError::MissingAtlasPatch { leaf_index, key });
            };
            work.instances = work.instances.saturating_add(1);
            work.vertices = work.vertices.saturating_add(entry.vertex_count as u64);
            work.triangles = work.triangles.saturating_add(entry.triangle_count as u64);
        }
        Ok(work)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenLeafLodError {
    LengthMismatch,
    InvalidTopology {
        leaf_index: usize,
    },
    InvalidLod {
        leaf_index: usize,
        edge_index: usize,
    },
    InvalidGradingRatio,
    AtlasCapExceeded {
        leaf_index: usize,
        edge_index: usize,
        required_lod: u32,
        max_lod: u32,
    },
    MissingAtlasPatch {
        leaf_index: usize,
        key: [u32; 3],
    },
    AbsoluteLodOverflow {
        leaf_index: usize,
        edge_index: usize,
    },
    OverlappingLeaves {
        first_index: usize,
        second_index: usize,
    },
    MissingFaceMetadata {
        leaf_index: usize,
        source_face: u32,
    },
    MissingSourceFace {
        source_face: u32,
    },
}

impl std::fmt::Display for ScreenLeafLodError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch => write!(formatter, "leaf topology and LoD lengths differ"),
            Self::InvalidTopology { leaf_index } => {
                write!(formatter, "adaptive leaf {leaf_index} has a non-dyadic domain")
            }
            Self::InvalidLod {
                leaf_index,
                edge_index,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} edge {edge_index} has an invalid LoD"
            ),
            Self::InvalidGradingRatio => write!(formatter, "invalid leaf grading ratio"),
            Self::AtlasCapExceeded {
                leaf_index,
                edge_index,
                required_lod,
                max_lod,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} edge {edge_index} needs LoD {required_lod}, atlas cap is {max_lod}"
            ),
            Self::MissingAtlasPatch { leaf_index, key } => write!(
                formatter,
                "adaptive leaf {leaf_index} needs missing atlas patch {key:?}"
            ),
            Self::AbsoluteLodOverflow {
                leaf_index,
                edge_index,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} edge {edge_index} absolute LoD overflows u32"
            ),
            Self::OverlappingLeaves {
                first_index,
                second_index,
            } => write!(
                formatter,
                "adaptive leaves {first_index} and {second_index} overlap"
            ),
            Self::MissingFaceMetadata {
                leaf_index,
                source_face,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} references source face {source_face} without complete render metadata"
            ),
            Self::MissingSourceFace { source_face } => write!(
                formatter,
                "adaptive frontier does not cover source face {source_face}",
            ),
        }
    }
}

impl std::error::Error for ScreenLeafLodError {}

/// Reject duplicate or ancestor/descendant leaves before reconciliation or
/// rendering. A visible frontier may omit culled regions, so this validates an
/// antichain rather than requiring complete coverage of every source face.
pub fn validate_screen_mesh_leaf_antichain(
    leaves: &[ScreenMeshLeafTopology],
) -> Result<(), ScreenLeafLodError> {
    let mut indices = BTreeMap::<(u32, ScreenPatchLeafId), usize>::new();
    for (leaf_index, leaf) in leaves.iter().copied().enumerate() {
        if leaf.id.domain() != Some(leaf.domain) {
            return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
        }
        if let Some(first_index) = indices.insert((leaf.source_face, leaf.id), leaf_index) {
            return Err(ScreenLeafLodError::OverlappingLeaves {
                first_index,
                second_index: leaf_index,
            });
        }
    }
    for (&(source_face, leaf_id), &leaf_index) in &indices {
        for ancestor_depth in 0..leaf_id.depth {
            let ancestor = leaf_id
                .ancestor_at_depth(ancestor_depth)
                .expect("validated leaf has every shallower ancestor");
            if let Some(&ancestor_index) = indices.get(&(source_face, ancestor)) {
                return Err(ScreenLeafLodError::OverlappingLeaves {
                    first_index: ancestor_index.min(leaf_index),
                    second_index: ancestor_index.max(leaf_index),
                });
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LineKey {
    constant_axis: u8,
    constant_numerator: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MeshLineKey {
    Interior {
        source_face: u32,
        constant_axis: u8,
        constant_numerator: u32,
    },
    SourceBoundary {
        canonical_half_edge: u32,
    },
}

type MeshTopologyLines = Vec<(MeshLineKey, Vec<EdgeSpan>)>;

#[derive(Clone, Copy, Debug)]
struct EdgeSpan {
    leaf_index: usize,
    edge_index: usize,
    start: u32,
    end: u32,
    depth: u8,
}

#[derive(Clone, Copy, Debug)]
struct QuantizedEdgeSpan {
    constant_axis: u8,
    constant_numerator: u32,
    endpoints: [[u32; 3]; 2],
    span: EdgeSpan,
}

fn quantized_domain_corners(
    leaf_index: usize,
    id: ScreenPatchLeafId,
    domain: QBPatchDomain,
    global_depth: u8,
) -> Result<[[u32; 3]; 3], ScreenLeafLodError> {
    if global_depth > 16 || id.depth > global_depth || id.domain() != Some(domain) {
        return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
    }
    let denominator = 1u32 << id.depth;
    let global_scale = 1u32 << (global_depth - id.depth);
    let mut corners = [[0u32; 3]; 3];
    for (corner_index, barycentric) in domain.corners.into_iter().enumerate() {
        let mut sum = 0u32;
        for coordinate in 0..3 {
            let scaled = barycentric[coordinate] * f64::from(denominator);
            let rounded = scaled.round();
            if !scaled.is_finite()
                || rounded < 0.0
                || rounded > f64::from(denominator)
                || (scaled - rounded).abs() > 1.0e-9
            {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
            let numerator = rounded as u32;
            corners[corner_index][coordinate] = numerator * global_scale;
            sum += numerator;
        }
        if sum != denominator {
            return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
        }
    }
    Ok(corners)
}

fn quantized_edge_spans(
    leaves: impl IntoIterator<Item = (usize, ScreenPatchLeafId, QBPatchDomain)>,
    global_depth: u8,
) -> Result<Vec<QuantizedEdgeSpan>, ScreenLeafLodError> {
    let mut edges = Vec::new();
    let edge_corners = [(1usize, 2usize), (0, 2), (0, 1)];
    for (leaf_index, id, domain) in leaves {
        let corners = quantized_domain_corners(leaf_index, id, domain, global_depth)?;
        let global_scale = 1u32 << (global_depth - id.depth);

        for (edge_index, (first, second)) in edge_corners.into_iter().enumerate() {
            let a = corners[first];
            let b = corners[second];
            let Some(constant_axis) = (0..3).find(|axis| a[*axis] == b[*axis]) else {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            };
            let parameter_axis = (constant_axis + 1) % 3;
            let start = a[parameter_axis].min(b[parameter_axis]);
            let end = a[parameter_axis].max(b[parameter_axis]);
            if start == end || end - start != global_scale {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
            edges.push(QuantizedEdgeSpan {
                constant_axis: constant_axis as u8,
                constant_numerator: a[constant_axis],
                endpoints: [a, b],
                span: EdgeSpan {
                    leaf_index,
                    edge_index,
                    start,
                    end,
                    depth: id.depth,
                },
            });
        }
    }
    Ok(edges)
}

fn topology_lines(
    leaves: &[ScreenLeafTopology],
) -> Result<BTreeMap<LineKey, Vec<EdgeSpan>>, ScreenLeafLodError> {
    let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
    let mut lines = BTreeMap::<LineKey, Vec<EdgeSpan>>::new();
    let edges = quantized_edge_spans(
        leaves
            .iter()
            .enumerate()
            .map(|(index, leaf)| (index, leaf.id, leaf.domain)),
        global_depth,
    )?;
    for edge in edges {
        lines
            .entry(LineKey {
                constant_axis: edge.constant_axis,
                constant_numerator: edge.constant_numerator,
            })
            .or_default()
            .push(edge.span);
    }
    for spans in lines.values_mut() {
        spans.sort_by_key(|span| (span.start, span.end, span.leaf_index, span.edge_index));
    }
    Ok(lines)
}

fn mesh_topology_lines_with<F>(
    leaves: &[ScreenMeshLeafTopology],
    source_half_edge_count: usize,
    mut face_edge: F,
) -> Result<MeshTopologyLines, ScreenLeafLodError>
where
    F: FnMut(u32, usize, usize) -> Result<CachedSourceEdge, ScreenLeafLodError>,
{
    let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
    let global_denominator = 1u32
        .checked_shl(u32::from(global_depth))
        .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
    let edges = quantized_edge_spans(
        leaves
            .iter()
            .enumerate()
            .map(|(index, leaf)| (index, leaf.id, leaf.domain)),
        global_depth,
    )?;
    let mut source_lines = vec![Vec::<EdgeSpan>::new(); source_half_edge_count];
    let mut interior_lines = BTreeMap::<MeshLineKey, Vec<EdgeSpan>>::new();
    for edge in edges {
        let leaf = &leaves[edge.span.leaf_index];
        if edge.constant_numerator == 0 {
            let source_edge = usize::from(edge.constant_axis);
            let cached_edge = face_edge(leaf.source_face, source_edge, edge.span.leaf_index)?;
            let parameter_axis = (source_edge + 2) % 3;
            let mut first = edge.endpoints[0][parameter_axis];
            let mut second = edge.endpoints[1][parameter_axis];
            if cached_edge.reversed() {
                first = global_denominator - first;
                second = global_denominator - second;
            }
            let mut span = edge.span;
            span.start = first.min(second);
            span.end = first.max(second);
            let canonical_half_edge = cached_edge.canonical_half_edge() as usize;
            source_lines
                .get_mut(canonical_half_edge)
                .ok_or(ScreenLeafLodError::InvalidTopology {
                    leaf_index: edge.span.leaf_index,
                })?
                .push(span);
        } else {
            interior_lines
                .entry(MeshLineKey::Interior {
                    source_face: leaf.source_face,
                    constant_axis: edge.constant_axis,
                    constant_numerator: edge.constant_numerator,
                })
                .or_default()
                .push(edge.span);
        }
    }
    for spans in interior_lines.values_mut() {
        spans.sort_by_key(|span| (span.start, span.end, span.leaf_index, span.edge_index));
    }
    for spans in &mut source_lines {
        spans.sort_by_key(|span| (span.start, span.end, span.leaf_index, span.edge_index));
    }
    let mut lines = interior_lines.into_iter().collect::<MeshTopologyLines>();
    lines.reserve(
        source_lines
            .iter()
            .filter(|spans| !spans.is_empty())
            .count(),
    );
    for (canonical_half_edge, spans) in source_lines.into_iter().enumerate() {
        if spans.is_empty() {
            continue;
        }
        lines.push((
            MeshLineKey::SourceBoundary {
                canonical_half_edge: canonical_half_edge as u32,
            },
            spans,
        ));
    }
    Ok(lines)
}

fn mesh_topology_lines(
    leaves: &[ScreenMeshLeafTopology],
    topology: &ScreenMeshTopologyCache,
) -> Result<MeshTopologyLines, ScreenLeafLodError> {
    mesh_topology_lines_with(
        leaves,
        topology.source_half_edge_count,
        |source_face, source_edge, leaf_index| {
            topology.face_edge(source_face, source_edge, leaf_index)
        },
    )
}

fn mesh_topology_lines_from_half_edge_mesh(
    leaves: &[ScreenMeshLeafTopology],
    source_topology: &quilting_mesh::HalfEdgeMesh,
) -> Result<MeshTopologyLines, ScreenLeafLodError> {
    mesh_topology_lines_with(
        leaves,
        source_topology.half_edges.len(),
        |source_face, source_edge, leaf_index| {
            if source_face >= source_topology.num_faces {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
            let half_edge = source_topology.face_half_edges(source_face)[(source_edge + 1) % 3];
            let canonical_half_edge = source_topology.canonical_edge(half_edge);
            CachedSourceEdge::new(canonical_half_edge, half_edge != canonical_half_edge)
                .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index })
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MeshVertexKey {
    SourceVertex(u32),
    SourceEdge {
        canonical_half_edge: u32,
        parameter: u32,
    },
    FaceInterior {
        source_face: u32,
        barycentric: [u32; 3],
    },
}

/// Dense source vertices dominate a root-heavy frontier, while edge/interior
/// vertices exist only around adaptively split faces. Index the former
/// directly and reserve ordered lookup for the sparse refined remainder.
struct MeshVertexIndex {
    source_vertices: Vec<usize>,
    refined_vertices: BTreeMap<MeshVertexKey, usize>,
    len: usize,
}

impl MeshVertexIndex {
    const MISSING: usize = usize::MAX;

    fn new(source_vertex_count: usize) -> Self {
        Self {
            source_vertices: vec![Self::MISSING; source_vertex_count],
            refined_vertices: BTreeMap::new(),
            len: 0,
        }
    }

    fn get_or_insert(&mut self, key: MeshVertexKey) -> Option<usize> {
        match key {
            MeshVertexKey::SourceVertex(vertex) => {
                let slot = self.source_vertices.get_mut(vertex as usize)?;
                if *slot == Self::MISSING {
                    *slot = self.len;
                    self.len = self.len.checked_add(1)?;
                }
                Some(*slot)
            }
            _ => {
                if let Some(&index) = self.refined_vertices.get(&key) {
                    return Some(index);
                }
                let index = self.len;
                self.len = self.len.checked_add(1)?;
                self.refined_vertices.insert(key, index);
                Some(index)
            }
        }
    }

    fn get(&self, key: &MeshVertexKey) -> Option<usize> {
        match *key {
            MeshVertexKey::SourceVertex(vertex) => self
                .source_vertices
                .get(vertex as usize)
                .copied()
                .filter(|&index| index != Self::MISSING),
            _ => self.refined_vertices.get(key).copied(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

fn find_vertex_root(parents: &mut [u32], vertex: u32) -> u32 {
    let mut root = vertex;
    while parents[root as usize] != root {
        root = parents[root as usize];
    }
    let mut current = vertex;
    while parents[current as usize] != current {
        let next = parents[current as usize];
        parents[current as usize] = root;
        current = next;
    }
    root
}

fn union_vertices(parents: &mut [u32], first: u32, second: u32) {
    let first_root = find_vertex_root(parents, first);
    let second_root = find_vertex_root(parents, second);
    if first_root == second_root {
        return;
    }
    let (minimum, maximum) = if first_root < second_root {
        (first_root, second_root)
    } else {
        (second_root, first_root)
    };
    parents[maximum as usize] = minimum;
}

/// Canonicalize source vertices joined only through actual half-edge twins.
/// Exact-welded glTF seams retain their distinct attribute vertices, but a
/// joined opposing edge proves that its corresponding endpoints are the same
/// geometric points for resident-density visualization.
fn canonical_source_vertices(
    source_topology: &quilting_mesh::HalfEdgeMesh,
) -> Result<Vec<u32>, ScreenLeafLodError> {
    let mut parents = (0..source_topology.num_vertices).collect::<Vec<_>>();
    for half_edge in 0..source_topology.half_edges.len() as u32 {
        let Some(twin) = source_topology.twin(half_edge) else {
            continue;
        };
        if half_edge >= twin {
            continue;
        }
        let edge = &source_topology.half_edges[half_edge as usize];
        let opposing = &source_topology.half_edges[twin as usize];
        let from = source_topology.half_edges[edge.prev as usize].vertex;
        let to = edge.vertex;
        let opposing_from = source_topology.half_edges[opposing.prev as usize].vertex;
        let opposing_to = opposing.vertex;
        if [from, to, opposing_from, opposing_to]
            .into_iter()
            .any(|vertex| vertex >= source_topology.num_vertices)
        {
            return Err(ScreenLeafLodError::InvalidTopology { leaf_index: 0 });
        }
        union_vertices(&mut parents, from, opposing_to);
        union_vertices(&mut parents, to, opposing_from);
    }
    for vertex in 0..source_topology.num_vertices {
        parents[vertex as usize] = find_vertex_root(&mut parents, vertex);
    }
    Ok(parents)
}

#[derive(Clone, Copy, Debug)]
struct CachedSourceEdge(u32);

impl CachedSourceEdge {
    const REVERSED: u32 = 1 << 31;

    fn new(canonical_half_edge: u32, reversed: bool) -> Option<Self> {
        (canonical_half_edge < Self::REVERSED).then_some(Self(
            canonical_half_edge | if reversed { Self::REVERSED } else { 0 },
        ))
    }

    fn canonical_half_edge(self) -> u32 {
        self.0 & !Self::REVERSED
    }

    fn reversed(self) -> bool {
        self.0 & Self::REVERSED != 0
    }
}

/// Source-mesh identities needed by every camera-dependent adaptive frontier.
/// Build this once with the welded half-edge mesh; leaf repartitioning then
/// avoids rescanning the complete authored topology. The source
/// [`quilting_mesh::HalfEdgeMesh`] must be treated as immutable while this
/// cache or any frontier built from it remains in use; rebuild after mutation.
#[derive(Debug)]
pub struct ScreenMeshTopologyCache {
    source_vertex_count: usize,
    source_half_edge_count: usize,
    canonical_face_vertices: Vec<[u32; 3]>,
    face_edges: Vec<[CachedSourceEdge; 3]>,
}

impl ScreenMeshTopologyCache {
    pub fn from_half_edge_mesh(
        source_topology: &quilting_mesh::HalfEdgeMesh,
    ) -> Result<Self, ScreenLeafLodError> {
        let canonical_vertices = canonical_source_vertices(source_topology)?;
        let mut canonical_face_vertices = Vec::with_capacity(source_topology.num_faces as usize);
        let mut face_edges = Vec::with_capacity(source_topology.num_faces as usize);
        for source_face in 0..source_topology.num_faces {
            let vertices = source_topology.face_vertices(source_face);
            let mut canonical = [0u32; 3];
            for (corner, vertex) in vertices.into_iter().enumerate() {
                canonical[corner] = canonical_vertices
                    .get(vertex as usize)
                    .copied()
                    .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
            }
            canonical_face_vertices.push(canonical);
            let half_edges = source_topology.face_half_edges(source_face);
            let mut cached_edges = [CachedSourceEdge(0); 3];
            for source_edge in 0..3 {
                let half_edge = half_edges[(source_edge + 1) % 3];
                let canonical_half_edge = source_topology.canonical_edge(half_edge);
                cached_edges[source_edge] =
                    CachedSourceEdge::new(canonical_half_edge, half_edge != canonical_half_edge)
                        .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
            }
            face_edges.push(cached_edges);
        }
        Ok(Self {
            source_vertex_count: canonical_vertices.len(),
            source_half_edge_count: source_topology.half_edges.len(),
            canonical_face_vertices,
            face_edges,
        })
    }

    fn face_vertices(
        &self,
        source_face: u32,
        leaf_index: usize,
    ) -> Result<[u32; 3], ScreenLeafLodError> {
        self.canonical_face_vertices
            .get(source_face as usize)
            .copied()
            .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index })
    }

    fn face_edge(
        &self,
        source_face: u32,
        source_edge: usize,
        leaf_index: usize,
    ) -> Result<CachedSourceEdge, ScreenLeafLodError> {
        self.face_edges
            .get(source_face as usize)
            .and_then(|edges| edges.get(source_edge))
            .copied()
            .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index })
    }
}

fn mesh_vertex_key(
    leaf_index: usize,
    leaf: ScreenMeshLeafTopology,
    corner: [u32; 3],
    global_denominator: u32,
    topology: &ScreenMeshTopologyCache,
) -> Result<MeshVertexKey, ScreenLeafLodError> {
    let zero_axes = (0..3).filter(|&axis| corner[axis] == 0).collect::<Vec<_>>();
    match zero_axes.as_slice() {
        [first, second] => {
            let source_corner = 3usize - first - second;
            if corner[source_corner] != global_denominator {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
            let canonical = topology.face_vertices(leaf.source_face, leaf_index)?[source_corner];
            Ok(MeshVertexKey::SourceVertex(canonical))
        }
        [source_edge] => {
            let source_edge = *source_edge;
            let edge = topology.face_edge(leaf.source_face, source_edge, leaf_index)?;
            let parameter_axis = (source_edge + 2) % 3;
            let mut parameter = corner[parameter_axis];
            if edge.reversed() {
                parameter = global_denominator - parameter;
            }
            Ok(MeshVertexKey::SourceEdge {
                canonical_half_edge: edge.canonical_half_edge(),
                parameter,
            })
        }
        [] => Ok(MeshVertexKey::FaceInterior {
            source_face: leaf.source_face,
            barycentric: corner,
        }),
        _ => Err(ScreenLeafLodError::InvalidTopology { leaf_index }),
    }
}

fn absolute_leaf_edge_lods(
    leaf_index: usize,
    leaf: ScreenMeshLeafTopology,
    edge_lods: [u32; 3],
) -> Result<[u32; 3], ScreenLeafLodError> {
    if leaf.id.domain() != Some(leaf.domain) {
        return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
    }
    let scale = 1u32.checked_shl(u32::from(leaf.id.depth)).ok_or(
        ScreenLeafLodError::AbsoluteLodOverflow {
            leaf_index,
            edge_index: 0,
        },
    )?;
    let mut absolute_edges = [0u32; 3];
    for (edge_index, local_lod) in edge_lods.into_iter().enumerate() {
        if !local_lod.is_power_of_two() {
            return Err(ScreenLeafLodError::InvalidLod {
                leaf_index,
                edge_index,
            });
        }
        absolute_edges[edge_index] =
            local_lod
                .checked_mul(scale)
                .ok_or(ScreenLeafLodError::AbsoluteLodOverflow {
                    leaf_index,
                    edge_index,
                })?;
    }
    Ok(absolute_edges)
}

fn mesh_line_vertex_key(
    line: MeshLineKey,
    parameter: u32,
    global_denominator: u32,
    leaf_index: usize,
) -> Result<MeshVertexKey, ScreenLeafLodError> {
    match line {
        MeshLineKey::SourceBoundary {
            canonical_half_edge,
        } => Ok(MeshVertexKey::SourceEdge {
            canonical_half_edge,
            parameter,
        }),
        MeshLineKey::Interior {
            source_face,
            constant_axis,
            constant_numerator,
        } => {
            let constant_axis = usize::from(constant_axis);
            let parameter_axis = (constant_axis + 1) % 3;
            let remaining_axis = (constant_axis + 2) % 3;
            let Some(remaining) = global_denominator
                .checked_sub(constant_numerator)
                .and_then(|rest| rest.checked_sub(parameter))
            else {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            };
            let mut barycentric = [0u32; 3];
            barycentric[constant_axis] = constant_numerator;
            barycentric[parameter_axis] = parameter;
            barycentric[remaining_axis] = remaining;
            Ok(MeshVertexKey::FaceInterior {
                source_face,
                barycentric,
            })
        }
    }
}

/// Camera-dependent dyadic frontier with its welded line and corner incidence
/// prepared once. Reconciliation and draw grouping share this object, avoiding
/// a second topology reconstruction on every classification update.
#[derive(Debug)]
pub struct ScreenMeshLeafFrontier {
    leaves: Vec<ScreenMeshLeafTopology>,
    depths: Vec<u8>,
    lines: MeshTopologyLines,
    corner_vertices: Vec<[usize; 3]>,
    hanging_offsets: Vec<u32>,
    hanging_sources: Vec<u32>,
}

/// Retained work buffers for converting reconciled edge LoDs into physical
/// corner overrides. One renderer-owned value makes repeated grouping
/// allocation-free when the frontier cardinalities are stable.
#[derive(Debug, Default)]
pub struct ScreenMeshLeafLodScratch {
    absolute_edges: Vec<[u32; 3]>,
    vertex_max: Vec<u32>,
    hanging_lods: Vec<u32>,
    corner_lods: Vec<[u32; 3]>,
}

impl ScreenMeshLeafFrontier {
    pub fn build(
        leaves: &[ScreenMeshLeafTopology],
        topology: &ScreenMeshTopologyCache,
    ) -> Result<Self, ScreenLeafLodError> {
        validate_screen_mesh_leaf_antichain(leaves)?;
        let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
        let global_denominator = 1u32
            .checked_shl(u32::from(global_depth))
            .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
        let lines = mesh_topology_lines(leaves, topology)?;
        let mut vertex_indices = MeshVertexIndex::new(topology.source_vertex_count);
        let mut corner_vertices = Vec::<[usize; 3]>::with_capacity(leaves.len());

        for (leaf_index, leaf) in leaves.iter().copied().enumerate() {
            let corners = quantized_domain_corners(leaf_index, leaf.id, leaf.domain, global_depth)?;
            let mut leaf_vertices = [0usize; 3];
            for corner_index in 0..3 {
                let key = mesh_vertex_key(
                    leaf_index,
                    leaf,
                    corners[corner_index],
                    global_denominator,
                    topology,
                )?;
                leaf_vertices[corner_index] = vertex_indices
                    .get_or_insert(key)
                    .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index })?;
            }
            corner_vertices.push(leaf_vertices);
        }

        let mut hanging_pairs = Vec::<(usize, u32)>::new();
        let mut endpoints = Vec::<u32>::new();
        for (line, spans) in &lines {
            if spans.len() < 2
                || spans
                    .iter()
                    .all(|span| span.start == spans[0].start && span.end == spans[0].end)
            {
                continue;
            }
            endpoints.clear();
            endpoints.reserve(spans.len().saturating_mul(2));
            endpoints.extend(spans.iter().flat_map(|span| [span.start, span.end]));
            endpoints.sort_unstable();
            endpoints.dedup();
            for span in spans {
                let first_interior = endpoints.partition_point(|&value| value <= span.start);
                let after_interior = endpoints.partition_point(|&value| value < span.end);
                for &parameter in &endpoints[first_interior..after_interior] {
                    let key = mesh_line_vertex_key(
                        *line,
                        parameter,
                        global_denominator,
                        span.leaf_index,
                    )?;
                    let Some(vertex_index) = vertex_indices.get(&key) else {
                        return Err(ScreenLeafLodError::InvalidTopology {
                            leaf_index: span.leaf_index,
                        });
                    };
                    let leaf_index = u32::try_from(span.leaf_index)
                        .ok()
                        .filter(|index| *index <= u32::MAX >> 2)
                        .ok_or(ScreenLeafLodError::InvalidTopology {
                            leaf_index: span.leaf_index,
                        })?;
                    hanging_pairs.push((vertex_index, (leaf_index << 2) | span.edge_index as u32));
                }
            }
        }
        hanging_pairs.sort_unstable();
        hanging_pairs.dedup();
        let mut hanging_offsets = Vec::with_capacity(vertex_indices.len() + 1);
        let mut hanging_sources = Vec::with_capacity(hanging_pairs.len());
        let mut pair_index = 0usize;
        for vertex_index in 0..vertex_indices.len() {
            hanging_offsets.push(
                u32::try_from(hanging_sources.len())
                    .map_err(|_| ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?,
            );
            while pair_index < hanging_pairs.len() && hanging_pairs[pair_index].0 == vertex_index {
                hanging_sources.push(hanging_pairs[pair_index].1);
                pair_index += 1;
            }
        }
        debug_assert_eq!(pair_index, hanging_pairs.len());
        hanging_offsets.push(
            u32::try_from(hanging_sources.len())
                .map_err(|_| ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?,
        );

        Ok(Self {
            leaves: leaves.to_vec(),
            depths: leaves.iter().map(|leaf| leaf.id.depth).collect(),
            lines,
            corner_vertices,
            hanging_offsets,
            hanging_sources,
        })
    }

    pub fn leaves(&self) -> &[ScreenMeshLeafTopology] {
        &self.leaves
    }

    pub fn reconcile_lods(
        &self,
        requested: &[[u32; 3]],
        max_face_edge_ratio: u32,
        max_lod: u32,
    ) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
        reconcile_mesh_lods(
            &self.depths,
            &self.lines,
            requested,
            max_face_edge_ratio,
            max_lod,
        )
    }

    /// Physical-vertex corner overrides used beside the normative edge-density
    /// field. Conforming corners max-reduce all incident edges; a hanging
    /// corner inherits the absolute LoD of the coarser edge containing it.
    pub fn rebuild_vertex_lods(
        &self,
        resident: &[[u32; 3]],
    ) -> Result<Vec<[u32; 3]>, ScreenLeafLodError> {
        let mut scratch = ScreenMeshLeafLodScratch::default();
        self.rebuild_vertex_lods_into(resident, &mut scratch)?;
        Ok(scratch.corner_lods)
    }

    pub fn rebuild_vertex_lods_into<'a>(
        &self,
        resident: &[[u32; 3]],
        scratch: &'a mut ScreenMeshLeafLodScratch,
    ) -> Result<&'a [[u32; 3]], ScreenLeafLodError> {
        if self.leaves.len() != resident.len() {
            return Err(ScreenLeafLodError::LengthMismatch);
        }
        scratch.absolute_edges.clear();
        scratch.absolute_edges.reserve(self.leaves.len());
        for (leaf_index, (leaf, lods)) in self
            .leaves
            .iter()
            .copied()
            .zip(resident.iter().copied())
            .enumerate()
        {
            scratch
                .absolute_edges
                .push(absolute_leaf_edge_lods(leaf_index, leaf, lods)?);
        }
        scratch.vertex_max.clear();
        let vertex_count = self.hanging_offsets.len().saturating_sub(1);
        scratch.vertex_max.resize(vertex_count, 1);
        for (leaf_index, vertices) in self.corner_vertices.iter().copied().enumerate() {
            let edges = scratch.absolute_edges[leaf_index];
            let corner_lods = [
                edges[1].max(edges[2]),
                edges[0].max(edges[2]),
                edges[0].max(edges[1]),
            ];
            for (vertex, lod) in vertices.into_iter().zip(corner_lods) {
                scratch.vertex_max[vertex] = scratch.vertex_max[vertex].max(lod);
            }
        }
        scratch.hanging_lods.clear();
        scratch.hanging_lods.reserve(vertex_count);
        for vertex_index in 0..vertex_count {
            let start = self.hanging_offsets[vertex_index] as usize;
            let end = self.hanging_offsets[vertex_index + 1] as usize;
            let mut maximum = 0u32;
            for packed in &self.hanging_sources[start..end] {
                let leaf_index = (packed >> 2) as usize;
                let edge_index = (packed & 3) as usize;
                maximum = maximum.max(scratch.absolute_edges[leaf_index][edge_index]);
            }
            scratch.hanging_lods.push(maximum);
        }
        scratch.corner_lods.clear();
        scratch.corner_lods.reserve(self.corner_vertices.len());
        for vertices in &self.corner_vertices {
            scratch.corner_lods.push(vertices.map(|vertex| {
                let hanging = scratch.hanging_lods[vertex];
                if hanging == 0 {
                    scratch.vertex_max[vertex]
                } else {
                    hanging
                }
            }));
        }
        Ok(&scratch.corner_lods)
    }
}

/// Convenience path for callers that do not retain a source topology cache or
/// adaptive frontier. Frequent classification paths should build both once.
pub fn rebuild_screen_mesh_leaf_vertex_lods(
    leaves: &[ScreenMeshLeafTopology],
    resident: &[[u32; 3]],
    source_topology: &quilting_mesh::HalfEdgeMesh,
) -> Result<Vec<[u32; 3]>, ScreenLeafLodError> {
    let topology = ScreenMeshTopologyCache::from_half_edge_mesh(source_topology)?;
    ScreenMeshLeafFrontier::build(leaves, &topology)?.rebuild_vertex_lods(resident)
}

fn lod_exponent(lod: u32) -> Option<u8> {
    lod.is_power_of_two().then_some(lod.trailing_zeros() as u8)
}

fn required_local_lod(
    absolute_exponent: u8,
    span: EdgeSpan,
    max_lod: u32,
) -> Result<u32, ScreenLeafLodError> {
    let local_exponent = absolute_exponent
        .checked_sub(span.depth)
        .expect("component maximum includes every member");
    let required_lod = 1u32.checked_shl(u32::from(local_exponent)).ok_or(
        ScreenLeafLodError::AtlasCapExceeded {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
            required_lod: u32::MAX,
            max_lod,
        },
    )?;
    if required_lod > max_lod {
        return Err(ScreenLeafLodError::AtlasCapExceeded {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
            required_lod,
            max_lod,
        });
    }
    Ok(required_lod)
}

fn reconcile_component(
    component: &[EdgeSpan],
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut absolute_exponent = 0u8;
    for span in component {
        let lod = resident[span.leaf_index][span.edge_index];
        let exponent = lod_exponent(lod).ok_or(ScreenLeafLodError::InvalidLod {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
        })?;
        absolute_exponent = absolute_exponent.max(span.depth.saturating_add(exponent));
    }
    let mut promotions = 0;
    for span in component {
        let required = required_local_lod(absolute_exponent, *span, max_lod)?;
        let current = &mut resident[span.leaf_index][span.edge_index];
        if *current < required {
            *current = required;
            promotions += 1;
        }
    }
    Ok(promotions)
}

fn reconcile_line_spans<'a>(
    lines: impl IntoIterator<Item = &'a [EdgeSpan]>,
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut promotions = 0;
    for spans in lines {
        let mut component_start = 0usize;
        while component_start < spans.len() {
            let mut component_end = component_start + 1;
            let mut covered_until = spans[component_start].end;
            while component_end < spans.len() && spans[component_end].start < covered_until {
                covered_until = covered_until.max(spans[component_end].end);
                component_end += 1;
            }
            promotions +=
                reconcile_component(&spans[component_start..component_end], resident, max_lod)?;
            component_start = component_end;
        }
    }
    Ok(promotions)
}

fn reconcile_lines<K: Ord>(
    lines: &BTreeMap<K, Vec<EdgeSpan>>,
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    reconcile_line_spans(lines.values().map(Vec::as_slice), resident, max_lod)
}

fn reconcile_mesh_lines(
    lines: &MeshTopologyLines,
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    reconcile_line_spans(
        lines.iter().map(|(_, spans)| spans.as_slice()),
        resident,
        max_lod,
    )
}

fn apply_grading(
    resident: &mut [[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut promotions = 0;
    for (leaf_index, lods) in resident.iter_mut().enumerate() {
        let largest = *lods.iter().max().expect("three edge LoDs");
        let minimum = (largest / max_face_edge_ratio).max(1);
        for (edge_index, lod) in lods.iter_mut().enumerate() {
            if *lod < minimum {
                if minimum > max_lod {
                    return Err(ScreenLeafLodError::AtlasCapExceeded {
                        leaf_index,
                        edge_index,
                        required_lod: minimum,
                        max_lod,
                    });
                }
                *lod = minimum;
                promotions += 1;
            }
        }
    }
    Ok(promotions)
}

fn reconcile_lods_with(
    depths: &[u8],
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
    mut reconcile_shared: impl FnMut(&mut [[u32; 3]], u32) -> Result<usize, ScreenLeafLodError>,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    if depths.len() != requested.len() {
        return Err(ScreenLeafLodError::LengthMismatch);
    }
    if max_face_edge_ratio < 2
        || !max_face_edge_ratio.is_power_of_two()
        || max_lod == 0
        || !max_lod.is_power_of_two()
    {
        return Err(ScreenLeafLodError::InvalidGradingRatio);
    }
    for (leaf_index, lods) in requested.iter().enumerate() {
        for (edge_index, lod) in lods.iter().copied().enumerate() {
            if !lod.is_power_of_two() || lod > max_lod {
                return Err(ScreenLeafLodError::InvalidLod {
                    leaf_index,
                    edge_index,
                });
            }
        }
    }

    let mut resident = requested.to_vec();
    let mut iterations = 0usize;
    let mut shared_edge_promotions = 0usize;
    let mut grading_promotions = 0usize;
    loop {
        iterations += 1;
        let shared = reconcile_shared(&mut resident, max_lod)?;
        let graded = apply_grading(&mut resident, max_face_edge_ratio, max_lod)?;
        shared_edge_promotions += shared;
        grading_promotions += graded;
        if shared == 0 && graded == 0 {
            break;
        }
    }

    let max_absolute_exponent = depths
        .iter()
        .zip(&resident)
        .flat_map(|(&depth, lods)| lods.map(|lod| depth.saturating_add(lod.trailing_zeros() as u8)))
        .max()
        .unwrap_or(0);
    Ok(ScreenLeafLodResult {
        resident,
        iterations,
        shared_edge_promotions,
        grading_promotions,
        max_absolute_exponent,
    })
}

fn reconcile_lods<K: Ord>(
    depths: &[u8],
    lines: &BTreeMap<K, Vec<EdgeSpan>>,
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    reconcile_lods_with(
        depths,
        requested,
        max_face_edge_ratio,
        max_lod,
        |resident, max_lod| reconcile_lines(lines, resident, max_lod),
    )
}

fn reconcile_mesh_lods(
    depths: &[u8],
    lines: &MeshTopologyLines,
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    reconcile_lods_with(
        depths,
        requested,
        max_face_edge_ratio,
        max_lod,
        |resident, max_lod| reconcile_mesh_lines(lines, resident, max_lod),
    )
}

/// Reconcile local leaf LoDs across one-to-many shared edges inside one source
/// patch and apply within-leaf atlas grading. Inputs and outputs use logical
/// edge order A/B/C.
pub fn reconcile_screen_leaf_lods(
    leaves: &[ScreenLeafTopology],
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    let lines = topology_lines(leaves)?;
    let depths = leaves.iter().map(|leaf| leaf.id.depth).collect::<Vec<_>>();
    reconcile_lods(&depths, &lines, requested, max_face_edge_ratio, max_lod)
}

/// Reconcile adaptive leaves across both their source-face interiors and the
/// welded authored-mesh boundaries. Different neighboring faces may choose
/// different dyadic trees; overlapping physical edge intervals still receive
/// one absolute sampling resolution.
pub fn reconcile_screen_mesh_leaf_lods(
    leaves: &[ScreenMeshLeafTopology],
    requested: &[[u32; 3]],
    source_topology: &quilting_mesh::HalfEdgeMesh,
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    validate_screen_mesh_leaf_antichain(leaves)?;
    let lines = mesh_topology_lines_from_half_edge_mesh(leaves, source_topology)?;
    let depths = leaves.iter().map(|leaf| leaf.id.depth).collect::<Vec<_>>();
    reconcile_mesh_lods(&depths, &lines, requested, max_face_edge_ratio, max_lod)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::BuildMode;
    use crate::sampling::PatchConfig;

    fn mixed_depth_partition() -> Vec<ScreenLeafTopology> {
        let root = QBPatchDomain::FULL.quarter();
        let mut leaves = Vec::new();
        for child_index in 0..3u8 {
            leaves.push(ScreenLeafTopology {
                id: ScreenPatchLeafId::ROOT.child(child_index).unwrap(),
                domain: root[child_index as usize],
            });
        }
        let centre_id = ScreenPatchLeafId::ROOT.child(3).unwrap();
        for (grandchild_index, local) in QBPatchDomain::FULL.quarter().into_iter().enumerate() {
            leaves.push(ScreenLeafTopology {
                id: centre_id.child(grandchild_index as u8).unwrap(),
                domain: root[3].compose(local),
            });
        }
        leaves
    }

    fn assert_overlaps_match(leaves: &[ScreenLeafTopology], lods: &[[u32; 3]]) {
        let lines = topology_lines(leaves).unwrap();
        for spans in lines.values() {
            for (index, a) in spans.iter().enumerate() {
                for b in &spans[index + 1..] {
                    if a.start < b.end && b.start < a.end {
                        let absolute_a =
                            a.depth + lods[a.leaf_index][a.edge_index].trailing_zeros() as u8;
                        let absolute_b =
                            b.depth + lods[b.leaf_index][b.edge_index].trailing_zeros() as u8;
                        assert_eq!(absolute_a, absolute_b, "overlapping spans {a:?} / {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn coarse_edges_promote_to_match_multiple_fine_neighbors() {
        let leaves = mixed_depth_partition();
        let result =
            reconcile_screen_leaf_lods(&leaves, &vec![[1; 3]; leaves.len()], 4, 512).unwrap();
        assert!(result.shared_edge_promotions > 0);
        assert_overlaps_match(&leaves, &result.resident);
        for lods in &result.resident {
            let minimum = *lods.iter().min().unwrap();
            let maximum = *lods.iter().max().unwrap();
            assert!(maximum <= 4 * minimum);
        }
        let mut keys = result
            .resident
            .iter()
            .copied()
            .map(|lods| canonical_form(lods).res)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        let mut levels = keys.iter().flatten().copied().collect::<Vec<_>>();
        levels.sort_unstable();
        levels.dedup();
        let atlas = TessellationAtlas::build_for_keys(
            &levels,
            &keys,
            &PatchConfig::default(),
            BuildMode::Hierarchical,
        );
        let work = result.atlas_work(&atlas).unwrap();
        assert_eq!(work.instances, leaves.len() as u64);
        assert!(work.vertices >= work.instances * 3);
        assert!(work.triangles >= work.instances);
    }

    #[test]
    fn screen_demand_and_grading_reach_one_fixed_point() {
        let leaves = mixed_depth_partition();
        let mut requested = vec![[1; 3]; leaves.len()];
        requested[3][1] = 16;
        let result = reconcile_screen_leaf_lods(&leaves, &requested, 2, 512).unwrap();
        assert!(result.iterations >= 2);
        assert!(result.shared_edge_promotions > 0);
        assert!(result.grading_promotions > 0);
        assert_overlaps_match(&leaves, &result.resident);
        for lods in result.resident {
            let minimum = *lods.iter().min().unwrap();
            let maximum = *lods.iter().max().unwrap();
            assert!(maximum <= 2 * minimum);
        }
    }

    #[test]
    fn atlas_overflow_is_reported_instead_of_dropping_a_leaf() {
        let root = ScreenLeafTopology {
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        };
        let deep_domain = (0..10).fold(QBPatchDomain::FULL, |domain, _| {
            domain.compose(QBPatchDomain::FULL.quarter()[0])
        });
        let deep = ScreenLeafTopology {
            id: (0..10).fold(ScreenPatchLeafId::ROOT, |id, _| id.child(0).unwrap()),
            domain: deep_domain,
        };
        let error =
            reconcile_screen_leaf_lods(&[root, deep], &[[1; 3], [512; 3]], 4, 512).unwrap_err();
        assert!(matches!(error, ScreenLeafLodError::AtlasCapExceeded { .. }));
    }

    #[test]
    fn different_face_trees_reconcile_over_welded_source_edges() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let source_topology = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [4, 3, 5]],
        );
        let quarters = QBPatchDomain::FULL.quarter();
        let mut leaves = (0..4u8)
            .map(|child| ScreenMeshLeafTopology {
                source_face: 0,
                id: ScreenPatchLeafId::ROOT.child(child).unwrap(),
                domain: quarters[child as usize],
            })
            .collect::<Vec<_>>();
        leaves.push(ScreenMeshLeafTopology {
            source_face: 1,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let mut requested = vec![[1; 3]; leaves.len()];
        // Face 1 logical edge C is the complete physical edge shared with
        // face 0 logical edge A, which is represented by two depth-1 spans.
        requested[4][2] = 8;

        let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&source_topology).unwrap();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology).unwrap();
        let result = frontier.reconcile_lods(&requested, 4, 512).unwrap();
        assert_eq!(
            result,
            reconcile_screen_mesh_leaf_lods(&leaves, &requested, &source_topology, 4, 512).unwrap()
        );
        assert!(result.shared_edge_promotions >= 2);
        assert_eq!(result.resident[4][2], 8);

        let lines = mesh_topology_lines(&leaves, &topology).unwrap();
        for (_, spans) in &lines {
            for (index, a) in spans.iter().enumerate() {
                for b in &spans[index + 1..] {
                    if a.start < b.end && b.start < a.end {
                        let absolute_a = a.depth
                            + result.resident[a.leaf_index][a.edge_index].trailing_zeros() as u8;
                        let absolute_b = b.depth
                            + result.resident[b.leaf_index][b.edge_index].trailing_zeros() as u8;
                        assert_eq!(absolute_a, absolute_b, "overlapping spans {a:?} / {b:?}");

                        let edge_midpoints = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
                        let absolute_lods = |span: &EdgeSpan| {
                            result.resident[span.leaf_index].map(|lod| f64::from(lod << span.depth))
                        };
                        let density_a = crate::interpolation::tri_edge_density(
                            edge_midpoints[a.edge_index],
                            absolute_lods(a),
                        );
                        let density_b = crate::interpolation::tri_edge_density(
                            edge_midpoints[b.edge_index],
                            absolute_lods(b),
                        );
                        assert_eq!(density_a, density_b);
                    }
                }
            }
        }

        let corner_lods = frontier.rebuild_vertex_lods(&result.resident).unwrap();
        assert_eq!(
            corner_lods,
            rebuild_screen_mesh_leaf_vertex_lods(&leaves, &result.resident, &source_topology)
                .unwrap()
        );
        let mut scratch = ScreenMeshLeafLodScratch::default();
        assert_eq!(
            frontier
                .rebuild_vertex_lods_into(&result.resident, &mut scratch)
                .unwrap(),
            corner_lods,
        );
        let capacities = (
            scratch.absolute_edges.capacity(),
            scratch.vertex_max.capacity(),
            scratch.hanging_lods.capacity(),
            scratch.corner_lods.capacity(),
        );
        frontier
            .rebuild_vertex_lods_into(&result.resident, &mut scratch)
            .unwrap();
        assert_eq!(
            capacities,
            (
                scratch.absolute_edges.capacity(),
                scratch.vertex_max.capacity(),
                scratch.hanging_lods.capacity(),
                scratch.corner_lods.capacity(),
            )
        );
        let global_depth = 1;
        let global_denominator = 1 << global_depth;
        let face_one_edge_c = source_topology.face_half_edges(1)[0];
        let shared_edge = source_topology.canonical_edge(face_one_edge_c);
        let mut hanging_corners = 0;
        for (leaf_index, leaf) in leaves.iter().copied().enumerate() {
            let corners =
                quantized_domain_corners(leaf_index, leaf.id, leaf.domain, global_depth).unwrap();
            for (corner_index, corner) in corners.into_iter().enumerate() {
                if mesh_vertex_key(leaf_index, leaf, corner, global_denominator, &topology)
                    == Ok(MeshVertexKey::SourceEdge {
                        canonical_half_edge: shared_edge,
                        parameter: 1,
                    })
                {
                    assert_eq!(corner_lods[leaf_index][corner_index], 8);
                    hanging_corners += 1;
                }
            }
        }
        // Both boundary leaves and the centre quarter share this physical
        // hanging point on the neighboring root edge.
        assert_eq!(hanging_corners, 3);
    }

    #[test]
    fn absolute_leaf_corner_lods_detect_shifted_value_overflow() {
        let id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let leaves = [ScreenMeshLeafTopology {
            source_face: 0,
            id,
            domain: id.domain().unwrap(),
        }];
        let topology = quilting_mesh::HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        let error = rebuild_screen_mesh_leaf_vertex_lods(&leaves, &[[1 << 31, 1, 1]], &topology)
            .unwrap_err();
        assert_eq!(
            error,
            ScreenLeafLodError::AbsoluteLodOverflow {
                leaf_index: 0,
                edge_index: 0,
            },
        );
    }

    #[test]
    fn mesh_leaf_frontiers_reject_duplicate_and_ancestor_overlap() {
        let child = ScreenPatchLeafId::ROOT.child(2).unwrap();
        let root = ScreenMeshLeafTopology {
            source_face: 0,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        };
        let descendant = ScreenMeshLeafTopology {
            source_face: 0,
            id: child,
            domain: child.domain().unwrap(),
        };
        assert!(matches!(
            validate_screen_mesh_leaf_antichain(&[root, root]),
            Err(ScreenLeafLodError::OverlappingLeaves { .. })
        ));
        assert!(matches!(
            validate_screen_mesh_leaf_antichain(&[root, descendant]),
            Err(ScreenLeafLodError::OverlappingLeaves { .. })
        ));
        let other_face = ScreenMeshLeafTopology {
            source_face: 1,
            ..descendant
        };
        assert_eq!(
            validate_screen_mesh_leaf_antichain(&[root, other_face]),
            Ok(())
        );
    }

    #[test]
    fn hanging_corner_override_matches_the_containing_coarse_edge() {
        let leaf_topology = mixed_depth_partition();
        let leaves = leaf_topology
            .iter()
            .map(|leaf| ScreenMeshLeafTopology {
                source_face: 0,
                id: leaf.id,
                domain: leaf.domain,
            })
            .collect::<Vec<_>>();
        let topology = quilting_mesh::HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        let lines = topology_lines(&leaf_topology).unwrap();
        let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap();
        let edge_corners = [(1usize, 2usize), (0, 2), (0, 1)];
        let mut checked = 0usize;

        for sample in 0..256u32 {
            let requested = (0..leaves.len())
                .map(|leaf| {
                    std::array::from_fn(|edge| {
                        let hash = sample
                            .wrapping_mul(747_796_405)
                            .wrapping_add((leaf as u32 + 1).wrapping_mul(2_891_336_453))
                            .wrapping_add((edge as u32 + 1).wrapping_mul(277_803_737));
                        1 << ((hash >> 29) & 3)
                    })
                })
                .collect::<Vec<[u32; 3]>>();
            let reconciled =
                reconcile_screen_mesh_leaf_lods(&leaves, &requested, &topology, 4, 512).unwrap();
            let values =
                rebuild_screen_mesh_leaf_vertex_lods(&leaves, &reconciled.resident, &topology)
                    .unwrap();

            for (line, spans) in &lines {
                let parameter_axis = (usize::from(line.constant_axis) + 1) % 3;
                for coarse in spans {
                    for fine in spans {
                        if fine.depth <= coarse.depth {
                            continue;
                        }
                        let fine_corners = quantized_domain_corners(
                            fine.leaf_index,
                            leaves[fine.leaf_index].id,
                            leaves[fine.leaf_index].domain,
                            global_depth,
                        )
                        .unwrap();
                        let (first, second) = edge_corners[fine.edge_index];
                        for fine_corner in [first, second] {
                            let parameter = fine_corners[fine_corner][parameter_axis];
                            if coarse.start < parameter && parameter < coarse.end {
                                let coarse_absolute = reconciled.resident[coarse.leaf_index]
                                    [coarse.edge_index]
                                    << coarse.depth;
                                assert_eq!(
                                    values[fine.leaf_index][fine_corner], coarse_absolute,
                                    "sample={sample} coarse={coarse:?} fine={fine:?}",
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn conforming_corner_maxima_cross_exact_welded_attribute_seams() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let topology = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [4, 3, 5]],
        );
        let leaves = [
            ScreenMeshLeafTopology {
                source_face: 0,
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            },
            ScreenMeshLeafTopology {
                source_face: 1,
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            },
        ];
        let corner_lods =
            rebuild_screen_mesh_leaf_vertex_lods(&leaves, &[[2, 1, 16], [1, 1, 2]], &topology)
                .unwrap();

        assert_eq!(corner_lods[0][1], 16);
        assert_eq!(corner_lods[1][1], 16);
        assert_eq!(corner_lods[0][2], 2);
        assert_eq!(corner_lods[1][0], 2);
    }
}
