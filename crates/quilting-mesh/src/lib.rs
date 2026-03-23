/// Half-edge mesh data structure for efficient adjacency queries.
///
/// Built once from an indexed triangle list, provides O(1) lookups for:
/// - Vertex -> outgoing half-edges
/// - Half-edge -> twin, next, prev, face, vertex
/// - Face -> one half-edge (iterate to get all 3)
/// - Edge -> both half-edges (or one if boundary)

use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HalfEdgeId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceId(pub u32);

fn pack_twin(idx: u32) -> Option<NonZeroU32> {
    Some(NonZeroU32::new(idx + 1).unwrap())
}

pub fn unpack_twin(val: Option<NonZeroU32>) -> Option<u32> {
    val.map(|v| v.get() - 1)
}

#[derive(Debug, Clone)]
pub struct HalfEdge {
    pub vertex: u32,
    pub twin: Option<NonZeroU32>,
    pub next: u32,
    pub prev: u32,
    pub face: u32,
}

#[derive(Debug, Clone)]
pub struct HalfEdgeMesh {
    pub half_edges: Vec<HalfEdge>,
    pub vertex_half_edge: Vec<Option<u32>>,
    pub face_half_edge: Vec<u32>,
    pub num_vertices: u32,
    pub num_faces: u32,
}

impl HalfEdgeMesh {
    pub fn from_triangles(num_vertices: u32, faces: &[[u32; 3]]) -> Self {
        let num_faces = faces.len() as u32;
        let num_half_edges = (num_faces * 3) as usize;

        let mut half_edges = Vec::with_capacity(num_half_edges);
        let mut vertex_half_edge: Vec<Option<u32>> = vec![None; num_vertices as usize];
        let mut face_half_edge = Vec::with_capacity(num_faces as usize);

        for (fi, face) in faces.iter().enumerate() {
            let base = (fi * 3) as u32;
            face_half_edge.push(base);

            for i in 0u32..3 {
                let he_idx = base + i;
                let next_i = (i + 1) % 3;
                let prev_i = (i + 2) % 3;

                half_edges.push(HalfEdge {
                    vertex: face[next_i as usize],
                    twin: None,
                    next: base + next_i,
                    prev: base + prev_i,
                    face: fi as u32,
                });

                let from_v = face[i as usize] as usize;
                if vertex_half_edge[from_v].is_none() {
                    vertex_half_edge[from_v] = Some(he_idx);
                }
            }
        }

        use rustc_hash::FxHashMap;
        let mut edge_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();

        for he_idx in 0..half_edges.len() {
            let face_idx = half_edges[he_idx].face;
            let from = faces[face_idx as usize][(he_idx % 3) as usize];
            let to = half_edges[he_idx].vertex;

            if let Some(&twin_idx) = edge_map.get(&(to, from)) {
                half_edges[he_idx].twin = pack_twin(twin_idx);
                half_edges[twin_idx as usize].twin = pack_twin(he_idx as u32);
            }
            edge_map.insert((from, to), he_idx as u32);
        }

        Self {
            half_edges,
            vertex_half_edge,
            face_half_edge,
            num_vertices,
            num_faces,
        }
    }

    pub fn face_vertices(&self, face: u32) -> [u32; 3] {
        let he0 = self.face_half_edge[face as usize] as usize;
        let he1 = self.half_edges[he0].next as usize;
        let v0 = self.half_edges[self.half_edges[he0].prev as usize].vertex;
        let v1 = self.half_edges[he0].vertex;
        let v2 = self.half_edges[he1].vertex;
        [v0, v1, v2]
    }

    pub fn face_half_edges(&self, face: u32) -> [u32; 3] {
        let he0 = self.face_half_edge[face as usize];
        let he1 = self.half_edges[he0 as usize].next;
        let he2 = self.half_edges[he1 as usize].next;
        [he0, he1, he2]
    }

    pub fn adjacent_face(&self, half_edge: u32) -> Option<u32> {
        unpack_twin(self.half_edges[half_edge as usize].twin)
            .map(|t| self.half_edges[t as usize].face)
    }

    pub fn edge_vertices(&self, half_edge: u32) -> (u32, u32) {
        let he = &self.half_edges[half_edge as usize];
        let from = self.half_edges[he.prev as usize].vertex;
        (from, he.vertex)
    }

    pub fn is_boundary_edge(&self, half_edge: u32) -> bool {
        self.half_edges[half_edge as usize].twin.is_none()
    }

    pub fn vertex_outgoing(&self, vertex: u32) -> Vec<u32> {
        let start = match self.vertex_half_edge[vertex as usize] {
            Some(he) => he,
            None => return vec![],
        };

        let mut result = vec![];
        let mut current = start;
        loop {
            result.push(current);
            let twin = unpack_twin(self.half_edges[current as usize].twin);
            match twin {
                None => break,
                Some(t) => {
                    current = self.half_edges[t as usize].next;
                    if current == start { break; }
                }
            }
        }
        result
    }

    pub fn num_boundary_edges(&self) -> usize {
        self.half_edges.iter().filter(|he| he.twin.is_none()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_edge_is_20_bytes() {
        assert_eq!(std::mem::size_of::<HalfEdge>(), 20);
    }

    #[test]
    fn single_triangle() {
        let mesh = HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);
        assert_eq!(mesh.num_faces, 1);
        assert_eq!(mesh.half_edges.len(), 3);
        assert_eq!(mesh.face_vertices(0), [0, 1, 2]);
        assert_eq!(mesh.num_boundary_edges(), 3);
    }

    #[test]
    fn two_triangles_shared_edge() {
        let mesh = HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        assert_eq!(mesh.num_faces, 2);
        assert_eq!(mesh.half_edges.len(), 6);

        let shared = mesh.half_edges.iter().filter(|he| he.twin.is_some()).count();
        assert_eq!(shared, 2, "shared edge should have 2 twin half-edges");
        assert_eq!(mesh.num_boundary_edges(), 4);
    }

    #[test]
    fn cube() {
        let faces = vec![
            [0,1,2],[0,2,3], [5,4,7],[5,7,6], [4,0,3],[4,3,7],
            [1,5,6],[1,6,2], [3,2,6],[3,6,7], [4,5,1],[4,1,0],
        ];
        let mesh = HalfEdgeMesh::from_triangles(8, &faces);
        assert_eq!(mesh.num_faces, 12);
        assert_eq!(mesh.half_edges.len(), 36);
        assert_eq!(mesh.num_boundary_edges(), 0);
    }

    #[test]
    fn adjacency() {
        let faces = vec![[0,1,2],[0,2,3],[0,3,1]];
        let mesh = HalfEdgeMesh::from_triangles(4, &faces);

        let outgoing = mesh.vertex_outgoing(0);
        assert_eq!(outgoing.len(), 3, "vertex 0 should have 3 outgoing edges");
    }
}
