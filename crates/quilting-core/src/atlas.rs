use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::delaunay::triangulate_2d;
use crate::mesh::TessellationMesh;
use crate::permutation::{canonical_form, remap_position};
use crate::sampling::{tri_patch, PatchConfig};
use crate::subdivide;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchEntry {
    pub base_vertex: usize,
    pub vertex_count: usize,
    pub base_triangle: usize,
    pub triangle_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TessellationAtlas {
    pub positions: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
    pub patches: HashMap<[u32; 3], PatchEntry>,
    pub lod_levels: Vec<u32>,
}

/// How to generate patches for the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildMode {
    /// Generate every patch independently via Bridson sampling + Delaunay.
    Direct,
    /// Generate base patches (where min resolution = min LOD) via sampling,
    /// then derive higher levels by midpoint subdivision. Much faster for
    /// large LOD ranges since subdivision is O(N) with no spatial index.
    Hierarchical,
}

/// Enumerate all canonical (sorted) triples from the given LOD levels.
fn canonical_triples(lod_levels: &[u32]) -> Vec<[u32; 3]> {
    let mut triples = Vec::new();
    for &a in lod_levels {
        for &b in lod_levels {
            for &c in lod_levels {
                let mut key = [a, b, c];
                key.sort();
                if !triples.contains(&key) {
                    triples.push(key);
                }
            }
        }
    }
    triples.sort();
    triples
}

/// Generate a single patch via sampling + triangulation.
fn generate_patch(
    key: [u32; 3],
    config: &PatchConfig,
) -> Option<(Vec<[f64; 2]>, Vec<[usize; 3]>)> {
    let res = [key[0] as f64, key[1] as f64, key[2] as f64];
    let sample = tri_patch(res, config);
    if sample.positions.len() < 3 {
        return None;
    }
    let tri = triangulate_2d(&sample.positions);
    Some((tri.positions, tri.triangles))
}

/// Merge patches into a flat atlas.
fn merge_patches(
    lod_levels: &[u32],
    patch_results: Vec<([u32; 3], Vec<[f64; 2]>, Vec<[usize; 3]>)>,
) -> TessellationAtlas {
    let mut atlas = TessellationAtlas {
        positions: Vec::new(),
        triangles: Vec::new(),
        patches: HashMap::new(),
        lod_levels: lod_levels.to_vec(),
    };

    for (key, positions, triangles) in patch_results {
        let base_vertex = atlas.positions.len();
        let base_triangle = atlas.triangles.len();
        atlas.positions.extend_from_slice(&positions);
        for t in &triangles {
            atlas.triangles.push([
                t[0] + base_vertex,
                t[1] + base_vertex,
                t[2] + base_vertex,
            ]);
        }
        atlas.patches.insert(key, PatchEntry {
            base_vertex,
            vertex_count: positions.len(),
            base_triangle,
            triangle_count: triangles.len(),
        });
    }

    atlas
}

impl TessellationAtlas {
    /// Build atlas with the specified mode.
    pub fn build_with_mode(
        lod_levels: &[u32],
        config: &PatchConfig,
        mode: BuildMode,
    ) -> Self {
        match mode {
            BuildMode::Direct => Self::build_direct(lod_levels, config),
            BuildMode::Hierarchical => Self::build_hierarchical(lod_levels, config),
        }
    }

    /// Build atlas using direct generation (backward compatible).
    pub fn build(lod_levels: &[u32], config: &PatchConfig) -> Self {
        Self::build_direct(lod_levels, config)
    }

    fn build_direct(lod_levels: &[u32], config: &PatchConfig) -> Self {
        let triples = canonical_triples(lod_levels);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let results: Vec<_> = triples
                .par_iter()
                .filter_map(|&key| generate_patch(key, config).map(|(p, t)| (key, p, t)))
                .collect();
            merge_patches(lod_levels, results)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let results: Vec<_> = triples
                .iter()
                .filter_map(|&key| generate_patch(key, config).map(|(p, t)| (key, p, t)))
                .collect();
            merge_patches(lod_levels, results)
        }
    }

    /// Hierarchical build: generate base patches, derive higher levels by subdivision.
    ///
    /// A canonical triple (p, q, r) where all values are divisible by 2 is derived
    /// from (p/2, q/2, r/2) by one round of midpoint subdivision. This recurses
    /// down to the base level where at least one value equals the minimum LOD.
    fn build_hierarchical(lod_levels: &[u32], config: &PatchConfig) -> Self {
        let triples = canonical_triples(lod_levels);
        let min_lod = *lod_levels.iter().min().unwrap_or(&1);

        // Sort triples by their minimum value so we process bases first
        let mut sorted = triples.clone();
        sorted.sort_by_key(|k| k[0]);

        // Store individual patch meshes for subdivision lookup
        let mut patch_meshes: HashMap<[u32; 3], (Vec<[f64; 2]>, Vec<[usize; 3]>)> = HashMap::new();

        // Collect base triples (min value = min_lod) and derived triples
        let mut base_triples = Vec::new();
        let mut derived: Vec<([u32; 3], [u32; 3], u32)> = Vec::new(); // (key, parent_key, subdivisions)

        for &key in &sorted {
            // Find how many times we can halve all three values
            let mut parent = key;
            let mut n_sub = 0u32;
            while parent[0] > min_lod
                && parent[0] % 2 == 0
                && parent[1] % 2 == 0
                && parent[2] % 2 == 0
            {
                parent = [parent[0] / 2, parent[1] / 2, parent[2] / 2];
                n_sub += 1;
            }

            if n_sub == 0 {
                base_triples.push(key);
            } else {
                derived.push((key, parent, n_sub));
            }
        }

        // Generate base patches (these need actual sampling)
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let base_results: Vec<_> = base_triples
                .par_iter()
                .filter_map(|&key| generate_patch(key, config).map(|mesh| (key, mesh)))
                .collect();
            for (key, (pos, tris)) in base_results {
                patch_meshes.insert(key, (pos, tris));
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for &key in &base_triples {
                if let Some((pos, tris)) = generate_patch(key, config) {
                    patch_meshes.insert(key, (pos, tris));
                }
            }
        }

        // Derive higher-level patches by subdivision
        // Sort derived by n_sub to process in order (1 subdivision first, then 2, etc.)
        let mut derived = derived;
        derived.sort_by_key(|&(_, _, n)| n);

        for (key, parent_key, n_sub) in &derived {
            if let Some((parent_pos, parent_tris)) = patch_meshes.get(parent_key) {
                let (pos, tris) = subdivide::subdivide_n(parent_pos, parent_tris, *n_sub);
                patch_meshes.insert(*key, (pos, tris));
            }
        }

        // Merge all patches into the atlas
        let results: Vec<_> = sorted
            .iter()
            .filter_map(|key| {
                patch_meshes.remove(key).map(|(pos, tris)| (*key, pos, tris))
            })
            .collect();

        merge_patches(lod_levels, results)
    }

    /// Look up a patch for arbitrary (possibly non-sorted) resolutions.
    pub fn get_patch(&self, res: [u32; 3]) -> Option<TessellationMesh> {
        let key = canonical_form(res);
        let entry = self.patches.get(&key.res)?;

        let positions: Vec<[f64; 2]> = self.positions
            [entry.base_vertex..entry.base_vertex + entry.vertex_count]
            .iter()
            .map(|&p| remap_position(key.perm_index, p))
            .collect();

        let triangles: Vec<[usize; 3]> = self.triangles
            [entry.base_triangle..entry.base_triangle + entry.triangle_count]
            .iter()
            .map(|t| [
                t[0] - entry.base_vertex,
                t[1] - entry.base_vertex,
                t[2] - entry.base_vertex,
            ])
            .collect();

        Some(TessellationMesh::from_2d(positions, triangles))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("atlas serialization failed")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_small_atlas() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);
        assert!(atlas.patches.len() >= 3);
        assert!(atlas.patches.contains_key(&[2, 2, 2]));
        assert!(atlas.patches.contains_key(&[4, 4, 4]));
    }

    #[test]
    fn build_hierarchical_matches_direct() {
        let config = PatchConfig::default();
        let lods = &[1, 2, 4];
        let direct = TessellationAtlas::build_with_mode(lods, &config, BuildMode::Direct);
        let hier = TessellationAtlas::build_with_mode(lods, &config, BuildMode::Hierarchical);

        // Same set of patches
        assert_eq!(direct.patches.len(), hier.patches.len(),
            "direct has {} patches, hierarchical has {}", direct.patches.len(), hier.patches.len());
        for key in direct.patches.keys() {
            assert!(hier.patches.contains_key(key), "hierarchical missing patch {:?}", key);
        }
    }

    #[test]
    fn hierarchical_subdivision_produces_more_points() {
        let config = PatchConfig::default();
        let lods = &[1, 2, 4, 8];
        let hier = TessellationAtlas::build_with_mode(lods, &config, BuildMode::Hierarchical);

        // (8,8,8) should have more vertices than (4,4,4) — it's a subdivision
        let m4 = hier.get_patch([4, 4, 4]).unwrap();
        let m8 = hier.get_patch([8, 8, 8]).unwrap();
        assert!(
            m8.vertex_count() > m4.vertex_count(),
            "(8,8,8) has {} verts, (4,4,4) has {} — expected more",
            m8.vertex_count(), m4.vertex_count()
        );
    }

    #[test]
    fn get_patch_canonical_and_permuted() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);
        let mesh1 = atlas.get_patch([2, 2, 4]);
        let mesh2 = atlas.get_patch([4, 2, 2]);
        assert!(mesh1.is_some() && mesh2.is_some());
        assert_eq!(mesh1.unwrap().vertex_count(), mesh2.unwrap().vertex_count());
    }

    #[test]
    fn get_patch_positions_in_triangle() {
        use crate::triangle;
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[4], &config);
        let mesh = atlas.get_patch([4, 4, 4]).unwrap();
        for &[x, y] in &mesh.positions {
            let [u, v, w] = triangle::cartesian_to_bary(x, y);
            assert!(u >= -1e-10 && v >= -1e-10 && w >= -1e-10,
                "[{}, {}] outside triangle", x, y);
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);
        let bytes = atlas.to_bytes();
        let restored = TessellationAtlas::from_bytes(&bytes).unwrap();
        assert_eq!(atlas.positions.len(), restored.positions.len());
        assert_eq!(atlas.patches.len(), restored.patches.len());
    }
}
