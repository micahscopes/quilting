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

    /// Hierarchical build: generate base patches at min LOD, then derive
    /// higher-resolution patches by adaptive subdivision using the density
    /// function. Each patch starts from its min-resolution uniform base and
    /// refines until the local density is met everywhere.
    fn build_hierarchical(lod_levels: &[u32], config: &PatchConfig) -> Self {
        use crate::interpolation::tri_edge_weight;
        use crate::triangle;

        let triples = canonical_triples(lod_levels);
        let _min_lod = *lod_levels.iter().min().unwrap_or(&1);

        // For each triple, generate a base mesh at the minimum edge resolution,
        // then adaptively subdivide to match the target density.
        let generate_hierarchical = |key: [u32; 3]| -> Option<(Vec<[f64; 2]>, Vec<[usize; 3]>)> {
            let base_res = key[0]; // sorted, so key[0] is the minimum

            // Generate base mesh at uniform min resolution
            let base_key = [base_res, base_res, base_res];
            let (base_pos, base_tris) = generate_patch(base_key, config)?;

            if key == base_key {
                // Uniform — no adaptive refinement needed
                return Some((base_pos, base_tris));
            }

            // Adaptive refinement with the target density function
            let res = [key[0] as f64, key[1] as f64, key[2] as f64];
            let density_fn = |p: [f64; 2]| -> f64 {
                let bary = triangle::cartesian_to_bary(p[0], p[1]);
                tri_edge_weight(bary, res)
            };

            // Max subdivisions needed = log2(max_res / min_res)
            let max_ratio = key[2] as f64 / key[0] as f64;
            let max_iters = (max_ratio.log2().ceil() as usize).max(1);

            let (pos, tris) = subdivide::subdivide_adaptive(
                &base_pos, &base_tris, &density_fn, 1.2, max_iters,
            );
            Some((pos, tris))
        };

        #[cfg(feature = "parallel")]
        let results = {
            use rayon::prelude::*;
            triples
                .par_iter()
                .filter_map(|&key| {
                    generate_hierarchical(key).map(|(p, t)| (key, p, t))
                })
                .collect::<Vec<_>>()
        };
        #[cfg(not(feature = "parallel"))]
        let results = {
            triples
                .iter()
                .filter_map(|&key| {
                    generate_hierarchical(key).map(|(p, t)| (key, p, t))
                })
                .collect::<Vec<_>>()
        };

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
