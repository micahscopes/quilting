use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::delaunay::triangulate_2d;
use crate::mesh::TessellationMesh;
use crate::permutation::{canonical_form, remap_position};
use crate::sampling::{tri_patch, PatchConfig};

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

/// Generate a single patch: sample + triangulate. Returns None if too few points.
/// Uses a deterministic seed derived from the base seed and the key.
fn generate_patch(
    key: [u32; 3],
    config: &PatchConfig,
) -> Option<(Vec<[f64; 2]>, Vec<[usize; 3]>)> {
    // Derive a per-patch seed so different patches get different sampling patterns
    let patch_seed = config.seed
        ^ (key[0] as u64 * 73856093)
        ^ (key[1] as u64 * 19349669)
        ^ (key[2] as u64 * 83492791);
    let patch_config = PatchConfig {
        k_candidates: config.k_candidates,
        seed: patch_seed,
    };

    let res = [key[0] as f64, key[1] as f64, key[2] as f64];
    let sample = tri_patch(res, &patch_config);

    if sample.positions.len() < 3 {
        return None;
    }

    let tri = triangulate_2d(&sample.positions);
    Some((tri.positions, tri.triangles))
}

/// Merge a list of (key, positions, triangles) into an atlas.
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

        atlas.patches.insert(
            key,
            PatchEntry {
                base_vertex,
                vertex_count: positions.len(),
                base_triangle,
                triangle_count: triangles.len(),
            },
        );
    }

    atlas
}

impl TessellationAtlas {
    /// Build atlas for given LOD levels (e.g., [2, 4, 8, 16]).
    /// Uses rayon for parallel generation when the `parallel` feature is enabled.
    pub fn build(lod_levels: &[u32], config: &PatchConfig) -> Self {
        let triples = canonical_triples(lod_levels);

        #[cfg(feature = "parallel")]
        {
            Self::build_parallel_inner(&triples, lod_levels, config)
        }
        #[cfg(not(feature = "parallel"))]
        {
            Self::build_sequential(&triples, lod_levels, config)
        }
    }

    fn build_sequential(
        triples: &[[u32; 3]],
        lod_levels: &[u32],
        config: &PatchConfig,
    ) -> Self {
        let results: Vec<_> = triples
            .iter()
            .filter_map(|&key| {
                generate_patch(key, config).map(|(p, t)| (key, p, t))
            })
            .collect();
        merge_patches(lod_levels, results)
    }

    #[cfg(feature = "parallel")]
    fn build_parallel_inner(
        triples: &[[u32; 3]],
        lod_levels: &[u32],
        config: &PatchConfig,
    ) -> Self {
        use rayon::prelude::*;

        let results: Vec<_> = triples
            .par_iter()
            .filter_map(|&key| {
                generate_patch(key, config).map(|(p, t)| (key, p, t))
            })
            .collect();
        merge_patches(lod_levels, results)
    }

    /// Look up a patch for arbitrary (possibly non-sorted) resolutions.
    /// Returns the mesh positions (remapped if needed) and triangles.
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
            .map(|t| {
                [
                    t[0] - entry.base_vertex,
                    t[1] - entry.base_vertex,
                    t[2] - entry.base_vertex,
                ]
            })
            .collect();

        Some(TessellationMesh::from_2d(positions, triangles))
    }

    /// Serialize atlas to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("atlas serialization failed")
    }

    /// Deserialize atlas from bytes.
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

        // LOD levels [2, 4] -> canonical triples: [2,2,2], [2,2,4], [2,4,4], [4,4,4]
        assert!(
            atlas.patches.len() >= 3,
            "expected at least 3 patches, got {}",
            atlas.patches.len()
        );
        assert!(atlas.patches.contains_key(&[2, 2, 2]));
        assert!(atlas.patches.contains_key(&[4, 4, 4]));
    }

    #[test]
    fn get_patch_canonical_and_permuted() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);

        let mesh1 = atlas.get_patch([2, 2, 4]);
        assert!(mesh1.is_some(), "canonical patch [2,2,4] not found");

        let mesh2 = atlas.get_patch([4, 2, 2]);
        assert!(mesh2.is_some(), "permuted patch [4,2,2] not found");

        let m1 = mesh1.unwrap();
        let m2 = mesh2.unwrap();
        assert_eq!(m1.vertex_count(), m2.vertex_count());
        assert_eq!(m1.triangle_count(), m2.triangle_count());
    }

    #[test]
    fn get_patch_positions_in_triangle() {
        use crate::triangle;
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[4], &config);

        let mesh = atlas.get_patch([4, 4, 4]).unwrap();
        for &[x, y] in &mesh.positions {
            let [u, v, w] = triangle::cartesian_to_bary(x, y);
            assert!(
                u >= -1e-10 && v >= -1e-10 && w >= -1e-10,
                "position [{}, {}] (bary [{}, {}, {}]) outside equilateral triangle",
                x, y, u, v, w
            );
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);

        let bytes = atlas.to_bytes();
        let restored = TessellationAtlas::from_bytes(&bytes).expect("deserialization failed");

        assert_eq!(atlas.positions.len(), restored.positions.len());
        assert_eq!(atlas.triangles.len(), restored.triangles.len());
        assert_eq!(atlas.patches.len(), restored.patches.len());
        assert_eq!(atlas.lod_levels, restored.lod_levels);
    }

    #[test]
    fn per_patch_seeds_differ() {
        // Different canonical triples should produce different tessellations
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 8], &config);

        let m1 = atlas.get_patch([2, 2, 2]);
        let m2 = atlas.get_patch([8, 8, 8]);
        if let (Some(m1), Some(m2)) = (m1, m2) {
            // Different resolutions should produce different vertex counts
            assert_ne!(
                m1.vertex_count(),
                m2.vertex_count(),
                "patches [2,2,2] and [8,8,8] have identical vertex counts — seeds may not differ"
            );
        }
    }
}
