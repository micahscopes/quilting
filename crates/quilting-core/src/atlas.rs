use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::delaunay::triangulate_2d_constrained;
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
pub fn canonical_triples(lod_levels: &[u32]) -> Vec<[u32; 3]> {
    let mut levels = lod_levels.to_vec();
    levels.sort_unstable();
    levels.dedup();

    let mut triples = Vec::new();
    for (a_index, &a) in levels.iter().enumerate() {
        for (b_index, &b) in levels.iter().enumerate().skip(a_index) {
            for &c in levels.iter().skip(b_index) {
                triples.push([a, b, c]);
            }
        }
    }
    triples
}

/// Canonical triples reachable after enforcing a maximum within-face edge-LOD
/// ratio. Keeping this in the backend-neutral atlas contract prevents startup
/// builders from baking topology that neither WebGL2 nor a future WebGPU
/// classifier is allowed to submit.
pub fn ratio_bounded_canonical_triples(
    lod_levels: &[u32],
    max_edge_ratio: u32,
) -> Vec<[u32; 3]> {
    canonical_triples(lod_levels)
        .into_iter()
        .filter(|key| key[2] <= key[0].saturating_mul(max_edge_ratio))
        .collect()
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
    let tri = triangulate_2d_constrained(&sample.positions, &sample.bary);
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
        insert_patch(&mut atlas, key, positions, triangles);
    }

    atlas
}

fn insert_patch(
    atlas: &mut TessellationAtlas,
    key: [u32; 3],
    positions: Vec<[f64; 2]>,
    triangles: Vec<[usize; 3]>,
) {
    let base_vertex = atlas.positions.len();
    let base_triangle = atlas.triangles.len();
    atlas.positions.extend_from_slice(&positions);
    for triangle in &triangles {
        atlas.triangles.push([
            triangle[0] + base_vertex,
            triangle[1] + base_vertex,
            triangle[2] + base_vertex,
        ]);
    }
    atlas.patches.insert(key, PatchEntry {
        base_vertex,
        vertex_count: positions.len(),
        base_triangle,
        triangle_count: triangles.len(),
    });
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

    /// Build only a caller-selected canonical subset with an explicit topology
    /// policy.
    ///
    /// `Direct` independently blue-noise samples and constrains every requested
    /// patch. `Hierarchical` blue-noise samples only irreducible ancestors and
    /// derives their power-of-two descendants by midpoint subdivision. Keeping
    /// both policies behind one subset API lets runtime and offline tools
    /// measure the quality/startup tradeoff over the exact same reachable keys.
    pub fn build_for_keys(
        lod_levels: &[u32],
        keys: &[[u32; 3]],
        config: &PatchConfig,
        mode: BuildMode,
    ) -> Self {
        match mode {
            BuildMode::Direct => Self::build_direct_for_keys(lod_levels, keys, config),
            BuildMode::Hierarchical => {
                Self::build_hierarchical_for_keys(lod_levels, keys, config)
            }
        }
    }

    /// Build exactly the requested canonical patches and the ancestors needed
    /// to derive them by midpoint subdivision.
    ///
    /// This is the runtime-atlas path: callers can pass the subset reachable
    /// under their LOD grading contract instead of generating the full
    /// combinatorial product of `lod_levels`.
    pub fn build_hierarchical_for_keys(
        lod_levels: &[u32],
        keys: &[[u32; 3]],
        config: &PatchConfig,
    ) -> Self {
        let mut atlas = TessellationAtlas {
            positions: Vec::new(),
            triangles: Vec::new(),
            patches: HashMap::new(),
            lod_levels: lod_levels.to_vec(),
        };
        for &key in keys {
            atlas.ensure_hierarchical_patch(key, config);
        }
        atlas
    }

    /// Ensure one canonical patch exists, recursively deriving it from its
    /// half-resolution ancestor when all three edge resolutions are even.
    pub fn ensure_hierarchical_patch(
        &mut self,
        mut key: [u32; 3],
        config: &PatchConfig,
    ) -> bool {
        key.sort_unstable();
        if self.patches.contains_key(&key) {
            return true;
        }

        if key[0] > 0 && key.iter().all(|resolution| resolution % 2 == 0) {
            let parent = [key[0] / 2, key[1] / 2, key[2] / 2];
            if self.ensure_hierarchical_patch(parent, config) {
                let Some(entry) = self.patches.get(&parent).cloned() else {
                    return false;
                };
                let triangles: Vec<[usize; 3]> = self.triangles
                    [entry.base_triangle..entry.base_triangle + entry.triangle_count]
                    .iter()
                    .map(|triangle| [
                        triangle[0] - entry.base_vertex,
                        triangle[1] - entry.base_vertex,
                        triangle[2] - entry.base_vertex,
                    ])
                    .collect();
                let positions = &self.positions
                    [entry.base_vertex..entry.base_vertex + entry.vertex_count];
                let (positions, triangles) = subdivide::subdivide(positions, &triangles);
                insert_patch(self, key, positions, triangles);
                return true;
            }
        }

        let Some((positions, triangles)) = generate_patch(key, config) else {
            return false;
        };
        insert_patch(self, key, positions, triangles);
        true
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

    fn build_direct_for_keys(
        lod_levels: &[u32],
        keys: &[[u32; 3]],
        config: &PatchConfig,
    ) -> Self {
        let mut canonical_keys: Vec<[u32; 3]> = keys
            .iter()
            .copied()
            .map(|mut key| {
                key.sort_unstable();
                key
            })
            .collect();
        canonical_keys.sort_unstable();
        canonical_keys.dedup();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let results: Vec<_> = canonical_keys
                .par_iter()
                .filter_map(|&key| generate_patch(key, config).map(|(p, t)| (key, p, t)))
                .collect();
            merge_patches(lod_levels, results)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let results: Vec<_> = canonical_keys
                .iter()
                .filter_map(|&key| generate_patch(key, config).map(|(p, t)| (key, p, t)))
                .collect();
            merge_patches(lod_levels, results)
        }
    }

    /// Hierarchical build: derive patches by uniform subdivision where possible.
    ///
    /// (2p, 2q, 2r) is derived from (p, q, r) by splitting every triangle into 4.
    /// This preserves boundary invariants exactly — midpoints of parent boundary
    /// edges become the child's boundary vertices at the correct stitching positions.
    ///
    /// Triples that can't be halved evenly to reach an existing patch (e.g., (1,1,2048)
    /// where 1 can't be halved) are generated directly.
    fn build_hierarchical(lod_levels: &[u32], config: &PatchConfig) -> Self {
        let triples = canonical_triples(lod_levels);
        let min_lod = *lod_levels.iter().min().unwrap_or(&1);

        let mut patch_meshes: HashMap<[u32; 3], (Vec<[f64; 2]>, Vec<[usize; 3]>)> = HashMap::new();

        // Classify: for each triple, find its base (halve all until we can't)
        // and how many subdivisions are needed.
        struct Entry {
            key: [u32; 3],
            base: [u32; 3],
            n_sub: u32,
        }
        let mut entries: Vec<Entry> = triples.iter().map(|&key| {
            let mut base = key;
            let mut n = 0u32;
            while base[0] > min_lod
                && base[0] % 2 == 0
                && base[1] % 2 == 0
                && base[2] % 2 == 0
            {
                base = [base[0] / 2, base[1] / 2, base[2] / 2];
                n += 1;
            }
            Entry { key, base, n_sub: n }
        }).collect();

        // Sort by n_sub so bases are processed first
        entries.sort_by_key(|e| e.n_sub);

        // Collect all base triples that need direct generation
        let base_keys: Vec<[u32; 3]> = entries.iter()
            .filter(|e| e.n_sub == 0)
            .map(|e| e.key)
            .collect();

        // Generate bases (in parallel if available)
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let results: Vec<_> = base_keys
                .par_iter()
                .filter_map(|&key| generate_patch(key, config).map(|m| (key, m)))
                .collect();
            for (key, mesh) in results {
                patch_meshes.insert(key, mesh);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for &key in &base_keys {
                if let Some(mesh) = generate_patch(key, config) {
                    patch_meshes.insert(key, mesh);
                }
            }
        }

        // Derive higher levels by subdivision, level by level
        for entry in &entries {
            if entry.n_sub == 0 || patch_meshes.contains_key(&entry.key) {
                continue;
            }
            // Find the parent (one level down)
            let parent = [entry.key[0] / 2, entry.key[1] / 2, entry.key[2] / 2];
            if let Some((pos, tris)) = patch_meshes.get(&parent) {
                let (new_pos, new_tris) = subdivide::subdivide(pos, tris);
                patch_meshes.insert(entry.key, (new_pos, new_tris));
            } else if let Some((pos, tris)) = patch_meshes.get(&entry.base) {
                // Parent not computed yet but base exists — subdivide from base
                let (new_pos, new_tris) = subdivide::subdivide_n(pos, tris, entry.n_sub);
                patch_meshes.insert(entry.key, (new_pos, new_tris));
            }
        }

        let results: Vec<_> = triples.iter()
            .filter_map(|key| patch_meshes.remove(key).map(|(p, t)| (*key, p, t)))
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

    /// Build only a subset of canonical triples (for parallel atlas construction).
    /// `worker_index` and `num_workers` partition the triples round-robin.
    pub fn build_subset(
        lod_levels: &[u32],
        config: &PatchConfig,
        mode: BuildMode,
        worker_index: usize,
        num_workers: usize,
    ) -> Self {
        let all_triples = canonical_triples(lod_levels);
        let my_triples: Vec<[u32; 3]> = all_triples.into_iter()
            .enumerate()
            .filter(|(i, _)| i % num_workers == worker_index)
            .map(|(_, t)| t)
            .collect();

        // For hierarchical mode, we need base patches available to subdivide.
        // Build a minimal atlas with just the base triples first, then
        // generate our assigned triples.
        match mode {
            BuildMode::Hierarchical => {
                // Build full hierarchical (fast for bases, subdivides the rest)
                let full = Self::build_hierarchical(lod_levels, config);
                // Extract only our assigned triples
                let mut atlas = TessellationAtlas {
                    positions: Vec::new(),
                    triangles: Vec::new(),
                    patches: HashMap::new(),
                    lod_levels: lod_levels.to_vec(),
                };
                for key in &my_triples {
                    if let Some(entry) = full.patches.get(key) {
                        let positions = full.positions
                            [entry.base_vertex..entry.base_vertex + entry.vertex_count].to_vec();
                        let triangles: Vec<[usize; 3]> = full.triangles
                            [entry.base_triangle..entry.base_triangle + entry.triangle_count]
                            .iter()
                            .map(|t| [t[0] - entry.base_vertex, t[1] - entry.base_vertex, t[2] - entry.base_vertex])
                            .collect();
                        let base_vertex = atlas.positions.len();
                        let base_triangle = atlas.triangles.len();
                        atlas.positions.extend_from_slice(&positions);
                        for t in &triangles {
                            atlas.triangles.push([t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex]);
                        }
                        atlas.patches.insert(*key, PatchEntry {
                            base_vertex,
                            vertex_count: positions.len(),
                            base_triangle,
                            triangle_count: triangles.len(),
                        });
                    }
                }
                atlas
            }
            BuildMode::Direct => {
                let results: Vec<_> = my_triples.iter()
                    .filter_map(|&key| generate_patch(key, config).map(|(p, t)| (key, p, t)))
                    .collect();
                merge_patches(lod_levels, results)
            }
        }
    }

    /// Merge another atlas into this one (for combining parallel build results).
    pub fn merge_from(&mut self, other: &TessellationAtlas) {
        for (key, entry) in &other.patches {
            if self.patches.contains_key(key) { continue; }
            let positions = other.positions
                [entry.base_vertex..entry.base_vertex + entry.vertex_count].to_vec();
            let triangles: Vec<[usize; 3]> = other.triangles
                [entry.base_triangle..entry.base_triangle + entry.triangle_count]
                .iter()
                .map(|t| [t[0] - entry.base_vertex, t[1] - entry.base_vertex, t[2] - entry.base_vertex])
                .collect();
            let base_vertex = self.positions.len();
            let base_triangle = self.triangles.len();
            self.positions.extend_from_slice(&positions);
            for t in &triangles {
                self.triangles.push([t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex]);
            }
            self.patches.insert(*key, PatchEntry {
                base_vertex,
                vertex_count: positions.len(),
                base_triangle,
                triangle_count: triangles.len(),
            });
        }
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

    fn assert_boundary_vertex_counts(atlas: &TessellationAtlas, key: [u32; 3]) {
        let patch = atlas.get_patch(key).expect("requested atlas patch");
        let mut counts = [0_usize; 3];
        for &[x, y] in &patch.positions {
            let bary = crate::triangle::cartesian_to_bary(x, y);
            for (edge, coordinate) in bary.into_iter().enumerate() {
                counts[edge] += usize::from(coordinate.abs() <= 1.0e-9);
            }
        }
        assert_eq!(counts, key.map(|resolution| resolution as usize + 1));
    }

    #[test]
    fn build_small_atlas() {
        let config = PatchConfig::default();
        let atlas = TessellationAtlas::build(&[2, 4], &config);
        assert!(atlas.patches.len() >= 3);
        assert!(atlas.patches.contains_key(&[2, 2, 2]));
        assert!(atlas.patches.contains_key(&[4, 4, 4]));
    }

    #[test]
    fn two_to_one_atlas_contains_only_the_three_reachable_families() {
        let lods: Vec<u32> = (0..=7).map(|exponent| 1 << exponent).collect();
        let triples = ratio_bounded_canonical_triples(&lods, 2);

        assert_eq!(triples.len(), 22);
        assert_eq!(triples.first(), Some(&[1, 1, 1]));
        assert_eq!(triples.last(), Some(&[128, 128, 128]));
        assert!(triples.contains(&[64, 64, 128]));
        assert!(triples.contains(&[64, 128, 128]));
        assert!(!triples.contains(&[1, 1, 128]));
        assert!(triples.iter().all(|key| key[2] <= key[0] * 2));
    }

    #[test]
    fn hierarchical_subset_builds_only_reachable_runtime_patches() {
        let lods: Vec<u32> = (0..=7).map(|exponent| 1 << exponent).collect();
        let keys = ratio_bounded_canonical_triples(&lods, 2);
        let atlas = TessellationAtlas::build_hierarchical_for_keys(
            &lods,
            &keys,
            &PatchConfig::default(),
        );

        assert_eq!(keys.len(), 22);
        assert_eq!(atlas.patches.len(), keys.len());
        assert!(keys.iter().all(|key| atlas.patches.contains_key(key)));
    }

    #[test]
    fn hierarchical_subset_matches_full_hierarchical_topology() {
        let config = PatchConfig::default();
        let lods = [1, 2, 4, 8];
        let keys = ratio_bounded_canonical_triples(&lods, 2);
        let subset = TessellationAtlas::build_hierarchical_for_keys(&lods, &keys, &config);
        let full = TessellationAtlas::build_with_mode(&lods, &config, BuildMode::Hierarchical);

        for key in keys {
            let subset_patch = subset.get_patch(key).unwrap();
            let full_patch = full.get_patch(key).unwrap();
            assert_eq!(subset_patch.positions, full_patch.positions, "patch {key:?}");
            assert_eq!(subset_patch.triangles, full_patch.triangles, "patch {key:?}");
        }
    }

    #[test]
    fn direct_subset_matches_full_direct_topology() {
        let config = PatchConfig::default();
        let lods = [1, 2, 4];
        let requested = [[4, 2, 1], [1, 2, 4], [4, 4, 4]];
        let expected = [[1, 2, 4], [4, 4, 4]];
        let subset = TessellationAtlas::build_for_keys(
            &lods,
            &requested,
            &config,
            BuildMode::Direct,
        );
        let full = TessellationAtlas::build_with_mode(&lods, &config, BuildMode::Direct);

        assert_eq!(subset.patches.len(), expected.len());
        for key in expected {
            let subset_patch = subset.get_patch(key).unwrap();
            let full_patch = full.get_patch(key).unwrap();
            assert_eq!(subset_patch.positions, full_patch.positions, "patch {key:?}");
            assert_eq!(subset_patch.triangles, full_patch.triangles, "patch {key:?}");
        }
    }

    #[test]
    fn direct_and_hierarchical_four_to_one_patches_preserve_edge_counts() {
        let config = PatchConfig::default();
        let lods = [1, 2, 4, 8, 16];
        let keys = [[1, 1, 4], [2, 8, 8], [4, 8, 16], [16, 16, 16]];

        for mode in [BuildMode::Direct, BuildMode::Hierarchical] {
            let atlas = TessellationAtlas::build_for_keys(&lods, &keys, &config, mode);
            for key in keys {
                assert_boundary_vertex_counts(&atlas, key);
            }
        }
    }

    #[test]
    fn ensuring_existing_hierarchical_patch_is_idempotent() {
        let config = PatchConfig::default();
        let mut atlas = TessellationAtlas::build_hierarchical_for_keys(
            &[1, 2, 4, 8],
            &[[8, 8, 8]],
            &config,
        );
        let counts = (
            atlas.patches.len(),
            atlas.positions.len(),
            atlas.triangles.len(),
        );

        assert!(atlas.ensure_hierarchical_patch([8, 8, 8], &config));
        assert_eq!(
            counts,
            (
                atlas.patches.len(),
                atlas.positions.len(),
                atlas.triangles.len(),
            ),
        );
    }

    #[test]
    fn lod_one_is_exactly_the_source_triangle() {
        let atlas = TessellationAtlas::build(&[1], &PatchConfig::default());
        let patch = atlas.get_patch([1, 1, 1]).unwrap();
        assert_eq!(patch.vertex_count(), 3);
        assert_eq!(patch.triangle_count(), 1);
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
    fn every_s3_permutation_preserves_exact_edge_subdivisions() {
        use crate::permutation::S3_PERMUTATIONS;
        use crate::triangle;

        let atlas = TessellationAtlas::build(&[2, 4, 8], &PatchConfig::default());
        let canonical = [2u32, 4, 8];
        for (permutation_index, permutation) in S3_PERMUTATIONS.iter().enumerate() {
            let requested = [
                canonical[permutation[0]],
                canonical[permutation[1]],
                canonical[permutation[2]],
            ];
            let patch = atlas.get_patch(requested).unwrap();
            let barycentrics: Vec<[f64; 3]> = patch.positions.iter()
                .map(|position| triangle::cartesian_to_bary(position[0], position[1]))
                .collect();

            for edge in 0..3 {
                let parameter_component = (edge + 2) % 3;
                let mut parameters: Vec<f64> = barycentrics.iter()
                    .filter(|bary| bary[edge].abs() < 1.0e-9)
                    .map(|bary| bary[parameter_component])
                    .collect();
                parameters.sort_by(|a, b| a.total_cmp(b));
                parameters.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);

                let subdivisions = requested[edge] as usize;
                assert_eq!(
                    parameters.len(),
                    subdivisions + 1,
                    "permutation {permutation_index}, edge {edge} requested {} subdivisions",
                    requested[edge],
                );
                for (sample, parameter) in parameters.iter().enumerate() {
                    let expected = sample as f64 / subdivisions as f64;
                    assert!(
                        (*parameter - expected).abs() < 1.0e-9,
                        "permutation {permutation_index}, edge {edge}, sample {sample}: {parameter} != {expected}",
                    );
                }
            }
        }
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
    fn no_corner_spanning_triangles_above_lod1() {
        use crate::triangle;
        let config = PatchConfig::default();
        let lods = &[1, 2, 4, 8];
        let atlas = TessellationAtlas::build(lods, &config);
        let eps = 0.02;

        fn corner_id(bary: [f64; 3], eps: f64) -> Option<usize> {
            if bary[0] > 1.0 - eps && bary[1] < eps && bary[2] < eps { return Some(0); }
            if bary[1] > 1.0 - eps && bary[0] < eps && bary[2] < eps { return Some(1); }
            if bary[2] > 1.0 - eps && bary[0] < eps && bary[1] < eps { return Some(2); }
            None
        }

        // Edge between corners i,j has resolution: the LOD component opposite the third corner
        // Corners are A=0, B=1, C=2. Edge AB is opposite C → key[2]. Etc.
        fn edge_res(c0: usize, c1: usize, key: [u32; 3]) -> u32 {
            let opposite = 3 - c0 - c1; // the corner NOT on this edge
            key[opposite]
        }

        let mut bad = 0;
        for (&key, entry) in &atlas.patches {
            for tri_idx in entry.base_triangle..entry.base_triangle + entry.triangle_count {
                let t = atlas.triangles[tri_idx];
                let barys: Vec<[f64; 3]> = t.iter().map(|&idx| {
                    let p = atlas.positions[idx];
                    triangle::cartesian_to_bary(p[0], p[1])
                }).collect();
                let corners: Vec<Option<usize>> = barys.iter().map(|b| corner_id(*b, eps)).collect();
                let corner_verts: Vec<usize> = corners.iter().filter_map(|c| *c).collect();
                if corner_verts.len() >= 2 {
                    let res = edge_res(corner_verts[0], corner_verts[1], key);
                    if res > 1 {
                        bad += 1;
                        eprintln!("BAD: patch {:?} has triangle spanning corners {:?} (edge res={})",
                            key, corner_verts, res);
                    }
                }
            }
        }
        assert_eq!(bad, 0, "found {} corner-spanning triangles on edges with res > 1", bad);
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
