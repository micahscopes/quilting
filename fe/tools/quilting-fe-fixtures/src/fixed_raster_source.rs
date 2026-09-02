//! Deterministic expansion of one checked atlas patch into Fe-authored topology.

use std::fmt::{self, Write};

use crate::{Artifact, AtlasPatch};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    MissingPatch(usize),
    EmptyPatch(usize),
    ArithmeticOverflow,
    VertexOutsidePatch { patch: usize, vertex: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPatch(patch) => write!(formatter, "artifact has no patch {patch}"),
            Self::EmptyPatch(patch) => write!(formatter, "patch {patch} has no triangles"),
            Self::ArithmeticOverflow => formatter.write_str("fixed-raster expansion overflow"),
            Self::VertexOutsidePatch { patch, vertex } => {
                write!(
                    formatter,
                    "triangle vertex {vertex} is outside patch {patch}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

fn patch_range(patch: AtlasPatch) -> Result<(usize, usize, usize, usize), Error> {
    let first_vertex =
        usize::try_from(patch.first_vertex).map_err(|_| Error::ArithmeticOverflow)?;
    let vertex_count =
        usize::try_from(patch.vertex_count).map_err(|_| Error::ArithmeticOverflow)?;
    let first_triangle =
        usize::try_from(patch.first_triangle).map_err(|_| Error::ArithmeticOverflow)?;
    let triangle_count =
        usize::try_from(patch.triangle_count).map_err(|_| Error::ArithmeticOverflow)?;
    Ok((first_vertex, vertex_count, first_triangle, triangle_count))
}

fn write_barycentric_lookup(
    source: &mut String,
    barycentrics: &[[f32; 3]],
    first_index: usize,
    indent: usize,
) {
    let padding = "    ".repeat(indent);
    if let [[a, b, c]] = barycentrics {
        writeln!(source, "{padding}bary3f({a:?}, {b:?}, {c:?})").unwrap();
        return;
    }

    let middle = barycentrics.len() / 2;
    let pivot = first_index + middle;
    writeln!(source, "{padding}if vertex_index < {pivot} {{").unwrap();
    write_barycentric_lookup(source, &barycentrics[..middle], first_index, indent + 1);
    writeln!(source, "{padding}}} else {{").unwrap();
    write_barycentric_lookup(source, &barycentrics[middle..], pivot, indent + 1);
    writeln!(source, "{padding}}}").unwrap();
}

/// Render one artifact patch as a fixed, non-indexed Fe vertex stream.
///
/// # Errors
///
/// Returns an error for a missing/empty patch, arithmetic overflow, or a
/// triangle that escapes the selected patch's checked vertex range.
pub fn render(
    artifact: &Artifact,
    fixture_label: &str,
    patch_index: usize,
) -> Result<String, Error> {
    let patch = artifact
        .patches
        .get(patch_index)
        .copied()
        .ok_or(Error::MissingPatch(patch_index))?;
    let (first_vertex, vertex_count, first_triangle, triangle_count) = patch_range(patch)?;
    if triangle_count == 0 {
        return Err(Error::EmptyPatch(patch_index));
    }
    let last_vertex = first_vertex
        .checked_add(vertex_count)
        .ok_or(Error::ArithmeticOverflow)?;
    let last_triangle = first_triangle
        .checked_add(triangle_count)
        .ok_or(Error::ArithmeticOverflow)?;
    let triangles = artifact
        .triangles
        .get(first_triangle..last_triangle)
        .ok_or(Error::ArithmeticOverflow)?;
    let expanded_count = triangle_count
        .checked_mul(3)
        .ok_or(Error::ArithmeticOverflow)?;
    let mut expanded = Vec::with_capacity(expanded_count);
    for triangle in triangles {
        for vertex in triangle.indices {
            let vertex_index = usize::try_from(vertex).map_err(|_| Error::ArithmeticOverflow)?;
            if !(first_vertex..last_vertex).contains(&vertex_index) {
                return Err(Error::VertexOutsidePatch {
                    patch: patch_index,
                    vertex,
                });
            }
            expanded.push(vertex_index);
        }
    }

    let label = fixture_label.replace(['\n', '\r'], " ");
    let mut source = String::new();
    writeln!(
        source,
        "//! @generated from fixtures/classic-quilting/v1/{label}."
    )
    .unwrap();
    source
        .push_str("//! Regenerate with the classic-quilting artifact tool; do not hand-edit.\n\n");
    source.push_str("use quilting_domain::{Bary3F, bary3f}\n");
    source.push_str("use std::webgpu::TriangleList\n\n");
    writeln!(
        source,
        "pub const SOURCE_VERTEX_COUNT: usize = {vertex_count}"
    )
    .unwrap();
    writeln!(
        source,
        "pub const SOURCE_TRIANGLE_COUNT: usize = {triangle_count}"
    )
    .unwrap();
    writeln!(
        source,
        "pub const EXPANDED_VERTEX_COUNT: usize = {expanded_count}\n"
    )
    .unwrap();
    writeln!(
        source,
        "pub type FixedTopology = TriangleList<{expanded_count}>\n"
    )
    .unwrap();
    writeln!(
        source,
        "/// Checked indexed topology beginning with `{:?}`, expanded in draw order.",
        triangles[0].indices
    )
    .unwrap();
    source.push_str("pub fn expanded_barycentric(_ vertex_index: u32) -> Bary3F {\n");
    let barycentrics = expanded
        .into_iter()
        .map(|vertex_index| artifact.vertices[vertex_index].barycentric)
        .collect::<Vec<_>>();
    write_barycentric_lookup(&mut source, &barycentrics, 0, 1);
    source.push_str("}\n");
    source.push_str("\n/// Triangle-local barycentrics for a non-indexed triangle-list stream.\n");
    source.push_str("pub fn local_barycentric(_ vertex_index: u32) -> Bary3F {\n");
    source.push_str("    let corner = vertex_index % 3\n");
    source.push_str("    if corner == 0 {\n");
    source.push_str("        bary3f(1.0, 0.0, 0.0)\n");
    source.push_str("    } else if corner == 1 {\n");
    source.push_str("        bary3f(0.0, 1.0, 0.0)\n");
    source.push_str("    } else {\n");
    source.push_str("        bary3f(0.0, 0.0, 1.0)\n");
    source.push_str("    }\n}\n");
    Ok(source)
}

#[cfg(test)]
mod tests {
    #[test]
    fn committed_wire_topology_is_exactly_regenerated() {
        let artifact = crate::decode(include_bytes!(
            "../../../fixtures/classic-quilting/v1/direct-seed42-k2-4-8.cqa"
        ))
        .expect("checked asymmetric wire fixture");
        let generated =
            super::render(&artifact, "direct-seed42-k2-4-8.cqa", 0).expect("expand wire topology");
        assert_eq!(
            generated,
            include_str!(
                "../../../ingots/demos/classic_quilting_fixed_raster/src/fixed_topology.fe"
            )
        );
    }

    #[test]
    fn wire_topology_lookup_depth_stays_bounded() {
        let source = include_str!(
            "../../../ingots/demos/classic_quilting_fixed_raster/src/fixed_topology.fe"
        );
        let lookup = source
            .split_once("pub fn expanded_barycentric")
            .expect("expanded lookup")
            .1
            .split_once("pub fn local_barycentric")
            .expect("local lookup follows expanded lookup")
            .0;
        let maximum_indent = lookup
            .lines()
            .map(|line| line.len() - line.trim_start_matches(' ').len())
            .max()
            .unwrap_or_default();

        assert!(
            maximum_indent <= 8 * 4,
            "generated selector nesting regressed beyond the browser-safe balanced tree"
        );
    }
}
