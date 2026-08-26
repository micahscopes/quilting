use quilting_core::polytope4::{exploded_cell_projection, ProjectedPolytope4};
use quilting_gltf::export::{encode_static_mesh_glb, StaticMeshGlb};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let mut check = false;
    let mut output = PathBuf::from("examples/polytopes");
    for argument in env::args().skip(1) {
        if argument == "--check" {
            check = true;
        } else if output == Path::new("examples/polytopes") {
            output = PathBuf::from(argument);
        } else {
            return Err("usage: export_polytope_glbs [OUTPUT_DIRECTORY] [--check]".to_owned());
        }
    }
    if !check {
        fs::create_dir_all(&output)
            .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    }

    for (kind, filename, name, color) in [
        (
            ProjectedPolytope4::Simplex,
            "4-simplex.glb",
            "Projected 4-simplex",
            [0.95, 0.42, 0.24, 1.0],
        ),
        (
            ProjectedPolytope4::Tesseract,
            "tesseract.glb",
            "Projected tesseract",
            [0.12, 0.68, 0.94, 1.0],
        ),
        (
            ProjectedPolytope4::CrossPolytope,
            "16-cell.glb",
            "Projected 16-cell",
            [0.68, 0.38, 0.94, 1.0],
        ),
    ] {
        let projection = exploded_cell_projection(kind);
        let bytes = encode_static_mesh_glb(StaticMeshGlb {
            name,
            positions: &projection.positions,
            triangles: &projection.faces,
            base_color: color,
        })
        .map_err(|error| format!("cannot encode {name}: {error}"))?;
        let path = output.join(filename);
        if check {
            let resident = fs::read(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if resident != bytes {
                return Err(format!(
                    "{} is stale; regenerate it with export_polytope_glbs",
                    path.display(),
                ));
            }
        } else {
            fs::write(&path, &bytes)
                .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        }
        println!(
            "{}: {} vertices, {} faces, {} bytes{}",
            path.display(),
            projection.positions.len(),
            projection.faces.len(),
            bytes.len(),
            if check { " (exact)" } else { "" },
        );
    }
    Ok(())
}
