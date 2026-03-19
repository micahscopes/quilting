use quilting_core::atlas::{BuildMode, TessellationAtlas};
use quilting_core::delaunay::triangulate_2d;
use quilting_core::mesh::TessellationMesh;
use quilting_core::sampling::{tri_patch, tri_patch_jittered, PatchConfig};
use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

const CONFIG: PatchConfig = PatchConfig { k_candidates: 30, seed: 42 };
const MAX_LOD_EXP: u32 = 8; // atlas covers 2^0..2^8

struct CachedAtlas {
    atlas: TessellationAtlas,
    mode: BuildMode,
    sampler: String,
}

fn build_atlas(mode: BuildMode) -> TessellationAtlas {
    let lods: Vec<u32> = (0..=MAX_LOD_EXP).map(|n| 1u32 << n).collect();
    let t0 = Instant::now();
    let atlas = TessellationAtlas::build_with_mode(&lods, &CONFIG, mode);
    eprintln!(
        "Atlas ({:?}): {} patches, {} verts ({:.0}ms)",
        mode, atlas.patches.len(), atlas.positions.len(),
        t0.elapsed().as_secs_f64() * 1000.0,
    );
    atlas
}

fn generate_on_demand(res: [u32; 3], sampler: &str) -> (TessellationMesh, f64, f64) {
    let res_f = [res[0] as f64, res[1] as f64, res[2] as f64];
    let t0 = Instant::now();
    let sample = match sampler {
        "jittered" => tri_patch_jittered(res_f, &CONFIG),
        _ => tri_patch(res_f, &CONFIG),
    };
    let sample_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if sample.positions.len() < 3 {
        return (TessellationMesh::from_2d(vec![], vec![]), sample_ms, 0.0);
    }

    let t0 = Instant::now();
    let tri = triangulate_2d(&sample.positions);
    let tri_ms = t0.elapsed().as_secs_f64() * 1000.0;
    (TessellationMesh::from_2d(tri.positions, tri.triangles), sample_ms, tri_ms)
}

/// Find the best ancestor triple in the atlas and subdivide up to the target.
/// Returns (ancestor_key, n_subdivisions) or None if no ancestor exists.
fn find_ancestor(res: [u32; 3], max_atlas_lod: u32) -> Option<([u32; 3], u32)> {
    let mut ancestor = res;
    let mut n = 0u32;
    loop {
        if ancestor.iter().all(|&r| r <= max_atlas_lod) {
            return Some((ancestor, n));
        }
        // All must be even to halve
        if ancestor.iter().any(|&r| r % 2 != 0 || r == 0) {
            return None;
        }
        ancestor = [ancestor[0] / 2, ancestor[1] / 2, ancestor[2] / 2];
        n += 1;
    }
}

fn serve_patch(
    res: [u32; 3],
    sampler: &str,
    mode: BuildMode,
    max_atlas_lod: u32,
    cache: &RefCell<CachedAtlas>,
) -> (TessellationMesh, f64, f64, String) {
    // Try atlas lookup (exact match)
    if res.iter().all(|&r| r <= max_atlas_lod) {
        let c = cache.borrow();
        let t0 = Instant::now();
        if let Some(mesh) = c.atlas.get_patch(res) {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            return (mesh, 0.0, ms, format!("atlas/{:?}", mode).to_lowercase());
        }
    }

    // Hierarchical: subdivide from nearest ancestor in the atlas
    if mode == BuildMode::Hierarchical {
        if let Some((ancestor, n_sub)) = find_ancestor(res, max_atlas_lod) {
            let c = cache.borrow();
            if let Some(mesh) = c.atlas.get_patch(ancestor) {
                let t0 = Instant::now();
                let (pos, tris) = quilting_core::subdivide::subdivide_n(
                    &mesh.positions, &mesh.triangles, n_sub,
                );
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return (
                    TessellationMesh::from_2d(pos, tris),
                    0.0, ms,
                    format!("subdivide({}x)", n_sub),
                );
            }
        }
    }

    // On-demand generation (unreachable by subdivision)
    eprintln!("fallback: ({},{},{}) mode={:?} — no ancestor in atlas", res[0], res[1], res[2], mode);
    let (m, sm, tm) = generate_on_demand(res, sampler);
    (m, sm, tm, sampler.to_string())
}

fn mesh_to_json(mesh: &TessellationMesh, sample_ms: f64, tri_ms: f64, source: &str) -> String {
    if mesh.positions.is_empty() {
        return format!(
            r#"{{"positions":[],"triangles":[],"vertex_count":0,"triangle_count":0,"sample_ms":0,"tri_ms":0,"source":"{}"}}"#,
            source
        );
    }
    let positions: Vec<f64> = mesh.positions.iter().flat_map(|p| [p[0], p[1]]).collect();
    let triangles: Vec<usize> = mesh.triangles.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();
    format!(
        r#"{{"positions":[{}],"triangles":[{}],"vertex_count":{},"triangle_count":{},"sample_ms":{:.1},"tri_ms":{:.1},"source":"{}"}}"#,
        positions.iter().map(|v| format!("{:.8}", v)).collect::<Vec<_>>().join(","),
        triangles.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        mesh.positions.len(), mesh.triangles.len(),
        sample_ms, tri_ms, source,
    )
}

const HTML: &str = include_str!("web_demo.html");

fn handle_request(request: &str, cache: &RefCell<CachedAtlas>) -> (String, String) {
    if request.starts_with("GET /patch?") {
        let query = request
            .split('?').nth(1).unwrap_or("")
            .split(' ').next().unwrap_or("");

        let mut res = [4u32, 4, 4];
        let mut sampler = "bridson".to_string();
        let mut build_mode = "direct".to_string();

        for pair in query.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let val = kv.next().unwrap_or("");
            match key {
                "a" => res[0] = val.parse().unwrap_or(4),
                "b" => res[1] = val.parse().unwrap_or(4),
                "c" => res[2] = val.parse().unwrap_or(4),
                "method" => sampler = val.to_string(),
                "build" => build_mode = val.to_string(),
                _ => {}
            }
        }

        let mode = match build_mode.as_str() {
            "hierarchical" => BuildMode::Hierarchical,
            _ => BuildMode::Direct,
        };
        let max_atlas_lod = 1u32 << MAX_LOD_EXP;

        // Rebuild atlas if mode or sampler changed
        {
            let c = cache.borrow();
            if c.mode != mode || c.sampler != sampler {
                drop(c);
                let atlas = build_atlas(mode);
                *cache.borrow_mut() = CachedAtlas {
                    atlas,
                    mode,
                    sampler: sampler.clone(),
                };
            }
        }

        let (mesh, sample_ms, tri_ms, source) = serve_patch(
            res, &sampler, mode, max_atlas_lod, &cache,
        );

        let json = mesh_to_json(&mesh, sample_ms, tri_ms, &source);
        (
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n".to_string(),
            json,
        )
    } else if request.starts_with("GET / ") || request.starts_with("GET /index.html") {
        (
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n".to_string(),
            HTML.to_string(),
        )
    } else {
        (
            "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n".to_string(),
            "not found".to_string(),
        )
    }
}

fn main() {
    let cache = RefCell::new(CachedAtlas {
        atlas: build_atlas(BuildMode::Direct),
        mode: BuildMode::Direct,
        sampler: "bridson".to_string(),
    });

    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080u16);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap_or_else(|_| {
        TcpListener::bind(format!("127.0.0.1:{}", port + 1)).expect("failed to bind")
    });

    let addr = listener.local_addr().unwrap();
    eprintln!("Serving demo at http://{}", addr);

    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open")
        .arg(format!("http://{}", addr))
        .spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open")
        .arg(format!("http://{}", addr))
        .spawn();

    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let mut buf = [0u8; 2048];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request = String::from_utf8_lossy(&buf[..n]);
        let (headers, body) = handle_request(&request, &cache);
        let response = format!(
            "{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            headers, body.len(), body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}
