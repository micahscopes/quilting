use quilting_core::atlas::TessellationAtlas;
use quilting_core::delaunay::triangulate_2d;
use quilting_core::mesh::TessellationMesh;
use quilting_core::sampling::{tri_patch, tri_patch_jittered, PatchConfig};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

/// Pre-built atlas + on-demand generation for high-res patches
struct PatchServer {
    atlas: TessellationAtlas,
    max_atlas_lod: u32,
    config: PatchConfig,
}

impl PatchServer {
    fn build(max_lod_exp: u32) -> Self {
        let config = PatchConfig { k_candidates: 30, seed: 42 };
        let lods: Vec<u32> = (0..=max_lod_exp).map(|n| 1u32 << n).collect();

        let t0 = Instant::now();
        let atlas = TessellationAtlas::build(&lods, &config);
        let elapsed = t0.elapsed();

        eprintln!(
            "Atlas built: {} patches, {} verts, {} tris ({:.1}ms)",
            atlas.patches.len(),
            atlas.positions.len(),
            atlas.triangles.len(),
            elapsed.as_secs_f64() * 1000.0,
        );

        Self {
            atlas,
            max_atlas_lod: 1 << max_lod_exp,
            config,
        }
    }

    fn get_patch(&self, res: [u32; 3], method: &str) -> (TessellationMesh, f64, f64) {
        // Try atlas first (for resolutions within the pre-built range)
        if res.iter().all(|&r| r <= self.max_atlas_lod) {
            let t0 = Instant::now();
            if let Some(mesh) = self.atlas.get_patch(res) {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                return (mesh, 0.0, ms); // sample_ms=0 (precomputed), tri_ms=lookup time
            }
        }

        // Generate on demand
        let res_f = [res[0] as f64, res[1] as f64, res[2] as f64];
        let t0 = Instant::now();
        let sample = match method {
            "jittered" => tri_patch_jittered(res_f, &self.config),
            _ => tri_patch(res_f, &self.config),
        };
        let sample_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if sample.positions.len() < 3 {
            return (
                TessellationMesh::from_2d(vec![], vec![]),
                sample_ms,
                0.0,
            );
        }

        let t0 = Instant::now();
        let tri = triangulate_2d(&sample.positions);
        let tri_ms = t0.elapsed().as_secs_f64() * 1000.0;

        (
            TessellationMesh::from_2d(tri.positions, tri.triangles),
            sample_ms,
            tri_ms,
        )
    }
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
        mesh.positions.len(),
        mesh.triangles.len(),
        sample_ms,
        tri_ms,
        source,
    )
}

const HTML: &str = include_str!("web_demo.html");

fn handle_request(request: &str, server: &PatchServer) -> (String, String) {
    if request.starts_with("GET /patch?") {
        let query = request
            .split('?').nth(1).unwrap_or("")
            .split(' ').next().unwrap_or("");

        let mut res_a = 4u32;
        let mut res_b = 4u32;
        let mut res_c = 4u32;
        let mut method = "bridson";

        for pair in query.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let val = kv.next().unwrap_or("");
            match key {
                "a" => res_a = val.parse().unwrap_or(4),
                "b" => res_b = val.parse().unwrap_or(4),
                "c" => res_c = val.parse().unwrap_or(4),
                "method" => method = if val == "jittered" { "jittered" } else { "bridson" },
                _ => {}
            }
        }

        let res = [res_a, res_b, res_c];
        let (mesh, sample_ms, tri_ms) = server.get_patch(res, method);
        let source = if res.iter().all(|&r| r <= server.max_atlas_lod) {
            "atlas"
        } else {
            method
        };
        let json = mesh_to_json(&mesh, sample_ms, tri_ms, source);

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
    // Pre-build atlas for LODs 2^0 through 2^7 (1..128)
    // With parallel feature, this uses rayon and takes ~25ms in release
    let server = PatchServer::build(7);

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
        let (headers, body) = handle_request(&request, &server);
        let response = format!(
            "{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            headers, body.len(), body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}
