use quilting_core::atlas::TessellationAtlas;
use quilting_core::delaunay::triangulate_2d;
use quilting_core::sampling::{tri_patch, PatchConfig};
use std::io::{Read, Write};
use std::net::TcpListener;

fn generate_patch_json(res_a: f64, res_b: f64, res_c: f64, seed: u64) -> String {
    let config = PatchConfig {
        k_candidates: 30,
        seed,
    };
    let sample = tri_patch([res_a, res_b, res_c], &config);

    if sample.positions.len() < 3 {
        return r#"{"positions":[],"triangles":[],"vertex_count":0,"triangle_count":0}"#.to_string();
    }

    let tri = triangulate_2d(&sample.positions);

    let positions: Vec<f64> = tri.positions.iter().flat_map(|p| [p[0], p[1]]).collect();
    let triangles: Vec<usize> = tri.triangles.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();

    format!(
        r#"{{"positions":[{}],"triangles":[{}],"vertex_count":{},"triangle_count":{}}}"#,
        positions
            .iter()
            .map(|v| format!("{:.8}", v))
            .collect::<Vec<_>>()
            .join(","),
        triangles
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        tri.positions.len(),
        tri.triangles.len(),
    )
}

const HTML: &str = include_str!("web_demo.html");

fn handle_request(request: &str) -> (String, String) {
    if request.starts_with("GET /patch?") {
        // Parse query string
        let query = request
            .split('?')
            .nth(1)
            .unwrap_or("")
            .split(' ')
            .next()
            .unwrap_or("");

        let mut res_a = 4.0;
        let mut res_b = 4.0;
        let mut res_c = 4.0;
        let mut seed = 42u64;

        for pair in query.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let val = kv.next().unwrap_or("");
            match key {
                "a" => res_a = val.parse().unwrap_or(4.0),
                "b" => res_b = val.parse().unwrap_or(4.0),
                "c" => res_c = val.parse().unwrap_or(4.0),
                "seed" => seed = val.parse().unwrap_or(42),
                _ => {}
            }
        }

        let json = generate_patch_json(res_a, res_b, res_c, seed);
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
    // Print atlas stats
    let config = PatchConfig::default();
    let lods: Vec<u32> = (1..=4).map(|n| 1u32 << n).collect();
    eprintln!("Building atlas for LODs: {:?}", lods);
    let t0 = std::time::Instant::now();
    let atlas = TessellationAtlas::build(&lods, &config);
    let elapsed = t0.elapsed();
    eprintln!(
        "Atlas: {} patches, {} vertices, {} triangles ({:.1}ms)",
        atlas.patches.len(),
        atlas.positions.len(),
        atlas.triangles.len(),
        elapsed.as_secs_f64() * 1000.0
    );

    // Start HTTP server
    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080u16);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap_or_else(|_| {
        // Try next port if busy
        TcpListener::bind(format!("127.0.0.1:{}", port + 1)).expect("failed to bind")
    });

    let addr = listener.local_addr().unwrap();
    eprintln!("Serving demo at http://{}", addr);

    // Try to open browser
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
        let (headers, body) = handle_request(&request);
        let response = format!(
            "{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            headers,
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}
