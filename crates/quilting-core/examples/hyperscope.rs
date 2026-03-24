use std::io::{Read, Write};
use std::net::TcpListener;

const HTML: &str = include_str!("../../../hyperscope.html");
const PROTOTYPE_HTML: &str = include_str!("../../../hyperscope-prototype.html");
const PROTOTYPE_WORKER_JS: &str = include_str!("../../../hyperscope-prototype_worker.js");
const WASM_JS: &str = include_str!("../../../pkg/quilting_wasm.js");
const WASM_BIN: &[u8] = include_bytes!("../../../pkg/quilting_wasm_bg.wasm");

fn handle_request(request: &str) -> Option<(&'static str, &'static str, Option<&'static [u8]>)> {
    if request.starts_with("GET / ") || request.starts_with("GET /index.html") || request.starts_with("GET /hyperscope.html") {
        Some(("text/html", HTML, None))
    } else if request.starts_with("GET /prototype") || request.starts_with("GET /hyperscope-prototype.html") {
        Some(("text/html", PROTOTYPE_HTML, None))
    } else if request.starts_with("GET /hyperscope-prototype_worker.js") {
        Some(("application/javascript", PROTOTYPE_WORKER_JS, None))
    } else if request.starts_with("GET /pkg/quilting_wasm.js") {
        Some(("application/javascript", WASM_JS, None))
    } else if request.starts_with("GET /pkg/quilting_wasm_bg.wasm") {
        Some(("application/wasm", "", Some(WASM_BIN)))
    } else {
        None
    }
}

fn main() {
    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8093u16);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap_or_else(|_| {
        TcpListener::bind(format!("127.0.0.1:{}", port + 1)).expect("failed to bind")
    });

    let addr = listener.local_addr().unwrap();
    eprintln!("Hyperscope demo at http://{}", addr);

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
        let mut buf = [0u8; 4096];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request = String::from_utf8_lossy(&buf[..n]);

        // Serve static assets from disk (matcaps, env maps)
        let serve_from_disk = |prefix: &str, dir: &str, mime: &str| -> Option<Vec<u8>> {
            if !request.starts_with(&format!("GET /{}/", prefix)) { return None; }
            let filename = request.split(' ').nth(1)?
                .trim_start_matches(&format!("/{}/", prefix));
            if !filename.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
                return None;
            }
            let path = std::path::Path::new(dir).join(filename);
            let data = std::fs::read(&path).ok()?;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
                mime, data.len()
            );
            let mut resp = headers.into_bytes();
            resp.extend_from_slice(&data);
            Some(resp)
        };

        // Serve .glb files from current directory
        let serve_glb = || -> Option<Vec<u8>> {
            let path = request.split(' ').nth(1)?;
            let path = path.trim_start_matches('/');
            if !path.ends_with(".glb") && !path.ends_with(".gltf") { return None; }
            if !path.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') { return None; }
            let data = std::fs::read(path).ok()?;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: model/gltf-binary\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
                data.len()
            );
            let mut resp = headers.into_bytes();
            resp.extend_from_slice(&data);
            Some(resp)
        };

        if let Some(resp) = serve_glb()
            .or_else(|| serve_from_disk("matcaps", "matcaps", "image/png"))
            .or_else(|| serve_from_disk("envmaps", "envmaps", "application/octet-stream"))
        {
            let _ = stream.write_all(&resp);
            continue;
        }

        match handle_request(&request) {
            Some((content_type, text_body, binary_body)) => {
                if let Some(bin) = binary_body {
                    let headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
                        content_type, bin.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(bin);
                } else {
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n{}",
                        content_type, text_body.len(), text_body
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            }
            None => {
                let body = "not found";
                let response = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        }
    }
}
