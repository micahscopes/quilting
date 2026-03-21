use std::io::{Read, Write};
use std::net::TcpListener;

const HTML: &str = include_str!("../../../hyperscope.html");
const WORKER_JS: &str = include_str!("../../../hyperscope_worker.js");
const WASM_JS: &str = include_str!("../../../pkg/quilting_wasm.js");
const WASM_BIN: &[u8] = include_bytes!("../../../pkg/quilting_wasm_bg.wasm");

fn handle_request(request: &str) -> Option<(&'static str, &'static str, Option<&'static [u8]>)> {
    if request.starts_with("GET / ") || request.starts_with("GET /index.html") || request.starts_with("GET /hyperscope.html") {
        Some(("text/html", HTML, None))
    } else if request.starts_with("GET /hyperscope_worker.js") {
        Some(("application/javascript", WORKER_JS, None))
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

        // Serve matcap PNGs from disk
        if request.starts_with("GET /matcaps/") {
            let filename = request.split(' ').nth(1).unwrap_or("")
                .trim_start_matches("/matcaps/");
            if filename.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '.') && filename.ends_with(".png") {
                let path = std::path::Path::new("matcaps").join(filename);
                if let Ok(data) = std::fs::read(&path) {
                    let headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        data.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(&data);
                    continue;
                }
            }
        }

        match handle_request(&request) {
            Some((content_type, text_body, binary_body)) => {
                if let Some(bin) = binary_body {
                    let headers = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        content_type, bin.len()
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(bin);
                } else {
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        content_type, text_body.len(), text_body
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            }
            None => {
                let body = "not found";
                let response = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        }
    }
}
