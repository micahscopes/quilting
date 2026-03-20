/// Showcase example: serves the comprehensive quilting demo over HTTP.
/// All rendering happens in the browser via WASM + WebGL2.
///
/// Run: cargo run --example showcase
/// Then open http://127.0.0.1:8090 in your browser.

use std::io::{Read, Write};
use std::net::TcpListener;

const SHOWCASE_HTML: &str = include_str!("../../../examples/showcase/index.html");
const SHOWCASE_WORKER: &str = include_str!("../../../examples/showcase/worker.js");
const WASM_JS: &str = include_str!("../../../pkg/quilting_wasm.js");
const WASM_BIN: &[u8] = include_bytes!("../../../pkg/quilting_wasm_bg.wasm");

fn handle_request(request: &str) -> Option<(&'static str, Vec<u8>)> {
    if request.starts_with("GET / ") || request.starts_with("GET /index.html") {
        Some(("text/html", SHOWCASE_HTML.as_bytes().to_vec()))
    } else if request.starts_with("GET /worker.js") {
        Some(("application/javascript", SHOWCASE_WORKER.as_bytes().to_vec()))
    } else if request.starts_with("GET /pkg/quilting_wasm.js") {
        Some(("application/javascript", WASM_JS.as_bytes().to_vec()))
    } else if request.starts_with("GET /pkg/quilting_wasm_bg.wasm") {
        Some(("application/wasm", WASM_BIN.to_vec()))
    } else {
        None
    }
}

fn main() {
    let port = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8090u16);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap_or_else(|_| {
        TcpListener::bind(format!("127.0.0.1:{}", port + 1)).expect("failed to bind")
    });

    let addr = listener.local_addr().unwrap();
    eprintln!("Showcase demo at http://{}", addr);

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

        if let Some((content_type, body)) = handle_request(&request) {
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                content_type, body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            let _ = stream.write_all(&body);
        } else {
            let body = b"not found";
            let headers = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            let _ = stream.write_all(body);
        }
    }
}
