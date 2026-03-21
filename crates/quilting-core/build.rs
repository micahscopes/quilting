use std::process::Command;
use std::path::Path;

fn main() {
    // Don't run wasm-pack when we're being compiled FOR wasm — that causes infinite recursion.
    if std::env::var("TARGET").map_or(false, |t| t.contains("wasm")) {
        return;
    }

    let pkg_wasm = Path::new("../../pkg/quilting_wasm_bg.wasm");

    let source_dirs = [
        "../quilting-wasm/src",
        "../quilting-mesh/src",
        "../quilting-spacetime/src",
        "src",  // quilting-core's own source (atlas.rs, evaluate.rs, etc.)
    ];
    let cargo_tomls = [
        "../quilting-wasm/Cargo.toml",
        "../quilting-mesh/Cargo.toml",
    ];

    for dir in &source_dirs {
        if Path::new(dir).exists() {
            println!("cargo:rerun-if-changed={}", dir);
        }
    }
    for toml in &cargo_tomls {
        println!("cargo:rerun-if-changed={}", toml);
    }

    if pkg_wasm.exists() && !any_source_newer(pkg_wasm, &source_dirs, &cargo_tomls) {
        return;
    }

    eprintln!("Building WASM (release, opt-level 2)...");
    // Use a separate target dir to avoid cargo lock deadlock —
    // the parent cargo process holds the lock on the main target dir.
    let status = Command::new("wasm-pack")
        .args([
            "build", "../quilting-wasm",
            "--target", "web",
            "--out-dir", "../../pkg",
        ])
        .env("CARGO_TARGET_DIR", "/tmp/quilting-wasm-target")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status();

    match status {
        Ok(s) if s.success() => eprintln!("WASM build complete."),
        Ok(s) => eprintln!("wasm-pack exited with: {}", s),
        Err(e) => eprintln!("wasm-pack not found: {} (skip WASM build)", e),
    }
}

fn any_source_newer(target: &Path, dirs: &[&str], tomls: &[&str]) -> bool {
    let Ok(target_meta) = target.metadata() else { return true };
    let Ok(target_time) = target_meta.modified() else { return true };

    for dir in dirs {
        let dir_path = Path::new(dir);
        if !dir_path.exists() { continue; }
        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        if mtime > target_time {
                            return true;
                        }
                    }
                }
            }
        }
    }

    for toml in tomls {
        let p = Path::new(toml);
        if let Ok(meta) = p.metadata() {
            if let Ok(mtime) = meta.modified() {
                if mtime > target_time {
                    return true;
                }
            }
        }
    }

    false
}
