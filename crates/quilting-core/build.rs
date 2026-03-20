use std::process::Command;
use std::path::Path;

fn main() {
    let wasm_crate = Path::new("../quilting-wasm/src/lib.rs");
    let pkg_wasm = Path::new("../../pkg/quilting_wasm_bg.wasm");

    if !wasm_crate.exists() {
        return;
    }

    // Track all WASM source files so we rebuild when any of them change.
    // But DON'T rebuild if the binary is already up to date.
    for dir in &["../quilting-wasm/src", "../quilting-mesh/src", "../quilting-spacetime/src"] {
        if Path::new(dir).exists() {
            println!("cargo:rerun-if-changed={}", dir);
        }
    }
    println!("cargo:rerun-if-changed=../quilting-wasm/Cargo.toml");

    // Skip if pkg already exists and is newer than all sources
    if pkg_wasm.exists() && !any_source_newer(pkg_wasm) {
        return;
    }

    eprintln!("Building WASM (--dev)...");
    let status = Command::new("wasm-pack")
        .args(["build", "../quilting-wasm", "--target", "web", "--dev", "--out-dir", "../../pkg"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status();

    match status {
        Ok(s) if s.success() => eprintln!("WASM build complete."),
        Ok(s) => eprintln!("wasm-pack exited with: {}", s),
        Err(e) => eprintln!("wasm-pack not found: {} (skip WASM build)", e),
    }
}

fn any_source_newer(target: &Path) -> bool {
    let Ok(target_meta) = target.metadata() else { return true };
    let Ok(target_time) = target_meta.modified() else { return true };

    let source_dirs = [
        "../quilting-wasm/src",
        "../quilting-mesh/src",
        "../quilting-spacetime/src",
        "../quilting-core/src",
    ];

    for dir in &source_dirs {
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

    // Also check Cargo.toml files
    for toml in &["../quilting-wasm/Cargo.toml", "../quilting-mesh/Cargo.toml"] {
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
