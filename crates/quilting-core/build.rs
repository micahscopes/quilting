use std::process::Command;
use std::path::Path;

fn main() {
    // Auto-build WASM when examples are compiled.
    // Only runs if wasm-pack is available and the source changed.
    let wasm_crate = Path::new("../quilting-wasm/src/lib.rs");
    let pkg_dir = Path::new("../../pkg");

    if wasm_crate.exists() {
        println!("cargo:rerun-if-changed=../quilting-wasm/src/lib.rs");
        println!("cargo:rerun-if-changed=../quilting-wasm/Cargo.toml");

        // Only rebuild if pkg doesn't exist or WASM source is newer
        let needs_build = !pkg_dir.join("quilting_wasm_bg.wasm").exists()
            || is_newer(wasm_crate, &pkg_dir.join("quilting_wasm_bg.wasm"));

        if needs_build {
            eprintln!("Building WASM...");
            let status = Command::new("wasm-pack")
                .args(["build", "../quilting-wasm", "--target", "web", "--dev", "--out-dir", "../../pkg"])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .status();

            match status {
                Ok(s) if s.success() => eprintln!("WASM build complete."),
                Ok(s) => eprintln!("wasm-pack exited with: {}", s),
                Err(e) => eprintln!("wasm-pack not found or failed: {} (WASM won't be rebuilt)", e),
            }
        }
    }
}

fn is_newer(a: &Path, b: &Path) -> bool {
    let Ok(ma) = a.metadata() else { return false };
    let Ok(mb) = b.metadata() else { return true };
    let Ok(ta) = ma.modified() else { return false };
    let Ok(tb) = mb.modified() else { return true };
    ta > tb
}
