fn main() {
    // WASM is built manually: wasm-pack build crates/quilting-wasm --target web --dev --out-dir ../../pkg
    // Auto-build was removed because it caused cargo lock deadlocks when
    // cargo run triggers build.rs which triggers wasm-pack which needs the same lock.
}
