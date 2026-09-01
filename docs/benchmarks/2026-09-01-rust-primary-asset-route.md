# Rust-authoritative primary asset route packet

Date: 2026-09-01

## Outcome

The startup primary-scene URI, optional animation clip, and playback state now
leave route admission as one typed `RoutePrimaryAssetSettings` packet. The
packet deliberately contains no asset or request ID: `AppStore` still allocates
those identities when the browser adapter asks for the primary scene.

On a Rust-admitted route:

- model acquisition uses the packet URI;
- playback uses the packet boolean rather than another string comparison;
- clip `-1` becomes `None`, while a selected integral clip becomes `Some(u32)`;
- JavaScript `parseInt` remains only in the explicit browser fallback lane.

## Evidence

- `cargo test -p hyperscope-app`: 142 passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- `cargo clippy -p hyperscope-app --lib --no-deps -- -D warnings`: passed.
- development WASM rebuilt with `HYPERSCOPE_WASM_PROFILE=dev` and
  `HYPERSCOPE_WASM_OPT=0`.
- `node scripts/smoke-rust-primary-asset-route.mjs`: passed.
- presentation, asset-request, and animation-clip neighboring smokes passed.
- the extracted browser module parses under `node --check`.
- `git diff --check`: passed.
