# Composed WebGPU focus presentation — 2026-08-30

## Outcome

The retained WebGPU focus frame can now terminate directly at an acquired
presentation-surface view. The same encoder used by the offscreen oracle:

1. draws unsuppressed resident roots into retained scene/raw-field MRTs;
2. loads both attachments and draws sparse adaptive dyadic replacements;
3. executes the Rust-authored focus schedule; and
4. writes the final scheduled pass to the surface in the same submission.

The surface path does not copy classifier, resident topology, visibility,
raw-field, or image data back to the CPU. Intermediate texture formats remain
backend-owned; only the final focus pipeline is specialized for the configured
surface format.

The adapter conformance fixture now exercises both offscreen and presentation
encoders and requires identical logical draw/pass evidence. Validation errors,
missing depth, mismatched target formats, suspended surfaces, and acquisition
failures remain explicit rather than silently falling back inside the encoder.

This closes the rendering-architecture gap between a complete retained focus
frame and browser presentation. It does **not** enable the live WebGPU
capability predicate yet: image parity against the frozen WebGL2 oracle and a
browser adapter run remain promotion gates.

## Verification

- `quilting-webgpu` compilation and library/native conformance tests.
- Strict crate-local Clippy.
- `quilting-wasm` with `leptos-ui,webgpu-backend` for
  `wasm32-unknown-unknown`.
- No browser tab or user-run server touched.

## Next measured cuts

1. Run the frozen WebGL2/WebGPU browser image comparison on this surface path.
2. Continue migrating browser-owned camera/navigation and URL state into Rust
   reducers plus FRP read models.
3. Consolidate retained resource lifecycles behind the shared render-frame
   boundary before enabling the backend capability.
