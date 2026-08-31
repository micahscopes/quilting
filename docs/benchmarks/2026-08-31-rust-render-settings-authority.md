# Rust render-settings authority cutover

Date: 2026-08-31

## Outcome

`renderstateimpl=rust` is the canonical Hyperscope route. The Leptos render
controls dispatch complete typed settings through `AppStore`; JavaScript only
projects the committed packet into the existing renderer and URL adapters.
`renderstateimpl=js` and `renderstateimpl=shadow` remain explicit rollback and
comparison lanes.

Focus-compositor inspection is no longer disguised as a scene render style.
Replay 0.26 adds the backend-neutral `Composite`, `Weight`, `DistanceField`, and
`Firmness` diagnostic value. Historical `mode=fz-weight|fz-jfa|fz-firmness`
URLs normalize to PBR plus `fdebug=1|2|3`; replay 0.25 accepts only Composite.
The semantic name `DistanceField` deliberately does not promise JFA as the
renderer implementation.

## Live evidence

The isolated Chromium audit exercised the same edit sequence in explicit Rust
and JavaScript lanes: tessellation density, focus blur radius, focus Weight
diagnostic, canonical URL publication, and exact restoration. Rust committed
the typed projection with zero comparison mismatches, errors, or rollbacks;
the JavaScript lane performed no Rust dispatch or comparison.

The paused WebGPU presentation audit then exercised PBR/Composite to Weight and
back. Weight retired WebGPU with the explicit `unsupported-mode` reason and
incumbent WebGL2 presentation. Returning to Composite presented a fresh WebGPU
frame and classified a fresh device-resident LOD epoch; diagnostics reported
the same effective authority as the runtime. A controlled paused WebGL2/WebGPU
capture also showed identical horse orientation and background in both the
ordinary and sphere-reflected focus-composited views.

## Verification

- `cargo test -p hyperscope-app --all-features`: 148 passed.
- `cargo test -p hyperscope-web --all-features`: 41 passed (38 library, 3 bin).
- `cargo check -p quilting-wasm --tests --target wasm32-unknown-unknown --features leptos-ui,webgpu-backend` passed.
- Route, app-shadow, and render-settings source smokes passed.
- Live Rust, JavaScript rollback, WebGPU policy, and paused presentation recovery audits passed.

The presentation recovery is event-driven: the actual transition to a
presentable WebGPU surface requests the missing complete LOD epoch. It does not
add polling or readback to the ordinary frame path.
