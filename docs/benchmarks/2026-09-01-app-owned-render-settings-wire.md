# App-owned render-settings wire projection

Date: 2026-09-01

Route startup and live reducer state now share one application-owned conversion
for renderer-independent settings. `RenderSettingsWire` owns style, resolution,
tessellation density and attenuation, pixel floor, atlas level, grading ratio,
and the complete focus-postprocess packet. `AppRenderSnapshotWire` adds the
decimal-string revision while reusing that conversion field-for-field.

The live DTO deliberately spells its fields instead of using serde flattening.
The first generated-WASM gate demonstrated why: `serde_wasm_bindgen` projected a
flattened nested struct as a JavaScript `Map`, whereas the established ABI and
browser consumers require an ordinary object. The revised implementation
destructures the shared `RenderSettingsWire` conversion into explicit live DTO
fields. This preserves one policy owner and the exact JavaScript container type.

Input remains in `quilting-wasm`: it decodes the browser packet, resolves closed
focus-mode/diagnostic enums, and calls the validating semantic constructors.
This migration changes trusted output only.

The WASM facade decreased from 5,492 to 5,429 lines and the route wire module
from 480 to 418. The application wire module grew from 497 to 627 lines,
including the cross-projection oracle. The net five lines are test evidence;
125 lines of duplicate adapter conversion were removed.

Verification:

- `cargo test -p hyperscope-app --features replay`: 186 passed, including the
  route/live render-settings field oracle.
- `cargo clippy -p hyperscope-app --lib --features wire --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- Generated-WASM app-shadow and route smokes plus focused render-settings and
  Patch Lab smokes passed after the plain-object correction.
- All 40 CPU-only, source, and generated-WASM smoke scripts passed. The three
  process-owning server/build-lifecycle smokes remained intentionally excluded.

No browser, renderer, GPU context, server, or user-owned process was started.
