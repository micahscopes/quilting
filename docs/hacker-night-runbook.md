# Hyperscope hacker-night runbook

This is the exact offline-friendly path for the checked-in six-cue Quilting
presentation. Rehearse these commands and this URL on the presentation machine;
do not rely on a development tab with cached assets.

## 1. Build and preflight

From the repository root:

```sh
trunk build --release
cargo run -p hyperscope-app --features replay --bin hyperscope-replay -- --check
cargo run -p hyperscope-app --features replay --bin hyperscope-replay -- --navigation --check
cargo run -p hyperscope-app --features replay --bin hyperscope-replay -- --orchestration --check
cargo run -p hyperscape --bin hyperscope-preflight
node --test tests/*.test.mjs
node scripts/smoke-hyperscope-presentation.mjs
node scripts/smoke-hyperscope-app-shadow.mjs
node scripts/smoke-surface-walk.mjs
node scripts/smoke-hyperscope-route-shadow.mjs
node scripts/smoke-render-shadow.mjs
```

The ordinary preflight must print `PASS`. It validates the Rust presentation
document, byte-for-byte manifest freshness, every presentation GLB, the runtime
JS/WASM pairs, environment maps, licenses, and generated Trunk
bootstrap. Trunk also embeds a deterministic receipt for the Rust, shader,
HTML/module, and copied-asset inputs; preflight recomputes it and rejects a
coherent but stale `dist/`. It reports the matching fingerprints and checked
bundle size. The generated-WASM smoke verifies start, cue deep-link, malformed
cue, unknown-cue, application projection, and camera/focus transition parity
without needing a GPU or browser. It also loads synthetic authored and ordinary
GLBs through the real WASM parser to prove that durable node IDs remain dense
and asset-scoped while ordinary high-node-count files pay no null-table clone.
The surface-walk smoke additionally checks the Rust scale/speed/near-plane
policy against the shared live-page helpers, then runs deterministic animated
contact, pitch-recapture, orientation-retention, reset, and invalid-input
traces without a renderer. It also calls the composed attach/step exports
before renderer initialization and verifies that their generated TypeScript
return type remains `ComposedSurfaceWalkResult`; the browser-independent test
suite checks that partitioned and single-step re-anchor clocks finish on the
same exact virtual-time endpoint.

The replay checks must print
`PASS fnv1a-128-json:4d8598faf9db62e8500d49d94ead89ed`,
`PASS fnv1a-128-json:4b6f0b82cf471af7af17b99ed37317d4`, and
`PASS fnv1a-128-json:2cb74a642b3d4fc40b4eda777addb833`. The first executes
the complete six-cue semantic walkthrough. The second exercises every current
navigation action, including focus/inversion, camera and surface transitions,
stable selection anchoring, clicked source/output pivot projection, complete
perspective-lens edits, explicit free-tangent/point-target mode, and an atomic
rejected input. The third covers
asset supersession/cancellation/completion, stale and failed effects, presence
ordering/expiry, authored revision admission, and rejected wire input. All run
through `hyperscope-app` independently of browser timing, input adapters, and
the renderer. A mismatch means reducer, presentation, navigation,
orchestration, or trace behavior changed and must be reviewed; the fingerprints
are deterministic regression oracles, not cryptographic signatures.

For a public downloadable archive, use the stricter gate:

```sh
cargo run -p hyperscape --bin hyperscope-preflight -- --strict
```

`--strict` currently fails intentionally: the horse has a known CC BY-NC-SA
3.0 license that requires an explicit noncommercial mixed-license release
decision (or replacement/exclusion). It also catches accidental untracked files
copied from `local-glbs/`. The four matcap looks are now analytic WGSL profiles,
so they add no separately licensed image payload; preflight also rejects stale
copies of the retired PNGs left by a non-clean build. See
[`ASSET_ATTRIBUTION.md`](../ASSET_ATTRIBUTION.md). Do not publish `dist/` as an
MIT/Apache-only or commercial-use asset bundle until the strict gate is green.

For the noncommercial hacker-night presentation, make the policy choice
explicit and stage a bundle that excludes `local-glbs/` and any stale retired
matcaps:

```sh
scripts/stage-hyperscope-release.sh dist-release
cargo run -p hyperscape --bin hyperscope-preflight -- \
  --dist dist-release \
  --distribution-policy noncommercial-mixed \
  --strict
```

The staging command refuses to overwrite an existing directory. Remove or
rename an older `dist-release/` yourself before preparing a new archive. Under
the default `permissive-only` policy, the horse remains a hard warning.

The preflight has `--json` for an archival or scripted report. It is
filesystem-only and cannot certify browser GPU/HID behavior.

## 2. Serve the built directory

The release directory has no network runtime dependency. Serve it over local
HTTP rather than opening `index.html` as a `file:` URL:

```sh
python3 -m http.server 8888 --directory dist
```

Open:

```text
http://127.0.0.1:8888/?presentation=1&glb=horse.glb
```

The presentation does not require the optional SpaceMouse. WebHID permission
is browser- and operating-system-specific, so grant and test it before the
event if it will be used. The core mouse/keyboard demo must remain usable when
WebHID is unavailable.

## 3. Rehearse the story

Advance with Right Arrow, Page Down, or the on-screen right arrow. Reverse with
Left Arrow or Page Up. Verify all six cues in order:

After advancing, copy the URL and reload it once. Its `cue` UUID must restore
the same numbered cue rather than returning to cue one.

1. **Curved patches, continuously animated** — visible patch boundaries on the
   animated horse.
2. **One canonical topology** — the wire view shows the atlas topology stamped
   across faces.
3. **A shared edge chooses one resolution** — the coarse 64-pixel threshold
   should show a stable two-level LOD split without open seams.
4. **Curvature and normals are evaluated analytically** — the horse remains
   recognizable in the normals view.
5. **The Möbius map folds into each patch's weights** — inversion and the
   red/blue stretch field complete without a pole warning.
6. **Scenes stay composable** — PBR returns and both the animated horse and
   Blender-authored scene are visible; diagnostics report 2 assets and 4,264
   faces.

Keep the sidebar available during rehearsal so a presenter can inspect or
recover state, then collapse it for the talk if desired. The text card remains
interactive rather than locking the camera.

## 4. Interaction demonstrations

- **Fly/orbit:** mouse and keyboard controls remain the baseline. A SpaceMouse
  is optional.
- **Select/focus:** left-click a rendered object. Focus and inversion share the
  selected object's smoothly interpolated sphere.
- **Surface walk:** right-click a surface to attach, then use `W`, `A`, `S`, and
  `D`. The walker follows animated QB geometry and detaches safely when contact
  becomes invalid.
- **Load a fallback model:** drag a local `.glb` onto the canvas or use the file
  picker. This is a recovery/demo path, not part of the deterministic cue deck.
- **Compare grading policy:** open
  `?lab=triangle&labfield=edges&laba=1&labb=6&labc=6`, then repeat with
  `&lodratio=4`. The page reload is intentional: diagnostics must report the
  active policy, zero shared-edge mismatches, the reconciled request, and the
  resulting resident triangle count from the matching cached atlas.

## 5. Browser smoke check

Use a fresh browser tab after the release build and verify:

- no `phase wasm failed`, atlas-upload failure, GLB parse error, or uncaught
  exception appears in the console;
- the horse animates continuously and LOD does not flash to a stale low level;
- picking/selection tint and surface attachment use the visible object;
- moving through all six cues leaves exactly one requested visualization mode
  active;
- the final cue reports no failed or unsupported required asset;
- reloading with the network disabled still succeeds through local HTTP.

For the focus-authority migration gate, open a separate disposable tab at
`?animate=0&fuzzy=1&fmode=3&navshadow=1&appshadow=1&rendershadow=1`.
Verify that the incumbent navigation snapshot and `AppStore` snapshot agree on
focus enablement, shell coordinate, angular aperture, sphere, lens, and aim
policy. Toggle Fuzzy off and on and move the focus slider: each synchronous UI
burst should produce one focus synchronization, no application mismatch, and
no render-shadow mismatch. A focus-only route must remain active even when
spherical inversion is disabled. While settled, `frameCalls` should track the
page frame counter one-for-one, `frameSnapshotCalls` must remain unchanged, and
`frameErrors` must stay zero. A mapped authored selection should increment
`selectionTransitionFrames` and `rendererFocusPacketComparisons` without
incrementing either mismatch counter. Close the disposable tab after the check.

The preload `integrity` warning emitted by Chrome for unsupported preload
destinations is informational; a renderer initialization or asset error is not.

For migration diagnostics, add `walkimpl=shadow`. The rendered camera and
attachment still come from the JavaScript oracle, while the composed Rust
candidate reports samples, topology/camera drift, and its last packet at
`globalThis.__hyperscopeSurfaceWalkRustShadow`. `walkimpl=rust` is now a real
opt-in authority mode: Rust owns the contact response, camera, and re-anchor
transition while the legacy walker remains a same-input rollback diagnostic.
The release URL deliberately stays on the soaked `js` default for the talk;
use `walkimpl=rust` only when explicitly demonstrating the migration.

## 6. Recovery during the talk

1. Reload the exact URL above. Presentation state deterministically returns to
   cue one.
2. If an interaction leaves the camera disoriented, reload rather than trying
   to reconstruct an improvised view on stage.
3. If WebHID fails, continue with mouse/keyboard; do not change operating-system
   permissions during the talk.
4. If a dropped model fails, return to the bundled presentation—the checked-in
   assets are the rehearsed path.
5. Keep a known-good copy of `dist/` made after a passing preflight. Do not run a
   development rebuild during the presentation.

## 7. Release boundary

The presentation bundle and the source-code release are different artifacts.
The code can be released under the repository licenses while third-party model
and image policy is resolved. The horse has documented CC BY-NC-SA 3.0 terms;
it is not part of the MIT/Apache grant and cannot be included in a commercial
or permissive-only bundle. Exclude `local-glbs/` and any test downloads from
every archive. A public bundle under the intended
policy is ready only when the strict preflight has no errors or warnings and
the browser smoke check passes on the target machine.
