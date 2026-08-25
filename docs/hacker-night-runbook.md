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
node scripts/smoke-local-peer-relay.mjs
```

With Blender installed, also run
`node scripts/smoke-blender-browser-relay.mjs`. It must report one
Blender-authored edit, one browser presence frame, three received frames, the
exact browser presence sequence `18446744073709551614`, and projected
translation `[3,4,5]`. Blender must project the remote camera, inversion sphere,
and bound selection into at least 166 transient overlay segments without
creating an object.

### Authority and rollback check

An ordinary deck uses Rust for presentation, asset effects, packed-scene
extraction, and canonical routes. Before the talk, verify:

- `__hyperscopePresentation` is ready under `hyperscope-app` authority;
- `__hyperscopeAppShadowDiagnostics` reports Rust asset effects, no failed
  loads, no mismatches, and no frame errors;
- `__hyperscopeSceneExtraction` is authoritative with no semantic mismatch or
  fallback; and
- `__hyperscopeRouteShadowDiagnostics` is authoritative, has byte-identical
  browser/Rust queries, and has no mismatch or fallback write.

The default `rust` values stay out of canonical links. For recovery, append
`presentimpl=js`, `assetimpl=js`, `sceneimpl=js`, or `routeimpl=js`; an explicit
rollback must remain in the synchronized URL. The corresponding `shadow` value
compares implementations without granting Rust authority. Detailed parity
fields and ownership boundaries live in the
[release architecture](hacker-night-release-architecture.md) and
[presentation contract](hyperscope-presentation.md).

The ordinary preflight must print `PASS`, matching source/build fingerprints,
and the checked bundle size. It rejects a coherent but stale `dist/` as well as
missing presentation/runtime assets. The three replay commands must print:

```text
PASS fnv1a-128-json:ea73662b9602640b487943ba2d9880f7
PASS fnv1a-128-json:632127d93ad2417225544b3d14819302
PASS fnv1a-128-json:9bc89c319883e25f7e91d001656d924b
```

They are deterministic regression oracles for the cue walkthrough,
navigation, and orchestration/history paths; any mismatch must be reviewed.

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

If `trunk serve` is currently using `dist/`, build to a separate directory so
its development live-reload client cannot race into the release bundle:

```sh
trunk build --release --dist dist-build-release
scripts/stage-hyperscope-release.sh dist-release dist-build-release
```

The staging script rejects any source HTML containing Trunk's live-reload
WebSocket client. Offline preflight enforces the same invariant independently.

The staging command refuses to overwrite an existing directory. Remove or
rename an older `dist-release/` yourself before preparing a new archive. Under
the default `permissive-only` policy, the horse remains a hard warning.

The preflight has `--json` for an archival or scripted report. It is
filesystem-only and cannot certify browser GPU/HID behavior.

## 2. Serve the exact staged directory

The release directory has no network runtime dependency. Serve it over local
HTTP rather than opening `index.html` as a `file:` URL:

```sh
python3 -m http.server 8888 --directory dist-release
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
   Blender-authored scene are visible; diagnostics report 2 assets and 4,252
   faces.

Keep the sidebar available during rehearsal so a presenter can inspect or
recover state, then collapse it for the talk if desired. The text card remains
interactive rather than locking the camera.

### Stage-ready choreography

Prepare three views before the audience arrives rather than configuring them
during the talk:

1. Open the six-cue deck at the known URL from section 2 and leave it on cue
   one.
2. Open the Patch Lab comparison from section 4 in a second browser tab. Keep
   both the 2:1 URL and its `lodratio=4` variant in browser history.
3. Open the authored demo in Blender, keep the **Hyperscape** sidebar visible,
   and connect the browser and Blender to the relay using the steps below.

The short talk path is:

1. Advance through cues one to five without leaving the canvas: animated patch
   boundaries, reusable atlas topology, shared-edge LOD, analytic normals, then
   the conformal stretch field.
2. Advance to cue six and show that the horse and Blender neighborhood remain
   distinct authored assets in one renderer scene.
3. Move one bound object in Blender. Its corresponding Hyperscope object should
   move without reloading the GLB or replacing the presentation.
4. Move the Hyperscope camera or select an authored object. Blender should show
   the transient remote camera, focus/inversion sphere, and selection bounds
   without creating or saving Blender objects.
5. Return to Hyperscope for one interaction: select, invert, or attach to a
   surface and walk. Use only the interaction rehearsed on the presentation
   machine.
6. If time permits, switch to the prepared Patch Lab tab and compare the 2:1
   and 4:1 resident triangle counts.

Do not improvise relay setup, HID permissions, or a development rebuild while
speaking. If live sync is unavailable, continue with cue six as the composed
scene finale; the checked-in scene demonstrates the same asset and identity
boundary without the optional carrier.

### Connect the live Blender finale

Use a disposable loopback-only token chosen before the event. This explicit
example keeps the rehearsal repeatable; replace it if the machine is shared:

```sh
cargo run -p hyperscope-web --features local-peer-relay \
  --bin hyperscope-local-peer-relay -- \
  --token quilting-hacker-night-local
```

The relay must report `127.0.0.1:42117`, the chosen bearer token, the
presentation origin, and `delivery only: no persistence, repair, or projection
authority`.

In Hyperscope's **Local Blender Peer (opt-in)** section:

1. Leave the URL as `http://127.0.0.1:42117`.
2. Enter the same token and choose **Connect**.
3. Confirm the status says connected and does not report a gap, restart, or
   projection error.

In Blender's **3D View > Sidebar > Hyperscape > Local Blender ↔ Hyperscope**
panel:

1. Choose **Connect Local Hyperscope Peer**.
2. Leave the relay URL at `http://127.0.0.1:42117`, enter the same token, and
   confirm.
3. Confirm **State: connected** and at least one remote peer before moving an
   authored object.

The token is runtime-only: the browser does not persist it and Blender does not
save it into the `.blend` or add-on preferences. Disconnect both peers and stop
the relay after the presentation.

## 4. Interaction demonstrations

- **Fly/orbit:** mouse and keyboard controls remain the baseline. A SpaceMouse
  is optional.
- **Select/focus:** left-click a rendered object. Focus and inversion share the
  selected object's smoothly interpolated sphere.
- **Surface walk:** right-click a surface to attach, then use `W`, `A`, `S`, and
  `D`. The walker follows animated QB geometry and detaches safely when contact
  becomes invalid.
- **Load a fallback model:** drag a local `.glb` onto the canvas or use the file
  picker. From the cue deck, a drop opens a clean ordinary-model route after
  persisting the file; Back returns to the exact deck URL. This is a
  recovery/demo path, not part of the deterministic cue deck.
- **Compare grading policy:** open
  `?mode=both&animate=0&lab=triangle&labfield=edges&laba=1&labb=6&labc=6`,
  then add `&lodratio=4`. The page reload is intentional: diagnostics must report the
  active policy, zero shared-edge mismatches, the reconciled request, and the
  resulting resident triangle count from the matching cached atlas.
- **Optional Blender presence:** start the loopback relay with the
  `local-peer-relay` feature, then explicitly connect both clients using its
  runtime-only token. Moving the browser camera should advance `sentFrames`
  and `publishedPresenceFrames` at no more than 20 Hz; a settled view refreshes
  every 500 ms. Disconnect, wait longer than the 1,500 ms TTL, and verify
  `peerPresenceSnapshot().peers` no longer contains the browser sender. This
  lane is delivery-only: it must not create HHHS history, browser persistence,
  or Blender datablocks. In Blender, the same live sample should appear as a
  camera glyph, focus/inversion sphere, and selected-object wire bounds; an
  inverted browser view must appear in Blender's ordinary source chart.

## 5. Browser smoke check

Use a fresh browser tab after the release build and verify:

- no `phase wasm failed`, atlas-upload failure, GLB parse error, or uncaught
  exception appears in the console;
- the horse animates continuously and LOD does not flash to a stale low level;
- on the final two-asset cue, `__hyperscopePresentation.lodCadence` reports a
  scene classification with `lastSceneSubjectRecords: 9` and
  `lastSceneGpuPasses: 1`; later animation-only classifications report
  `lastPrimaryAnimationSubjectRecords: 1` and
  `lastPrimaryAnimationGpuPasses: 1` without implying that the static
  Blender-authored asset has left LOD residency. The unsuffixed `last*` fields
  remain the most recent classification of either scope and will normally
  return to the animation-only values while the horse is playing;
- picking/selection tint and surface attachment use the visible object;
- moving through all six cues leaves exactly one requested visualization mode
  active;
- the final cue reports no failed or unsupported required asset;
- reloading with the network disabled still succeeds through local HTTP.

The preload `integrity` warning emitted by Chrome for unsupported preload
destinations is informational; a renderer initialization or asset error is not.

Selection and surface walking deliberately retain their rehearsed JavaScript
defaults for the talk. Do not add `selectionimpl=rust` or `walkimpl=rust` on
stage unless that exact opt-in path was rehearsed on the presentation machine.
The shadow/Rust parity procedure remains in the
[focus and navigation roadmap](focus-navigation-roadmap.md).

## 6. Recovery during the talk

1. Reload the exact URL above. Presentation state deterministically returns to
   cue one.
2. If an interaction leaves the camera disoriented, reload rather than trying
   to reconstruct an improvised view on stage.
3. If WebHID fails, continue with mouse/keyboard; do not change operating-system
   permissions during the talk.
4. If a dropped model fails, go Back or reload the known deck URL—the checked-in
   assets are the rehearsed path.
5. Keep the exact `dist-release/` that passed strict preflight. Do not serve the
   development `dist/` or run a rebuild during the presentation.

## 7. Release boundary

The presentation bundle and the source-code release are different artifacts.
The code can be released under the repository licenses while third-party model
and image policy is resolved. The horse has documented CC BY-NC-SA 3.0 terms;
it is not part of the MIT/Apache grant and cannot be included in a commercial
or permissive-only bundle. Exclude `local-glbs/` and any test downloads from
every archive. A public bundle under the intended
policy is ready only when the strict preflight has no errors or warnings and
the browser smoke check passes on the target machine.

After that gate passes, make a deterministic transport archive at a new path:

```sh
release_archive=../hyperscope-hacker-night.tar.gz
test ! -e "$release_archive"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  --format=gnu -C dist-release -cf - . | gzip -9 -n > "$release_archive"
sha256sum "$release_archive"
```

Extract the archive into an empty directory and repeat strict preflight before
publishing or moving it to the presentation machine. The deterministic options
make identical staged contents produce an identical compressed archive.
