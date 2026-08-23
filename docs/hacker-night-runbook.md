# Hyperscope hacker-night runbook

This is the exact offline-friendly path for the checked-in six-cue Quilting
presentation. Rehearse these commands and this URL on the presentation machine;
do not rely on a development tab with cached assets.

## 1. Build and preflight

From the repository root:

```sh
trunk build --release
cargo run -p hyperscape --bin hyperscope-preflight
```

The ordinary preflight must print `PASS`. It validates the Rust presentation
document, byte-for-byte manifest freshness, every presentation GLB, the runtime
JS/WASM pairs, environment maps, licenses, and generated Trunk
bootstrap. It also reports the checked bundle size.

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

The preload `integrity` warning emitted by Chrome for unsupported preload
destinations is informational; a renderer initialization or asset error is not.

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
