import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, HyperscopeAppShadow, HyperscopeNavigation } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const app = new HyperscopeAppShadow();
const asset = 'f0000000-0000-4000-8000-000000000001';
const first = 'e0000000-0000-4000-8000-000000000001';
const second = 'e0000000-0000-4000-8000-000000000002';

const requested = app.requestAsset(
  1,
  0,
  first,
  asset,
  'horse.glb',
  'model/gltf-binary',
);
assert.deepEqual(requested.effects.map(effect => effect.type), ['fetch_asset']);

const replaced = app.requestAsset(
  2,
  0,
  second,
  asset,
  'horse.glb',
  'model/gltf-binary',
);
assert.deepEqual(
  replaced.effects.map(effect => effect.type),
  ['cancel_asset_load', 'fetch_asset'],
);
assert.equal(replaced.effects[0].request_id, first);

const stale = app.completeAssetLoaded(first, asset, 181_808);
assert.equal(stale.disposition, 'ignored_stale');
const afterStale = app.snapshot();
assert.equal(afterStale.loadingAssets, 1);
assert.equal(afterStale.assets[0].status.state, 'loading');
assert.equal(afterStale.assets[0].status.request_id, second);
assert.equal(afterStale.diagnostics[0].code, 'stale_effect_completion');

const applied = app.completeAssetLoaded(second, asset, 181_808);
assert.equal(applied.disposition, 'applied');
const ready = app.snapshot();
assert.equal(ready.loadingAssets, 0);
assert.equal(ready.assets[0].status.state, 'ready');
assert.equal(ready.assets[0].status.byte_length, 181_808);

const presentationDocument = readFileSync(
  `${repository}/examples/hacker-night.presentation.json`,
  'utf8',
);
const presentation = JSON.parse(presentationDocument);
const loadedPresentation = app.loadPresentation(presentationDocument);
const incumbent = new HyperscopeNavigation();
incumbent.loadPresentation(presentationDocument);
assert.equal(loadedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.cueCount, 6);
assert.equal(app.snapshot().presentation.active, undefined);

const eye = new Float64Array([0, 0, 3]);
const forward = new Float64Array([0, 0, -1]);
const up = new Float64Array([0, 1, 0]);
const target = new Float64Array([0, 0, 0]);
const focusCenter = new Float64Array([0.5, 0, 0]);
function assertNavigationParity(actual, expected) {
  for (const field of ['eye', 'right', 'up', 'forward', 'semantic_target']) {
    assert.deepEqual(actual.camera[field], expected.camera[field]);
  }
  for (const field of [
    'control_distance', 'camera_transition_remaining',
  ]) {
    assert.equal(actual.camera[field], expected.camera[field]);
  }
  assert.deepEqual(actual.focus.center, expected.focus.center);
  for (const field of [
    'radius', 'anchored', 'focus_enabled', 'inversion_enabled', 'focus_coordinate',
    'angular_aperture', 'focus_transition_remaining',
  ]) {
    assert.equal(actual.focus[field], expected.focus[field]);
  }
  assert.equal(actual.reflection, expected.reflection);
}
incumbent.synchronizeState(
  eye, forward, up, 3, target, focusCenter, 2, false, false, 0.5, 0.1,
);
const synchronized = app.synchronizeNavigation(
  eye,
  forward,
  up,
  3,
  target,
  focusCenter,
  2,
  false,
  false,
  0.5,
  0.1,
);
assert.equal(synchronized.publishedUi, false);
const startedPresentation = app.present(3, 'start', '');
const incumbentStart = incumbent.startPresentation();
assert.equal(startedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.active.cue_id, presentation.cues[0].id);
assert.deepEqual(app.snapshot().presentation.active, incumbentStart);
const midTransition = app.tickPresentation(0.35);
const incumbentMidTransition = incumbent.tick(0.35);
assert.equal(midTransition.elapsed_seconds, 0.35);
assert.ok(Math.abs(midTransition.camera.camera_transition_remaining - 0.35) < 1e-12);
assertNavigationParity(midTransition, incumbentMidTransition);
assertNavigationParity(app.tickPresentation(0.35), incumbent.tick(0.35));

const linkedCue = presentation.cues[4].id;
const linkedApp = app.present(4, 'jump', linkedCue);
const linkedIncumbent = incumbent.jumpToPresentationCue(linkedCue);
assert.equal(linkedApp.disposition, 'applied');
assert.deepEqual(app.snapshot().presentation.active, linkedIncumbent);
assert.equal(app.snapshot().presentation.active.cue_id, linkedCue);
for (let step = 0; step < 12; step++) {
  assertNavigationParity(app.tickPresentation(0.1), incumbent.tick(0.1));
}
const inverted = app.tickPresentation(0);
assert.equal(inverted.reflection, 'sphere_reflection');
assert.equal(inverted.focus.inversion_enabled, true);
assert.throws(
  () => app.present(5, 'jump', 'not-a-uuid'),
  /cue ID must be a UUID/,
);
assert.equal(
  app.snapshot().presentation.active.cue_id,
  linkedCue,
  'a malformed shadow cue must preserve the preceding reducer state',
);

app.advanceFrame(2, 0.1);
assert.throws(
  () => app.requestAsset(
    3,
    3,
    'e0000000-0000-4000-8000-000000000003',
    asset,
    'horse.glb',
    'model/gltf-binary',
  ),
  /effect-producing input cannot be scheduled/,
);

const finalSnapshot = app.snapshot();
incumbent.free();
app.free();
console.log(JSON.stringify({
  requested: requested.effects.length,
  replacementEffects: replaced.effects.map(effect => effect.type),
  staleDisposition: stale.disposition,
  readyBytes: ready.assets[0].status.byte_length,
  diagnostics: ready.diagnostics.map(diagnostic => diagnostic.code),
  presentationCue: finalSnapshot.presentation.active.cue_id,
}));
