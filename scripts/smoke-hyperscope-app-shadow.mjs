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
  assert.equal(actual.elapsed_seconds, expected.elapsed_seconds);
  assert.equal(actual.preset, expected.preset);
  assert.equal(actual.pending_actions, expected.pending_actions);
  assert.equal(actual.last_applied_sequence, expected.last_applied_sequence);
  for (const field of ['eye', 'orientation', 'right', 'up', 'forward', 'semantic_target']) {
    assert.deepEqual(actual.camera[field], expected.camera[field]);
  }
  for (const field of [
    'control_distance', 'camera_transition_remaining',
    'surface_anchor_transition_remaining', 'surface_anchor_hop_height',
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
  assert.deepEqual(actual.diagnostics, expected.diagnostics);
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

// Presentation and direct navigation deliberately share these same two
// instances. This catches collisions between cue-authored and adapter-authored
// sequence numbers before the explicit re-synchronization below.
assert.equal(app.toggleInversion(), incumbent.toggleInversion());
assertNavigationParity(app.navigationSnapshot(), incumbent.snapshot());
assertNavigationParity(app.tickNavigation(0), incumbent.tick(0));

const navigationApp = app;
const navigationIncumbent = incumbent;
navigationIncumbent.synchronizeState(
  eye, forward, up, 3, target, focusCenter, 2, false, false, 0.5, 0.1,
);
navigationApp.synchronizeNavigation(
  eye, forward, up, 3, target, focusCenter, 2, false, false, 0.5, 0.1,
);
assert.equal(navigationApp.setPreset('fly'), navigationIncumbent.setPreset('fly'));
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
const reverseInterleaveApp = navigationApp.present(6, 'advance', '');
const reverseInterleaveIncumbent = navigationIncumbent.advancePresentation();
assert.deepEqual(
  navigationApp.snapshot().presentation.active,
  reverseInterleaveIncumbent,
  'direct navigation followed by presentation must preserve shared sequence order',
);
assert.equal(reverseInterleaveApp.disposition, 'applied');
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());

navigationIncumbent.synchronizeState(
  eye, forward, up, 3, target, focusCenter, 2, false, false, 0.5, 0.1,
);
navigationApp.synchronizeNavigation(
  eye, forward, up, 3, target, focusCenter, 2, false, false, 0.5, 0.1,
);
const firstNavigationSequence = navigationApp.setPreset('fly');
assert.equal(firstNavigationSequence, navigationIncumbent.setPreset('fly'));
assert.equal(firstNavigationSequence, 0n, 'synchronization resets the shared sequence authority');
assert.equal(
  navigationApp.applyFrame(
    new Float64Array([0.2, -0.1, -0.4]),
    new Float64Array([0.03, -0.02, 0.01]),
    0,
    false,
  ),
  navigationIncumbent.applyFrame(
    new Float64Array([0.2, -0.1, -0.4]),
    new Float64Array([0.03, -0.02, 0.01]),
    0,
    false,
  ),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(
  navigationApp.tickNavigation(1 / 60),
  navigationIncumbent.tick(1 / 60),
);

const transitionEye = new Float64Array([0.5, 0.25, 4]);
assert.equal(
  navigationApp.transitionCamera(
    transitionEye, forward, up, 4, target, 0.5, 'smootherstep',
  ),
  navigationIncumbent.transitionCamera(
    transitionEye, forward, up, 4, target, 0.5, 'smootherstep',
  ),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.setFreeFocusSphere(new Float64Array([0.25, 0.5, -0.25]), 1.5),
  navigationIncumbent.setFreeFocusSphere(new Float64Array([0.25, 0.5, -0.25]), 1.5),
);
assert.equal(
  navigationApp.setFocusEnabled(true),
  navigationIncumbent.setFocusEnabled(true),
);
assert.equal(
  navigationApp.setFocusField(0.35, 0.075),
  navigationIncumbent.setFocusField(0.35, 0.075),
);
assert.equal(
  navigationApp.setInversionEnabled(true),
  navigationIncumbent.setInversionEnabled(true),
);
assert.equal(
  navigationApp.translateFocus(new Float64Array([0.1, -0.2, 0.05])),
  navigationIncumbent.translateFocus(new Float64Array([0.1, -0.2, 0.05])),
);
assert.equal(
  navigationApp.scaleFocusLog(Math.log(1.2)),
  navigationIncumbent.scaleFocusLog(Math.log(1.2)),
);
assert.equal(
  navigationApp.toggleInversion(),
  navigationIncumbent.toggleInversion(),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(navigationApp.tickNavigation(0), navigationIncumbent.tick(0));

const anchorEye = new Float64Array([1, 0.5, 2]);
const anchorForward = new Float64Array([0, 0, -1]);
const anchorUp = new Float64Array([0, 1, 0]);
const anchorNormal = new Float64Array([0, 1, 0]);
assert.equal(
  navigationApp.beginSurfaceAnchorTransition(
    anchorEye, anchorForward, anchorUp, 2, anchorNormal, 10, 1, 'smootherstep',
  ),
  navigationIncumbent.beginSurfaceAnchorTransition(
    anchorEye, anchorForward, anchorUp, 2, anchorNormal, 10, 1, 'smootherstep',
  ),
);
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.updateSurfaceAnchorTarget(
    new Float64Array([1.25, 0.6, 2]),
    anchorForward,
    anchorUp,
    2,
    anchorNormal,
  ),
  navigationIncumbent.updateSurfaceAnchorTarget(
    new Float64Array([1.25, 0.6, 2]),
    anchorForward,
    anchorUp,
    2,
    anchorNormal,
  ),
);
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.cancelSurfaceAnchorTransition(),
  navigationIncumbent.cancelSurfaceAnchorTransition(),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(navigationApp.tickNavigation(0), navigationIncumbent.tick(0));

const finalFrameTime = app.navigationSnapshot().elapsed_seconds + 0.1;
app.advanceFrame(finalFrameTime, 0.1);
assert.throws(
  () => app.requestAsset(
    3,
    finalFrameTime + 1,
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
  navigationBoundaryParity: true,
}));
