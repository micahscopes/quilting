import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const {
  default: init,
  HyperscopeAppShadow,
  HyperscopeNavigation,
  mapSpaceMouseCameraFrame,
} = await import(packageUrl);
const { mapSpaceMouseNavigationAxes } = await import(
  pathToFileURL(`${repository}/spacemouse.mjs`).href
);
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

// A generated-WASM authority gate for the exact browser rollback oracle: the
// proposed inversion sphere is centered at the camera eye. The queued action
// is consumed, but camera, focus intent, and active reflection remain one
// coherent identity-chart transaction after the transport reaches its pole.
const poleApp = new HyperscopeAppShadow();
const poleCenter = new Float64Array([0, 0, 3]);
poleApp.synchronizeNavigation(
  eye, forward, up, 3, target, poleCenter, 2, false, false, 0.5, 0.1,
);
const beforePole = poleApp.navigationSnapshot();
assert.equal(poleApp.setInversionEnabled(true), 0n);
assert.equal(poleApp.navigationSnapshot().pending_actions, 1);
const afterPole = poleApp.tickNavigation(0);
assert.equal(afterPole.pending_actions, 0);
assert.equal(afterPole.last_applied_sequence, 0);
assert.equal(afterPole.reflection, 'identity');
assert.equal(afterPole.focus.inversion_enabled, false);
assert.deepEqual(afterPole.focus.center, beforePole.focus.center);
assert.equal(afterPole.focus.radius, beforePole.focus.radius);
assert.deepEqual(afterPole.camera, beforePole.camera);
assert.match(
  afterPole.diagnostics.at(-1),
  /camera transport reached a spherical-reflection pole/,
);
poleApp.free();

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

function browserSpaceMouseCameraFrame(normalizedAxes, sample) {
  const mapped = Array.from(mapSpaceMouseNavigationAxes(normalizedAxes, {
    mode: sample.preset,
    swapYZ: sample.swapYZ,
    invertPan: sample.invertPan,
    invertRotate: sample.invertRotate,
  }));
  const translationScale = sample.registeredLinearSpeed
    * sample.moveGain * sample.deltaSeconds;
  const rotationScale = 1.5 * sample.rotateGain * sample.deltaSeconds;
  const translation = mapped.slice(0, 3).map(axis => axis * translationScale);
  const rotation = mapped.slice(3, 6).map(axis => axis * rotationScale);
  let dollyLog = 0;
  if (sample.preset === 'object') {
    translation[2] = 0;
    dollyLog = mapped[2] * 1.5 * sample.moveGain * sample.deltaSeconds;
  }
  return {
    translation,
    rotation,
    dolly_log: dollyLog,
    horizon_locked: sample.preset === 'drone'
      || (sample.preset !== 'hyperscope' && sample.horizonLockRequested),
  };
}

function rustSpaceMouseCameraFrame(normalizedAxes, sample) {
  return mapSpaceMouseCameraFrame(
    normalizedAxes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
}

const presets = ['hyperscope', 'object', 'fly', 'drone'];
const normalizedAxes = new Float32Array([0.25, -0.5, 0.75, -1, 0.125, -0.25]);
const spaceMouseAxisVectors = [new Float32Array(6)];
for (let axis = 0; axis < 6; axis++) {
  const positive = new Float32Array(6);
  positive[axis] = 1;
  spaceMouseAxisVectors.push(positive);
  const negative = new Float32Array(6);
  negative[axis] = -1;
  spaceMouseAxisVectors.push(negative);
}
spaceMouseAxisVectors.push(normalizedAxes);

let exhaustiveMappingCases = 0;
for (const preset of presets) {
  for (const swapYZ of [false, true]) {
    for (let invertPan = 0; invertPan < 8; invertPan++) {
      for (let invertRotate = 0; invertRotate < 8; invertRotate++) {
        for (const axes of spaceMouseAxisVectors) {
          const sample = {
            preset,
            swapYZ,
            invertPan,
            invertRotate,
            deltaSeconds: 0.25,
            registeredLinearSpeed: 2,
            moveGain: 0.5,
            rotateGain: 1.25,
            horizonLockRequested: false,
          };
          assert.deepEqual(
            rustSpaceMouseCameraFrame(axes, sample),
            { preset, frame: browserSpaceMouseCameraFrame(axes, sample) },
          );
          exhaustiveMappingCases++;
        }
      }
    }
  }
}

let responsePolicyCases = 0;
for (const preset of presets) {
  for (const horizonLockRequested of [false, true]) {
    for (const deltaSeconds of [0, 0.125, 0.5]) {
      for (const registeredLinearSpeed of [0, 0.5, 4]) {
        for (const moveGain of [0, 0.25, 3]) {
          for (const rotateGain of [0, 0.5, 4]) {
            const sample = {
              preset,
              swapYZ: true,
              invertPan: 0b101,
              invertRotate: 0b010,
              deltaSeconds,
              registeredLinearSpeed,
              moveGain,
              rotateGain,
              horizonLockRequested,
            };
            assert.deepEqual(
              rustSpaceMouseCameraFrame(normalizedAxes, sample),
              { preset, frame: browserSpaceMouseCameraFrame(normalizedAxes, sample) },
            );
            responsePolicyCases++;
          }
        }
      }
    }
  }
}

const spaceMouseCases = [
  {
    preset: 'hyperscope', swapYZ: false, invertPan: 0b010, invertRotate: 0b001,
    deltaSeconds: 0.25, registeredLinearSpeed: 2, moveGain: 0.5, rotateGain: 4 / 3,
    horizonLockRequested: true,
  },
  {
    preset: 'object', swapYZ: false, invertPan: 0, invertRotate: 0,
    deltaSeconds: 0.5, registeredLinearSpeed: 4, moveGain: 0.5, rotateGain: 2 / 3,
    horizonLockRequested: true,
  },
  {
    preset: 'fly', swapYZ: true, invertPan: 0b101, invertRotate: 0b010,
    deltaSeconds: 0.125, registeredLinearSpeed: 8, moveGain: 0.75, rotateGain: 2,
    horizonLockRequested: false,
  },
  {
    preset: 'drone', swapYZ: false, invertPan: 0b111, invertRotate: 0b111,
    deltaSeconds: 0.5, registeredLinearSpeed: 0.5, moveGain: 2, rotateGain: 1,
    horizonLockRequested: false,
  },
];
const spaceMouseCameraStates = [
  { eye, forward, up, target },
  {
    eye: new Float64Array([2, -1, 4]),
    forward,
    up,
    target: new Float64Array(),
  },
  {
    eye: new Float64Array([1, 2, 3]),
    forward: new Float64Array([0, -1, 0]),
    up: new Float64Array([1, 0, 0]),
    target: new Float64Array(),
  },
  {
    eye: new Float64Array([3, 0, 0]),
    forward: new Float64Array([-1, 0, 0]),
    up: new Float64Array([0, 0, 1]),
    target,
  },
];
for (const [caseIndex, sample] of spaceMouseCases.entries()) {
  const mappedApp = new HyperscopeAppShadow();
  const semanticApp = new HyperscopeAppShadow();
  const camera = spaceMouseCameraStates[caseIndex];
  for (const candidate of [mappedApp, semanticApp]) {
    candidate.synchronizeNavigation(
      camera.eye, camera.forward, camera.up, 3, camera.target,
      focusCenter, 2, false, false, 0.5, 0.1,
    );
  }
  const expectedFrame = browserSpaceMouseCameraFrame(normalizedAxes, sample);
  const dispatch = mappedApp.queueSpaceMouseCamera(
    normalizedAxes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
  assert.deepEqual(dispatch.frame, expectedFrame);
  assert.equal(dispatch.preset, sample.preset);
  assert.equal(dispatch.preset_sequence, '0');
  assert.equal(dispatch.frame_sequence, '1');
  assert.equal(semanticApp.setPreset(sample.preset), 0n);
  assert.equal(
    semanticApp.applyFrame(
      new Float64Array(expectedFrame.translation),
      new Float64Array(expectedFrame.rotation),
      expectedFrame.dolly_log,
      expectedFrame.horizon_locked,
    ),
    1n,
  );
  assertNavigationParity(mappedApp.navigationSnapshot(), semanticApp.navigationSnapshot());
  assertNavigationParity(mappedApp.tickNavigation(0), semanticApp.tickNavigation(0));
  mappedApp.free();
  semanticApp.free();
}

const traceMappedApp = new HyperscopeAppShadow();
const traceSemanticApp = new HyperscopeAppShadow();
for (const candidate of [traceMappedApp, traceSemanticApp]) {
  candidate.synchronizeNavigation(
    new Float64Array([2, -1, 4]), forward, up, 3, new Float64Array(),
    focusCenter, 2, false, false, 0.5, 0.1,
  );
}
const traceDeltas = [1 / 128, 1 / 64, 1 / 32];
const traceSpeeds = [0.5, 2, 8];
const traceMoveGains = [0.25, 1, 3];
const traceRotateGains = [0.5, 1, 2];
const traceFrames = 120;
for (let frame = 0; frame < traceFrames; frame++) {
  const sample = {
    preset: presets[frame % presets.length],
    swapYZ: (frame & 1) !== 0,
    invertPan: frame % 8,
    invertRotate: (frame * 3) % 8,
    deltaSeconds: traceDeltas[frame % traceDeltas.length],
    registeredLinearSpeed: traceSpeeds[(frame + 1) % traceSpeeds.length],
    moveGain: traceMoveGains[(frame + 2) % traceMoveGains.length],
    rotateGain: traceRotateGains[frame % traceRotateGains.length],
    horizonLockRequested: (frame & 2) !== 0,
  };
  const axes = spaceMouseAxisVectors[frame % spaceMouseAxisVectors.length];
  const expectedFrame = browserSpaceMouseCameraFrame(axes, sample);
  const dispatch = traceMappedApp.queueSpaceMouseCamera(
    axes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
  assert.deepEqual(dispatch.frame, expectedFrame);
  assert.equal(dispatch.preset_sequence, String(frame * 2));
  assert.equal(dispatch.frame_sequence, String(frame * 2 + 1));
  assert.equal(traceSemanticApp.setPreset(sample.preset), BigInt(frame * 2));
  assert.equal(
    traceSemanticApp.applyFrame(
      new Float64Array(expectedFrame.translation),
      new Float64Array(expectedFrame.rotation),
      expectedFrame.dolly_log,
      expectedFrame.horizon_locked,
    ),
    BigInt(frame * 2 + 1),
  );
  assertNavigationParity(traceMappedApp.navigationSnapshot(), traceSemanticApp.navigationSnapshot());
  assertNavigationParity(
    traceMappedApp.tickNavigation(sample.deltaSeconds),
    traceSemanticApp.tickNavigation(sample.deltaSeconds),
  );
}
traceMappedApp.free();
traceSemanticApp.free();

const invalidSpaceMouseApp = new HyperscopeAppShadow();
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array(5), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /exactly six normalized axes/,
);
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array([NaN, 0, 0, 0, 0, 0]), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /remain finite/,
);
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array([1.01, 0, 0, 0, 0, 0]), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /within \[-1, 1\]/,
);
for (const response of [
  [-1, 1, 1, 1],
  [1, -1, 1, 1],
  [1, 1, -1, 1],
  [1, 1, 1, -1],
]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      normalizedAxes, 'fly', false, 0, 0, ...response, false,
    ),
    /must be nonnegative/,
  );
}
for (const invalidMask of [8, 256, 263, 1.5, NaN]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      normalizedAxes, 'object', false, invalidMask, 0, 1, 1, 1, 1, false,
    ),
    /finite integers from 0 through 7/,
  );
}
const overflowSpaceMouseAxes = new Float32Array([1, 0, 0, 0, 0, 0]);
for (const [preset, deltaSeconds, registeredLinearSpeed, moveGain, rotateGain] of [
  ['fly', 1, Number.MAX_VALUE, 2, 1],
  ['fly', 1, 1, 1, Number.MAX_VALUE],
  ['object', 1, 0, Number.MAX_VALUE, 1],
]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      overflowSpaceMouseAxes, preset, false, 0, 0,
      deltaSeconds, registeredLinearSpeed, moveGain, rotateGain, false,
    ),
    /remain finite/,
  );
}
assert.equal(invalidSpaceMouseApp.navigationSnapshot().pending_actions, 0);
assert.equal(invalidSpaceMouseApp.navigationSnapshot().preset, 'hyperscope');
invalidSpaceMouseApp.free();

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
  spaceMouseInputCases: {
    exhaustiveMapping: exhaustiveMappingCases,
    responsePolicy: responsePolicyCases,
    queuedCameraStates: spaceMouseCases.length,
    deterministicTraceFrames: traceFrames,
  },
}));
