import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const manifestPath = `${repository}/examples/hacker-night.presentation.json`;
const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
const workerSource = readFileSync(`${repository}/hyperscope_worker.js`, 'utf8');
const { default: init, HyperscopeNavigation } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const cueActivation = browserSource.slice(
  browserSource.indexOf('function activateRustPresentation('),
  browserSource.indexOf('async function initializeRustPresentation('),
);
const guardedSelectionClear = cueActivation.indexOf(
  'if (selectedObject || rustAppShadowSelectionQueued)',
);
assert.ok(guardedSelectionClear >= 0);
assert.ok(
  guardedSelectionClear < cueActivation.indexOf('synchronizeRustApplicationPresentationFromBrowser();'),
  'a cue must detach a real selection before synchronizing its free-focus start state',
);

const rendererSelection = browserSource.slice(
  browserSource.indexOf('function selectObjectFromFace('),
  browserSource.indexOf('function selectObjectAtViewCenter('),
);
for (const handoff of [
  'rustPresentationTransitionActive = false;',
  'rustAppPresentationPoseReady = false;',
]) {
  assert.ok(
    rendererSelection.indexOf(handoff) >= 0
      && rendererSelection.indexOf(handoff) < rendererSelection.indexOf('anchorFocusSphereToSelection('),
    `renderer selection must execute ${handoff} before anchoring`,
  );
}
assert.ok(
  browserSource.match(/if \(rustPresentationController && rustAppPresentationPoseReady\)/g)?.length >= 2,
  'selection handoff must stop focus/lens mutations of the inactive presentation observer',
);

const controller = new HyperscopeNavigation();
const presentation = controller.loadPresentation(readFileSync(manifestPath, 'utf8'));
assert.equal(presentation.cues.length, 6);

const first = controller.startPresentation();
assert.equal(first.cue_id, presentation.cues[0].id);
assert.equal(first.cue_index, 0);
assert.deepEqual(first.animations, presentation.cues[0].animations);

for (const requiredAnimationAdapterStep of [
  'primaryPresentationAnimation(snapshot)',
  'selectAnimationIndex(clipIndex)',
  'animTime = time;',
  'animating_sig.set(Boolean(animation.playing));',
  'deltaSeconds * presentationAnimationSpeed',
]) {
  assert.ok(
    browserSource.includes(requiredAnimationAdapterStep),
    `browser animation adapter is missing ${requiredAnimationAdapterStep}`,
  );
}

for (const requiredAuthorityStep of [
  "get('presentimpl') || 'js'",
  "RUST_PRESENTATION_IMPLEMENTATION === 'rust'",
  "ensureRustAppShadow('presentation-authority')",
  'rustPresentationController = null;',
  'rustPresentationManifest = appPresentationManifest(refreshAppShadowSnapshot());',
  'const result = dispatchAppPresentation(direction, cueId);',
  'navigation = appFrame?.navigation || rustAppShadow.navigationSnapshot();',
  'rustAppShadow.extractActivePresentationScene(',
  'const presentationBindings = [];',
  'const semantics = resolved.semanticNodes.get(state.node);',
]) {
  assert.ok(
    browserSource.includes(requiredAuthorityStep),
    `browser presentation authority gate is missing ${requiredAuthorityStep}`,
  );
}

const presentationBindingAdapter = browserSource.slice(
  browserSource.indexOf('const presentationBinding = layer ? {'),
  browserSource.indexOf('} : null;', browserSource.indexOf('const presentationBinding = layer ? {')),
);
assert.ok(!presentationBindingAdapter.includes('layerTransform'));
const activationAdapter = browserSource.slice(
  browserSource.indexOf('function activateRustPresentation('),
  browserSource.indexOf('async function initializeRustPresentation('),
);
assert.ok(
  activationAdapter.indexOf('mirrorAppPresentation(direction, cueId, snapshot)')
    < activationAdapter.indexOf('renderRustPresentationSnapshot(snapshot)'),
  'AppStore cue authority must commit before active-scene extraction during rendering',
);

const layerAdapter = browserSource.slice(
  browserSource.indexOf('function applyPresentationLayerState('),
  browserSource.indexOf('function applyRustPresentationNavigation('),
);
assert.ok(
  layerAdapter.includes('lodRecords.push(...lodRecord);'),
  'every resident presentation node must contribute its authored LOD state',
);
assert.ok(
  layerAdapter.indexOf('lodRecords.push(...lodRecord);')
    < layerAdapter.indexOf('if (state.primary)'),
  'primary animation filtering must happen after the complete scene LOD state is retained',
);
const compositionAdapter = browserSource.slice(
  browserSource.indexOf('async function uploadPresentationCompositionToLodWorker('),
  browserSource.indexOf('// --- Init ---'),
);
for (const requiredSceneLodStep of [
  "workerCall('upload_composed_model_to_compute'",
  'resident.faceOffset + resident.faceCount',
  'await uploadPresentationCompositionToLodWorker(',
  'residentFaces !== presentationComposition.totalFaces',
  'recordPresentationLodUpdate(faceIndices);',
  'faceLimit: primaryOnly ? currentPrimaryFaceCount : 0',
  'presentationComposition.primaryLodStates',
]) {
  assert.ok(
    compositionAdapter.includes(requiredSceneLodStep)
      || browserSource.includes(requiredSceneLodStep),
    `packed-scene LOD adapter is missing ${requiredSceneLodStep}`,
  );
}
for (const requiredWorkerStep of [
  "type === 'upload_composed_model_to_compute'",
  'wasm.cancel_animated_lods();',
  'wasm.upload_composed_model_to_compute(',
  'faceLimit || 0',
]) {
  assert.ok(
    workerSource.includes(requiredWorkerStep),
    `worker packed-scene LOD path is missing ${requiredWorkerStep}`,
  );
}

const linkedCue = presentation.cues[2].id;
const linked = controller.jumpToPresentationCue(linkedCue);
assert.equal(linked.cue_id, linkedCue);
assert.equal(linked.cue_index, 2);
assert.equal(controller.presentationSnapshot().cue_id, linkedCue);

function captureFailure(operation) {
  try {
    operation();
  } catch (error) {
    return String(error);
  }
  assert.fail('expected presentation operation to fail');
}

assert.match(
  captureFailure(() => controller.jumpToPresentationCue('not-a-uuid')),
  /invalid presentation cue UUID/,
);
assert.match(
  captureFailure(() => controller.jumpToPresentationCue(
    'f0000000-0000-4000-8000-000000000099',
  )),
  /unknown presentation cue/,
);
assert.equal(
  controller.presentationSnapshot().cue_id,
  linkedCue,
  'a rejected deep link must preserve the last valid cue',
);

console.log(JSON.stringify({
  cues: presentation.cues.length,
  initialCue: first.cue_id,
  linkedCue: linked.cue_id,
  rejectedCuePreserved: controller.presentationSnapshot().cue_id,
}));
