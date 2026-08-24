import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const manifestPath = `${repository}/examples/hacker-night.presentation.json`;
const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
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
