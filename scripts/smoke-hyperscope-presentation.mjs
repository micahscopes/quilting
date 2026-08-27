import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const manifestPath = `${repository}/crates/hyperscape/fixtures/hacker-night.presentation.json`;
const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
const workerSource = readFileSync(`${repository}/hyperscope_worker.js`, 'utf8');
const mainRendererSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/main_renderer.rs`,
  'utf8',
);
const {
  default: init,
  HyperscopeNavigation,
  hyperscopeControlSpecs,
  load_patch_lab,
  mr_acceptLodDeltaSequence,
  mr_uploadComposedLodModel,
  update_patch_lab_lods,
} = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

assert.equal(typeof mr_uploadComposedLodModel, 'function');
assert.deepEqual(
  hyperscopeControlSpecs().find(spec => spec.key === 'lodimpl'),
  { key: 'lodimpl', defaultValue: 'js', kind: 'implementation' },
);

assert.equal(mr_acceptLodDeltaSequence(91, 0, 1, true), true);
assert.equal(mr_acceptLodDeltaSequence(91, 1, 2, false), false);
assert.throws(
  () => mr_acceptLodDeltaSequence(91, 3, 4, false),
  /does not extend the resident base/,
);
assert.equal(mr_acceptLodDeltaSequence(92, 0, 1, true), true);

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
  'claimManualCameraFromActiveView()',
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

const selectedInversionIngress = browserSource.slice(
  browserSource.indexOf('function dispatchAppSelectedInversionGesture('),
  browserSource.indexOf('function clearSelectedObject('),
);
for (const semanticStep of [
  'advanceRustApplicationClockToSelectionEvent(nowMs);',
  'rustAppShadow.refitFocusAndToggleInversion(',
  'const navigation = rustAppShadow.tickNavigation(0);',
  "? 'inversionGestureAccepts'",
]) {
  assert.ok(
    selectedInversionIngress.includes(semanticStep),
    `selected inversion ingress is missing ${semanticStep}`,
  );
}

const selectedInversionGesture = browserSource.slice(
  browserSource.indexOf('function fitOrToggleInversionToSelection('),
  browserSource.indexOf('function advanceFocusTransitions('),
);
assert.ok(
  selectedInversionGesture.indexOf('dispatchAppSelectedInversionGesture(nowMs)')
    < selectedInversionGesture.indexOf('anchorFocusSphereToSelection(nowMs)'),
  'the typed Rust gesture must be admitted before the incumbent fallback mutates focus',
);
assert.ok(
  selectedInversionGesture.includes("RUST_SELECTION_IMPLEMENTATION === 'rust'"),
  'selected inversion must retain an explicit Rust authority branch',
);
assert.ok(
  selectedInversionGesture.includes('applyRustSelectedFocusNavigation(rustGesture.navigation)'),
  'Rust authority must apply the coherent camera/focus snapshot',
);
assert.ok(
  browserSource.includes('globalThis.__hyperscopeSelection = Object.freeze({'),
  'target-browser acceptance must use the real semantic selection gesture path',
);

const rustSelectionFrame = browserSource.slice(
  browserSource.indexOf('function advanceRustApplicationFrame('),
  browserSource.indexOf('function compareAppPresentationPose('),
);
assert.ok(
  rustSelectionFrame.indexOf('applyRustSelectedFocusNavigation(navigation)')
    < rustSelectionFrame.indexOf('applyRustAppFocusToRenderer(navigation)'),
  'each Rust selection frame must apply navigation before the identity-checked renderer packet',
);

const sharedNavigationAdapter = browserSource.slice(
  browserSource.indexOf('function applyRustNavigationSnapshot('),
  browserSource.indexOf('function tickRustPresentation('),
);
for (const adapterStep of [
  'function applyRustPresentationNavigation(snapshot)',
  'function applyRustSelectedFocusNavigation(snapshot)',
  'selection.outputAnchor = selectedFocus?.output_pivot',
  'selection.outputRadius = Number.isFinite(selectedFocus?.output_radius)',
]) {
  assert.ok(
    sharedNavigationAdapter.includes(adapterStep),
    `shared navigation adapter is missing ${adapterStep}`,
  );
}

const controller = new HyperscopeNavigation();
const presentation = controller.loadPresentation(readFileSync(manifestPath, 'utf8'));
assert.equal(presentation.assets.length, 5);
assert.equal(presentation.cues.length, 8);
assert.deepEqual(
  presentation.assets.slice(2).map(asset => asset.uri),
  [
    '/polytopes/4-simplex.glb',
    '/polytopes/tesseract.glb',
    '/polytopes/16-cell.glb',
  ],
);
assert.deepEqual(
  presentation.cues.map(cue => cue.id),
  [
    'e0000000-0000-4000-8000-000000000007',
    'e0000000-0000-4000-8000-000000000008',
    'e0000000-0000-4000-8000-000000000009',
    'e0000000-0000-4000-8000-00000000000a',
    'e0000000-0000-4000-8000-000000000001',
    'e0000000-0000-4000-8000-000000000004',
    'e0000000-0000-4000-8000-000000000006',
    'e0000000-0000-4000-8000-000000000002',
  ],
);

const first = controller.startPresentation();
assert.equal(first.cue_id, presentation.cues[0].id);
assert.equal(first.cue_index, 0);
assert.deepEqual(first.animations, presentation.cues[0].animations);

for (const requiredAnimationAdapterStep of [
  'primaryPresentationAnimation(snapshot)',
  'selectAnimationIndex(clipIndex)',
  'animTime = time;',
  'animating_sig.set(playing);',
  'deltaSeconds * presentationAnimationSpeed',
  'toggleAnimationPlaybackIntent()',
  "e.code === 'Space'",
  '!e.repeat',
]) {
  assert.ok(
    browserSource.includes(requiredAnimationAdapterStep),
    `browser animation adapter is missing ${requiredAnimationAdapterStep}`,
  );
}

const lodAdapter = browserSource.slice(
  browserSource.indexOf('async function recomputeLods()'),
  browserSource.indexOf('async function loadModel('),
);
for (const requiredPoseGate of [
  'const lodPose = gpuSkinned ? animationPoseApplied : null;',
  'const lodPoseAnimated = lodPose !== null;',
  't: lodPose?.t ?? 0',
  'poseSampleTime: lodPose?.sampleTime ?? 0',
  'poseRevision: lodPose?.revision ?? 0',
  'poseContinuityEpoch: lodPose?.continuityEpoch ?? 0',
  'skipAnimation: !lodPoseAnimated',
  'capturePose: lodPoseAnimated && (',
  'RUST_ROUND_SHADOW_ENABLED && !roundAuthoredScene',
  'acceptLodPoseStamp(resp, lodPose);',
  'acceptLodDeltaSequence(resp, !!wt.full_snapshot);',
  'let resetDelta = lodDeltaResetPending;',
]) {
  assert.ok(
    lodAdapter.includes(requiredPoseGate),
    `LOD animation work must be gated by a resident deforming pose: ${requiredPoseGate}`,
  );
}
assert.ok(
  browserSource.includes('lodDeltaResetPending = true;'),
  'a rejected sparse publication must force a full worker snapshot',
);
for (const sameContextResidencyStep of [
  "initialBrowserParams, 'lodimpl', 'js'",
  "effectiveAuthority: 'worker'",
  'function uploadSameContextLodResidency(totalVertices, primaryFaces)',
  'mr_uploadComposedLodModel(totalVertices, primaryFaces)',
  "? 'resident-authority-ready'",
  ": 'resident-shadow'",
  'mr_dispatchSameContextLod(',
  "LOD_IMPLEMENTATION !== 'js' && sameContextReady",
  'sameContextDispatched ? sameContextRequest : 0',
  "resp.full_fingerprint || ''",
  "mr_pollSameContextLod(LOD_IMPLEMENTATION === 'rust')",
  "? 'renderer-context' : 'worker'",
  'if (sameContextDispatched) {',
  'lodDirty = true;',
  'sameContextWorkerFallbackActive',
  'A dormant worker delta encoder cannot extend renderer-authoritative',
  'lodDeltaResetPending = true;',
  'mr_recordSameContextLodAuthority(',
  'mr_recordSameContextLodBatchPublication(sameContextRequest)',
  'mr_cancelSameContextLod(sameContextRequest)',
  'sameContextLodDiagnostics.effectiveAuthority = \'worker\';',
]) {
  assert.ok(
    browserSource.includes(sameContextResidencyStep),
    `same-context LOD residency is missing ${sameContextResidencyStep}`,
  );
}
for (const rustResidencyStep of [
  'same_context_lod: Option<SameContextLod>',
  'prepare_lod_atlas_lookup(',
  '.lod_animation_source(total_vertices)',
  'build_composed_lod_model(',
  '.and_then(prepare_lod_model)',
  'LodCompute::new(state.renderer.gl(), model.residency.num_faces)',
  'clear_same_context_lod(state);',
  'pub fn mr_dispatch_same_context_lod(',
  'surface_runtime',
  '.lod_pose_source(',
  'pub fn mr_poll_same_context_lod(authoritative: bool)',
  'publish_same_context_lod_completion(state)',
  'authoritative_publications',
  'stale_authoritative_completions',
  'PACKED_LOD_OUTPUT_BYTES_PER_FACE',
  'packed_readback_bytes_per_face',
  'last_readback_bytes',
  'RetainedLodPublication::PackedWords',
  'admit_face_lod_classification_publication(',
  'legacy_float_decodes',
  'last_legacy_float_decode_bytes',
  'diff_packed_lod_classifications(',
  'packed_publication_noops',
  'packed_sparse_publications',
  'packed_changed_records',
  'packed_admission_skips',
  'last_packed_changed_records',
  'RetainedLodPublication::PackedWords(&changed_packed)',
  'last_packed_admission_skipped',
  'refresh_adaptive_picked_batches(state)',
  'same_context_lod_pose_continuity_matches(',
  'candidate.continuity_epoch == continuity_epoch',
  'apply_lod_classification_publication(',
  'same_context_lod_authority_stamp(',
  'publication_fingerprint_comparisons',
  'compare_lod_classifications(',
  'pub fn mr_record_same_context_lod_authority(',
  'pub fn mr_record_same_context_lod_batch_publication(',
  'SameContextLodBatchAuthoritySnapshot',
  'worker_batch_snapshot',
  'delayed worker batch snapshot has the wrong stamp',
  'try_compare_same_context_lod_batches(state)',
]) {
  assert.ok(
    mainRendererSource.includes(rustResidencyStep),
    `Rust same-context residency is missing ${rustResidencyStep}`,
  );
}
for (const requiredWorkerDeltaStep of [
  'if (resetDelta',
  'wasm.reset_animated_lod_delta();',
  'delta_epoch: result.delta_epoch',
  'delta_base_revision: result.delta_base_revision',
  'delta_revision: result.delta_revision',
  'pose_sample_time: result.pose_sample_time',
  'pose_revision: result.pose_revision',
  'pose_continuity_epoch: result.pose_continuity_epoch',
  'full_fingerprint: result.full_fingerprint',
  'stampedPoseEpoch !== lodPoseContinuityEpoch',
]) {
  assert.ok(
    workerSource.includes(requiredWorkerDeltaStep),
    `LOD worker is missing sequenced-delta recovery: ${requiredWorkerDeltaStep}`,
  );
}
assert.ok(
  !lodAdapter.includes('const lodAnimated = animating;'),
  'global playback intent must not classify a static asset as pose animated',
);
assert.ok(
  browserSource.includes("debouncedLodRecompute('primary-animation');")
    && browserSource.includes('animationPoseApplied = request;'),
  'LOD scheduling must originate from the exact pose accepted by the renderer',
);
const animationFrame = browserSource.slice(
  browserSource.indexOf('if (animating && !patchLab.active && gpuSkinned && meshInfo) {'),
  browserSource.indexOf('if (patchLab.active && patchLab.animate'),
);
assert.ok(
  !animationFrame.includes("debouncedLodRecompute('primary-animation')"),
  'the RAF clock must not independently sample a different LOD pose',
);

for (const requiredAnimatedAnchorStep of [
  'mr_attachSurfaceCameraAnchor',
  'mr_stepSurfaceCameraAnchor',
  'mr_transportSurfaceCameraAnchorReflection',
  'fixedAnchorRequested = e.shiftKey',
  'surfaceCameraAnchor.inputRevision !== manualCameraInputRevision',
  'ordinary camera controls remain surface-relative',
]) {
  assert.ok(
    browserSource.includes(requiredAnimatedAnchorStep),
    `hyperscope browser adapter is missing animated-anchor step: ${requiredAnimatedAnchorStep}`,
  );
}

for (const requiredAuthorityStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'presentimpl', 'rust',\n)",
  "presentimpl: 'rust'",
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

for (const requiredBakedSecondaryStep of [
  'function bakedPresentationNodeWorldTransforms(faceNodes)',
  'if (nodeWorldTransforms.length === 0 && info.has_hyperscape !== true)',
  'nodeWorldTransforms = bakedPresentationNodeWorldTransforms(faceNodes);',
]) {
  assert.ok(
    browserSource.includes(requiredBakedSecondaryStep),
    `ordinary baked presentation assets are missing ${requiredBakedSecondaryStep}`,
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
  'cadence.sceneClassifications += 1;',
  'cadence.lastSceneSubjectRecords = subjectRecords;',
  'cadence.lastSceneGpuPasses = gpuPasses;',
  'cadence.primaryAnimationClassifications += 1;',
  'cadence.lastPrimaryAnimationSubjectRecords = subjectRecords;',
  'cadence.lastPrimaryAnimationGpuPasses = gpuPasses;',
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
  'subject_records: result.subject_records',
  'gpu_passes: result.gpu_passes',
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

const patchMesh = load_patch_lab('triangle', 8, 0.55);
assert.equal(patchMesh.num_faces, 1);
const patchLabTwoToOne = update_patch_lab_lods('edges', 0, 0, 7, 1, 6, 6, 2);
const patchLabFourToOne = update_patch_lab_lods('edges', 0, 0, 7, 1, 6, 6, 4);
assert.deepEqual(Array.from(patchLabTwoToOne.requested), [2, 64, 64]);
assert.deepEqual(Array.from(patchLabTwoToOne.actual), [32, 64, 64]);
assert.equal(patchLabTwoToOne.promoted_edges, 1);
assert.equal(patchLabTwoToOne.shared_edge_mismatches, 0);
assert.equal(patchLabTwoToOne.policy_face_edge_ratio, 2);
assert.deepEqual(Array.from(patchLabFourToOne.requested), [2, 64, 64]);
assert.deepEqual(Array.from(patchLabFourToOne.actual), [16, 64, 64]);
assert.equal(patchLabFourToOne.promoted_edges, 1);
assert.equal(patchLabFourToOne.shared_edge_mismatches, 0);
assert.equal(patchLabFourToOne.policy_face_edge_ratio, 4);

console.log(JSON.stringify({
  cues: presentation.cues.length,
  initialCue: first.cue_id,
  linkedCue: linked.cue_id,
  rejectedCuePreserved: controller.presentationSnapshot().cue_id,
  patchLab: {
    requested: Array.from(patchLabTwoToOne.requested),
    twoToOne: Array.from(patchLabTwoToOne.actual),
    fourToOne: Array.from(patchLabFourToOne.actual),
  },
}));
