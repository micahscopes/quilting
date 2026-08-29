import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const manifestPath = `${repository}/crates/hyperscape/fixtures/hacker-night.presentation.json`;
const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
const animationControlSource = [
  'animation_control.rs',
  'animation_control/csr.rs',
].map(path => readFileSync(
  `${repository}/crates/hyperscope-web/src/${path}`,
  'utf8',
)).join('\n');
const presentationCardSource = [
  'presentation_card.rs',
  'presentation_card/csr.rs',
].map(path => readFileSync(
  `${repository}/crates/hyperscope-web/src/${path}`,
  'utf8',
)).join('\n');
const workerSource = readFileSync(`${repository}/hyperscope_worker.js`, 'utf8');
const mainRendererSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/main_renderer.rs`,
  'utf8',
);
const wasmFacadeSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/lib.rs`,
  'utf8',
);
const {
  default: init,
  HyperscopeNavigation,
  hyperscopeControlSpecs,
  load_gltf_data: loadGltfData,
  load_patch_lab,
  mr_acceptLodDeltaSequence,
  mr_measureRootGrouping,
  mr_incrementalRootGroupShadowDiagnostics,
  mr_setIncrementalRootGroupShadowEnabled,
  mr_resetRuntimeTimingDiagnostics,
  mr_runtimeTimingDiagnostics,
  mr_uploadComposedLodModel,
  set_active_animation_preserving_topology: setActiveAnimationPreservingTopology,
  update_patch_lab_lods,
} = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

assert.equal(typeof mr_uploadComposedLodModel, 'function');
assert.equal(typeof setActiveAnimationPreservingTopology, 'function');
assert.equal(typeof mr_measureRootGrouping, 'function');
assert.equal(typeof mr_incrementalRootGroupShadowDiagnostics, 'function');
assert.equal(typeof mr_setIncrementalRootGroupShadowEnabled, 'function');
assert.equal(typeof mr_runtimeTimingDiagnostics, 'function');
assert.equal(mr_measureRootGrouping(4), null);
assert.equal(typeof mr_resetRuntimeTimingDiagnostics, 'function');
assert.throws(() => mr_runtimeTimingDiagnostics(), /renderer is not initialized/);
assert.throws(() => mr_resetRuntimeTimingDiagnostics(), /renderer is not initialized/);
const animatedFixture = loadGltfData(new Uint8Array(readFileSync(`${repository}/horse.glb`)));
assert.ok(animatedFixture.animations.length > 0);
assert.equal(
  setActiveAnimationPreservingTopology(
    0,
    animatedFixture.num_vertices + 1,
    animatedFixture.num_faces,
  ),
  null,
  'a stale packed-topology witness must reject the clip switch',
);
const retainedTopology = setActiveAnimationPreservingTopology(
  0,
  animatedFixture.num_vertices,
  animatedFixture.num_faces,
);
assert.equal(retainedTopology.topology_preserved, true);
assert.equal(retainedTopology.num_vertices, animatedFixture.num_vertices);
assert.equal(retainedTopology.num_faces, animatedFixture.num_faces);
assert.deepEqual(
  hyperscopeControlSpecs().find(spec => spec.key === 'lodimpl'),
  { key: 'lodimpl', defaultValue: 'js', kind: 'implementation' },
);
assert.deepEqual(
  hyperscopeControlSpecs().find(spec => spec.key === 'presentimpl'),
  { key: 'presentimpl', defaultValue: 'rust', kind: 'implementation' },
);

assert.equal(mr_acceptLodDeltaSequence(91, 0, 1, true), true);
assert.equal(mr_acceptLodDeltaSequence(91, 1, 2, false), false);
assert.throws(
  () => mr_acceptLodDeltaSequence(91, 3, 4, false),
  /does not extend the resident base/,
);
assert.equal(mr_acceptLodDeltaSequence(92, 0, 1, true), true);

const cuePreparation = browserSource.slice(
  browserSource.indexOf('function prepareRustPresentationAction('),
  browserSource.indexOf('function applyRustPresentationCommit('),
);
const guardedSelectionClear = cuePreparation.indexOf(
  'if (selectedObject || rustAppShadowSelectionQueued)',
);
assert.ok(guardedSelectionClear >= 0);
assert.ok(
  guardedSelectionClear < cuePreparation.indexOf(
    'const applicationSynchronized = synchronizeRustApplicationPresentationFromBrowser();',
  ),
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
assert.deepEqual(
  presentation.cues.map(cue => controller.jumpToPresentationCue(cue.id).render_style),
  [
    'matcap_wire', 'wire', 'lod', 'normals',
    'matcap_wire', 'lod', 'stretch', 'pbr',
  ],
  'Rust presentation snapshots must resolve the backend-neutral render style',
);
controller.startPresentation();

assert.deepEqual(
  presentation.cues.map(cue => controller.jumpToPresentationCue(cue.id).tessellation),
  presentation.cues.map(cue => cue.tessellation),
  'Rust presentation snapshots must preserve every validated tessellation policy exactly',
);
controller.startPresentation();

const presentationVisualizationAdapter = browserSource.slice(
  browserSource.indexOf('const PRESENTATION_RENDER_MODES = Object.freeze({'),
  browserSource.indexOf('function applyPresentationTessellation('),
);
for (const thinAdapterStep of [
  'function applyPresentationVisualization(renderStyle, overlays = [])',
  'const mode = PRESENTATION_RENDER_MODES[renderStyle];',
  "matcap_wire: 'both'",
  "overlays.filter(overlay => overlay === 'control_net')",
]) {
  assert.ok(
    presentationVisualizationAdapter.includes(thinAdapterStep),
    `presentation render adapter is missing ${thinAdapterStep}`,
  );
}
assert.ok(
  browserSource.includes(
    'applyPresentationVisualization(snapshot.render_style, snapshot.overlays || []);',
  ),
  'the browser must consume the Rust-owned presentation render style',
);
assert.equal(
  browserSource.includes('PRESENTATION_SURFACE_VISUALIZATIONS'),
  false,
  'the browser must not retain an overlay-to-render-style policy table',
);

const presentationTessellationAdapter = browserSource.slice(
  browserSource.indexOf('function applyPresentationTessellation('),
  browserSource.indexOf('function applyPresentationLayerState('),
);
for (const exactAdapterStep of [
  'const density = tessellation?.density;',
  'const minPixels = tessellation?.min_pixels_per_subdivision;',
  'const screenAttenuation = tessellation?.screen_attenuation;',
  "throw new Error('invalid Rust presentation tessellation snapshot')",
]) {
  assert.ok(
    presentationTessellationAdapter.includes(exactAdapterStep),
    `presentation tessellation adapter is missing ${exactAdapterStep}`,
  );
}
for (const forbiddenBrowserPolicy of ['Math.max(', 'Math.min(', '|| 100', '|| 16']) {
  assert.equal(
    presentationTessellationAdapter.includes(forbiddenBrowserPolicy),
    false,
    `presentation tessellation adapter retained browser policy ${forbiddenBrowserPolicy}`,
  );
}

for (const requiredAnimationAdapterStep of [
  'primaryPresentationAnimation(snapshot)',
  '{ restoreRouteClock: false },',
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
const animationClipSwitchAdapter = browserSource.slice(
  browserSource.indexOf('async function selectAnimationIndex('),
  browserSource.indexOf("$('anim-sel').addEventListener('change'"),
);
for (const topologyPreservingStep of [
  'const preservePackedComposition = presentationComposition.ready;',
  'preserveTopology: preservePackedComposition,',
  'expectedVertices: expectedPrimaryVertices,',
  'expectedFaces: expectedPrimaryFaces,',
  'result.topology_preserved !== true',
  'rustPresentationDiagnostics.compositionPreservingClipSwitches += 1;',
]) {
  assert.ok(
    animationClipSwitchAdapter.includes(topologyPreservingStep),
    `packed presentation clip switching is missing ${topologyPreservingStep}`,
  );
}
assert.equal(
  browserSource.includes('presentation clip switching after scene packing is not supported'),
  false,
  'packed presentation composition must not reject a topology-preserving clip switch',
);
assert.ok(
  workerSource.includes('wasm.set_active_animation_preserving_topology('),
  'the worker must ask Rust to reject a stale packed-topology witness before mutation',
);
const topologyCheckedSwitch = wasmFacadeSource.slice(
  wasmFacadeSource.indexOf('fn set_active_animation_impl('),
  wasmFacadeSource.indexOf('pub fn evaluate_animation_frame('),
);
assert.ok(
  topologyCheckedSwitch.indexOf('expected_topology.is_some_and(')
    < topologyCheckedSwitch.indexOf('data.active_animation = index;'),
  'Rust must validate retained topology before replacing the active evaluator',
);
const animationPlaybackAdapter = browserSource.slice(
  browserSource.indexOf('function setAnimationPlaybackIntent('),
  browserSource.indexOf('function appShadowUuid('),
);
for (const directPlaybackStep of [
  'app.dispatchAnimationPlaying(requested)',
  "observeRustAppShadowSequence(receipt.sequence, 'Rust animation playback')",
  'app.dispatchAnimationToggle()',
  "observeRustAppShadowSequence(receipt.sequence, 'Rust animation toggle')",
]) {
  assert.ok(
    animationPlaybackAdapter.includes(directPlaybackStep),
    `browser playback adapter is missing store-allocated dispatch step: ${directPlaybackStep}`,
  );
}
for (const forbiddenPlaybackStep of [
  'app.setAnimationPlaying(++rustAppShadowSequence',
  'app.toggleAnimationPlaying(++rustAppShadowSequence',
]) {
  assert.equal(
    animationPlaybackAdapter.includes(forbiddenPlaybackStep),
    false,
    `browser playback adapter still allocates an application sequence: ${forbiddenPlaybackStep}`,
  );
}
const animationControlMountBoundary = browserSource.slice(
  browserSource.indexOf('rustAppShadow.mountAnimationControl('),
  browserSource.indexOf(
    'host.hidden = false;',
    browserSource.indexOf('rustAppShadow.mountAnimationControl('),
  ),
);
assert.ok(
  !animationControlMountBoundary.includes('toggleAnimationPlaybackIntent()')
    && animationControlMountBoundary.includes("observeRustAppShadowSequence(sequence, 'Rust animation control');")
    && animationControlMountBoundary.includes('animating_sig.set(Boolean(playing));')
    && animationControlMountBoundary.includes("'animation_control_rejection'"),
  'the Leptos animation callback must adapt committed Rust state without dispatching browser intent',
);
for (const directAnimationStep of [
  '.dispatch_semantic(SemanticAction::Animate(AnimationAction::TogglePlaying))',
  'store.frame_snapshot().animation.playing',
  'arguments.push(&JsValue::from(committed.sequence));',
  'arguments.push(&JsValue::from(committed.revision));',
]) {
  assert.ok(
    animationControlSource.includes(directAnimationStep),
    `Leptos animation control is missing direct Rust dispatch step: ${directAnimationStep}`,
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
  browserSource.includes('const POINTER_LOD_SETTLE_MS = 100;'),
  'pointer LOD topology must use the measured trailing settle boundary',
);
assert.equal(
  Array.from(browserSource.matchAll(/schedulePointerLodSettle\(\);/g)).length,
  2,
  'mouse drag and wheel must share one pointer LOD settle scheduler',
);
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
for (const promotionTelemetryStep of [
  'mr_runtimeTimingDiagnostics, mr_resetRuntimeTimingDiagnostics',
  'const snapshot = mr_runtimeTimingDiagnostics();',
  'mr_resetRuntimeTimingDiagnostics();',
]) {
  assert.ok(
    browserSource.includes(promotionTelemetryStep),
    `browser promotion telemetry is missing ${promotionTelemetryStep}`,
  );
}
assert.ok(
  !browserSource.includes('function recordFrameTiming('),
  'frame timing must not retain a duplicate JavaScript accumulator',
);
for (const rustTimingStep of [
  'render_timing: RenderTimingDiagnostics',
  'dispatch_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  'publication_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  'bucket_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  'vertex_lod_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  'render_node_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  'group_member_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>',
  '&"batchTiming".into()',
  'pub fn mr_runtime_timing_diagnostics()',
  'pub fn mr_reset_runtime_timing_diagnostics()',
]) {
  assert.ok(
    mainRendererSource.includes(rustTimingStep),
    `Rust promotion telemetry is missing ${rustTimingStep}`,
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
const presentationCardMountBoundary = browserSource.slice(
  browserSource.indexOf('rustAppShadow.mountPresentationCard('),
  browserSource.indexOf(
    'rustPresentationViewMounted = true;',
    browserSource.indexOf('rustAppShadow.mountPresentationCard('),
  ),
);
assert.ok(
  browserSource.includes('if (!presentationAppAuthority()) {\n    useBrowserPresentationView();')
    && !presentationCardMountBoundary.includes('activateRustPresentation(')
    && presentationCardMountBoundary.includes('() => prepareRustPresentationAction()')
    && presentationCardMountBoundary.includes('consumeRustPresentationCardCommit(')
    && presentationCardMountBoundary.includes('effect,')
    && presentationCardMountBoundary.includes('cancellations,')
    && presentationCardMountBoundary.includes("'presentation_view_rejection'"),
  'the Rust-authority Leptos card must dispatch directly while rollback lanes retain HTML controls',
);
for (const directPresentationStep of [
  '.dispatch_semantic(SemanticAction::Present(action.semantic()))',
  'prepare_callback.call1(',
  'activate_presentation_card(store, action)',
  'arguments.push(&JsValue::from(committed.sequence));',
  'arguments.push(&JsValue::from(committed.revision));',
  'presentation_clip_effect_to_js("select_animation_clip", effect)',
]) {
  assert.ok(
    presentationCardSource.includes(directPresentationStep),
    `Leptos presentation card is missing direct Rust dispatch step: ${directPresentationStep}`,
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
    < activationAdapter.indexOf('return applyRustPresentationCommit(snapshot, navigation, commit);'),
  'AppStore cue authority must commit before active-scene extraction during rendering',
);
const directPresentationAdapter = browserSource.slice(
  browserSource.indexOf('function dispatchAppPresentation('),
  browserSource.indexOf('function mirrorAppPresentation('),
);
assert.ok(
  directPresentationAdapter.includes('if (presentationAppAuthority())')
    && directPresentationAdapter.includes('rustAppShadow.dispatchPresentation(direction, cueId || \'\')')
    && directPresentationAdapter.includes("observeRustAppShadowSequence(receipt.sequence, 'Rust presentation adapter')")
    && directPresentationAdapter.includes('rustAppShadow.present('),
  'Rust presentation adapters must allocate inside AppStore while shadow replay stays explicitly sequenced',
);
const presentationCommitAdapter = browserSource.slice(
  browserSource.indexOf('function applyRustPresentationCommit('),
  browserSource.indexOf('function consumeRustPresentationCardCommit('),
);
assert.ok(
  presentationCommitAdapter.includes('renderRustPresentationSnapshot(snapshot);')
    && presentationCommitAdapter.includes('applyCommittedPresentationAnimationEffects(')
    && presentationCommitAdapter.includes('applyRustPresentationNavigation(navigation);'),
  'all cue paths must share one committed renderer/navigation adapter',
);
for (const residencyStep of [
  'rustAppShadow.bindPresentationAnimationResidency(',
  'await bindPrimaryPresentationAnimationResidency(primaryAsset.id);',
  "effect.type === 'select_animation_clip'",
  'committedClipJob: job,',
  'writeRustAnimationSample(rustAppShadow, null);',
]) {
  assert.ok(
    browserSource.includes(residencyStep),
    `presentation animation residency is missing ${residencyStep}`,
  );
}

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
