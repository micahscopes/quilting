import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runInNewContext } from 'node:vm';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const webGpuBackendSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/webgpu_backend.rs`,
  'utf8',
);
const mainRendererSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/main_renderer.rs`,
  'utf8',
);
const appShadowSource = readFileSync(
  `${repository}/crates/quilting-wasm/src/app_shadow.rs`,
  'utf8',
);
const appStoreSource = readFileSync(
  `${repository}/crates/hyperscope-app/src/lib.rs`,
  'utf8',
);
const cameraControlsSource = [
  `${repository}/crates/hyperscope-web/src/camera_controls.rs`,
  `${repository}/crates/hyperscope-web/src/camera_controls/csr.rs`,
].map(path => readFileSync(path, 'utf8')).join('\n');
const {
  default: init,
  canonicalizeHyperscopeRoute,
  hyperscopeControlSpecs,
} = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const specs = hyperscopeControlSpecs();
const rustDefaults = Object.fromEntries(
  specs.map(spec => [spec.key, spec.defaultValue]),
);
assert.equal(new Set(specs.map(spec => spec.key)).size, specs.length);
assert.equal(specs.find(spec => spec.key === 'mode').kind, 'render_mode');
assert.equal(specs.find(spec => spec.key === 'res').kind, 'resolution_level');
assert.deepEqual(specs.find(spec => spec.key === 'res').numericDomain, {
  minimum: 0,
  maximum: 6,
  integral: true,
  step: 1,
});
assert.equal(specs.find(spec => spec.key === 'density').kind, 'tessellation_density');
assert.deepEqual(specs.find(spec => spec.key === 'density').numericDomain, {
  minimum: 1,
  maximum: 500,
  integral: true,
  step: 1,
});
assert.equal(specs.find(spec => spec.key === 'minpx').kind, 'pixel_floor');
assert.equal(specs.find(spec => spec.key === 'minpx').defaultValue, '16');
assert.deepEqual(specs.find(spec => spec.key === 'minpx').numericDomain, {
  minimum: 1,
  maximum: 64,
  integral: false,
  step: 0.1,
});
assert.equal(specs.find(spec => spec.key === 'atlas').kind, 'atlas_exponent');
assert.deepEqual(specs.find(spec => spec.key === 'atlas').numericDomain, {
  minimum: 3,
  maximum: 9,
  integral: true,
  step: 1,
});
assert.deepEqual(specs.find(spec => spec.key === 'xform').choices, [
  'identity', 'sphere_reflection', 'rotation', 'translation',
]);
assert.deepEqual(specs.find(spec => spec.key === 'fmode').choices, ['0', '1', '2', '3']);
assert.deepEqual(specs.find(spec => spec.key === 'smnav').choices, [
  'hyperscope', 'object', 'fly', 'drone',
]);
assert.deepEqual(specs.find(spec => spec.key === 'anim').numericDomain, {
  minimum: -1,
  maximum: 2147483647,
  integral: true,
  step: 1,
});
assert.deepEqual(specs.find(spec => spec.key === 'lab').choices, [
  '0', 'triangle', 'plane', 'cube',
]);
assert.equal(specs.find(spec => spec.key === 'lodratio').defaultValue, '2');
assert.equal(specs.find(spec => spec.key === 'lodratio').kind, 'lod_ratio');
assert.equal(specs.find(spec => spec.key === 'appshadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'rendershadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'adaptiveshadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'rootgroupshadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'navstateimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'navstateimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'walkimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'navimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'navimpl').defaultValue, 'js');
assert.equal(specs.find(spec => spec.key === 'animclockimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'animclockimpl').defaultValue, 'js');
assert.equal(specs.find(spec => spec.key === 'animclipimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'animclipimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'animtime').kind, 'number');
assert.equal(specs.find(spec => spec.key === 'animtime').defaultValue, '0');
assert.equal(specs.find(spec => spec.key === 'animspeed').kind, 'number');
assert.equal(specs.find(spec => spec.key === 'animspeed').defaultValue, '1');
assert.equal(specs.find(spec => spec.key === 'aim').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'aim').defaultValue, '0');
assert.equal(specs.find(spec => spec.key === 'selasset').kind, 'optional_uuid');
assert.equal(specs.find(spec => spec.key === 'selentity').kind, 'optional_uuid');
assert.equal(specs.find(spec => spec.key === 'selectionimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'selectionimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'pickimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'pickimpl').defaultValue, 'js');
assert.equal(specs.find(spec => spec.key === 'presentimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'presentimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'gfxpresentimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'gfxpresentimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'assetimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'assetimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'sceneimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'sceneimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'routeimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'routeimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'renderstateimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'renderstateimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'cue').kind, 'optional_uuid');
assert.equal(specs.find(spec => spec.key === 'cue').defaultValue, '');
assert.equal(specs.find(spec => spec.key === 'fdebug').kind, 'choice');
assert.equal(specs.find(spec => spec.key === 'fdebug').defaultValue, '0');

for (const mode of ['pbr', 'matcap', 'wire', 'normals', 'both', 'lod', 'stretch']) {
  assert.equal(canonicalizeHyperscopeRoute([['mode', mode]]).diagnostics.length, 0);
}
for (const mode of ['matcap_wire', 'PBR', 'browser_magic']) {
  assert.equal(canonicalizeHyperscopeRoute([['mode', mode]]).diagnostics[0].code, 'invalid_value');
}

const typedRenderRoute = canonicalizeHyperscopeRoute([
  ['mode', 'both'],
  ['res', '4'],
  ['density', '237'],
  ['atten', '0'],
  ['minpx', '48.25'],
  ['atlas', '9'],
  ['lodratio', '4'],
]);
assert.deepEqual(typedRenderRoute.renderSettings, {
  style: 'matcap_wire',
  resolutionLevel: 4,
  density: 237,
  screenAttenuation: false,
  minPixelsPerSubdivision: 48.25,
  atlasExponent: 9,
  maxFaceEdgeRatio: 4,
  focusPostprocess: {
    enabled: false,
    mode: 1,
    diagnosticView: 0,
    blurRadiusPixels: 11,
    blurStrength: 3,
    focusCoordinate: 0.62,
    bandwidth: 0.1,
    normalizeRange: false,
    gaussianPasses: 1,
    kawasePasses: 3,
    kawaseOffset: 1.5,
  },
});
assert.equal(typedRenderRoute.resolvedPairs.length, specs.length);
assert.deepEqual(
  Object.fromEntries(typedRenderRoute.resolvedPairs),
  Object.fromEntries(specs.map(spec => [spec.key, spec.defaultValue]).concat([
    ['mode', 'both'],
    ['res', '4'],
    ['density', '237'],
    ['atten', '0'],
    ['minpx', '48.25'],
    ['atlas', '9'],
    ['lodratio', '4'],
  ])),
);
assert.equal(canonicalizeHyperscopeRoute([['mode', 'matcap_wire']]).renderSettings, undefined);
const typedSelectionRoute = canonicalizeHyperscopeRoute([
  ['selasset', '60000000-0000-4000-8000-00000000000A'],
  ['selentity', '70000000-0000-4000-8000-00000000000B'],
]);
assert.deepEqual(typedSelectionRoute.selection, {
  assetId: '60000000-0000-4000-8000-00000000000a',
  entityId: '70000000-0000-4000-8000-00000000000b',
});
assert.equal(canonicalizeHyperscopeRoute([]).selection, undefined);
assert.equal(canonicalizeHyperscopeRoute([]).animationClock, undefined);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['animtime', '0']]).animationClock,
  { timeSeconds: 0, speed: undefined },
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['animspeed', '-0.5']]).animationClock,
  { timeSeconds: undefined, speed: -0.5 },
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['animtime', '1.25'], ['animspeed', '-0.5']]).animationClock,
  { timeSeconds: 1.25, speed: -0.5 },
);
for (const [key, value] of [
  ['res', '7'], ['res', '3.5'],
  ['density', '0'], ['density', '12.5'], ['density', '501'],
  ['minpx', '0.99'], ['minpx', '64.01'],
  ['atlas', '2'], ['atlas', '9.5'], ['atlas', '10'],
]) {
  assert.equal(
    canonicalizeHyperscopeRoute([[key, value]]).diagnostics[0].code,
    'invalid_value',
    `${key}=${value} must fail Rust route admission`,
  );
}

const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
const renderControlsSource = [
  `${repository}/crates/hyperscope-web/src/render_controls.rs`,
  `${repository}/crates/hyperscope-web/src/render_controls/csr.rs`,
].map(path => readFileSync(path, 'utf8')).join('\n');
const legacyRouteNormalizerSource = browserSource.match(
  /function normalizeLegacyRouteShadow\(params\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(legacyRouteNormalizerSource, 'could not locate legacy route-shadow normalizer');
const normalizeLegacyRouteShadow = runInNewContext(
  `${legacyRouteNormalizerSource}; normalizeLegacyRouteShadow`,
  { URLSearchParams },
);
assert.equal(
  normalizeLegacyRouteShadow(new URLSearchParams('routeshadow=1')).toString(),
  'routeimpl=shadow',
);
assert.equal(
  normalizeLegacyRouteShadow(
    new URLSearchParams('routeshadow=1&routeimpl=js'),
  ).toString(),
  'routeimpl=shadow',
);
assert.equal(
  normalizeLegacyRouteShadow(
    new URLSearchParams('routeshadow=1&routeimpl=rust'),
  ).toString(),
  'routeimpl=rust',
);
assert.equal(
  normalizeLegacyRouteShadow(new URLSearchParams('routeshadow=0')).toString(),
  '',
);
assert.equal(
  specs.some(spec => spec.key === 'routeshadow'),
  false,
  'the legacy route-shadow alias must not remain a canonical Rust control',
);
const legacyNavigationNormalizerSource = browserSource.match(
  /function normalizeLegacyNavigationShadow\(params\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(
  legacyNavigationNormalizerSource,
  'could not locate legacy navigation-shadow normalizer',
);
const normalizeLegacyNavigationShadow = runInNewContext(
  `${legacyNavigationNormalizerSource}; normalizeLegacyNavigationShadow`,
  { URLSearchParams },
);
assert.equal(
  normalizeLegacyNavigationShadow(new URLSearchParams('navshadow=1')).toString(),
  'navimpl=shadow',
);
assert.equal(
  normalizeLegacyNavigationShadow(
    new URLSearchParams('navshadow=1&navimpl=js'),
  ).toString(),
  'navimpl=shadow',
);
assert.equal(
  normalizeLegacyNavigationShadow(
    new URLSearchParams('navshadow=1&navimpl=rust'),
  ).toString(),
  'navimpl=rust',
);
assert.equal(
  normalizeLegacyNavigationShadow(new URLSearchParams('navshadow=0')).toString(),
  '',
);
assert.equal(
  specs.some(spec => spec.key === 'navshadow'),
  false,
  'the legacy navigation-shadow alias must not remain a canonical Rust control',
);
const policyDefaultsSource = browserSource.match(
  /const BOOTSTRAP_POLICY_DEFAULTS = Object\.freeze\((\{[\s\S]*?\n\})\);/,
)?.[1];
assert.ok(policyDefaultsSource, 'could not locate pre-WASM adapter policy defaults');
const policyDefaults = JSON.parse(JSON.stringify(
  runInNewContext(`(${policyDefaultsSource})`),
));
assert.deepEqual(Object.keys(policyDefaults), [
  'gfx',
  'pickimpl',
  'presentation',
  'presentimpl',
  'gfxpresentimpl',
  'roundshadow',
  'selectionimpl',
  'sceneimpl',
  'assetimpl',
  'animclockimpl',
  'animclipimpl',
  'renderstateimpl',
  'patchlabimpl',
  'appshadow',
  'routeimpl',
  'rendershadow',
  'lodimpl',
  'adaptiveshadow',
  'rootgroupshadow',
  'navstateimpl',
  'walkimpl',
  'navimpl',
]);
for (const [key, value] of Object.entries(policyDefaults)) {
  assert.equal(
    rustDefaults[key],
    value,
    `Rust route default for ${key} drifted from pre-WASM adapter policy`,
  );
}
const policyDefaultsDeclaration =
  `const BOOTSTRAP_POLICY_DEFAULTS = Object.freeze(${policyDefaultsSource});`;
const implementationFromRouteSource = browserSource.match(
  /function implementationFromRoute\(params, key\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(implementationFromRouteSource, 'could not locate implementation-mode parser');
const implementationFromRoute = runInNewContext(
  `${policyDefaultsDeclaration}\n${implementationFromRouteSource}; implementationFromRoute`,
);
for (const implementation of ['js', 'shadow', 'rust']) {
  assert.equal(
    implementationFromRoute(
      new URLSearchParams(`navimpl=${implementation}`),
      'navimpl',
    ),
    implementation,
  );
  assert.equal(
    implementationFromRoute(
      new URLSearchParams(`navstateimpl=${implementation}`),
      'navstateimpl',
    ),
    implementation,
  );
  assert.deepEqual(
    canonicalizeHyperscopeRoute([['navstateimpl', implementation]]).pairs,
    implementation === 'rust' ? [] : [['navstateimpl', implementation]],
  );
}
assert.equal(
  implementationFromRoute(new URLSearchParams(), 'navimpl'),
  'js',
);
assert.equal(
  implementationFromRoute(new URLSearchParams(), 'navstateimpl'),
  'rust',
);
assert.equal(
  implementationFromRoute(new URLSearchParams(), 'routeimpl'),
  'rust',
);
assert.equal(
  implementationFromRoute(new URLSearchParams('navimpl=invalid'), 'navimpl'),
  'js',
);
assert.equal(
  implementationFromRoute(new URLSearchParams('navstateimpl=invalid'), 'navstateimpl'),
  'rust',
);
assert.throws(
  () => implementationFromRoute(new URLSearchParams(), 'missingimpl'),
  /No implementation bootstrap policy/,
);
const graphicsBackendFromRouteSource = browserSource.match(
  /function graphicsBackendFromRoute\(params\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(graphicsBackendFromRouteSource, 'could not locate graphics-backend parser');
const graphicsBackendFromRoute = runInNewContext(
  `${policyDefaultsDeclaration}\n${graphicsBackendFromRouteSource}; graphicsBackendFromRoute`,
);
assert.equal(graphicsBackendFromRoute(new URLSearchParams()), 'webgl2');
assert.equal(
  graphicsBackendFromRoute(new URLSearchParams('gfx=webgpu-shadow')),
  'webgpu-shadow',
);
assert.equal(graphicsBackendFromRoute(new URLSearchParams('gfx=invalid')), 'webgl2');
const bootstrapToggleFromRouteSource = browserSource.match(
  /function bootstrapToggleFromRoute\(params, key\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(bootstrapToggleFromRouteSource, 'could not locate bootstrap-toggle parser');
const bootstrapToggleFromRoute = runInNewContext(
  `${policyDefaultsDeclaration}\n${bootstrapToggleFromRouteSource}; bootstrapToggleFromRoute`,
);
assert.equal(bootstrapToggleFromRoute(new URLSearchParams(), 'presentation'), false);
assert.equal(
  bootstrapToggleFromRoute(new URLSearchParams('presentation=1'), 'presentation'),
  true,
);
assert.equal(
  bootstrapToggleFromRoute(new URLSearchParams('presentation=invalid'), 'presentation'),
  false,
);
assert.throws(
  () => bootstrapToggleFromRoute(new URLSearchParams(), 'gfx'),
  /No toggle bootstrap policy/,
);
const canonicalFixedSource = browserSource.match(
  /function canonicalFixedRouteNumber\(value, fractionDigits\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(canonicalFixedSource, 'could not locate fixed route-number canonicalizer');
const canonicalFixedRouteNumber = runInNewContext(
  `${canonicalFixedSource}; canonicalFixedRouteNumber`,
);
assert.equal(canonicalFixedRouteNumber(-0, 3), '0.000');
assert.equal(canonicalFixedRouteNumber(-0.0004, 3), '0.000');
assert.equal(canonicalFixedRouteNumber(0.0004, 3), '0.000');
assert.equal(canonicalFixedRouteNumber(-0.0006, 3), '-0.001');
assert.equal(canonicalFixedRouteNumber(1.25, 2), '1.25');
for (const routeDefaultStep of [
  "implementationFromRoute(\n  initialRouteParams, 'routeimpl',\n)",
]) {
  assert.ok(
    browserSource.includes(routeDefaultStep),
    `browser route default is missing ${routeDefaultStep}`,
  );
}
for (const selectionDefaultStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'selectionimpl',\n)",
  "implementationFromRoute(\n  initialBrowserParams, 'pickimpl',\n)",
]) {
  assert.ok(
    browserSource.includes(selectionDefaultStep),
    `browser selection default is missing ${selectionDefaultStep}`,
  );
}
for (const animationClockDefaultStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'animclockimpl',\n)",
  "implementationFromRoute(\n  initialBrowserParams, 'animclipimpl',\n)",
]) {
  assert.ok(
    browserSource.includes(animationClockDefaultStep),
    `browser animation-clock default is missing ${animationClockDefaultStep}`,
  );
}
for (const animationClockAuthorityStep of [
  "!pendingRouteAnimationClock\n        && (!RUST_PRESENTATION_ENABLED",
  "compareRustAnimationSample('frame')",
  "applyRustAnimationSample('frame')",
  'app.writeInstalledAnimationSample(installedAnimationSamplePacket);',
  'const range = animationClipRangeForClock(app);',
  "throw new Error('Rust animation frame sampled a nonresident clip');",
  "ANIMATION_CLOCK_IMPLEMENTATION === 'rust'",
  "restorePendingRouteAnimationClock('clip-switch')",
  "restorePendingRouteAnimationClock('presentation-animation')",
  '{ restoreRouteClock: false },',
  'await selectAnimationIndex(animIdx);',
  "restorePendingRouteAnimationClock('startup-animation')",
  "decoded.animtimeProvided = explicitParams.has('animtime');",
  "decoded.animspeedProvided = explicitParams.has('animspeed');",
  "const receipt = app.dispatchAnimationClock(playing, timeSeconds, speed);",
  'observeRustAppShadowSequence(receipt.sequence, context);',
  'return app.setAnimationClock(',
  'const receipt = app.dispatchAnimationSeek(timeSeconds);',
  'return app.seekAnimation(++rustAppShadowSequence, timeSeconds);',
]) {
  assert.ok(
    browserSource.includes(animationClockAuthorityStep),
    `browser animation-clock adapter is missing ${animationClockAuthorityStep}`,
  );
}
for (const animationClipAuthorityStep of [
  "ANIMATION_CLIP_IMPLEMENTATION === 'rust'",
  "const receipt = app.requestAnimationClip(index);",
  'const effect = receipt.selection;',
  'const cancellations = receipt.cancellations;',
  'if (!receipt.matchesRequest)',
  'rustAppShadow.finishAnimationClipSelected(',
  'rustAppShadow.finishAnimationClipSelectionFailed(',
  'beginAppAnimationClipSelection(idx)',
  'completeAppAnimationClipSelection(clipJob);',
  'let rendererSwitched = false;',
  'if (!rendererSwitched) {',
  'Animation switched, but Rust completion failed:',
  'globalThis.__hyperscopeAnimationClipDiagnostics = animationClipDiagnostics;',
  "observeAnimationClipResidency('scene-installed', snapshot);",
  "observeAnimationClipResidency('selection-pending', residencySnapshot);",
  "error ? 'selection-failed' : 'selection-complete',",
  '{ animationClipSelection: receipt.selection },',
  'clipJob.cancellations.length === 0',
  'animationClipDiagnostics.repairs += 1;',
  'rendererAnimationClipIndex = null;',
  'rendererAnimationClipIndex = idx;',
  'rustAppShadow.mountAnimationClipControl(',
  "rustAppShadowDiagnostics.animationClipControlAuthority = 'hyperscope-web';",
  'committedClipJob: { effect, cancellations },',
  "throw new Error('only Rust clip authority may supply a committed clip job');",
]) {
  assert.ok(
    browserSource.includes(animationClipAuthorityStep),
    `browser animation-clip authority is missing ${animationClipAuthorityStep}`,
  );
}
const animationClipRequestBoundary = browserSource.slice(
  browserSource.indexOf('function beginAppAnimationClipSelection(index) {'),
  browserSource.indexOf('function completeAppAnimationClipSelection(',
));
for (const retiredAnimationClipParsing of [
  'receipt.commit.effects.filter(',
  "effect.type === 'select_animation_clip'",
  "effect.type === 'cancel_animation_clip_selection'",
  'snapshot?.animationClipSelection?.active?.clip?.index',
]) {
  assert.equal(
    animationClipRequestBoundary.includes(retiredAnimationClipParsing),
    false,
    `browser clip request must not rediscover ${retiredAnimationClipParsing}`,
  );
}
for (const navigationAuthorityStep of [
  "implementationFromRoute(\n  initialNavigationParams, 'navimpl',\n)",
  "RUST_NAVIGATION_IMPLEMENTATION !== 'js'",
  'rustAppShadow.stepSpaceMouseCamera(',
  'rustAppShadow.stepPointerCamera(',
  'rustAppShadow.aimAtSelection(',
  'dispatchAppSelectedCameraTransition(',
  'compareAppSelectedCameraTransitionFrame(',
  'rustAppSelectedCameraTransitionKind',
  'spaceMouseAxes,\n          navigationMode,',
  "RUST_NAVIGATION_IMPLEMENTATION === 'rust' && rustCameraReady",
  'applyRustManualCameraPacket(rustManualCameraPacket)',
  "RUST_NAVIGATION_IMPLEMENTATION === 'shadow' && rustCameraReady",
  "recordRustManualCameraComparison(rustManualCameraPacket, 'SpaceMouse')",
  'integratePointerCamera(dx, dy, e.shiftKey ? 1 : 0);',
  'integratePointerCamera(0, e.deltaY, 2);',
  '&& !rustManualCameraActive) {',
  'An idle HID report does not make the Rust camera stale.',
  'rustAppShadowDiagnostics.cameraFallbackWrites += 1;',
]) {
  assert.ok(
    browserSource.includes(navigationAuthorityStep),
    `browser navigation authority adapter is missing ${navigationAuthorityStep}`,
  );
}
for (const focusSphereAuthorityStep of [
  'function dispatchRustFocusSphereGeometry(sphere, preserveAnchor, source)',
  'function dispatchManualFocusSphereControl(id, requested)',
  'function dispatchManualInversionControl(requested)',
  'rustAppShadow.setInversionEnabled(enabled);',
  'rustAppShadow.editFocusSphere(',
  'const navigation = rustAppShadow.tickNavigation(0);',
  '? applyRustSelectedFocusNavigation(navigation)',
  ': applyRustNavigationSnapshot(navigation);',
  "{ center: nextCenter, radius: nextRadius },\n          'spacemouse-inversion',",
  "}, 'focus-wheel');",
  "requested => dispatchManualFocusSphereControl('mr', requested)",
  "bindBtnGroup('xform-btns', mob.xform, dispatchManualInversionControl);",
  'rustNavigationProjectionDepth > 0',
  'const reflectionMobius = packedMobiusOrNull(snapshot.reflection_mobius);',
  '? rustProjectedReflectionMobius',
  'const retainsSelectedAnchor = Boolean(rustGesture.navigation?.selected_focus);',
]) {
  assert.ok(
    browserSource.includes(focusSphereAuthorityStep),
    `browser focus-sphere authority adapter is missing ${focusSphereAuthorityStep}`,
  );
}
assert.ok(
  appShadowSource.includes('#[wasm_bindgen(js_name = editFocusSphere)]')
    && appShadowSource.includes('.dispatch_focus_sphere_edit(target, preserve_anchor)'),
  'generated WASM facade must delegate focus-sphere semantics to AppStore',
);
assert.ok(
  appShadowSource.includes('frame.reflection.mobius().coefficients_f32()'),
  'generated WASM navigation packet must carry the Rust-authored conformal chart',
);
assert.ok(
  appStoreSource.includes('pub fn dispatch_focus_sphere_edit(')
    && appStoreSource.includes('NavigationAction::ScaleFocusLog(ratio.ln())')
    && appStoreSource.includes('NavigationAction::SetFreeFocusSphere(target)'),
  'AppStore must atomically distinguish anchored and detached focus-sphere edits',
);
for (const interactionBoundaryStep of [
  '#[wasm_bindgen(js_name = setInteractionHover)]',
  '#[wasm_bindgen(js_name = replaceInteractionTargets)]',
  '#[wasm_bindgen(js_name = setPackedInteractionHover)]',
  '#[wasm_bindgen(js_name = clearInteractionHover)]',
  '#[wasm_bindgen(js_name = pressInteractionPrimary)]',
  '#[wasm_bindgen(js_name = releaseInteractionPrimary)]',
  '#[wasm_bindgen(js_name = cancelInteractionPrimary)]',
  '#[wasm_bindgen(js_name = interactionSnapshot)]',
  '#[wasm_bindgen(js_name = recordBackendPickStage)]',
  '#[wasm_bindgen(js_name = recordBackendPickEvidence)]',
  '#[wasm_bindgen(js_name = recordBackendPickError)]',
  '#[wasm_bindgen(js_name = backendPickDiagnostics)]',
  '#[wasm_bindgen(js_name = stageBackendPickEvidence)]',
  '#[wasm_bindgen(js_name = readBackendPickEvidence)]',
  '.dispatch_semantic(SemanticAction::Interact(action))',
]) {
  assert.ok(
    appShadowSource.includes(interactionBoundaryStep),
    `generated WASM interaction boundary is missing ${interactionBoundaryStep}`,
  );
}
for (const exactPickStep of [
  '.sample_face_in_parent_chart(',
  '.output_patch_for_face(',
  '("source_position", surface.source_position)',
  '("output_position", surface.output_position)',
]) {
  assert.ok(
    mainRendererSource.includes(exactPickStep),
    `renderer interaction pick is missing ${exactPickStep}`,
  );
}
for (const interactionAuthorityStep of [
  'interaction: InteractionController,',
  'SemanticAction::Interact(action) => {',
  '.advance_to(frame.elapsed_seconds, &self.navigation.focus)',
  'activation.navigation_action(self.interaction.policy)',
  'InteractionSnapshot::from_state(',
]) {
  assert.ok(
    appStoreSource.includes(interactionAuthorityStep),
    `AppStore interaction authority is missing ${interactionAuthorityStep}`,
  );
}
for (const browserInteractionShadowStep of [
  'function mirrorSelectedObjectInteractionToApp(nowMs = performance.now())',
  "RUST_SELECTION_IMPLEMENTATION !== 'shadow'",
  'function replaceRustInteractionTargets()',
  'rustAppShadow.replaceInteractionTargets(JSON.stringify(targets))',
  'rustAppShadow.setPackedInteractionHover(',
  'rustAppShadow.setInteractionHover(',
  'rustAppShadow.pressInteractionPrimary();',
  'rustAppShadow.releaseInteractionPrimary();',
  'const interaction = rustAppShadow.interactionSnapshot();',
  'function mirrorSelectedObjectDirectToApp(nowMs = performance.now())',
  'const pickedSurface = pickSurfaceAtCanvasPixel(x, y);',
  'pickedSurface?.source_position',
  'pickedSurface?.output_position',
  'globalThis.__hyperscopeInteractionDiagnostics = rustInteractionDiagnostics;',
  'globalThis.__hyperscopeBackendPickDiagnostics = backendPickDiagnostics;',
  "PICK_IMPLEMENTATION !== 'js'",
  'const staged = app.stageBackendPickEvidence(',
  'Promise.resolve(app.readBackendPickEvidence())',
  'updateBackendPickDiagnostics(app.backendPickDiagnostics())',
]) {
  assert.ok(
    browserSource.includes(browserInteractionShadowStep),
    `browser interaction shadow is missing ${browserInteractionShadowStep}`,
  );
}
assert.equal(
  browserSource.includes('Number(report.targetEpoch) !== currentTargetEpoch'),
  false,
  'browser glue must not own retained-pick target-epoch rejection',
);
for (const retiredBrowserPickShuttle of [
  'quiltingWasmBackend.mr_stageBackendPickEvidence(',
  'quiltingWasmBackend.mr_readBackendPickEvidence()',
  'const targetEpoch = rustInteractionDiagnostics.targetEpoch;',
]) {
  assert.equal(
    browserSource.includes(retiredBrowserPickShuttle),
    false,
    `browser glue must not retain ${retiredBrowserPickShuttle}`,
  );
}
for (const navigationSettingsAuthorityStep of [
  "implementationFromRoute(\n  initialNavigationParams, 'navstateimpl',\n)",
  "RUST_NAVIGATION_SETTINGS_IMPLEMENTATION !== 'js'",
  'function browserNavigationSettingsState()',
  'app.synchronizeNavigationSettings(',
  'if (!receipt.matchesInput)',
  'function applyRustNavigationSettingsProjection(navigation)',
  'function ensureRustNavigationSettingsView()',
  'rustAppShadow.mountNavigationControls(',
  "rustNavigationSettingsDiagnostics.viewAuthority = 'hyperscope-web';",
  "RUST_NAVIGATION_SETTINGS_IMPLEMENTATION === 'rust'",
  'batchSignals(() => {',
  'scheduleRustNavigationSettingsSynchronization();',
  "set(\n        'navstateimpl',",
  'globalThis.__hyperscopeNavigationSettings = rustNavigationSettingsDiagnostics;',
]) {
  assert.ok(
    browserSource.includes(navigationSettingsAuthorityStep),
    `browser navigation-settings adapter is missing ${navigationSettingsAuthorityStep}`,
  );
}
assert.equal(
  browserSource.includes('function navigationSettingsContentEqual(left, right)'),
  false,
  'browser glue must not own navigation-settings equality',
);
assert.equal(
  browserSource.includes('app.setNavigationSettings('),
  false,
  'browser glue must not allocate explicitly sequenced navigation-settings events',
);
for (const graphicsBackendStep of [
  "import * as quiltingWasmBackend from './pkg/quilting_wasm.js';",
  "const GRAPHICS_BACKEND_REQUEST = graphicsBackendFromRoute(initialBrowserParams);",
  "set('gfx', GRAPHICS_BACKEND_REQUEST, PARAM_DEFAULTS.gfx);",
  'await quiltingWasmBackend.mr_initWebGpuBackend()',
  'await quiltingWasmBackend.mr_initWebGpuPresentation(',
  'quiltingWasmBackend.mr_uploadWebGpuComposedModel(',
  "graphicsBackendDiagnostics.state = 'presentation-ready';",
  'graphicsBackendDiagnostics.state = decision.phase;',
  'function webGpuPresentationSupportsRenderMode(mode, residency = null)',
  'residency?.presentationStyle === graphicsBackendDiagnostics.renderMode',
  "presentationCanvas.classList.toggle('webgpu-presenting', presenting);",
  "webglCanvas.classList.toggle('webgpu-input-layer', presenting);",
  "globalThis.__hyperscopeGraphicsBackend = graphicsBackendDiagnostics;",
  'return refreshWebGpuBackendDiagnostics();',
]) {
  assert.ok(
    browserSource.includes(graphicsBackendStep),
    `browser graphics backend adapter is missing ${graphicsBackendStep}`,
  );
}
const webGpuModeSupportSource = browserSource.match(
  /function webGpuPresentationSupportsRenderMode\(mode, residency = null\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(webGpuModeSupportSource, 'could not locate WebGPU mode support predicate');
const webGpuPresentationSupportsRenderMode = runInNewContext(
  `const graphicsBackendDiagnostics = { focusPostprocessRequested: false };
  ${webGpuModeSupportSource}; webGpuPresentationSupportsRenderMode`,
);
for (const mode of ['matcap', 'wire', 'normals', 'both', 'lod', 'stretch']) {
  assert.equal(webGpuPresentationSupportsRenderMode(mode), true, `${mode} should use WebGPU`);
}
assert.equal(webGpuPresentationSupportsRenderMode('pbr'), false);
assert.equal(webGpuPresentationSupportsRenderMode('pbr', {
  pbrPresentationReady: true,
}), true, 'resident PBR should arm its first WebGPU presentation');
assert.equal(webGpuPresentationSupportsRenderMode('pbr', {
  pbrPresentationReady: false,
  presentationStyle: 'wire',
  presentationFrames: 1,
}), false, 'nonresident PBR must not arm from another style retained frame');
const webGpuFocusPresentationSupportsRenderMode = runInNewContext(
  `const graphicsBackendDiagnostics = { focusPostprocessRequested: true };
  ${webGpuModeSupportSource}; webGpuPresentationSupportsRenderMode`,
);
assert.equal(
  webGpuFocusPresentationSupportsRenderMode('matcap', {
    focusPipelineReady: true,
    focusSceneReady: true,
    environmentReady: true,
  }),
  false,
  'focus post-processing must not claim unsupported diagnostic composition',
);
assert.equal(
  webGpuFocusPresentationSupportsRenderMode('pbr', {
    pbrPresentationReady: true,
    focusPipelineReady: true,
    focusSceneReady: true,
    environmentReady: true,
  }),
  true,
  'ready PBR focus composition should remain presentable',
);
assert.ok(
  browserSource.includes(
    'residency?.presentationStyle === graphicsBackendDiagnostics.renderMode',
  ) && browserSource.includes('(residency?.presentationFrames || 0) > 0'),
  'canvas promotion must still require a submitted frame in the requested style',
);
for (const stalePresentationGuard of [
  'fn incumbent_required(&mut self) -> LiveFrameDisposition',
  'self.last_frame_input = None;',
  'scene.supports_resident_patch_presentation_frame(style, options)',
]) {
  assert.ok(
    webGpuBackendSource.includes(stalePresentationGuard),
    `WebGPU stale-presentation guard is missing ${stalePresentationGuard}`,
  );
}
for (const sharedFramePlanStep of [
  'fn refresh_render_command_plan(renderer: &mut MainState, backend_plan_required: bool)',
  'fn webgpu_render_style_requested(',
  'fn webgpu_frame_requested(renderer: &MainState) -> bool',
  'quilting_webgpu::supports_patch_presentation_style(style)',
  'crate::webgpu_backend::live_presentation_requested()',
  'RenderCommandPlan::build(scene, style, options)',
  'fn current_render_frame(',
  'RenderFrame::from_command_plan(',
]) {
  assert.ok(
    mainRendererSource.includes(sharedFramePlanStep),
    `main renderer shared frame-plan path is missing ${sharedFramePlanStep}`,
  );
}
for (const borrowedFramePlanStep of [
  'frame: &RenderFrame',
  '.execution(scene.scene())',
  'validated_scene().shares_snapshot_with(scene)',
]) {
  assert.ok(
    webGpuBackendSource.includes(borrowedFramePlanStep),
    `WebGPU borrowed frame-plan path is missing ${borrowedFramePlanStep}`,
  );
}
assert.equal(
  webGpuBackendSource.includes('command_plan: Option<RenderCommandPlan>'),
  false,
  'WebGPU must not retain a duplicate browser command-plan cache',
);
assert.equal(
  webGpuBackendSource.includes('RenderCommandPlan::build('),
  false,
  'WebGPU must not rebuild the main renderer command plan',
);
assert.equal(
  webGpuBackendSource.includes('RenderFrame::from_command_plan('),
  false,
  'WebGPU must not reconstruct the main renderer frame',
);
assert.equal(
  webGpuBackendSource.includes('RenderFrame::build('),
  false,
  'ordinary WebGPU browser frames must not rebuild and revalidate commands',
);
for (const selectedFaceEvidenceStep of [
  'fn backend_frame_evidence_supports_composition(',
  'crate::webgpu_backend::focus_evidence_prerequisites_ready()',
  'focus backend evidence requires WebGPU focus pipelines and environment residency',
  'backend evidence focus packets do not match',
  'render_highlight_to(',
  'Some(target)',
  'render_highlight(state.renderer.gl(), state, &camera);',
  'backend image evidence requires resident WebGL highlight resources',
]) {
  assert.ok(
    mainRendererSource.includes(selectedFaceEvidenceStep),
    `WebGPU selected-face evidence is missing ${selectedFaceEvidenceStep}`,
  );
}
const routeDefaultsAdapterSource = browserSource.match(
  /const BOOTSTRAP_PARAM_DEFAULTS = Object\.freeze\(\{[\s\S]*?\n\}\);\nlet PARAM_DEFAULTS = BOOTSTRAP_PARAM_DEFAULTS;\n\nfunction installRustControlDefaults\(specs\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(routeDefaultsAdapterSource, 'could not locate Rust-default installation adapter');
assert.equal(
  browserSource.includes('const PARAM_DEFAULTS = {'),
  false,
  'the browser must not retain a complete route-default authority',
);
const installRouteDefaults = specsInput => runInNewContext(
  `${policyDefaultsDeclaration}\n${routeDefaultsAdapterSource}; input => {
    const installed = installRustControlDefaults(input);
    return { frozen: Object.isFrozen(installed), entries: Object.entries(installed) };
  }`,
)(specsInput);
const installedDefaults = installRouteDefaults(specs);
assert.equal(installedDefaults.frozen, true);
assert.deepEqual(
  Object.fromEntries(JSON.parse(JSON.stringify(installedDefaults.entries))),
  rustDefaults,
  'the browser must install every Rust control default without another policy',
);
const bootstrapDefaultsSource = browserSource.match(
  /const BOOTSTRAP_PARAM_DEFAULTS = Object\.freeze\((\{[\s\S]*?\n\})\);/,
)?.[1];
assert.ok(bootstrapDefaultsSource, 'could not locate pre-WASM bootstrap defaults');
const bootstrapDefaults = JSON.parse(JSON.stringify(
  runInNewContext(`(${bootstrapDefaultsSource})`),
));
assert.equal(Object.keys(bootstrapDefaults).length, 10);
const preWasmStartup = browserSource.slice(
  browserSource.indexOf('// --- Init ---'),
  browserSource.indexOf("phase('wasm', [], async () =>"),
);
assert.deepEqual(
  Array.from(new Set(
    Array.from(preWasmStartup.matchAll(/initParams\.([A-Za-z0-9_]+)/g), match => match[1]),
  )),
  Object.keys(bootstrapDefaults),
  'the pre-WASM preview must consume only its explicit bootstrap defaults',
);
for (const [key, value] of Object.entries(bootstrapDefaults)) {
  assert.equal(
    rustDefaults[key],
    value,
    `Rust route default for ${key} drifted from the inert browser preview`,
  );
}
assert.throws(() => installRouteDefaults([]), /empty control registry/);
assert.throws(
  () => installRouteDefaults(specs.concat(specs[0])),
  /duplicate control key/,
);
assert.throws(
  () => installRouteDefaults(specs.filter(spec => spec.key !== 'glb')),
  /omitted bootstrap key/,
);
assert.throws(
  () => installRouteDefaults(specs.map(spec => (
    spec.key === 'gfx' ? { ...spec, defaultValue: 2 } : spec
  ))),
  /non-string default/,
);
assert.throws(
  () => installRouteDefaults(specs.map(spec => (
    spec.key === 'zoom' ? { ...spec, defaultValue: '4' } : spec
  ))),
  /drifted from bootstrap/,
);
assert.throws(
  () => installRouteDefaults(specs.map(spec => (
    spec.key === 'navimpl' ? { ...spec, defaultValue: 'rust' } : spec
  ))),
  /drifted from bootstrap/,
);
assert.throws(
  () => installRouteDefaults(specs.map(spec => (
    spec.key === 'navstateimpl' ? { ...spec, defaultValue: 'js' } : spec
  ))),
  /drifted from bootstrap/,
);
const implicitBrowserDefaults = {
  presentation: '0',
  roundshadow: '0',
  appshadow: '0',
  rendershadow: '0',
  adaptiveshadow: '0',
  rootgroupshadow: '0',
};
assert.deepEqual(
  Object.fromEntries(
    Object.keys(implicitBrowserDefaults).map(key => [key, rustDefaults[key]]),
  ),
  implicitBrowserDefaults,
  'Rust implicit flag defaults drifted from the browser rollback',
);
const syncSource = browserSource.match(
  /function syncURL\(\) \{([\s\S]*?)\/\/ Apply URL params to controls on load/,
)?.[1];
assert.ok(syncSource, 'could not locate browser URL serializer');
assert.equal(
  Array.from(syncSource.matchAll(/canonicalFixedRouteNumber\(/g)).length,
  18,
  'every camera/animation value and default must canonicalize signed zero before comparison',
);
const browserKeyOrder = Array.from(
  syncSource.matchAll(/(?:set|ss)\(\s*'([^']+)'/g),
  match => match[1],
);
assert.deepEqual(
  specs.map(spec => spec.key),
  browserKeyOrder,
  'Rust route order/default registry drifted from the browser oracle',
);
for (const authorityStep of [
  "RUST_ROUTE_IMPLEMENTATION === 'rust'",
  'committedParams = new URLSearchParams(rustRoute.pairs);',
  "rustRouteShadowDiagnostics.state = 'fallback';",
  'history.replaceState(null, \'\', url);',
]) {
  assert.ok(
    syncSource.includes(authorityStep),
    `browser route authority adapter is missing ${authorityStep}`,
  );
}
for (const sceneExtractionStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'sceneimpl',\n)",
  'rustAppShadow.extractActivePresentationScene(',
  'JSON.stringify(presentationBindings)',
  "rustNode.source === 'authored_absolute'",
  'semanticNodes.set(node, {',
  "RUST_SCENE_IMPLEMENTATION === 'rust'",
  "rustSceneExtractionDiagnostics.state = 'fallback';",
]) {
  assert.ok(
    browserSource.includes(sceneExtractionStep),
    `browser scene extraction rollback gate is missing ${sceneExtractionStep}`,
  );
}
for (const renderSettingsStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'renderstateimpl',\n)",
  'function browserRenderSettingsState()',
  "const style = mode === 'both' ? 'matcap_wire' : String(mode);",
  'app.synchronizeRenderSettings(settings)',
  'function synchronizeRustRenderSettingsPacket(app, settings, source)',
  'if (!receipt.matchesInput)',
  "RUST_RENDER_SETTINGS_IMPLEMENTATION === 'rust'",
  'applyRustRenderSettingsProjection(app.snapshot().renderSettings);',
  'scheduleRustRenderSettingsSynchronization();',
  'rustAppShadow.mountRenderControls(',
  'rustAppShadowSequence = Math.max(rustAppShadowSequence, sequenceNumber);',
  "rustRenderSettingsDiagnostics.viewAuthority = 'hyperscope-web';",
  "RUST_RENDER_SETTINGS_IMPLEMENTATION !== 'rust'",
  "'render_settings_view_projection'",
  "'render_settings_view_rejection'",
]) {
  assert.ok(
    browserSource.includes(renderSettingsStep),
    `browser render-settings boundary is missing ${renderSettingsStep}`,
  );
}
const renderSettingsBoundary = browserSource.slice(
  browserSource.indexOf('const rustRenderSettingsDiagnostics = {'),
  browserSource.indexOf('function ensureRustAppShadow('),
);
assert.equal(
  Array.from(renderSettingsBoundary.matchAll(/\.synchronizeRenderSettings\(/g)).length,
  1,
  'all browser render-setting synchronization must share one Rust sequence/effect adapter',
);
for (const retiredRenderSettingsSemantic of [
  '.setRenderSettings(',
  'function renderSettingsContentEqual(left, right)',
  'function focusPostprocessContentEqual(left, right)',
]) {
  assert.equal(
    renderSettingsBoundary.includes(retiredRenderSettingsSemantic),
    false,
    `browser render-settings boundary must not retain ${retiredRenderSettingsSemantic}`,
  );
}
const renderControlMountBoundary = browserSource.slice(
  browserSource.indexOf('rustAppShadow.mountRenderControls('),
  browserSource.indexOf(
    'rustRenderSettingsViewMounted = true;',
    browserSource.indexOf('rustAppShadow.mountRenderControls('),
  ),
);
assert.ok(
  !renderControlMountBoundary.includes('synchronizeRustRenderSettingsPacket(')
    && renderControlMountBoundary.includes('acceptRustRenderSettingsViewCommit,'),
  'the Leptos callback must adapt an already committed Rust projection, not dispatch browser intent',
);
const renderControlCommitBoundary = browserSource.slice(
  browserSource.indexOf('function acceptRustRenderSettingsViewCommit('),
  browserSource.indexOf('function rejectRustRenderSettingsViewCommit('),
);
assert.ok(
  renderControlCommitBoundary.includes('applyRustRenderSettingsProjection(rust);')
    && !renderControlCommitBoundary.includes('synchronizeRustRenderSettingsPacket('),
  'the shared render/focus callback must project committed Rust state without redispatch',
);
for (const directDispatchStep of [
  '.dispatch_semantic(SemanticAction::SetRenderSettings(settings))',
  'project_render_controls(&store.render_snapshot())',
  'emit_committed(',
  'emit_error(error_callback,',
  'arguments.push(&JsValue::from(committed.sequence));',
]) {
  assert.ok(
    renderControlsSource.includes(directDispatchStep),
    `Leptos render controls are missing direct Rust dispatch step: ${directDispatchStep}`,
  );
}
for (const cameraLensStep of [
  "numeric_control_domain(\"fov\")",
  'NavigationAction::SetPerspectiveLens(requested_lens)',
  'project_camera_lens_control(&store.navigation_snapshot())',
  'queued.requested_lens.vertical_fov_radians.to_degrees()',
]) {
  assert.ok(
    cameraControlsSource.includes(cameraLensStep),
    `Rust camera-lens control is missing ${cameraLensStep}`,
  );
}
for (const cameraLensBoundaryStep of [
  "RUST_NAVIGATION_IMPLEMENTATION !== 'rust'",
  'rustAppShadow.mountCameraLensControl(',
  'const navigation = rustAppShadow.tickNavigation(0);',
  'cameraFovDegrees.set(committedDegrees);',
  "rustAppShadowDiagnostics.cameraLensControlAuthority = 'hyperscope-web';",
  'useBrowserCameraLensControl();',
]) {
  assert.ok(
    browserSource.includes(cameraLensBoundaryStep),
    `browser camera-lens boundary is missing ${cameraLensBoundaryStep}`,
  );
}
const cameraLensMountBoundary = browserSource.slice(
  browserSource.indexOf('rustAppShadow.mountCameraLensControl('),
  browserSource.indexOf(
    'host.hidden = false;',
    browserSource.indexOf('rustAppShadow.mountCameraLensControl('),
  ),
);
assert.ok(
  !cameraLensMountBoundary.includes('.setPerspectiveLens(')
    && cameraLensMountBoundary.includes('rustAppShadow.tickNavigation(0)'),
  'the camera-lens island must queue in Rust and project only the integrated camera',
);
for (const assetAuthorityStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'assetimpl',\n)",
  "RUST_ASSET_IMPLEMENTATION !== 'js'",
  'EXPLICIT_RUST_APP_SHADOW_ENABLED',
  "import { BrowserAssetEffectHost } from './asset_effect_host.mjs",
  'const browserAssetEffectHost = new BrowserAssetEffectHost(RUST_ASSET_IMPLEMENTATION);',
  'rustAppShadow.requestAssetLoad(',
  "observeRustAppShadowSequence(receipt.sequence, 'Rust asset request')",
  'fetch: receipt.fetch,',
  'loadCancellations: receipt.loadCancellations,',
  'installCancellations: receipt.installCancellations,',
  'browserAssetEffectHost.begin({',
  'browserAssetEffectHost.beginInstall(token, receipt.install)',
  'browserAssetEffectHost.runProcess(assetToken, async () => {',
  'browserAssetEffectHost.runInstall(assetToken, async () => {',
  'rustAppShadow.finishAssetLoadedWithMetadata(',
  'rustAppShadow.finishAssetFailed(',
  'rustAppShadow.finishPrimarySceneInstalled(',
  'rustAppShadow.finishPrimarySceneInstallFailed(',
  'snapshot?.installedPrimaryScene?.asset?.assetId',
  "beginAppAssetShadow(file.name, 'drop', null, 'primary_scene')",
  'function standaloneDroppedModelUrl(filename)',
  "url.searchParams.set('glb', filename);",
  'location.assign(standaloneDroppedModelUrl(file.name));',
  "beginAppAssetShadow(currentGlb, 'startup', null, 'primary_scene')",
  'fetch(candidate, appAssetFetchOptions(assetShadow))',
  'if (!appAssetMayProcess(assetShadow)) return;',
  "if (RUST_ASSET_IMPLEMENTATION === 'rust') throw error;",
]) {
  assert.ok(
    browserSource.includes(assetAuthorityStep),
    `browser asset authority adapter is missing ${assetAuthorityStep}`,
  );
}
const assetRequestBoundary = browserSource.slice(
  browserSource.indexOf('function beginAppAssetShadow('),
  browserSource.indexOf(
    'function completeAppAssetShadow(',
    browserSource.indexOf('function beginAppAssetShadow('),
  ),
);
for (const retiredAssetRequestStep of [
  'rustAppShadow.requestPrimaryAsset.bind(',
  'rustAppShadow.requestAsset.bind(',
  '++rustAppShadowSequence',
  'commit.effects.filter(',
]) {
  assert.equal(
    assetRequestBoundary.includes(retiredAssetRequestStep),
    false,
    `ordinary asset requests must not retain ${retiredAssetRequestStep}`,
  );
}
const dropAdapter = browserSource.slice(
  browserSource.indexOf('// --- File drop with overlay ---'),
  browserSource.indexOf('// --- Environment maps (IBL) ---'),
);
assert.ok(
  dropAdapter.indexOf('await idbPut(IDB_GLB_STORE, file.name, buf);')
    < dropAdapter.indexOf('location.assign(standaloneDroppedModelUrl(file.name));'),
  'presentation drop must persist the file before entering its standalone route',
);
assert.ok(
  dropAdapter.indexOf('location.assign(standaloneDroppedModelUrl(file.name));')
    < dropAdapter.indexOf('const installed = await loadModel('),
  'presentation drop must leave the cue composition before attempting an in-place install',
);
assert.ok(
  browserSource.includes("if (EXPLICIT_RUST_APP_SHADOW_ENABLED) p.set('appshadow', '1');"),
  'implicit Rust asset authority must not pollute canonical URLs with appshadow=1',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['aim', '0']]).pairs,
  [],
  'canonical routes must omit the free-camera target-policy default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['aim', '1']]).pairs,
  [['aim', '1']],
  'canonical routes must retain an explicit finite camera target',
);
const selectedRoute = canonicalizeHyperscopeRoute([
  ['selentity', '70000000-0000-4000-8000-000000000001'],
  ['selasset', '60000000-0000-4000-8000-000000000001'],
]);
assert.deepEqual(selectedRoute.pairs, [
  ['selasset', '60000000-0000-4000-8000-000000000001'],
  ['selentity', '70000000-0000-4000-8000-000000000001'],
]);
assert.deepEqual(selectedRoute.diagnostics, []);
assert.deepEqual(
  canonicalizeHyperscopeRoute([
    ['selasset', '60000000-0000-4000-8000-000000000001'],
  ]).diagnostics.map(diagnostic => diagnostic.code),
  ['invalid_value'],
  'a route selection must carry asset and entity identity atomically',
);
for (const targetPolicyStep of [
  "set(\n      'aim',\n      manualCameraSemanticTargetEnabled ? '1' : '0',",
  "manualCameraSemanticTargetEnabled = initParams.aim === '1';",
  "manualCameraSemanticTargetEnabled = routeCamera?.semanticTargetEnabled\n    ?? params.aim === '1';",
]) {
  assert.ok(
    browserSource.includes(targetPolicyStep),
    `camera target-policy route is missing ${targetPolicyStep}`,
  );
}
for (const selectedRouteStep of [
  "set(\n      'selasset',",
  "set(\n      'selentity',",
  "restorePendingRouteSelection('primary-model');",
  "restorePendingRouteSelection('presentation-composition');",
  'retainRouteSelectionIdentity(selectedObject.identity);',
]) {
  assert.ok(
    browserSource.includes(selectedRouteStep),
    `selected-identity route is missing ${selectedRouteStep}`,
  );
}
const clearSelectionSource = browserSource.match(
  /function clearSelectedObject\(message = ''\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(clearSelectionSource, 'could not locate selection-clear adapter');
assert.equal(
  clearSelectionSource.includes('manualCameraSemanticTargetEnabled = false'),
  false,
  'selection detach must not silently rewrite independent camera target policy',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['assetimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust asset-authority default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['assetimpl', 'js']]).pairs,
  [['assetimpl', 'js']],
  'canonical routes must retain an explicit JavaScript rollback',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['presentimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust presentation-authority default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['presentimpl', 'js']]).pairs,
  [['presentimpl', 'js']],
  'canonical routes must retain an explicit presentation rollback',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['gfxpresentimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust graphics-presentation policy default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['gfxpresentimpl', 'shadow']]).pairs,
  [['gfxpresentimpl', 'shadow']],
  'canonical routes must retain an explicit graphics-presentation shadow',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['sceneimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust scene-authority default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['sceneimpl', 'js']]).pairs,
  [['sceneimpl', 'js']],
  'canonical routes must retain an explicit scene-extraction rollback',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['renderstateimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust render-settings default',
);
for (const implementation of ['js', 'shadow']) {
  assert.deepEqual(
    canonicalizeHyperscopeRoute([['renderstateimpl', implementation]]).pairs,
    [['renderstateimpl', implementation]],
    `canonical routes must retain explicit ${implementation} render settings`,
  );
}
assert.deepEqual(
  canonicalizeHyperscopeRoute([['animclipimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust animation-clip default',
);
for (const implementation of ['js', 'shadow']) {
  assert.deepEqual(
    canonicalizeHyperscopeRoute([['animclipimpl', implementation]]).pairs,
    [['animclipimpl', implementation]],
    `canonical routes must retain explicit ${implementation} animation-clip authority`,
  );
}
assert.deepEqual(
  canonicalizeHyperscopeRoute([['routeimpl', 'rust']]).pairs,
  [],
  'canonical routes must omit the Rust route-authority default',
);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['routeimpl', 'js']]).pairs,
  [['routeimpl', 'js']],
  'canonical routes must retain an explicit route-authority rollback',
);
const startupAdapter = browserSource.slice(
  browserSource.indexOf("phase('wasm', [], async () =>"),
  browserSource.indexOf("phase('workers', ['wasm'], async () =>"),
);
for (const startupStep of [
  'const controlSpecs = hyperscopeControlSpecs();',
  'installRustControlDefaults(controlSpecs);',
  'rustRouteShadowDiagnostics.specs = controlSpecs;',
  'initParams = readParams(initialRouteParams);',
  'const startupRoute = evaluateRustRoute(startupBrowserParams, false);',
  'startupRoute.diagnostics.length === 0',
  '&& startupRoute.renderSettings',
  '&& startupRoute.navigationSettings',
  'new URLSearchParams(startupRoute.resolvedPairs),',
  'new URLSearchParams(startupRoute.pairs),',
  'initRenderSettings = startupRoute.renderSettings;',
  'initRustRouteAdmitted = true;',
  'initRouteSelection = startupRoute.selection ?? null;',
  'initRouteAnimationClock = startupRoute.animationClock ?? null;',
  "rustRouteShadowDiagnostics.startupSource = 'browser-fallback';",
  "'missing-typed-route-settings'",
  'initRouteSelection,',
  'initRouteAnimationClock,',
  "synchronizeRustRenderSettingsPacket(\n        app,\n        initRenderSettings,\n        'route-startup',",
  "rustRenderSettingsDiagnostics.state = 'route-committed';",
]) {
  assert.ok(
    startupAdapter.includes(startupStep),
    `browser startup route adapter is missing ${startupStep}`,
  );
}
assert.ok(
  startupAdapter.indexOf('installRustControlDefaults(controlSpecs);')
    < startupAdapter.indexOf('const startupRoute = evaluateRustRoute(startupBrowserParams, false);'),
  'Rust defaults must be installed before either startup authority decodes state',
);
assert.ok(
  startupAdapter.indexOf('new URLSearchParams(startupRoute.resolvedPairs),')
    < startupAdapter.indexOf('initRouteSelection,'),
  'Rust startup decoding must finish before browser state is applied',
);
assert.ok(
  browserSource.includes('function readParams(p, explicitParams = p) {')
    && browserSource.includes(
      'for (const [key, defaultValue] of Object.entries(PARAM_DEFAULTS))',
    )
    && browserSource.includes("decoded.animtimeProvided = explicitParams.has('animtime')")
    && browserSource.includes("decoded.animspeedProvided = explicitParams.has('animspeed')"),
  'resolved Rust defaults must not masquerade as explicitly linked animation-clock values',
);
const readParamsDeclaration = browserSource.match(
  /function readParams\(p, explicitParams = p\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(readParamsDeclaration, 'could not isolate route decoder');
const decodeRoute = Function(
  `${policyDefaultsDeclaration}\n${routeDefaultsAdapterSource}\n`
    + `installRustControlDefaults(${JSON.stringify(specs)});\n`
    + `${readParamsDeclaration}\nreturn readParams;`,
)();
const resolvedDefaults = new URLSearchParams(
  canonicalizeHyperscopeRoute([]).resolvedPairs,
);
const implicitClock = decodeRoute(resolvedDefaults, new URLSearchParams());
assert.equal(implicitClock.animtimeProvided, false);
assert.equal(implicitClock.animspeedProvided, false);
assert.equal(Object.keys(implicitClock).length, specs.length + 2);
for (const spec of specs) {
  assert.equal(
    implicitClock[spec.key],
    spec.defaultValue,
    `the registry-driven decoder omitted or changed ${spec.key}`,
  );
}
const explicitClock = decodeRoute(
  resolvedDefaults,
  new URLSearchParams([['animtime', '0'], ['animspeed', '1']]),
);
assert.equal(explicitClock.animtimeProvided, true);
assert.equal(explicitClock.animspeedProvided, true);
assert.equal(
  decodeRoute(new URLSearchParams('mode=wire&mode=pbr')).mode,
  'wire',
  'the browser decoder must retain Rust first-value duplicate semantics',
);

const applyParamsSource = browserSource.match(
  /applyParams = function\([\s\S]*?validatedRouteAnimationClock = null,[\s\S]*?\) \{([\s\S]*?)\n\};\n\n\/\/ --- IndexedDB/,
)?.[1];
assert.ok(applyParamsSource, 'could not locate browser startup state adapter');
for (const exactProjection of [
  'renderMode.set(routeRenderMode ?? params.mode ?? \'pbr\');',
  'lod.res.set(validatedRenderSettings.resolutionLevel);',
  'lod.density.set(validatedRenderSettings.density);',
  'lod.screenAtten.set(validatedRenderSettings.screenAttenuation);',
  'lod.minPx.set(validatedRenderSettings.minPixelsPerSubdivision);',
  'lod.atlas.set(validatedRenderSettings.atlasExponent);',
  'lod.grading.set(String(validatedRenderSettings.maxFaceEdgeRatio));',
]) {
  assert.ok(
    applyParamsSource.includes(exactProjection),
    `Rust-admitted startup render state is missing exact projection: ${exactProjection}`,
  );
}
for (const exactAdmissionStep of [
  'rustRouteAdmitted ? Number(value) : finite(value, fallback, minimum, maximum);',
  'rustRouteAdmitted ? Number(value) : integer(value, fallback, minimum, maximum);',
  'mob.mx.set(routeTransform?.centerControls[0]',
  '?? admittedNumber(params.mx, 5, -30, 30));',
  'cameraFovDegrees.set(routeCamera?.verticalFovDegrees',
  '?? admittedInteger(params.fov, 75, 35, 110));',
  '$(id).max = atlasExp;',
  '$(id).value = admittedInteger(value, fallback, 0, atlasExp);',
  'fz.radius.set(admittedFocus?.blurRadiusPixels',
  '?? admittedInteger(params.fradius, 11, 4, 128));',
  'pendingRouteSelection = rustRouteAdmitted',
  '? validatedRouteSelection',
  'pendingRouteAnimationClock = rustRouteAdmitted',
  '? validatedRouteAnimationClock',
]) {
  assert.ok(
    applyParamsSource.includes(exactAdmissionStep),
    `Rust-admitted startup values are missing exact projection: ${exactAdmissionStep}`,
  );
}
assert.ok(
  browserSource.includes('const animIdx = initRustRouteAdmitted')
    && browserSource.includes('? Number(initParams.anim)')
    && browserSource.includes(': parseInt(initParams.anim);'),
  'Rust-admitted animation indices must bypass legacy parseInt coercion',
);

const canonical = canonicalizeHyperscopeRoute([
  ['zoom', '3.00'],
  ['rx', '0.125'],
  ['mode', 'lod'],
  ['glb', 'horse.glb'],
  ['minpx', '16.0'],
  ['lodratio', '4'],
  ['routeimpl', 'shadow'],
]);
assert.deepEqual(canonical.pairs, [
  ['mode', 'lod'],
  ['lodratio', '4'],
  ['rx', '0.125'],
  ['routeimpl', 'shadow'],
]);
assert.deepEqual(canonical.diagnostics, []);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['routeshadow', '1']]).diagnostics
    .map(diagnostic => diagnostic.code),
  ['unknown_key'],
  'the legacy browser-only alias must not survive in the canonical Rust schema',
);

const malformed = canonicalizeHyperscopeRoute([
  ['mode', 'wire'],
  ['mode', 'pbr'],
  ['atten', 'yes'],
  ['rx', 'NaN'],
  ['lodratio', '3'],
  ['selectionimpl', 'sometimes'],
  ['routeimpl', 'sometimes'],
  ['mystery', '1'],
]);
assert.deepEqual(
  malformed.diagnostics.map(diagnostic => diagnostic.code),
  [
    'duplicate_key', 'invalid_value', 'invalid_value', 'invalid_value',
    'invalid_value', 'invalid_value', 'unknown_key',
  ],
);
assert.deepEqual(malformed.pairs, [
  ['mode', 'wire'],
  ['atten', 'yes'],
  ['lodratio', '3'],
  ['selectionimpl', 'sometimes'],
  ['rx', 'NaN'],
  ['routeimpl', 'sometimes'],
]);

const cue = 'e0000000-0000-4000-8000-000000000004';
const linkedCue = canonicalizeHyperscopeRoute([
  ['presentation', '1'],
  ['cue', cue],
]);
assert.deepEqual(linkedCue.pairs, [
  ['presentation', '1'],
  ['cue', cue],
]);
assert.deepEqual(linkedCue.diagnostics, []);
assert.deepEqual(
  canonicalizeHyperscopeRoute([['cue', 'not-a-uuid']]).diagnostics
    .map(diagnostic => diagnostic.code),
  ['invalid_value'],
);

console.log(JSON.stringify({
  specs: specs.length,
  canonicalPairs: canonical.pairs,
  diagnosticCodes: malformed.diagnostics.map(diagnostic => diagnostic.code),
}));
