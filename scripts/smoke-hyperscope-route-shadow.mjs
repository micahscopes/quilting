import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runInNewContext } from 'node:vm';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const {
  default: init,
  canonicalizeHyperscopeRoute,
  hyperscopeControlSpecs,
} = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const specs = hyperscopeControlSpecs();
assert.equal(new Set(specs.map(spec => spec.key)).size, specs.length);
assert.equal(specs.find(spec => spec.key === 'minpx').defaultValue, '16');
assert.equal(specs.find(spec => spec.key === 'lodratio').defaultValue, '2');
assert.equal(specs.find(spec => spec.key === 'lodratio').kind, 'lod_ratio');
assert.equal(specs.find(spec => spec.key === 'appshadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'rendershadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'adaptiveshadow').kind, 'toggle');
assert.equal(specs.find(spec => spec.key === 'walkimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'navimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'navimpl').defaultValue, 'js');
assert.equal(specs.find(spec => spec.key === 'animclockimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'animclockimpl').defaultValue, 'js');
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
assert.equal(specs.find(spec => spec.key === 'presentimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'presentimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'assetimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'assetimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'sceneimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'sceneimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'routeimpl').kind, 'implementation');
assert.equal(specs.find(spec => spec.key === 'routeimpl').defaultValue, 'rust');
assert.equal(specs.find(spec => spec.key === 'cue').kind, 'optional_uuid');
assert.equal(specs.find(spec => spec.key === 'cue').defaultValue, '');

const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
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
const implementationFromRouteSource = browserSource.match(
  /function implementationFromRoute\(params, key, defaultImplementation\) \{[\s\S]*?\n\}/,
)?.[0];
assert.ok(implementationFromRouteSource, 'could not locate implementation-mode parser');
const implementationFromRoute = runInNewContext(
  `${implementationFromRouteSource}; implementationFromRoute`,
);
for (const implementation of ['js', 'shadow', 'rust']) {
  assert.equal(
    implementationFromRoute(
      new URLSearchParams(`mode=${implementation}`),
      'mode',
      'rust',
    ),
    implementation,
  );
}
assert.equal(
  implementationFromRoute(new URLSearchParams(), 'mode', 'js'),
  'js',
);
assert.equal(
  implementationFromRoute(new URLSearchParams('mode=invalid'), 'mode', 'rust'),
  'rust',
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
  "implementationFromRoute(\n  initialRouteParams, 'routeimpl', 'rust',\n)",
  "routeimpl: 'rust'",
]) {
  assert.ok(
    browserSource.includes(routeDefaultStep),
    `browser route default is missing ${routeDefaultStep}`,
  );
}
for (const selectionDefaultStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'selectionimpl', 'rust',\n)",
  "selectionimpl: 'rust'",
]) {
  assert.ok(
    browserSource.includes(selectionDefaultStep),
    `browser selection default is missing ${selectionDefaultStep}`,
  );
}
for (const animationClockDefaultStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'animclockimpl', 'js',\n)",
  "animclockimpl: 'js'",
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
  'rustAppShadow.writeAnimationSample(',
  "ANIMATION_CLOCK_IMPLEMENTATION === 'rust'",
  "restorePendingRouteAnimationClock('clip-switch')",
  "restorePendingRouteAnimationClock('presentation-animation')",
  '{ restoreRouteClock: false },',
  'await selectAnimationIndex(animIdx);',
  "restorePendingRouteAnimationClock('startup-animation')",
  'animtimeProvided:',
  'animspeedProvided:',
]) {
  assert.ok(
    browserSource.includes(animationClockAuthorityStep),
    `browser animation-clock adapter is missing ${animationClockAuthorityStep}`,
  );
}
for (const navigationAuthorityStep of [
  "implementationFromRoute(\n  initialNavigationParams, 'navimpl', 'js',\n)",
  "navimpl: 'js'",
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
const browserDefaultsSource = browserSource.match(
  /const PARAM_DEFAULTS = (\{[\s\S]*?\n\});/,
)?.[1];
assert.ok(browserDefaultsSource, 'could not locate browser URL defaults');
const browserDefaults = JSON.parse(JSON.stringify(
  runInNewContext(`(${browserDefaultsSource})`),
));
const rustDefaults = Object.fromEntries(
  specs.map(spec => [spec.key, spec.defaultValue]),
);
for (const [key, value] of Object.entries(browserDefaults)) {
  assert.equal(
    rustDefaults[key],
    value,
    `Rust route default for ${key} drifted from the browser rollback`,
  );
}
const implicitBrowserDefaults = {
  presentation: '0',
  roundshadow: '0',
  appshadow: '0',
  rendershadow: '0',
  adaptiveshadow: '0',
};
assert.deepEqual(
  Object.fromEntries(
    Object.keys(implicitBrowserDefaults).map(key => [key, rustDefaults[key]]),
  ),
  implicitBrowserDefaults,
  'Rust implicit flag defaults drifted from the browser rollback',
);
assert.equal(
  Object.keys(browserDefaults).length + Object.keys(implicitBrowserDefaults).length,
  specs.length,
  'the browser/Rust default parity oracle does not cover every route control',
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
  "implementationFromRoute(\n  initialBrowserParams, 'sceneimpl', 'rust',\n)",
  "sceneimpl: 'rust'",
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
for (const assetAuthorityStep of [
  "implementationFromRoute(\n  initialBrowserParams, 'assetimpl', 'rust',\n)",
  "assetimpl: 'rust'",
  "RUST_ASSET_IMPLEMENTATION !== 'js'",
  'EXPLICIT_RUST_APP_SHADOW_ENABLED',
  "import { BrowserAssetEffectHost } from './asset_effect_host.mjs",
  'const browserAssetEffectHost = new BrowserAssetEffectHost(RUST_ASSET_IMPLEMENTATION);',
  'rustAppShadow.requestPrimaryAsset.bind(rustAppShadow)',
  'browserAssetEffectHost.begin({',
  'browserAssetEffectHost.runProcess(assetToken, async () => {',
  'rustAppShadow.completeAssetLoadedWithMetadata(',
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
  "manualCameraSemanticTargetEnabled = params.aim === '1';",
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
  'const startupRoute = evaluateRustRoute(startupBrowserParams, false);',
  'startupRoute && startupRoute.diagnostics.length === 0',
  'initParams = readParams(new URLSearchParams(startupRoute.pairs));',
  "rustRouteShadowDiagnostics.startupSource = 'browser-fallback';",
  'applyParams(initParams);',
]) {
  assert.ok(
    startupAdapter.includes(startupStep),
    `browser startup route adapter is missing ${startupStep}`,
  );
}
assert.ok(
  startupAdapter.indexOf('initParams = readParams(new URLSearchParams(startupRoute.pairs));')
    < startupAdapter.indexOf('applyParams(initParams);'),
  'Rust startup decoding must finish before browser state is applied',
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
