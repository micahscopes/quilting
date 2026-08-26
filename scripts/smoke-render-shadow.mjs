import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const {
  default: init,
  mr_adaptivePickedDiagnostics,
  mr_clearAdaptivePickedFace,
  mr_debugFocusState,
  mr_refreshAdaptivePicked,
  mr_renderShadowDiagnostics,
  mr_setAdaptiveCurrentView,
  mr_setAdaptivePickedFace,
  mr_setLodGradingRatio,
  mr_setRenderShadowEnabled,
} = await import(packageUrl);

await init({ module_or_path: readFileSync(wasmPath) });
assert.equal(
  mr_renderShadowDiagnostics(),
  null,
  'diagnostics should be inert before a renderer exists',
);
assert.equal(
  mr_setRenderShadowEnabled(true),
  null,
  'enabling should be inert before a renderer exists',
);
assert.equal(mr_setLodGradingRatio(4), false, 'grading should be inert before a renderer exists');
assert.equal(mr_setLodGradingRatio(3), false, 'unsupported grading ratios must fail closed');
assert.equal(mr_debugFocusState(), null, 'focus diagnostics should be inert before a renderer exists');
assert.equal(
  mr_adaptivePickedDiagnostics(),
  null,
  'adaptive diagnostics should be inert before a renderer exists',
);
assert.equal(
  mr_refreshAdaptivePicked(),
  null,
  'adaptive refresh should be inert before a renderer exists',
);
assert.equal(
  mr_clearAdaptivePickedFace(),
  null,
  'adaptive clear should be inert before a renderer exists',
);
const rejectedAdaptiveConfiguration =
  mr_setAdaptivePickedFace(0, 16, 32, 5, 256, 256, 2_000_000);
assert.equal(
  rejectedAdaptiveConfiguration.get('ok'),
  false,
  'adaptive configuration should fail closed before renderer initialization',
);
assert.equal(rejectedAdaptiveConfiguration.get('error'), 'renderer is not initialized');
const rejectedCurrentViewConfiguration =
  mr_setAdaptiveCurrentView(16, 32, 5, 8, 64, 512, 512, 2_000_000);
assert.equal(
  rejectedCurrentViewConfiguration.get('ok'),
  false,
  'current-view adaptive configuration should fail closed before renderer initialization',
);
assert.equal(rejectedCurrentViewConfiguration.get('error'), 'renderer is not initialized');

console.log(JSON.stringify({ generatedExports: true, inertBeforeRenderer: true }));
