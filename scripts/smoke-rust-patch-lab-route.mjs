import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, canonicalizeHyperscopeRoute } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const defaults = canonicalizeHyperscopeRoute([]);
assert.deepEqual(defaults.patchLabSession, {
  active: false,
  controls: {
    shape: 'triangle',
    field: 'manual_edges',
    manualEdgeExponents: [3, 4, 4],
    minExponent: 1,
    maxExponent: 6,
    phaseMicroradians: 0,
    phaseRadians: 0,
    bendPercent: 55,
    grid: 8,
    animate: false,
  },
});
assert.deepEqual(defaults.startupSettings.patchLabSession, defaults.patchLabSession);

const normalized = canonicalizeHyperscopeRoute([
  ['atlas', '5'],
  ['lab', 'plane'],
  ['labfield', 'edges'],
  ['laba', '5'],
  ['labb', '4'],
  ['labc', '3'],
  ['labmin', '4'],
  ['labmax', '2'],
  ['labphase', '6.283185'],
  ['labgrid', '16'],
]);
assert.deepEqual(normalized.diagnostics, []);
assert.deepEqual(normalized.patchLabSession, {
  active: true,
  controls: {
    shape: 'plane',
    field: 'wave',
    manualEdgeExponents: [4, 4, 4],
    minExponent: 4,
    maxExponent: 4,
    phaseMicroradians: 0,
    phaseRadians: 0,
    bendPercent: 55,
    grid: 16,
    animate: false,
  },
});
assert.deepEqual(normalized.startupSettings.patchLabSession, normalized.patchLabSession);

const browser = readFileSync(`${repository}/hyperscope.html`, 'utf8');
for (const required of [
  '&& startupRoute.startupSettings',
  'initRustStartupSettings = startupRoute.startupSettings;',
  'const validatedPatchLabSession = startupSettings?.patchLabSession ?? null;',
  '? validatedPatchLabSession?.controls : null;',
  '? patchLabWireField(routePatchLabControls.field)',
  'applyParams(initParams, initRustStartupSettings)',
]) {
  assert.ok(browser.includes(required), `browser typed Patch Lab route is missing ${required}`);
}

console.log('Rust typed Patch Lab route smoke passed');
