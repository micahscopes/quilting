import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const {
  default: init,
  mr_renderShadowDiagnostics,
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

console.log(JSON.stringify({ generatedExports: true, inertBeforeRenderer: true }));
