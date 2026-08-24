import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

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
assert.equal(specs.find(spec => spec.key === 'cue').kind, 'optional_uuid');
assert.equal(specs.find(spec => spec.key === 'cue').defaultValue, '');

const browserSource = readFileSync(`${repository}/hyperscope.html`, 'utf8');
const syncSource = browserSource.match(
  /function syncURL\(\) \{([\s\S]*?)\/\/ Apply URL params to controls on load/,
)?.[1];
assert.ok(syncSource, 'could not locate browser URL serializer');
const browserKeyOrder = Array.from(
  syncSource.matchAll(/(?:set|ss)\(\s*'([^']+)'/g),
  match => match[1],
);
assert.deepEqual(
  specs.map(spec => spec.key),
  browserKeyOrder,
  'Rust route order/default registry drifted from the browser oracle',
);

const canonical = canonicalizeHyperscopeRoute([
  ['routeshadow', '1'],
  ['zoom', '3.00'],
  ['rx', '0.125'],
  ['mode', 'lod'],
  ['glb', 'horse.glb'],
  ['minpx', '16.0'],
  ['lodratio', '4'],
]);
assert.deepEqual(canonical.pairs, [
  ['mode', 'lod'],
  ['lodratio', '4'],
  ['rx', '0.125'],
  ['routeshadow', '1'],
]);
assert.deepEqual(canonical.diagnostics, []);

const malformed = canonicalizeHyperscopeRoute([
  ['mode', 'wire'],
  ['mode', 'pbr'],
  ['atten', 'yes'],
  ['rx', 'NaN'],
  ['lodratio', '3'],
  ['mystery', '1'],
]);
assert.deepEqual(
  malformed.diagnostics.map(diagnostic => diagnostic.code),
  ['duplicate_key', 'invalid_value', 'invalid_value', 'invalid_value', 'unknown_key'],
);
assert.deepEqual(malformed.pairs, [
  ['mode', 'wire'],
  ['atten', 'yes'],
  ['lodratio', '3'],
  ['rx', 'NaN'],
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
