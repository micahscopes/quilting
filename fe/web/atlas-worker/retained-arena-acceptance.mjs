// Ownership check against the emitted Fe pool and the shared render runtime.
// This does not start workers or claim worker-to-renderer acceptance.
// node retained-arena-acceptance.mjs <parent.wasm> <fe-render-runtime.js>
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [wasmPath, runtimePath] = process.argv.slice(2);
if (!wasmPath || !runtimePath) throw Error('expected parent.wasm and render runtime paths');
globalThis.HTMLElement = class {};
globalThis.customElements = { define() {} };
const { FeSurfaceElement } = await import(pathToFileURL(resolve(runtimePath)).href);
const module = await WebAssembly.compile(await readFile(wasmPath));
const imports = Object.create(null);
for (const entry of WebAssembly.Module.imports(module)) {
  assert.equal(entry.kind, 'function');
  (imports[entry.module] ??= Object.create(null))[entry.name] = () => {
    throw Error(`unexpected effect during ownership check: ${entry.module}.${entry.name}`);
  };
}
const { exports: wasm } = await WebAssembly.instantiate(module, imports);
const surface = Object.create(FeSurfaceElement.prototype);
surface._wasmArenaCheckpoint = wasm.fe_cabi_checkpoint;
surface._wasmArenaRewind = wasm.fe_cabi_rewind;
surface._wasmArenaReset = () => { throw Error('must not reset retained memory'); };
const before = wasm.fe_cabi_checkpoint();
const initial = surface._runWasmInitialization(() => wasm.fe_actor_initialize_v1());
const retainedEnd = wasm.fe_cabi_checkpoint();
assert(retainedEnd > before + 64 * 1024 * 1024, 'real pool reserves its retained arena');
const projection = wasm.fe_actor_project_v1();
assert.equal(projection[3], 1200);
const header = new Uint8Array(wasm.memory.buffer, initial[0], 256).slice();
// Simulate an outstanding task allocation with the real canonical allocator.
const task = wasm.cabi_realloc(0, 0, 4, 256);
new Uint8Array(wasm.memory.buffer, task, 256).fill(77);
const taskEnd = wasm.fe_cabi_checkpoint();
for (let i = 0; i < 100; i++) {
  surface._runWasmArenaEpoch(() => {
    const pointer = wasm.cabi_realloc(0, 0, 4, 4096);
    assert(pointer >= taskEnd);
    new Uint8Array(wasm.memory.buffer, pointer, 4096).fill(i);
    assert.deepEqual(wasm.fe_actor_project_v1(), projection);
  });
  assert.equal(wasm.fe_cabi_checkpoint(), taskEnd);
}
assert.throws(() => surface._runWasmArenaEpoch(() => {
  wasm.cabi_realloc(0, 0, 4, 8192);
  throw Error('test trap');
}), /test trap/);
assert.equal(wasm.fe_cabi_checkpoint(), taskEnd);
assert.deepEqual(new Uint8Array(wasm.memory.buffer, initial[0], 256), header);
assert(new Uint8Array(wasm.memory.buffer, task, 256).every(x => x === 77));
console.log(JSON.stringify({ before, retainedEnd, taskEnd, scratchCalls: 100,
  canonicalTiles: projection[3], linearMemoryBytes: wasm.memory.buffer.byteLength,
  retainedHeaderPreserved: true, outstandingAllocationPreserved: true }));
