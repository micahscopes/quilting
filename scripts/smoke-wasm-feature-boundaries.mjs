import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const root = read('crates/quilting-wasm/src/lib.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');

assert.ok(
  root.includes('#[cfg(feature = "webgpu-backend")]\nmod webgpu_backend;'),
  'the WebGPU adapter module must remain feature-gated',
);

for (const call of [
  'crate::webgpu_backend::replace_image_bitmaps(&images)',
  'crate::webgpu_backend::replace_environment_maps(',
]) {
  const index = renderer.indexOf(call);
  assert.ok(index >= 0, `renderer is missing ${call}`);
  const prefix = renderer.slice(Math.max(0, index - 96), index);
  assert.ok(
    prefix.includes('#[cfg(feature = "webgpu-backend")]'),
    `${call} must not resolve in the WebGL2-only build`,
  );
}

console.log('WASM backend feature-boundary source smoke passed');
