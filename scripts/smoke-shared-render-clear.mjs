#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const core = read('crates/quilting-core/src/render.rs');
const webgl = read('crates/quilting-renderer/src/lib.rs');
const wasm = read('crates/quilting-wasm/src/main_renderer.rs');
const webgpu = read('crates/quilting-webgpu/src/lib.rs');
const presentation = read('crates/quilting-webgpu/src/presentation.rs');
const focus = read('crates/quilting-webgpu/src/focus_postprocess.rs');
const adaptive = read('crates/quilting-webgpu/src/adaptive_overlay.rs');
const resident = read('crates/quilting-webgpu/src/resident_roots.rs');

assert.ok(
  core.includes('DEFAULT_RENDER_CLEAR_COLOR: [f32; 4] = [0.2, 0.2, 77.0 / 255.0, 1.0]'),
  'quilting-core does not own the canonical render clear',
);
for (const [name, source, required] of [
  ['WebGL2 renderer', webgl, 'DEFAULT_RENDER_CLEAR_COLOR'],
  ['WASM renderer', wasm, 'DEFAULT_RENDER_CLEAR_COLOR'],
  ['WebGPU renderer', webgpu, 'DEFAULT_FRAME_CLEAR'],
  ['WebGPU focus renderer', focus, 'crate::DEFAULT_FRAME_CLEAR'],
  ['WebGPU adaptive renderer', adaptive, 'DEFAULT_FRAME_CLEAR'],
  ['WebGPU resident renderer', resident, 'DEFAULT_FRAME_CLEAR'],
]) {
  assert.ok(source.includes(required), `${name} does not consume the shared render clear`);
}

const renderSources = [webgl, wasm, webgpu, focus, adaptive, resident].join('\n');
for (const duplicate of [
  /clear_color\(\s*0\.2\s*,\s*0\.2\s*,\s*0\.3/,
  /r:\s*0\.2\s*,\s*g:\s*0\.2\s*,\s*b:\s*0\.3/,
]) {
  assert.equal(duplicate.test(renderSources), false, 'a backend duplicated the clear policy');
}

for (const required of [
  'preferred_alpha_mode(&capabilities.alpha_modes)',
  '.contains(&wgpu::CompositeAlphaMode::Opaque)',
]) {
  assert.ok(presentation.includes(required), `WebGPU presentation is missing ${required}`);
}

console.log('Shared render-clear source smoke passed');
