#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const policy = read('crates/hyperscope-app/src/backend_presentation.rs');
const settings = read('crates/hyperscope-app/src/settings.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub struct WebGpuPresentationEvidence',
  'pub struct GraphicsPresentationDecision',
  'pub fn resolve_graphics_presentation(',
  'device_lod_recovery_eligible',
  'focus_postprocess_requested',
  'focus_presentation_ready',
  'device_lod_authority_eligible: present_webgpu',
  'evidence.presented_style == evidence.requested_style',
  'historical_surface_loss_prevents_recovery_and_presentation',
  'recovery_can_preheat_before_the_first_admitted_frame',
]) {
  assert.ok(policy.includes(required), `Rust presentation policy is missing ${required}`);
}
for (const forbidden of ['web_sys', 'HtmlCanvas', 'GpuDevice', 'WebGl']) {
  assert.equal(policy.includes(forbidden), false,
    `backend-neutral presentation policy contains platform type ${forbidden}`);
}
assert.ok(settings.includes('spec!("gfxpresentimpl", "rust", Implementation)'),
  'Rust controls do not expose the graphics-presentation rollback');
for (const required of [
  '#[wasm_bindgen(js_name = resolveGraphicsBackendPresentation)]',
  'hyperscope_app::resolve_graphics_presentation(',
  'WebGpuPresentationEvidence {',
]) {
  assert.ok(adapter.includes(required), `WASM presentation adapter is missing ${required}`);
}
for (const required of [
  "gfxpresentimpl: 'rust'",
  "initialBrowserParams, 'gfxpresentimpl'",
  'globalThis.__hyperscopeGraphicsPresentationPolicy',
  'function browserGraphicsPresentationDecision(residency)',
  'function resolveRustGraphicsPresentationDecision(residency)',
  'function chooseGraphicsPresentationDecision(browser, rust)',
  'graphicsBackendDiagnostics.focusPostprocessRequested',
  "graphicsPresentationPolicyDiagnostics.authority = 'hyperscope-app'",
  'const decision = chooseGraphicsPresentationDecision(browserDecision, rustDecision)',
  'const presenting = decision.presentWebgpu',
]) {
  assert.ok(browser.includes(required), `browser cutover is missing ${required}`);
}

console.log('Rust graphics-presentation policy source smoke passed');
