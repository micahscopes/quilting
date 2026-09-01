import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const model = read('crates/hyperscope-web/src/interaction_status.rs');
const view = read('crates/hyperscope-web/src/interaction_status/csr.rs');
const manifest = read('crates/hyperscope-web/Cargo.toml');
const webLibrary = read('crates/hyperscope-web/src/lib.rs');
const appShadow = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

for (const required of [
  'pub struct InteractionStatusViewModel',
  'pub struct InteractionSelectionViewModel',
  'pub fn project_interaction_status(frame: &AppFrameSnapshot)',
  'frame.selected_focus.map(',
  'hovered.identity == selected.identity',
  'hovered.surface.map(|surface| surface.face)',
]) {
  assert.ok(model.includes(required), `interaction projection is missing ${required}`);
}

for (const required of [
  'pub fn mount_interaction_status(parent: web_sys::HtmlElement, store: AppStore)',
  'store.navigation_signal().for_each',
  'project_interaction_status(&navigation)',
  'role="status"',
  'aria-live="polite"',
]) {
  assert.ok(view.includes(required), `interaction CSR view is missing ${required}`);
}

assert.ok(manifest.includes('interaction-status = ["dep:hyperscope-app"]'));
assert.ok(manifest.includes('"interaction-status",'));
assert.ok(webLibrary.includes('pub mod interaction_status;'));

for (const required of [
  '#[wasm_bindgen(js_name = mountInteractionStatus)]',
  'hyperscope_web::interaction_status::mount_interaction_status(',
]) {
  assert.ok(appShadow.includes(required), `WASM interaction mount is missing ${required}`);
}

for (const required of [
  'id="interaction-status-rust" hidden',
  'function ensureRustInteractionStatusView()',
  "rustAppShadow.mountInteractionStatus(host);",
  "rustAppShadowDiagnostics.interactionStatusAuthority = 'hyperscope-web';",
  'ensureRustInteractionStatusView();',
  'rustInteractionStatusMounted',
  "status.textContent = '';",
  'status.hidden = true;',
]) {
  assert.ok(browser.includes(required), `browser interaction view adapter is missing ${required}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Rust interaction status source smoke passed');
