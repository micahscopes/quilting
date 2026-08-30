import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const patchLab = read('crates/hyperscope-app/src/patch_lab.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');
const settings = read('crates/hyperscope-app/src/settings.rs');

for (const required of [
  'spec!("renderstateimpl", "js", Implementation)',
  'pub enum RenderSettingsSynchronizationDisposition',
  'pub fn synchronize_render_settings(',
  'if snapshot.settings == settings',
  'value: SemanticAction::SetRenderSettings(settings)',
  'RenderSettingsSynchronizationDisposition::Unchanged',
  'RenderSettingsSynchronizationDisposition::Committed',
]) {
  assert.ok(
    settings.includes(required) || app.includes(required),
    `Rust render-settings authority is missing ${required}`,
  );
}

for (const required of [
  'render_settings_changed(settings, &mut effects)',
  'let changed = store.synchronize_render_settings(render).unwrap();',
  'changed.commit.unwrap().effects',
  'PatchLabEffect::EvaluateLod',
]) {
  assert.ok(
    app.includes(required) || patchLab.includes(required),
    `render-settings Patch Lab effect fence is missing ${required}`,
  );
}

for (const required of [
  '#[wasm_bindgen(js_name = setRenderSettings)]',
  '#[wasm_bindgen(js_name = synchronizeRenderSettings)]',
  'let input: ShadowRenderSettingsInput =',
  '.synchronize_render_settings(settings)',
  'let matches_input = synchronization.snapshot.settings == settings;',
  'commit: synchronization.commit.as_ref().map(shadow_commit)',
  '#[serde(rename_all = "camelCase", deny_unknown_fields)]',
]) {
  assert.ok(adapter.includes(required), `WASM render-settings port is missing ${required}`);
}

const packetStart = browser.indexOf(
  'function synchronizeRustRenderSettingsPacket(app, settings, source) {',
);
const packetEnd = browser.indexOf('\n}', packetStart);
assert.ok(packetStart >= 0 && packetEnd > packetStart,
  'could not locate browser render-settings packet adapter');
const packet = browser.slice(packetStart, packetEnd);
for (const required of [
  'app.synchronizeRenderSettings(settings)',
  'if (receipt.sequence != null)',
  'observeRustPatchLabEffects(receipt.commit.effects || []',
  "receipt.disposition === 'committed'",
]) {
  assert.ok(packet.includes(required), `thin render-settings packet adapter is missing ${required}`);
}

const synchronizationStart = browser.indexOf('function synchronizeRustRenderSettings() {');
const synchronizationEnd = browser.indexOf('  } catch (error) {', synchronizationStart);
assert.ok(synchronizationStart >= 0 && synchronizationEnd > synchronizationStart,
  'could not locate browser render-settings synchronization');
const synchronization = browser.slice(synchronizationStart, synchronizationEnd);
for (const required of [
  'synchronizeRustRenderSettingsPacket(',
  'if (!receipt.matchesInput)',
  'applyRustRenderSettingsProjection(rust)',
]) {
  assert.ok(synchronization.includes(required), `thin render adapter is missing ${required}`);
}
for (const retired of [
  'renderSettingsContentEqual',
  'app.snapshot().renderSettings',
  '++rustAppShadowSequence',
  'app.setRenderSettings(',
]) {
  assert.equal(synchronization.includes(retired), false,
    `browser synchronization must not retain ${retired}`);
}
for (const retired of [
  'function renderSettingsContentEqual(left, right)',
  'function focusPostprocessContentEqual(left, right)',
  'app.setRenderSettings(',
]) {
  assert.equal(browser.includes(retired), false, `browser glue must not retain ${retired}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Render-settings Rust boundary source smoke passed');
