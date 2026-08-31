import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const adapter = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');
const settings = read('crates/hyperscope-app/src/settings.rs');

for (const required of [
  'spec!("navstateimpl", "rust", Implementation)',
  'pub enum NavigationSettingsSynchronizationDisposition',
  'pub fn synchronize_navigation_settings(',
  'if snapshot.settings == settings',
  'value: SemanticAction::SetNavigationSettings(settings)',
  'NavigationSettingsSynchronizationDisposition::Unchanged',
  'NavigationSettingsSynchronizationDisposition::Committed',
]) {
  assert.ok(
    settings.includes(required) || app.includes(required),
    `Rust navigation-settings authority is missing ${required}`,
  );
}

for (const required of [
  '#[wasm_bindgen(js_name = setNavigationSettings)]',
  '#[wasm_bindgen(js_name = synchronizeNavigationSettings)]',
  '.synchronize_navigation_settings(settings)',
  'let matches_input = synchronization.snapshot.settings == settings;',
  'commit: synchronization.commit.as_ref().map(shadow_commit)',
]) {
  assert.ok(adapter.includes(required), `WASM navigation-settings port is missing ${required}`);
}

const synchronizationStart = browser.indexOf('function synchronizeRustNavigationSettings() {');
const synchronizationEnd = browser.indexOf('  } catch (error) {', synchronizationStart);
assert.ok(synchronizationStart >= 0 && synchronizationEnd > synchronizationStart,
  'could not locate browser navigation-settings synchronization');
const synchronization = browser.slice(synchronizationStart, synchronizationEnd);
for (const required of [
  'app.synchronizeNavigationSettings(',
  'if (receipt.sequence != null)',
  "receipt.disposition === 'committed'",
  'if (!receipt.matchesInput)',
  'applyRustNavigationSettingsProjection(rust)',
]) {
  assert.ok(synchronization.includes(required), `thin browser adapter is missing ${required}`);
}
for (const retired of [
  'navigationSettingsContentEqual',
  'app.snapshot().navigationSettings',
  '++rustAppShadowSequence',
]) {
  assert.equal(
    synchronization.includes(retired),
    false,
    `browser synchronization must not retain ${retired}`,
  );
}
assert.equal(
  browser.includes('function navigationSettingsContentEqual(left, right)'),
  false,
  'browser glue must not own navigation-settings equality',
);
assert.equal(
  browser.includes('app.setNavigationSettings('),
  false,
  'browser glue must not allocate explicitly sequenced navigation-settings events',
);

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Navigation-settings Rust boundary source smoke passed');
