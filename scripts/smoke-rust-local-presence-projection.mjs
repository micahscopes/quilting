import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const app = read('crates/hyperscope-app/src/lib.rs');
const bridge = read('crates/quilting-wasm/src/app_shadow.rs');
const browser = read('hyperscope.html');

assert.ok(app.includes('pub fn local_presence_snapshot('),
  'AppStore must own the local semantic presence projection');
for (const field of [
  'camera: Some(CameraPresence {',
  'selection,',
  'focus: include_focus.then_some(FocusPresence {',
  'active_cue,',
  'animation_seconds,',
]) {
  assert.ok(app.includes(field), `Rust presence projection is missing ${field}`);
}
assert.ok(bridge.includes('#[wasm_bindgen(js_name = localPresenceSample)]'),
  'the WASM adapter must expose the typed local presence projection');

const sampleStart = browser.indexOf('function localPeerPresenceSample()');
const sampleEnd = browser.indexOf('\nfunction ', sampleStart + 1);
const sample = browser.slice(sampleStart, sampleEnd);
for (const gate of [
  "RUST_NAVIGATION_IMPLEMENTATION === 'rust'",
  "RUST_SELECTION_IMPLEMENTATION === 'rust'",
  "ANIMATION_CLOCK_IMPLEMENTATION === 'rust'",
  "ANIMATION_CLIP_IMPLEMENTATION === 'rust'",
  "RUST_PRESENTATION_IMPLEMENTATION === 'rust'",
]) {
  assert.ok(sample.includes(gate), `local presence is missing rollback gate ${gate}`);
}
assert.ok(sample.includes('rustAppShadow.localPresenceSample(LOCAL_PEER_PRESENCE_TTL_MS)'),
  'the Rust-authority presence lane must not reconstruct viewport semantics');
assert.ok(sample.includes('const navigation = manualNavigationState();'),
  'the measured JS/shadow rollback sample must remain available');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Rust local-presence projection source smoke passed');
