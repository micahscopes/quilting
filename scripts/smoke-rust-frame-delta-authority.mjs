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

assert.ok(
  app.includes('pub fn dispatch_frame_delta(&self, delta_seconds: f64)'),
  'the app store needs an atomic delta-only frame port',
);
assert.ok(
  bridge.includes('#[wasm_bindgen(js_name = advanceFrameDeltaQuiet)]'),
  'the WASM adapter must expose a delta-only Rust frame port',
);
assert.ok(
  bridge.includes('.dispatch_frame_delta(delta_seconds)'),
  'the WASM bridge must derive time through the atomic store operation',
);
assert.ok(
  bridge.includes('self.advance_frame_delta_quiet(0.0)?;'),
  'synchronous camera integration must not reconstruct absolute frame time',
);

const frameStart = browser.indexOf('function advanceRustApplicationFrame(');
const frameEnd = browser.indexOf('\nfunction ', frameStart + 1);
const frameLane = browser.slice(frameStart, frameEnd);
assert.ok(frameLane.includes('rustAppShadow.advanceFrameDeltaQuiet(deltaSeconds)'),
  'ordinary RAF adaptation must use the delta-only port');
assert.equal(frameLane.includes('rustAppShadow.advanceFrameQuiet('), false,
  'ordinary RAF adaptation must not round-trip browser elapsed time');

const selectionStart = browser.indexOf('function advanceRustApplicationClockToSelectionEvent(');
const selectionEnd = browser.indexOf('\nfunction ', selectionStart + 1);
const selectionLane = browser.slice(selectionStart, selectionEnd);
assert.ok(selectionLane.includes('rustAppShadow.advanceFrameDeltaQuiet(partialSeconds)'),
  'selection event-time splitting must use the delta-only port');
assert.equal(selectionLane.includes('rustAppShadow.advanceFrameQuiet('), false,
  'selection event-time splitting must not round-trip browser elapsed time');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Rust frame-delta authority source smoke passed');
