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
  app.includes('pub animation_pose_sample_time_seconds: f64'),
  'frame snapshots must expose the Rust-owned pose sample clock',
);
assert.ok(
  app.includes('if self.animation.playing {'),
  'the pose sample clock must advance only with playing transport',
);
assert.ok(
  bridge.includes('#[wasm_bindgen(js_name = writeAnimationPoseRequestFromFrame)]'),
  'the WASM adapter must expose the frame-clock pose request port',
);
assert.ok(
  bridge.includes('self.store.animation_pose_sample_time_seconds()'),
  'the per-frame port must not clone a complete frame snapshot',
);
assert.match(
  browser,
  /ANIMATION_CLOCK_IMPLEMENTATION === 'rust'\s*\? app\.writeAnimationPoseRequestFromFrame\(t, animationPosePacket\)/,
  'Rust authority must not round-trip a browser pose clock',
);
assert.match(
  browser,
  /if \(ANIMATION_CLOCK_IMPLEMENTATION !== 'rust'\) \{\s*animationPoseClockSeconds \+= deltaSeconds;/,
  'the legacy pose clock must remain only for JS/shadow parity',
);

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Rust animation pose frame-clock source smoke passed');
