import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const protocol = read('crates/hyperscape-protocol/src/lib.rs');
const application = read('crates/hyperscope-app/src/lib.rs');
const appBridge = read('crates/quilting-wasm/src/app_shadow.rs');
const runtime = read('crates/hyperscape/src/interchange.rs');
const renderer = read('crates/quilting-wasm/src/main_renderer.rs');
const browser = read('hyperscope.html');

for (const required of [
  'SetConformalFrameTransform {',
  'generators: Vec<WireConformalGenerator>',
  'MAX_CONFORMAL_GENERATORS_PER_FRAME',
]) {
  assert.ok(protocol.includes(required), `protocol frame command is missing ${required}`);
}
for (const required of [
  'pub conformal_frames: Vec<AuthoredConformalFrameReadModel>',
  'authored_conformal_frames: BTreeMap<ConformalFrameId, Vec<WireConformalGenerator>>',
  'generators: generators.clone()',
]) {
  assert.ok(application.includes(required), `application projection is missing ${required}`);
}
assert.ok(appBridge.includes('authored_conformal_frames: authored'),
  'the application WASM snapshot must carry the materialized frame projection');
assert.ok(appBridge.includes('frame_id: frame.frame.to_string()'),
  'the application WASM snapshot must preserve the stable frame ID');

const runtimeApplyStart = runtime.indexOf('pub fn apply_authored_conformal_frame_transforms(');
const runtimeApplyEnd = runtime.indexOf('\n    pub fn app(', runtimeApplyStart);
assert.ok(runtimeApplyStart >= 0 && runtimeApplyEnd > runtimeApplyStart,
  'could not locate the conformal-frame runtime boundary');
const runtimeApply = runtime.slice(runtimeApplyStart, runtimeApplyEnd);
for (const required of [
  'let mut staged = self.authored_base_frames.clone();',
  '.frame_id(stable_id)',
  'SurfacePinned(stable_id)',
  'staged.set_local_to_parent(frame, chain)',
  'resource_mut::<ConformalScene>().frames = staged;',
]) {
  assert.ok(runtimeApply.includes(required), `runtime atomic apply is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = "mr_applyAuthoredConformalFrames")]',
  'serde_json::from_str::<Vec<AuthoredConformalFrameInput>>(frames_json)',
  '.apply_authored_conformal_frame_transforms(&transforms)',
  'apply_hyperscape_packets(&browser.runtime.packets_by_node());',
]) {
  assert.ok(renderer.includes(required), `renderer bridge is missing ${required}`);
}

const adapterStart = browser.indexOf('function applyRustAuthoredConformalFrameProjection(');
const adapterEnd = browser.indexOf('\nfunction ', adapterStart + 1);
assert.ok(adapterStart >= 0 && adapterEnd > adapterStart,
  'could not locate the browser frame-projection adapter');
const adapter = browser.slice(adapterStart, adapterEnd);
for (const required of [
  'snapshot.authoredConformalFrames',
  'JSON.stringify(frames)',
  'authoredConformalFrameRuntimeGeneration',
  'mr_applyAuthoredConformalFrames(serialized)',
  "authoredLodReasons.add('authored-conformal-frame')",
]) {
  assert.ok(adapter.includes(required), `thin browser adapter is missing ${required}`);
}
for (const forbidden of ['WireConformalGenerator', 'switch (', 'case \'translation\'']) {
  assert.equal(adapter.includes(forbidden), false,
    `browser adapter must not interpret conformal semantics through ${forbidden}`);
}
assert.ok(browser.includes('applyRustAuthoredConformalFrameProjection(snapshot);'),
  'application snapshot refresh must deliver a changed frame projection');
assert.ok(browser.includes(
  'applyRustAuthoredConformalFrameProjection(rustAppShadowDiagnostics.snapshot);',
), 'a new resident Hyperscape runtime must receive the current complete projection');

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

console.log('Authored conformal-frame projection source smoke passed');
