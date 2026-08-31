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
const host = read('asset_effect_host.mjs');
const settings = read('crates/hyperscope-app/src/settings.rs');

for (const required of [
  'spec!("assetimpl", "rust", Implementation)',
  'pub struct AssetFetchJob',
  'pub struct AssetJobIdentity',
  'pub struct AssetEffectJobs',
  'pub struct AssetLoadRequest',
  'pub fn request_asset_load(',
  'self.dispatch_semantic(SemanticAction::RequestAsset',
  'let mut jobs = AssetEffectJobs::from_commit(&commit)?;',
  'load_cancellations: jobs.load_cancellations',
  'install_cancellations: jobs.install_cancellations',
]) {
  assert.ok(settings.includes(required) || app.includes(required),
    `application asset-request port is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = requestAsset)]',
  '#[wasm_bindgen(js_name = requestPrimaryAsset)]',
  '#[wasm_bindgen(js_name = requestAssetLoad)]',
  '.request_asset_load(',
  'ShadowAssetLoadRequest',
  'ShadowAssetFetchJob::from(request.fetch)',
  'ShadowAssetJobIdentity::from',
]) {
  assert.ok(adapter.includes(required), `WASM asset-request port is missing ${required}`);
}

const requestStart = browser.indexOf('function beginAppAssetShadow(');
const requestEnd = browser.indexOf('function completeAppAssetShadow(', requestStart);
assert.ok(requestStart >= 0 && requestEnd > requestStart,
  'could not locate browser asset-request adapter');
const request = browser.slice(requestStart, requestEnd);
for (const required of [
  'rustAppShadow.requestAssetLoad(',
  "observeRustAppShadowSequence(receipt.sequence, 'Rust asset request')",
  'fetch: receipt.fetch,',
  'loadCancellations: receipt.loadCancellations,',
  'installCancellations: receipt.installCancellations,',
]) {
  assert.ok(request.includes(required), `thin asset-request adapter is missing ${required}`);
}
for (const retired of [
  'rustAppShadow.requestAsset.bind(',
  'rustAppShadow.requestPrimaryAsset.bind(',
  '++rustAppShadowSequence',
  'commit.effects.filter(',
  "effect.type === 'cancel_asset_load'",
]) {
  assert.equal(request.includes(retired), false,
    `ordinary asset requests must not retain ${retired}`);
}

const hostRequestStart = host.indexOf('  begin({');
const hostRequestEnd = host.indexOf('\n  beginInstall(', hostRequestStart);
assert.ok(hostRequestStart >= 0 && hostRequestEnd > hostRequestStart,
  'could not locate browser platform asset host request path');
const hostRequest = host.slice(hostRequestStart, hostRequestEnd);
for (const required of [
  'fetch = null,',
  'loadCancellations = [],',
  'installCancellations = [],',
  'validateFetchJob(fetch)',
  "validateJobList(loadCancellations, 'load cancellations', 'load')",
  "validateJobList(installCancellations, 'install cancellations', 'install')",
]) {
  assert.ok(hostRequest.includes(required), `platform asset host is missing ${required}`);
}
for (const retired of ['commit = null', 'commitEffects(commit)', 'effect.type ===']) {
  assert.equal(hostRequest.includes(retired), false,
    `platform request host must not interpret generic commits through ${retired}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
  encoding: 'utf8',
  input: moduleSource,
});
assert.equal(syntax.status, 0, syntax.stderr);

const hostSyntax = spawnSync(process.execPath, ['--check', 'asset_effect_host.mjs'], {
  cwd: repository,
  encoding: 'utf8',
});
assert.equal(hostSyntax.status, 0, hostSyntax.stderr);

console.log('Asset-request Rust boundary source smoke passed');
