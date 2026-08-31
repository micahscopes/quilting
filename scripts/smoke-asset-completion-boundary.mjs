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

for (const required of [
  'pub struct AssetLoadCompletionDispatch',
  'pub fn complete_asset_load(',
  'EffectCompletion::AssetLoad(completion)',
  'let mut jobs = AssetEffectJobs::from_commit(&commit)?;',
  'install: jobs.install.take()',
  '.find(|asset| asset.descriptor.id == expected.asset_id)',
  'decoded.asset,',
]) {
  assert.ok(app.includes(required), `application asset-completion port is missing ${required}`);
}

for (const required of [
  '#[wasm_bindgen(js_name = completeAssetLoaded)]',
  '#[wasm_bindgen(js_name = completeAssetLoadedWithMetadata)]',
  '#[wasm_bindgen(js_name = completeAssetFailed)]',
  '#[wasm_bindgen(js_name = finishAssetLoaded)]',
  '#[wasm_bindgen(js_name = finishAssetLoadedWithMetadata)]',
  '#[wasm_bindgen(js_name = finishAssetFailed)]',
  '.complete_asset_load(AssetLoadCompletion {',
  'struct ShadowAssetLoadCompletion',
  'install: dispatch.install.map(ShadowAssetJobIdentity::from)',
  'asset: dispatch.asset.map(ShadowAsset::from)',
]) {
  assert.ok(adapter.includes(required), `WASM asset-completion port is missing ${required}`);
}

const completionStart = browser.indexOf('function completeAppAssetShadow(');
const completionEnd = browser.indexOf('function primarySceneInstallFacts(', completionStart);
assert.ok(completionStart >= 0 && completionEnd > completionStart,
  'could not locate browser asset-completion adapter');
const completion = browser.slice(completionStart, completionEnd);
for (const required of [
  'rustAppShadow.finishAssetLoadedWithMetadata(',
  'rustAppShadow.finishAssetLoaded(',
  'const commit = receipt.commit;',
  'cacheAppAssetReadModel(receipt.asset, commit)',
  'browserAssetEffectHost.beginInstall(token, receipt.install)',
]) {
  assert.ok(completion.includes(required), `thin asset-completion adapter is missing ${required}`);
}
for (const retired of [
  'rustAppShadow.completeAssetLoadedWithMetadata(',
  'rustAppShadow.completeAssetLoaded(',
  'browserAssetEffectHost.beginInstall(token, commit)',
  'commit.effects',
  'refreshAppShadowSnapshot()',
]) {
  assert.equal(completion.includes(retired), false,
    `ordinary asset completion must not retain ${retired}`);
}

const failureStart = browser.indexOf('function failAppAssetShadow(');
const failureEnd = browser.indexOf('function appAssetMayProcess(', failureStart);
assert.ok(failureStart >= 0 && failureEnd > failureStart,
  'could not locate browser asset-failure adapter');
const failure = browser.slice(failureStart, failureEnd);
assert.ok(failure.includes('rustAppShadow.finishAssetFailed('),
  'asset failures must use the typed completion receipt');
assert.ok(failure.includes('cacheAppAssetReadModel(receipt.asset, commit)'),
  'asset failures must consume the compact asset state');
assert.equal(failure.includes('rustAppShadow.completeAssetFailed('), false,
  'ordinary asset failures must not use the generic completion seam');
assert.equal(failure.includes('refreshAppShadowSnapshot()'), false,
  'asset failures must not serialize the complete application state');

const installStart = host.indexOf('  beginInstall(');
const installEnd = host.indexOf('\n  recordCompletion(', installStart);
assert.ok(installStart >= 0 && installEnd > installStart,
  'could not locate platform install authorization');
const install = host.slice(installStart, installEnd);
for (const required of [
  'beginInstall(token, install)',
  "validateJobIdentity(install, 'primary install')",
  'matchingInstall.assetId !== token.assetId',
]) {
  assert.ok(install.includes(required), `platform install host is missing ${required}`);
}
for (const retired of [
  'commit.effects',
  'installCommitEffects(',
  'validateInstallEffect(',
  '.filter(',
  'effect.type',
]) {
  assert.equal(host.includes(retired), false,
    `asset platform host must not interpret generic effects through ${retired}`);
}

const moduleSource = browser.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(moduleSource, 'could not extract the Hyperscope inline module');
for (const [label, source] of [
  ['Hyperscope inline module', moduleSource],
  ['asset platform host', host],
]) {
  const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], {
    encoding: 'utf8',
    input: source,
  });
  assert.equal(syntax.status, 0, `${label}: ${syntax.stderr}`);
}

console.log('Asset-completion Rust boundary source smoke passed');
