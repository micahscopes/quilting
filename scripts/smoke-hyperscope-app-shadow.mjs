import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, HyperscopeAppShadow } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const app = new HyperscopeAppShadow();
const asset = 'f0000000-0000-4000-8000-000000000001';
const first = 'e0000000-0000-4000-8000-000000000001';
const second = 'e0000000-0000-4000-8000-000000000002';

const requested = app.requestAsset(
  1,
  0,
  first,
  asset,
  'horse.glb',
  'model/gltf-binary',
);
assert.deepEqual(requested.effects.map(effect => effect.type), ['fetch_asset']);

const replaced = app.requestAsset(
  2,
  0,
  second,
  asset,
  'horse.glb',
  'model/gltf-binary',
);
assert.deepEqual(
  replaced.effects.map(effect => effect.type),
  ['cancel_asset_load', 'fetch_asset'],
);
assert.equal(replaced.effects[0].request_id, first);

const stale = app.completeAssetLoaded(first, asset, 181_808);
assert.equal(stale.disposition, 'ignored_stale');
const afterStale = app.snapshot();
assert.equal(afterStale.loadingAssets, 1);
assert.equal(afterStale.assets[0].status.state, 'loading');
assert.equal(afterStale.assets[0].status.request_id, second);
assert.equal(afterStale.diagnostics[0].code, 'stale_effect_completion');

const applied = app.completeAssetLoaded(second, asset, 181_808);
assert.equal(applied.disposition, 'applied');
const ready = app.snapshot();
assert.equal(ready.loadingAssets, 0);
assert.equal(ready.assets[0].status.state, 'ready');
assert.equal(ready.assets[0].status.byte_length, 181_808);

const presentationDocument = readFileSync(
  `${repository}/examples/hacker-night.presentation.json`,
  'utf8',
);
const presentation = JSON.parse(presentationDocument);
const loadedPresentation = app.loadPresentation(presentationDocument);
assert.equal(loadedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.cueCount, 6);
assert.equal(app.snapshot().presentation.active, undefined);

const startedPresentation = app.present(3, 0, 'start', '');
assert.equal(startedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.active.cue_id, presentation.cues[0].id);
const linkedCue = presentation.cues[2].id;
app.present(4, 0, 'jump', linkedCue);
assert.equal(app.snapshot().presentation.active.cue_id, linkedCue);
assert.throws(
  () => app.present(5, 0, 'jump', 'not-a-uuid'),
  /cue ID must be a UUID/,
);
assert.equal(
  app.snapshot().presentation.active.cue_id,
  linkedCue,
  'a malformed shadow cue must preserve the preceding reducer state',
);

app.advanceFrame(1, 1);
assert.throws(
  () => app.requestAsset(
    3,
    2,
    'e0000000-0000-4000-8000-000000000003',
    asset,
    'horse.glb',
    'model/gltf-binary',
  ),
  /effect-producing input cannot be scheduled/,
);

const finalSnapshot = app.snapshot();
app.free();
console.log(JSON.stringify({
  requested: requested.effects.length,
  replacementEffects: replaced.effects.map(effect => effect.type),
  staleDisposition: stale.disposition,
  readyBytes: ready.assets[0].status.byte_length,
  diagnostics: ready.diagnostics.map(diagnostic => diagnostic.code),
  presentationCue: finalSnapshot.presentation.active.cue_id,
}));
