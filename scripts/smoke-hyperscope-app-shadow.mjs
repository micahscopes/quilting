import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const {
  default: init,
  encodeLocalPresenceEnvelope,
  HyperscopeAppShadow,
  HyperscopeNavigation,
  build_required_atlas: buildRequiredAtlas,
  export_all_patches: exportAllPatches,
  load_gltf_data: loadGltfData,
  mapPointerTurntableFrame,
  mapSpaceMouseCameraFrame,
  required_tessellation_atlas_triples: requiredAtlasTriples,
} = await import(packageUrl);
const { mapSpaceMouseNavigationAxes } = await import(
  pathToFileURL(`${repository}/spacemouse.mjs`).href
);
const { framedSphereDistance, transportCameraAcrossSphereReflections } = await import(
  pathToFileURL(`${repository}/hyperscope_focus.mjs`).href
);
await init({ module_or_path: readFileSync(wasmPath) });

const encodedPresence = encodeLocalPresenceEnvelope(
  '75000000-0000-4000-8000-000000000001',
  '75000000-0000-4000-8000-000000000002',
  '18446744073709551615',
  1500,
  new Float64Array([8, 9, 10]),
  new Float64Array([0, 0, -1]),
  new Float64Array([0, 1, 0]),
  '["75000000-0000-4000-8000-000000000003"]',
  true,
  new Float64Array([1, 2, 3]),
  4,
  true,
  '',
  2.5,
);
const decodedPresence = JSON.parse(encodedPresence);
assert.match(encodedPresence, /"sequence":18446744073709551615/);
assert.deepEqual(decodedPresence.presence.camera.eye, [8, 9, 10]);
assert.deepEqual(decodedPresence.presence.focus, {
  center: [1, 2, 3],
  radius: 4,
  inversion_enabled: true,
});
assert.throws(
  () => encodeLocalPresenceEnvelope(
    '75000000-0000-4000-8000-000000000001',
    '75000000-0000-4000-8000-000000000002',
    '18446744073709551616',
    1500,
    new Float64Array([0, 0, 3]),
    new Float64Array([0, 0, -1]),
    new Float64Array([0, 1, 0]),
    '[]', false, new Float64Array(), 1, false, '', -1,
  ),
  /invalid decimal u64/,
);
assert.throws(
  () => encodeLocalPresenceEnvelope(
    '75000000-0000-4000-8000-000000000001',
    '75000000-0000-4000-8000-000000000002',
    '1',
    0,
    new Float64Array([0, 0, 3]),
    new Float64Array([0, 0, -1]),
    new Float64Array([0, 0, -1]),
    '[]', false, new Float64Array(), 1, false, '', -1,
  ),
  /presence TTL|independent directions/,
);

assert.equal(requiredAtlasTriples(6, 2).length / 3, 19);
assert.equal(requiredAtlasTriples(6, 4).length / 3, 34);
assert.equal(requiredAtlasTriples(6, 3).length, 0);
assert.ok(buildRequiredAtlas(6, 4) >= 0);
assert.equal(exportAllPatches().patches.length / 7, 34);
assert.equal(buildRequiredAtlas(6, 3), -1);
assert.equal(
  exportAllPatches().patches.length / 7,
  34,
  'a rejected policy must retain the last valid atlas',
);

function authoredTriangleGlb(stableEntityId) {
  const binary = new Uint8Array(44);
  new Float32Array(binary.buffer, 0, 9).set([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  new Uint16Array(binary.buffer, 36, 3).set([0, 1, 2]);
  const document = {
    asset: {
      version: '2.0',
      copyright: 'Example model copyright',
      generator: 'Hyperscope smoke fixture',
      extras: {
        title: 'Authored triangle',
        author: 'Example Author (https://example.test/author)',
        license: 'CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)',
        source: 'https://example.test/authored-triangle',
        unrelated: { retained: false },
      },
    },
    buffers: [{ byteLength: 42 }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 },
      { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 },
    ],
    accessors: [
      {
        bufferView: 0, componentType: 5126, count: 3, type: 'VEC3',
        min: [0, 0, 0], max: [1, 1, 0],
      },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    nodes: [
      {
        mesh: 0,
        translation: [2, 3, 4],
        ...(stableEntityId
          ? { extras: { hyperscape: { stable_id: stableEntityId, frame: 0 } } }
          : {}),
      },
      { name: 'unbound-camera-guide' },
    ],
    scenes: [{ nodes: [0, 1] }],
    scene: 0,
    ...(stableEntityId ? {
      extras: {
        hyperscape: {
          version: '0.1',
          frames: [{ name: 'world', parent: null, generators: [] }],
        },
      },
    } : {}),
  };
  const jsonSource = new TextEncoder().encode(JSON.stringify(document));
  const jsonLength = Math.ceil(jsonSource.length / 4) * 4;
  const result = new Uint8Array(12 + 8 + jsonLength + 8 + binary.length);
  const view = new DataView(result.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, result.length, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  result.fill(0x20, 20, 20 + jsonLength);
  result.set(jsonSource, 20);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binary.length, true);
  view.setUint32(binaryHeader + 4, 0x004e4942, true);
  result.set(binary, binaryHeader + 8);
  return result;
}

const authoredEntity = '71000000-0000-4000-8000-000000000001';
const authoredModel = loadGltfData(authoredTriangleGlb(authoredEntity));
assert.deepEqual(authoredModel.asset_metadata, {
  copyright: 'Example model copyright',
  generator: 'Hyperscope smoke fixture',
  title: 'Authored triangle',
  author: 'Example Author (https://example.test/author)',
  license: 'CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)',
  source: 'https://example.test/authored-triangle',
});
assert.deepEqual(
  authoredModel.node_stable_entity_ids,
  [authoredEntity, null],
  'the loader must export a dense authored node-identity table',
);
assert.deepEqual(Array.from(authoredModel.face_node_indices), [0]);
assert.deepEqual(
  Array.from(authoredModel.node_world_transforms),
  [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    2, 3, 4, 1,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ],
  'the loader must retain dense authored node world transforms',
);
const releaseAuthoredModel = loadGltfData(new Uint8Array(readFileSync(
  `${repository}/examples/hyperscape-blender-demo.glb`,
)));
const releaseStableIds = [
  'f0000000-0000-4000-8000-000000000001',
  'f0000000-0000-4000-8000-000000000002',
  'f0000000-0000-4000-8000-000000000003',
  'f0000000-0000-4000-8000-000000000004',
  'f0000000-0000-4000-8000-000000000005',
];
assert.deepEqual(
  releaseAuthoredModel.node_stable_entity_ids.filter(Boolean).sort(),
  releaseStableIds.slice().sort(),
  'the checked Blender scene must retain its five authored entity IDs',
);
const pickableReleaseIds = new Set(
  Array.from(releaseAuthoredModel.face_node_indices)
    .map(node => releaseAuthoredModel.node_stable_entity_ids[node])
    .filter(Boolean),
);
assert.deepEqual(
  Array.from(pickableReleaseIds).sort(),
  releaseStableIds.filter(id => !id.endsWith('0003')).sort(),
  'four checked release entities must join durable IDs to pickable faces',
);
const ordinaryModel = loadGltfData(authoredTriangleGlb(null));
assert.deepEqual(
  ordinaryModel.node_stable_entity_ids,
  [],
  'ordinary assets must not clone one null identity per renderer node',
);
assert.equal(
  ordinaryModel.node_world_transforms.length,
  0,
  'ordinary assets must not clone one matrix per renderer node',
);

const app = new HyperscopeAppShadow();
assert.equal(app.snapshot().animationPlaying, true);
assert.deepEqual(app.snapshot().renderSettings, {
  revision: '0',
  style: 'pbr',
  resolutionLevel: 0,
  density: 100,
  screenAttenuation: true,
  minPixelsPerSubdivision: 16,
  atlasExponent: 7,
  maxFaceEdgeRatio: 2,
});
const renderSettingsReceipt = app.setRenderSettings(
  80, 'stretch', 6, 12, false, 64, 9, 4,
);
assert.equal(renderSettingsReceipt.commit.disposition, 'applied');
assert.deepEqual(renderSettingsReceipt.render, app.snapshot().renderSettings);
assert.deepEqual(
  {
    style: renderSettingsReceipt.render.style,
    resolutionLevel: renderSettingsReceipt.render.resolutionLevel,
    density: renderSettingsReceipt.render.density,
    screenAttenuation: renderSettingsReceipt.render.screenAttenuation,
    minPixelsPerSubdivision: renderSettingsReceipt.render.minPixelsPerSubdivision,
    atlasExponent: renderSettingsReceipt.render.atlasExponent,
    maxFaceEdgeRatio: renderSettingsReceipt.render.maxFaceEdgeRatio,
  },
  {
    style: 'stretch',
    resolutionLevel: 6,
    density: 12,
    screenAttenuation: false,
    minPixelsPerSubdivision: 64,
    atlasExponent: 9,
    maxFaceEdgeRatio: 4,
  },
);
const beforeRejectedRenderSettings = app.snapshot();
assert.throws(
  () => app.setRenderSettings(81, 'browser_magic', 0, 100, true, 16, 7, 2),
  /unknown backend-neutral render style/,
);
assert.throws(
  () => app.setRenderSettings(82, 'pbr', 0, 100, true, 16, 10, 2),
  /resident atlas exponent must be in \[3,9\]/,
);
assert.deepEqual(app.snapshot(), beforeRejectedRenderSettings);
const pausedAnimation = app.setAnimationPlaying(90, false);
assert.equal(pausedAnimation.playing, false);
assert.equal(pausedAnimation.commit.disposition, 'applied');
assert.equal(app.snapshot().animationPlaying, false);
const resumedAnimation = app.toggleAnimationPlaying(91);
assert.equal(resumedAnimation.playing, true);
assert.equal(app.snapshot().animationPlaying, true);
const directAnimationApp = new HyperscopeAppShadow();
const directPause = directAnimationApp.dispatchAnimationPlaying(false);
assert.equal(directPause.sequence, '0');
assert.equal(directPause.commit.disposition, 'applied');
assert.equal(directPause.playing, false);
const directResume = directAnimationApp.dispatchAnimationToggle();
assert.equal(directResume.sequence, '1');
assert.equal(directResume.commit.disposition, 'applied');
assert.equal(directResume.playing, true);
directAnimationApp.free();
const animationClockApp = new HyperscopeAppShadow();
const restoredAnimation = animationClockApp.setAnimationClock(1, true, 2, -0.5);
assert.deepEqual(
  {
    playing: restoredAnimation.playing,
    timeSeconds: restoredAnimation.timeSeconds,
    speed: restoredAnimation.speed,
  },
  { playing: true, timeSeconds: 2, speed: -0.5 },
);
const animationPacket = new Float64Array(3);
animationClockApp.writeAnimationClock(animationPacket);
assert.deepEqual(Array.from(animationPacket), [1, 2, -0.5]);
animationClockApp.advanceFrameQuiet(0.5, 0.5);
animationClockApp.writeAnimationClock(animationPacket);
assert.deepEqual(Array.from(animationPacket), [1, 1.75, -0.5]);
animationClockApp.writeAnimationSample(3, 2, animationPacket);
assert.deepEqual(Array.from(animationPacket), [1, 4.75, -0.5]);
const soughtAnimation = animationClockApp.seekAnimation(2, 0.25);
assert.equal(soughtAnimation.timeSeconds, 0.25);
const spedAnimation = animationClockApp.setAnimationSpeed(3, 2);
assert.equal(spedAnimation.speed, 2);
assert.throws(
  () => animationClockApp.setAnimationClock(4, true, Number.NaN, 1),
  /animation time and speed/,
);
assert.throws(
  () => animationClockApp.writeAnimationClock(new Float64Array(2)),
  /exactly 3 numbers/,
);
assert.throws(
  () => animationClockApp.writeAnimationSample(0, 0, animationPacket),
  /finite and positive/,
);
assert.deepEqual(
  {
    playing: animationClockApp.snapshot().animationPlaying,
    timeSeconds: animationClockApp.snapshot().animationTimeSeconds,
    speed: animationClockApp.snapshot().animationSpeed,
  },
  { playing: true, timeSeconds: 0.25, speed: 2 },
);
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
assert.throws(
  () => app.sessionNodeIdentities(asset, new Int32Array([0])),
  /requires a ready AppStore asset/,
  'a stale completion must not grant session selection authority',
);

const applied = app.completeAssetLoadedWithMetadata(
  second,
  asset,
  181_808,
  authoredModel.asset_metadata,
);
assert.equal(applied.disposition, 'applied');
const ready = app.snapshot();
assert.equal(ready.loadingAssets, 0);
assert.equal(ready.assets[0].status.state, 'ready');
assert.equal(ready.assets[0].status.byte_length, 181_808);
assert.deepEqual(ready.assets[0].status.metadata, authoredModel.asset_metadata);

const primaryApp = new HyperscopeAppShadow();
const primaryHorse = 'f0000000-0000-4000-8000-000000000010';
const primaryChess = 'f0000000-0000-4000-8000-000000000011';
const primaryFirst = 'e0000000-0000-4000-8000-000000000010';
const primarySecond = 'e0000000-0000-4000-8000-000000000011';
primaryApp.requestPrimaryAsset(
  1,
  0,
  primaryFirst,
  primaryHorse,
  'horse.glb',
  'model/gltf-binary',
);
const primaryReplacement = primaryApp.requestPrimaryAsset(
  2,
  0,
  primarySecond,
  primaryChess,
  'local-glbs/chess.glb',
  'model/gltf-binary',
);
assert.deepEqual(
  primaryReplacement.effects.map(effect => effect.type),
  ['cancel_asset_load', 'fetch_asset'],
  'primary scene replacement must cancel across different asset IDs',
);
assert.equal(primaryReplacement.effects[0].request_id, primaryFirst);
assert.equal(primaryReplacement.effects[0].asset_id, primaryHorse);
let primarySnapshot = primaryApp.snapshot();
assert.equal(primarySnapshot.loadingPrimarySceneAsset, primaryChess);
assert.equal(primarySnapshot.loadingPrimarySceneRequest, primarySecond);
assert.equal(
  primaryApp.completeAssetLoaded(primaryFirst, primaryHorse, 181_808).disposition,
  'ignored_stale',
);
primarySnapshot = primaryApp.snapshot();
assert.equal(primarySnapshot.loadingPrimarySceneRequest, primarySecond);
assert.equal(
  primaryApp.completeAssetLoaded(primarySecond, primaryChess, 200_000).disposition,
  'applied',
);
primarySnapshot = primaryApp.snapshot();
assert.equal(primarySnapshot.loadingPrimarySceneAsset, undefined);
assert.equal(primarySnapshot.loadingPrimarySceneRequest, undefined);

const sessionNodeIdentities = app.sessionNodeIdentities(
  asset,
  new Int32Array([7, 0, 7]),
);
assert.deepEqual(sessionNodeIdentities, [
  {
    assetId: asset,
    entityId: 'eeeeeeee-0000-4000-8000-000000000001',
    sourceNode: 0,
    durable: false,
  },
  {
    assetId: asset,
    entityId: 'eeeeeeee-0000-4000-8000-000000000008',
    sourceNode: 7,
    durable: false,
  },
]);
assert.throws(
  () => app.sessionNodeIdentities(asset, new Int32Array([-1])),
  /must be nonnegative/,
);

// Transport-neutral authored checkpoints cross generated WASM as canonical
// protocol JSON while the projection fence remains lossless beyond JS's safe
// integer range. Stale and malformed batches must leave the materialization
// exactly unchanged.
const authoredTransformEnvelope = JSON.parse(readFileSync(
  `${repository}/crates/hyperscape-protocol/fixtures/authored-set-transform-v0.1.json`,
  'utf8',
));
const authoredAsset = '62000000-0000-4000-8000-000000000001';
const authoredProjectionRevision = '9007199254740993';
const authoredUpsertEnvelope = {
  header: {
    version: { major: 0, minor: 1 },
    message_id: '62000000-0000-4000-8000-000000000002',
    sender: '62000000-0000-4000-8000-000000000003',
    sequence: 2,
  },
  command: {
    type: 'upsert_asset',
    asset: {
      id: authoredAsset,
      uri: 'blender-live.glb',
      media_type: 'model/gltf-binary',
    },
  },
};
const authoredApplied = app.applyAuthoredRevision(
  authoredProjectionRevision,
  JSON.stringify([authoredTransformEnvelope, authoredUpsertEnvelope]),
);
assert.equal(authoredApplied.disposition, 'applied');
const authoredSnapshot = app.snapshot();
assert.equal(
  authoredSnapshot.authoredProjectionRevision,
  authoredProjectionRevision,
  'the authored fence must not round through a JavaScript number',
);
assert.deepEqual(authoredSnapshot.authoredAssets.map(entry => entry.id), [authoredAsset]);
assert.equal(authoredSnapshot.authoredAssets[0].uri, 'blender-live.glb');
assert.equal(authoredSnapshot.authoredAssets[0].mediaType, 'model/gltf-binary');
assert.deepEqual(authoredSnapshot.authoredEntities, [{
  entityId: authoredTransformEnvelope.command.entity,
  translation: [1, 2, 3],
  rotationWxyz: [1, 0, 0, 0],
  scale: [1, 1, 1],
}]);

const staleRemoval = structuredClone(authoredTransformEnvelope);
staleRemoval.header.message_id = '62000000-0000-4000-8000-000000000004';
staleRemoval.header.sequence = 4;
staleRemoval.command = {
  type: 'remove_entity',
  entity: authoredTransformEnvelope.command.entity,
};
assert.equal(
  app.applyAuthoredRevision(
    authoredProjectionRevision,
    JSON.stringify([staleRemoval]),
  ).disposition,
  'ignored_stale',
);
assert.deepEqual(app.snapshot().authoredEntities, authoredSnapshot.authoredEntities);

const validThenInvalid = structuredClone(authoredTransformEnvelope);
validThenInvalid.header.message_id = '62000000-0000-4000-8000-000000000005';
validThenInvalid.header.sequence = 5;
validThenInvalid.command.transform.translation = [9, 9, 9];
const invalidTransform = structuredClone(validThenInvalid);
invalidTransform.header.message_id = '62000000-0000-4000-8000-000000000006';
invalidTransform.header.sequence = 6;
invalidTransform.command.transform.scale = [0, 1, 1];
const beforeInvalidAuthored = app.snapshot();
assert.throws(
  () => app.applyAuthoredRevision(
    '9007199254740994',
    JSON.stringify([validThenInvalid, invalidTransform]),
  ),
  /transform must be finite with nonzero rotation and scale/,
);
const afterInvalidAuthored = app.snapshot();
assert.equal(afterInvalidAuthored.revision, beforeInvalidAuthored.revision);
assert.equal(
  afterInvalidAuthored.authoredProjectionRevision,
  beforeInvalidAuthored.authoredProjectionRevision,
);
assert.deepEqual(afterInvalidAuthored.authoredAssets, beforeInvalidAuthored.authoredAssets);
assert.deepEqual(afterInvalidAuthored.authoredEntities, beforeInvalidAuthored.authoredEntities);
assert.throws(
  () => app.applyAuthoredRevision('18446744073709551616', '[]'),
  /authored projection revision is invalid/,
);
assert.deepEqual(app.snapshot().authoredEntities, beforeInvalidAuthored.authoredEntities);

// The generated application facade owns local-peer admission. A carrier only
// supplies canonical JSON and receipt time; duplicate, stale, echo, atomic
// failure, and TTL policy all remain in Rust.
const peerApp = new HyperscopeAppShadow();
const peerEntity = '63000000-0000-4000-8000-000000000001';
const peerSender = '63000000-0000-4000-8000-000000000002';
const peerEnvelope = structuredClone(authoredTransformEnvelope);
peerEnvelope.header.message_id = '63000000-0000-4000-8000-000000000003';
peerEnvelope.header.sender = peerSender;
peerEnvelope.header.sequence = 10;
peerEnvelope.command.entity = peerEntity;
const peerFrame = { lane: 'authored', envelope: peerEnvelope };
const peerApplied = peerApp.receiveLocalPeerEnvelope(0, JSON.stringify(peerFrame));
assert.equal(peerApplied.lane, 'authored');
assert.equal(peerApplied.disposition, 'applied');
assert.equal(peerApplied.projectionRevision, '0');
assert.equal(peerApplied.commit.disposition, 'applied');
const peerAppliedRevision = peerApp.snapshot().revision;

const peerDuplicate = peerApp.receiveLocalPeerEnvelope(0, JSON.stringify(peerFrame));
assert.equal(peerDuplicate.disposition, 'ignored_duplicate');
assert.equal(peerDuplicate.projectionRevision, undefined);
assert.equal(peerDuplicate.commit, undefined);
assert.equal(peerApp.snapshot().revision, peerAppliedRevision);

const peerStaleEnvelope = structuredClone(peerEnvelope);
peerStaleEnvelope.header.message_id = '63000000-0000-4000-8000-000000000004';
peerStaleEnvelope.header.sequence = 9;
peerStaleEnvelope.command.transform.translation = [9, 9, 9];
const peerStale = peerApp.receiveLocalPeerEnvelope(0, JSON.stringify({
  lane: 'authored',
  envelope: peerStaleEnvelope,
}));
assert.equal(peerStale.disposition, 'ignored_stale');
assert.equal(peerStale.commit, undefined);
assert.equal(peerApp.snapshot().revision, peerAppliedRevision);

const peerInvalidEnvelope = structuredClone(peerEnvelope);
peerInvalidEnvelope.header.message_id = '63000000-0000-4000-8000-000000000005';
peerInvalidEnvelope.header.sequence = 11;
peerInvalidEnvelope.command.transform.scale = [0, 1, 1];
const beforeInvalidPeer = peerApp.snapshot();
assert.throws(
  () => peerApp.receiveLocalPeerEnvelope(0, JSON.stringify({
    lane: 'authored',
    envelope: peerInvalidEnvelope,
  })),
  /transform must be finite with nonzero rotation and scale/,
);
assert.deepEqual(peerApp.snapshot(), beforeInvalidPeer);
peerInvalidEnvelope.command.transform.scale = [1, 1, 1];
const peerCorrected = peerApp.receiveLocalPeerEnvelope(0, JSON.stringify({
  lane: 'authored',
  envelope: peerInvalidEnvelope,
}));
assert.equal(peerCorrected.disposition, 'applied');
assert.equal(peerCorrected.projectionRevision, '1');

const localEnvelope = structuredClone(peerEnvelope);
localEnvelope.header.message_id = '63000000-0000-4000-8000-000000000006';
localEnvelope.header.sequence = 12;
localEnvelope.command.transform.translation = [12, 0, 0];
assert.equal(
  peerApp.applyAuthoredRevision('2', JSON.stringify([localEnvelope])).disposition,
  'applied',
);
peerApp.recordLocalAuthoredEnvelope(JSON.stringify(localEnvelope));
const localEchoFrame = JSON.stringify({ lane: 'authored', envelope: localEnvelope });
const peerEcho = peerApp.receiveLocalPeerEnvelope(0, localEchoFrame);
assert.equal(peerEcho.disposition, 'ignored_echo');
assert.equal(peerEcho.commit, undefined);
assert.equal(
  peerApp.receiveLocalPeerEnvelope(0, localEchoFrame).disposition,
  'ignored_duplicate',
  'a consumed local echo must remain retry-safe',
);

const presenceEnvelope = JSON.parse(readFileSync(
  `${repository}/crates/hyperscape-protocol/fixtures/presence-camera-v0.1.json`,
  'utf8',
));
presenceEnvelope.header.message_id = '63000000-0000-4000-8000-000000000007';
presenceEnvelope.header.sender = '63000000-0000-4000-8000-000000000008';
presenceEnvelope.header.sequence = 1;
presenceEnvelope.presence.ttl_millis = 100;
presenceEnvelope.presence.selection = [peerEntity];
const presenceFrame = JSON.stringify({ lane: 'presence', envelope: presenceEnvelope });
const presenceApplied = peerApp.receiveLocalPeerEnvelope(5, presenceFrame);
assert.equal(presenceApplied.lane, 'presence');
assert.equal(presenceApplied.disposition, 'applied');
assert.equal(presenceApplied.projectionRevision, undefined);
assert.equal(presenceApplied.commit.publishedUi, false);
const livePresence = peerApp.peerPresenceSnapshot();
assert.equal(livePresence.elapsedSeconds, 0);
assert.equal(livePresence.peers.length, 1);
assert.equal(livePresence.peers[0].peerId, presenceEnvelope.header.sender);
assert.equal(livePresence.peers[0].sequence, '1');
assert.equal(livePresence.peers[0].expiresAtSeconds, 5.1);
assert.deepEqual(livePresence.peers[0].presence.selection, [peerEntity]);
assert.equal(
  peerApp.receiveLocalPeerEnvelope(5.01, presenceFrame).disposition,
  'ignored_duplicate',
);
peerApp.advanceFrame(5.2, 5.2);
const expiredPresence = peerApp.peerPresenceSnapshot();
assert.equal(expiredPresence.elapsedSeconds, 5.2);
assert.deepEqual(expiredPresence.peers, []);

const localPresenceEnvelope = structuredClone(presenceEnvelope);
localPresenceEnvelope.header.message_id = '63000000-0000-4000-8000-000000000009';
localPresenceEnvelope.header.sender = '63000000-0000-4000-8000-00000000000a';
peerApp.recordLocalPresenceEnvelope(JSON.stringify(localPresenceEnvelope));
const localPresenceFrame = JSON.stringify({
  lane: 'presence',
  envelope: localPresenceEnvelope,
});
assert.equal(
  peerApp.receiveLocalPeerEnvelope(5.3, localPresenceFrame).disposition,
  'ignored_echo',
);
assert.deepEqual(peerApp.peerPresenceSnapshot().peers, []);
assert.equal(
  peerApp.receiveLocalPeerEnvelope(5.31, localPresenceFrame).disposition,
  'ignored_duplicate',
);
peerApp.free();

const identityMatrix = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];
const authoredPackedSceneInput = [{
  layer: '62000000-0000-4000-8000-000000000010',
  asset: authoredAsset,
  layerTransform: {
    translation: [10, 0, 0],
    rotation: [1, 0, 0, 0],
    scale: [2, 1, 1],
  },
  // Deliberately reverse packed order. Rust owns output ordering.
  nodes: [
    {
      packedNode: 9,
      sourceNode: 2,
      entityId: null,
      sourceWorld: identityMatrix.map((value, index) => index === 12 ? 4 : value),
    },
    {
      packedNode: 3,
      sourceNode: 1,
      entityId: authoredTransformEnvelope.command.entity,
      sourceWorld: identityMatrix.map((value, index) => index === 12 ? 99 : value),
    },
  ],
}];
const extractedScene = app.extractPackedScene(JSON.stringify(authoredPackedSceneInput));
assert.equal(extractedScene.appRevision, app.snapshot().revision);
assert.equal(extractedScene.authoredProjectionRevision, authoredProjectionRevision);
assert.deepEqual(extractedScene.nodes.map(node => node.packedNode), [3, 9]);
assert.deepEqual(extractedScene.nodes.map(node => node.source), [
  'authored_absolute',
  'gltf_world',
]);
assert.deepEqual(extractedScene.nodes[0].matrix, [
  2, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  12, 2, 3, 1,
]);
assert.deepEqual(extractedScene.nodes[1].matrix, [
  2, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  18, 0, 0, 1,
]);
assert.deepEqual(extractedScene.unmatchedAuthoredEntities, []);
const beforeInvalidExtraction = app.snapshot();
const duplicatePackedSceneInput = structuredClone(authoredPackedSceneInput);
duplicatePackedSceneInput[0].nodes[0].packedNode = 3;
assert.throws(
  () => app.extractPackedScene(JSON.stringify(duplicatePackedSceneInput)),
  /repeats renderer node 3/,
);
assert.deepEqual(app.snapshot(), beforeInvalidExtraction);

const presentationDocument = readFileSync(
  `${repository}/crates/hyperscape/fixtures/hacker-night.presentation.json`,
  'utf8',
);
const presentation = JSON.parse(presentationDocument);
const loadedPresentation = app.loadPresentation(presentationDocument);
const incumbent = new HyperscopeNavigation();
incumbent.loadPresentation(presentationDocument);
assert.equal(loadedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.cueCount, presentation.cues.length);
assert.deepEqual(app.snapshot().presentation.assets, presentation.assets);
assert.equal(app.snapshot().presentation.active, undefined);

const directPresentationApp = new HyperscopeAppShadow();
directPresentationApp.loadPresentation(presentationDocument);
const directPresentationStart = directPresentationApp.dispatchPresentation('start', '');
assert.equal(directPresentationStart.sequence, '0');
assert.equal(directPresentationStart.commit.disposition, 'applied');
assert.equal(
  directPresentationApp.snapshot().presentation.active.cue_id,
  presentation.cues[0].id,
);
const directPresentationAdvance = directPresentationApp.dispatchPresentation('advance', '');
assert.equal(directPresentationAdvance.sequence, '1');
assert.equal(directPresentationAdvance.commit.disposition, 'applied');
assert.equal(
  directPresentationApp.snapshot().presentation.active.cue_id,
  presentation.cues[1].id,
);
const beforeRejectedDirectPresentation = directPresentationApp.snapshot();
assert.throws(
  () => directPresentationApp.dispatchPresentation('jump', 'not-a-uuid'),
  /cue ID must be a UUID/,
);
assert.deepEqual(
  directPresentationApp.snapshot(),
  beforeRejectedDirectPresentation,
  'rejected direct cue input must preserve state and sequence authority',
);
const directPresentationReverse = directPresentationApp.dispatchPresentation('reverse', '');
assert.equal(
  directPresentationReverse.sequence,
  '2',
  'rejected direct cue input must not consume a sequence number',
);
directPresentationApp.free();

const eye = new Float64Array([0, 0, 3]);
const forward = new Float64Array([0, 0, -1]);
const up = new Float64Array([0, 1, 0]);
const target = new Float64Array([0, 0, 0]);
const focusCenter = new Float64Array([0.5, 0, 0]);
const projectionLens = [1.25, 0.002, 25_000];
function assertNavigationParity(actual, expected) {
  assertNavigationContentParity(actual, expected);
  assert.equal(actual.pending_actions, expected.pending_actions);
  assert.equal(actual.last_applied_sequence, expected.last_applied_sequence);
}

function assertNavigationContentParity(actual, expected) {
  assert.equal(actual.elapsed_seconds, expected.elapsed_seconds);
  assert.equal(actual.preset, expected.preset);
  for (const field of ['eye', 'orientation', 'right', 'up', 'forward', 'semantic_target']) {
    assert.deepEqual(actual.camera[field], expected.camera[field]);
  }
  for (const field of [
    'control_distance', 'camera_transition_remaining',
    'vertical_fov_radians', 'near', 'far',
    'surface_anchor_transition_remaining', 'surface_anchor_hop_height',
  ]) {
    assert.equal(actual.camera[field], expected.camera[field]);
  }
  assert.deepEqual(actual.focus.center, expected.focus.center);
  for (const field of [
    'radius', 'anchored', 'focus_enabled', 'inversion_enabled', 'focus_coordinate',
    'angular_aperture', 'focus_transition_remaining',
  ]) {
    assert.equal(actual.focus[field], expected.focus[field]);
  }
  assert.deepEqual(actual.selected_focus, expected.selected_focus);
  assert.equal(actual.reflection, expected.reflection);
  assert.deepEqual(actual.diagnostics, expected.diagnostics);
}

function assertSpaceMouseCameraPacket(packet, snapshot) {
  assert.equal(packet.length, 17);
  assert.deepEqual(Array.from(packet.slice(0, 3)), snapshot.camera.eye);
  assert.deepEqual(Array.from(packet.slice(3, 6)), snapshot.camera.right);
  assert.deepEqual(Array.from(packet.slice(6, 9)), snapshot.camera.up);
  assert.deepEqual(Array.from(packet.slice(9, 12)), snapshot.camera.forward);
  assert.equal(packet[12], snapshot.camera.control_distance);
  assert.equal(packet[13], snapshot.camera.semantic_target === undefined ? 0 : 1);
  assert.deepEqual(
    Array.from(packet.slice(14, 17)),
    snapshot.camera.semantic_target ?? [0, 0, 0],
  );
}

// A generated-WASM authority gate for the exact browser rollback oracle: the
// proposed inversion sphere is centered at the camera eye. The queued action
// is consumed, but camera, focus intent, and active reflection remain one
// coherent identity-chart transaction after the transport reaches its pole.
const poleApp = new HyperscopeAppShadow();
const poleCenter = new Float64Array([0, 0, 3]);
poleApp.synchronizeNavigation(
  eye, forward, up, 3, target, ...projectionLens,
  poleCenter, 2, false, false, 0.5, 0.1,
);
const beforePole = poleApp.navigationSnapshot();
assert.equal(poleApp.setInversionEnabled(true), 0n);
assert.equal(poleApp.navigationSnapshot().pending_actions, 1);
const afterPole = poleApp.tickNavigation(0);
assert.equal(afterPole.pending_actions, 0);
assert.equal(afterPole.last_applied_sequence, 0);
assert.equal(afterPole.reflection, 'identity');
assert.equal(afterPole.focus.inversion_enabled, false);
assert.deepEqual(afterPole.focus.center, beforePole.focus.center);
assert.equal(afterPole.focus.radius, beforePole.focus.radius);
assert.deepEqual(afterPole.camera, beforePole.camera);
assert.match(
  afterPole.diagnostics.at(-1),
  /camera transport reached a spherical-reflection pole/,
);
poleApp.free();

// Lens and semantic-target presence are camera semantics, not incidental
// browser projection state. Exercise both generated facades against the
// independent JavaScript conformal transport oracle in both aim modes.
const assertArrayClose = (actual, expected, tolerance = 1e-11) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= tolerance,
      `axis ${index}: ${value} != ${expected[index]}`,
    );
  });
};
for (const semanticTargetEnabled of [false, true]) {
  const policyApp = new HyperscopeAppShadow();
  const policyIncumbent = new HyperscopeNavigation();
  for (const candidate of [policyApp, policyIncumbent]) {
    const synchronize = candidate instanceof HyperscopeAppShadow
      ? candidate.synchronizeNavigation.bind(candidate)
      : candidate.synchronizeState.bind(candidate);
    synchronize(
      eye, forward, up, 3, new Float64Array(), ...projectionLens,
      focusCenter, 2, false, false, 0.5, 0.1,
    );
  }
  assertNavigationParity(policyApp.navigationSnapshot(), policyIncumbent.snapshot());
  const beforeInvalidLensApp = policyApp.navigationSnapshot();
  const beforeInvalidLensIncumbent = policyIncumbent.snapshot();
  assert.throws(
    () => policyApp.setPerspectiveLens(Number.NaN, 0.01, 10_000),
    /camera lens values are invalid/,
  );
  assert.throws(
    () => policyIncumbent.setPerspectiveLens(Number.NaN, 0.01, 10_000),
    /camera lens values are invalid/,
  );
  assert.deepEqual(policyApp.navigationSnapshot(), beforeInvalidLensApp);
  assert.deepEqual(policyIncumbent.snapshot(), beforeInvalidLensIncumbent);

  assert.equal(
    policyApp.setPerspectiveLens(1.1, 0.003, 30_000),
    policyIncumbent.setPerspectiveLens(1.1, 0.003, 30_000),
  );
  assertNavigationParity(policyApp.tickNavigation(0), policyIncumbent.tick(0));
  assert.equal(policyApp.navigationSnapshot().camera.vertical_fov_radians, 1.1);
  assert.equal(policyApp.navigationSnapshot().camera.near, 0.003);
  assert.equal(policyApp.navigationSnapshot().camera.far, 30_000);
  assert.equal(
    policyApp.setPerspectiveLens(...projectionLens),
    policyIncumbent.setPerspectiveLens(...projectionLens),
  );
  assertNavigationParity(policyApp.tickNavigation(0), policyIncumbent.tick(0));

  if (semanticTargetEnabled) {
    assert.equal(
      policyApp.setSemanticTargetEnabled(true),
      policyIncumbent.setSemanticTargetEnabled(true),
    );
    assertNavigationParity(policyApp.tickNavigation(0), policyIncumbent.tick(0));
    assert.deepEqual(policyApp.navigationSnapshot().camera.semantic_target, [0, 0, 0]);
  }
  const reflectionCenter = new Float64Array([1, 0, 0]);
  for (const candidate of [policyApp, policyIncumbent]) {
    candidate.setFreeFocusSphere(reflectionCenter, 2);
  }
  assertNavigationParity(policyApp.tickNavigation(0), policyIncumbent.tick(0));
  const beforeReflection = policyApp.navigationSnapshot();
  const expectedReflection = transportCameraAcrossSphereReflections(
    {
      eye: beforeReflection.camera.eye,
      target: semanticTargetEnabled ? beforeReflection.camera.semantic_target : null,
      basis: [
        ...beforeReflection.camera.right,
        ...beforeReflection.camera.up,
        ...beforeReflection.camera.forward,
      ],
      orbitDistance: beforeReflection.camera.control_distance,
    },
    { enabled: false },
    { enabled: true, center: [1, 0, 0], radius: 2 },
  );
  assert.ok(expectedReflection);
  assert.equal(
    policyApp.setInversionEnabled(true),
    policyIncumbent.setInversionEnabled(true),
  );
  assertNavigationParity(policyApp.tickNavigation(0), policyIncumbent.tick(0));
  const reflected = policyApp.navigationSnapshot();
  assertArrayClose(reflected.camera.eye, expectedReflection.eye);
  assertArrayClose(
    [
      ...reflected.camera.right,
      ...reflected.camera.up,
      ...reflected.camera.forward,
    ],
    expectedReflection.basis,
  );
  assert.ok(
    Math.abs(reflected.camera.control_distance - expectedReflection.orbitDistance) <= 1e-11,
  );
  if (semanticTargetEnabled) {
    assertArrayClose(reflected.camera.semantic_target, expectedReflection.target);
  } else {
    assert.equal(reflected.camera.semantic_target, undefined);
  }
  assert.equal(reflected.camera.vertical_fov_radians, projectionLens[0]);
  assert.equal(reflected.camera.near, projectionLens[1]);
  assert.equal(reflected.camera.far, projectionLens[2]);
  policyApp.free();
  policyIncumbent.free();
}

// Selection-pivot aiming preserves camera distance/orientation while moving
// the explicit target along the established browser target-orbit path.
const aimApp = new HyperscopeAppShadow();
const aimIncumbent = new HyperscopeNavigation();
for (const candidate of [aimApp, aimIncumbent]) {
  const synchronize = candidate instanceof HyperscopeAppShadow
    ? candidate.synchronizeNavigation.bind(candidate)
    : candidate.synchronizeState.bind(candidate);
  synchronize(
    eye, forward, up, 3, new Float64Array(), ...projectionLens,
    focusCenter, 2, false, false, 0.5, 0.1,
  );
  candidate.anchorFocus(
    '60000000-0000-4000-8000-000000000001',
    '70000000-0000-4000-8000-000000000001',
    new Float64Array([4, 0, 0]),
    1,
    new Float64Array([4, 0, 0]),
    1.1,
    0,
    'linear',
  );
}
assertNavigationParity(aimApp.tickNavigation(0), aimIncumbent.tick(0));
assert.equal(
  aimApp.aimAtSelection(0.6, 'smootherstep'),
  aimIncumbent.aimAtSelection(0.6, 'smootherstep'),
);
assertNavigationParity(aimApp.tickNavigation(0), aimIncumbent.tick(0));
assert.deepEqual(aimApp.navigationSnapshot().camera.semantic_target, [0, 0, 0]);
const aimMidpoint = aimApp.tickNavigation(0.3);
assertNavigationParity(aimMidpoint, aimIncumbent.tick(0.3));
assertArrayClose(aimMidpoint.camera.semantic_target, [2, 0, 0]);
assertArrayClose(aimMidpoint.camera.eye, [2, 0, 3]);
assert.ok(Math.abs(aimMidpoint.camera.control_distance - 3) <= 1e-12);
assertNavigationParity(aimApp.tickNavigation(0.3), aimIncumbent.tick(0.3));
assertArrayClose(aimApp.navigationSnapshot().camera.semantic_target, [4, 0, 0]);
assertArrayClose(aimApp.navigationSnapshot().camera.eye, [4, 0, 3]);
aimApp.free();
aimIncumbent.free();

// Selection remains one source-chart identity while the derived pivot/radius
// follows the active reflection chart. A selected pivot at the reflection pole
// becomes absent without destroying the source selection.
const selectionApp = new HyperscopeAppShadow();
const selectionIncumbent = new HyperscopeNavigation();
for (const candidate of [selectionApp, selectionIncumbent]) {
  const synchronize = candidate instanceof HyperscopeAppShadow
    ? candidate.synchronizeNavigation.bind(candidate)
    : candidate.synchronizeState.bind(candidate);
  synchronize(
    eye, forward, up, 3, new Float64Array(), ...projectionLens,
    focusCenter, 2, false, false, 0.5, 0.1,
  );
}
const selectedEntity = '70000000-0000-4000-8000-000000000001';
const selectedAsset = '60000000-0000-4000-8000-000000000001';
const selectedBoundCenter = new Float64Array([0, 0, 0]);
const selectedPivot = new Float64Array([4, 0, 0]);
const nilEntity = '00000000-0000-0000-0000-000000000000';
const appBeforeNilEntity = selectionApp.navigationSnapshot();
const incumbentBeforeNilEntity = selectionIncumbent.snapshot();
assert.throws(
  () => selectionApp.anchorFocus(
    nilEntity, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
  /asset ID must not be nil/,
);
assert.throws(
  () => selectionIncumbent.anchorFocus(
    nilEntity, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
  /asset ID must not be nil/,
);
assert.throws(
  () => selectionApp.anchorFocus(
    selectedAsset, nilEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
  /entity ID must not be nil/,
);
assert.throws(
  () => selectionIncumbent.anchorFocus(
    selectedAsset, nilEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
  /entity ID must not be nil/,
);
assert.deepEqual(selectionApp.navigationSnapshot(), appBeforeNilEntity);
assert.deepEqual(selectionIncumbent.snapshot(), incumbentBeforeNilEntity);
assert.equal(
  selectionApp.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
  selectionIncumbent.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1, 0, 'smootherstep',
  ),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(selectionApp.navigationSnapshot().selected_focus, {
  asset_id: selectedAsset,
  entity_id: selectedEntity,
  source_bound_center: [0, 0, 0],
  source_bound_radius: 2,
  source_pivot: [4, 0, 0],
  margin: 1,
  output_pivot: [4, 0, 0],
  output_radius: 2,
});
const reframeAspect = 16 / 9;
const reframeMargin = 1.15;
const reframeDuration = 0.7;
const beforeSelectedReframe = selectionApp.navigationSnapshot();
assert.equal(
  selectionApp.reframeSelection(
    reframeAspect, reframeMargin, reframeDuration, 'smootherstep',
  ),
  selectionIncumbent.reframeSelection(
    reframeAspect, reframeMargin, reframeDuration, 'smootherstep',
  ),
);
assert.deepEqual(
  selectionApp.navigationSnapshot().camera,
  beforeSelectedReframe.camera,
  'semantic reframe remains queued until the shared frame boundary',
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.equal(
  selectionApp.navigationSnapshot().camera.camera_transition_remaining,
  reframeDuration,
);
const expectedReframeDistance = Math.min(
  Math.max(
    framedSphereDistance(2, reframeAspect, projectionLens[0], reframeMargin),
    0.1,
  ),
  100,
);
const midpointReframe = selectionApp.tickNavigation(0.35);
assertNavigationParity(midpointReframe, selectionIncumbent.tick(0.35));
const expectedMidpointDistance = Math.sqrt(
  beforeSelectedReframe.camera.control_distance * expectedReframeDistance,
);
assertArrayClose(midpointReframe.camera.semantic_target, [2, 0, 0]);
assertArrayClose(midpointReframe.camera.eye, [2, 0, expectedMidpointDistance]);
assert.ok(
  Math.abs(midpointReframe.camera.control_distance - expectedMidpointDistance) <= 1e-12,
);
assertNavigationParity(selectionApp.tickNavigation(0.35), selectionIncumbent.tick(0.35));
const reframedSelection = selectionApp.navigationSnapshot();
assertArrayClose(reframedSelection.camera.eye, [4, 0, expectedReframeDistance]);
assertArrayClose(reframedSelection.camera.semantic_target, [4, 0, 0]);
assert.ok(
  Math.abs(reframedSelection.camera.control_distance - expectedReframeDistance) <= 1e-12,
);
assert.equal(reframedSelection.camera.camera_transition_remaining, undefined);

const beforeInvalidReframe = selectionApp.navigationSnapshot();
assert.equal(
  selectionApp.reframeSelection(0, reframeMargin, reframeDuration, 'smootherstep'),
  selectionIncumbent.reframeSelection(0, reframeMargin, reframeDuration, 'smootherstep'),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(
  selectionApp.navigationSnapshot().camera,
  beforeInvalidReframe.camera,
  'invalid framing must not partially mutate the camera',
);
assert.match(
  selectionApp.navigationSnapshot().diagnostics.at(-1),
  /camera framing radius, aspect, field of view, and margin are invalid/,
);
assert.equal(
  selectionApp.applyFocusToRenderer(17, selectedAsset, selectedEntity),
  false,
  'identity-checked renderer application must be inert before renderer initialization',
);
assert.throws(
  () => selectionApp.applyFocusToRenderer(
    17,
    '60000000-0000-4000-8000-000000000002',
    selectedEntity,
  ),
  /identity does not match/,
);
assert.equal(
  selectionApp.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1.25, 0.8, 'linear',
  ),
  selectionIncumbent.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, selectedPivot, 1.25, 0.8, 'linear',
  ),
);
assertNavigationParity(selectionApp.tickNavigation(0.25), selectionIncumbent.tick(0.25));
const beforeSelectedGesture = selectionApp.navigationSnapshot();
assert.ok(beforeSelectedGesture.focus.radius > 2 && beforeSelectedGesture.focus.radius < 2.5);
assert.equal(
  selectionApp.refitFocusAndToggleInversion(0.6, 'smootherstep'),
  selectionIncumbent.refitFocusAndToggleInversion(0.6, 'smootherstep'),
);
const selectedGesture = selectionApp.tickNavigation(0);
assertNavigationParity(selectedGesture, selectionIncumbent.tick(0));
assert.equal(selectedGesture.focus.inversion_enabled, true);
assert.equal(selectedGesture.focus.focus_transition_remaining, 0.6);
assert.equal(selectedGesture.selected_focus.margin, 1.25);
assertNavigationParity(selectionApp.tickNavigation(0.6), selectionIncumbent.tick(0.6));
assert.ok(Math.abs(selectionApp.navigationSnapshot().focus.radius - 2.5) <= 1e-12);
assert.equal(
  selectionApp.setInversionEnabled(false),
  selectionIncumbent.setInversionEnabled(false),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.equal(
  selectionApp.setInversionEnabled(true),
  selectionIncumbent.setInversionEnabled(true),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(selectionApp.navigationSnapshot().selected_focus.output_pivot, [1.5625, 0, 0]);
assert.equal(selectionApp.navigationSnapshot().selected_focus.output_radius, 0.78125);

assert.equal(
  selectionApp.detachFocus(),
  selectionIncumbent.detachFocus(),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.equal(selectionApp.navigationSnapshot().selected_focus, undefined);
assert.equal(selectionApp.navigationSnapshot().reflection, 'sphere_reflection');
assert.equal(selectionApp.applyFocusToRenderer(-1, '', ''), false);

const polePivot = new Float64Array([0, 0, 0]);
assert.equal(
  selectionApp.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, polePivot, 1, 0, 'smootherstep',
  ),
  selectionIncumbent.anchorFocus(
    selectedAsset, selectedEntity, selectedBoundCenter, 2, polePivot, 1, 0, 'smootherstep',
  ),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(
  selectionApp.navigationSnapshot().selected_focus.source_pivot,
  [0, 0, 0],
);
assert.equal(selectionApp.navigationSnapshot().selected_focus.output_pivot, undefined);
assert.equal(selectionApp.navigationSnapshot().selected_focus.output_radius, undefined);
const beforePoleAim = selectionApp.navigationSnapshot();
assert.equal(
  selectionApp.aimAtSelection(0.7, 'smootherstep'),
  selectionIncumbent.aimAtSelection(0.7, 'smootherstep'),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(
  selectionApp.navigationSnapshot().camera,
  beforePoleAim.camera,
  'aiming at a selected reflection pole must leave the camera untouched',
);
assert.equal(selectionApp.navigationSnapshot().camera.camera_transition_remaining, undefined);
assert.match(selectionApp.navigationSnapshot().diagnostics.at(-1), /reflection pole/);
const beforePoleReframe = selectionApp.navigationSnapshot();
assert.equal(
  selectionApp.reframeSelection(1, 1.15, 0.7, 'smootherstep'),
  selectionIncumbent.reframeSelection(1, 1.15, 0.7, 'smootherstep'),
);
assertNavigationParity(selectionApp.tickNavigation(0), selectionIncumbent.tick(0));
assert.deepEqual(
  selectionApp.navigationSnapshot().camera,
  beforePoleReframe.camera,
  'a selected pivot at the reflection pole must leave the camera untouched',
);
assert.equal(selectionApp.navigationSnapshot().camera.camera_transition_remaining, undefined);
assert.match(selectionApp.navigationSnapshot().diagnostics.at(-1), /reflection pole/);
selectionApp.free();
selectionIncumbent.free();

// The browser frame adapter can advance the Rust application clock without
// paying to serialize a navigation object on every settled frame. Snapshots
// remain explicit and the same transition stays cadence-equivalent to the
// standalone incumbent controller.
const clockApp = new HyperscopeAppShadow();
const clockIncumbent = new HyperscopeNavigation();
for (const candidate of [clockApp, clockIncumbent]) {
  const synchronize = candidate instanceof HyperscopeAppShadow
    ? candidate.synchronizeNavigation.bind(candidate)
    : candidate.synchronizeState.bind(candidate);
  synchronize(
    eye, forward, up, 3, new Float64Array(), ...projectionLens,
    focusCenter, 2, true, false, 0.5, 0.1,
  );
}
assert.equal(
  clockApp.anchorFocus(
    selectedAsset,
    selectedEntity,
    new Float64Array([3, 1, -2]),
    0.75,
    new Float64Array([3, 1, -2]),
    1.1,
    1,
    'smootherstep',
  ),
  clockIncumbent.anchorFocus(
    selectedAsset,
    selectedEntity,
    new Float64Array([3, 1, -2]),
    0.75,
    new Float64Array([3, 1, -2]),
    1.1,
    1,
    'smootherstep',
  ),
);
assertNavigationParity(clockApp.tickNavigation(0), clockIncumbent.tick(0));
assert.equal(clockApp.advanceFrameQuiet(0.25, 0.25), undefined);
assertNavigationParity(clockApp.navigationSnapshot(), clockIncumbent.tick(0.25));
const beforeRejectedFrame = clockApp.navigationSnapshot();
assert.throws(() => clockApp.advanceFrameQuiet(0.24, -0.01), /time/);
assert.deepEqual(clockApp.navigationSnapshot(), beforeRejectedFrame);
assert.equal(clockApp.advanceFrameQuiet(1, 0.75), undefined);
assertNavigationParity(clockApp.navigationSnapshot(), clockIncumbent.tick(0.75));
assert.equal(clockApp.navigationSnapshot().focus.focus_transition_remaining, undefined);
clockApp.free();
clockIncumbent.free();

incumbent.synchronizeState(
  eye, forward, up, 3, target, ...projectionLens,
  focusCenter, 2, false, false, 0.5, 0.1,
);
const synchronized = app.synchronizeNavigation(
  eye,
  forward,
  up,
  3,
  target,
  ...projectionLens,
  focusCenter,
  2,
  false,
  false,
  0.5,
  0.1,
);
assert.equal(synchronized.publishedUi, false);
const startedPresentation = app.present(3, 'start', '');
const incumbentStart = incumbent.startPresentation();
assert.equal(startedPresentation.disposition, 'applied');
assert.equal(app.snapshot().presentation.active.cue_id, presentation.cues[0].id);
assert.deepEqual(app.snapshot().presentation.active, incumbentStart);
assert.equal(
  app.snapshot().presentation.active.render_style,
  'matcap_wire',
  'the AppStore presentation projection must retain Rust render semantics',
);
assert.deepEqual(
  app.snapshot().presentation.active.tessellation,
  incumbentStart.tessellation,
  'the AppStore presentation projection must retain Rust tessellation policy exactly',
);
assert.deepEqual(
  app.snapshot().renderSettings,
  {
    revision: app.snapshot().revision,
    style: incumbentStart.render_style,
    resolutionLevel: 6,
    density: incumbentStart.tessellation.density,
    screenAttenuation: incumbentStart.tessellation.screen_attenuation,
    minPixelsPerSubdivision: incumbentStart.tessellation.min_pixels_per_subdivision,
    atlasExponent: 9,
    maxFaceEdgeRatio: 4,
  },
  'cue activation must replace only authored render policy in the AppStore',
);
const activePresentation = app.snapshot().presentation.active;
const activePresentationBindings = activePresentation.layers.map((layer, index) => ({
  layer: layer.id,
  asset: layer.asset,
  nodes: [{
    packedNode: index + 20,
    sourceNode: index,
    entityId: index === 0 ? authoredTransformEnvelope.command.entity : null,
    sourceWorld: identityMatrix.map((value, element) => element === 12 ? 99 : value),
  }],
}));
const activeComposition = app.extractActivePresentationScene(
  JSON.stringify(activePresentationBindings),
);
assert.equal(activeComposition.appRevision, app.snapshot().revision);
assert.equal(activeComposition.authoredProjectionRevision, authoredProjectionRevision);
assert.equal(activeComposition.cueId, activePresentation.cue_id);
assert.equal(activeComposition.sceneId, activePresentation.scene_id);
assert.deepEqual(
  activeComposition.nodes.map(node => node.packedNode),
  activePresentation.layers.map((_, index) => index + 20),
);
assert.equal(activeComposition.nodes[0].layer, activePresentation.layers[0].id);
assert.equal(activeComposition.nodes[0].asset, activePresentation.layers[0].asset);
assert.equal(activeComposition.nodes[0].source, 'authored_absolute');
assert.deepEqual(activeComposition.nodes[0].matrix, [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  1, 2, 3, 1,
]);
assert.equal(activeComposition.nodes[0].visible, activePresentation.layers[0].visible);
assert.equal(
  activeComposition.nodes[0].opacity,
  activePresentation.layers[0].visible ? activePresentation.layers[0].opacity : 0,
);
assert.deepEqual(activeComposition.unmatchedAuthoredEntities, []);
const beforeRejectedComposition = app.snapshot();
assert.throws(
  () => app.extractActivePresentationScene(JSON.stringify([])),
  /omitted active presentation layer/,
);
assert.throws(
  () => app.extractActivePresentationScene(JSON.stringify([{
    ...activePresentationBindings[0],
    layerTransform: { translation: [99, 0, 0] },
  }])),
  /unknown field `layerTransform`/,
);
assert.deepEqual(app.snapshot(), beforeRejectedComposition);
const midTransition = app.tickPresentation(0.35);
const incumbentMidTransition = incumbent.tick(0.35);
assert.equal(midTransition.elapsed_seconds, 0.35);
assert.ok(Math.abs(midTransition.camera.camera_transition_remaining - 0.35) < 1e-12);
assertNavigationParity(midTransition, incumbentMidTransition);
assertNavigationParity(app.tickPresentation(0.35), incumbent.tick(0.35));

const invertedView = presentation.views.find(view => view.focus?.inversion_enabled === true);
const linkedCueRecord = presentation.cues.find(cue => cue.view === invertedView?.id);
const linkedCue = linkedCueRecord?.id;
assert.ok(linkedCueRecord, 'presentation fixture must retain one inverted-chart cue');
const linkedApp = app.present(4, 'jump', linkedCue);
const linkedIncumbent = incumbent.jumpToPresentationCue(linkedCue);
assert.equal(linkedApp.disposition, 'applied');
assert.deepEqual(app.snapshot().presentation.active, linkedIncumbent);
assert.equal(app.snapshot().presentation.active.cue_id, linkedCue);
const linkedTransitionSteps = Math.ceil(
  (linkedCueRecord.transition?.duration_seconds || 0) / 0.1,
) + 1;
for (let step = 0; step < linkedTransitionSteps; step++) {
  assertNavigationParity(app.tickPresentation(0.1), incumbent.tick(0.1));
}
const inverted = app.tickPresentation(0);
assert.equal(inverted.reflection, 'sphere_reflection');
assert.equal(inverted.focus.inversion_enabled, true);
assert.throws(
  () => app.present(5, 'jump', 'not-a-uuid'),
  /cue ID must be a UUID/,
);
assert.equal(
  app.snapshot().presentation.active.cue_id,
  linkedCue,
  'a malformed shadow cue must preserve the preceding reducer state',
);

// Presentation and direct navigation deliberately share these same two
// instances. This catches collisions between cue-authored and adapter-authored
// sequence numbers before the explicit re-synchronization below.
assert.equal(app.toggleInversion(), incumbent.toggleInversion());
assertNavigationParity(app.navigationSnapshot(), incumbent.snapshot());
assertNavigationParity(app.tickNavigation(0), incumbent.tick(0));

const navigationApp = app;
const navigationIncumbent = incumbent;
navigationIncumbent.synchronizeState(
  eye, forward, up, 3, target, ...projectionLens,
  focusCenter, 2, false, false, 0.5, 0.1,
);
navigationApp.synchronizeNavigation(
  eye, forward, up, 3, target, ...projectionLens,
  focusCenter, 2, false, false, 0.5, 0.1,
);
assert.equal(navigationApp.setPreset('fly'), navigationIncumbent.setPreset('fly'));
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
const reverseInterleaveApp = navigationApp.present(6, 'advance', '');
const reverseInterleaveIncumbent = navigationIncumbent.advancePresentation();
assert.deepEqual(
  navigationApp.snapshot().presentation.active,
  reverseInterleaveIncumbent,
  'direct navigation followed by presentation must preserve shared sequence order',
);
assert.equal(reverseInterleaveApp.disposition, 'applied');
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());

navigationIncumbent.synchronizeState(
  eye, forward, up, 3, target, ...projectionLens,
  focusCenter, 2, false, false, 0.5, 0.1,
);
navigationApp.synchronizeNavigation(
  eye, forward, up, 3, target, ...projectionLens,
  focusCenter, 2, false, false, 0.5, 0.1,
);
const firstNavigationSequence = navigationApp.setPreset('fly');
assert.equal(firstNavigationSequence, navigationIncumbent.setPreset('fly'));
assert.equal(firstNavigationSequence, 0n, 'synchronization resets the shared sequence authority');
assert.equal(
  navigationApp.applyFrame(
    new Float64Array([0.2, -0.1, -0.4]),
    new Float64Array([0.03, -0.02, 0.01]),
    0,
    false,
  ),
  navigationIncumbent.applyFrame(
    new Float64Array([0.2, -0.1, -0.4]),
    new Float64Array([0.03, -0.02, 0.01]),
    0,
    false,
  ),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(
  navigationApp.tickNavigation(1 / 60),
  navigationIncumbent.tick(1 / 60),
);

const transitionEye = new Float64Array([0.5, 0.25, 4]);
assert.equal(
  navigationApp.transitionCamera(
    transitionEye, forward, up, 4, target, 0.5, 'smootherstep',
  ),
  navigationIncumbent.transitionCamera(
    transitionEye, forward, up, 4, target, 0.5, 'smootherstep',
  ),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.setFreeFocusSphere(new Float64Array([0.25, 0.5, -0.25]), 1.5),
  navigationIncumbent.setFreeFocusSphere(new Float64Array([0.25, 0.5, -0.25]), 1.5),
);
assert.equal(
  navigationApp.setFocusFieldState(true, 0.35, 0.075),
  navigationIncumbent.setFocusFieldState(true, 0.35, 0.075),
);
assert.equal(
  navigationApp.setInversionEnabled(true),
  navigationIncumbent.setInversionEnabled(true),
);
assert.equal(
  navigationApp.translateFocus(new Float64Array([0.1, -0.2, 0.05])),
  navigationIncumbent.translateFocus(new Float64Array([0.1, -0.2, 0.05])),
);
assert.equal(
  navigationApp.scaleFocusLog(Math.log(1.2)),
  navigationIncumbent.scaleFocusLog(Math.log(1.2)),
);
assert.equal(
  navigationApp.toggleInversion(),
  navigationIncumbent.toggleInversion(),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(navigationApp.tickNavigation(0), navigationIncumbent.tick(0));
const acceptedFocusField = navigationApp.navigationSnapshot().focus;
assert.equal(
  navigationApp.setFocusFieldState(false, Number.NaN, 0.2),
  navigationIncumbent.setFocusFieldState(false, Number.NaN, 0.2),
);
assertNavigationParity(navigationApp.tickNavigation(0), navigationIncumbent.tick(0));
assert.deepEqual(
  navigationApp.navigationSnapshot().focus,
  acceptedFocusField,
  'invalid field geometry must also roll back its enabled-state edit',
);

const anchorEye = new Float64Array([1, 0.5, 2]);
const anchorForward = new Float64Array([0, 0, -1]);
const anchorUp = new Float64Array([0, 1, 0]);
const anchorNormal = new Float64Array([0, 1, 0]);
assert.equal(
  navigationApp.beginSurfaceAnchorTransition(
    anchorEye, anchorForward, anchorUp, 2, anchorNormal, 10, 1, 'smootherstep',
  ),
  navigationIncumbent.beginSurfaceAnchorTransition(
    anchorEye, anchorForward, anchorUp, 2, anchorNormal, 10, 1, 'smootherstep',
  ),
);
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.updateSurfaceAnchorTarget(
    new Float64Array([1.25, 0.6, 2]),
    anchorForward,
    anchorUp,
    2,
    anchorNormal,
  ),
  navigationIncumbent.updateSurfaceAnchorTarget(
    new Float64Array([1.25, 0.6, 2]),
    anchorForward,
    anchorUp,
    2,
    anchorNormal,
  ),
);
assertNavigationParity(
  navigationApp.tickNavigation(0.25),
  navigationIncumbent.tick(0.25),
);
assert.equal(
  navigationApp.cancelSurfaceAnchorTransition(),
  navigationIncumbent.cancelSurfaceAnchorTransition(),
);
assertNavigationParity(navigationApp.navigationSnapshot(), navigationIncumbent.snapshot());
assertNavigationParity(navigationApp.tickNavigation(0), navigationIncumbent.tick(0));

function browserSpaceMouseCameraFrame(normalizedAxes, sample) {
  const mapped = Array.from(mapSpaceMouseNavigationAxes(normalizedAxes, {
    mode: sample.preset,
    swapYZ: sample.swapYZ,
    invertPan: sample.invertPan,
    invertRotate: sample.invertRotate,
  }));
  const translationScale = sample.registeredLinearSpeed
    * sample.moveGain * sample.deltaSeconds;
  const rotationScale = 1.5 * sample.rotateGain * sample.deltaSeconds;
  const translation = mapped.slice(0, 3).map(axis => axis * translationScale);
  const rotation = mapped.slice(3, 6).map(axis => axis * rotationScale);
  let dollyLog = 0;
  if (sample.preset === 'object') {
    translation[2] = 0;
    dollyLog = mapped[2] * 1.5 * sample.moveGain * sample.deltaSeconds;
  }
  return {
    translation,
    rotation,
    dolly_log: dollyLog,
    horizon_locked: sample.preset === 'drone'
      || (sample.preset !== 'hyperscope' && sample.horizonLockRequested),
  };
}

function rustSpaceMouseCameraFrame(normalizedAxes, sample) {
  return mapSpaceMouseCameraFrame(
    normalizedAxes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
}

function browserPointerTurntableFrame(deltaX, deltaY, gesture, controlDistance) {
  if (gesture === 0) {
    return { pan: [0, 0], pitch: -deltaY * 0.005, yaw: deltaX * 0.005, dolly_log: 0 };
  }
  if (gesture === 1) {
    return {
      pan: [
        -deltaX * 0.003 * controlDistance,
        -deltaY * 0.003 * controlDistance,
      ],
      pitch: 0,
      yaw: 0,
      dolly_log: 0,
    };
  }
  return {
    pan: [0, 0],
    pitch: 0,
    yaw: 0,
    dolly_log: Math.log(deltaY > 0 ? 1.1 : 0.9),
  };
}

function rotateVectorAroundAxis(vector, axis, angle) {
  const length = Math.hypot(...axis);
  const unit = axis.map(value => value / length);
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const dot = vector.reduce((sum, value, index) => sum + value * unit[index], 0);
  const cross = [
    unit[1] * vector[2] - unit[2] * vector[1],
    unit[2] * vector[0] - unit[0] * vector[2],
    unit[0] * vector[1] - unit[1] * vector[0],
  ];
  return vector.map((value, index) => (
    value * cosine + cross[index] * sine + unit[index] * dot * (1 - cosine)
  ));
}

function browserPointerCameraStep(camera, frame, semanticTargetEnabled) {
  const pivot = camera.semanticTarget ?? camera.eye.map((value, index) => (
    value + camera.forward[index] * camera.controlDistance
  ));
  const worldUp = [0, 1, 0];
  const yawed = {
    right: rotateVectorAroundAxis(camera.right, worldUp, frame.yaw),
    up: rotateVectorAroundAxis(camera.up, worldUp, frame.yaw),
    forward: rotateVectorAroundAxis(camera.forward, worldUp, frame.yaw),
  };
  const basis = {
    right: rotateVectorAroundAxis(yawed.right, yawed.right, frame.pitch),
    up: rotateVectorAroundAxis(yawed.up, yawed.right, frame.pitch),
    forward: rotateVectorAroundAxis(yawed.forward, yawed.right, frame.pitch),
  };
  const translatedPivot = pivot.map((value, index) => (
    value + basis.right[index] * frame.pan[0] + basis.up[index] * frame.pan[1]
  ));
  const controlDistance = Math.min(
    100,
    Math.max(0.1, camera.controlDistance * Math.exp(frame.dolly_log)),
  );
  return {
    eye: translatedPivot.map((value, index) => (
      value - basis.forward[index] * controlDistance
    )),
    ...basis,
    controlDistance,
    semanticTarget: semanticTargetEnabled ? translatedPivot : undefined,
  };
}

function assertNumbersNear(actual, expected, tolerance = 2e-12) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `numeric drift at ${index}: ${actual[index]} vs ${expected[index]}`,
    );
  }
}

const presets = ['hyperscope', 'object', 'fly', 'drone'];
const normalizedAxes = new Float32Array([0.25, -0.5, 0.75, -1, 0.125, -0.25]);
const spaceMouseAxisVectors = [new Float32Array(6)];
for (let axis = 0; axis < 6; axis++) {
  const positive = new Float32Array(6);
  positive[axis] = 1;
  spaceMouseAxisVectors.push(positive);
  const negative = new Float32Array(6);
  negative[axis] = -1;
  spaceMouseAxisVectors.push(negative);
}
spaceMouseAxisVectors.push(normalizedAxes);

let exhaustiveMappingCases = 0;
for (const preset of presets) {
  for (const swapYZ of [false, true]) {
    for (let invertPan = 0; invertPan < 8; invertPan++) {
      for (let invertRotate = 0; invertRotate < 8; invertRotate++) {
        for (const axes of spaceMouseAxisVectors) {
          const sample = {
            preset,
            swapYZ,
            invertPan,
            invertRotate,
            deltaSeconds: 0.25,
            registeredLinearSpeed: 2,
            moveGain: 0.5,
            rotateGain: 1.25,
            horizonLockRequested: false,
          };
          assert.deepEqual(
            rustSpaceMouseCameraFrame(axes, sample),
            { preset, frame: browserSpaceMouseCameraFrame(axes, sample) },
          );
          exhaustiveMappingCases++;
        }
      }
    }
  }
}

let responsePolicyCases = 0;
for (const preset of presets) {
  for (const horizonLockRequested of [false, true]) {
    for (const deltaSeconds of [0, 0.125, 0.5]) {
      for (const registeredLinearSpeed of [0, 0.5, 4]) {
        for (const moveGain of [0, 0.25, 3]) {
          for (const rotateGain of [0, 0.5, 4]) {
            const sample = {
              preset,
              swapYZ: true,
              invertPan: 0b101,
              invertRotate: 0b010,
              deltaSeconds,
              registeredLinearSpeed,
              moveGain,
              rotateGain,
              horizonLockRequested,
            };
            assert.deepEqual(
              rustSpaceMouseCameraFrame(normalizedAxes, sample),
              { preset, frame: browserSpaceMouseCameraFrame(normalizedAxes, sample) },
            );
            responsePolicyCases++;
          }
        }
      }
    }
  }
}

const spaceMouseCases = [
  {
    preset: 'hyperscope', swapYZ: false, invertPan: 0b010, invertRotate: 0b001,
    deltaSeconds: 0.25, registeredLinearSpeed: 2, moveGain: 0.5, rotateGain: 4 / 3,
    horizonLockRequested: true,
  },
  {
    preset: 'object', swapYZ: false, invertPan: 0, invertRotate: 0,
    deltaSeconds: 0.5, registeredLinearSpeed: 4, moveGain: 0.5, rotateGain: 2 / 3,
    horizonLockRequested: true,
  },
  {
    preset: 'fly', swapYZ: true, invertPan: 0b101, invertRotate: 0b010,
    deltaSeconds: 0.125, registeredLinearSpeed: 8, moveGain: 0.75, rotateGain: 2,
    horizonLockRequested: false,
  },
  {
    preset: 'drone', swapYZ: false, invertPan: 0b111, invertRotate: 0b111,
    deltaSeconds: 0.5, registeredLinearSpeed: 0.5, moveGain: 2, rotateGain: 1,
    horizonLockRequested: false,
  },
];
const spaceMouseCameraStates = [
  { eye, forward, up, target },
  {
    eye: new Float64Array([2, -1, 4]),
    forward,
    up,
    target: new Float64Array(),
  },
  {
    eye: new Float64Array([1, 2, 3]),
    forward: new Float64Array([0, -1, 0]),
    up: new Float64Array([1, 0, 0]),
    target: new Float64Array(),
  },
  {
    eye: new Float64Array([3, 0, 0]),
    forward: new Float64Array([-1, 0, 0]),
    up: new Float64Array([0, 0, 1]),
    target,
  },
];
for (const [caseIndex, sample] of spaceMouseCases.entries()) {
  const mappedApp = new HyperscopeAppShadow();
  const semanticApp = new HyperscopeAppShadow();
  const intentApp = new HyperscopeAppShadow();
  const decomposedIntentApp = new HyperscopeAppShadow();
  const camera = spaceMouseCameraStates[caseIndex];
  for (const candidate of [mappedApp, semanticApp, intentApp, decomposedIntentApp]) {
    candidate.synchronizeNavigation(
      camera.eye, camera.forward, camera.up, 3, camera.target,
      ...projectionLens, focusCenter, 2, false, false, 0.5, 0.1,
    );
  }
  const expectedFrame = browserSpaceMouseCameraFrame(normalizedAxes, sample);
  const dispatch = mappedApp.queueSpaceMouseCamera(
    normalizedAxes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
  assert.deepEqual(dispatch.frame, expectedFrame);
  assert.equal(dispatch.preset, sample.preset);
  assert.equal(dispatch.preset_sequence, '0');
  assert.equal(dispatch.frame_sequence, '1');
  assert.equal(semanticApp.setPreset(sample.preset), 0n);
  assert.equal(
    semanticApp.applyFrame(
      new Float64Array(expectedFrame.translation),
      new Float64Array(expectedFrame.rotation),
      expectedFrame.dolly_log,
      expectedFrame.horizon_locked,
    ),
    1n,
  );
  assertNavigationParity(mappedApp.navigationSnapshot(), semanticApp.navigationSnapshot());
  assertNavigationParity(mappedApp.tickNavigation(0), semanticApp.tickNavigation(0));

  const cameraPacket = new Float64Array(17);
  intentApp.stepSpaceMouseCamera(
    normalizedAxes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
    cameraPacket,
  );
  decomposedIntentApp.setPreset(sample.preset);
  decomposedIntentApp.setSemanticTargetEnabled(sample.preset === 'object');
  decomposedIntentApp.applyFrame(
    new Float64Array(expectedFrame.translation),
    new Float64Array(expectedFrame.rotation),
    expectedFrame.dolly_log,
    expectedFrame.horizon_locked,
  );
  const expectedIntent = decomposedIntentApp.tickNavigation(0);
  const actualIntent = intentApp.navigationSnapshot();
  assertNavigationContentParity(actualIntent, expectedIntent);
  assert.equal(actualIntent.pending_actions, 0);
  assert.equal(actualIntent.last_applied_sequence, 0);
  assert.equal(actualIntent.camera.semantic_target !== undefined, sample.preset === 'object');
  assertSpaceMouseCameraPacket(cameraPacket, actualIntent);
  mappedApp.free();
  semanticApp.free();
  intentApp.free();
  decomposedIntentApp.free();
}

const traceMappedApp = new HyperscopeAppShadow();
const traceSemanticApp = new HyperscopeAppShadow();
for (const candidate of [traceMappedApp, traceSemanticApp]) {
  candidate.synchronizeNavigation(
    new Float64Array([2, -1, 4]), forward, up, 3, new Float64Array(),
    ...projectionLens, focusCenter, 2, false, false, 0.5, 0.1,
  );
}
const traceDeltas = [1 / 128, 1 / 64, 1 / 32];
const traceSpeeds = [0.5, 2, 8];
const traceMoveGains = [0.25, 1, 3];
const traceRotateGains = [0.5, 1, 2];
const traceFrames = 120;
for (let frame = 0; frame < traceFrames; frame++) {
  const sample = {
    preset: presets[frame % presets.length],
    swapYZ: (frame & 1) !== 0,
    invertPan: frame % 8,
    invertRotate: (frame * 3) % 8,
    deltaSeconds: traceDeltas[frame % traceDeltas.length],
    registeredLinearSpeed: traceSpeeds[(frame + 1) % traceSpeeds.length],
    moveGain: traceMoveGains[(frame + 2) % traceMoveGains.length],
    rotateGain: traceRotateGains[frame % traceRotateGains.length],
    horizonLockRequested: (frame & 2) !== 0,
  };
  const axes = spaceMouseAxisVectors[frame % spaceMouseAxisVectors.length];
  const expectedFrame = browserSpaceMouseCameraFrame(axes, sample);
  const dispatch = traceMappedApp.queueSpaceMouseCamera(
    axes,
    sample.preset,
    sample.swapYZ,
    sample.invertPan,
    sample.invertRotate,
    sample.deltaSeconds,
    sample.registeredLinearSpeed,
    sample.moveGain,
    sample.rotateGain,
    sample.horizonLockRequested,
  );
  assert.deepEqual(dispatch.frame, expectedFrame);
  assert.equal(dispatch.preset_sequence, String(frame * 2));
  assert.equal(dispatch.frame_sequence, String(frame * 2 + 1));
  assert.equal(traceSemanticApp.setPreset(sample.preset), BigInt(frame * 2));
  assert.equal(
    traceSemanticApp.applyFrame(
      new Float64Array(expectedFrame.translation),
      new Float64Array(expectedFrame.rotation),
      expectedFrame.dolly_log,
      expectedFrame.horizon_locked,
    ),
    BigInt(frame * 2 + 1),
  );
  assertNavigationParity(traceMappedApp.navigationSnapshot(), traceSemanticApp.navigationSnapshot());
  assertNavigationParity(
    traceMappedApp.tickNavigation(sample.deltaSeconds),
    traceSemanticApp.tickNavigation(sample.deltaSeconds),
  );
}
traceMappedApp.free();
traceSemanticApp.free();

const invalidSpaceMouseApp = new HyperscopeAppShadow();
const unchangedCameraPacket = new Float64Array(17).fill(123);
const shortCameraPacket = new Float64Array(16).fill(456);
const beforeInvalidSpaceMouseStep = invalidSpaceMouseApp.navigationSnapshot();
assert.throws(
  () => invalidSpaceMouseApp.stepSpaceMouseCamera(
    normalizedAxes, 'fly', false, 0, 0, 1, 1, 1, 1, false, shortCameraPacket,
  ),
  /exactly 17 numbers/,
);
assert.deepEqual(Array.from(shortCameraPacket), new Array(16).fill(456));
assert.throws(
  () => invalidSpaceMouseApp.stepSpaceMouseCamera(
    new Float32Array([NaN, 0, 0, 0, 0, 0]),
    'fly', false, 0, 0, 1, 1, 1, 1, false, unchangedCameraPacket,
  ),
  /remain finite/,
);
assert.deepEqual(Array.from(unchangedCameraPacket), new Array(17).fill(123));
assertNavigationParity(invalidSpaceMouseApp.navigationSnapshot(), beforeInvalidSpaceMouseStep);
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array(5), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /exactly six normalized axes/,
);
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array([NaN, 0, 0, 0, 0, 0]), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /remain finite/,
);
assert.throws(
  () => invalidSpaceMouseApp.queueSpaceMouseCamera(
    new Float32Array([1.01, 0, 0, 0, 0, 0]), 'fly', false, 0, 0, 1, 1, 1, 1, false,
  ),
  /within \[-1, 1\]/,
);
for (const response of [
  [-1, 1, 1, 1],
  [1, -1, 1, 1],
  [1, 1, -1, 1],
  [1, 1, 1, -1],
]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      normalizedAxes, 'fly', false, 0, 0, ...response, false,
    ),
    /must be nonnegative/,
  );
}
for (const invalidMask of [8, 256, 263, 1.5, NaN]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      normalizedAxes, 'object', false, invalidMask, 0, 1, 1, 1, 1, false,
    ),
    /finite integers from 0 through 7/,
  );
}
const overflowSpaceMouseAxes = new Float32Array([1, 0, 0, 0, 0, 0]);
for (const [preset, deltaSeconds, registeredLinearSpeed, moveGain, rotateGain] of [
  ['fly', 1, Number.MAX_VALUE, 2, 1],
  ['fly', 1, 1, 1, Number.MAX_VALUE],
  ['object', 1, 0, Number.MAX_VALUE, 1],
]) {
  assert.throws(
    () => invalidSpaceMouseApp.queueSpaceMouseCamera(
      overflowSpaceMouseAxes, preset, false, 0, 0,
      deltaSeconds, registeredLinearSpeed, moveGain, rotateGain, false,
    ),
    /remain finite/,
  );
}
assert.equal(invalidSpaceMouseApp.navigationSnapshot().pending_actions, 0);
assert.equal(invalidSpaceMouseApp.navigationSnapshot().preset, 'hyperscope');
invalidSpaceMouseApp.free();

let pointerMappingCases = 0;
for (const gesture of [0, 1, 2]) {
  for (const deltaX of [-40, 0, 17.5]) {
    for (const deltaY of [-23.25, 0, 31]) {
      for (const controlDistance of [0.1, 3, 100]) {
        const actual = mapPointerTurntableFrame(
          deltaX,
          deltaY,
          gesture,
          controlDistance,
        );
        const expected = browserPointerTurntableFrame(
          deltaX,
          deltaY,
          gesture,
          controlDistance,
        );
        assertNumbersNear(actual.pan, expected.pan);
        assertNumbersNear(
          [actual.pitch, actual.yaw, actual.dolly_log],
          [expected.pitch, expected.yaw, expected.dolly_log],
        );
        pointerMappingCases++;
      }
    }
  }
}

const pointerCameraCases = [
  { deltaX: 19, deltaY: -11, gesture: 0, semanticTargetEnabled: false },
  { deltaX: -7, deltaY: 13, gesture: 1, semanticTargetEnabled: false },
  { deltaX: 0, deltaY: 1, gesture: 2, semanticTargetEnabled: false },
  { deltaX: -31, deltaY: 5, gesture: 0, semanticTargetEnabled: true },
  { deltaX: 9, deltaY: -15, gesture: 1, semanticTargetEnabled: true },
  { deltaX: 0, deltaY: -1, gesture: 2, semanticTargetEnabled: true },
];
const pointerCameraApp = new HyperscopeAppShadow();
const pointerEye = new Float64Array([2, -1, 4]);
const pointerForward = new Float64Array([0.36, -0.48, -0.8]);
const pointerUp = new Float64Array([0.8, 0.6, 0]);
pointerCameraApp.synchronizeNavigation(
  pointerEye,
  pointerForward,
  pointerUp,
  4,
  new Float64Array(),
  ...projectionLens,
  focusCenter,
  2,
  false,
  false,
  0.5,
  0.1,
);
let pointerOracle = {
  eye: Array.from(pointerEye),
  right: [0.48, -0.64, 0.6],
  up: Array.from(pointerUp),
  forward: Array.from(pointerForward),
  controlDistance: 4,
  semanticTarget: undefined,
};
for (const sample of pointerCameraCases) {
  const expectedFrame = browserPointerTurntableFrame(
    sample.deltaX,
    sample.deltaY,
    sample.gesture,
    pointerOracle.controlDistance,
  );
  pointerOracle = browserPointerCameraStep(
    pointerOracle,
    expectedFrame,
    sample.semanticTargetEnabled,
  );
  const packet = new Float64Array(17);
  pointerCameraApp.stepPointerCamera(
    sample.deltaX,
    sample.deltaY,
    sample.gesture,
    sample.semanticTargetEnabled,
    packet,
  );
  const snapshot = pointerCameraApp.navigationSnapshot();
  assertSpaceMouseCameraPacket(packet, snapshot);
  assertNumbersNear(snapshot.camera.eye, pointerOracle.eye);
  assertNumbersNear(snapshot.camera.right, pointerOracle.right);
  assertNumbersNear(snapshot.camera.up, pointerOracle.up);
  assertNumbersNear(snapshot.camera.forward, pointerOracle.forward);
  assert.ok(Math.abs(
    snapshot.camera.control_distance - pointerOracle.controlDistance,
  ) <= 2e-12);
  if (pointerOracle.semanticTarget === undefined) {
    assert.equal(snapshot.camera.semantic_target, undefined);
  } else {
    assertNumbersNear(snapshot.camera.semantic_target, pointerOracle.semanticTarget);
  }
  assert.equal(snapshot.pending_actions, 0);
}
pointerCameraApp.free();

const invalidPointerApp = new HyperscopeAppShadow();
const pointerPacket = new Float64Array(17).fill(789);
const invalidPointerBefore = invalidPointerApp.navigationSnapshot();
assert.throws(
  () => invalidPointerApp.stepPointerCamera(NaN, 0, 0, false, pointerPacket),
  /finite/,
);
assert.throws(
  () => invalidPointerApp.stepPointerCamera(0, 0, 3, false, pointerPacket),
  /pointer gesture/,
);
assert.throws(
  () => invalidPointerApp.stepPointerCamera(
    0,
    0,
    0,
    false,
    new Float64Array(16),
  ),
  /exactly 17 numbers/,
);
assert.deepEqual(Array.from(pointerPacket), new Array(17).fill(789));
assertNavigationParity(invalidPointerApp.navigationSnapshot(), invalidPointerBefore);
invalidPointerApp.free();

const finalFrameTime = app.navigationSnapshot().elapsed_seconds + 0.1;
app.advanceFrame(finalFrameTime, 0.1);
assert.throws(
  () => app.requestAsset(
    3,
    finalFrameTime + 1,
    'e0000000-0000-4000-8000-000000000003',
    asset,
    'horse.glb',
    'model/gltf-binary',
  ),
  /effect-producing input cannot be scheduled/,
);

const finalSnapshot = app.snapshot();
incumbent.free();
app.free();
console.log(JSON.stringify({
  requested: requested.effects.length,
  replacementEffects: replaced.effects.map(effect => effect.type),
  staleDisposition: stale.disposition,
  readyBytes: ready.assets[0].status.byte_length,
  sessionSelectionNodes: sessionNodeIdentities.length,
  authoredProjectionRevision: finalSnapshot.authoredProjectionRevision,
  authoredAssets: finalSnapshot.authoredAssets.length,
  authoredEntities: finalSnapshot.authoredEntities.length,
  packedSceneNodes: extractedScene.nodes.length,
  activePresentationNodes: activeComposition.nodes.length,
  peerIngress: {
    authored: peerApplied.disposition,
    duplicate: peerDuplicate.disposition,
    stale: peerStale.disposition,
    corrected: peerCorrected.disposition,
    echo: peerEcho.disposition,
    presence: presenceApplied.disposition,
    expiredPeers: expiredPresence.peers.length,
  },
  diagnostics: ready.diagnostics.map(diagnostic => diagnostic.code),
  presentationCue: finalSnapshot.presentation.active.cue_id,
  animationPlaying: finalSnapshot.animationPlaying,
  navigationBoundaryParity: true,
  spaceMouseInputCases: {
    exhaustiveMapping: exhaustiveMappingCases,
    responsePolicy: responsePolicyCases,
    queuedCameraStates: spaceMouseCases.length,
    deterministicTraceFrames: traceFrames,
  },
  pointerInputCases: {
    mapping: pointerMappingCases,
    cameraStates: pointerCameraCases.length,
  },
}));
