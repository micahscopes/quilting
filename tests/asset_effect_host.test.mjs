import assert from 'node:assert/strict';
import test from 'node:test';
import { BrowserAssetEffectHost } from '../asset_effect_host.mjs';

const fetchEffect = (requestId, assetId, uri) => ({
  type: 'fetch_asset',
  request_id: requestId,
  asset_id: assetId,
  uri,
});

const cancelEffect = (requestId, assetId) => ({
  type: 'cancel_asset_load',
  request_id: requestId,
  asset_id: assetId,
});

function begin(host, {
  requestId,
  assetId,
  uri,
  scope = 'asset',
  effects = [fetchEffect(requestId, assetId, uri)],
}) {
  return host.begin({
    requestId,
    assetId,
    uri,
    source: 'test',
    scope,
    commit: { effects },
  });
}

test('the platform host requires an explicit authority policy', () => {
  assert.throws(
    () => new BrowserAssetEffectHost(),
    /unsupported asset implementation/,
  );
});

test('rust mode aborts and fences a superseded primary scene', () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1',
    assetId: 'horse',
    uri: 'horse.glb',
    scope: 'primary_scene',
  }).token;
  host.recordCompletion(first, 'applied');
  assert.equal(host.mayInstall(first), true);

  const second = begin(host, {
    requestId: 'request-2',
    assetId: 'chess',
    uri: 'chess.glb',
    scope: 'primary_scene',
    effects: [
      cancelEffect('request-1', 'horse'),
      fetchEffect('request-2', 'chess', 'chess.glb'),
    ],
  }).token;
  assert.equal(first.signal.aborted, true);
  assert.equal(host.mayInstall(first), false);
  host.recordCompletion(second, 'applied');
  assert.equal(host.mayInstall(second), true);
});

test('shadow mode observes cancellation without changing incumbent behavior', () => {
  const host = new BrowserAssetEffectHost('shadow');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  const second = begin(host, {
    requestId: 'request-2',
    assetId: 'chess',
    uri: 'chess.glb',
    scope: 'primary_scene',
    effects: [
      cancelEffect('request-1', 'horse'),
      fetchEffect('request-2', 'chess', 'chess.glb'),
    ],
  }).token;
  host.recordCompletion(first, 'ignored_stale');
  host.recordCompletion(second, 'applied');
  assert.equal(first.signal.aborted, false);
  assert.equal(host.mayInstall(first), true);
  assert.equal(host.mayInstall(second), true);
});

test('rust mode validates a full commit before superseding the current job', () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  assert.throws(
    () => begin(host, {
      requestId: 'request-2',
      assetId: 'chess',
      uri: 'chess.glb',
      scope: 'primary_scene',
      effects: [fetchEffect('wrong-request', 'chess', 'chess.glb')],
    }),
    /exactly one matching fetch effect/,
  );
  assert.equal(first.signal.aborted, false);
  assert.equal(host.primary, first);
});

test('Rust cancellation effects abort independent same-asset jobs', () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb',
  }).token;
  begin(host, {
    requestId: 'request-2',
    assetId: 'horse',
    uri: 'horse.glb',
    effects: [
      cancelEffect('request-1', 'horse'),
      fetchEffect('request-2', 'horse', 'horse.glb'),
    ],
  });
  assert.equal(first.signal.aborted, true);
});

test('js mode preserves incumbent logical URIs without requiring Rust effects', () => {
  const host = new BrowserAssetEffectHost('js');
  const { token, mismatches } = host.begin({
    requestId: 'request-1',
    assetId: 'horse',
    uri: 'horse.glb',
    source: 'test',
    scope: 'primary_scene',
  });
  assert.equal(token.uri, 'horse.glb');
  assert.deepEqual(mismatches, []);
  assert.equal(host.mayInstall(token), true);
});

test('rust mode serializes primary installations and skips a queued stale job', async () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  host.recordCompletion(first, 'applied');
  let releaseFirst;
  const firstBlocked = new Promise(resolve => { releaseFirst = resolve; });
  const order = [];
  const firstInstall = host.runInstall(first, async () => {
    order.push('first-start');
    await firstBlocked;
    order.push('first-end');
    return true;
  });
  await Promise.resolve();

  const second = begin(host, {
    requestId: 'request-2',
    assetId: 'chess',
    uri: 'chess.glb',
    scope: 'primary_scene',
    effects: [
      cancelEffect('request-1', 'horse'),
      fetchEffect('request-2', 'chess', 'chess.glb'),
    ],
  }).token;
  host.recordCompletion(second, 'applied');
  const secondInstall = host.runInstall(second, async () => {
    order.push('second');
    return true;
  });
  releaseFirst();
  assert.equal(await firstInstall, true);
  assert.equal(await secondInstall, true);
  assert.deepEqual(order, ['first-start', 'first-end', 'second']);

  const staleQueued = host.runInstall(first, async () => {
    order.push('stale');
    return true;
  });
  assert.equal(await staleQueued, false);
  assert.deepEqual(order, ['first-start', 'first-end', 'second']);
});
