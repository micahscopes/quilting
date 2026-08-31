import assert from 'node:assert/strict';
import test from 'node:test';
import { BrowserAssetEffectHost } from '../asset_effect_host.mjs';

const fetchJob = (requestId, assetId, uri) => ({
  requestId,
  assetId,
  uri,
});

const cancellationJob = (requestId, assetId) => ({
  requestId,
  assetId,
});

const installJob = (requestId, assetId) => ({
  requestId,
  assetId,
});

function authorizeInstall(host, token) {
  return host.beginInstall(token, installJob(token.requestId, token.assetId));
}

function begin(host, {
  requestId,
  assetId,
  uri,
  scope = 'asset',
  fetch = fetchJob(requestId, assetId, uri),
  loadCancellations = [],
  installCancellations = [],
}) {
  return host.begin({
    requestId,
    assetId,
    uri,
    source: 'test',
    scope,
    fetch,
    loadCancellations,
    installCancellations,
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
  assert.equal(first.disposition, null);
  assert.equal(host.mayProcess(first), true);
  assert.equal(host.mayInstall(first), false);
  host.recordCompletion(first, 'applied');
  assert.equal(host.mayInstall(first), false);
  authorizeInstall(host, first);
  assert.equal(host.mayInstall(first), true);

  const second = begin(host, {
    requestId: 'request-2',
    assetId: 'chess',
    uri: 'chess.glb',
    scope: 'primary_scene',
    fetch: fetchJob('request-2', 'chess', 'chess.glb'),
    installCancellations: [cancellationJob('request-1', 'horse')],
  }).token;
  assert.equal(first.signal.aborted, true);
  assert.equal(host.mayProcess(first), false);
  assert.equal(host.mayInstall(first), false);
  assert.equal(host.mayProcess(second), true);
  assert.equal(host.mayInstall(second), false);
  host.recordCompletion(second, 'applied');
  authorizeInstall(host, second);
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
    fetch: fetchJob('request-2', 'chess', 'chess.glb'),
    loadCancellations: [cancellationJob('request-1', 'horse')],
  }).token;
  host.recordCompletion(first, 'ignored_stale');
  host.recordCompletion(second, 'applied');
  assert.equal(first.signal.aborted, false);
  assert.equal(host.mayInstall(first), true);
  assert.equal(host.mayInstall(second), true);
});

test('rust mode validates a typed request before superseding the current job', () => {
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
      fetch: fetchJob('wrong-request', 'chess', 'chess.glb'),
    }),
    /one matching fetch job/,
  );
  assert.equal(first.signal.aborted, false);
  assert.equal(host.primary, first);
});

test('rust mode requires the decoded completion to authorize primary installation', () => {
  const host = new BrowserAssetEffectHost('rust');
  const token = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  host.recordCompletion(token, 'applied');
  assert.equal(host.mayInstall(token), false);
  assert.throws(
    () => host.beginInstall(token, null),
    /primary install must be a job object/,
  );
  assert.equal(host.mayInstall(token), false);
  authorizeInstall(host, token);
  assert.equal(host.mayInstall(token), true);
  assert.equal(host.recordInstallCompletion(token, 'applied'), 'applied');
  assert.equal(token.installDisposition, 'applied');
});

test('rust mode rejects a replacement that omits the active install cancellation', () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  host.recordCompletion(first, 'applied');
  authorizeInstall(host, first);
  assert.throws(
    () => begin(host, {
      requestId: 'request-2',
      assetId: 'chess',
      uri: 'chess.glb',
      scope: 'primary_scene',
      fetch: fetchJob('request-2', 'chess', 'chess.glb'),
    }),
    /omitted install cancellation/,
  );
  assert.equal(first.signal.aborted, false);
  assert.equal(host.primary, first);
});

test('Rust cancellation jobs abort independent same-asset jobs', () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb',
  }).token;
  begin(host, {
    requestId: 'request-2',
    assetId: 'horse',
    uri: 'horse.glb',
    fetch: fetchJob('request-2', 'horse', 'horse.glb'),
    loadCancellations: [cancellationJob('request-1', 'horse')],
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
  assert.equal(host.mayProcess(token), true);
  assert.equal(host.mayInstall(token), true);
});

test('rust mode serializes primary installations and skips a queued stale job', async () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  host.recordCompletion(first, 'applied');
  authorizeInstall(host, first);
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
    fetch: fetchJob('request-2', 'chess', 'chess.glb'),
    installCancellations: [cancellationJob('request-1', 'horse')],
  }).token;
  host.recordCompletion(second, 'applied');
  authorizeInstall(host, second);
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

test('rust mode serializes decode work before completion and skips stale jobs', async () => {
  const host = new BrowserAssetEffectHost('rust');
  const first = begin(host, {
    requestId: 'request-1', assetId: 'horse', uri: 'horse.glb', scope: 'primary_scene',
  }).token;
  let releaseFirst;
  const firstBlocked = new Promise(resolve => { releaseFirst = resolve; });
  const order = [];
  const firstProcess = host.runProcess(first, async () => {
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
    fetch: fetchJob('request-2', 'chess', 'chess.glb'),
    loadCancellations: [cancellationJob('request-1', 'horse')],
  }).token;
  const secondProcess = host.runProcess(second, async () => {
    order.push('second');
    return true;
  });
  const staleQueued = host.runProcess(first, async () => {
    order.push('stale');
    return true;
  });

  releaseFirst();
  assert.equal(await firstProcess, true);
  assert.equal(await secondProcess, true);
  assert.equal(await staleQueued, false);
  assert.deepEqual(order, ['first-start', 'first-end', 'second']);
});
