import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BrowserLocalPeerSession,
  BrowserLocalPeerSessionError,
} from '../local_peer_session.mjs';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function fixture() {
  const events = [];
  const peer = {
    free() { events.push('peer.free'); },
  };
  const app = {
    receiveLocalPeerEnvelope() {},
    frameElapsedSeconds() { return 12.5; },
    async openDurableAuthoredPeerWithRole(projectId, role) {
      events.push(['app.open', projectId, role]);
      return peer;
    },
  };
  let options = null;
  const relay = {
    start() { events.push('relay.start'); },
    async stop() { events.push('relay.stop'); },
    snapshot() { return { state: 'connecting' }; },
    sendAppliedAuthoredEnvelope(value) { return `authored:${value}`; },
    sendPresenceEnvelope(value) { return `presence:${value}`; },
  };
  const session = new BrowserLocalPeerSession({
    app,
    relayFactory(value) {
      options = value;
      return relay;
    },
  });
  return { app, events, peer, relay, session, options: () => options };
}

test('legacy mode retains the direct rollback lane without allocating durability', async () => {
  const value = fixture();
  await value.session.connect({
    baseUrl: 'http://127.0.0.1:42117',
    token: 'runtime only',
    mode: 'legacy',
  });
  assert.equal(value.options().durablePeer, null);
  assert.equal(value.options().authoredProposalPolicy, 'legacy');
  assert.deepEqual(value.events, ['relay.start']);
  assert.equal(value.session.sendAppliedAuthoredEnvelope('x'), 'authored:x');
  await value.session.disconnect();
  assert.deepEqual(value.events, ['relay.start', 'relay.stop']);
});

test('durable replica consumes authorized records but cannot promote proposals', async () => {
  const value = fixture();
  await value.session.connect({
    baseUrl: 'http://127.0.0.1:42117',
    token: 'runtime only',
    mode: 'durable',
    projectId: ' 00000000-0000-4000-8000-000000000042 ',
  });
  assert.deepEqual(value.events, [
    ['app.open', '00000000-0000-4000-8000-000000000042', 'replica'],
    'relay.start',
  ]);
  assert.equal(value.options().durablePeer, value.peer);
  assert.equal(value.options().authoredProposalPolicy, 'ignore');
  await value.session.disconnect();
  assert.deepEqual(value.events.slice(-2), ['relay.stop', 'peer.free']);
});

test('admission authority must be explicit and is handed to both Rust and carrier', async () => {
  const value = fixture();
  await value.session.connect({
    baseUrl: 'http://127.0.0.1:42117',
    token: 'runtime only',
    mode: 'durable',
    projectId: '00000000-0000-4000-8000-000000000042',
    proposalRole: 'admission_authority',
  });
  assert.deepEqual(value.events[0], [
    'app.open',
    '00000000-0000-4000-8000-000000000042',
    'admission_authority',
  ]);
  assert.equal(value.options().authoredProposalPolicy, 'admit');
});

test('relay construction failure releases the opened durable writer lease', async () => {
  const value = fixture();
  value.session.relayFactory = () => { throw new Error('relay failed'); };
  await assert.rejects(
    value.session.connect({
      baseUrl: 'http://127.0.0.1:42117',
      token: 'runtime only',
      mode: 'durable',
      projectId: '00000000-0000-4000-8000-000000000042',
    }),
    /relay failed/,
  );
  assert.deepEqual(value.events.slice(-1), ['peer.free']);
  assert.equal(value.session.isConnected(), false);
});

test('a stale asynchronous open cannot resurrect a disconnected session', async () => {
  const value = fixture();
  const opening = deferred();
  value.app.openDurableAuthoredPeerWithRole = () => opening.promise;
  const connect = value.session.connect({
    baseUrl: 'http://127.0.0.1:42117',
    token: 'runtime only',
    mode: 'durable',
    projectId: '00000000-0000-4000-8000-000000000042',
  });
  assert.equal(value.session.isOpening(), true);
  await value.session.disconnect();
  opening.resolve(value.peer);
  await assert.rejects(connect, /opening was cancelled/);
  assert.deepEqual(value.events, ['peer.free']);
  assert.equal(value.session.isConnected(), false);
});

test('durable mode fails closed when the optional WASM feature is absent', async () => {
  const app = { receiveLocalPeerEnvelope() {}, frameElapsedSeconds() { return 0; } };
  const session = new BrowserLocalPeerSession({ app });
  await assert.rejects(
    session.connect({ mode: 'durable', projectId: 'project', token: 'runtime only' }),
    BrowserLocalPeerSessionError,
  );
});
