import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BrowserLocalPeerRelay,
  BrowserLocalPeerRelayError,
} from '../local_peer_browser.mjs';


const authoredEnvelope = sequence => `{
  "header": {
    "version": {"major": 0, "minor": 1},
    "message_id": "10000000-0000-4000-8000-000000000001",
    "sender": "10000000-0000-4000-8000-000000000002",
    "sequence": ${sequence}
  },
  "command": {
    "type": "set_entity_transform",
    "entity": "10000000-0000-4000-8000-000000000003",
    "transform": {
      "translation": [1, 2, 3],
      "rotation_wxyz": [1, 0, 0, 0],
      "scale": [1, 1, 1]
    }
  }
}`;

const presenceEnvelope = `{
  "header": {
    "version": {"major": 0, "minor": 1},
    "message_id": "20000000-0000-4000-8000-000000000001",
    "sender": "20000000-0000-4000-8000-000000000002",
    "sequence": 1
  },
  "presence": {"ttl_millis": 1500, "selection": []}
}`;

function appOracle({ reject = null } = {}) {
  return {
    received: [],
    recorded: [],
    recordedPresence: [],
    receiveLocalPeerEnvelope(atSeconds, frameJson) {
      if (reject?.(frameJson)) throw new Error('deliberate Rust rejection');
      this.received.push({ atSeconds, frameJson });
      return {
        lane: frameJson.includes('"lane":"presence"') ? 'presence' : 'authored',
        disposition: 'applied',
      };
    },
    recordLocalAuthoredEnvelope(envelopeJson) {
      this.recorded.push(envelopeJson);
    },
    recordLocalPresenceEnvelope(envelopeJson) {
      if (reject?.(envelopeJson)) throw new Error('deliberate Rust rejection');
      this.recordedPresence.push(envelopeJson);
    },
  };
}

function batch({
  generation = 'generation-a',
  requestedAfter = '0',
  resumeAfter = '0',
  latestCursor = '0',
  oldestCursor = null,
  gap = false,
  hasMore = false,
  frames = [],
} = {}) {
  return {
    generation,
    requestedAfter,
    resumeAfter,
    oldestCursor,
    latestCursor,
    gap,
    hasMore,
    frames,
  };
}

function transport(app = appOracle(), options = {}) {
  return new BrowserLocalPeerRelay({
    baseUrl: 'http://127.0.0.1:42117',
    token: 'test-token',
    app,
    fetchImpl: async () => { throw new Error('unexpected fetch'); },
    nowSeconds: () => 12.5,
    ...options,
  });
}

function jsonResponse(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  return {
    ok: true,
    status: 200,
    async arrayBuffer() {
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    },
    async text() {
      return new TextDecoder().decode(bytes);
    },
  };
}

test('browser carrier rejects ambiguous configuration and never accepts a token in the URL', () => {
  const app = appOracle();
  for (const baseUrl of [
    'ws://127.0.0.1:42117',
    'http://user@127.0.0.1:42117',
    'http://127.0.0.1:42117/path',
    'http://127.0.0.1:42117?token=secret',
  ]) {
    assert.throws(
      () => new BrowserLocalPeerRelay({ baseUrl, token: 'token', app }),
      BrowserLocalPeerRelayError,
    );
  }
  assert.throws(
    () => new BrowserLocalPeerRelay({
      baseUrl: 'http://127.0.0.1:42117',
      token: 'not a token',
      app,
    }),
    /token/,
  );
});

test('exact u64 application JSON reaches Rust without JavaScript numeric parsing', () => {
  const app = appOracle();
  const relay = transport(app);
  const frameJson = `{"lane":"authored","envelope":${authoredEnvelope(
    '18446744073709551615',
  )}}`;
  const hasMore = relay.acceptBatch(batch({
    latestCursor: '1',
    oldestCursor: '1',
    frames: [{ cursor: '1', frameJson }],
  }), 0n);
  assert.equal(hasMore, false);
  assert.equal(app.received.length, 1);
  assert.match(app.received[0].frameJson, /18446744073709551615/);
  assert.equal(relay.snapshot().cursor, '1');
  assert.equal(relay.snapshot().appliedFrames, 1);
});

test('Rust semantic rejection cannot advance the delivery cursor', () => {
  const app = appOracle({ reject: frame => frame.includes('bad_semantics') });
  const relay = transport(app);
  const frameJson = '{"lane":"bad_semantics","envelope":{}}';
  assert.throws(
    () => relay.acceptBatch(batch({
      latestCursor: '1',
      oldestCursor: '1',
      frames: [{ cursor: '1', frameJson }],
    }), 0n),
    /Rust rejected relay cursor 1/,
  );
  assert.equal(relay.snapshot().cursor, '0');
});

test('generation changes discard the stale response and repoll from zero', () => {
  const app = appOracle();
  const relay = transport(app);
  assert.equal(relay.acceptBatch(batch(), 0n), false);
  const frameJson = `{"lane":"authored","envelope":${authoredEnvelope('1')}}`;
  const changed = batch({
    generation: 'generation-b',
    latestCursor: '1',
    oldestCursor: '1',
    frames: [{ cursor: '1', frameJson }],
  });
  assert.equal(relay.acceptBatch(changed, 0n), true);
  assert.equal(app.received.length, 0);
  assert.deepEqual(
    {
      generation: relay.snapshot().generation,
      cursor: relay.snapshot().cursor,
      restarts: relay.snapshot().restarts,
      gaps: relay.snapshot().gaps,
    },
    { generation: 'generation-b', cursor: '0', restarts: 1, gaps: 1 },
  );
  assert.equal(relay.acceptBatch(changed, 0n), false);
  assert.equal(app.received.length, 1);
});

test('bounded-history gaps surface degraded state while applying the retained suffix', () => {
  const relay = transport();
  const first = `{"lane":"authored","envelope":${authoredEnvelope('1')}}`;
  const second = `{"lane":"presence","envelope":${presenceEnvelope}}`;
  relay.acceptBatch(batch({
    resumeAfter: '1',
    latestCursor: '3',
    oldestCursor: '2',
    gap: true,
    frames: [
      { cursor: '2', frameJson: first },
      { cursor: '3', frameJson: second },
    ],
  }), 0n);
  assert.equal(relay.snapshot().state, 'stopped');
  assert.equal(relay.degraded, true);
  assert.equal(relay.snapshot().gaps, 1);
  assert.equal(relay.snapshot().cursor, '3');
});

test('failed posts retain strict authored order through retry', async () => {
  const app = appOracle();
  const relay = transport(app);
  const calls = [];
  let failFirst = true;
  relay.requestJson = async (method, path, body) => {
    assert.deepEqual([method, path], ['POST', '/v1/frame']);
    calls.push(body);
    if (failFirst) {
      failFirst = false;
      throw new Error('deliberate failure');
    }
    return { generation: 'generation-a', cursor: String(calls.length) };
  };
  const first = relay.sendAppliedAuthoredEnvelope(authoredEnvelope('1'));
  const secondEnvelope = authoredEnvelope('2').replace(
    '10000000-0000-4000-8000-000000000001',
    '10000000-0000-4000-8000-000000000004',
  );
  const second = relay.sendAppliedAuthoredEnvelope(secondEnvelope);
  await assert.rejects(relay.flushOutbound(), /deliberate failure/);
  await relay.flushOutbound();
  await Promise.all([first, second]);
  assert.equal(calls.length, 3);
  assert.equal(calls[0], calls[1]);
  assert.notEqual(calls[1], calls[2]);
  assert.deepEqual(app.recorded, [authoredEnvelope('1'), secondEnvelope]);
});

test('local presence is validated by Rust before it enters the outbound queue', async () => {
  const app = appOracle();
  const relay = transport(app);
  relay.requestJson = async () => ({ generation: 'generation-a', cursor: '1' });
  const completion = relay.sendPresenceEnvelope(presenceEnvelope);
  assert.equal(app.received.length, 0);
  assert.deepEqual(app.recordedPresence, [presenceEnvelope]);
  await relay.flushOutbound();
  await completion;
});

test('start and stop abort the polling loop without persisting credentials', async () => {
  const app = appOracle();
  const relay = transport(app, {
    pollIntervalMs: 5,
    fetchImpl: async url => {
      const requestedAfter = new URL(url).searchParams.get('after');
      return jsonResponse(batch({ requestedAfter }));
    },
  });
  relay.start();
  await new Promise(resolve => setTimeout(resolve, 15));
  await relay.stop();
  assert.equal(relay.snapshot().state, 'stopped');
  assert.equal('token' in relay.snapshot(), false);
});
