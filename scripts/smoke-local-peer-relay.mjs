import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const repository = fileURLToPath(new URL('..', import.meta.url));
const token = 'smoke_token';
const origin = 'http://localhost:9999';
const build = spawnSync(
  'cargo',
  [
    'build',
    '-p', 'hyperscope-web',
    '--features', 'local-peer-relay',
    '--bin', 'hyperscope-local-peer-relay',
  ],
  { cwd: repository, encoding: 'utf8' },
);
if (build.status !== 0) {
  throw new Error(`relay build failed\n${build.stdout}\n${build.stderr}`);
}

const relay = spawn(
  `${repository}/target/debug/hyperscope-local-peer-relay`,
  ['--bind', '127.0.0.1:0', '--token', token, '--origin', origin],
  { cwd: repository, stdio: ['ignore', 'pipe', 'pipe'] },
);
let stdout = '';
let stderr = '';
relay.stdout.setEncoding('utf8');
relay.stderr.setEncoding('utf8');
relay.stdout.on('data', chunk => { stdout += chunk; });
relay.stderr.on('data', chunk => { stderr += chunk; });

async function waitForAddress() {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const match = stdout.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
    if (match) return match[1];
    if (relay.exitCode !== null) {
      throw new Error(`relay exited before listening\n${stdout}\n${stderr}`);
    }
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error(`timed out waiting for relay\n${stdout}\n${stderr}`);
}

async function stopRelay() {
  if (relay.exitCode !== null) return;
  relay.kill('SIGTERM');
  await Promise.race([
    new Promise(resolve => relay.once('close', resolve)),
    new Promise(resolve => setTimeout(resolve, 2_000)),
  ]);
  if (relay.exitCode === null) relay.kill('SIGKILL');
}

const headers = {
  Authorization: `Bearer ${token}`,
  Origin: origin,
};
let summary;
try {
  const base = await waitForAddress();
  assert.equal((await fetch(`${base}/v1/health`)).status, 401);
  assert.equal((await fetch(`${base}/v1/health`, {
    headers: { Authorization: `Bearer ${token}`, Origin: 'http://evil.invalid' },
  })).status, 403);

  const preflight = await fetch(`${base}/v1/frame`, {
    method: 'OPTIONS',
    headers: {
      Origin: origin,
      'Access-Control-Request-Method': 'POST',
      'Access-Control-Request-Headers': 'authorization,content-type',
    },
  });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('access-control-allow-origin'), origin);
  assert.match(preflight.headers.get('access-control-allow-headers'), /Authorization/);

  const healthResponse = await fetch(`${base}/v1/health`, { headers });
  assert.equal(healthResponse.status, 200);
  const health = await healthResponse.json();
  assert.equal(health.durable, false);
  assert.equal(health.latestCursor, '0');
  assert.equal(health.retainedFrames, 0);

  const envelope = JSON.parse(readFileSync(
    `${repository}/crates/hyperscape-protocol/fixtures/authored-set-transform-v0.1.json`,
    'utf8',
  ));
  const frame = { lane: 'authored', envelope };
  const frameJson = JSON.stringify(frame).replace(
    /}$/,
    ',"futureExact":18446744073709551615}',
  );
  const postResponse = await fetch(`${base}/v1/frame`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: frameJson,
  });
  assert.equal(postResponse.status, 202);
  const posted = await postResponse.json();
  assert.equal(posted.generation, health.generation);
  assert.equal(posted.cursor, '1');

  const invalidResponse = await fetch(`${base}/v1/frame`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: '{',
  });
  assert.equal(invalidResponse.status, 400);

  const pollResponse = await fetch(`${base}/v1/frames?after=0&limit=10`, { headers });
  assert.equal(pollResponse.status, 200);
  const batch = await pollResponse.json();
  assert.equal(batch.generation, health.generation);
  assert.equal(batch.gap, false);
  assert.equal(batch.hasMore, false);
  assert.equal(batch.latestCursor, '1');
  assert.equal(batch.frames[0].cursor, '1');
  const decodedFrame = JSON.parse(batch.frames[0].frameJson);
  assert.deepEqual(decodedFrame.envelope, envelope);
  assert.equal(decodedFrame.lane, frame.lane);
  assert.match(batch.frames[0].frameJson, /18446744073709551615/);

  const futureCursorResponse = await fetch(`${base}/v1/frames?after=99&limit=10`, { headers });
  assert.equal(futureCursorResponse.status, 200);
  const futureCursor = await futureCursorResponse.json();
  assert.equal(futureCursor.gap, true);
  assert.equal(futureCursor.resumeAfter, '0');
  assert.deepEqual(futureCursor.frames, []);

  summary = {
    authenticated: healthResponse.status,
    rejectedUnauthorized: 401,
    rejectedOrigin: 403,
    corsPreflight: preflight.status,
    acceptedCursor: posted.cursor,
    retainedFrames: batch.frames.length,
    futureCursorGap: futureCursor.gap,
    durable: health.durable,
  };
} finally {
  await stopRelay();
}

console.log(JSON.stringify(summary));
