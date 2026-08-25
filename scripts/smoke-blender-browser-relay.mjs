import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { BrowserLocalPeerRelay } from '../local_peer_browser.mjs';


const repository = fileURLToPath(new URL('..', import.meta.url));
const token = 'blender_browser_smoke_token';
const origin = 'http://localhost:9999';
const entity = '73000000-0000-4000-8000-000000000001';
const peer = '73000000-0000-4000-8000-000000000002';
const blenderConfig = mkdtempSync(join(tmpdir(), 'hyperscape-blender-relay-smoke-'));
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
let relayStdout = '';
let relayStderr = '';
relay.stdout.setEncoding('utf8');
relay.stderr.setEncoding('utf8');
relay.stdout.on('data', chunk => { relayStdout += chunk; });
relay.stderr.on('data', chunk => { relayStderr += chunk; });

async function waitUntil(predicate, context, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = predicate();
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error(`timed out waiting for ${context}`);
}

async function stopProcess(process) {
  if (process.exitCode !== null) return;
  process.kill('SIGTERM');
  await Promise.race([
    new Promise(resolve => process.once('close', resolve)),
    new Promise(resolve => setTimeout(resolve, 2_000)),
  ]);
  if (process.exitCode === null) process.kill('SIGKILL');
}

let browserRelay = null;
let app = null;
let blender = null;
let summary;
try {
  const baseUrl = await waitUntil(() => {
    const match = relayStdout.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
    if (match) return match[1];
    if (relay.exitCode !== null) {
      throw new Error(`relay exited before listening\n${relayStdout}\n${relayStderr}`);
    }
    return null;
  }, 'relay address');

  const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
  const { default: init, HyperscopeAppShadow } = await import(packageUrl);
  await init({
    module_or_path: readFileSync(`${repository}/pkg/quilting_wasm_bg.wasm`),
  });
  app = new HyperscopeAppShadow();
  const exactFrames = [];
  browserRelay = new BrowserLocalPeerRelay({
    baseUrl,
    token,
    app,
    pollIntervalMs: 5,
    onReceipt(_receipt, frameJson) {
      exactFrames.push(frameJson);
    },
  });
  browserRelay.start();

  blender = spawn(
    'blender',
    [
      '--background',
      '--factory-startup',
      '--python-exit-code', '1',
      '--python',
      `${repository}/tools/blender_hyperscape/tests/blender_relay_publish.py`,
    ],
    {
      cwd: repository,
      env: {
        ...process.env,
        HYPERSCAPE_RELAY_URL: baseUrl,
        HYPERSCAPE_RELAY_TOKEN: token,
        HYPERSCAPE_ENTITY_ID: entity,
        HYPERSCAPE_PEER_ID: peer,
        BLENDER_USER_CONFIG: blenderConfig,
        BLENDER_USER_SCRIPTS: join(blenderConfig, 'scripts'),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  let blenderStdout = '';
  let blenderStderr = '';
  blender.stdout.setEncoding('utf8');
  blender.stderr.setEncoding('utf8');
  blender.stdout.on('data', chunk => { blenderStdout += chunk; });
  blender.stderr.on('data', chunk => { blenderStderr += chunk; });
  const blenderExit = await new Promise(resolve => blender.once('close', resolve));
  assert.equal(blenderExit, 0, `${blenderStdout}\n${blenderStderr}`);
  assert.match(blenderStdout, /Hyperscape Blender relay publish passed/);

  await waitUntil(
    () => browserRelay.snapshot().appliedFrames >= 2,
    'Rust/WASM authored and presence admission',
  );
  const authored = app.snapshot().authoredEntities;
  assert.deepEqual(authored, [{
    entityId: entity,
    translation: [3, 4, 5],
    rotationWxyz: [1, 0, 0, 0],
    scale: [1, 1, 1],
  }]);
  const authoredFrame = exactFrames.find(frame => frame.includes('"lane":"authored"'));
  assert.ok(authoredFrame);
  const sequence = authoredFrame.match(/"sequence":([0-9]+)/)?.[1];
  assert.ok(sequence);
  assert.ok(BigInt(sequence) > BigInt(Number.MAX_SAFE_INTEGER));

  const identity = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
  const extraction = app.extractPackedScene(JSON.stringify([{
    layer: '73000000-0000-4000-8000-000000000003',
    asset: '73000000-0000-4000-8000-000000000004',
    layerTransform: {
      translation: [0, 0, 0],
      rotation: [1, 0, 0, 0],
      scale: [1, 1, 1],
    },
    nodes: [{
      packedNode: 0,
      sourceNode: 0,
      entityId: entity,
      sourceWorld: identity,
    }],
  }]));
  assert.equal(extraction.nodes[0].source, 'authored_absolute');
  assert.deepEqual(extraction.nodes[0].matrix.slice(12, 15), [3, 4, 5]);

  summary = {
    blenderAuthoredFrames: 1,
    browserReceivedFrames: browserRelay.snapshot().receivedFrames,
    rustAppliedFrames: browserRelay.snapshot().appliedFrames,
    exactSequence: sequence,
    projectedTranslation: extraction.nodes[0].matrix.slice(12, 15),
  };
} finally {
  if (browserRelay) await browserRelay.stop();
  app?.free();
  if (blender) await stopProcess(blender);
  await stopProcess(relay);
  rmSync(blenderConfig, { recursive: true, force: true });
}

console.log(JSON.stringify(summary));
