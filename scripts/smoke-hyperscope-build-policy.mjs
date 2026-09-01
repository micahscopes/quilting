import assert from 'node:assert/strict';
import {
  chmodSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const buildScript = join(repository, 'scripts/build-hyperscope-wasm.sh');
const temporary = mkdtempSync(join(tmpdir(), 'hyperscope-build-policy-'));
const fakeWasmPack = join(temporary, 'wasm-pack');
const cleanEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(([key]) => ![
    'CARGO_BUILD_JOBS',
    'HYPERSCOPE_ARTIFACT_BUILD',
    'HYPERSCOPE_BUILD_JOBS',
    'HYPERSCOPE_DURABLE_HISTORY',
    'HYPERSCOPE_WASM_OPT',
    'HYPERSCOPE_WASM_PROFILE',
  ].includes(key)),
);

writeFileSync(fakeWasmPack, `#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$HYPERSCOPE_CAPTURE_ARGS"
printf '%s\n' "$CARGO_BUILD_JOBS" > "$HYPERSCOPE_CAPTURE_JOBS"
`);
chmodSync(fakeWasmPack, 0o755);

function run(name, overrides = {}) {
  const argumentsPath = join(temporary, `${name}.args`);
  const jobsPath = join(temporary, `${name}.jobs`);
  const result = spawnSync('bash', [buildScript], {
    cwd: repository,
    encoding: 'utf8',
    env: {
      ...cleanEnvironment,
      PATH: `${temporary}:${cleanEnvironment.PATH}`,
      HYPERSCOPE_CAPTURE_ARGS: argumentsPath,
      HYPERSCOPE_CAPTURE_JOBS: jobsPath,
      ...overrides,
    },
  });
  return {
    ...result,
    args: result.status === 0
      ? readFileSync(argumentsPath, 'utf8').trim().split('\n')
      : [],
    jobs: result.status === 0 ? readFileSync(jobsPath, 'utf8').trim() : null,
  };
}

try {
  const ordinary = run('ordinary');
  assert.equal(ordinary.status, 0, ordinary.stderr);
  assert.ok(ordinary.args.includes('--release'));
  assert.ok(ordinary.args.includes('--no-opt'));
  assert.equal(ordinary.args.includes('--dev'), false);
  assert.equal(ordinary.jobs, '2');
  assert.equal(
    ordinary.args[ordinary.args.indexOf('--features') + 1],
    'leptos-ui,webgpu-backend',
  );

  const durable = run('durable', { HYPERSCOPE_DURABLE_HISTORY: '1' });
  assert.equal(durable.status, 0, durable.stderr);
  assert.equal(
    durable.args[durable.args.indexOf('--features') + 1],
    'leptos-ui,webgpu-backend,durable-history',
  );

  const fast = run('fast', {
    HYPERSCOPE_WASM_PROFILE: 'dev',
    HYPERSCOPE_BUILD_JOBS: '1',
  });
  assert.equal(fast.status, 0, fast.stderr);
  assert.ok(fast.args.includes('--dev'));
  assert.ok(fast.args.includes('--no-opt'));
  assert.equal(fast.args.includes('--release'), false);
  assert.equal(fast.jobs, '1');

  const artifact = run('artifact', {
    HYPERSCOPE_ARTIFACT_BUILD: '1',
    HYPERSCOPE_WASM_OPT: '1',
    HYPERSCOPE_BUILD_JOBS: '3',
  });
  assert.equal(artifact.status, 0, artifact.stderr);
  assert.ok(artifact.args.includes('--release'));
  assert.equal(artifact.args.includes('--no-opt'), false);
  assert.equal(artifact.jobs, '3');

  const accidentalOptimizer = run('accidental-optimizer', {
    HYPERSCOPE_WASM_OPT: '1',
  });
  assert.equal(accidentalOptimizer.status, 2);
  assert.match(accidentalOptimizer.stderr, /requires HYPERSCOPE_ARTIFACT_BUILD=1/);

  const invalid = run('invalid', { HYPERSCOPE_WASM_PROFILE: 'fast-ish' });
  assert.equal(invalid.status, 2);
  assert.match(invalid.stderr, /must be release, dev, or profiling/);

  const invalidDurability = run('invalid-durability', {
    HYPERSCOPE_DURABLE_HISTORY: 'sometimes',
  });
  assert.equal(invalidDurability.status, 2);
  assert.match(invalidDurability.stderr, /must be 0 or 1/);

  console.log('Hyperscope WASM build policy smoke passed');
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
