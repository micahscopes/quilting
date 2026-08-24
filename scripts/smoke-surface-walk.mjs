import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  composeSurfaceRelativeForward,
  resolveSurfaceWalkView,
  scaleRelativeNearPlane,
  sceneRelativeWalkSpeed,
} from '../hyperscope_focus.mjs';

const repository = fileURLToPath(new URL('..', import.meta.url));
const packageUrl = pathToFileURL(`${repository}/pkg/quilting_wasm.js`).href;
const wasmPath = `${repository}/pkg/quilting_wasm_bg.wasm`;
const { default: init, HyperscopeSurfaceWalk } = await import(packageUrl);
await init({ module_or_path: readFileSync(wasmPath) });

const defaultControls = Object.freeze({
  baseRadiiPerSecond: 0.2,
  baseEyeHeight: 0.035,
  speedOctaveSteps: 0,
  bodyScaleOctaveSteps: 0,
  eyeHeightOctaveSteps: 0,
  smoothingSeconds: 0.18,
  tangentPullFraction: 0.7,
  fastMultiplier: 3,
  defaultNear: 0.01,
  minimumNear: 1e-7,
  nearEyeFraction: 0.08,
});

const canonicalCamera = Object.freeze({
  eye: [0, 1, 3],
  forward: [0, 0, -1],
  up: [0, 1, 0],
  controlDistance: 3,
  verticalFovRadians: Math.PI / 3,
  near: 0.01,
  far: 10_000,
});

function close(actual, expected, tolerance = 2e-11) {
  assert(Number.isFinite(actual), `expected finite value, got ${actual}`);
  assert(
    Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`,
  );
}

function vectorClose(actual, expected, tolerance = 2e-11) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, axis) => close(value, expected[axis], tolerance));
}

function length(vector) {
  return Math.hypot(...vector);
}

function dot(left, right) {
  return left.reduce((sum, value, axis) => sum + value * right[axis], 0);
}

function assertOrthonormal(camera) {
  close(length(camera.right), 1, 2e-10);
  close(length(camera.up), 1, 2e-10);
  close(length(camera.forward), 1, 2e-10);
  close(dot(camera.right, camera.up), 0, 2e-10);
  close(dot(camera.right, camera.forward), 0, 2e-10);
  close(dot(camera.up, camera.forward), 0, 2e-10);
}

function cameraRequest(camera) {
  return {
    eye: camera.eye,
    forward: camera.forward,
    up: camera.up,
    controlDistance: camera.controlDistance,
    verticalFovRadians: camera.verticalFovRadians,
    near: camera.near,
    far: camera.far,
  };
}

function normalized(vector) {
  const magnitude = Math.hypot(...vector);
  return vector.map(value => value / magnitude);
}

function cross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function cameraBasis(camera) {
  const forward = normalized(camera.forward);
  const right = normalized(camera.right || cross(forward, camera.up));
  const up = normalized(cross(right, forward));
  return [...right, ...up, ...forward];
}

function walkEyeHeight(controls) {
  return controls.baseEyeHeight
    * 2 ** (controls.bodyScaleOctaveSteps / 100)
    * 2 ** (controls.eyeHeightOctaveSteps / 100);
}

function assertViewParity(rustFrame, oracleFrame) {
  assert(oracleFrame, 'incumbent JavaScript oracle rejected a Rust-accepted frame');
  vectorClose(rustFrame.filteredPosition, oracleFrame.filteredPosition, 2e-10);
  vectorClose(rustFrame.filteredNormal, oracleFrame.filteredNormal, 2e-10);
  vectorClose(rustFrame.camera.eye, oracleFrame.eye, 2e-10);
  vectorClose(
    [...rustFrame.camera.right, ...rustFrame.camera.up, ...rustFrame.camera.forward],
    oracleFrame.basis,
    2e-10,
  );
  if (oracleFrame.tangentForward) {
    vectorClose(rustFrame.tangentForward, oracleFrame.tangentForward, 2e-10);
  } else {
    assert.equal(rustFrame.tangentForward, undefined);
  }
  if (oracleFrame.relativePitch == null) {
    assert.equal(rustFrame.relativePitchRadians, undefined);
  } else {
    close(rustFrame.relativePitchRadians, oracleFrame.relativePitch, 2e-10);
  }
}

const mapper = new HyperscopeSurfaceWalk();
let mappingCases = 0;
for (const sceneRadius of [0.1, 1, 10]) {
  for (const speedOctaveSteps of [-400, -100, 0, 100, 400]) {
    for (const bodyScaleOctaveSteps of [-800, 0, 800]) {
      for (const eyeHeightOctaveSteps of [-400, 0, 400]) {
        for (const fast of [false, true]) {
          const controls = {
            ...defaultControls,
            speedOctaveSteps,
            bodyScaleOctaveSteps,
            eyeHeightOctaveSteps,
          };
          const bodyScale = 2 ** (bodyScaleOctaveSteps / 100);
          const radiiPerSecond = 0.2 * 2 ** (speedOctaveSteps / 100);
          const expectedSpeed = sceneRelativeWalkSpeed(
            sceneRadius,
            radiiPerSecond,
            bodyScale,
          ) * (fast ? 3 : 1);
          const expectedEyeHeight = 0.035
            * bodyScale * 2 ** (eyeHeightOctaveSteps / 100);
          const expectedNear = scaleRelativeNearPlane(expectedEyeHeight);
          for (const [forwardAxis, rightAxis] of [
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [2, -3],
          ]) {
            const motion = mapper.planMotion({
              camera: canonicalCamera,
              outputNormal: [0, 1, 0],
              sceneRadius,
              controls,
              input: { forwardAxis, rightAxis, fast },
            });
            close(motion.metrics.bodyScale, bodyScale);
            close(motion.metrics.radiiPerSecond, radiiPerSecond);
            close(motion.metrics.speed, expectedSpeed);
            close(motion.metrics.eyeHeight, expectedEyeHeight);
            close(motion.metrics.near, expectedNear);
            const inputLength = Math.max(1, Math.hypot(forwardAxis, rightAxis));
            vectorClose(motion.desiredOutputVelocity, [
              rightAxis * expectedSpeed / inputLength,
              0,
              -forwardAxis * expectedSpeed / inputLength,
            ]);
            mappingCases += 1;
          }
        }
      }
    }
  }
}

const initialPitch = -Math.PI / 6;
const pitchedForward = composeSurfaceRelativeForward(
  [0, 0, -1],
  [0, 1, 0],
  initialPitch,
);
const pitchedCamera = {
  ...canonicalCamera,
  forward: pitchedForward,
  up: [0, Math.cos(initialPitch), Math.sin(initialPitch)],
};
const firstWalker = new HyperscopeSurfaceWalk();
let frame = firstWalker.followFrame({
  camera: pitchedCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 2,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
let oracleState = {
  filteredPosition: null,
  filteredNormal: null,
  tangentForward: null,
  relativePitch: null,
};
let oracleBasis = cameraBasis(pitchedCamera);
let oracleFrame = resolveSurfaceWalkView(
  oracleState,
  { basis: oracleBasis },
  { outputPosition: [0, 0, 0], outputNormal: [0, 1, 0] },
  {
    active: false,
    deltaSeconds: 0,
    smoothingSeconds: defaultControls.smoothingSeconds,
    tangentPullFraction: defaultControls.tangentPullFraction,
    eyeHeight: walkEyeHeight(defaultControls),
    orient: true,
    captureRelativeView: false,
  },
);
assertViewParity(frame, oracleFrame);
oracleState = oracleFrame;
oracleBasis = oracleFrame.basis;
close(frame.relativePitchRadians, initialPitch);
vectorClose(frame.camera.eye, [0, 0.035, 0]);
assertOrthonormal(frame.camera);

const recaptureWalker = new HyperscopeSurfaceWalk();
let recaptureFrame = recaptureWalker.followFrame({
  camera: canonicalCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
const recapturedPitch = Math.PI / 5;
const recapturedForward = composeSurfaceRelativeForward(
  [0, 0, -1],
  [0, 1, 0],
  recapturedPitch,
);
recaptureFrame = recaptureWalker.followFrame({
  camera: {
    ...cameraRequest(recaptureFrame.camera),
    forward: recapturedForward,
    up: [0, Math.cos(recapturedPitch), Math.sin(recapturedPitch)],
  },
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: true,
});
close(recaptureFrame.relativePitchRadians, recapturedPitch);

const positionOnlyWalker = new HyperscopeSurfaceWalk();
let positionOnlyFrame = positionOnlyWalker.followFrame({
  camera: canonicalCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
const priorOrientation = {
  right: positionOnlyFrame.camera.right,
  up: positionOnlyFrame.camera.up,
  forward: positionOnlyFrame.camera.forward,
};
positionOnlyFrame = positionOnlyWalker.followFrame({
  camera: cameraRequest(positionOnlyFrame.camera),
  outputPosition: [3, 2, 1],
  outputNormal: [1, 0, 0],
  sceneRadius: 1,
  controls: { ...defaultControls, smoothingSeconds: 0 },
  deltaSeconds: 1 / 60,
  orient: false,
  captureRelativeView: false,
});
vectorClose(positionOnlyFrame.camera.right, priorOrientation.right);
vectorClose(positionOnlyFrame.camera.up, priorOrientation.up);
vectorClose(positionOnlyFrame.camera.forward, priorOrientation.forward);
vectorClose(positionOnlyFrame.camera.eye, [3.035, 2, 1]);

const traceRequests = [];
for (let index = 1; index <= 600; index++) {
  const phase = index / 37;
  const normal = [0.35 * Math.sin(phase), 1, 0.3 * Math.cos(phase)];
  const normalLength = Math.hypot(...normal);
  traceRequests.push({
    outputPosition: [
      index / 300,
      0.08 * Math.sin(index / 23),
      0.12 * Math.cos(index / 31),
    ],
    outputNormal: normal.map(value => value / normalLength),
    sceneRadius: 2,
    controls: {
      ...defaultControls,
      bodyScaleOctaveSteps: (index % 161) - 80,
      eyeHeightOctaveSteps: (index % 101) - 50,
      smoothingSeconds: [0, 0.18, 0.7][index % 3],
      tangentPullFraction: [0, 0.7, 1][index % 3],
    },
    deltaSeconds: [1 / 240, 1 / 60, 1 / 30, 0.125][index % 4],
    orient: true,
    captureRelativeView: false,
  });
}

const secondWalker = new HyperscopeSurfaceWalk();
const secondInitialFrame = secondWalker.followFrame({
  camera: pitchedCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 2,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
assert.deepEqual(secondInitialFrame, frame);
let firstTraceCamera = cameraRequest(frame.camera);
let secondTraceCamera = cameraRequest(secondInitialFrame.camera);
let lastTraceFrame = null;
for (const request of traceRequests) {
  const first = firstWalker.followFrame({ ...request, camera: firstTraceCamera });
  const second = secondWalker.followFrame({ ...request, camera: secondTraceCamera });
  oracleFrame = resolveSurfaceWalkView(
    oracleState,
    { basis: oracleBasis },
    {
      outputPosition: request.outputPosition,
      outputNormal: request.outputNormal,
    },
    {
      active: true,
      deltaSeconds: request.deltaSeconds,
      smoothingSeconds: request.controls.smoothingSeconds,
      tangentPullFraction: request.controls.tangentPullFraction,
      eyeHeight: walkEyeHeight(request.controls),
      orient: request.orient,
      captureRelativeView: request.captureRelativeView,
    },
  );
  assert.deepEqual(first, second);
  assertViewParity(first, oracleFrame);
  assertOrthonormal(first.camera);
  close(length(first.filteredNormal), 1, 2e-10);
  close(first.relativePitchRadians, initialPitch);
  firstTraceCamera = cameraRequest(first.camera);
  secondTraceCamera = cameraRequest(second.camera);
  oracleState = oracleFrame;
  oracleBasis = oracleFrame.basis;
  lastTraceFrame = first;
}

const atomicBaseline = new HyperscopeSurfaceWalk();
let baselineFrame = atomicBaseline.followFrame({
  camera: canonicalCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
const atomicSubject = new HyperscopeSurfaceWalk();
let subjectFrame = atomicSubject.followFrame({
  camera: canonicalCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 0,
  orient: true,
  captureRelativeView: false,
});
assert.throws(() => atomicSubject.followFrame({
  camera: cameraRequest(subjectFrame.camera),
  outputPosition: [1, 0, 0],
  outputNormal: [0, 0, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 1 / 60,
  orient: true,
  captureRelativeView: false,
}));
const nextRequest = {
  outputPosition: [2, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: 1 / 60,
  orient: true,
  captureRelativeView: false,
};
baselineFrame = atomicBaseline.followFrame({
  ...nextRequest,
  camera: cameraRequest(baselineFrame.camera),
});
subjectFrame = atomicSubject.followFrame({
  ...nextRequest,
  camera: cameraRequest(subjectFrame.camera),
});
assert.deepEqual(subjectFrame, baselineFrame);

assert.equal(firstWalker.active, true);
firstWalker.reset();
assert.equal(firstWalker.active, false);
assert(lastTraceFrame);
assert.throws(() => mapper.planMotion({
  camera: canonicalCamera,
  outputNormal: [0, 1, 0],
  sceneRadius: 0,
  controls: defaultControls,
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
}));
assert.throws(() => mapper.planMotion({
  camera: canonicalCamera,
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: { ...defaultControls, bodyScaleOctaveSteps: 1_000_000 },
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
}));
assert.throws(() => mapper.planMotion({
  camera: canonicalCamera,
  outputNormal: [0, 1, 0],
  sceneRadius: 1e-308,
  controls: {
    ...defaultControls,
    baseRadiiPerSecond: Number.MAX_VALUE,
    speedOctaveSteps: 100,
  },
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
}));
assert.throws(() => mapper.planMotion({
  camera: { ...canonicalCamera, eye: [0, 1, 3, 4] },
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
}));
assert.throws(() => mapper.planMotion({
  camera: canonicalCamera,
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
  unexpected: true,
}));
assert.throws(() => mapper.planMotion({
  camera: canonicalCamera,
  outputNormal: [Number.NaN, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  input: { forwardAxis: 1, rightAxis: 0, fast: false },
}));
assert.throws(() => mapper.followFrame({
  camera: canonicalCamera,
  outputPosition: [0, 0, 0],
  outputNormal: [0, 1, 0],
  sceneRadius: 1,
  controls: defaultControls,
  deltaSeconds: -1 / 60,
  orient: true,
  captureRelativeView: false,
}));

console.log(
  `surface walk smoke passed: ${mappingCases} mapping cases, 600 incumbent-parity frames, recapture/orientation modes, strict atomic rejection`,
);
