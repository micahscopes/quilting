import assert from 'node:assert/strict';
import test from 'node:test';

import {
  SPACEMOUSE_AXIS_MAX,
  SpaceMouseController,
  createSpaceMouseState,
  decodeSpaceMouseReport,
  mapSpaceMouseBlenderAxes,
  mapSpaceMouseFlyAxes,
  mapSpaceMouseNavigationAxes,
  shapeSpaceMouseAxis,
  spaceMouseLocalRotationVector,
  spaceMouseModifierMode,
  spaceMouseNavigationPolicy,
} from '../spacemouse.mjs';

function report(values) {
  const buffer = new ArrayBuffer(values.length * 2);
  const view = new DataView(buffer);
  values.forEach((value, index) => view.setInt16(index * 2, value, true));
  return view;
}

test('Bluetooth report 1 decodes all six signed axes', () => {
  const state = createSpaceMouseState();
  assert.equal(
    decodeSpaceMouseReport(1, report([350, -350, 175, -175, 0, 35]), state),
    true,
  );
  assert.deepEqual(
    Array.from(state.axes).map(value => Number(value.toFixed(3))),
    [1, -1, 0.5, -0.5, 0, 0.1],
  );
});

test('classic split translation and rotation reports retain the other half', () => {
  const state = createSpaceMouseState();
  decodeSpaceMouseReport(1, report([350, 175, -350]), state);
  decodeSpaceMouseReport(2, report([-175, 70, 350]), state);
  assert.deepEqual(
    Array.from(state.axes).map(value => Number(value.toFixed(3))),
    [1, 0.5, -1, -0.5, 0.2, 1],
  );
});

test('button reports preserve the complete bitmask', () => {
  const state = createSpaceMouseState();
  const data = new DataView(new Uint8Array([0x03, 0x01]).buffer);
  assert.equal(decodeSpaceMouseReport(3, data, state), true);
  assert.equal(state.buttons, 0x0103);
});

test('the two primary buttons select inversion and depth-of-field layers', () => {
  assert.equal(spaceMouseModifierMode(0), 'camera');
  assert.equal(spaceMouseModifierMode(1), 'inversion');
  assert.equal(spaceMouseModifierMode(2), 'depth-of-field');
  assert.equal(spaceMouseModifierMode(3), 'depth-of-field');
  assert.equal(spaceMouseModifierMode(0x103), 'depth-of-field');
});

test('Hyperscope fly uses the requested positive Rx and Ry directions', () => {
  const mapped = mapSpaceMouseFlyAxes([1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(mapped), [1, 2, 3, 4, 5, 6]);
});

test('Blender device normalization matches the c63a backend transform', () => {
  const mapped = mapSpaceMouseBlenderAxes([1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(mapped), [1, -3, 2, -4, 6, -5]);
});

test('Blender Object reverses all normalized navigation axes relative to Fly', () => {
  const fly = mapSpaceMouseNavigationAxes([1, 2, 3, 4, 5, 6], { mode: 'fly' });
  const object = mapSpaceMouseNavigationAxes([1, 2, 3, 4, 5, 6], { mode: 'object' });
  assert.deepEqual(Array.from(fly), [1, -3, 2, -4, 6, -5]);
  assert.deepEqual(Array.from(object), [-1, 3, -2, 4, -6, 5]);
});

test('Blender Y/Z swap and independent inversion masks match NDOF preferences', () => {
  const mapped = mapSpaceMouseNavigationAxes(
    [1, 2, 3, 4, 5, 6],
    { mode: 'fly', swapYZ: true, invertPan: 0b101, invertRotate: 0b010 },
  );
  assert.deepEqual(Array.from(mapped), [-1, -2, 3, -4, -5, 6]);
});

test('Hyperscope pitch and yaw both follow the rolled local camera frame', () => {
  // right=(0,-1,0), up=(1,0,0) is a 90-degree rolled camera frame.
  const rolledBasis = [0, -1, 0, 1, 0, 0, 0, 0, -1];
  assert.deepEqual(
    Array.from(spaceMouseLocalRotationVector(rolledBasis, 1, 0, 0)),
    [0, -1, 0, 1],
  );
  assert.deepEqual(
    Array.from(spaceMouseLocalRotationVector(rolledBasis, 0, 1, 0)),
    [1, 0, 0, 1],
  );
});

test('navigation policies distinguish pivot orbit, free flight, and drone', () => {
  assert.deepEqual(
    spaceMouseNavigationPolicy('object'),
    { mode: 'object', orbit: true, drone: false, horizonLocked: false },
  );
  assert.deepEqual(
    spaceMouseNavigationPolicy('fly'),
    { mode: 'fly', orbit: false, drone: false, horizonLocked: false },
  );
  assert.deepEqual(
    spaceMouseNavigationPolicy('drone'),
    { mode: 'drone', orbit: false, drone: true, horizonLocked: true },
  );
  assert.deepEqual(
    spaceMouseNavigationPolicy('future-mode'),
    { mode: 'hyperscope', orbit: false, drone: false, horizonLocked: false },
  );
});

test('unknown and truncated reports do not mutate state', () => {
  const state = createSpaceMouseState();
  state.axes[0] = 0.25;
  assert.equal(decodeSpaceMouseReport(9, new DataView(new ArrayBuffer(2)), state), false);
  assert.equal(decodeSpaceMouseReport(1, new DataView(new ArrayBuffer(4)), state), false);
  assert.equal(state.axes[0], 0.25);
});

test('axis response removes drift and preserves full-scale sign', () => {
  assert.equal(shapeSpaceMouseAxis(0.04), 0);
  assert.equal(shapeSpaceMouseAxis(-0.08), 0);
  assert.equal(shapeSpaceMouseAxis(1), 1);
  assert.equal(shapeSpaceMouseAxis(-1), -1);
  assert(shapeSpaceMouseAxis(0.5) > 0 && shapeSpaceMouseAxis(0.5) < 0.5);
});

test('frame sampling smooths fresh input and decays stale input', () => {
  const controller = new SpaceMouseController({ hid: null, staleAfterMs: 100, responseHz: 20 });
  controller.state.axes[0] = SPACEMOUSE_AXIS_MAX / SPACEMOUSE_AXIS_MAX;
  controller.lastAxisReportAt = 0;
  const fresh = controller.sample(50, 1 / 60)[0];
  assert(fresh > 0 && fresh < 1);
  const stale = controller.sample(200, 1)[0];
  assert(stale < fresh * 0.01);
});
