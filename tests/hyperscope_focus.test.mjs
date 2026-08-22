import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyMobiusPoint,
  buildNodeFocusRecords,
  faceSourceCentroid,
  focusSphereFromBound,
  focusRelativeNavigationSpeed,
  framedSphereDistance,
  interpolateSphereFit,
  isPrimarySelectionClick,
  mobiusConformalScaleAt,
  perspectiveNavigationSpeed,
  scaleRadiusMultiplicatively,
  scaleAnchoredFocusRadius,
  smootherstep01,
} from '../hyperscope_focus.mjs';

const STRIDE = 52;

function instanceFace(points, vertexIds = [0, 1, 2]) {
  const data = new Float32Array(STRIDE);
  for (let corner = 0; corner < 3; corner++) {
    const offset = corner * 4;
    data[offset] = vertexIds[corner];
    data.set(points[corner], offset + 1);
  }
  return data;
}

test('node focus records derive stable per-node source spheres from face instances', () => {
  const first = instanceFace([[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [0, 1, 2]);
  const second = instanceFace([[-1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 2, 3]);
  const instances = new Float32Array(STRIDE * 2);
  instances.set(first, 0);
  instances.set(second, STRIDE);
  const records = buildNodeFocusRecords(instances, new Int32Array([7, 7]));
  const record = records.get(7);
  assert.deepEqual(record.center, [0, 0, 0]);
  assert.equal(record.vertexCount, 4);
  assert(Math.abs(record.radius - Math.sqrt(2)) < 1e-6);
});

test('face source centroid ignores packed vertex IDs', () => {
  const face = instanceFace([[0, 0, 0], [3, 0, 0], [0, 6, 0]], [91, 92, 93]);
  assert.deepEqual(faceSourceCentroid(face, 0), [1, 2, 0]);
  assert.equal(faceSourceCentroid(face, 1), null);
});

test('sphere reflection maps points and reports its local conformal scale', () => {
  // Unit sphere inversion around the origin: x -> x / |x|^2.
  const sphereReflection = new Float32Array([
    0, 0, 0, 0,
    -1, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
  ]);
  assert.deepEqual(applyMobiusPoint(sphereReflection, [2, 0, 0]), [0.5, 0, 0]);
  assert(Math.abs(mobiusConformalScaleAt(sphereReflection, [2, 0, 0]) - 0.25) < 1e-9);
  assert.equal(applyMobiusPoint(sphereReflection, [0, 0, 0]), null);
});

test('sphere fitting eases endpoints and interpolates radius without crossing zero', () => {
  assert.equal(smootherstep01(0), 0);
  assert.equal(smootherstep01(1), 1);
  const halfway = interpolateSphereFit(
    { center: [0, 0, 0], radius: 1 },
    { center: [2, 4, 6], radius: 4 },
    0.5,
  );
  assert.deepEqual(halfway.center, [1, 2, 3]);
  assert(Math.abs(halfway.radius - 2) < 1e-12);
});

test('object focus sphere applies a stable margin and rejects invalid bounds', () => {
  assert.deepEqual(
    focusSphereFromBound({ center: [1, 2, 3], radius: 2 }),
    { center: [1, 2, 3], radius: 2.2, margin: 1.1 },
  );
  assert.equal(focusSphereFromBound({ center: [1, 2], radius: 2 }), null);
  assert.equal(focusSphereFromBound({ center: [1, 2, 3], radius: -1 }), null);
});

test('selection click policy preserves orbit drags and non-primary buttons', () => {
  assert.equal(isPrimarySelectionClick(0, 0), true);
  assert.equal(isPrimarySelectionClick(0, 4), true);
  assert.equal(isPrimarySelectionClick(0, 4.01), false);
  assert.equal(isPrimarySelectionClick(2, 0), false);
});

test('perspective navigation speed is screen-relative and linear in depth', () => {
  const near = perspectiveNavigationSpeed(2, 1000);
  const far = perspectiveNavigationSpeed(8, 1000);
  assert(near > 0);
  assert(Math.abs(far / near - 4) < 1e-12);
  assert(Math.abs(perspectiveNavigationSpeed(2, 2000) / near - 0.5) < 1e-12);
});

test('focus-relative navigation caps tiny selections without accelerating large ones', () => {
  const screenSpeed = perspectiveNavigationSpeed(4, 1000);
  assert.equal(focusRelativeNavigationSpeed(4, 1000, null), screenSpeed);
  assert.equal(focusRelativeNavigationSpeed(4, 1000, 0.1), 0.2);
  assert.equal(focusRelativeNavigationSpeed(4, 1000, 100), screenSpeed);
});

test('multiplicative radius control applies the same ratio at every scale', () => {
  const small = scaleRadiusMultiplicatively(0.1, 0.5, 3, 0.2);
  const large = scaleRadiusMultiplicatively(2, 0.5, 3, 0.2);
  assert(Math.abs(small / 0.1 - large / 2) < 1e-12);
  assert.equal(scaleRadiusMultiplicatively(5, 1, 3, 1), 5);
});

test('anchored radius editing preserves the selected object and caps its margin', () => {
  assert.equal(scaleAnchoredFocusRadius(2, 2.2, -1, 3, 1), 2);
  assert.equal(scaleAnchoredFocusRadius(2, 2.2, 1, 30, 1), 8);
});

test('sphere framing respects the narrower viewport axis and requested margin', () => {
  const landscape = framedSphereDistance(2, 16 / 9);
  assert(Math.abs(landscape - 4.6) < 1e-12);
  assert(framedSphereDistance(2, 0.5) > landscape);
});
