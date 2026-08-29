import assert from 'node:assert/strict';
import test from 'node:test';

import {
  advanceVirtualTransitionClock,
  appendPackedNodeIdentities,
  applyMobiusPoint,
  buildFocusBounds,
  buildNodeFocusRecords,
  compactifiedRadialCoordinate,
  composeSurfaceRelativeForward,
  decomposeSurfaceRelativeForward,
  durablePresentationAssetId,
  faceSourceCentroid,
  focusSphereFromBound,
  focusRelativeNavigationSpeed,
  framedSphereDistance,
  interpolateWalkAnchorEye,
  interpolateSphereFit,
  isPrimarySelectionClick,
  mobiusConformalScaleAt,
  perspectiveNavigationSpeed,
  resolveSurfaceWalkView,
  scaleRadiusMultiplicatively,
  scaleAnchoredFocusRadius,
  scaleRelativeNearPlane,
  sceneRelativeWalkSpeed,
  selectionTransitionFrameDeltaSeconds,
  sharedFocusSphereActive,
  spheroidalFocusEnabled,
  spheroidalDefocus,
  smootherstep01,
  transportCameraAcrossSphereReflections,
  transportPointAndDirectionsAcrossSphereReflections,
} from '../hyperscope_focus.mjs';

const STRIDE = 52;

test('selection frame clock retains a future event boundary across stale RAF timestamps', () => {
  let eventAtMs = 110;
  let elapsed = 0;
  const staleFrameAtMs = 100;
  elapsed += selectionTransitionFrameDeltaSeconds(
    staleFrameAtMs, eventAtMs, true, 0.016, 0.016,
  );
  if (staleFrameAtMs >= eventAtMs) eventAtMs = null;
  assert.equal(elapsed, 0);
  assert.equal(eventAtMs, 110, 'a stale RAF must not consume the event boundary');

  const nextFrameAtMs = 126;
  elapsed += selectionTransitionFrameDeltaSeconds(
    nextFrameAtMs, eventAtMs, true, 0.026, 0.026,
  );
  if (nextFrameAtMs >= eventAtMs) eventAtMs = null;
  assert.equal(elapsed, 0.016);
  assert.equal(eventAtMs, null);
  assert.equal(
    selectionTransitionFrameDeltaSeconds(142, eventAtMs, true, 0.016, 0.016),
    0.016,
  );
});

test('durable presentation asset scope requires exact manifest-fetch provenance', () => {
  const assetId = '60000000-0000-4000-8000-000000000001';
  const assets = [{ id: assetId, uri: '/horse.glb' }];
  assert.equal(
    durablePresentationAssetId(assets, '/horse.glb', 'manifest-fetch'),
    assetId,
  );
  assert.equal(durablePresentationAssetId(assets, '/horse.glb', 'indexeddb'), null);
  assert.equal(durablePresentationAssetId(assets, 'horse.glb', 'manifest-fetch'), null);
  assert.equal(
    durablePresentationAssetId(assets, '/local-glbs/horse.glb', 'manifest-fetch'),
    null,
  );
});

test('composed node identity keeps durable asset scope and source offsets', () => {
  const primaryAsset = '60000000-0000-4000-8000-000000000001';
  const secondaryAsset = '60000000-0000-4000-8000-000000000002';
  const firstEntity = '70000000-0000-4000-8000-000000000001';
  const secondEntity = '70000000-0000-4000-8000-000000000002';
  const nonPickableCamera = '70000000-0000-4000-8000-000000000003';
  const identities = new Map();

  assert.equal(appendPackedNodeIdentities(
    identities,
    primaryAsset.toUpperCase(),
    [firstEntity, null, secondEntity, nonPickableCamera],
    [0, 1, 2],
  ), 2);
  assert.equal(appendPackedNodeIdentities(
    identities,
    secondaryAsset,
    [firstEntity],
    [0],
    3,
  ), 1);
  assert.deepEqual(identities.get(0), {
    assetId: primaryAsset,
    entityId: firstEntity,
    sourceNode: 0,
    durable: true,
  });
  assert.deepEqual(identities.get(2), {
    assetId: primaryAsset,
    entityId: secondEntity,
    sourceNode: 2,
    durable: true,
  });
  assert.deepEqual(identities.get(3), {
    assetId: secondaryAsset,
    entityId: firstEntity,
    sourceNode: 0,
    durable: true,
  });
  assert.notDeepEqual(identities.get(0), identities.get(3));
  assert.throws(
    () => appendPackedNodeIdentities(identities, primaryAsset, [firstEntity], [0]),
    /already has a durable identity/,
  );
  const rejected = new Map();
  assert.throws(
    () => appendPackedNodeIdentities(
      rejected,
      primaryAsset,
      [firstEntity, 'not-a-uuid'],
      [0, 1],
    ),
    /must be a non-nil UUID/,
  );
  assert.equal(rejected.size, 0, 'a rejected batch must not leave a partial catalog');
});

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
  assert.equal('vertexIds' in record, false);
  assert.equal('vertices' in record, false);
});

test('focus bound pass also encloses the complete untransformed source scene', () => {
  const first = instanceFace([[-2, -1, 0], [0, -1, 0], [0, 1, 0]], [0, 1, 2]);
  const second = instanceFace([[2, -1, 0], [4, -1, 0], [4, 1, 0]], [0, 1, 2]);
  const instances = new Float32Array(STRIDE * 2);
  instances.set(first, 0);
  instances.set(second, STRIDE);
  const bounds = buildFocusBounds(instances, new Int32Array([3, 8]));

  assert.equal(bounds.nodes.size, 2);
  assert.deepEqual(bounds.scene.center, [1, 0, 0]);
  assert.equal(bounds.scene.vertexCount, 6);
  assert(Math.abs(bounds.scene.radius - Math.sqrt(10)) < 1e-6);
});

test('walking pace is normalized by scene radius and avatar scale', () => {
  assert.equal(sceneRelativeWalkSpeed(2, 0.25), 0.5);
  assert.equal(sceneRelativeWalkSpeed(2, 0.25, 0.125), 0.0625);
  assert.equal(sceneRelativeWalkSpeed(Number.NaN, 0.25), null);
  assert.equal(sceneRelativeWalkSpeed(0, 0.25), null);
});

test('surface-walk view oracle retains pitch and rejects degenerate contact frames', () => {
  const pitch = -Math.PI / 6;
  const forward = composeSurfaceRelativeForward([0, 0, -1], [0, 1, 0], pitch);
  const basis = [1, 0, 0, 0, Math.cos(pitch), Math.sin(pitch), ...forward];
  const options = {
    active: false,
    deltaSeconds: 0,
    smoothingSeconds: 0.18,
    tangentPullFraction: 0.7,
    eyeHeight: 0.035,
    orient: true,
    captureRelativeView: false,
  };
  const first = resolveSurfaceWalkView(
    {},
    { basis },
    { outputPosition: [0, 0, 0], outputNormal: [0, 1, 0] },
    options,
  );
  assert(first);
  assert(Math.abs(first.relativePitch - pitch) < 1e-12);
  assert.deepEqual(first.eye, [0, 0.035, 0]);

  const next = resolveSurfaceWalkView(
    first,
    { basis: first.basis },
    { outputPosition: [1, 0, 0], outputNormal: [0, 0, 1] },
    { ...options, active: true, deltaSeconds: 10, smoothingSeconds: 0, tangentPullFraction: 1 },
  );
  assert(next);
  assert(Math.abs(next.relativePitch - pitch) < 1e-12);
  assert.equal(resolveSurfaceWalkView(
    first,
    { basis: first.basis },
    { outputPosition: [0, 0, 0], outputNormal: [0, 1e-10, 0] },
    options,
  ), null);
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

test('camera transport follows the exact eye and line of sight and reports local scale', () => {
  const camera = {
    eye: [2, 0, 0],
    target: [4, 0, 0],
    basis: [0, 0, 1, 0, 1, 0, 1, 0, 0],
    orbitDistance: 2,
  };
  const identity = { enabled: false };
  const inversion = { enabled: true, center: [0, 0, 0], radius: 1 };
  const transported = transportCameraAcrossSphereReflections(camera, identity, inversion);
  assert.deepEqual(transported.eye, [0.5, 0, 0]);
  assert.deepEqual(transported.target, [0.25, 0, 0]);
  assert.equal(transported.localScale, 0.25);
  assert.equal(transported.orbitDistance, 0.25);
  assert.deepEqual(transported.basis.slice(6), [-1, 0, 0]);
  assert.deepEqual(transported.basis.slice(3, 6), [0, 1, 0]);

  const restored = transportCameraAcrossSphereReflections(transported, inversion, identity);
  assert.deepEqual(restored.eye, camera.eye);
  assert.deepEqual(restored.target, camera.target);
  assert(Math.abs(restored.orbitDistance - camera.orbitDistance) < 1e-12);
  assert.deepEqual(restored.basis, camera.basis);
});

test('surface point and tangent use the differential at the contact point', () => {
  const transported = transportPointAndDirectionsAcrossSphereReflections(
    [2, 0, 0],
    [[0, 1, 0], [1, 0, 0]],
    { enabled: false },
    { enabled: true, center: [0, 0, 0], radius: 1 },
  );
  assert.deepEqual(transported.point, [0.5, 0, 0]);
  assert.deepEqual(transported.directions[0], [0, 1, 0]);
  assert.deepEqual(transported.directions[1], [-1, 0, 0]);
  assert.equal(transported.localScale, 0.25);
});

test('surface-relative view pitch survives a changing walk normal', () => {
  const pitch = Math.PI / 6;
  const initialForward = [0, Math.sin(pitch), -Math.cos(pitch)];
  const relative = decomposeSurfaceRelativeForward(
    initialForward,
    [0, 1, 0],
    [0, 0, -1],
  );
  assert.ok(relative);
  assert.ok(Math.abs(relative.pitch - pitch) < 1e-12);
  assert.deepEqual(relative.tangent, [0, 0, -1]);

  const carried = composeSurfaceRelativeForward(relative.tangent, [1, 0, 0], relative.pitch);
  assert.ok(carried);
  assert.ok(Math.abs(carried[0] - Math.sin(pitch)) < 1e-12);
  assert.ok(Math.abs(carried[1]) < 1e-12);
  assert.ok(Math.abs(carried[2] + Math.cos(pitch)) < 1e-12);
});

test('walk near plane follows tiny eye heights without exceeding the ordinary plane', () => {
  assert.equal(scaleRelativeNearPlane(1), 0.01);
  assert.ok(Math.abs(scaleRelativeNearPlane(0.035) - 0.0028) < 1e-12);
  assert.ok(Math.abs(scaleRelativeNearPlane(0.035, 0.02, 1e-8, 0.1) - 0.0035) < 1e-12);
  assert.equal(scaleRelativeNearPlane(1e-8), 1e-7);
  assert.equal(scaleRelativeNearPlane(Number.NaN), 0.01);
});

test('target-free camera transport maps the sight tangent without inventing an aim point', () => {
  const camera = {
    eye: [2, 0, 0],
    basis: [0, 0, -1, 0, 1, 0, -1, 0, 0],
    orbitDistance: 2,
  };
  const transported = transportCameraAcrossSphereReflections(
    camera,
    { enabled: false },
    { enabled: true, center: [0, 0, 0], radius: 1 },
  );
  assert.deepEqual(transported.eye, [0.5, 0, 0]);
  assert.deepEqual(transported.basis.slice(6), [1, 0, 0]);
  assert.equal(transported.orbitDistance, 0.5);
  assert.deepEqual(transported.target, [1, 0, 0]);
});

test('point-target camera transport rejects a target pole instead of changing aim mode', () => {
  const camera = {
    eye: [2, 0, 0],
    target: [0, 0, 0],
    basis: [0, 0, -1, 0, 1, 0, -1, 0, 0],
    orbitDistance: 2,
  };
  assert.equal(transportCameraAcrossSphereReflections(
    camera,
    { enabled: false },
    { enabled: true, center: [0, 0, 0], radius: 1 },
  ), null);
});

test('camera transport is stable while editing an unchanged inversion sphere', () => {
  const inversion = { enabled: true, center: [1, -2, 0.5], radius: 3 };
  const camera = {
    eye: [4, 1, 2],
    target: [4, 1, -3],
    basis: [1, 0, 0, 0, 1, 0, 0, 0, -1],
    orbitDistance: 5,
  };
  const transported = transportCameraAcrossSphereReflections(camera, inversion, inversion);
  camera.eye.forEach((value, axis) => assert(Math.abs(transported.eye[axis] - value) < 1e-12));
  camera.basis.forEach((value, axis) => {
    assert(Math.abs(transported.basis[axis] - value) < 1e-12);
  });
  assert(Math.abs(transported.orbitDistance - camera.orbitDistance) < 1e-12);
});

test('camera transport round-trips across a moving and resizing sphere', () => {
  const first = { enabled: true, center: [0, 0, 0], radius: 1 };
  const second = { enabled: true, center: [1, -0.5, 0.25], radius: 2.5 };
  const camera = {
    eye: [0.5, 1, -0.25],
    target: [0.5, 1, -3.25],
    basis: [1, 0, 0, 0, 1, 0, 0, 0, -1],
    orbitDistance: 3,
  };
  const moved = transportCameraAcrossSphereReflections(camera, first, second);
  const restored = transportCameraAcrossSphereReflections(moved, second, first);
  camera.eye.forEach((value, axis) => assert(Math.abs(restored.eye[axis] - value) < 1e-12));
  camera.basis.forEach((value, axis) => {
    assert(Math.abs(restored.basis[axis] - value) < 1e-12);
  });
  assert(Math.abs(restored.orbitDistance - camera.orbitDistance) < 1e-12);
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

test('walk anchor glide eases exactly between endpoints with a normal hop', () => {
  const start = interpolateWalkAnchorEye([0, 0, 0], [10, 0, 0], [0, 2, 0], 0, 2);
  const middle = interpolateWalkAnchorEye([0, 0, 0], [10, 0, 0], [0, 2, 0], 0.5, 2);
  const end = interpolateWalkAnchorEye([0, 0, 0], [10, 0, 0], [0, 2, 0], 1, 2);
  assert.deepEqual(start.eye, [0, 0, 0]);
  assert.deepEqual(middle.eye, [5, 2, 0]);
  assert.equal(end.eye[0], 10);
  assert(Math.abs(end.eye[1]) < 1e-12);
  assert.equal(end.eye[2], 0);
  assert.equal(start.hop, 0);
  assert(Math.abs(end.hop) < 1e-12);
  assert.equal(interpolateWalkAnchorEye([0, 0], [1, 2, 3], [0, 1, 0], 0.5), null);
});

test('walk anchor clock is virtual, cadence independent, and strictly validated', () => {
  let partitioned = { elapsedSeconds: 0 };
  for (let frame = 0; frame < 10; frame++) {
    partitioned = advanceVirtualTransitionClock(partitioned.elapsedSeconds, 1, 0.1);
  }
  const single = advanceVirtualTransitionClock(0, 1, 1);
  assert.equal(partitioned.elapsedSeconds, single.elapsedSeconds);
  assert.equal(partitioned.progress, 1);
  assert.equal(partitioned.complete, true);
  assert.deepEqual(advanceVirtualTransitionClock(0.25, 1, 0), {
    elapsedSeconds: 0.25,
    progress: 0.25,
    complete: false,
  });
  assert.equal(advanceVirtualTransitionClock(0, 0, 0.1), null);
  assert.equal(advanceVirtualTransitionClock(0, 1, -0.1), null);
  assert.equal(advanceVirtualTransitionClock(Infinity, 1, 0.1), null);
  assert.equal(advanceVirtualTransitionClock(0, 1, '0.1'), null);
  assert.equal(advanceVirtualTransitionClock(0, 1, null), null);
});

test('object focus sphere applies a stable margin and rejects invalid bounds', () => {
  assert.deepEqual(
    focusSphereFromBound({ center: [1, 2, 3], radius: 2 }),
    { center: [1, 2, 3], radius: 2.2, margin: 1.1 },
  );
  assert.equal(focusSphereFromBound({ center: [1, 2], radius: 2 }), null);
  assert.equal(focusSphereFromBound({ center: [1, 2, 3], radius: -1 }), null);
});

test('spheroidal focus enablement is independent of inversion and legacy blur modes', () => {
  assert.equal(spheroidalFocusEnabled(true, '3'), true);
  assert.equal(spheroidalFocusEnabled(true, 3), true);
  assert.equal(spheroidalFocusEnabled('1', '3'), true);
  assert.equal(spheroidalFocusEnabled(false, '3'), false);
  assert.equal(spheroidalFocusEnabled('0', '3'), false);
  assert.equal(spheroidalFocusEnabled(true, '0'), false);
  assert.equal(sharedFocusSphereActive(false, true, '3'), true);
  assert.equal(sharedFocusSphereActive(true, false, '0'), true);
  assert.equal(sharedFocusSphereActive(false, true, '0'), false);
});

test('compactified radial focus is exact at the origin, sphere, and infinity', () => {
  assert.equal(compactifiedRadialCoordinate(0, 3), 0);
  assert.equal(compactifiedRadialCoordinate(3, 3), 0.5);
  assert.equal(compactifiedRadialCoordinate(Infinity, 3), 1);
  assert.equal(compactifiedRadialCoordinate(1, 0), null);
});

test('sphere reflection complements the compactified focus coordinate', () => {
  const radius = 2;
  const distance = 8;
  const reflectedDistance = radius * radius / distance;
  const sum = compactifiedRadialCoordinate(distance, radius)
    + compactifiedRadialCoordinate(reflectedDistance, radius);
  assert(Math.abs(sum - 1) < 1e-12);
});

test('spheroidal defocus is symmetric and aperture-normalized', () => {
  assert.equal(spheroidalDefocus(0.5, 0.5, 0.1), 0);
  assert(Math.abs(spheroidalDefocus(0.4, 0.5, 0.1) - 0.5) < 1e-12);
  assert(Math.abs(spheroidalDefocus(0.6, 0.5, 0.1) - 0.5) < 1e-12);
  assert.equal(spheroidalDefocus(0.5, 0.5, 0), null);
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
