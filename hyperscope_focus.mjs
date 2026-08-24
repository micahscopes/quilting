// Object-focus and conformal-navigation helpers shared by the Hyperscope UI.
// Keep this module dependency-free so its geometry and gesture-adjacent policy
// can be exercised by Node without constructing a browser or WebGL context.

export const HYPERSCOPE_INSTANCE_STRIDE = 52;
const MOBIUS_EPSILON = 1e-12;

function finitePoint(x, y, z) {
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
}

/**
 * Build one normalized source-space bounding sphere per stable glTF node.
 *
 * The instance stream stores three `[vertex_id, x, y, z]` records per face.
 * Vertex IDs deduplicate shared corners without requiring another mesh export.
 */
export function buildFocusBounds(
  instances,
  faceNodes,
  stride = HYPERSCOPE_INSTANCE_STRIDE,
) {
  const records = new Map();
  if (!instances || !faceNodes || stride < 12) {
    return { nodes: records, scene: null };
  }
  const faceCount = Math.min(faceNodes.length, Math.floor(instances.length / stride));
  const sceneScratch = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
    vertexCount: 0,
  };

  for (let face = 0; face < faceCount; face++) {
    const node = Number(faceNodes[face]);
    if (!Number.isInteger(node) || node < 0) continue;
    let record = records.get(node);
    if (!record) {
      record = {
        node,
        min: [Infinity, Infinity, Infinity],
        max: [-Infinity, -Infinity, -Infinity],
        vertexIds: new Set(),
      };
      records.set(node, record);
    }

    const base = face * stride;
    for (let corner = 0; corner < 3; corner++) {
      const offset = base + corner * 4;
      const x = Number(instances[offset + 1]);
      const y = Number(instances[offset + 2]);
      const z = Number(instances[offset + 3]);
      if (!finitePoint(x, y, z)) continue;
      const rawVertex = Number(instances[offset]);
      const vertexKey = Number.isFinite(rawVertex)
        ? rawVertex
        : face * 3 + corner;
      record.vertexIds.add(vertexKey);
      record.min[0] = Math.min(record.min[0], x);
      record.min[1] = Math.min(record.min[1], y);
      record.min[2] = Math.min(record.min[2], z);
      record.max[0] = Math.max(record.max[0], x);
      record.max[1] = Math.max(record.max[1], y);
      record.max[2] = Math.max(record.max[2], z);
      sceneScratch.min[0] = Math.min(sceneScratch.min[0], x);
      sceneScratch.min[1] = Math.min(sceneScratch.min[1], y);
      sceneScratch.min[2] = Math.min(sceneScratch.min[2], z);
      sceneScratch.max[0] = Math.max(sceneScratch.max[0], x);
      sceneScratch.max[1] = Math.max(sceneScratch.max[1], y);
      sceneScratch.max[2] = Math.max(sceneScratch.max[2], z);
    }
  }

  for (const [node, record] of records) {
    if (!record.vertexIds.size) {
      records.delete(node);
      continue;
    }
    const center = [
      (record.min[0] + record.max[0]) * 0.5,
      (record.min[1] + record.max[1]) * 0.5,
      (record.min[2] + record.max[2]) * 0.5,
    ];
    record.center = center;
    record.radiusSquared = 0;
    record.vertexCount = record.vertexIds.size;
    sceneScratch.vertexCount += record.vertexCount;
  }

  const scene = sceneScratch.vertexCount > 0 ? {
    center: [
      (sceneScratch.min[0] + sceneScratch.max[0]) * 0.5,
      (sceneScratch.min[1] + sceneScratch.max[1]) * 0.5,
      (sceneScratch.min[2] + sceneScratch.max[2]) * 0.5,
    ],
    radiusSquared: 0,
    vertexCount: sceneScratch.vertexCount,
  } : null;

  // A second linear scan is dramatically cheaper at chess scale than
  // retaining one JS coordinate array per unique vertex in every node Map.
  // Duplicate corners cannot change a maximum, so no coordinate cache is
  // needed to obtain the exact same AABB-centred sphere.
  for (let face = 0; face < faceCount; face++) {
    const record = records.get(Number(faceNodes[face]));
    if (!record) continue;
    const base = face * stride;
    for (let corner = 0; corner < 3; corner++) {
      const offset = base + corner * 4;
      const x = Number(instances[offset + 1]);
      const y = Number(instances[offset + 2]);
      const z = Number(instances[offset + 3]);
      if (!finitePoint(x, y, z)) continue;
      const dx = x - record.center[0];
      const dy = y - record.center[1];
      const dz = z - record.center[2];
      record.radiusSquared = Math.max(
        record.radiusSquared,
        dx * dx + dy * dy + dz * dz,
      );
      if (scene) {
        const sceneDx = x - scene.center[0];
        const sceneDy = y - scene.center[1];
        const sceneDz = z - scene.center[2];
        scene.radiusSquared = Math.max(
          scene.radiusSquared,
          sceneDx * sceneDx + sceneDy * sceneDy + sceneDz * sceneDz,
        );
      }
    }
  }

  for (const record of records.values()) {
    record.radius = Math.sqrt(record.radiusSquared);
    delete record.min;
    delete record.max;
    delete record.radiusSquared;
    delete record.vertexIds;
  }
  if (scene) {
    scene.radius = Math.sqrt(scene.radiusSquared);
    delete scene.radiusSquared;
  }
  return { nodes: records, scene };
}

/** Backward-compatible node-only view of the shared source-bound pass. */
export function buildNodeFocusRecords(
  instances,
  faceNodes,
  stride = HYPERSCOPE_INSTANCE_STRIDE,
) {
  return buildFocusBounds(instances, faceNodes, stride).nodes;
}

/** Convert a scene-relative walking pace to source/output-chart units per second. */
export function sceneRelativeWalkSpeed(sceneRadius, radiiPerSecond, bodyScale = 1) {
  if (![sceneRadius, radiiPerSecond, bodyScale].every(Number.isFinite)
      || !(sceneRadius > 0) || radiiPerSecond < 0 || !(bodyScale > 0)) {
    return null;
  }
  return sceneRadius * radiiPerSecond * bodyScale;
}

/** Return the normalized source-space centroid of a source face. */
export function faceSourceCentroid(
  instances,
  face,
  stride = HYPERSCOPE_INSTANCE_STRIDE,
) {
  if (!instances || !Number.isInteger(face) || face < 0) return null;
  const base = face * stride;
  if (base + 12 > instances.length) return null;
  const center = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const offset = base + corner * 4;
    const x = Number(instances[offset + 1]);
    const y = Number(instances[offset + 2]);
    const z = Number(instances[offset + 3]);
    if (!finitePoint(x, y, z)) return null;
    center[0] += x / 3;
    center[1] += y / 3;
    center[2] += z / 3;
  }
  return center;
}

function qmul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function qadd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
}

function qsub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
}

function qnormSquared(q) {
  return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
}

function qinverse(q) {
  const normSquared = qnormSquared(q);
  if (!Number.isFinite(normSquared) || normSquared <= MOBIUS_EPSILON) return null;
  const inverseNorm = 1 / normSquared;
  return [
    q[0] * inverseNorm,
    -q[1] * inverseNorm,
    -q[2] * inverseNorm,
    -q[3] * inverseNorm,
  ];
}

function mobiusParts(matrix) {
  if (!matrix || matrix.length < 16) return null;
  return [
    Array.from(matrix.slice(0, 4), Number),
    Array.from(matrix.slice(4, 8), Number),
    Array.from(matrix.slice(8, 12), Number),
    Array.from(matrix.slice(12, 16), Number),
  ];
}

/** Apply the renderer's quaternion Möbius matrix to an ordinary 3-D point. */
export function applyMobiusPoint(matrix, point) {
  const parts = mobiusParts(matrix);
  if (!parts || !point || !finitePoint(point[0], point[1], point[2])) return null;
  const [a, b, c, d] = parts;
  const q = [0, Number(point[0]), Number(point[1]), Number(point[2])];
  const denominator = qadd(qmul(c, q), d);
  const inverse = qinverse(denominator);
  if (!inverse) return null;
  const mapped = qmul(qadd(qmul(a, q), b), inverse);
  if (!finitePoint(mapped[1], mapped[2], mapped[3])) return null;
  return [mapped[1], mapped[2], mapped[3]];
}

/** Local isotropic length scale of a quaternion Möbius map at a 3-D point. */
export function mobiusConformalScaleAt(matrix, point) {
  const parts = mobiusParts(matrix);
  if (!parts || !point || !finitePoint(point[0], point[1], point[2])) return null;
  const [a, b, c, d] = parts;
  const q = [0, Number(point[0]), Number(point[1]), Number(point[2])];
  const denominator = qadd(qmul(c, q), d);
  const denominatorNormSquared = qnormSquared(denominator);
  const inverse = qinverse(denominator);
  if (!inverse) return null;
  const mapped = qmul(qadd(qmul(a, q), b), inverse);
  const left = qsub(a, qmul(mapped, c));
  const scale = Math.sqrt(qnormSquared(left) / denominatorNormSquared);
  return Number.isFinite(scale) ? scale : null;
}

function reflectionState(state) {
  if (!state?.enabled) return { enabled: false, center: [0, 0, 0], radius: 1 };
  const center = Array.from(state.center || [], Number);
  const radius = Number(state.radius);
  if (center.length !== 3 || !finitePoint(center[0], center[1], center[2])
      || !(radius > 0) || !Number.isFinite(radius)) return null;
  return { enabled: true, center, radius };
}

function reflectPointAndFrame(point, directions, state) {
  if (!state.enabled) {
    return {
      point: Array.from(point, Number),
      directions: directions.map(direction => Array.from(direction, Number)),
      scale: 1,
    };
  }
  const delta = point.map((coordinate, axis) => coordinate - state.center[axis]);
  const distanceSquared = delta.reduce((sum, coordinate) => sum + coordinate * coordinate, 0);
  if (!Number.isFinite(distanceSquared) || distanceSquared <= MOBIUS_EPSILON) return null;
  const inverseDistance = 1 / Math.sqrt(distanceSquared);
  const normal = delta.map(coordinate => coordinate * inverseDistance);
  const scale = state.radius * state.radius / distanceSquared;
  return {
    point: delta.map((coordinate, axis) => state.center[axis] + scale * coordinate),
    directions: directions.map(direction => {
      const projection = direction.reduce(
        (sum, coordinate, axis) => sum + coordinate * normal[axis],
        0,
      );
      return direction.map((coordinate, axis) => coordinate - 2 * projection * normal[axis]);
    }),
    scale,
  };
}

/** Transport a point and attached directions through F_next o inverse(F_previous). */
export function transportPointAndDirectionsAcrossSphereReflections(
  point,
  directions,
  previous,
  next,
) {
  const before = reflectionState(previous);
  const after = reflectionState(next);
  const sourcePoint = Array.from(point || [], Number);
  const sourceDirections = Array.from(directions || [], direction => Array.from(direction || [], Number));
  if (!before || !after || sourcePoint.length !== 3 || !finitePoint(...sourcePoint)
      || sourceDirections.some(direction => direction.length !== 3 || !finitePoint(...direction))) {
    return null;
  }
  const unmapped = reflectPointAndFrame(sourcePoint, sourceDirections, before);
  if (!unmapped) return null;
  const remapped = reflectPointAndFrame(unmapped.point, unmapped.directions, after);
  if (!remapped) return null;
  return {
    point: remapped.point,
    directions: remapped.directions,
    localScale: unmapped.scale * remapped.scale,
  };
}

function normalizedDirection(direction) {
  const values = Array.from(direction || [], Number);
  if (values.length !== 3 || !finitePoint(...values)) return null;
  const length = Math.hypot(...values);
  if (!(length > MOBIUS_EPSILON) || !Number.isFinite(length)) return null;
  return values.map(value => value / length);
}

function surfaceFrameDirection(direction) {
  const values = Array.from(direction || [], Number);
  if (values.length !== 3 || !finitePoint(...values)) return null;
  const length = Math.hypot(...values);
  if (!(length > 1e-8) || !Number.isFinite(length)) return null;
  return values.map(value => value / length);
}

/** Normalize a surface-frame direction using the incumbent walk tolerance. */
export function surfaceNormalize(direction) {
  return surfaceFrameDirection(direction);
}

function surfaceFrameCross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

/** Project a direction into a normalized local surface tangent. */
export function surfaceProjectTangent(direction, normal) {
  const value = Array.from(direction || [], Number);
  const surfaceNormal = surfaceFrameDirection(normal);
  if (value.length !== 3 || !finitePoint(...value) || !surfaceNormal) return null;
  const normalPart = value.reduce(
    (sum, coordinate, axis) => sum + coordinate * surfaceNormal[axis],
    0,
  );
  return surfaceFrameDirection(value.map(
    (coordinate, axis) => coordinate - surfaceNormal[axis] * normalPart,
  ));
}

/** Spherical direction response used by the incumbent surface-walk camera. */
export function surfaceSmoothDirection(from, to, amount) {
  const destination = surfaceFrameDirection(to);
  if (!destination || !Number.isFinite(amount)) return null;
  if (!from || amount >= 1) return destination;
  const source = surfaceFrameDirection(from);
  if (!source) return null;
  if (amount <= 0) return source;
  const cosine = Math.max(-1, Math.min(1, source.reduce(
    (sum, value, axis) => sum + value * destination[axis],
    0,
  )));
  if (cosine > 0.9995) {
    return surfaceFrameDirection(source.map(
      (value, axis) => value + (destination[axis] - value) * amount,
    )) || destination;
  }
  if (cosine < -0.9995) {
    const orthogonal = surfaceFrameDirection(surfaceFrameCross(source, [1, 0, 0]))
      || surfaceFrameDirection(surfaceFrameCross(source, [0, 1, 0]));
    if (!orthogonal) return destination;
    return surfaceFrameDirection(source.map(
      (value, axis) => value * Math.cos(Math.PI * amount)
        + orthogonal[axis] * Math.sin(Math.PI * amount),
    )) || destination;
  }
  const angle = Math.acos(cosine);
  const sine = Math.sin(angle);
  if (Math.abs(sine) < 1e-8) return destination;
  const firstWeight = Math.sin((1 - amount) * angle) / sine;
  const secondWeight = Math.sin(amount * angle) / sine;
  return surfaceFrameDirection(source.map(
    (value, axis) => value * firstWeight + destination[axis] * secondWeight,
  )) || destination;
}

/**
 * Pure incumbent oracle for one animated surface-walk camera response.
 * Acquisition, topology walking, and camera application stay outside.
 */
export function resolveSurfaceWalkView(state, camera, contact, options) {
  const basis = Array.from(camera?.basis || [], Number);
  const targetPosition = Array.from(contact?.outputPosition || [], Number);
  const targetNormal = surfaceFrameDirection(contact?.outputNormal);
  const deltaSeconds = Number(options?.deltaSeconds);
  const smoothingSeconds = Number(options?.smoothingSeconds);
  const tangentPullFraction = Number(options?.tangentPullFraction);
  const eyeHeight = Number(options?.eyeHeight);
  if (basis.length !== 9 || !basis.every(Number.isFinite)
      || targetPosition.length !== 3 || !targetPosition.every(Number.isFinite)
      || !targetNormal || !Number.isFinite(deltaSeconds) || deltaSeconds < 0
      || !Number.isFinite(smoothingSeconds) || smoothingSeconds < 0
      || !Number.isFinite(tangentPullFraction) || tangentPullFraction < 0
      || tangentPullFraction > 1 || !(eyeHeight > 0) || !Number.isFinite(eyeHeight)) {
    return null;
  }
  const active = Boolean(options?.active);
  const priorPosition = state?.filteredPosition == null
    ? null : Array.from(state.filteredPosition, Number);
  const priorNormal = state?.filteredNormal == null
    ? null : surfaceFrameDirection(state.filteredNormal);
  const priorTangent = state?.tangentForward == null
    ? null : surfaceFrameDirection(state.tangentForward);
  if ((priorPosition && (priorPosition.length !== 3 || !priorPosition.every(Number.isFinite)))
      || (state?.filteredNormal != null && !priorNormal)
      || (state?.tangentForward != null && !priorTangent)) return null;

  const filterAmount = !active || !(deltaSeconds > 0) || smoothingSeconds <= 0
    ? 1 : 1 - Math.exp(-deltaSeconds / smoothingSeconds);
  const filteredPosition = priorPosition
    ? priorPosition.map(
      (value, axis) => value + (targetPosition[axis] - value) * filterAmount,
    )
    : targetPosition;
  const filteredNormal = surfaceSmoothDirection(priorNormal, targetNormal, filterAmount);
  if (!filteredNormal) return null;
  const eye = filteredPosition.map(
    (value, axis) => value + filteredNormal[axis] * eyeHeight,
  );

  let tangentForward = priorTangent;
  let relativePitch = Number.isFinite(state?.relativePitch) ? state.relativePitch : null;
  let outputBasis = basis.slice();
  if (options?.orient) {
    const oldForward = basis.slice(6, 9);
    const captureRelativeView = !active || Boolean(options?.captureRelativeView)
      || !Number.isFinite(relativePitch) || !tangentForward;
    const relativeView = captureRelativeView
      ? decomposeSurfaceRelativeForward(
        oldForward,
        filteredNormal,
        tangentForward || basis.slice(3, 6),
      )
      : null;
    tangentForward = relativeView?.tangent
      || surfaceProjectTangent(tangentForward, filteredNormal)
      || surfaceProjectTangent(basis.slice(3, 6), filteredNormal)
      || surfaceFrameDirection(surfaceFrameCross(filteredNormal, [1, 0, 0]))
      || surfaceFrameDirection(surfaceFrameCross(filteredNormal, [0, 0, 1]));
    if (!tangentForward) return null;
    if (relativeView) relativePitch = relativeView.pitch;
    const targetForward = composeSurfaceRelativeForward(
      tangentForward,
      filteredNormal,
      Number.isFinite(relativePitch) ? relativePitch : 0,
    ) || tangentForward;
    const framePullAmount = !active || !(deltaSeconds > 0)
      ? 1 : 1 - Math.exp(-tangentPullFraction * 8 * deltaSeconds);
    const forward = surfaceSmoothDirection(oldForward, targetForward, framePullAmount);
    const right = forward && surfaceFrameDirection(surfaceFrameCross(forward, filteredNormal));
    const up = right && surfaceFrameDirection(surfaceFrameCross(right, forward));
    if (!right || !up) return null;
    outputBasis = [...right, ...up, ...forward];
  }
  return {
    filteredPosition,
    filteredNormal,
    tangentForward,
    relativePitch: Number.isFinite(relativePitch) ? relativePitch : null,
    eye,
    basis: outputBasis,
  };
}

/** Split a view direction into a surface tangent heading and relative pitch. */
export function decomposeSurfaceRelativeForward(forward, normal, tangentHint = null) {
  const view = normalizedDirection(forward);
  const surfaceNormal = normalizedDirection(normal);
  if (!view || !surfaceNormal) return null;
  const pitchSine = Math.max(-1, Math.min(1, view.reduce(
    (sum, value, axis) => sum + value * surfaceNormal[axis],
    0,
  )));
  let tangent = normalizedDirection(
    view.map((value, axis) => value - surfaceNormal[axis] * pitchSine),
  );
  if (!tangent && tangentHint) {
    const hint = normalizedDirection(tangentHint);
    if (hint) {
      const projection = hint.reduce(
        (sum, value, axis) => sum + value * surfaceNormal[axis],
        0,
      );
      tangent = normalizedDirection(
        hint.map((value, axis) => value - surfaceNormal[axis] * projection),
      );
    }
  }
  if (!tangent) return null;
  return { tangent, pitch: Math.asin(pitchSine) };
}

/** Rebuild a view direction at the same pitch relative to a new surface frame. */
export function composeSurfaceRelativeForward(tangent, normal, pitch) {
  const surfaceNormal = normalizedDirection(normal);
  if (!surfaceNormal || !Number.isFinite(pitch)) return null;
  const heading = normalizedDirection(tangent);
  if (!heading) return null;
  const projected = decomposeSurfaceRelativeForward(heading, surfaceNormal);
  if (!projected) return null;
  const limitedPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));
  const tangentScale = Math.cos(limitedPitch);
  const normalScale = Math.sin(limitedPitch);
  return normalizedDirection(projected.tangent.map(
    (value, axis) => value * tangentScale + surfaceNormal[axis] * normalScale,
  ));
}

/** Keep the walk near plane below the eye-to-surface offset at every body scale. */
export function scaleRelativeNearPlane(
  eyeHeight,
  defaultNear = 0.01,
  minimumNear = 1e-7,
) {
  const height = Math.abs(Number(eyeHeight));
  const fallback = Number(defaultNear);
  const minimum = Number(minimumNear);
  if (!(fallback > 0) || !(minimum > 0)) return null;
  if (!(height > 0) || !Number.isFinite(height)) return fallback;
  return Math.max(minimum, Math.min(fallback, height * 0.08));
}

function normalizedCameraBasis(forward, up) {
  let forwardLength = Math.hypot(...forward);
  if (!(forwardLength > 1e-12) || !Number.isFinite(forwardLength)) return null;
  const f = forward.map(value => value / forwardLength);
  let right = [
    f[1] * up[2] - f[2] * up[1],
    f[2] * up[0] - f[0] * up[2],
    f[0] * up[1] - f[1] * up[0],
  ];
  const rightLength = Math.hypot(...right);
  if (!(rightLength > 1e-12) || !Number.isFinite(rightLength)) return null;
  right = right.map(value => value / rightLength);
  const u = [
    right[1] * f[2] - right[2] * f[1],
    right[2] * f[0] - right[0] * f[2],
    right[0] * f[1] - right[1] * f[0],
  ];
  return [...right, ...u, ...f];
}

function transportUpAlongSightline(previousForward, nextForward, previousUp) {
  const oldLength = Math.hypot(...previousForward);
  const nextLength = Math.hypot(...nextForward);
  if (!(oldLength > 1e-12) || !(nextLength > 1e-12)) return null;
  const from = previousForward.map(value => value / oldLength);
  const to = nextForward.map(value => value / nextLength);
  const cosine = Math.max(-1, Math.min(1,
    from.reduce((sum, value, axis) => sum + value * to[axis], 0)));
  if (cosine > 1 - 1e-12) return Array.from(previousUp, Number);
  if (cosine < -1 + 1e-12) {
    // The previous up axis is perpendicular to the sightline, so a half-turn
    // around it maps forward to backward while retaining roll.
    return Array.from(previousUp, Number);
  }
  const cross = [
    from[1] * to[2] - from[2] * to[1],
    from[2] * to[0] - from[0] * to[2],
    from[0] * to[1] - from[1] * to[0],
  ];
  const firstCross = [
    cross[1] * previousUp[2] - cross[2] * previousUp[1],
    cross[2] * previousUp[0] - cross[0] * previousUp[2],
    cross[0] * previousUp[1] - cross[1] * previousUp[0],
  ];
  const secondCross = [
    cross[1] * firstCross[2] - cross[2] * firstCross[1],
    cross[2] * firstCross[0] - cross[0] * firstCross[2],
    cross[0] * firstCross[1] - cross[1] * firstCross[0],
  ];
  return previousUp.map(
    (value, axis) => value + firstCross[axis] + secondCross[axis] / (1 + cosine),
  );
}

/**
 * Transport a camera through F_next o inverse(F_previous).
 *
 * A sphere reflection is self-inverse. Its differential is a positive scale
 * times a Householder reflection. The eye and optional look-at target are
 * transported exactly; roll follows the shortest sightline rotation. A camera
 * without a target falls back to the exact local conformal differential.
 */
export function transportCameraAcrossSphereReflections(camera, previous, next) {
  const before = reflectionState(previous);
  const after = reflectionState(next);
  const eye = Array.from(camera?.eye || [], Number);
  const basis = Array.from(camera?.basis || [], Number);
  const orbitDistance = Number(camera?.orbitDistance);
  const requestedTarget = camera?.target == null
    ? null
    : Array.from(camera.target, Number);
  if (!before || !after || eye.length !== 3 || !finitePoint(...eye)
      || basis.length !== 9 || !basis.every(Number.isFinite)
      || (requestedTarget && (requestedTarget.length !== 3 || !finitePoint(...requestedTarget)))
      || !(orbitDistance > 0) || !Number.isFinite(orbitDistance)) return null;

  const unmap = reflectPointAndFrame(
    eye,
    [basis.slice(3, 6), basis.slice(6, 9)],
    before,
  );
  if (!unmap) return null;
  const remap = reflectPointAndFrame(unmap.point, unmap.directions, after);
  if (!remap) return null;
  const scale = unmap.scale * remap.scale;
  if (!(scale > 0) || !Number.isFinite(scale)) return null;

  let target = null;
  if (requestedTarget) {
    const targetUnmap = reflectPointAndFrame(requestedTarget, [], before);
    const targetRemap = targetUnmap
      ? reflectPointAndFrame(targetUnmap.point, [], after)
      : null;
    // Point-target transport is an atomic semantic contract. Reaching the
    // target pole must reject the chart edit instead of silently switching to
    // free-tangent transport and changing camera meaning.
    if (!targetRemap) return null;
    target = targetRemap.point;
  }
  let transportedDistance = orbitDistance * scale;
  let transportedForward = remap.directions[1];
  let transportedUp = remap.directions[0];
  if (target) {
    transportedForward = target.map((coordinate, axis) => coordinate - remap.point[axis]);
    transportedDistance = Math.hypot(...transportedForward);
    if (!(transportedDistance > 1e-12) || !Number.isFinite(transportedDistance)) return null;
    transportedUp = transportUpAlongSightline(
      basis.slice(6, 9),
      transportedForward,
      basis.slice(3, 6),
    );
  }
  const transportedBasis = transportedUp
    ? normalizedCameraBasis(transportedForward, transportedUp)
    : null;
  if (!transportedBasis) return null;
  if (!target) {
    target = remap.point.map(
      (coordinate, axis) => coordinate + transportedBasis[axis + 6] * transportedDistance,
    );
  }
  return {
    eye: remap.point,
    target,
    basis: transportedBasis,
    orbitDistance: transportedDistance,
    localScale: scale,
  };
}

/** Quintic easing with zero velocity and acceleration at both endpoints. */
export function smootherstep01(value) {
  const t = Math.max(0, Math.min(1, Number(value) || 0));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * Advance a transition on an explicit virtual-time delta.
 *
 * Keeping this independent of `performance.now()` makes camera glides
 * deterministic under replay and gives the JavaScript oracle and Rust owner
 * one clock during authority migration.
 */
export function advanceVirtualTransitionClock(elapsedSeconds, durationSeconds, deltaSeconds) {
  if (typeof elapsedSeconds !== 'number'
      || typeof durationSeconds !== 'number'
      || typeof deltaSeconds !== 'number') {
    return null;
  }
  const elapsed = elapsedSeconds;
  const duration = durationSeconds;
  const delta = deltaSeconds;
  if (!Number.isFinite(elapsed) || elapsed < 0
      || !Number.isFinite(duration) || duration <= 0
      || !Number.isFinite(delta) || delta < 0) {
    return null;
  }
  const accumulated = Math.min(duration, elapsed + delta);
  const progress = accumulated / duration;
  // Match the Rust transition owner's endpoint rule. Decimal frame cadences
  // such as ten 0.1-second ticks must land on the same terminal state as one
  // 1-second tick instead of leaving an extra one-frame sliver.
  const complete = progress >= 1 - 1e-12;
  const nextElapsed = complete ? duration : accumulated;
  return {
    elapsedSeconds: nextElapsed,
    progress: complete ? 1 : progress,
    complete,
  };
}

/**
 * Glide an eye point to a surface anchor with a minimum-jerk quintic ease and
 * a gentle hop along the destination surface normal.
 */
export function interpolateWalkAnchorEye(
  startEye,
  targetEye,
  targetNormal,
  progress,
  hopHeight = 0,
) {
  const start = Array.from(startEye || [], Number);
  const target = Array.from(targetEye || [], Number);
  const normal = normalizedDirection(targetNormal);
  if (start.length !== 3 || target.length !== 3
      || !finitePoint(...start) || !finitePoint(...target) || !normal) {
    return null;
  }
  const t = smootherstep01(progress);
  const safeHopHeight = Number.isFinite(Number(hopHeight))
    ? Math.max(0, Number(hopHeight))
    : 0;
  const hop = Math.sin(Math.PI * t) * safeHopHeight;
  return {
    eye: start.map((value, axis) =>
      value + (target[axis] - value) * t + normal[axis] * hop),
    easedProgress: t,
    hop,
  };
}

/** Interpolate sphere center linearly and positive radius logarithmically. */
export function interpolateSphereFit(start, target, progress) {
  const t = smootherstep01(progress);
  const startRadius = Math.max(Number(start.radius) || 0, 1e-4);
  const targetRadius = Math.max(Number(target.radius) || 0, 1e-4);
  return {
    center: [0, 1, 2].map(axis =>
      Number(start.center[axis]) + (Number(target.center[axis]) - Number(start.center[axis])) * t),
    radius: Math.exp(Math.log(startRadius) + (Math.log(targetRadius) - Math.log(startRadius)) * t),
  };
}

/** Build the persistent focus/inversion sphere around an object bound. */
export function focusSphereFromBound(bound, margin = 1.1, minRadius = 0.02) {
  const center = Array.from(bound?.center || [], Number);
  if (center.length !== 3 || !finitePoint(center[0], center[1], center[2])) return null;
  const sourceRadius = Number(bound?.radius);
  if (!(sourceRadius >= 0) || !Number.isFinite(sourceRadius)) return null;
  const safeMargin = Math.max(Number(margin) || 0, 1);
  return {
    center,
    radius: Math.max(sourceRadius * safeMargin, Math.max(Number(minRadius) || 0, 1e-4)),
    margin: safeMargin,
  };
}

/**
 * The shared spheroidal field is a semantic focus effect, not a synonym for
 * the legacy fuzzy post-process toggle or for the existence of its sphere.
 */
export function spheroidalFocusEnabled(postprocessEnabled, mode) {
  const enabled = postprocessEnabled === true
    || postprocessEnabled === 1
    || postprocessEnabled === '1';
  return enabled && String(mode) === '3';
}

/** Restore whether the retained shared sphere is interactively active. */
export function sharedFocusSphereActive(inversionEnabled, postprocessEnabled, mode) {
  const inversion = inversionEnabled === true
    || inversionEnabled === 1
    || inversionEnabled === '1';
  return inversion || spheroidalFocusEnabled(postprocessEnabled, mode);
}

/**
 * Exact radial coordinate of the round S3 compactification induced by a
 * sphere: origin=0, sphere=1/2, and the pole at infinity=1.
 *
 * Sphere reflection sends distance d to radius^2/d, and therefore sends this
 * coordinate u to 1-u exactly (away from the origin/infinity endpoints).
 */
export function compactifiedRadialCoordinate(distance, radius) {
  const d = Math.max(Number(distance) || 0, 0);
  const r = Number(radius);
  if (!(r > 0) || !Number.isFinite(r)) return null;
  if (!Number.isFinite(d)) return 1;
  return (2 / Math.PI) * Math.atan(d / r);
}

/** Smooth circle-of-confusion response around a spheroidal focal shell. */
export function spheroidalDefocus(coordinate, focus, angularAperture) {
  const u = Number(coordinate);
  const focalShell = Number(focus);
  const aperture = Number(angularAperture);
  if (![u, focalShell, aperture].every(Number.isFinite) || aperture <= 0) return null;
  const coc = Math.abs(u - focalShell) / aperture;
  return coc / (1 + coc);
}

/** Distinguish an intentional primary-button pick from the orbit gesture. */
export function isPrimarySelectionClick(button, dragDistance, threshold = 4) {
  const distance = Number(dragDistance);
  return Number(button) === 0
    && Number.isFinite(distance)
    && distance <= Math.max(Number(threshold) || 0, 0);
}

/**
 * Radius editing while object-anchored changes only the margin around the
 * object. It cannot shrink through the selected object or grow without bound.
 */
export function scaleAnchoredFocusRadius(
  boundRadius,
  radius,
  input,
  sensitivity,
  deltaSeconds,
  minMargin = 1,
  maxMargin = 4,
) {
  const base = Math.max(Number(boundRadius) || 0, 1e-4);
  const lowerMargin = Math.max(Number(minMargin) || 0, 1);
  const upperMargin = Math.max(Number(maxMargin) || 0, lowerMargin);
  return scaleRadiusMultiplicatively(
    radius,
    input,
    sensitivity,
    deltaSeconds,
    0.6,
    base * lowerMargin,
    base * upperMargin,
  );
}

/**
 * Blender-style perspective pan speed in world units per second per unit input.
 * The reference depth is captured when motion begins so speed cannot feed back
 * while the camera is moving.
 */
export function perspectiveNavigationSpeed(
  referenceDepth,
  viewportHeight,
  fovY = Math.PI / 3,
  pixelsPerSecond = 600,
) {
  const depth = Math.max(Math.abs(Number(referenceDepth) || 0), 1e-3);
  const height = Math.max(Number(viewportHeight) || 0, 1);
  return 2 * depth * Math.tan(fovY * 0.5) * pixelsPerSecond / height;
}

/**
 * Screen-relative navigation speed capped by the selected object's diameter.
 *
 * Depth alone is appropriate for an unbounded scene, but it makes a tiny
 * object at that depth traversable in a fraction of a gesture. The cap keeps
 * motion proportional to the current focus scale without accelerating large
 * objects beyond the familiar screen-space rate.
 */
export function focusRelativeNavigationSpeed(
  referenceDepth,
  viewportHeight,
  focusRadius = null,
  fovY = Math.PI / 3,
  pixelsPerSecond = 600,
  focusDiametersPerSecond = 1,
) {
  const screenSpeed = perspectiveNavigationSpeed(
    referenceDepth,
    viewportHeight,
    fovY,
    pixelsPerSecond,
  );
  const radius = Number(focusRadius);
  if (!(radius > 0) || !Number.isFinite(radius)) return screenSpeed;
  const diameterSpeed = 2 * radius * Math.max(Number(focusDiametersPerSecond) || 0, 0);
  return Math.min(screenSpeed, diameterSpeed);
}

/** Scale-independent radius editing: equal puck motion produces equal ratios. */
export function scaleRadiusMultiplicatively(
  radius,
  input,
  sensitivity,
  deltaSeconds,
  rate = 0.6,
  minRadius = 0.011,
  maxRadius = 5,
) {
  const lower = Math.max(Number(minRadius) || 0, 1e-6);
  const upper = Math.max(Number(maxRadius) || 0, lower);
  const current = Math.min(Math.max(Number(radius) || lower, lower), upper);
  const exponent = (Number(input) || 0)
    * (Number(sensitivity) || 0)
    * Math.max(Number(deltaSeconds) || 0, 0)
    * (Number(rate) || 0);
  return Math.min(Math.max(current * Math.exp(exponent), lower), upper);
}

/** Camera distance that contains a sphere in the narrower perspective axis. */
export function framedSphereDistance(
  radius,
  aspect,
  fovY = Math.PI / 3,
  margin = 1.15,
) {
  const safeRadius = Math.max(Number(radius) || 0, 1e-4);
  const safeAspect = Math.max(Number(aspect) || 0, 1e-4);
  const halfY = Math.max(Math.min(Number(fovY) * 0.5, Math.PI * 0.49), 1e-4);
  const halfX = Math.atan(Math.tan(halfY) * safeAspect);
  const limitingHalfAngle = Math.min(halfX, halfY);
  return safeRadius * Math.max(Number(margin) || 0, 1) / Math.sin(limitingHalfAngle);
}
