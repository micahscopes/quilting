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
export function buildNodeFocusRecords(
  instances,
  faceNodes,
  stride = HYPERSCOPE_INSTANCE_STRIDE,
) {
  const records = new Map();
  if (!instances || !faceNodes || stride < 12) return records;
  const faceCount = Math.min(faceNodes.length, Math.floor(instances.length / stride));

  for (let face = 0; face < faceCount; face++) {
    const node = Number(faceNodes[face]);
    if (!Number.isInteger(node) || node < 0) continue;
    let record = records.get(node);
    if (!record) {
      record = {
        node,
        min: [Infinity, Infinity, Infinity],
        max: [-Infinity, -Infinity, -Infinity],
        vertices: new Map(),
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
      if (!record.vertices.has(vertexKey)) record.vertices.set(vertexKey, [x, y, z]);
      record.min[0] = Math.min(record.min[0], x);
      record.min[1] = Math.min(record.min[1], y);
      record.min[2] = Math.min(record.min[2], z);
      record.max[0] = Math.max(record.max[0], x);
      record.max[1] = Math.max(record.max[1], y);
      record.max[2] = Math.max(record.max[2], z);
    }
  }

  for (const [node, record] of records) {
    if (!record.vertices.size) {
      records.delete(node);
      continue;
    }
    const center = [
      (record.min[0] + record.max[0]) * 0.5,
      (record.min[1] + record.max[1]) * 0.5,
      (record.min[2] + record.max[2]) * 0.5,
    ];
    let radiusSquared = 0;
    for (const point of record.vertices.values()) {
      const dx = point[0] - center[0];
      const dy = point[1] - center[1];
      const dz = point[2] - center[2];
      radiusSquared = Math.max(radiusSquared, dx * dx + dy * dy + dz * dz);
    }
    record.center = center;
    record.radius = Math.sqrt(radiusSquared);
    record.vertexCount = record.vertices.size;
    delete record.min;
    delete record.max;
  }
  return records;
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

/** Quintic easing with zero velocity and acceleration at both endpoints. */
export function smootherstep01(value) {
  const t = Math.max(0, Math.min(1, Number(value) || 0));
  return t * t * t * (t * (t * 6 - 15) + 10);
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
