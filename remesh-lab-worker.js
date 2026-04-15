// Remesh Lab Worker — minimal worker for shape generation + QEM simplification.
// No atlas, no LOD, no animation — just mesh operations.

let wasm = null;

self.onmessage = async function(e) {
  const { type, data, id } = e.data;

  if (type === 'init') {
    const mod = await import('./pkg/quilting_wasm.js?v=' + Date.now());
    await mod.default();
    wasm = mod;
    self.postMessage({ type: 'ready', id });
    return;
  }

  if (type === 'generate_shape') {
    const { shape, subdivisions } = data;
    const result = wasm.generate_test_mesh(shape, subdivisions || 3);
    if (result) {
      self.postMessage({
        type: 'shape_generated', id,
        positions: result.positions,
        faces: result.faces,
        num_vertices: result.num_vertices,
        num_faces: result.num_faces,
      });
    } else {
      self.postMessage({ type: 'shape_generated', id, error: 'failed to generate' });
    }
    return;
  }

  if (type === 'remesh') {
    const { positions, faces, target } = data;
    const result = wasm.simplify_mesh(
      new Float64Array(positions.flat()),
      new Uint32Array(faces.flat()),
      target,
    );
    if (result) {
      self.postMessage({
        type: 'remeshed', id,
        positions: result.positions,
        faces: result.faces,
        max_error: result.max_error,
      });
    } else {
      self.postMessage({ type: 'remeshed', id, error: 'simplification failed' });
    }
    return;
  }
};
