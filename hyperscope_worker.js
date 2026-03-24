// Hyperscope Worker: loads WASM, handles glTF loading, animation evaluation, and atlas management.

let wasm = null;

self.onmessage = async function(e) {
  const { type, data, id } = e.data;

  if (type === 'init') {
    const mod = await import('./pkg/quilting_wasm.js');
    await mod.default();
    wasm = mod;
    // Try to init GPU compute (OffscreenCanvas + WebGL2 for transform feedback)
    const gpuOk = wasm.init_gpu_compute(200000); // up to 200K faces
    console.log(`Worker: GPU compute ${gpuOk ? 'OK' : 'UNAVAILABLE'}`);
    self.postMessage({ type: 'ready', id, gpuCompute: gpuOk });
    return;
  }

  if (type === 'get_shaders') {
    const vs = wasm.get_vertex_glsl();
    const fs_matcap = wasm.get_fragment_glsl('matcap');
    const fs_wire = wasm.get_fragment_glsl('wire');
    const fs_normals = wasm.get_fragment_glsl('normals');
    const fs_pbr = wasm.get_fragment_glsl('pbr');
    const fs_pick = wasm.get_fragment_glsl('pick');
    self.postMessage({ type: 'shaders', id, vs, fs_matcap, fs_wire, fs_normals, fs_pbr, fs_pick });
    return;
  }

  if (type === 'set_sliver') {
    wasm.set_sliver_threshold(data.threshold);
    self.postMessage({ type: 'sliver_set', id });
    return;
  }

  if (type === 'set_tess_params') {
    wasm.set_tess_params(data.density || 20, !!data.screenAtten);
    if (data.minPx != null) wasm.set_min_px_per_sub(data.minPx);
    self.postMessage({ type: 'tess_params_set', id });
    return;
  }

  if (type === 'build_atlas') {
    const { maxLodExp, mode } = data;
    const ms = wasm.build_atlas(maxLodExp, mode);
    self.postMessage({ type: 'atlas_built', id, ms });
    return;
  }

  if (type === 'build_atlas_subset') {
    const { maxLodExp, mode, workerIndex, numWorkers } = data;
    const ms = wasm.build_atlas_subset(maxLodExp, mode, workerIndex, numWorkers);
    self.postMessage({ type: 'atlas_subset_built', id, ms });
    return;
  }

  if (type === 'merge_atlas_bytes') {
    const ok = wasm.merge_atlas_bytes(new Uint8Array(data.bytes));
    self.postMessage({ type: 'atlas_merged', id, ok });
    return;
  }

  if (type === 'export_atlas_bytes') {
    const bytes = wasm.export_atlas_bytes();
    self.postMessage({ type: 'atlas_bytes', id, bytes: bytes.buffer }, [bytes.buffer]);
    return;
  }

  if (type === 'import_atlas_bytes') {
    const ok = wasm.import_atlas_bytes(new Uint8Array(data.bytes));
    self.postMessage({ type: 'atlas_imported', id, ok });
    return;
  }

  if (type === 'export_patches') {
    const patches = wasm.export_all_patches();
    self.postMessage({ type: 'patches_exported', id, patches });
    return;
  }

  if (type === 'extend_atlas') {
    const ms = wasm.extend_atlas(data.newLod);
    self.postMessage({ type: 'atlas_extended', id, ms });
    return;
  }

  if (type === 'load_gltf_data') {
    const result = wasm.load_gltf_data(new Uint8Array(data.bytes));
    self.postMessage({ type: 'gltf_loaded', id, result });
    return;
  }

  if (type === 'set_face_materials') {
    wasm.set_face_materials(new Int32Array(data.materials));
    self.postMessage({ type: 'face_materials_set', id });
    return;
  }

  if (type === 'list_animations') {
    const animations = wasm.list_animations();
    self.postMessage({ type: 'animations_listed', id, animations });
    return;
  }

  if (type === 'set_active_animation') {
    const result = wasm.set_active_animation(data.index);
    self.postMessage({ type: 'animation_switched', id, result });
    return;
  }

  if (type === 'evaluate_animation_frame') {
    const pose = wasm.evaluate_animation_frame(data.t);
    self.postMessage({ type: 'animation_pose', id, pose });
    return;
  }

  if (type === 'get_skinning_data') {
    const skinData = wasm.get_skinning_data();
    self.postMessage({ type: 'skinning_data', id, skinData });
    return;
  }

  if (type === 'get_rest_pose_instances') {
    const rpData = wasm.get_rest_pose_instances(data.lod_time || 0);
    self.postMessage({ type: 'rest_pose_instances', id, data: rpData });
    return;
  }

  if (type === 'upload_model_to_compute') {
    const ok = wasm.upload_model_to_compute();
    self.postMessage({ type: 'model_uploaded_to_compute', id, ok });
    return;
  }

  if (type === 'recompute_lods') {
    const { animTime, transformType, params, vpMatrix, vpWidth, vpHeight, tessDensity, screenAtten, minPxSub } = data;
    if (tessDensity != null) {
      wasm.set_tess_params(tessDensity, !!screenAtten);
      if (minPxSub != null) wasm.set_min_px_per_sub(minPxSub);
    }
    const result = wasm.recompute_lods(
      animTime != null ? animTime : -1.0,
      transformType || 'identity',
      new Float64Array(params || []),
      new Float64Array(vpMatrix || new Array(16).fill(0)),
      vpWidth || 0, vpHeight || 0,
    );
    if (result) {
      self.postMessage({ type: 'lods_recomputed', id, lods: result }, [result.buffer]);
    } else {
      self.postMessage({ type: 'lods_recomputed', id, lods: null });
    }
    return;
  }

  if (type === 'compute_animated_lods') {
    const { t, mobius, density, meshRadius, minPx, vpMatrix, vpWidth, vpHeight } = data;
    const result = wasm.compute_animated_lods(
      t,
      new Float32Array(mobius || [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0]),
      density || 20.0,
      meshRadius || 1.0,
      minPx || 0.0,
      new Float32Array(vpMatrix || new Array(16).fill(0)),
      vpWidth || 0,
      vpHeight || 0,
    );
    if (result) {
      self.postMessage({ type: 'animated_lods', id, lods: result }, [result.buffer]);
    } else {
      self.postMessage({ type: 'animated_lods', id, lods: null });
    }
    return;
  }

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }
};
