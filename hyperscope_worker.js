// Hyperscope Worker: loads WASM, handles spacetime slicing + Mobius transforms.

let wasm = null;
let sharedOrigView = null, sharedXformView = null;

self.onmessage = async function(e) {
  const { type, data, id } = e.data;

  if (type === 'set_shared_buffer') {
    const { sharedBuf, origOffset, xformOffset, slotSize } = e.data;
    sharedOrigView = new Float32Array(sharedBuf, origOffset, slotSize / 4);
    sharedXformView = new Float32Array(sharedBuf, xformOffset, slotSize / 4);
    return;
  }

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

  if (type === 'create_hypermesh') {
    const result = wasm.create_hypermesh(data.name);
    // Auto-prebake animation frames to GPU
    if (result && result.time_min != null && result.time_max != null) {
      const nframes = Math.min(240, Math.max(60, Math.ceil((result.time_max - result.time_min) * 30)));
      const baked = wasm.prebake_animation(nframes, result.time_min, result.time_max);
      console.log(`Prebaked ${baked} animation frames (t=${result.time_min}-${result.time_max}, nframes=${nframes})`);
      if (baked === 0) console.warn('Prebake returned 0 — GPU compute may have failed');
    } else {
      console.warn('create_hypermesh: no time range, prebake skipped', result);
    }
    self.postMessage({ type: 'hypermesh_created', id, result });
    return;
  }

  if (type === 'slice_and_transform') {
    const { normal, offset, transformType, params, overrideRes, vpMatrix, vpWidth, vpHeight, toroidal, tessDensity, screenAtten, minPxSub } = data;
    if (tessDensity != null) {
      wasm.set_tess_params(tessDensity, !!screenAtten);
      if (minPxSub != null) wasm.set_min_px_per_sub(minPxSub);
    }
    const result = wasm.slice_and_transform(
      new Float64Array(normal),
      offset,
      transformType,
      !!toroidal,
      new Float64Array(params),
      overrideRes,
      new Float64Array(vpMatrix || []),
      vpWidth || 0,
      vpHeight || 0,
    );
    if (sharedOrigView && result.all_orig && result.all_xform) {
      // Write into SharedArrayBuffer — main thread reads directly, zero copy
      sharedOrigView.set(result.all_orig);
      sharedXformView.set(result.all_xform);
      // Send only metadata (tiny) — instance data is in shared memory
      const metaTransfer = [];
      if (result.batch_meta) metaTransfer.push(result.batch_meta.buffer);
      if (result.face_indices) metaTransfer.push(result.face_indices.buffer);
      self.postMessage({ type: 'batches', id, result: {
        total_faces: result.total_faces,
        num_batches: result.num_batches,
        batch_meta: result.batch_meta,
        face_indices: result.face_indices,
        shared: true,
        data_len: result.all_orig.length,
      }}, metaTransfer);
    } else {
      // Fallback: transfer flat buffers
      const transferList = [];
      if (result.all_orig) transferList.push(result.all_orig.buffer);
      if (result.all_xform) transferList.push(result.all_xform.buffer);
      if (result.batch_meta) transferList.push(result.batch_meta.buffer);
      if (result.face_indices) transferList.push(result.face_indices.buffer);
      self.postMessage({ type: 'batches', id, result }, transferList);
    }
    return;
  }

  if (type === 'load_gltf_data') {
    const result = wasm.load_gltf_data(new Uint8Array(data.bytes));
    // Check if model has GPU skinning data — skip prebake if so
    const skinData = wasm.get_skinning_data();
    const hasGpuAnimation = skinData && (skinData.num_joints > 0 || skinData.num_vertices > 0);
    if (hasGpuAnimation) {
      console.log(`GPU animated model: ${skinData.num_joints} joints, ${skinData.num_vertices} verts — no prebake`);
    } else if (result && result.time_min != null && result.time_max != null) {
      const nframes = Math.min(240, Math.max(60, Math.ceil((result.time_max - result.time_min) * 30)));
      const baked = wasm.prebake_animation(nframes, result.time_min, result.time_max);
      console.log(`Prebaked ${baked} glTF animation frames`);
    }
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
    // Re-prebake with the new animation's time range
    if (result && result.time_min != null && result.time_max != null) {
      const nframes = Math.min(240, Math.max(60, Math.ceil((result.time_max - result.time_min) * 30)));
      const baked = wasm.prebake_animation(nframes, result.time_min, result.time_max);
      console.log(`Switched to animation ${data.index}, prebaked ${baked} frames`);
    }
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

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }
};
