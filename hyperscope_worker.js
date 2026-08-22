// Hyperscope Worker: loads WASM, handles glTF loading, animation evaluation, and atlas management.

let wasm = null;
let lodJobGeneration = 0;

self.onmessage = async function(e) {
  const { type, data, id } = e.data;

  if (type === 'init') {
    const startedAt = performance.now();
    const mod = await import('./pkg/quilting_wasm.js');
    const importedAt = performance.now();
    await mod.default(data.wasmModule
      ? { module_or_path: data.wasmModule }
      : undefined);
    const instantiatedAt = performance.now();
    wasm = mod;
    // Only the retained runtime worker needs LOD compute. Atlas helpers avoid a
    // 4096² OffscreenCanvas, a WebGL context, and 500K-face compute buffers.
    const gpuRequested = data.gpuCompute !== false;
    const gpuOk = gpuRequested && wasm.init_gpu_compute(500000);
    console.log(gpuRequested
      ? `Worker: GPU compute ${gpuOk ? 'OK' : 'UNAVAILABLE'}`
      : 'Worker: atlas helper (GPU compute not requested)');
    const finishedAt = performance.now();
    self.postMessage({
      type: 'ready',
      id,
      gpuCompute: gpuOk,
      timing: {
        importMs: importedAt - startedAt,
        instantiateMs: instantiatedAt - importedAt,
        gpuInitMs: finishedAt - instantiatedAt,
        sharedCompiledModule: !!data.wasmModule,
      },
    });
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
    const atlas = wasm.export_all_patches();
    const transfers = [atlas.patches.buffer, atlas.bary.buffer, atlas.tris.buffer, atlas.lines.buffer];
    self.postMessage({ type: 'patches_exported', id, atlas }, transfers);
    return;
  }

  if (type === 'extend_atlas') {
    const ms = wasm.extend_atlas(data.newLod);
    self.postMessage({ type: 'atlas_extended', id, ms });
    return;
  }

  if (type === 'load_gltf_data') {
    lodJobGeneration += 1;
    wasm.cancel_animated_lods();
    const result = wasm.load_gltf_data(new Uint8Array(data.bytes));
    // Decode raw image blobs using browser-native decoders (parallel, fast)
    if (result && result.textures && result.textures.length > 0) {
      const t0 = performance.now();
      const decoded = await Promise.all(result.textures.map(async (tex) => {
        const raw = tex.raw_data;
        if (!raw || raw.length === 0) return { width: 0, height: 0, pixels: null, wrap_s: tex.wrap_s, wrap_t: tex.wrap_t };
        const blob = new Blob([raw], { type: tex.mime_type });
        const bitmap = await createImageBitmap(blob);
        const w = bitmap.width, h = bitmap.height;
        const canvas = new OffscreenCanvas(w, h);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, w, h);
        bitmap.close();
        return { width: w, height: h, data: imageData.data, wrap_s: tex.wrap_s, wrap_t: tex.wrap_t };
      }));
      result.textures = decoded;
      console.log(`Browser-native image decode: ${result.textures.length} textures in ${(performance.now() - t0).toFixed(0)}ms`);
    }
    // Transfer pixel ArrayBuffers to avoid cloning (prevents OOM on large models)
    const transfers = [];
    if (result && result.textures) {
      for (const t of result.textures) {
        if (t.data && t.data.buffer) transfers.push(t.data.buffer);
      }
    }
    self.postMessage({ type: 'gltf_loaded', id, result }, transfers);
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
    lodJobGeneration += 1;
    wasm.cancel_animated_lods();
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
    const rpData = wasm.get_rest_pose_instances();
    self.postMessage({ type: 'rest_pose_instances', id, data: rpData });
    return;
  }

  if (type === 'upload_model_to_compute') {
    const ok = wasm.upload_model_to_compute();
    self.postMessage({ type: 'model_uploaded_to_compute', id, ok });
    return;
  }

  if (type === 'sample_stretch_range') {
    const { mobius, instances, numFaces } = data;
    const range = wasm.sample_stretch_range(
      new Float32Array(mobius),
      new Float32Array(instances),
      numFaces,
    );
    self.postMessage({ type: 'stretch_range', id, min: range[0], max: range[1] });
    return;
  }

  if (type === 'compute_animated_lods') {
    const generation = ++lodJobGeneration;
    wasm.cancel_animated_lods();
    const { t, mobius, subjectStates, density, minPx, vpMatrix, vpWidth, vpHeight, tessDensity, screenAtten, minPxSub, skipAnimation } = data;
    const wt0 = performance.now();
    // Set tess params before compute
    if (tessDensity != null) {
      wasm.set_tess_params(tessDensity, !!screenAtten);
      if (minPxSub != null) wasm.set_min_px_per_sub(minPxSub);
    }
    const wt1 = performance.now();
    let dispatched = false;
    try {
      dispatched = wasm.dispatch_animated_lods(
        skipAnimation ? -1.0 : t,  // t < 0 signals: skip animation, use rest pose
        new Float32Array(mobius || [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0]),
        new Float32Array(subjectStates || []),
        density || 20.0,
        minPx || 0.0,
        new Float32Array(vpMatrix || new Array(16).fill(0)),
        vpWidth || 0,
        vpHeight || 0,
      );
    } catch (e) {
      console.error('dispatch_animated_lods threw:', e.message, e.stack);
    }
    if (!dispatched) {
      self.postMessage({ type: 'animated_lods', id, lods: null,
        timing: { tess_params: wt1-wt0, wasm_total: performance.now()-wt1 },
        debug: { dispatchFailed: true } });
      return;
    }

    const poll = () => {
      if (generation !== lodJobGeneration) {
        self.postMessage({ type: 'animated_lods', id, lods: null,
          timing: { tess_params: wt1-wt0, wasm_total: performance.now()-wt1 },
          debug: { canceled: true } });
        return;
      }
      let result;
      try {
        result = wasm.poll_animated_lods();
      } catch (e) {
        result = null;
        console.error('poll_animated_lods threw:', e.message, e.stack);
      }
      if (result === undefined) {
        setTimeout(poll, 1);
        return;
      }

      const wt2 = performance.now();
      // Grab WASM-side perf measures (they land in worker's performance context)
      const wasmMeasures = {};
      for (const m of performance.getEntriesByType('measure')) {
        if (m.name.startsWith('lod-')) {
          if (!wasmMeasures[m.name]) wasmMeasures[m.name] = [];
          wasmMeasures[m.name].push(Math.round(m.duration * 100) / 100);
        }
      }
      performance.clearMeasures();
      performance.clearMarks();
      const timing = { tess_params: wt1-wt0, wasm_total: wt2-wt1, wasm_phases: wasmMeasures };
      if (result && result.length > 0) {
        const dbgSamples = [];
        for (const fi of [0, 1, Math.floor(result.length/12), Math.floor(result.length/6)-1]) {
          const o = fi * 6;
          if (o+5 < result.length) dbgSamples.push(`f${fi}=[${result[o]},${result[o+1]},${result[o+2]} p${result[o+3]} par${result[o+4]} a${result[o+5]}]`);
        }
        timing.dbgSamples = dbgSamples.join(', ');
        try { timing.gpuState = wasm.debug_gpu_compute_state(); } catch(e) {}
        self.postMessage({ type: 'animated_lods', id, lods: result, timing }, [result.buffer]);
      } else {
        let gpuState = 'unknown';
        try { gpuState = wasm.debug_gpu_compute_state ? wasm.debug_gpu_compute_state() : 'no debug fn'; } catch(e) { gpuState = e.message; }
        self.postMessage({ type: 'animated_lods', id, lods: null, timing,
          debug: { resultType: typeof result, resultNull: result === null, gpuState, wt: wt2-wt1 } });
      }
    };
    setTimeout(poll, 0);
    return;
  }

  if (type === 'generate_single') {
    const { a, b, c } = data;
    wasm.generate_and_store_patch(a, b, c);
    self.postMessage({ type: 'generated', id });
    return;
  }

  if (type === 'load_test_shape') {
    const { shape, param1, param2 } = data;
    const result = wasm.load_test_shape(shape, param1 || 2, param2 || 8);
    if (result) {
      const transfers = [];
      if (result.instances?.buffer) transfers.push(result.instances.buffer);
      if (result.face_lods?.buffer) transfers.push(result.face_lods.buffer);
      if (result.face_materials?.buffer) transfers.push(result.face_materials.buffer);
      self.postMessage({ type: 'test_shape_loaded', id, result }, transfers);
    } else {
      self.postMessage({ type: 'test_shape_loaded', id, result: null });
    }
    return;
  }

  if (type === 'remesh') {
    const targetPatches = data.targetPatches || 200;
    const stats = wasm.remesh_current_model(targetPatches);
    self.postMessage({ type: 'remeshed', id, stats });
    return;
  }

  if (type === 'compute_remeshed') {
    const { mobius, lod } = data;
    const result = wasm.compute_remeshed_instances(
      new Float32Array(mobius),
      lod || 4,
    );
    if (result) {
      const transfers = [];
      if (result.instances?.buffer) transfers.push(result.instances.buffer);
      if (result.face_lods?.buffer) transfers.push(result.face_lods.buffer);
      if (result.face_materials?.buffer) transfers.push(result.face_materials.buffer);
      self.postMessage({ type: 'remeshed_instances', id, result }, transfers);
    } else {
      self.postMessage({ type: 'remeshed_instances', id, result: null });
    }
    return;
  }

  if (type === 'clear_remesh') {
    wasm.clear_remeshed_data();
    self.postMessage({ type: 'remesh_cleared', id });
    return;
  }
};
