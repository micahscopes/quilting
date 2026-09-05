// Browser diagnostic only. No production page imports this module.
// Keep the generated candidate/mesh pipeline and replace exactly two stages.
export async function installReferenceSampling(surface, source) {
  await surface.live();
  await surface._render();
  const gpu = surface._gpu, changes = [];
  gpu.device.pushErrorScope('validation');
  let scope = true;
  try {
    const module = gpu.device.createShaderModule({code: source});
    const diagnostics = [...(await module.getCompilationInfo()).messages];
    if (diagnostics.some(m => m.type === 'error')) throw Error(diagnostics.map(m => m.message).join('\n'));
    for (const record of gpu.passRecords.filter(r => ['propose', 'retire'].includes(r.pass.source_entry))) {
      const pipeline = await gpu.device.createComputePipelineAsync({
        ...record.pipelineDescriptor,
        compute: {...record.pipelineDescriptor.compute, module, entryPoint: record.pass.source_entry},
      });
      changes.push({record, original: record.pipeline, pipeline});
    }
    const error = await gpu.device.popErrorScope(); scope = false;
    if (error) throw Error(error.message);
    if (changes.length !== 2 || gpu !== surface._gpu) throw Error('unexpected generator or device generation');
    for (const c of changes) c.record.pipeline = c.pipeline;
    return {
      changes,
      restore() { for (const c of changes) c.record.pipeline = c.original; },
    };
  } finally { if (scope) await gpu.device.popErrorScope(); }
}

// Queue completion timing, NOT GPU timestamp timing. Measures an initialized
// sampling job (candidate/bounds build + 64 immutable proposal/retirement rounds),
// not triangulation, rendering, downloading, or whole-atlas startup. Compile
// time and CPU command encoding are excluded from submitToCompletionMs.
export async function compareSamplingQueueTime(surface, changes, pairs = 12,
  labels = {original: 'fe', replacement: 'manual'}) {
  if (!Number.isInteger(pairs) || pairs < 1 ||
      typeof labels.original !== 'string' || typeof labels.replacement !== 'string' ||
      !labels.original || !labels.replacement || labels.original === labels.replacement)
    throw Error('timing requires positive pairs and distinct implementation labels');
  const gpu = surface._gpu, device = gpu.device;
  await device.queue.onSubmittedWorkDone();
  const records = gpu.passRecords.slice(0, 7);
  const names = records.map(r => r.pass.source_entry).join(',');
  if (names !== 'initialize,candidate_index_leaves,candidate_index_reduce,candidate_index_advance,propose,retire,advance')
    throw Error('unsupported generator prelude');
  const observations = [];
  for (let pair = -2; pair < pairs; ++pair) {
    for (const replacement of pair % 2 === 0 ? [false, true] : [true, false]) {
      const began = performance.now(), encoder = device.createCommandEncoder();
      const dispatch = record => {
        const c = changes.find(c => c.record === record);
        const pass = encoder.beginComputePass();
        pass.setPipeline(c ? (replacement ? c.pipeline : c.original) : record.pipeline);
        pass.setBindGroup(0, record.bindGroup);
        pass.dispatchWorkgroups(...record.pass.dispatch); pass.end();
      };
      dispatch(records[0]); dispatch(records[1]);
      for (let i = 0; i < records[2].pass.cycle.repeat; ++i) { dispatch(records[2]); dispatch(records[3]); }
      for (let i = 0; i < records[4].pass.cycle.repeat; ++i) { dispatch(records[4]); dispatch(records[5]); dispatch(records[6]); }
      const commands = encoder.finish(), submitted = performance.now();
      device.queue.submit([commands]); await device.queue.onSubmittedWorkDone();
      const finished = performance.now();
      if (pair >= 0) observations.push({pair, implementation: replacement ? labels.replacement : labels.original,
        encodingMs: submitted-began, submitToCompletionMs: finished-submitted});
    }
  }
  if (gpu !== surface._gpu) throw Error('device changed while timing');
  return {kind: 'queue-completion-not-gpu-timestamp', params: {...surface.params}, observations};
}
