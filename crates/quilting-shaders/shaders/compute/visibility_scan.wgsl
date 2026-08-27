#import quilting::compute::visibility_compaction_types::{VisibilityCompactionUniforms, VisibilityBatchRecord, CompactedBatchRangeRecord, IndexedIndirectArguments}

// Batch count is small relative to source instances. One deterministic scan
// freezes exact CPU-oracle offsets and emits the fixed-size indirect table;
// instance counting and stable scatter remain parallel across batches.

@group(0) @binding(0) var<uniform> dispatch: VisibilityCompactionUniforms;
@group(0) @binding(1) var<storage, read> batches: array<VisibilityBatchRecord>;
@group(0) @binding(2) var<storage, read> batch_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(4) var<storage, read_write> indirect_arguments: array<IndexedIndirectArguments>;

@compute @workgroup_size(1)
fn scan_visible_batches(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x != 0u {
        return;
    }
    var compacted_first = 0u;
    for (var batch_index = 0u; batch_index < dispatch.counts.x; batch_index++) {
        let batch = batches[batch_index];
        let instance_count = batch_counts[batch_index];
        compacted_ranges[batch_index] = CompactedBatchRangeRecord(
            batch_index,
            batch.source_first_instance,
            batch.source_instance_count,
            compacted_first,
            instance_count,
        );
        indirect_arguments[batch_index] = IndexedIndirectArguments(
            batch.index_count,
            instance_count,
            0u,
            0,
            // WebGPU requires the optional indirect-first-instance feature for
            // nonzero values. The vertex stage indexes from the range prefix.
            0u,
        );
        compacted_first += instance_count;
    }
}
