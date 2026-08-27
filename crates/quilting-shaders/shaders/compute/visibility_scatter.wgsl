#import quilting::compute::visibility_compaction_types::{VisibilityCompactionUniforms, VisibilityBatchRecord, CompactedBatchRangeRecord}

// One workgroup owns one batch. A 64-lane inclusive scan compacts each chunk
// in stable source-member order; chunk_base serializes only chunks inside that
// one batch. Distinct batches scatter concurrently into prefix-assigned ranges.

@group(0) @binding(0) var<uniform> dispatch: VisibilityCompactionUniforms;
@group(0) @binding(1) var<storage, read> batches: array<VisibilityBatchRecord>;
@group(0) @binding(2) var<storage, read> source_eligibility: array<u32>;
@group(0) @binding(3) var<storage, read> source_visibility: array<u32>;
@group(0) @binding(4) var<storage, read> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(5) var<storage, read_write> compacted_source_instances: array<u32>;

var<workgroup> prefix: array<u32, 64>;
var<workgroup> chunk_base: u32;

@compute @workgroup_size(64)
fn scatter_visible_instances(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let batch_index = group_id.x;
    if batch_index >= dispatch.counts.x {
        return;
    }
    let batch = batches[batch_index];
    let range = compacted_ranges[batch_index];
    if local_index == 0u {
        chunk_base = 0u;
    }
    workgroupBarrier();

    for (
        var chunk_first = 0u;
        chunk_first < batch.source_instance_count;
        chunk_first += 64u
    ) {
        let remaining = min(64u, batch.source_instance_count - chunk_first);
        var visible = 0u;
        var source_index = 0u;
        if local_index < remaining {
            source_index = batch.source_first_instance + chunk_first + local_index;
            if source_index < dispatch.counts.y
                && source_eligibility[source_index] != 0u
                && source_visibility[source_index] != 0u
            {
                visible = 1u;
            }
        }
        prefix[local_index] = visible;
        workgroupBarrier();

        for (var offset = 1u; offset < 64u; offset *= 2u) {
            var preceding = 0u;
            if local_index >= offset {
                preceding = prefix[local_index - offset];
            }
            workgroupBarrier();
            prefix[local_index] += preceding;
            workgroupBarrier();
        }

        if visible != 0u {
            let destination = range.compacted_first_instance
                + chunk_base
                + prefix[local_index]
                - 1u;
            compacted_source_instances[destination] = source_index;
        }
        workgroupBarrier();
        if local_index == 0u {
            chunk_base += prefix[remaining - 1u];
        }
        workgroupBarrier();
    }
}
