#import quilting::compute::visibility_compaction_types::{VisibilityCompactionUniforms, VisibilityBatchRecord}

// One workgroup counts one canonical render bucket. Atomics affect only the
// commutative count; survivor order is established by the scatter pass.

@group(0) @binding(0) var<uniform> dispatch: VisibilityCompactionUniforms;
@group(0) @binding(1) var<storage, read> batches: array<VisibilityBatchRecord>;
@group(0) @binding(2) var<storage, read> source_eligibility: array<u32>;
@group(0) @binding(3) var<storage, read> source_visibility: array<u32>;
@group(0) @binding(4) var<storage, read_write> batch_counts: array<u32>;

var<workgroup> visible_count: atomic<u32>;

@compute @workgroup_size(64)
fn count_visible_instances(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let batch_index = group_id.x;
    if batch_index >= dispatch.counts.x {
        return;
    }
    if local_index == 0u {
        atomicStore(&visible_count, 0u);
    }
    workgroupBarrier();

    let batch = batches[batch_index];
    for (
        var member_index = local_index;
        member_index < batch.source_instance_count;
        member_index += 64u
    ) {
        let source_index = batch.source_first_instance + member_index;
        if source_index < dispatch.counts.y
            && source_eligibility[source_index] != 0u
            && source_visibility[source_index] != 0u
        {
            atomicAdd(&visible_count, 1u);
        }
    }
    workgroupBarrier();
    if local_index == 0u {
        batch_counts[batch_index] = atomicLoad(&visible_count);
    }
}
