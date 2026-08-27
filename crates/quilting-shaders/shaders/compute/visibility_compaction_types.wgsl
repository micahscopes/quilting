#define_import_path quilting::compute::visibility_compaction_types

// Storage ABI shared by the deterministic CPU compaction oracle and WebGPU.
// Source eligibility is a separate u32 stream because batch enablement and
// retained-root suppression change much less often than current-pose
// visibility.

struct VisibilityCompactionUniforms {
    // x = batch count, y = source-instance count; z/w are reserved.
    counts: vec4<u32>,
}

struct VisibilityBatchRecord {
    source_first_instance: u32,
    source_instance_count: u32,
    index_count: u32,
    _padding: u32,
}

struct CompactedBatchRangeRecord {
    batch_index: u32,
    source_first_instance: u32,
    source_instance_count: u32,
    compacted_first_instance: u32,
    compacted_instance_count: u32,
}

// Exact five-word DrawIndexedIndirect ABI.
struct IndexedIndirectArguments {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
