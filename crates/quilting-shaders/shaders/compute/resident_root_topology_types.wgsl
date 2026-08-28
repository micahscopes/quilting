#define_import_path quilting::compute::resident_root_topology_types

struct ResidentRootTopologyDispatch {
    // x = source faces, y = compact vertices, z = affine subject rows,
    // w = reserved.
    counts: vec4<u32>,
}
