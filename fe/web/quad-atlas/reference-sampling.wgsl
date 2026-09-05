// Hand-written experimental reference, NOT part of the production Fe page.
// Replaces only proposal/retirement. Candidate generation, bounds construction,
// compaction, triangulation and rendering still come from Fe in this first probe.
// The layout is the current quad diagnostic ABI; capacity follows the buffers.
struct Params {
    bottom: f32, right: f32, top: f32, left: f32, seed: f32, wire_px: f32,
}
@group(0) @binding(0) var<storage, read_write> sampling: array<u32>;
@group(0) @binding(1) var<storage, read_write> bounds: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

const NONE = 0xffffffffu;
const PENDING = 0u;
const ACCEPTED = 1u;
const REJECTED = 2u;
override MAX_LOD: u32 = 3u; // Match the current preview's declared request range.

struct Candidate { point: vec2<u32>, radius2: u32, priority: u32 }
struct Query {
    node: u32, offset: u32, slot: u32,
    // A tight cell window is cheaper than the tree for uniform/small queries.
    origin: vec2<u32>, extent: vec2<u32>, side: u32, window_count: u32,
}

fn capacity() -> u32 { return (arrayLength(&sampling) - 1u) / 8u; }
fn load_candidate(slot: u32) -> Candidate {
    let word = slot * 5u;
    return Candidate(vec2(sampling[word], sampling[word+1u]), sampling[word+3u], sampling[word+4u]);
}
fn conflict(a: Candidate, b: Candidate) -> bool {
    let delta = vec2<i32>(a.point) - vec2<i32>(b.point);
    return u32(delta.x*delta.x + delta.y*delta.y) < max(a.radius2, b.radius2);
}
fn following_subtree(node: u32) -> u32 {
    var n = node;
    while n > 1u && (n & 1u) != 0u { n >>= 1u; }
    return select(n+1u, 0u, n <= 1u);
}
fn excludes(node: u32, candidate: Candidate) -> bool {
    let word = 1u + (node-1u)*5u;
    let lo = vec2(bounds[word], bounds[word+1u]);
    if lo.x == NONE { return true; }
    let hi = vec2(bounds[word+2u], bounds[word+3u]);
    if any(candidate.point > vec2(16384u)) || any(hi > vec2(16384u)) { return false; }
    let p = vec2<i32>(candidate.point);
    let delta = max(max(vec2<i32>(lo)-p, p-vec2<i32>(hi)), vec2(0));
    return u32(delta.x*delta.x + delta.y*delta.y) >= max(candidate.radius2, bounds[word+4u]);
}
fn levels() -> vec4<u32> {
    return vec4<u32>(clamp(vec4(params.bottom, params.right, params.top, params.left), vec4(0.0), vec4(f32(MAX_LOD))));
}
fn start_query(candidate: Candidate, lod: vec4<u32>) -> Query {
    let side = 2u << max(max(lod.x,lod.y),max(lod.z,lod.w));
    let radius = 16384u >> min(min(lod.x,lod.y),min(lod.z,lod.w));
    let cell = 16384u / side;
    // Candidate IDs are column-major in the packed square coordinates.
    let p = candidate.point.yx;
    let lo = vec2<u32>(max(vec2<i32>(p)-vec2<i32>(i32(radius)),vec2(0))) / cell;
    let hi = min((p+vec2(radius))/cell, vec2(side-1u));
    let extent = hi-lo+vec2(1u);
    return Query(1u,0u,NONE,lo,extent,side,extent.x*extent.y*2u);
}
fn next_query(candidate: Candidate, count: u32, cursor: Query) -> Query {
    var q = cursor;
    q.slot = NONE;
    if q.window_count <= 50u {
        if q.offset < q.window_count {
            let cell = q.offset / 2u;
            let xy = q.origin + vec2(cell/q.extent.y, cell%q.extent.y);
            q.slot = (xy.x*q.side+xy.y)*2u + q.offset%2u;
            q.offset++;
        }
        return q;
    }
    let leaves = ((arrayLength(&bounds)-1u)/5u+1u)/2u;
    while q.node != 0u {
        if q.offset == 0u && excludes(q.node,candidate) {
            q.node = following_subtree(q.node);
        } else if q.node < leaves {
            q.node *= 2u;
        } else {
            let slot = (q.node-leaves)*16u + q.offset;
            if slot >= count { q.node=0u; return q; }
            q.slot = slot;
            q.offset++;
            if q.offset == 16u { q.node=following_subtree(q.node); q.offset=0u; }
            return q;
        }
    }
    return q;
}

// One immutable state generation is read throughout each dispatch. Proposal
// writes a disjoint winner mask; retirement writes the other state generation.
fn sample_lane(slot: u32, retirement: bool) {
    let cap = capacity();
    if slot >= cap { return; }
    let lod = levels();
    let side = 2u << max(max(lod.x,lod.y),max(lod.z,lod.w));
    let count = min(2u*side*side,cap);
    let parity = sampling[cap*8u];
    let current_base = cap * select(6u,5u,parity==0u);
    let next_base = cap * select(5u,6u,parity==0u);
    let state = sampling[current_base+slot];
    if slot >= count || state != PENDING {
        if retirement { sampling[next_base+slot]=state; }
        else { sampling[cap*7u+slot]=0u; }
        return;
    }
    let candidate = load_candidate(slot);
    let winner = sampling[cap*7u+slot] != 0u;
    var query = start_query(candidate,lod);
    var blocked = false;
    loop {
        query = next_query(candidate,count,query);
        let other_slot = query.slot;
        if other_slot == NONE { break; }
        if other_slot >= count || other_slot == slot { continue; }
        let other_state = sampling[current_base+other_slot];
        var relevant = other_state == ACCEPTED || other_state == PENDING;
        if retirement { relevant = other_state == ACCEPTED || (!winner && sampling[cap*7u+other_slot] != 0u); }
        if !relevant { continue; }
        let other = load_candidate(other_slot);
        if !conflict(candidate,other) { continue; }
        if retirement || other_state == ACCEPTED || other.priority < candidate.priority
            || (other.priority == candidate.priority && other_slot < slot) {
            blocked=true; break;
        }
    }
    if retirement {
        var next = PENDING;
        if blocked { next=REJECTED; } else if winner { next=ACCEPTED; }
        sampling[next_base+slot]=next;
    } else { sampling[cap*7u+slot]=select(1u,0u,blocked); }
}
@compute @workgroup_size(64) fn propose(@builtin(global_invocation_id) id:vec3<u32>) { sample_lane(id.x,false); }
@compute @workgroup_size(64) fn retire(@builtin(global_invocation_id) id:vec3<u32>) { sample_lane(id.x,true); }
