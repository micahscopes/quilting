# Shader optimization audit

Scope: the quad atlas generator, including the manual sampling reference. This
is a compiler investigation, not a change to the production algorithm.

## Inspected versions

- Fe: `a8cee5c85562bc075e3e087a0c5e1ddd6aa16a1f`.
- Its pinned Sonatina: `2804b9caf4a58ed3afb690f6e993aca9e015668c`.
- Manual comparison: `05b64ef` (see `REFERENCE-WGSL.md`).

Inspect the pinned source, not an independently advancing Sonatina worktree.
Building the compiler in release mode does not itself select a shader
optimization preset.

## What already runs

Fe's `crates/codegen/src/sonatina/spirv_lower.rs` performs exact private-function
merging, helper classification and rooted inlining. Each inlining frontier runs
CFG cleanup, branch canonicalization, SCCP, scalar canonicalization and CFG
cleanup. Final cleanup adds known-bit, checked-arithmetic and range-branch
simplification before another SCCP/scalar/CFG sequence.

Sonatina's SCCP is composite: it includes aggressive dead-code elimination.
Adding standalone ADCE is therefore not equivalent to turning on previously
missing dead-code elimination.

Helpers outside the accepted shader ABI are expanded for backend legality.
Multiple surviving copies of a live loop are not dead code. Preserving a helper
and deleting unused instructions address different problems.

## Available experiments, not established improvements

Sonatina exposes LICM, load/store optimization, aggregate scalarization, code
sinking, loop strength reduction and global value numbering. These are absent
from the two shader cleanup lists above. Their applicability and benefit must
be established for this IR and its resource effects, not assumed from names.

The Fe source explicitly excludes GVN from frontier cleanup because the sparse
predicated solver can require many GiB on generated proof entries. Do not enable
the full optimization preset indiscriminately. The stock dead-function pass
also assumes object roots that this lowering does not populate; blindly running
that preset risks changing reachability semantics.

## Measurement order

1. Capture helper rejection reasons and pre/post-inline IR with the existing
   `FE_SPIRV_INLINE_TRACE` and `FE_SPIRV_INLINE_SNAPSHOT_DIR` diagnostics.
2. Identify the proposal/retirement query helpers and distinguish ABI rejection,
   unsupported control flow, rejected dependencies and cost policy.
3. Compare isolated additional passes, initially without altering inlining or
   the sampler. Record compilation wall time and peak memory, IR instruction
   counts, SPIR-V/WGSL sizes and helper counts.
4. Repeat all 56 quad baseline geometry hashes and independent selected-point
   checks. Include triangle and invalid-input coverage before promoting a
   generic compiler change.
5. Measure GPU work separately from browser compilation and command encoding.
   Keep the same cases and alternating measurement protocol as the reference.

The reference's 5,980 versus 134,557 WGSL bytes is a useful target, not an estimate
of removable dead code. Its mixed-case queue improvement and lack of measured
uniform-case improvement likewise do not establish a full-atlas speedup.

## First trace result

The uncached release build completed successfully with 47 passes and the same
599,272 pass-referenced WGSL bytes. Reported total compilation was 195,295 ms,
including 133,242 ms of lowering; tracing overhead is included. This is not an
optimization timing comparison. Snapshots are retained in
`/laboratory/quilting/scratch/atlas-optimization-trace-20260905`.

Both `0004-retire` and `0005-propose` contain an authored tree-iterator helper
`next_neighbor__g04d1`. Its pre-inline body has 51 textual IR instructions and
two call sites. It performs the buffer load through `load__g5c5a`, rather than
direct resource instructions. In the post-inline snapshots it is not marked
`inline(never)`; the live entry contains the expanded search loop instead. The
original helper remains as an unreachable definition in the snapshot, so whole-
module textual call counts must not be mistaken for calls from the live entry.

This points to a concrete cost-policy problem worth isolating: direct resource
access is checked locally, while a resource-passing helper must save at least
128 source instructions through sharing (unless another profitable helper
requires it). Two calls to this approximately 50-instruction iterator do not
meet that threshold. The iterator is more than a trivial cursor wrapper, yet
its actual memory work is delegated. Its callers can then become ineligible
because it was rejected.

This is evidence of lost existing helper structure, not proof that lowering the
threshold alone is correct or sufficient. The next regression should distinguish
a trivial resource wrapper from a loop-bearing helper with transitive resource
access, preserving existing ABI/control-flow legality checks. Verify emitted
calls and GPU results before claiming a fix. Riff-cat is not needed to invent
the helper in this case: the helper already exists before inlining.
