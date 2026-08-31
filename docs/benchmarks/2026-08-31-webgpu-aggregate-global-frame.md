# WebGPU aggregate-global frame residency

## Question

Can a resident-root baseline and its sparse adaptive replacement layer publish
one camera/focus/selection row without letting unrelated scenes share mutable
frame state?

## Finding

Yes, when sharing follows the retained aggregate's ownership boundary. The
browser already creates each root binding epoch before its matching diagnostic
or focus overlay. That exact pair now owns one reference-counted
`PatchRenderGlobalResidency`: a 176-byte GPU buffer plus its exact-word
publication witness.

This is deliberately not device-global. Ordinary fallback scenes retain their
own row, and separate root epochs—including focus and non-focus variants—stay
isolated. An overlay may share only when its model and resident draw-domain
identities match the supplied root bindings. Standalone overlay construction
remains available for conformance fixtures and owns an independent row.

The two families still retain independent 80-byte local-domain tables. Root
encoding publishes the aggregate-global row first; overlay encoding observes
the same memo and records an exact reuse. Rendering, GPU visibility, and
picking bind the same shared buffer, so no parallel camera contract exists.

## Bounded traffic result

Let `D` be resident-root draw domains and `O` adaptive overlay batches.

| Composite-frame case | Separate split families | Shared aggregate global |
| --- | ---: | ---: |
| First frame | `352 + 80(D+O)` | `176 + 80(D+O)` |
| Animation/LOD only | `0` | `0` |
| Camera/focus/selection only | `352` | `176` |
| Local domains only | `80(D+O)` | `80(D+O)` |
| Global and local change | `352 + 80(D+O)` | `176 + 80(D+O)` |

An aggregate without an overlay is unchanged. The reduction is one exact
176-byte queue publication for every composite global update and one fewer GPU
buffer allocation per retained overlay epoch.

## Failure and lifetime boundary

The shared row never outlives both owners because it is reference-counted.
Replacing either production aggregate constructs a new root binding and then
its overlay before the candidate is published. A foreign root/model/domain
combination is rejected before overlay allocation. Local packing failures
invalidate only the affected domain table; the exact shared global witness
remains valid.

## Verification

Zero-build ownership oracle:

```sh
node scripts/smoke-webgpu-aggregate-global-frame.mjs
```

Bounded compiler gates use one low-priority Cargo job:

```sh
cargo check -p quilting-webgpu --lib
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```

No Trunk server, `wasm-pack`, binding generation, linked WebGPU test binary,
or `wasm-opt` is involved.
