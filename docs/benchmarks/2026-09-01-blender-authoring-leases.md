# Blender advisory authoring leases — 2026-09-01

Blender's optional live peer now derives desired lease claims from the selected
stable bound entities and the scene's stable asset identity. Claim UUIDs remain
stable across periodic presence refreshes, selection omission releases them,
and a later reacquisition or asset change creates a fresh claim. Presence TTL
remains the crash/disconnect fallback.

Before sending a dirty transform, the main-thread adapter projects admitted,
unexpired remote presence by `AssetEntityId`. A remote claim holds the local
dirty edit in memory; overlapping local and remote claims are reported as
contention. When the remote sample expires or omits the claim, the retained edit
can publish normally. Remote authored commands are still admitted regardless of
claims, because the lease is coordination—not authority, capability material,
durable history, or an HHHS admission rule.

The dirty queue no longer clears all pending entities before transport sends.
Successfully published, unchanged, unrepresentable, or removed entries are
discarded individually; a transport failure leaves the failed and later edits
pending rather than losing the unsent suffix.

The dependency-free controller proves stable refresh IDs, omission release,
fresh reacquisition and asset-change IDs, the 256-claim fail-closed boundary,
asset scoping, holder deduplication, and deterministic ordering. The real
headless-Blender live-sync test source additionally checks declaration-before-
edit, contention gating, retained-edit publication after TTL expiry, and status
projection.

CPU-only evidence:

```text
python3 -m unittest discover -s tools/blender_hyperscape/tests -p 'test_*.py'
                                                        # 42 passed
python3 -m compileall -q tools/blender_hyperscape       # passed
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui                                  # passed
```

No browser, renderer, WebGPU device, server, relay process, or Blender process
was started. The real Blender live-sync script is an authored integration gate
and remains unexecuted while graphics contexts are externally contended.
