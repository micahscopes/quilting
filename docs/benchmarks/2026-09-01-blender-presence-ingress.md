# Blender presence-ingress parity — 2026-09-01

Blender's dependency-free protocol adapter now applies the same bounded ingress
policy as `hyperscope-app::LocalPeerIngress` to ephemeral presence: validate
first, consume known local echoes, reject duplicate message IDs, reject delayed
sender-local sequences, and only then install a receipt-relative-TTL sample.
The existing `PresenceInbox.accept` boolean remains available; `admit` exposes
the exact Rust-compatible disposition.

Outbound presence is recorded only after the delivery adapter accepts the
frame. A relay echo therefore creates neither a remote Blender peer nor a
second semantic sample. Authored and presence inboxes share the same bounded
message and sender-sequence primitives, while presence still has no conversion
to authored history or HHHS.

The Python fixture tests had retained references to the removed top-level
`fixtures/protocol` directory. They now read the canonical checked-in values
directly from `crates/hyperscape-protocol/fixtures`, restoring the intended
Rust/Python oracle instead of maintaining copies.

CPU-only evidence:

```text
python3 -m unittest discover -s tools/blender_hyperscape/tests -p 'test_*.py' -v
35 passed
```

No browser, renderer, WebGPU device, server, relay process, or Blender process
was started for this checkpoint. The real headless-Blender live-sync script was
extended to assert presence-echo consumption, but its execution remains a
separate integration gate while another workload owns the GPU runtime.
