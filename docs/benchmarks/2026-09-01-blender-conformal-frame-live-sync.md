# Blender conformal-frame live sync — 2026-09-01

Blender's direct local peer now reads both Hyperscape protocol 0.1 and 0.2 and
authors current 0.2 envelopes. The Python codec round-trips Rust's checked-in
`set_conformal_frame_transform` fixture exactly and enforces the same stable
frame ID, 256-generator bound, generator validation, and legacy-version gate.

The main-thread adapter polls the small authored frame collection at its
existing 20 Hz control cadence. It indexes frames by non-nil stable UUID,
excludes duplicate identities, and compares a quantized signature of each
complete ordered generator word. A local change publishes one atomic word;
an incoming word is fully validated before replacing the matching Blender
collection. Local echoes update neither Blender state nor the durable lane.

Timeline changes refresh the observed entity matrices and frame-word
signatures together. Evaluated animation therefore remains ephemeral
`animation_seconds` presence instead of generating durable edits per frame.
Parent topology, frame creation/removal, names, walls, anchors, and constraints
remain asset-authoring operations; this command only replaces an existing
frame's local-to-parent word.

The integration run also exposed and fixed a pre-existing Python 3.13/Blender
5.1 incompatibility: the default advisory-lease UUID factory returned a
`uuid.UUID`, while normalization only accepted strings.

Verification:

```text
python -m unittest discover -s tools/blender_hyperscape/tests -v
                                                        # 45 passed
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/tests/blender_live_sync.py
                                                        # passed (Blender 5.1.1)
python -m py_compile tools/blender_hyperscape/protocol.py \
  tools/blender_hyperscape/live_sync.py \
  tools/blender_hyperscape/ui.py                        # passed
```

No relay server, browser, renderer, GPU context, or user-run development
server was started.
