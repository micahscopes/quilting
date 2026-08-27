# Pointer camera authority gate — 2026-08-27

Mouse, trackpad, and one-finger touch camera gestures now cross the same
retained 17-number camera-packet boundary as SpaceMouse navigation. Rust owns
the incumbent mapping from browser deltas to a device-neutral turntable frame:
world-up yaw, post-yaw local pitch, screen-plane pan, and logarithmic dolly.
Target policy and pose integration commit as one application action.

The ordinary route remains `navimpl=js`. `navimpl=shadow` integrates an
independent Rust candidate before the incumbent browser path and compares the
result without granting write authority. `navimpl=rust` applies the Rust packet
and retains the browser calculation as a rejection fallback.

## Automated boundary evidence

`node scripts/smoke-hyperscope-app-shadow.mjs` passed against release WASM with:

- 81 pointer mapping combinations across orbit, pan, wheel, signed deltas, and
  control distances;
- six cumulative camera states beginning from a rolled basis and exercising
  target-free and finite-target policies;
- an independent JavaScript Rodrigues-rotation oracle; and
- invalid delta, gesture, and output-buffer checks proving no state or packet
  mutation.

The same gate retained the existing 7,168 SpaceMouse mapping cases, 648
response-policy cases, four camera states, and 120-frame deterministic trace.
All 118 `hyperscape` library tests and the release WASM build passed.

## Live Chromium evidence

Chrome DevTools drove one orbit, one shift-pan, and one wheel event through a
fresh local page for each authority route:

| Route | Steps | Comparisons | Writes | Fallbacks | Mismatches | Maximum error |
|---|---:|---:|---:|---:|---:|---:|
| `navimpl=shadow` | 3 | 3 | 0 | 0 | 0 | `2.9852427196885856e-9` |
| `navimpl=rust` | 3 | 3 | 3 | 0 | 0 | `0` |

Neither page emitted a warning or error. The shadow drift is below the existing
`2e-6` browser authority tolerance by roughly three orders of magnitude.

## Remaining gate

JavaScript stays the default until ordinary physical mouse/trackpad use is
accepted across orbit, pan, wheel, selection transitions, and an authored cue
handoff. The rollback remains a canonical `navimpl=js` route rather than an
untracked alternate implementation.
