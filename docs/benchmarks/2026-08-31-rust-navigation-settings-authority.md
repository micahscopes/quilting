# Rust navigation-settings authority cutover

Date: 2026-08-31

## Outcome

`navstateimpl=rust` is now the canonical route. Focus-transition duration and
the five device-independent surface-walk preferences are edited by a Leptos
CSR view over the committed `NavigationSettings` signal. The browser projects
only the committed packet into the existing runtime consumers and URL.

`navstateimpl=js` and `navstateimpl=shadow` remain explicit, canonical rollback
and measurement routes. Raw WebHID mapping, sensitivity, and focus policy are
separate boundaries and are not reclassified as navigation settings by this
cut.

## Live evidence

An isolated Chromium audit exercised every field in the packet:

- focus/navigation transition;
- walk-frame smoothing;
- tangent pull;
- walk speed;
- body scale; and
- eye height.

The Rust lane mounted six Leptos inputs, committed each edit through
`AppStore`, projected exact semantic units and canonical URL units, and restored
the initial packet with zero mismatch, error, or authority rollback. The shadow
lane produced the same six-field Rust and browser packet (apart from the Rust
projection revision) while retaining the HTML controls. The JavaScript lane
changed and restored the same URL/browser values with zero Rust dispatches,
comparisons, authority writes, or view mounts.

## Supporting gates

- Replay 0.23 records the packet and rejects it under 0.22.
- Native synchronization is atomic, idempotent, and sequence-owned.
- `hyperscope-web` preserves non-UI walk policy when editing the six exposed
  values and rejects invalid complete packets atomically.
- The source smoke rejects browser-owned equality, snapshot admission, and
  sequence allocation.

This migration changes low-rate control authority only. Camera integration,
animated surface attachment, and raw input remain on their existing direct
frame paths and do not wait for DOM/FRP publication.
