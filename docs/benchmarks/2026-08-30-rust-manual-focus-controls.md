# Rust-authoritative manual focus controls

Date: 2026-08-30

## Outcome

The four legacy Möbius/focus sliders now submit their requested sphere geometry
to `hyperscope-app` before changing browser signals whenever
`selectionimpl=rust` is active. Rust validates and queues one semantic focus
edit, integrates it at the application clock fence, transports the camera and
surface state across the resulting reflection chart, and publishes one
navigation snapshot. The browser then projects that committed snapshot into
the controls and renderer.

The interaction semantics are unchanged:

- changing an anchored radius preserves the selected identity and edits its
  margin;
- changing an anchored center deliberately detaches the focus sphere while
  retaining the selected object;
- changing a free sphere replaces its complete geometry; and
- a rejected pole-crossing edit restores the slider to the committed value
  without first publishing a transient browser sphere.

The `js` and unmapped-selection fallbacks remain available.

## Duplicate-transport removal

Browser signals run effects synchronously. Before this cut, projecting a Rust
focus snapshot through the Möbius signals could invoke the incumbent browser
camera transport before the Rust camera snapshot was installed. A scoped Rust
projection fence now tells the effect that chart transport has already been
resolved atomically by the reducer. The effect still updates the renderer,
atlas/LOD invalidation, URL, and compatibility projections, but it does not
integrate a second camera or surface transition.

A selected object may coexist with a deliberately detached free focus sphere.
The inversion gesture now projects either an anchored selected-focus snapshot
or a detached navigation snapshot accordingly instead of misreporting the
latter as a reflection-pole rejection.

## Verification

- the 88-spec Hyperscope route/source oracle passed with the manual-control and
  single-transport guards;
- the complete inline ES module passed `node --check`;
- the native `hyperscope-app` atomic focus-edit test passed; and
- no browser was launched, reloaded, or controlled for this cut. Interactive
  parity remains a separate user-run acceptance gate.
