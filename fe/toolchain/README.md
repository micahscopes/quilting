# Fe toolchain boundary

The workspace expects an isolated Fe checkout at repository-relative
`.toolchains/fe` (`../.toolchains/fe` from this workspace). That directory is
ignored deliberately: Quilting must not compile against an actively edited Fe
worktree. It lives outside `fe/` so recursive workspace tools see only
Quilting's ingots.

Historical atlas baseline (not the current development head):

```text
745044776d03a758471d2bf55de947d9e9f95d05
```

Canonical remote:

```text
https://github.com/micahscopes/fe.git
```

That baseline is the pushed `mb2` candidate used to
compile the first GPU-resident generated atlas patch and the Fe-authored
`Instanced<Draw, N>` raster policy. Quilting-specific source lives in this
repository; the pin supplies the Fe compiler, standard ingots, typed actor
graph, and WebGPU host. A machine that already has the commit can create a
private clone without modifying the source worktree:

```sh
git clone --shared --no-checkout /path/to/fe-repository ../.toolchains/fe
git -C ../.toolchains/fe checkout --detach 745044776d03a758471d2bf55de947d9e9f95d05
cargo build --release --locked --manifest-path ../.toolchains/fe/Cargo.toml --bin fe
```

Current development is on the same `mb2` integration branch, not a separate
Quilting compiler branch. The geometric arc-authoring code requires the pushed
`672432abf279258dc5d4b271dd596bb594c90a5a` or a successor containing it:
compile-time field reflection now resolves shared records through `self` and
`super` module paths. Its focused release regression passes. The broader
ground-type suite has 40 passing tests and one pre-existing expectation of an
obsolete forwarded-parameter rejection; that is not a fully green release gate.
The old atlas baseline above cannot compile the new cross-module arc provider.

The arc **renderer** additionally needs `c2e7b346e` or a successor: this includes
Fe-authored blending and typed native primitive assembly. The shared WebGPU API
checkpoint `8b7ff12ce` on `mb2` adds sample masks, alpha-to-coverage and all core
depth comparisons. Its 41 actor/compiler and 55 host regressions pass. These
are focused gates, not a declaration that the whole compiler release is green.

All release gates use `target/release/fe`. Fe project analysis additionally
passes `--profile release`; this profile is independent of the Cargo profile
used to build the compiler itself.

Once that exact commit (or a tested successor) is published, replace this
temporary local bootstrap with a normal pinned checkout and rerun every Fe,
Wasm, WGSL, and raster-oracle gate before changing the pin.

The pin passes release-profile checks for the atlas planner, packed WebGPU
sampling/topology provider, typed instanced draw derivation, and the
interactive standalone demos. The active `mb2` worktree's unrelated
Mandelbrot and emitter changes are not part of this pin and must not be
repaired from this repository.
