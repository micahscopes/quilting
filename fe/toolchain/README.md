# Fe toolchain boundary

The workspace expects an isolated Fe checkout at repository-relative
`.toolchains/fe` (`../.toolchains/fe` from this workspace). That directory is
ignored deliberately: Quilting must not compile against an actively edited Fe
worktree. It lives outside `fe/` so recursive workspace tools see only
Quilting's ingots.

Validated revision:

```text
6081b9b66acc7689768c1c0ece27d64a960403f7
```

Canonical remote:

```text
https://github.com/micahscopes/fe.git
```

As of 2026-09-02, the validated revision is the `mb2` commit that adds the
Fe-owned `SurfacePointerMotion` capability. The canonical GitHub remote may
still lag the local `mb2` branch, so a machine that already has the commit can
create a private clone without modifying the source worktree:

```sh
git clone --shared --no-checkout /path/to/fe-repository ../.toolchains/fe
git -C ../.toolchains/fe checkout --detach 6081b9b66acc7689768c1c0ece27d64a960403f7
cargo build --release --locked --manifest-path ../.toolchains/fe/Cargo.toml --bin fe
```

All release gates use `target/release/fe`. Fe project analysis additionally
passes `--profile release`; this profile is independent of the Cargo profile
used to build the compiler itself.

Once that exact commit (or a tested successor) is published, replace this
temporary local bootstrap with a normal pinned checkout and rerun every Fe,
Wasm, WGSL, and raster-oracle gate before changing the pin.

The pin passed the locked fixture, exact Quilting-export, and Fe/Wasm oracle
gates (8, 12, and 16 tests respectively). The active `mb2` worktree's unrelated
Mandelbrot changes are not part of this pin and must not be repaired from this
repository.
