# Fe toolchain boundary

The workspace expects an isolated Fe checkout at repository-relative
`.toolchains/fe` (`../.toolchains/fe` from this workspace). That directory is
ignored deliberately: Quilting must not compile against an actively edited Fe
worktree. It lives outside `fe/` so recursive workspace tools see only
Quilting's ingots.

Validated revision:

```text
c6ab98f222d5def80ad6d2bf6c99373666d6b48e
```

Canonical remote:

```text
https://github.com/micahscopes/fe.git
```

As of 2026-09-01, the validated revision is four commits ahead of the remote
`mb2` head (`ba0d41fdd21f9c3ad7523accc72843ba4aa025a9`). Therefore it cannot yet
be represented honestly as a portable git dependency or submodule. On a
machine that already has the commit, create a private clone without modifying
the source worktree:

```sh
git clone --shared --no-checkout /path/to/fe-repository ../.toolchains/fe
git -C ../.toolchains/fe checkout --detach c6ab98f222d5def80ad6d2bf6c99373666d6b48e
cargo build --release --locked --manifest-path ../.toolchains/fe/Cargo.toml --bin fe
```

All release gates use `target/release/fe`. Fe project analysis additionally
passes `--profile release`; this profile is independent of the Cargo profile
used to build the compiler itself.

Once that exact commit (or a tested successor) is published, replace this
temporary local bootstrap with a normal pinned checkout and rerun every Fe,
Wasm, WGSL, and raster-oracle gate before changing the pin.

The final M1 evidence also records a later uncommitted compiler regression in
the active development worktree. That dirt is not part of this pin and must
not be repaired from this repository.
