# 2026-08-27 hacker-night staged-release evidence

## Scope

This record covers source commit `36f4598` (`Cut polytope-first presentation`)
and an isolated `trunk build --release`. The build and staged bundle lived under
`/tmp`; neither the user-run development server nor its `dist/` directory was
stopped or replaced.

## Artifact gate

The release build reported source fingerprint
`a9e4dd50921eac5afbeb673eb133543b`. Staging excluded `local-glbs/` and the
retired matcap directory. Strict preflight passed with the explicit
`noncommercial-mixed` distribution policy:

- 8 cues and 5 assets;
- 22 checked files and a 24.00 MiB bundle;
- source and build fingerprints equal;
- the horse's CC BY-NC-SA 3.0 terms admitted explicitly, with attribution and
  license notice retained.

This does not make the bundle permissive-only or suitable for commercial use.

## Chrome/WebGL2 gate

The exact staged directory was served on a separate loopback port. Chrome loaded
the first cue, advanced through every cue, and reloaded the canonical final-cue
URL. Every cue reported the expected visualization and animation directive:

1. `both`, horse paused;
2. `wire`, horse paused;
3. `lod`, horse paused;
4. `normals`, horse paused;
5. `both`, horse playing;
6. `lod`, horse playing;
7. `stretch`, horse playing;
8. `pbr`, horse playing.

At every step presentation and assets remained ready, all 5 assets remained
resident, all 4,432 packed faces remained LOD-resident, and application and
scene-extraction mismatch counts stayed zero. The static packed scene contained
12 topology domains and classified all 12 subject records in one GPU pass. The
final canonical reload restored cue
`e0000000-0000-4000-8000-000000000002`, PBR mode, 5 resident assets, and 4,432
faces. Chrome reported no warning or error.

## Fresh-origin startup observation

The first load on the staging origin did not reuse its atlas or environment
cache. The measured startup included:

- WASM phase: 914.0 ms, including 169.3 ms compilation and 698.1 ms renderer
  initialization;
- atlas phase: 129.7 ms, including 29.0 ms generation, 40.7 ms packing
  round-trip, and 3.8 ms upload;
- primary model phase: 121.6 ms;
- render completion: 359.0 ms after its phase began;
- presentation asset completion: 112.1 ms after its phase began;
- environment generation: 1,686.4 ms total, dominated by 1,285.6 ms
  prefiltering and 260.9 ms irradiance generation.

The environment generator, not the tessellation atlas, was the largest
fresh-origin startup cost in this run. These are one-run Chrome measurements,
not generalized frame-time claims.

## Result

The isolated staged artifact passes the current noncommercial hacker-night
release gates. The remaining presentation-machine check is WebHID permission
and ordinary manual interaction (selection, walking, and recovery); those
capabilities are deliberately not certified by filesystem preflight.
