# Animation-clock route restoration gate — 2026-08-27

## Invariant

Canonical animation links use clip-relative `animtime` and signed `animspeed`.
The browser may not consume those values until the requested clip range is
resident. Restoration is one atomic Rust application action; the resulting
fixed three-`f64` clock sample updates renderer time, speed, and browser
controls together.

The Rust clock remains unwrapped while a canonical URL stores wrapped clip
time. Reload therefore restores the same visible pose and direction without
claiming that loop count is authored scene state.

## Regression and repair

The first live gate exposed split authority during startup. The primary upload
restored and cleared `animtime=0.375`, after which an asynchronous clip-0
selection reset only the browser pose to `0`. Rust still returned `0.375`, but
URL canonicalization erased the explicit time.

Startup now awaits the final requested clip selection before enabling URL
writes. A pending route clock survives the primary upload, is consumed against
that final clip range, and refreshes pose controls before rendering resumes.

## Evidence

On the ordinary 1.5-second horse clip:

- paused JavaScript-clock startup with `animtime=0.375&animspeed=-0.5`
  produced URL, slider, Rust clock sample, and Rust clip sample time `0.375`
  with speed `-0.5`;
- reloading the canonical URL reproduced the same values exactly, with one
  route restore and no application or frame error;
- a presentation URL carrying its canonical `anim=0` reloaded through the cue
  animation boundary with the same `0.375` pose and `-0.5` speed; cue-driven
  clip selection deferred route consumption and no independent startup clip
  reset ran afterward;
- Rust authority played backward across the clip origin and multiple loop
  boundaries, then paused with browser and Rust clip time both
  `0.6750000000000025` (browser display `0.675`) and speed `-0.5`;
- that reverse run accumulated 334 authoritative writes with zero fallback
  writes, clock errors, application mismatches, or frame errors;
- Chrome reported no warning or error.

The 82-control generated-WASM route smoke and all 74 `hyperscope-app`
all-feature tests passed. The temporary horse tab was used for the gate; the
user's chess tab was neither selected nor modified.
