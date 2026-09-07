# Composition update path: measured scope

## Background completions

`refresh_completion` now distinguishes residency progress from authored input.
When pending is zero, unrelated worker completions update completed/status but
retain the visible scene and upload prefix without recalculating edge demand,
child resolution or diagnostics. Pending requests still call full refresh.
Control edits continue to use full refresh; this is not a general input cache.

Release Wasm policy tests passed in24.72s, including both diagonal and fan:
resolve the requested tile through the completion path, complete an unrelated
LoD0 tile, observe progress+1 with unchanged visible geometry/revision/prefix,
then exercise changed controls and pending-scene retention.
Log: `/laboratory/quilting/scratch/composition-completion-tests-20260907.log`.

Chrome65 fresh load of `fe-render-3ac168ea1f5b580e.json` entered live mode with
4,296 vertices. Background completion count advanced43→44 while spacing
revision stayed1, pending0 and status0. No console errors/warnings. Release
dev rebuild35.046s, parent Wasm595,120 bytes, unchanged WGSL145,268 bytes /3passes.
These are build/behavior observations, not a measured completion speedup.

## Publication measurement before changing that API

On prior build `fe-render-48d316f5ea50b953.json`, with165 tiles complete, temporarily
wrapped the existing shared publication binding's `write` method with
`performance.now()` through Chrome MCP. Restored the original method afterward;
no instrumentation was added to application code.

- Idle:121 RAF callbacks, zero publication calls. Median RAF delta16.7ms,
  p95 delta16.8ms. The live viewer does not continuously republish idle state.
-120 ordinary DOM interior-twist edits, one per RAF:80 publication invocations,
  4.6ms total, median/p95 approximately0.1ms and maximum0.2ms. Restored the
  original twist afterward. This includes both publication policies and the
  synchronous writer, not overall control/render latency. Timing resolution,
  coalescing and instrumentation overhead limit precision.

Source tracing still identifies redundant work: each actual presentation invokes
the Fe publication callback before the writer checks its receipt. Spacing
rebuilds10×17 samples and a680-byte staging allocation even if its revision is
unchanged. GPU writes are deduplicated; this is CPU staging duplication, not
680bytes uploaded on every idle frame. Visible atlas prefix growth also republishes
the whole prefix; unrelated background growth does not.

Next shared API opportunity: separate a cheap Fe-owned publication identity from
effectful materialization, with receipts tied to physical buffer, queue, memory
owner and epoch. Skip materialization only on a valid receipt; recovery must
replay. Suffix publication additionally needs acknowledged ranges and failure/
recovery tests. Do not cache on a demo buffer name or assume queue submission
proves GPU execution.

Priority: this measurement does not establish staging as the dominant latency
source. Keep interior-quality measurement and controllable patch delivery ahead
of a larger publication redesign. Record future timings separately for control
planning, packing, upload bytes, pipeline realization and GPU rendering.
