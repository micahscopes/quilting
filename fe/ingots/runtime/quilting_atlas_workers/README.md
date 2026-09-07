# Atlas workers

Shared Fe orchestration for independently generated canonical atlas tiles.
`quilting_atlas_wasm` owns geometry and retained storage; this ingot adds the
typed mailbox, suspension, and completion-message effects used by a browser
worker pool. It has no rendering or DOM dependency.

An application supplies:

- A worker actor handling `Request -> Response`, using `packed_triangle` or
  `packed_square` for the requested key.
- Distinct typed `ActorInstance` slots and a bounded task family.
- A `CompletionMessage` implementation whose nominal type matches its resident
  transition. That transition decides how to project progress or render state.
- A retained `PoolHandle<N>` address whose owner outlives every running task.

`run<N, C, M>` claims jobs, awaits the selected worker, validates its original
lease, retains accepted bytes, and sends `M::completed()`. Failed message
delivery cancels the pool. `supervise<C>` uses the shared child supervision
mechanism with bounded restart policy. Neither operation creates geometry in
JavaScript or sends the parent's retained-memory address to a worker.

The `atlas_pool_workers` validation ingot uses this same loop. Its release
compiler gate derives four child packages and eight task machines (four child
supervisors, four job consumers). Browser execution and geometry audits remain
separate acceptance gates; successful compilation is not a performance claim.
