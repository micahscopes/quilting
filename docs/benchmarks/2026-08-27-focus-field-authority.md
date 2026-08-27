# Focus-field authority gate — 2026-08-27

Spheroidal focus enablement, shell coordinate, and angular aperture now enter
Hyperscape as one atomic navigation action. The legacy coordinate-only WASM and
replay APIs remain compatible, while interactive adapters include enablement so
invalid geometry cannot leave a partial toggle behind.

The generated-WASM application smoke compared `HyperscopeAppShadow` with the
standalone navigation facade. It accepted `(true, 0.35, 0.075)`, then rejected
`(false, NaN, 0.2)` while retaining the complete accepted focus state and exact
diagnostic parity. Native verification passed 113 Hyperscape tests and all 64
all-feature Hyperscope application tests.

Chrome DevTools MCP exercised the three retained migration modes against the
live WebGL2 application:

| Mode | Immediate AppStore state | Post-microtask state | Renderer authority writes |
| --- | --- | --- | ---: |
| `rust` | retained `0.30 / 0.12` while controls requested `0.77 / 0.23` | committed `0.77 / 0.23`; Leptos and URL matched | 2 including startup |
| `shadow` | retained `0.31 / 0.13` while controls requested `0.72 / 0.19` | observer reached `0.72 / 0.19` | 0 |
| `js` | retained its bootstrap `0.32 / 0.14` | unchanged; browser controls owned `0.68 / 0.17` | 0 |

The Rust run reported two comparisons, zero rejections, zero application
mismatches, and zero frame errors. The browser-independent suite passed all 66
tests; presentation, render-shadow, surface-walk, route, and application WASM
smokes also passed. Temporary test pages were closed, and the pre-existing
chess page was restored to its exact captured URL after Chrome reordered page
IDs during the rollback gate.
