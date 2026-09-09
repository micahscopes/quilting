# Interior counts with the layout held fixed

This is a bounded experiment, not an accepted automatic sampling policy.

Question: can measured physical edge lengths improve the existing two-diagonal
and four-triangle arrangements without moving their vertices or warping samples?
This separates count allocation from the earlier coarse-layout ranking experiment.

## Candidate

Keep the authored surface, outer counts, center, connectivity and placement maps.
Measure the four boundary curves and each visible interior curve using 16 equal
parameter intervals. Pool the outer lengths and segment counts into a target
world-space spacing. Round each interior length divided by that spacing upward
to an available dyadic segment count. Explicitly report demand beyond LoD8.

This is sampled chord length, not analytical arc length; the pooled spacing is
not a claim that unequal boundaries are individually uniform. It is also not a
screen-space policy. Compare 16 versus 32 intervals to detect unstable rounding.
Retain a fixed center so a successful count change cannot be mistaken for a
successful center optimizer.

## Smallest comparison and gates

- Use the existing three symmetric/asymmetric curved fixtures, both diagonals
  and the central four-triangle fan, with outer LoD3.
- Test count-demand multipliers 0.5, 1 and 1.5, alongside the existing rule.
- Record resolved counts, capped edges, planning evaluations, actual triangles,
  invalid/folded triangles, minimum/mean shape and area variation.
- Compare only equal triangle counts directly; otherwise report quality/cost
  tradeoffs. More triangles alone do not establish a better allocation.
- Verify outer counts stay fixed and admission/cap failures remain visible.
- Reject a universal replacement if hard fixtures regress without a compensating
  measured quality/cost benefit. A useful bounded shortlist is a weaker outcome.

The next slice is an oracle comparison, not a viewer default. A surviving policy
still needs low-tail/error comparisons, unequal outer counts, boundary identity
checks, pathological cases and edit-stability measurements before integration.

## September 8 result: not a replacement policy

The release Wasm gate passed 27 combinations in 114.51 seconds including Fe
compilation and mesh audits. All outer counts were preserved; no sampled mesh
had invalid vertices or UV folds. All 16/32-interval pairs chose identical
levels. The two plan calls together took 0.0451–0.0663 ms in Wasmtime, including
host calls, not browser-worker scheduling or atlas assembly. A 16-interval plan
uses 85 surface evaluations for a diagonal or 136 for a fan.

At multiplier 1 the symmetric fixture exactly reproduces the reference counts.
On the strongly asymmetric fan, it raises the budget from 376 to 596 triangles
while reducing minimum shape from 0.16118 to 0.13471 and mean shape from 0.57600
to 0.53496. The second asymmetric diagonal 0–2 similarly grows from 272 to 572
triangles while minimum shape falls from 0.25935 to 0.14135. More physical edge
samples do not fix the interior distortion under unchanged placement maps.

Multiplier 0.5 sometimes improves shape with fewer triangles, but not
consistently; no approximation-error claim follows from that observation.
Retain the experiment as a quality/cost ablation. Do not replace the existing
automatic rule with this pooled-length policy. Broader spacing/approximation
work remains necessary; this result rejects this shortcut, not the objective.

Raw log: `/laboratory/quilting/scratch/frozen-count-allocation-20260908.log`.
