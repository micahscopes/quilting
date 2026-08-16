# Lean roadmap for `formal/` (conformal mereology)

Handoff for the Codex agent. Produced by a Fable review of the current Lean against the
refined mathematics from an extended design conversation. Mathlib names verified against the
pinned checkout at `formal/.lake/packages/mathlib` (Lean 4.29.0 / mathlib v4.29.0).

## Implementation status (2026-08-15, completion pass)

The first foundation pass is now implemented in `ConformalMereology.lean`:

- T1 and T2: finite signed payload mass, complement accounting, bijection
  transport, and separating/nonseparating wall re-anchor laws;
- the open-side semantic correction: `RegularOpen X` is implemented via
  `Heyting.Regular (Opens X)`, with Boolean complement proved to have carrier
  `interior (set-complement)` and an explicit off-wall condition;
- a no-global-bottom finite Möbius inversion theorem, derived from
  `IncidenceAlgebra.zeta_mul_mu` rather than misapplying
  `moebius_inversion_bot` to `WithTop`;
- a finite `ChamberModel` in which region mass is proved to be the zeta
  transform of direct chamber mass and Möbius inversion recovers it;
- an indexed `FiniteLaminarFamily` whose pointwise laminarity theorem derives
  the unique least containing region after adjoining a background top;
- canonical geometric chambers `region w \ ⋃_{v<w} region v`, proved
  pairwise disjoint and exhaustive, with every region reconstructed from its
  lower chambers;
- the full T3–T4 semantic bridge: geometric-region payload mass is the zeta
  transform of geometric-chamber payload mass, and incidence Möbius inversion
  recovers the latter;
- a general semantic `ChamberReassignment` and the checked
  `Z_new * R * M_old` formula, including two-step path composition and
  arbitrary-length finite-chain reversal via `Fin.revPerm`;
- T6's chain-interval theorem `mu = I - C`, plus the matched-order update
  `mu_new - mu_old = C_old - C_new` and the changed-cover support corollary;
- T7's exact inversive-power scaling, absolute signed-distance invariance,
  separated/tangent/crossing preservation, indexed Gram law `D * G * D`, and
  single-wall row/column support;
- compactified inversion upgraded from an involutive equivalence to a
  `Homeomorph` under the necessary `ProperSpace` hypothesis, with continuity
  checked separately at ordinary points, the pole, and infinity;
- D4 and T9 completed: balls, strict exteriors, and affine half-spaces are
  concrete `IsOpenRoundSide`s; they are proved open and regular open, package
  into `RegularOpen`, and arbitrary-centre compactified inversion preserves
  them with exact pole/infinity bookkeeping;
- T10 completed at generator level: compactified translation, nonzero scale,
  orthogonal maps (hence rotations), and arbitrary-centre inversion preserve
  round sides, and `RoundSideAutomorphism.trans` certifies their compositions;
- T8's tractable rank result completed: every row/column-supported update is
  an explicit sum of two outer products and therefore has rank at most two;
  the single-wall signed-Gram update is a direct corollary;
- the four-chamber background regression theorem, proving the honest
  `(14,12,8)` result differs from the three-label `(7,6,4)` shuffle; and
- direct specialization of selected-side and payload transport to
  `extendedInversionEquiv`.

All tractable T1–T10 items in this roadmap are now implemented. The semantic
`R` theorem is intentionally general; applying it to a particular geometric
re-anchor still means supplying that re-anchor's physical chamber matching.
The research-open exclusions remain realizability, monodromy across crossing
events, conformal-structure reconstruction, and a full quaternionic
`SL(2,ℍ)` classification.

## 0. Original audit before the foundation pass

`ConformalMereology.lean` (351): `pushRegion` (:36) w/ `pushRegion_subset_iff` (:40),
`pushRegion_compl` (:53); `chosenSide` (:77) w/ `_eq_of_not_separates` (:88),
`_eq_compl_of_separates` (:97); generic conjugation (`transport` :116, `conjugate` :132,
`conjugate_leftInverse` :138); `containmentPivot` (:176) — proved for bare functions `A→A`
with mutual-inverse hyps, **no posets**; `zetaTransform` (:222), `zetaTransform_orderIso` (:227);
`fullPivot_transposes_mu` (:267) — one-line re-export of `IncidenceAlgebra.mu_toDual`; `ThreeLevel`
toy (:278–347) w/ `reverse₃` (:313), `pivotZeta₃` (:319), numeric `(1,2,4)↦(7,6,4)` (:339).

`SphericalInversion.lean` (617): `extendedInversion` (:34), `_involutive` (:65),
`extendedInversionEquiv` (:83); `spherePower` (:101), exact `spherePower_inversion_mul_norm_sq`
(:124); side-transport iffs (:185,:220); ball iffs (:254,:271); punctured set lemmas (:288,:304,:370);
δ=0 half-space (:336,:360); affine-pole replication (:385–613).

At the time of this audit, the only link between files was the import at `:3`: `pushRegion` was never
applied to `extendedInversionEquiv`. The foundation pass now supplies selected-side and payload
transport specializations. Preservation of compactified regular-open round sides,
which was open at the time of this audit, is now T9's checked bridge theorem.
The chamber hole diagnosed here is now covered by a four-chamber regression theorem; see §2.
Decorative at the time: `fullPivot_transposes_mu`,
the two `example`s (:249/:253), and `containmentPivot_*` (never touches a poset). Load-bearing:
`chosenSide_eq_compl_of_separates` and all of `SphericalInversion.lean`.

## 1. Definitional groundwork — decisions to make BEFORE writing theorems

**(D1) Walls abstractly indexed; geometry a side condition.** Don't build the poset on `Set X`
(`⊆` undecidable → kills `IncidenceAlgebra` + `decide`). Use `Fintype W` with
`[PartialOrder W] [DecidableLE W] [DecidableEq W]`, a `region : W → Set X`, and semantic hyps:
```lean
structure IsLaminarFamily (region : W → Set X) : Prop where
  mono    : ∀ v w, v ≤ w ↔ region v ⊆ region w
  laminar : ∀ v w, (region v ∩ region w).Nonempty → v ≤ w ∨ w ≤ v
```
`LocallyFiniteOrder W` via `letI := Fintype.toLocallyFiniteOrder` (`Order/Interval/Finset/Defs.lean:631`,
an `abbrev` not an instance, needs `[DecidableLT]`).

**(D2) Payloads are items at points; masses derived.** This makes chamber transport a *theorem*, not a
relabeling *definition*:
```lean
variable {ι : Type*} (s : Finset ι) (pos : ι → X) (val : ι → ℤ)
noncomputable def massIn (A : Set X) : ℤ := ∑ i ∈ s, A.indicator (fun _ => val i) (pos i)
def totalMass : ℤ := ∑ i ∈ s, val i
```
`Set.indicator` dodges `DecidablePred (· ∈ A)`. Items never move under re-anchoring; only accounting
changes. Start with `ℤ` (reuses `moebius_inversion_bot` verbatim); generalize to `AddCommGroup` later.
This is the conversation's "payloads are signed multisets of actual items."

**(D3) Background chamber is a real element: work in `WithTop W`.** `regionT ⊤ = univ`; chambers become
uniform and `chamberT ⊤` = background (outside all roots). This is what makes the honest pivot statable
(§2). `WithBot.decidableLE` at `Order/WithBot.lean:726` (dualize for `WithTop`).

**(D4) Compactified round sides: open sides primitive.**
```lean
inductive IsOpenRoundSide : Set (OnePoint E) → Prop
  | ball (a r) (hr : 0 < r) : IsOpenRoundSide ((↑) '' Metric.ball a r)
  | exterior (a r) (hr : 0 < r) : IsOpenRoundSide ((↑) '' (Metric.closedBall a r)ᶜ ∪ {∞})
  | halfspace (u : E) (c : ℝ) (hu : u ≠ 0) : IsOpenRoundSide ((↑) '' {x | c < inner ℝ x u})
```
Exactly the three images `SphericalInversion.lean` computes (δ>0/δ<0/δ=0); the class is closed under
`extendedInversion` where "open balls only" is not (open ball ↦ complement of a *closed* ball, :271).
A wall = the unordered `{S, complement}`; closed sides come via `pushRegion_compl` (:53). ∞ bookkeeping
baked in: exterior owns ∞; neither open half-space does.

## 2. The chamber hole (gap #2): concrete plan

### 2a. Diagnosis (worked example)
Three nested spheres cut compactified space into FOUR chambers: C₀(inside S₁), C₁, C₂, C₃(outside S₃,
holds old ∞). Old sides R₁=C₀, R₂=C₀∪C₁, R₃=C₀∪C₁∪C₂ → old direct labels (c₀,c₁,c₂); **c₃ (background)
is stored nowhere**. Re-anchor with new ∞ inside C₀: new sides R₁′=C₁∪C₂∪C₃, R₂′=C₂∪C₃, R₃′=C₃; order
reverses. Correct new direct labels: wall₁↦c₁, wall₂↦c₂, wall₃↦c₃ — a **shift by one along the flip
path, importing background, demoting c₀** — not `reverse₃`'s (f₃,f₂,f₁). With (c₀,c₁,c₂,c₃)=(1,2,4,8),
T=15: `pivotZeta₃(1,2,4)=(7,6,4)` (verified :339); honest new totals = (T−1,T−3,T−7)=**(14,12,8)**.
(7,6,4)≠(14,12,8). So `pivotZeta₃` is correct as a wall-indexed shuffle, false as chamber contents.

Redeeming structure: adjoin background as `⊤`. Old extended poset R₁<R₂<R₃<⊤ chambers (C₀,C₁,C₂,C₃);
new R₃′<R₂′<R₁′<⊤′ chambers (C₃,C₂,C₁,C₀). Chamber-matching map (element ↦ new element owning same
chamber): R₁↦⊤′, R₂↦R₁′, R₃↦R₂′, ⊤↦R₃′ = order-reversal of the **4-element extended chain**. So the
full pivot is order-reversal of the *extended* poset (with background), not the wall poset. The file
reversed one level too low.

### 2b. Targets (dependency order)
**T1 — Payload layer** (afternoon). `massIn`/`totalMass` (D2) + additivity on disjoint sets,
`massIn_compl : massIn Aᶜ = totalMass − massIn A`, and bijection-equivariance. Pure `Finset.sum` +
`Set.indicator_of_mem/of_notMem`.

**T2 — Wall-level re-anchoring law** (same afternoon). *Moving ∞ replaces each separating wall's total
by (grand total − old total); non-separating walls unchanged.*
```lean
theorem massIn_chosenSide_of_separates (hs : Separates q r h) :
    massIn s pos val (chosenSide r h) = totalMass s val - massIn s pos val (chosenSide q h)
theorem massIn_chosenSide_of_not_separates (hs : ¬ Separates q r h) :
    massIn s pos val (chosenSide r h) = massIn s pos val (chosenSide q h)
```
Two lemmas (not an `if` — `Separates` is a `Prop`). Rewrite with `chosenSide_eq_compl_of_separates`
(:97)/`_eq_of_not_separates` (:88), then `massIn_compl`. Fully general honest re-anchoring at cumulative
level. First theorem where re-anchoring moves actual data. **Do this first — absurd payoff/effort.**

**T3 — Chambers + partition — IMPLEMENTED.** `chamber w = region w \ ⋃_{v<w} region v`
(all strict predecessors, not covers — avoids `⋖` decidability, unverified in 4.29). Under
`IsLaminarFamily`: `chamber_pairwise_disjoint` and `region_eq_iUnion_chamber : region w = ⋃_{v∈Iic w} chamber v`.
Prove pointwise `x ∈ region w ↔ ∃! v, v ≤ w ∧ x ∈ chamber v` (min of a nonempty chain). Do it in
`WithTop W`/`regionT` so background is included.

**T4 — THE BRIDGE (highest-value theorem) — IMPLEMENTED.** *Region mass = zeta-transform of chamber
masses; chamber masses recovered via mathlib's `IncidenceAlgebra.mu`.*
```lean
theorem massIn_region_eq_sum_chamber (hL : IsLaminarFamily region) (w : W) :
    massIn s pos val (region w) = ∑ v ∈ Finset.Iic w, massIn s pos val (chamber region v)
theorem massIn_chamber_eq_moebius (hL : IsLaminarFamily region) (w : W) :
    massIn s pos val (chamber region w)
      = ∑ v ∈ Finset.Iic w, IncidenceAlgebra.mu ℤ v w * massIn s pos val (region v)
```
First from T1+T3 (each item in `region w` in exactly one chamber; `Finset.sum_comm`). Second is verbatim
`IncidenceAlgebra.moebius_inversion_bot` (`IncidenceAlgebra.lean:578`) — **don't re-prove inversion**.
First point where `mu` computes something *geometric* — the two Möbiuses provably meet. Everything routes
through this.

**T5 — Honest pivot + indictment of `reverse₃` — IMPLEMENTED AND GENERALIZED.** (1) `decide`-checked counterexample:
build the 4-chamber chain with `X := Fin 4`, payloads (1,2,4,8), prove
`naiveNewTotals=(7,6,4) ∧ honestNewTotals=(14,12,8) ∧ naive ≠ honest := by decide`; doc-comment
`pivotZeta₃` (:319) pointing at it so the file stops overclaiming. (2) Positive: on the extended chain,
chamber-matching map = order-reversal of `WithTop W`, transports chamber masses correctly
(`extended_full_pivot_is_reversal`). State for chains (`LinearOrder W`) first; general laminar (pole in
arbitrary deepest chamber) is a harder second wave.

## 3. Direction B (tractable new content)

**T6 — `μ = I − C` for interval-chain posets + sparse update — IMPLEMENTED.** *In a poset where every
interval is a chain (laminar forests qualify), μ = 1 on diagonal, −1 on covers, 0 else.*
```lean
theorem mu_of_covBy (hchain : ∀ a b, IsChain (·≤·) (Set.Icc a b)) (h : a ⋖ b) :
    IncidenceAlgebra.mu ℤ a b = -1
theorem mu_of_lt_not_covBy (hchain …) (hab : a < b) (h : ¬ a ⋖ b) :
    IncidenceAlgebra.mu ℤ a b = 0
```
Diagonal = `mu_self` (:387); off-≤ = `apply_eq_zero_of_not_le` (:95). Induct on `(Finset.Ioc a b).card`
via `mu_eq_neg_sum_Ioc_of_ne` (:493). **Not a re-proof — no `mu`-on-chains lemma in 4.29.** Corollary:
two orders on one carrier → `μ_new − μ_old = C_old − C_new`, supported on changed covers (the sparse
update, without Woodbury).

**T7 — Inversive power transform + RCC-skeleton invariance — IMPLEMENTED.** *For two spheres missing the
pole, `d²−r²−s²` transforms by `R⁴/(δ₁δ₂)`; unsigned inversive distance is invariant; disjoint/tangent/
overlapping is preserved.*
```lean
def inversivePower (a r b s) : ℝ := ‖a - b‖^2 - r^2 - s^2
theorem inversivePower_inversion (hδ₁ hδ₂) :
    inversivePower (invertedSphereCenter R a₁ r₁) (invertedSphereRadius R r₁ a₁)
                   (invertedSphereCenter R a₂ r₂) (invertedSphereRadius R r₂ a₂)
      = (R^4 / (δ₁ * δ₂)) * inversivePower a₁ r₁ a₂ r₂
```
Reuses `invertedSphereCenter/Radius` (:111/:115); proof mirrors `spherePower_inversion_mul_norm_sq` (:124)
(`norm_sub_sq_real`, `field_simp`, `ring`). Traps: (i) image center is `(R²/δ)•a`, NOT inversion of `a`
(don't reach for `dist_inversion_inversion`, `Inversion/Basic.lean:155`); (ii) SIGNED inversive distance
flips when the pole separates the spheres (`δ₁δ₂<0`) — correct geometry (one region inside-out); state
the signed identity primary, derive `|I|`-invariance + classification. Constant `R⁴/(δ₁δ₂)` not
hand-verified — treat RHS shape as conjectured-until-`ring`-closes; classification survives any positive-
scalar correction.

**T8 — Matrix rank-≤2 single-wall update (implemented).** State matrix-side to avoid two
`PartialOrder` instances on one type: for `r r' : W→W→Prop` agreeing off wall `w`,
`zetaMatrix r' − zetaMatrix r` vanishes outside row/col `w` → rank ≤ 2. Mathlib 4.29 has
`Matrix.rank_vecMulVec_le` but no matrix-facing `rank_add_le`, so the project derives subadditivity via
`mulVecLin` ranges and submodule finrank. T6 already gives the sharper laminar sparse update with explicit
coefficients.

Implemented as `matrix_rank_sub_le_two_of_eq_off`: a matrix supported on one
row and column is decomposed into `singleRowPart + singleColumnRemainder`, two
`Matrix.vecMulVec` terms. `rank_singleWallGramFlip_sub_le_two` instantiates it
for the signed inversive Gram matrix. Woodbury itself remains an optimization
consumer, not a mathematical prerequisite.

**Do NOT touch** (research-open): realizability (∃ℝ-hard), monodromy over crossings, conformal-structure
recovery, full oriented-matroid chirotope. Skip Helly/VC cost thread (mathlib has `helly_theorem'`,
`Analysis/Convex/Radon.lean` — a trap to re-prove; no consumer yet).

## 4. Direction A remainder: the glue theorem (gap #1)

**T9 — `extendedInversion` acts on round sides (implemented).** *Transporting an open round side
through compactified inversion yields an open round side.*
```lean
theorem isOpenRoundSide_pushRegion_extendedInversion (c : E) (R : ℝ) (hR : R ≠ 0)
    {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (pushRegion (extendedInversionEquiv c R hR) S)
```
This is the first theorem using both files' main objects. The implementation checks the pole and infinity
explicitly in the centered formulas, covers all ball/exterior/half-space cases, and lifts to arbitrary
centres through a proved translation–centered-inversion–translation factorization. It went further than
the original set-level target: `extendedInversionHomeomorph` is proved under `ProperSpace`, and every
round-side constructor is proved regular open.

## 5. Direction C: renderer bridge — mostly deferred

Renderer `Mobius` (`quaternion.rs:238`) = `x ↦ (ax+b)(cx+d)⁻¹` over quaternions on pure-imaginary `x`.
Mathlib has `ℍ[ℝ]` normed division ring but NO SL(2,ℍ) Möbius action, no compactified action, no
imaginary-subspace preservation. Full bridge = week-plus greenfield; payoff only for engine-level certified
claims. **Defer.**

**T10 (implemented generator slice):** the four generators the renderer builds — `translation`,
`scale`, `rotation`, `inversion` (`quaternion.rs:264–276`) — each induce `OnePoint E ≃ OnePoint E`
preserving `IsOpenRoundSide`. The Lean API is `RoundSideAutomorphism`, with certified `translation`,
`scale`, `orthogonal`, `inversion`, and `trans` constructors. “Every renderer map is a product of these
generators” remains an engine-level fact; the full quaternionic factorization theorem is deliberately
outside the formal claim.

## 6. Traps
1. Re-proving mathlib: `moebius_inversion_bot/_top`, `Matrix.invOf_add_mul_mul`, `helly_theorem'`,
   `mu_toDual`. Any "ζ,μ invert" is done.
2. Building on `OrderDual` re-exports (`fullPivot_transposes_mu`): wall-poset reversal is the semantically
   wrong op (§2); honest reversal is on `WithTop W`.
3. Two order instances on one type — use explicit relations (T8) or two index types + `Equiv` (T5).
4. `Decidable (a ⋖ b)` unverified — avoid `⋖` in definitions/`if`s, only in hypotheses.
5. `Set`-indexed posets — undecidable `⊆` poisons `IncidenceAlgebra`+`decide`; keep D1's split.
6. Open/closed asymmetry — "balls map to balls" without exterior/half-space cases is false (:271).
7. Signed vs unsigned inversive distance — the separating-pole sign flip is a feature.
8. Research-open — realizability, monodromy, conformal recovery, full SL(2,ℍ).

## 7. Completed implementation sequence

The work landed in this dependency order:
1. **T1** payload layer — afternoon
2. **T2** wall-level re-anchoring law — same afternoon
3. **T3** laminar chambers + partition on `WithTop W` — 1–2 days
4. **T4** the bridge (region mass = ζ of chamber masses; chamber via `mu`) — 1 day
   ← **HIGHEST VALUE: first place the two Möbiuses provably meet through geometry.**
5. **T5** honest chain pivot + `decide` counterexample `(7,6,4)≠(14,12,8)` — afternoon–1 day
6. **T6** `μ = I − C` for interval-chain posets + sparse update — 1–2 days
7. **T9** glue theorem: `IsOpenRoundSide` preserved by `pushRegion ∘ extendedInversionEquiv` — 2–4 days
8. **T7** inversive-power + RCC classification invariance — 2–3 days
9. **T10** generator-level renderer bridge
10. **T8** generic matrix rank-≤2 update, last

Steps 1–5 convert the formalization from "algebra narrated as geometry" to "geometry checked as algebra."
Steps 6–10 add the sparse-update, inversive, topology, and renderer-facing layers.

---

## 8. Refinement (implemented in T7): the inversive Gram matrix

T7's `inversivePower` values across all wall pairs form a symmetric matrix `G` (signed inversive
distances). Orientation (which side is "inside") is a diagonal ±1 matrix `D`. Reorientation = conjugation:
```
G_new = D · G_old · D
```
- `|Gᵢⱼ|` = crossing/tangency classification = **anchor-invariant (Layer 1)**.
- signs of `G` = current orientation = **Layer 2**.
- Flipping one wall = flipping one `D` entry = negating one row+column of `G` — the SAME support as the
  T8 rank-≤2 kernel update (the Gram picture and the Woodbury picture coincide).

This is a concrete finite symmetric matrix (decidable, no abstract pocset) and may be the single cleanest
Layer-1/Layer-2 target. Suggested theorems: define signed inversive distance (mathlib likely has no
inversive-distance/cross-ratio API — build from inner products); prove `|G|` Möbius-invariant; prove
`G_new = D·G_old·D` under reorientation; connect one-entry sign flip to the single-wall kernel update.

## 9. The `Z_new · R · M_old` framing of the chamber hole
The honest re-anchoring transport factors as `new = Z_new · R · M_old · old`, where `R` is the *semantic
chamber reassignment* — exactly Fable's `extended_full_pivot_is_reversal` map (§2a: R₁↦⊤′, R₂↦R₁′,
R₃↦R₂′, ⊤↦R₃′, the order-reversal of the extended `WithTop W` chain). `Z` and `M` are pure bookkeeping;
without a nontrivial `R` they telescope (why `Z_new·M_old` alone looked trivial). All geometric content
lives in `R`. This is now implemented as `ChamberReassignment`: the endpoint
mass theorem proves the factorization for arbitrary finite chamber models,
the path theorem composes two moves (and hence inductively any finite path),
and `reverseFiniteChain_regionMass_eq_semanticContainmentPivot` specializes it
to full reversal of any `Fin n` chain including background.

## 10. Corrected RCC8 allocation (reference)
| Information | Required structure |
|---|---|
| Equality | poset |
| Proper part (no tangency distinction) | containment poset |
| Overlap / PO | meet-semilattice / Boolean algebra (needs intersection element; orientation-dependent) |
| Wall transversality (all 4 sectors nonempty) | halfspace pocset (anchor-invariant) |
| DC vs EC | contact algebra |
| TPP vs NTPP | containment + contact-with-complement (A contacts B*) |
| Quantitative round-wall relation | inversive distance / Gram data |

Note: DC/EC and TPP/NTPP are the SAME contact primitive `C` applied to `(A,B)` vs `(A,B*)` — Lean needs
only ONE contact relation for both tangency splits. Wall transversality ≠ region-PO: transversal walls
give PO under every orientation, but non-transversal walls' region-relation is orientation-derived.
