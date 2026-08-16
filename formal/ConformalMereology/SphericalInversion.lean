import Mathlib.Geometry.Euclidean.Inversion.Basic
import Mathlib.Analysis.InnerProductSpace.Continuous
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.Topology.Compactification.OnePoint.Basic
import Mathlib.Topology.MetricSpace.ProperSpace
import Mathlib.Topology.MetricSpace.Bounded
import Mathlib.Topology.Algebra.Order.Field

/-!
# Spherical inversion and round sides

This file connects the abstract halfspace story to mathlib's Euclidean
inversion.  There are two deliberately separate models:

* mathlib's affine inversion fixes its centre, which keeps the function inside
  the affine space;
* `extendedInversion` acts on the one-point compactification and swaps the
  centre with infinity, as conformal geometry requires.

For the ball formula we work in an arbitrary real inner-product space.  This
is dimension-independent.  Translating the inversion centre to the origin is
only a choice of coordinates.
-/

open Metric Set Filter

namespace ConformalMereology

noncomputable section

section CompactifiedInversion

variable {V P : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MetricSpace P] [NormedAddTorsor V P]

/-- Spherical inversion on the conformal compactification.  Unlike mathlib's
affine-space convention, this sends the pole to infinity and infinity to the
pole. -/
def extendedInversion (c : P) (R : ℝ) : OnePoint P → OnePoint P := by
  classical
  exact OnePoint.rec (OnePoint.some c) fun x =>
    if x = c then OnePoint.infty else
      OnePoint.some (EuclideanGeometry.inversion c R x)

@[simp]
theorem extendedInversion_infty (c : P) (R : ℝ) :
    extendedInversion c R OnePoint.infty = (c : OnePoint P) :=
  rfl

@[simp]
theorem extendedInversion_center (c : P) (R : ℝ) :
    extendedInversion c R (c : OnePoint P) = OnePoint.infty := by
  classical
  change (if c = c then OnePoint.infty else
    OnePoint.some (EuclideanGeometry.inversion c R c)) = OnePoint.infty
  simp

@[simp]
theorem extendedInversion_coe_of_ne {c x : P} (R : ℝ) (hx : x ≠ c) :
    extendedInversion c R (x : OnePoint P) =
      OnePoint.some (EuclideanGeometry.inversion c R x) := by
  classical
  change (if x = c then OnePoint.infty else
    OnePoint.some (EuclideanGeometry.inversion c R x)) =
      OnePoint.some (EuclideanGeometry.inversion c R x)
  rw [if_neg hx]

/-- The compactified inversion is genuinely involutive: it swaps the affine
pole with infinity and uses mathlib's involution everywhere else. -/
theorem extendedInversion_involutive (c : P) {R : ℝ} (hR : R ≠ 0) :
    Function.Involutive (extendedInversion c R) := by
  intro z
  refine OnePoint.rec ?_ (fun x => ?_) z
  · simp
  · by_cases hx : x = c
    · subst x
      simp
    · rw [extendedInversion_coe_of_ne R hx]
      have hIx : EuclideanGeometry.inversion c R x ≠ c := by
        intro h
        exact hx ((EuclideanGeometry.inversion_eq_center (V := V) hR).mp h)
      rw [extendedInversion_coe_of_ne R hIx]
      exact congrArg (fun y : P => (y : OnePoint P))
        (EuclideanGeometry.inversion_inversion c hR x)

/-- The compactified spherical inversion as an actual equivalence, suitable
for transporting regions and containment orders. -/
def extendedInversionEquiv (c : P) (R : ℝ) (hR : R ≠ 0) :
    OnePoint P ≃ OnePoint P :=
  (extendedInversion_involutive c hR).toPerm

@[simp]
theorem extendedInversionEquiv_apply (c : P) (R : ℝ) (hR : R ≠ 0)
    (x : OnePoint P) :
    extendedInversionEquiv c R hR x = extendedInversion c R x :=
  rfl

@[simp]
theorem extendedInversionEquiv_symm_apply
    (c : P) (R : ℝ) (hR : R ≠ 0) (x : OnePoint P) :
    (extendedInversionEquiv c R hR).symm x = extendedInversion c R x := by
  apply (extendedInversionEquiv c R hR).injective
  rw [Equiv.apply_symm_apply]
  exact (extendedInversion_involutive c hR x).symm

/-- Membership in the image of compactified inversion can be tested by
applying the same inversion once more. -/
theorem mem_image_extendedInversionEquiv_iff
    (c : P) (R : ℝ) (hR : R ≠ 0) (S : Set (OnePoint P)) (x : OnePoint P) :
    x ∈ extendedInversionEquiv c R hR '' S ↔ extendedInversion c R x ∈ S := by
  rw [Equiv.image_eq_preimage_symm]
  simp only [Set.mem_preimage, extendedInversionEquiv_symm_apply]

/-- Compactified spherical inversion is continuous when closed bounded sets
are compact.  Properness is the precise hypothesis needed at infinity: it
turns metric escape into escape from every compact set.

The proof treats the two conformal points explicitly.  Near the pole, the
inverse-distance formula sends every sufficiently small punctured ball beyond
an arbitrary compact set.  At infinity, the same formula converges to the
pole. -/
theorem continuous_extendedInversion [ProperSpace P]
    (c : P) (R : ℝ) (hR : R ≠ 0) :
    Continuous (extendedInversion c R) := by
  rw [OnePoint.continuous_iff]
  constructor
  · have hdist : Tendsto (fun x : P => dist x c)
        (coclosedCompact P) atTop := by
      rw [Filter.coclosedCompact_eq_cocompact]
      exact tendsto_dist_right_cocompact_atTop c
    have hquot : Tendsto (fun x : P => R ^ 2 / dist x c)
        (coclosedCompact P) (nhds 0) := by
      simpa [div_eq_mul_inv] using
        (tendsto_const_nhds.mul hdist.inv_tendsto_atTop)
    have hinv : Tendsto (fun x : P => EuclideanGeometry.inversion c R x)
        (coclosedCompact P) (nhds c) := by
      rw [tendsto_iff_dist_tendsto_zero]
      simpa only [EuclideanGeometry.dist_inversion_center] using hquot
    have hcoe : Tendsto
        (fun x : P => OnePoint.some (EuclideanGeometry.inversion c R x))
        (coclosedCompact P) (nhds (c : OnePoint P)) := by
      simpa [Function.comp_def] using
        (OnePoint.continuous_coe.tendsto c).comp hinv
    apply hcoe.congr'
    filter_upwards [isCompact_singleton.compl_mem_coclosedCompact_of_isClosed
      isClosed_singleton] with x hx
    exact (extendedInversion_coe_of_ne R hx).symm
  · rw [continuous_iff_continuousAt]
    intro x
    by_cases hx : x = c
    · subst x
      change Tendsto (fun x : P => extendedInversion c R (x : OnePoint P))
        (nhds c) (nhds (extendedInversion c R (c : OnePoint P)))
      rw [extendedInversion_center]
      apply (OnePoint.le_nhds_infty).2
      intro s hsclosed hscompact
      rw [Filter.mem_map]
      obtain ⟨r, hsr⟩ := hscompact.isBounded.subset_closedBall c
      let B : ℝ := max r 1
      have hB : 0 < B := lt_of_lt_of_le zero_lt_one (le_max_right r 1)
      have hsB : s ⊆ Metric.closedBall c B :=
        hsr.trans (Metric.closedBall_subset_closedBall (le_max_left r 1))
      have hε : 0 < R ^ 2 / B := div_pos (sq_pos_of_ne_zero hR) hB
      refine mem_of_superset (Metric.ball_mem_nhds c hε) ?_
      intro y hy
      by_cases hyc : y = c
      · subst y
        simp
      · left
        refine ⟨EuclideanGeometry.inversion c R y, ?_, ?_⟩
        · intro hmem
          have hle : dist (EuclideanGeometry.inversion c R y) c ≤ B :=
            hsB hmem
          have hyd : dist y c < R ^ 2 / B := by
            simpa [Metric.mem_ball] using hy
          have hydpos : 0 < dist y c := dist_pos.mpr hyc
          have hmul : B * dist y c < R ^ 2 := by
            rw [mul_comm]
            exact (lt_div_iff₀ hB).mp hyd
          have hlt : B < R ^ 2 / dist y c :=
            (lt_div_iff₀ hydpos).mpr hmul
          rw [← EuclideanGeometry.dist_inversion_center] at hlt
          exact (not_lt_of_ge hle) hlt
        · exact (extendedInversion_coe_of_ne R hyc).symm
    · have hinv : ContinuousAt
          (fun y : P => EuclideanGeometry.inversion c R y) x :=
        continuousAt_const.inversion continuousAt_const continuousAt_id hx
      have hcoe : ContinuousAt
          (fun y : P => OnePoint.some (EuclideanGeometry.inversion c R y)) x := by
        simpa [Function.comp_def] using
          OnePoint.continuous_coe.continuousAt.comp hinv
      apply hcoe.congr_of_eventuallyEq
      filter_upwards [eventually_ne_nhds hx] with y hy
      exact extendedInversion_coe_of_ne R hy

/-- Spherical inversion on the conformal compactification as a genuine
homeomorphism. -/
def extendedInversionHomeomorph [ProperSpace P]
    (c : P) (R : ℝ) (hR : R ≠ 0) : OnePoint P ≃ₜ OnePoint P :=
  Continuous.homeoOfEquivCompactToT2 (f := extendedInversionEquiv c R hR)
    (continuous_extendedInversion c R hR)

@[simp]
theorem extendedInversionHomeomorph_apply [ProperSpace P]
    (c : P) (R : ℝ) (hR : R ≠ 0) (x : OnePoint P) :
    extendedInversionHomeomorph c R hR x = extendedInversion c R x :=
  rfl

end CompactifiedInversion

section BallFormula

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The bounded open side of a round sphere, embedded in the conformal
one-point compactification. -/
def compactifiedBall (a : E) (r : ℝ) : Set (OnePoint E) :=
  OnePoint.some '' Metric.ball a r

/-- The unbounded open side of a round sphere.  It contains the conformal
point at infinity and omits the boundary sphere. -/
def compactifiedExterior (a : E) (r : ℝ) : Set (OnePoint E) :=
  OnePoint.some '' (Metric.closedBall a r)ᶜ ∪ {OnePoint.infty}

/-- An open affine half-space in the conformal compactification.  Its
boundary hyperplane contains infinity, so neither open side owns infinity. -/
def compactifiedHalfspace (u : E) (q : ℝ) : Set (OnePoint E) :=
  OnePoint.some '' {x | q < inner ℝ x u}

/-- The three concrete kinds of open side bounded by a round hypersphere in
the conformal compactification: a ball, its strict exterior, or an affine
half-space. -/
inductive IsOpenRoundSide : Set (OnePoint E) → Prop
  | ball (a : E) (r : ℝ) (hr : 0 < r) :
      IsOpenRoundSide (compactifiedBall a r)
  | exterior (a : E) (r : ℝ) (hr : 0 < r) :
      IsOpenRoundSide (compactifiedExterior a r)
  | halfspace (u : E) (q : ℝ) (hu : u ≠ 0) :
      IsOpenRoundSide (compactifiedHalfspace u q)

omit [InnerProductSpace ℝ E] in
@[simp]
theorem infty_not_mem_compactifiedBall (a : E) (r : ℝ) :
    OnePoint.infty ∉ compactifiedBall a r :=
  OnePoint.infty_notMem_image_coe

omit [InnerProductSpace ℝ E] in
@[simp]
theorem infty_mem_compactifiedExterior (a : E) (r : ℝ) :
    OnePoint.infty ∈ compactifiedExterior a r := by
  simp [compactifiedExterior]

@[simp]
theorem infty_not_mem_compactifiedHalfspace (u : E) (q : ℝ) :
    OnePoint.infty ∉ compactifiedHalfspace u q :=
  OnePoint.infty_notMem_image_coe

/-- These concrete round sides really are open in the compactification of a
proper inner-product space. -/
theorem IsOpenRoundSide.isOpen [ProperSpace E]
    {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) : IsOpen S := by
  cases hS with
  | ball a r hr =>
      exact OnePoint.isOpen_image_coe.2 Metric.isOpen_ball
  | exterior a r hr =>
      rw [compactifiedExterior, ← OnePoint.compl_image_coe]
      exact OnePoint.isOpen_compl_image_coe.2
        ⟨Metric.isClosed_closedBall, isCompact_closedBall a r⟩
  | halfspace u q hu =>
      apply OnePoint.isOpen_image_coe.2
      exact isOpen_lt continuous_const (continuous_id.inner continuous_const)

/-- Signed power relative to a sphere.  Negative means open-ball interior,
zero means the boundary sphere, and positive means exterior. -/
def spherePower (a : E) (r : ℝ) (x : E) : ℝ :=
  ‖x - a‖ ^ 2 - r ^ 2

/-- The denominator controlling the image of a sphere under inversion about
the origin.  Its sign says whether the inversion pole is outside or inside
the original ball. -/
def inversionDenominator (a : E) (r : ℝ) : ℝ :=
  ‖a‖ ^ 2 - r ^ 2

/-- Centre of the image sphere when the original boundary misses the pole. -/
def invertedSphereCenter (R : ℝ) (a : E) (r : ℝ) : E :=
  (R ^ 2 / inversionDenominator a r) • a

/-- Radius of the image sphere when the original boundary misses the pole. -/
def invertedSphereRadius (R r : ℝ) (a : E) : ℝ :=
  R ^ 2 * r / |inversionDenominator a r|

/-- The exact quadratic identity behind the ball-side flip.  The positive
factor `‖x‖²` has been cleared, so the sign of the original sphere power is
the sign of the denominator times the sign of the image sphere power.

This is the dimension-independent spherical-inversion formula, in coordinates
where the pole is the origin. -/
theorem spherePower_inversion_mul_norm_sq
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : inversionDenominator a r ≠ 0) :
    spherePower a r (EuclideanGeometry.inversion (0 : E) R x) * ‖x‖ ^ 2 =
      inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x := by
  have hnx : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hnx2 : ‖x‖ ^ 2 ≠ 0 := pow_ne_zero 2 hnx
  have habsδ : |inversionDenominator a r| ≠ 0 := abs_ne_zero.mpr hδ
  unfold inversionDenominator at hδ habsδ
  simp only [spherePower, invertedSphereCenter, invertedSphereRadius,
    EuclideanGeometry.inversion, inversionDenominator]
  simp only [vsub_eq_sub, vadd_eq_add, sub_zero, add_zero, dist_zero_right]
  rw [norm_sub_sq_real, norm_sub_sq_real]
  simp only [real_inner_smul_left, real_inner_smul_right,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  field_simp [hnx, hδ, habsδ]
  simp only [sq_abs]
  ring

omit [InnerProductSpace ℝ E] in
/-- Negative sphere power is exactly membership in the corresponding open
ball (for a positive radius). -/
theorem spherePower_neg_iff_mem_ball
    (a x : E) {r : ℝ} (hr : 0 < r) :
    spherePower a r x < 0 ↔ x ∈ Metric.ball a r := by
  rw [Metric.mem_ball, dist_eq_norm, spherePower]
  constructor <;> intro h
  · nlinarith [norm_nonneg (x - a)]
  · nlinarith [norm_nonneg (x - a)]

omit [InnerProductSpace ℝ E] in
/-- Positive sphere power is exactly the strict exterior, i.e. the complement
of the closed ball. -/
theorem spherePower_pos_iff_not_mem_closedBall
    (a x : E) {r : ℝ} (hr : 0 < r) :
    0 < spherePower a r x ↔ x ∉ Metric.closedBall a r := by
  rw [Metric.mem_closedBall, dist_eq_norm, spherePower]
  constructor <;> intro h
  · nlinarith [norm_nonneg (x - a)]
  · have hlt : r < ‖x - a‖ := lt_of_not_ge h
    nlinarith [norm_nonneg (x - a)]

omit [InnerProductSpace ℝ E] in
/-- The denominator is negative exactly when the inversion pole is in the
original open ball. -/
theorem inversionDenominator_neg_iff_pole_mem_ball
    (a : E) {r : ℝ} (hr : 0 < r) :
    inversionDenominator a r < 0 ↔ (0 : E) ∈ Metric.ball a r := by
  simpa [spherePower, inversionDenominator, norm_neg] using
    (spherePower_neg_iff_mem_ball a (0 : E) hr)

omit [InnerProductSpace ℝ E] in
/-- The denominator is positive exactly when the pole is strictly outside
the original closed ball. -/
theorem inversionDenominator_pos_iff_pole_not_mem_closedBall
    (a : E) {r : ℝ} (hr : 0 < r) :
    0 < inversionDenominator a r ↔
      (0 : E) ∉ Metric.closedBall a r := by
  simpa [spherePower, inversionDenominator, norm_neg] using
    (spherePower_pos_iff_not_mem_closedBall a (0 : E) hr)

/-- If the pole is outside the original ball, inversion preserves the
interior side of the image sphere. -/
theorem spherePower_inversion_neg_iff_of_denominator_pos
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : 0 < inversionDenominator a r) :
    spherePower a r (EuclideanGeometry.inversion (0 : E) R x) < 0 ↔
      spherePower (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) x < 0 := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hid := spherePower_inversion_mul_norm_sq R r a x hx hδ.ne'
  constructor
  · intro h
    have hl : spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 < 0 := mul_neg_of_neg_of_pos h hn
    have hrhs : inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x < 0 := by
      rw [← hid]
      exact hl
    rcases (mul_neg_iff.mp hrhs) with hgood | hbad
    · exact hgood.2
    · exact (not_lt_of_ge hδ.le hbad.1).elim
  · intro h
    have hrhs : inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x < 0 :=
      mul_neg_of_pos_of_neg hδ h
    have hl : spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 < 0 := by
      rw [hid]
      exact hrhs
    rcases (mul_neg_iff.mp hl) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

/-- If the pole is inside the original ball, inversion flips interior to the
strict exterior of the image sphere. -/
theorem spherePower_inversion_neg_iff_of_denominator_neg
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : inversionDenominator a r < 0) :
    spherePower a r (EuclideanGeometry.inversion (0 : E) R x) < 0 ↔
      0 < spherePower (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) x := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hid := spherePower_inversion_mul_norm_sq R r a x hx hδ.ne
  constructor
  · intro h
    have hl : spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 < 0 := mul_neg_of_neg_of_pos h hn
    have hrhs : inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x < 0 := by
      rw [← hid]
      exact hl
    rcases (mul_neg_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hδ.le hbad.1).elim
    · exact hgood.2
  · intro h
    have hrhs : inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x < 0 :=
      mul_neg_of_neg_of_pos hδ h
    have hl : spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 < 0 := by
      rw [hid]
      exact hrhs
    rcases (mul_neg_iff.mp hl) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

/-- With the pole outside, inversion also preserves the strict exterior
side. -/
theorem spherePower_inversion_pos_iff_of_denominator_pos
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : 0 < inversionDenominator a r) :
    0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) ↔
      0 < spherePower (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) x := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hid := spherePower_inversion_mul_norm_sq R r a x hx hδ.ne'
  constructor
  · intro h
    have hl : 0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 := mul_pos h hn
    have hrhs : 0 < inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x := by
      rw [← hid]
      exact hl
    rcases (mul_pos_iff.mp hrhs) with hgood | hbad
    · exact hgood.2
    · exact (not_lt_of_ge hδ.le hbad.1).elim
  · intro h
    have hrhs : 0 < inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x := mul_pos hδ h
    have hl : 0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 := by
      rw [hid]
      exact hrhs
    rcases (mul_pos_iff.mp hl) with hgood | hbad
    · exact hgood.1
    · exact (not_lt_of_ge hn.le hbad.2).elim

/-- With the pole inside, inversion exchanges the strict exterior with the
interior of the image sphere. -/
theorem spherePower_inversion_pos_iff_of_denominator_neg
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : inversionDenominator a r < 0) :
    0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) ↔
      spherePower (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) x < 0 := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hid := spherePower_inversion_mul_norm_sq R r a x hx hδ.ne
  constructor
  · intro h
    have hl : 0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 := mul_pos h hn
    have hrhs : 0 < inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x := by
      rw [← hid]
      exact hl
    rcases (mul_pos_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hδ.le hbad.1).elim
    · exact hgood.2
  · intro h
    have hrhs : 0 < inversionDenominator a r *
        spherePower (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) x :=
      mul_pos_of_neg_of_neg hδ h
    have hl : 0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 := by
      rw [hid]
      exact hrhs
    rcases (mul_pos_iff.mp hl) with hgood | hbad
    · exact hgood.1
    · exact (not_lt_of_ge hn.le hbad.2).elim

/-- Ball-to-ball form of the outside-pole theorem, away from the pole. -/
theorem inversion_mem_ball_iff_mem_ball_of_pole_outside
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < inversionDenominator a r) :
    EuclideanGeometry.inversion (0 : E) R x ∈ Metric.ball a r ↔
      x ∈ Metric.ball (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  have hs : 0 < invertedSphereRadius R r a := by
    unfold invertedSphereRadius
    exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr)
      (abs_pos.mpr hδ.ne')
  rw [← spherePower_neg_iff_mem_ball a _ hr,
    spherePower_inversion_neg_iff_of_denominator_pos R r a x hx hδ,
    spherePower_neg_iff_mem_ball _ _ hs]

/-- Ball-to-complement form of the inside-pole theorem, away from the pole.
An open ball becomes the strict exterior, the complement of the image's
closed ball. -/
theorem inversion_mem_ball_iff_not_mem_closedBall_of_pole_inside
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r < 0) :
    EuclideanGeometry.inversion (0 : E) R x ∈ Metric.ball a r ↔
      x ∉ Metric.closedBall (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  have hs : 0 < invertedSphereRadius R r a := by
    unfold invertedSphereRadius
    exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr)
      (abs_pos.mpr hδ.ne)
  rw [← spherePower_neg_iff_mem_ball a _ hr,
    spherePower_inversion_neg_iff_of_denominator_neg R r a x hx hδ,
    spherePower_pos_iff_not_mem_closedBall _ _ hs]

/-- Exterior-to-exterior form when the pole is outside the sphere. -/
theorem inversion_not_mem_closedBall_iff_not_mem_closedBall_of_pole_outside
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < inversionDenominator a r) :
    EuclideanGeometry.inversion (0 : E) R x ∉ Metric.closedBall a r ↔
      x ∉ Metric.closedBall (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  have hs : 0 < invertedSphereRadius R r a := by
    unfold invertedSphereRadius
    exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr)
      (abs_pos.mpr hδ.ne')
  rw [← spherePower_pos_iff_not_mem_closedBall a _ hr,
    spherePower_inversion_pos_iff_of_denominator_pos R r a x hx hδ,
    spherePower_pos_iff_not_mem_closedBall _ _ hs]

/-- Exterior-to-ball form when the pole is inside the sphere. -/
theorem inversion_not_mem_closedBall_iff_mem_ball_of_pole_inside
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r < 0) :
    EuclideanGeometry.inversion (0 : E) R x ∉ Metric.closedBall a r ↔
      x ∈ Metric.ball (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  have hs : 0 < invertedSphereRadius R r a := by
    unfold invertedSphereRadius
    exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr)
      (abs_pos.mpr hδ.ne)
  rw [← spherePower_pos_iff_not_mem_closedBall a _ hr,
    spherePower_inversion_pos_iff_of_denominator_neg R r a x hx hδ,
    spherePower_neg_iff_mem_ball _ _ hs]

/-- Set-level ball image formula on the punctured affine space: outside-pole
inversion carries a ball side to a ball side.  Since inversion is involutive,
this preimage equality is also the corresponding image equality. -/
theorem preimage_ball_inter_punctured_of_pole_outside
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < inversionDenominator a r) :
    (EuclideanGeometry.inversion (0 : E) R ⁻¹' Metric.ball a r) ∩
        ({0} : Set E)ᶜ =
      Metric.ball (invertedSphereCenter R a r)
          (invertedSphereRadius R r a) ∩ ({0} : Set E)ᶜ := by
  ext x
  by_cases hx : x = 0
  · simp [hx]
  · simp [hx,
      inversion_mem_ball_iff_mem_ball_of_pole_outside R a x hx hR hr hδ]

/-- Set-level ball/complement formula on the punctured affine space:
inside-pole inversion carries an open ball to the complement of the image's
closed ball. -/
theorem preimage_ball_inter_punctured_of_pole_inside
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r < 0) :
    (EuclideanGeometry.inversion (0 : E) R ⁻¹' Metric.ball a r) ∩
        ({0} : Set E)ᶜ =
      (Metric.closedBall (invertedSphereCenter R a r)
          (invertedSphereRadius R r a))ᶜ ∩ ({0} : Set E)ᶜ := by
  ext x
  by_cases hx : x = 0
  · simp [hx]
  · simp [hx,
      inversion_mem_ball_iff_not_mem_closedBall_of_pole_inside
        R a x hx hR hr hδ]

/-- When the boundary sphere passes through the pole, its image is not a
sphere of finite radius.  The quadratic equation loses its `‖x‖²` term and
becomes an affine hyperplane equation. -/
theorem spherePower_inversion_mul_norm_sq_of_denominator_zero
    (R r : ℝ) (a x : E) (hx : x ≠ 0)
    (hδ : inversionDenominator a r = 0) :
    spherePower a r (EuclideanGeometry.inversion (0 : E) R x) * ‖x‖ ^ 2 =
      R ^ 2 * (R ^ 2 - 2 * inner ℝ x a) := by
  have hnx : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  simp only [spherePower, EuclideanGeometry.inversion, inversionDenominator] at hδ ⊢
  simp only [vsub_eq_sub, vadd_eq_add, sub_zero, add_zero, dist_zero_right]
  rw [norm_sub_sq_real]
  simp only [real_inner_smul_left, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  field_simp
  nlinarith

/-- The singular image side is the open affine half-space on the indicated
side of `2 * inner x a = R²`. -/
theorem spherePower_inversion_neg_iff_halfspace
    (R r : ℝ) (a x : E) (hx : x ≠ 0) (hR : R ≠ 0)
    (hδ : inversionDenominator a r = 0) :
    spherePower a r (EuclideanGeometry.inversion (0 : E) R x) < 0 ↔
      R ^ 2 < 2 * inner ℝ x a := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
  have hid :=
    spherePower_inversion_mul_norm_sq_of_denominator_zero R r a x hx hδ
  constructor
  · intro h
    have hl : spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 < 0 := mul_neg_of_neg_of_pos h hn
    rw [hid] at hl
    nlinarith
  · intro h
    have hrhs : R ^ 2 * (R ^ 2 - 2 * inner ℝ x a) < 0 :=
      mul_neg_of_pos_of_neg hR2 (sub_neg.mpr h)
    rw [← hid] at hrhs
    rcases (mul_neg_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

/-- The opposite open side in the singular case maps to the opposite affine
half-space. -/
theorem spherePower_inversion_pos_iff_opposite_halfspace
    (R r : ℝ) (a x : E) (hx : x ≠ 0) (hR : R ≠ 0)
    (hδ : inversionDenominator a r = 0) :
    0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) ↔
      2 * inner ℝ x a < R ^ 2 := by
  have hn : 0 < ‖x‖ ^ 2 := sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hx)
  have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
  have hid :=
    spherePower_inversion_mul_norm_sq_of_denominator_zero R r a x hx hδ
  constructor
  · intro h
    have hl : 0 < spherePower a r (EuclideanGeometry.inversion (0 : E) R x) *
        ‖x‖ ^ 2 := mul_pos h hn
    rw [hid] at hl
    nlinarith
  · intro h
    have hrhs : 0 < R ^ 2 * (R ^ 2 - 2 * inner ℝ x a) :=
      mul_pos hR2 (sub_pos.mpr h)
    rw [← hid] at hrhs
    rcases (mul_pos_iff.mp hrhs) with hgood | hbad
    · exact hgood.1
    · exact (not_lt_of_ge hn.le hbad.2).elim

/-- Ball-to-half-space form of the boundary-through-pole case. -/
theorem inversion_mem_ball_iff_halfspace
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r = 0) :
    EuclideanGeometry.inversion (0 : E) R x ∈ Metric.ball a r ↔
      R ^ 2 < 2 * inner ℝ x a := by
  rw [← spherePower_neg_iff_mem_ball a _ hr,
    spherePower_inversion_neg_iff_halfspace R r a x hx hR hδ]

/-- Exterior-to-opposite-half-space form when the boundary passes through
the inversion pole. -/
theorem inversion_not_mem_closedBall_iff_opposite_halfspace
    (R : ℝ) {r : ℝ} (a x : E) (hx : x ≠ 0) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r = 0) :
    EuclideanGeometry.inversion (0 : E) R x ∉ Metric.closedBall a r ↔
      2 * inner ℝ x a < R ^ 2 := by
  rw [← spherePower_pos_iff_not_mem_closedBall a _ hr,
    spherePower_inversion_pos_iff_opposite_halfspace R r a x hx hR hδ]

/-- Set-level singular formula: a sphere through the pole becomes an affine
hyperplane, and its ball side becomes the associated open half-space. -/
theorem preimage_ball_inter_punctured_of_pole_on_boundary
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r = 0) :
    (EuclideanGeometry.inversion (0 : E) R ⁻¹' Metric.ball a r) ∩
        ({0} : Set E)ᶜ =
      {x | R ^ 2 < 2 * inner ℝ x a} ∩ ({0} : Set E)ᶜ := by
  ext x
  by_cases hx : x = 0
  · simp [hx]
  · simp [hx, inversion_mem_ball_iff_halfspace R a x hx hR hr hδ]

/-! ### Exact compactified images of spherical sides -/

/-- When the pole is outside, the compactified image of a ball is the
corresponding image ball, including the pole and infinity cases. -/
theorem image_compactifiedBall_of_denominator_pos
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < inversionDenominator a r) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedBall a r =
      compactifiedBall (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedBall]
      have hnorm : r < ‖a‖ := by
        simpa [Metric.mem_closedBall, dist_eq_norm, norm_neg] using
          (inversionDenominator_pos_iff_pole_not_mem_closedBall a hr).mp hδ
      exact hnorm.le
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedBall, invertedSphereCenter, invertedSphereRadius]
        rw [abs_of_pos hδ]
        have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
        have hnorm : r < ‖a‖ := by
          unfold inversionDenominator at hδ
          nlinarith [norm_nonneg a]
        rw [norm_smul, Real.norm_eq_abs, abs_div,
          abs_of_nonneg hR2.le, abs_of_pos hδ]
        simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using
          mul_le_mul_of_nonneg_left hnorm.le
            (div_nonneg hR2.le hδ.le)
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedBall] using
          inversion_mem_ball_iff_mem_ball_of_pole_outside
            R a x hx hR hr hδ

/-- When the pole is inside, the compactified image of a ball is the strict
exterior and owns infinity exactly as required. -/
theorem image_compactifiedBall_of_denominator_neg
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r < 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedBall a r =
      compactifiedExterior (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simpa [compactifiedBall, compactifiedExterior] using
        (inversionDenominator_neg_iff_pole_mem_ball a hr).mp hδ
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedBall, compactifiedExterior,
          invertedSphereCenter, invertedSphereRadius]
        rw [abs_of_neg hδ]
        have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
        have hnorm : ‖a‖ < r := by
          unfold inversionDenominator at hδ
          nlinarith [norm_nonneg a]
        rw [norm_smul, Real.norm_eq_abs, abs_div,
          abs_of_nonneg hR2.le, abs_of_neg hδ]
        simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using
          mul_le_mul_of_nonneg_left hnorm.le
            (div_nonneg hR2.le (neg_nonneg.mpr hδ.le))
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedBall, compactifiedExterior] using
          inversion_mem_ball_iff_not_mem_closedBall_of_pole_inside
            R a x hx hR hr hδ

/-- A boundary sphere through the pole becomes a hyperplane, with no
punctured-space qualification left over. -/
theorem image_compactifiedBall_of_denominator_zero
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r = 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedBall a r =
      compactifiedHalfspace ((2 : ℝ) • a) (R ^ 2) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedBall, compactifiedHalfspace,
        inversionDenominator] at hδ ⊢
      nlinarith [norm_nonneg a]
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedBall, compactifiedHalfspace]
        exact sq_nonneg R
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedBall, compactifiedHalfspace,
          real_inner_smul_right] using
          inversion_mem_ball_iff_halfspace R a x hx hR hr hδ

/-- The strict exterior stays a strict exterior when the pole is outside. -/
theorem image_compactifiedExterior_of_denominator_pos
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < inversionDenominator a r) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedExterior a r =
      compactifiedExterior (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedExterior]
      have hnorm : r < ‖a‖ := by
        unfold inversionDenominator at hδ
        nlinarith [norm_nonneg a]
      exact hnorm
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedExterior, invertedSphereCenter, invertedSphereRadius]
        rw [abs_of_pos hδ, norm_smul, Real.norm_eq_abs, abs_div,
          abs_of_nonneg (sq_nonneg R), abs_of_pos hδ]
        have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
        have hnorm : r < ‖a‖ := by
          unfold inversionDenominator at hδ
          nlinarith [norm_nonneg a]
        simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using
          mul_lt_mul_of_pos_left hnorm (div_pos hR2 hδ)
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedExterior] using
          inversion_not_mem_closedBall_iff_not_mem_closedBall_of_pole_outside
            R a x hx hR hr hδ

/-- With the pole inside, the strict exterior becomes the image ball. -/
theorem image_compactifiedExterior_of_denominator_neg
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r < 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedExterior a r =
      compactifiedBall (invertedSphereCenter R a r)
        (invertedSphereRadius R r a) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedExterior]
      have hnorm : ‖a‖ < r := by
        unfold inversionDenominator at hδ
        nlinarith [norm_nonneg a]
      exact hnorm.le
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedExterior, compactifiedBall,
          invertedSphereCenter, invertedSphereRadius]
        rw [abs_of_neg hδ, norm_smul, Real.norm_eq_abs, abs_div,
          abs_of_nonneg (sq_nonneg R), abs_of_neg hδ]
        have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
        have hnorm : ‖a‖ < r := by
          unfold inversionDenominator at hδ
          nlinarith [norm_nonneg a]
        simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using
          mul_lt_mul_of_pos_left hnorm
            (div_pos hR2 (neg_pos.mpr hδ))
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedExterior, compactifiedBall] using
          inversion_not_mem_closedBall_iff_mem_ball_of_pole_inside
            R a x hx hR hr hδ

/-- The strict exterior of a boundary sphere through the pole becomes the
opposite affine half-space. -/
theorem image_compactifiedExterior_of_denominator_zero
    (R : ℝ) {r : ℝ} (a : E) (hR : R ≠ 0) (hr : 0 < r)
    (hδ : inversionDenominator a r = 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedExterior a r =
      compactifiedHalfspace ((-2 : ℝ) • a) (-(R ^ 2)) := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedExterior, compactifiedHalfspace,
        inversionDenominator] at hδ ⊢
      nlinarith [norm_nonneg a]
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedExterior, compactifiedHalfspace]
        exact sq_pos_of_ne_zero hR
      · simp only [extendedInversion_coe_of_ne R hx]
        simpa [compactifiedExterior, compactifiedHalfspace,
          real_inner_smul_right] using
          inversion_not_mem_closedBall_iff_opposite_halfspace
            R a x hx hR hr hδ

/-- Applying compactified inversion to the image of a set returns the set.
This is the set-level involution used to reverse the sphere-to-hyperplane
formulas. -/
theorem image_image_extendedInversionEquiv
    (R : ℝ) (hR : R ≠ 0) (S : Set (OnePoint E)) :
    extendedInversionEquiv (0 : E) R hR ''
        (extendedInversionEquiv (0 : E) R hR '' S) = S := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff,
    mem_image_extendedInversionEquiv_iff,
    extendedInversion_involutive (0 : E) hR z]

/-- A half-space through the inversion pole is fixed as an unoriented
hyperplane side: the positive radial factor preserves its sign. -/
theorem image_compactifiedHalfspace_zero
    (R : ℝ) (u : E) (hR : R ≠ 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedHalfspace u 0 =
      compactifiedHalfspace u 0 := by
  ext z
  rw [mem_image_extendedInversionEquiv_iff]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedHalfspace]
  | coe x =>
      by_cases hx : x = 0
      · subst x
        simp [compactifiedHalfspace]
      · simp only [extendedInversion_coe_of_ne R hx]
        simp only [compactifiedHalfspace]
        simp
        rw [EuclideanGeometry.inversion]
        simp only [vsub_eq_sub, vadd_eq_add, sub_zero, add_zero,
          dist_zero_right, real_inner_smul_left]
        have hcoef : 0 < (R / ‖x‖) ^ 2 :=
          sq_pos_of_ne_zero (div_ne_zero hR (norm_ne_zero_iff.mpr hx))
        exact mul_pos_iff_of_pos_left hcoef

/-- Centre of a sphere through the inversion pole whose image hyperplane has
normal `u` and offset `q`. -/
def halfspaceSourceCenter (R q : ℝ) (u : E) : E :=
  (R ^ 2 / (2 * q)) • u

theorem halfspaceSourceCenter_ne_zero
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : q ≠ 0) (hu : u ≠ 0) :
    halfspaceSourceCenter R q u ≠ 0 := by
  apply smul_ne_zero
  · exact div_ne_zero (pow_ne_zero 2 hR) (mul_ne_zero two_ne_zero hq)
  · exact hu

theorem halfspaceSourceRadius_pos
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : q ≠ 0) (hu : u ≠ 0) :
    0 < ‖halfspaceSourceCenter R q u‖ :=
  norm_pos_iff.mpr (halfspaceSourceCenter_ne_zero R q u hR hq hu)

theorem halfspaceSource_denominator_zero
    (R q : ℝ) (u : E) :
    inversionDenominator (halfspaceSourceCenter R q u)
      ‖halfspaceSourceCenter R q u‖ = 0 := by
  simp [inversionDenominator]

/-- The singular ball-image half-space normalizes to a prescribed positive
offset half-space. -/
theorem singular_positive_halfspace_eq
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : 0 < q) :
    compactifiedHalfspace
        ((2 : ℝ) • halfspaceSourceCenter R q u) (R ^ 2) =
      compactifiedHalfspace u q := by
  ext z
  induction z using OnePoint.rec with
  | infty => simp [compactifiedHalfspace]
  | coe x =>
      simp [compactifiedHalfspace, halfspaceSourceCenter,
        real_inner_smul_right]
      have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
      field_simp
      constructor <;> intro h <;> nlinarith

/-- A half-space not containing the pole maps back to the corresponding
open ball through the pole. -/
theorem image_compactifiedHalfspace_of_pos
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : 0 < q) (hu : u ≠ 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedHalfspace u q =
      compactifiedBall (halfspaceSourceCenter R q u)
        ‖halfspaceSourceCenter R q u‖ := by
  let a := halfspaceSourceCenter R q u
  have ha : a ≠ 0 := halfspaceSourceCenter_ne_zero R q u hR hq.ne' hu
  have hr : 0 < ‖a‖ := norm_pos_iff.mpr ha
  have hδ : inversionDenominator a ‖a‖ = 0 := by
    simp [inversionDenominator]
  have hforward := image_compactifiedBall_of_denominator_zero
    R a hR hr hδ
  rw [singular_positive_halfspace_eq R q u hR hq] at hforward
  rw [← hforward, image_image_extendedInversionEquiv]

/-- The singular exterior-image half-space normalizes to a prescribed
negative offset half-space. -/
theorem singular_negative_halfspace_eq
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : q < 0) :
    compactifiedHalfspace
        ((-2 : ℝ) • halfspaceSourceCenter R q u) (-(R ^ 2)) =
      compactifiedHalfspace u q := by
  ext z
  induction z using OnePoint.rec with
  | infty => simp [compactifiedHalfspace]
  | coe x =>
      simp [compactifiedHalfspace, halfspaceSourceCenter,
        real_inner_smul_right]
      have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
      have halg : R ^ 2 * inner ℝ x u / q < R ^ 2 ↔
          q < inner ℝ x u := by
        constructor
        · intro h
          apply (div_lt_one_of_neg hq).mp
          apply (mul_lt_mul_iff_of_pos_left hR2).mp
          simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using h
        · intro h
          have hd := (div_lt_one_of_neg hq).mpr h
          have hm := (mul_lt_mul_iff_of_pos_left hR2).mpr hd
          simpa [div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm] using hm
      convert halg using 1
      field_simp [hq.ne]

/-- A half-space containing the pole maps back to the strict exterior of the
corresponding sphere through the pole. -/
theorem image_compactifiedHalfspace_of_neg
    (R q : ℝ) (u : E) (hR : R ≠ 0) (hq : q < 0) (hu : u ≠ 0) :
    extendedInversionEquiv (0 : E) R hR '' compactifiedHalfspace u q =
      compactifiedExterior (halfspaceSourceCenter R q u)
        ‖halfspaceSourceCenter R q u‖ := by
  let a := halfspaceSourceCenter R q u
  have ha : a ≠ 0 := halfspaceSourceCenter_ne_zero R q u hR hq.ne hu
  have hr : 0 < ‖a‖ := norm_pos_iff.mpr ha
  have hδ : inversionDenominator a ‖a‖ = 0 := by
    simp [inversionDenominator]
  have hforward := image_compactifiedExterior_of_denominator_zero
    R a hR hr hδ
  rw [singular_negative_halfspace_eq R q u hR hq] at hforward
  rw [← hforward, image_image_extendedInversionEquiv]

/-- Centred compactified inversion preserves the class of open round sides,
with all ball/exterior/hyperplane and infinity cases covered. -/
theorem IsOpenRoundSide.image_centeredExtendedInversionEquiv
    (R : ℝ) (hR : R ≠ 0) {S : Set (OnePoint E)}
    (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (extendedInversionEquiv (0 : E) R hR '' S) := by
  cases hS with
  | ball a r hr =>
      rcases lt_trichotomy (inversionDenominator a r) 0 with hδ | hδ | hδ
      · rw [image_compactifiedBall_of_denominator_neg R a hR hr hδ]
        apply IsOpenRoundSide.exterior
        unfold invertedSphereRadius
        exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr) (abs_pos.mpr hδ.ne)
      · rw [image_compactifiedBall_of_denominator_zero R a hR hr hδ]
        apply IsOpenRoundSide.halfspace
        exact smul_ne_zero two_ne_zero fun ha => by
          subst a
          simp [inversionDenominator] at hδ
          nlinarith
      · rw [image_compactifiedBall_of_denominator_pos R a hR hr hδ]
        apply IsOpenRoundSide.ball
        unfold invertedSphereRadius
        exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr) (abs_pos.mpr hδ.ne')
  | exterior a r hr =>
      rcases lt_trichotomy (inversionDenominator a r) 0 with hδ | hδ | hδ
      · rw [image_compactifiedExterior_of_denominator_neg R a hR hr hδ]
        apply IsOpenRoundSide.ball
        unfold invertedSphereRadius
        exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr) (abs_pos.mpr hδ.ne)
      · rw [image_compactifiedExterior_of_denominator_zero R a hR hr hδ]
        apply IsOpenRoundSide.halfspace
        exact smul_ne_zero (neg_ne_zero.mpr two_ne_zero) fun ha => by
          subst a
          simp [inversionDenominator] at hδ
          nlinarith
      · rw [image_compactifiedExterior_of_denominator_pos R a hR hr hδ]
        apply IsOpenRoundSide.exterior
        unfold invertedSphereRadius
        exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr) (abs_pos.mpr hδ.ne')
  | halfspace u q hu =>
      rcases lt_trichotomy q 0 with hq | hq | hq
      · rw [image_compactifiedHalfspace_of_neg R q u hR hq hu]
        exact IsOpenRoundSide.exterior _ _
          (halfspaceSourceRadius_pos R q u hR hq.ne hu)
      · subst q
        rw [image_compactifiedHalfspace_zero R u hR]
        exact IsOpenRoundSide.halfspace u 0 hu
      · rw [image_compactifiedHalfspace_of_pos R q u hR hq hu]
        exact IsOpenRoundSide.ball _ _
          (halfspaceSourceRadius_pos R q u hR hq.ne' hu)

/-! ### Translation and arbitrary inversion poles -/

/-- Translation of the affine chart, fixing the compactification point. -/
def compactifiedTranslation (t : E) : OnePoint E ≃ OnePoint E :=
  (Homeomorph.onePointCongr (IsometryEquiv.vaddConst t).toHomeomorph).toEquiv

omit [InnerProductSpace ℝ E] in
theorem image_compactifiedTranslation_ball (t a : E) (r : ℝ) :
    compactifiedTranslation t '' compactifiedBall a r =
      compactifiedBall (a + t) r := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedTranslation, compactifiedBall]
  | coe x =>
      simp [compactifiedTranslation, compactifiedBall, Metric.mem_ball,
        dist_eq_norm]
      rw [show x - t - a = x - (a + t) by abel]

omit [InnerProductSpace ℝ E] in
theorem image_compactifiedTranslation_exterior (t a : E) (r : ℝ) :
    compactifiedTranslation t '' compactifiedExterior a r =
      compactifiedExterior (a + t) r := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedTranslation, compactifiedExterior]
  | coe x =>
      simp [compactifiedTranslation, compactifiedExterior,
        Metric.mem_closedBall, dist_eq_norm]
      rw [show x - t - a = x - (a + t) by abel]

theorem image_compactifiedTranslation_halfspace (t u : E) (q : ℝ) :
    compactifiedTranslation t '' compactifiedHalfspace u q =
      compactifiedHalfspace u (q + inner ℝ t u) := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedTranslation, compactifiedHalfspace]
  | coe x =>
      simp [compactifiedTranslation, compactifiedHalfspace,
        sub_eq_add_neg, inner_add_left]

/-- Translation preserves open round sides and updates their Euclidean
parameters explicitly. -/
theorem IsOpenRoundSide.image_compactifiedTranslation
    (t : E) {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (compactifiedTranslation t '' S) := by
  cases hS with
  | ball a r hr =>
      rw [image_compactifiedTranslation_ball]
      exact IsOpenRoundSide.ball _ _ hr
  | exterior a r hr =>
      rw [image_compactifiedTranslation_exterior]
      exact IsOpenRoundSide.exterior _ _ hr
  | halfspace u q hu =>
      rw [image_compactifiedTranslation_halfspace]
      exact IsOpenRoundSide.halfspace _ _ hu

/-- Inversion about an arbitrary centre factored into translation to the
origin, centred inversion, and translation back. -/
def compactifiedInversionViaTranslation
    (c : E) (R : ℝ) (hR : R ≠ 0) : OnePoint E ≃ OnePoint E :=
  (compactifiedTranslation (-c)).trans
    ((extendedInversionEquiv (0 : E) R hR).trans
      (compactifiedTranslation c))

@[simp]
theorem compactifiedInversionViaTranslation_apply
    (c : E) (R : ℝ) (hR : R ≠ 0) (z : OnePoint E) :
    compactifiedInversionViaTranslation c R hR z =
      extendedInversionEquiv c R hR z := by
  induction z using OnePoint.rec with
  | infty =>
      simp [compactifiedInversionViaTranslation, compactifiedTranslation]
  | coe x =>
      by_cases hx : x = c
      · subst x
        simp [compactifiedInversionViaTranslation, compactifiedTranslation]
      · simp [compactifiedInversionViaTranslation, compactifiedTranslation,
          extendedInversionEquiv_apply]
        have hxc : x + -c ≠ 0 := by
          intro h
          apply hx
          apply sub_eq_zero.mp
          simpa [sub_eq_add_neg] using h
        rw [extendedInversion_coe_of_ne R hxc]
        rw [extendedInversion_coe_of_ne R hx]
        simp [EuclideanGeometry.inversion, dist_eq_norm]
        rw [sub_eq_add_neg, smul_add, smul_neg]

/-- Compactified spherical inversion about an arbitrary centre preserves
open round sides.  This is the full set-level sphere/plane transport theorem,
not merely a result away from the pole. -/
theorem IsOpenRoundSide.image_extendedInversionEquiv
    (c : E) (R : ℝ) (hR : R ≠ 0) {S : Set (OnePoint E)}
    (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (extendedInversionEquiv c R hR '' S) := by
  have h₁ := hS.image_compactifiedTranslation (-c)
  have h₂ := h₁.image_centeredExtendedInversionEquiv R hR
  have h₃ := h₂.image_compactifiedTranslation c
  have heq : compactifiedInversionViaTranslation c R hR =
      extendedInversionEquiv c R hR := by
    ext z
    exact compactifiedInversionViaTranslation_apply c R hR z
  rw [← heq]
  have himage : compactifiedInversionViaTranslation c R hR '' S =
      compactifiedTranslation c ''
        (extendedInversionEquiv (0 : E) R hR ''
          (compactifiedTranslation (-c) '' S)) := by
    rw [Set.image_image, Set.image_image]
    rfl
  rw [himage]
  exact h₃

/-! ### Scale and orthogonal renderer generators -/

/-- Nonzero uniform scale on the affine chart, fixing infinity. -/
def compactifiedScale (s : ℝ) (hs : s ≠ 0) : OnePoint E ≃ OnePoint E :=
  (Homeomorph.onePointCongr (Homeomorph.smulOfNeZero s hs)).toEquiv

theorem image_compactifiedScale_ball
    (s : ℝ) (hs : s ≠ 0) (a : E) (r : ℝ) :
    compactifiedScale s hs '' compactifiedBall a r =
      compactifiedBall (s • a) (|s| * r) := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedScale, compactifiedBall]
  | coe x =>
      simp [compactifiedScale, compactifiedBall, Metric.mem_ball, dist_eq_norm]
      have hsabs : 0 < |s| := abs_pos.mpr hs
      have heq : ‖x - s • a‖ = |s| * ‖s⁻¹ • x - a‖ := by
        calc
          ‖x - s • a‖ = ‖s • (s⁻¹ • x - a)‖ := by
            congr 1
            simp [smul_sub, smul_smul, hs]
          _ = |s| * ‖s⁻¹ • x - a‖ := by
            rw [norm_smul, Real.norm_eq_abs]
      rw [heq]
      exact (mul_lt_mul_iff_of_pos_left hsabs).symm

theorem image_compactifiedScale_exterior
    (s : ℝ) (hs : s ≠ 0) (a : E) (r : ℝ) :
    compactifiedScale s hs '' compactifiedExterior a r =
      compactifiedExterior (s • a) (|s| * r) := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedScale, compactifiedExterior]
  | coe x =>
      simp [compactifiedScale, compactifiedExterior,
        Metric.mem_closedBall, dist_eq_norm]
      have hsabs : 0 < |s| := abs_pos.mpr hs
      have heq : ‖x - s • a‖ = |s| * ‖s⁻¹ • x - a‖ := by
        calc
          ‖x - s • a‖ = ‖s • (s⁻¹ • x - a)‖ := by
            congr 1
            simp [smul_sub, smul_smul, hs]
          _ = |s| * ‖s⁻¹ • x - a‖ := by
            rw [norm_smul, Real.norm_eq_abs]
      rw [heq]
      exact (mul_lt_mul_iff_of_pos_left hsabs).symm

theorem image_compactifiedScale_halfspace
    (s : ℝ) (hs : s ≠ 0) (u : E) (q : ℝ) :
    compactifiedScale s hs '' compactifiedHalfspace u q =
      compactifiedHalfspace (s⁻¹ • u) q := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedScale, compactifiedHalfspace]
  | coe x =>
      simp [compactifiedScale, compactifiedHalfspace,
        real_inner_smul_left, real_inner_smul_right]

/-- Uniform nonzero scale preserves the class of open round sides. -/
theorem IsOpenRoundSide.image_compactifiedScale
    (s : ℝ) (hs : s ≠ 0) {S : Set (OnePoint E)}
    (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (compactifiedScale s hs '' S) := by
  cases hS with
  | ball a r hr =>
      rw [image_compactifiedScale_ball]
      exact IsOpenRoundSide.ball _ _ (mul_pos (abs_pos.mpr hs) hr)
  | exterior a r hr =>
      rw [image_compactifiedScale_exterior]
      exact IsOpenRoundSide.exterior _ _ (mul_pos (abs_pos.mpr hs) hr)
  | halfspace u q hu =>
      rw [image_compactifiedScale_halfspace]
      exact IsOpenRoundSide.halfspace _ _
        (smul_ne_zero (inv_ne_zero hs) hu)

/-- An orthogonal linear map on the affine chart, fixing infinity.  In the
renderer this covers rotations and reflections; restricting the supplied
linear isometry to determinant `+1` recovers rotations. -/
def compactifiedOrthogonal (Q : E ≃ₗᵢ[ℝ] E) : OnePoint E ≃ OnePoint E :=
  (Homeomorph.onePointCongr Q.toHomeomorph).toEquiv

theorem image_compactifiedOrthogonal_ball
    (Q : E ≃ₗᵢ[ℝ] E) (a : E) (r : ℝ) :
    compactifiedOrthogonal Q '' compactifiedBall a r =
      compactifiedBall (Q a) r := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedOrthogonal, compactifiedBall]
  | coe x =>
      simp [compactifiedOrthogonal, compactifiedBall, Metric.mem_ball]
      have hdist : dist (Q.symm x) a = dist x (Q a) := by
        rw [dist_eq_norm, dist_eq_norm]
        symm
        simpa using Q.norm_map (Q.symm x - a)
      rw [hdist]

theorem image_compactifiedOrthogonal_exterior
    (Q : E ≃ₗᵢ[ℝ] E) (a : E) (r : ℝ) :
    compactifiedOrthogonal Q '' compactifiedExterior a r =
      compactifiedExterior (Q a) r := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedOrthogonal, compactifiedExterior]
  | coe x =>
      simp [compactifiedOrthogonal, compactifiedExterior,
        Metric.mem_closedBall]
      have hdist : dist (Q.symm x) a = dist x (Q a) := by
        rw [dist_eq_norm, dist_eq_norm]
        symm
        simpa using Q.norm_map (Q.symm x - a)
      rw [hdist]

theorem image_compactifiedOrthogonal_halfspace
    (Q : E ≃ₗᵢ[ℝ] E) (u : E) (q : ℝ) :
    compactifiedOrthogonal Q '' compactifiedHalfspace u q =
      compactifiedHalfspace (Q u) q := by
  ext z
  rw [Equiv.image_eq_preimage_symm]
  induction z using OnePoint.rec with
  | infty => simp [compactifiedOrthogonal, compactifiedHalfspace]
  | coe x =>
      simp only [compactifiedOrthogonal, compactifiedHalfspace, Set.mem_image]
      simp
      rw [← Q.inner_map_map]
      simp

/-- Every orthogonal linear equivalence preserves open round sides. -/
theorem IsOpenRoundSide.image_compactifiedOrthogonal
    (Q : E ≃ₗᵢ[ℝ] E) {S : Set (OnePoint E)}
    (hS : IsOpenRoundSide S) :
    IsOpenRoundSide (compactifiedOrthogonal Q '' S) := by
  cases hS with
  | ball a r hr =>
      rw [image_compactifiedOrthogonal_ball]
      exact IsOpenRoundSide.ball _ _ hr
  | exterior a r hr =>
      rw [image_compactifiedOrthogonal_exterior]
      exact IsOpenRoundSide.exterior _ _ hr
  | halfspace u q hu =>
      rw [image_compactifiedOrthogonal_halfspace]
      apply IsOpenRoundSide.halfspace
      intro hQu
      apply hu
      apply Q.injective
      simpa using hQu

/-- A compactified equivalence equipped with the exact property required by
the round-side renderer.  Generator words can be composed without reopening
their individual coordinate proofs. -/
structure RoundSideAutomorphism (E : Type*)
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  toEquiv : OnePoint E ≃ OnePoint E
  mapsRoundSide : ∀ {S : Set (OnePoint E)}, IsOpenRoundSide S →
    IsOpenRoundSide (toEquiv '' S)

namespace RoundSideAutomorphism

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Identity generator word. -/
def refl : RoundSideAutomorphism E where
  toEquiv := Equiv.refl _
  mapsRoundSide hS := by simpa using hS

/-- Composition of certified generator words. -/
def trans (A B : RoundSideAutomorphism E) : RoundSideAutomorphism E where
  toEquiv := A.toEquiv.trans B.toEquiv
  mapsRoundSide := by
    intro S hS
    have hA := A.mapsRoundSide hS
    have hB := B.mapsRoundSide hA
    change IsOpenRoundSide
      ((fun x => B.toEquiv (A.toEquiv x)) '' S)
    rw [← Set.image_image]
    exact hB

/-- Certified translation generator. -/
def translation (t : E) : RoundSideAutomorphism E where
  toEquiv := compactifiedTranslation t
  mapsRoundSide := IsOpenRoundSide.image_compactifiedTranslation t

/-- Certified nonzero uniform-scale generator. -/
def scale (s : ℝ) (hs : s ≠ 0) : RoundSideAutomorphism E where
  toEquiv := compactifiedScale s hs
  mapsRoundSide := IsOpenRoundSide.image_compactifiedScale s hs

/-- Certified orthogonal generator, including renderer rotations. -/
def orthogonal (Q : E ≃ₗᵢ[ℝ] E) : RoundSideAutomorphism E where
  toEquiv := compactifiedOrthogonal Q
  mapsRoundSide := IsOpenRoundSide.image_compactifiedOrthogonal Q

/-- Certified spherical-inversion generator about an arbitrary centre. -/
def inversion (c : E) (R : ℝ) (hR : R ≠ 0) : RoundSideAutomorphism E where
  toEquiv := extendedInversionEquiv c R hR
  mapsRoundSide := IsOpenRoundSide.image_extendedInversionEquiv c R hR

end RoundSideAutomorphism

/-! ### Regular-open semantics of concrete round sides -/

/-- Set-level regular openness.  This is the carrier equation underlying
`Heyting.Regular (Opens X)` without importing the higher-level mereology
module back into this geometric file. -/
def IsRegularOpenSet {X : Type*} [TopologicalSpace X] (S : Set X) : Prop :=
  interior (closure S) = S

/-- A homeomorphism sends regular-open sets to regular-open sets. -/
theorem IsRegularOpenSet.image_homeomorph
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    (h : X ≃ₜ Y) {S : Set X} (hS : IsRegularOpenSet S) :
    IsRegularOpenSet (h '' S) := by
  unfold IsRegularOpenSet at hS ⊢
  rw [← h.image_closure, ← h.image_interior, hS]

theorem isRegularOpenSet_image_homeomorph_iff
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    (h : X ≃ₜ Y) (S : Set X) :
    IsRegularOpenSet (h '' S) ↔ IsRegularOpenSet S := by
  constructor
  · intro himage
    have hback := IsRegularOpenSet.image_homeomorph h.symm himage
    have heq : h.symm '' (h '' S) = S := by
      ext x
      simp
    rw [heq] at hback
    exact hback
  · exact IsRegularOpenSet.image_homeomorph h

section RegularOpenRoundSides

variable [ProperSpace E]

/-- Closure of a compactified open ball includes its finite boundary sphere
but not infinity. -/
theorem closure_compactifiedBall (a : E) {r : ℝ} (hr : 0 < r) :
    closure (compactifiedBall a r) =
      OnePoint.some '' Metric.closedBall a r := by
  apply Set.Subset.antisymm
  · apply closure_minimal
    · exact Set.image_mono Metric.ball_subset_closedBall
    · exact OnePoint.isClosed_image_coe.2
        ⟨Metric.isClosed_closedBall, isCompact_closedBall a r⟩
  · rw [← closure_ball a hr.ne']
    exact image_closure_subset_closure_image OnePoint.continuous_coe

omit [ProperSpace E] in
/-- The interior of a finite compactified closed ball is its open ball. -/
theorem interior_image_closedBall (a : E) {r : ℝ} (hr : 0 < r) :
    interior (OnePoint.some '' Metric.closedBall a r) =
      compactifiedBall a r := by
  apply Set.Subset.antisymm
  · intro z hz
    obtain ⟨x, _, rfl⟩ := interior_subset hz
    refine ⟨x, ?_, rfl⟩
    rw [← interior_closedBall a hr.ne']
    let U : Set E :=
      OnePoint.some ⁻¹' interior (OnePoint.some '' Metric.closedBall a r)
    have hUopen : IsOpen U := isOpen_interior.preimage OnePoint.continuous_coe
    have hUsub : U ⊆ Metric.closedBall a r := by
      intro y hy
      have hy' : (y : OnePoint E) ∈
          OnePoint.some '' Metric.closedBall a r := interior_subset hy
      simpa using hy'
    exact interior_maximal hUsub hUopen hz
  · exact interior_maximal (Set.image_mono Metric.ball_subset_closedBall)
      (OnePoint.isOpen_image_coe.2 Metric.isOpen_ball)

theorem isRegularOpenSet_compactifiedBall
    (a : E) {r : ℝ} (hr : 0 < r) :
    IsRegularOpenSet (compactifiedBall a r) := by
  unfold IsRegularOpenSet
  rw [closure_compactifiedBall a hr, interior_image_closedBall a hr]

/-- The interior of the complement of an open set is always regular open. -/
theorem isRegularOpenSet_interior_compl_of_isOpen
    {X : Type*} [TopologicalSpace X] {U : Set X} (hU : IsOpen U) :
    IsRegularOpenSet (interior Uᶜ) := by
  unfold IsRegularOpenSet
  apply Set.Subset.antisymm
  · exact interior_mono (closure_minimal interior_subset hU.isClosed_compl)
  · exact interior_maximal subset_closure isOpen_interior

theorem compactifiedExterior_eq_interior_compl_ball
    (a : E) {r : ℝ} (hr : 0 < r) :
    compactifiedExterior a r = interior (compactifiedBall a r)ᶜ := by
  rw [interior_compl, closure_compactifiedBall a hr,
    OnePoint.compl_image_coe]
  rfl

theorem isRegularOpenSet_compactifiedExterior
    (a : E) {r : ℝ} (hr : 0 < r) :
    IsRegularOpenSet (compactifiedExterior a r) := by
  rw [compactifiedExterior_eq_interior_compl_ball a hr]
  exact isRegularOpenSet_interior_compl_of_isOpen
    (OnePoint.isOpen_image_coe.2 Metric.isOpen_ball)

theorem isRegularOpenSet_positive_compactifiedHalfspace
    (u : E) {q : ℝ} (hu : u ≠ 0) (hq : 0 < q) :
    IsRegularOpenSet (compactifiedHalfspace u q) := by
  let a := halfspaceSourceCenter 1 q u
  have ha : a ≠ 0 :=
    halfspaceSourceCenter_ne_zero 1 q u one_ne_zero hq.ne' hu
  have hr : 0 < ‖a‖ := norm_pos_iff.mpr ha
  have hforward := image_compactifiedHalfspace_of_pos
    1 q u one_ne_zero hq hu
  have hback : extendedInversionEquiv (0 : E) 1 one_ne_zero ''
      compactifiedBall a ‖a‖ = compactifiedHalfspace u q := by
    rw [← hforward, image_image_extendedInversionEquiv]
  have hreg := IsRegularOpenSet.image_homeomorph
    (extendedInversionHomeomorph (0 : E) 1 one_ne_zero)
    (isRegularOpenSet_compactifiedBall a hr)
  change IsRegularOpenSet
    (extendedInversionEquiv (0 : E) 1 one_ne_zero ''
      compactifiedBall a ‖a‖) at hreg
  rw [hback] at hreg
  exact hreg

/-- Every affine half-space in the compactification is regular open.  A
translation normalizes its offset to `1`, reducing to the sphere-through-pole
case above. -/
theorem isRegularOpenSet_compactifiedHalfspace
    (u : E) (q : ℝ) (hu : u ≠ 0) :
    IsRegularOpenSet (compactifiedHalfspace u q) := by
  let t : E := ((1 - q) / ‖u‖ ^ 2) • u
  have hnorm2 : ‖u‖ ^ 2 ≠ 0 :=
    pow_ne_zero 2 (norm_ne_zero_iff.mpr hu)
  have hoffset : q + inner ℝ t u = 1 := by
    simp only [t, real_inner_smul_left, real_inner_self_eq_norm_sq]
    field_simp
    ring
  have himage := image_compactifiedTranslation_halfspace t u q
  rw [hoffset] at himage
  have hregImage : IsRegularOpenSet
      (compactifiedTranslation t '' compactifiedHalfspace u q) := by
    rw [himage]
    exact isRegularOpenSet_positive_compactifiedHalfspace u hu zero_lt_one
  have hiff := isRegularOpenSet_image_homeomorph_iff
    (Homeomorph.onePointCongr (IsometryEquiv.vaddConst t).toHomeomorph)
    (compactifiedHalfspace u q)
  exact hiff.mp hregImage

/-- Every concrete open round side is genuinely regular open. -/
theorem IsOpenRoundSide.isRegularOpen
    {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) :
    IsRegularOpenSet S := by
  cases hS with
  | ball a r hr => exact isRegularOpenSet_compactifiedBall a hr
  | exterior a r hr => exact isRegularOpenSet_compactifiedExterior a hr
  | halfspace u q hu => exact isRegularOpenSet_compactifiedHalfspace u q hu

end RegularOpenRoundSides

/-! ### Two-sphere inversive invariants -/

/-- The unnormalized inversive power of two oriented spheres. -/
def inversivePower (a : E) (r : ℝ) (b : E) (s : ℝ) : ℝ :=
  ‖a - b‖ ^ 2 - r ^ 2 - s ^ 2

/-- Spherical inversion scales two-sphere inversive power by the signed
factor `R⁴ / (δ₁δ₂)`. -/
theorem inversivePower_inversion
    (R r s : ℝ) (a b : E)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    inversivePower
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) =
      (R ^ 4 / (inversionDenominator a r * inversionDenominator b s)) *
        inversivePower a r b s := by
  have habsa : |inversionDenominator a r| ≠ 0 := abs_ne_zero.mpr hδa
  have habsb : |inversionDenominator b s| ≠ 0 := abs_ne_zero.mpr hδb
  unfold inversionDenominator at hδa hδb habsa habsb
  simp only [inversivePower, invertedSphereCenter, invertedSphereRadius,
    inversionDenominator]
  rw [norm_sub_sq_real, norm_sub_sq_real]
  simp only [real_inner_smul_left, real_inner_smul_right,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  field_simp [hδa, hδb, habsa, habsb]
  simp only [sq_abs]
  ring

/-- Signed inversive distance. Its absolute value forgets side orientation
while retaining separation, tangency, and crossing. -/
noncomputable def signedInversiveDistance
    (a : E) (r : ℝ) (b : E) (s : ℝ) : ℝ :=
  inversivePower a r b s / (2 * r * s)

/-- The sign attached to one sphere by inversion. It is `+1` for positive
denominator and `-1` for negative denominator. -/
noncomputable def inversionOrientationSign (a : E) (r : ℝ) : ℝ :=
  |inversionDenominator a r| / inversionDenominator a r

omit [InnerProductSpace ℝ E] in
theorem abs_inversionOrientationSign (a : E) (r : ℝ)
    (hδ : inversionDenominator a r ≠ 0) :
    |inversionOrientationSign a r| = 1 := by
  unfold inversionOrientationSign
  rw [abs_div, abs_abs]
  field_simp [abs_ne_zero.mpr hδ]

/-- Signed inversive distance changes by the product of the two orientation
signs. -/
theorem signedInversiveDistance_inversion
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    signedInversiveDistance
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) =
      (|inversionDenominator a r| * |inversionDenominator b s| /
          (inversionDenominator a r * inversionDenominator b s)) *
        signedInversiveDistance a r b s := by
  rw [signedInversiveDistance,
    inversivePower_inversion R r s a b hδa hδb]
  unfold invertedSphereRadius signedInversiveDistance
  have habsa : |inversionDenominator a r| ≠ 0 := abs_ne_zero.mpr hδa
  have habsb : |inversionDenominator b s| ≠ 0 := abs_ne_zero.mpr hδb
  field_simp [hR, hδa, hδb, habsa, habsb]

theorem signedInversiveDistance_inversion_eq_orientationSigns
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    signedInversiveDistance
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) =
      inversionOrientationSign a r * signedInversiveDistance a r b s *
        inversionOrientationSign b s := by
  rw [signedInversiveDistance_inversion R r s a b hR hδa hδb]
  unfold inversionOrientationSign
  field_simp [hδa, hδb]

/-- Absolute inversive distance is invariant under spherical inversion. -/
theorem abs_signedInversiveDistance_inversion
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    |signedInversiveDistance
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b)| =
      |signedInversiveDistance a r b s| := by
  rw [signedInversiveDistance_inversion R r s a b hR hδa hδb,
    abs_mul]
  have hfactor :
      abs (|inversionDenominator a r| * |inversionDenominator b s| /
          (inversionDenominator a r * inversionDenominator b s)) = 1 := by
    rw [abs_div, abs_mul, abs_mul, abs_abs, abs_abs]
    field_simp [abs_ne_zero.mpr hδa, abs_ne_zero.mpr hδb]
  rw [hfactor, one_mul]

/-- Coarse wall separation: absolute inversive distance greater than one. -/
def InversivelySeparated (a : E) (r : ℝ) (b : E) (s : ℝ) : Prop :=
  1 < |signedInversiveDistance a r b s|

/-- The positive branch of separation.  For positive radii this is external
disjointness; with signed radii it records the corresponding oriented branch. -/
def InversivelyExternallySeparated (a : E) (r : ℝ) (b : E) (s : ℝ) : Prop :=
  1 < signedInversiveDistance a r b s

/-- The negative branch of separation.  For positive radii this is strict
nesting; with signed radii it records the corresponding oriented branch. -/
def InversivelyNested (a : E) (r : ℝ) (b : E) (s : ℝ) : Prop :=
  signedInversiveDistance a r b s < -1

omit [InnerProductSpace ℝ E] in
/-- Absolute separation forgets exactly which of its two oriented branches
holds: external disjointness or nesting (for positive radii). -/
theorem inversivelySeparated_iff_externallySeparated_or_nested
    (a : E) (r : ℝ) (b : E) (s : ℝ) :
    InversivelySeparated a r b s ↔
      InversivelyExternallySeparated a r b s ∨ InversivelyNested a r b s := by
  unfold InversivelySeparated InversivelyExternallySeparated InversivelyNested
  let I := signedInversiveDistance a r b s
  change 1 < |I| ↔ 1 < I ∨ I < -1
  constructor
  · intro h
    by_cases hI : 0 ≤ I
    · left
      simpa [abs_of_nonneg hI] using h
    · right
      rw [abs_of_neg (lt_of_not_ge hI)] at h
      linarith
  · rintro (h | h)
    · have hI : 0 < I := lt_trans (by norm_num) h
      simpa [abs_of_pos hI] using h
    · have hI : I < 0 := lt_trans h (by norm_num)
      rw [abs_of_neg hI]
      linarith

/-- Coarse wall tangency: absolute inversive distance equal to one. -/
def InversivelyTangent (a : E) (r : ℝ) (b : E) (s : ℝ) : Prop :=
  |signedInversiveDistance a r b s| = 1

/-- Transverse sphere crossing: absolute inversive distance less than one. -/
def InversivelyCrossing (a : E) (r : ℝ) (b : E) (s : ℝ) : Prop :=
  |signedInversiveDistance a r b s| < 1

theorem inversivelySeparated_inversion_iff
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    InversivelySeparated
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) ↔
      InversivelySeparated a r b s := by
  unfold InversivelySeparated
  rw [abs_signedInversiveDistance_inversion R r s a b hR hδa hδb]

theorem inversivelyTangent_inversion_iff
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    InversivelyTangent
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) ↔
      InversivelyTangent a r b s := by
  unfold InversivelyTangent
  rw [abs_signedInversiveDistance_inversion R r s a b hR hδa hδb]

theorem inversivelyCrossing_inversion_iff
    (R r s : ℝ) (a b : E) (hR : R ≠ 0)
    (hδa : inversionDenominator a r ≠ 0)
    (hδb : inversionDenominator b s ≠ 0) :
    InversivelyCrossing
        (invertedSphereCenter R a r) (invertedSphereRadius R r a)
        (invertedSphereCenter R b s) (invertedSphereRadius R s b) ↔
      InversivelyCrossing a r b s := by
  unfold InversivelyCrossing
  rw [abs_signedInversiveDistance_inversion R r s a b hR hδa hδb]

end BallFormula

section AffineBallFormula

variable {V P : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MetricSpace P] [NormedAddTorsor V P]

/-- Signed sphere power in an affine Euclidean space. -/
def affineSpherePower (a : P) (r : ℝ) (x : P) : ℝ :=
  dist x a ^ 2 - r ^ 2

/-- The arbitrary-pole version of the inversion denominator. -/
def affineInversionDenominator (c a : P) (r : ℝ) : ℝ :=
  dist a c ^ 2 - r ^ 2

/-- Image-sphere centre for inversion about the sphere `(c,R)`. -/
def affineInvertedSphereCenter (c : P) (R : ℝ) (a : P) (r : ℝ) : P :=
  (R ^ 2 / affineInversionDenominator c a r) • (a -ᵥ c) +ᵥ c

/-- Image-sphere radius for inversion about the sphere `(c,R)`. -/
def affineInvertedSphereRadius (c : P) (R r : ℝ) (a : P) : ℝ :=
  R ^ 2 * r / |affineInversionDenominator c a r|

omit [InnerProductSpace ℝ V] in
theorem affineSpherePower_eq_centered (c a x : P) (r : ℝ) :
    affineSpherePower a r x =
      spherePower (a -ᵥ c) r (x -ᵥ c) := by
  simp [affineSpherePower, spherePower, dist_eq_norm_vsub,
    vsub_sub_vsub_cancel_right]

omit [InnerProductSpace ℝ V] in
theorem affineInversionDenominator_eq_centered (c a : P) (r : ℝ) :
    affineInversionDenominator c a r =
      inversionDenominator (a -ᵥ c) r := by
  simp [affineInversionDenominator, inversionDenominator,
    dist_eq_norm_vsub]

theorem affineInvertedSphereCenter_vsub (c : P) (R : ℝ) (a : P) (r : ℝ) :
    affineInvertedSphereCenter c R a r -ᵥ c =
      invertedSphereCenter R (a -ᵥ c) r := by
  simp only [affineInvertedSphereCenter, vadd_vsub, invertedSphereCenter]
  rw [affineInversionDenominator_eq_centered c a r]

/-- The exact formula with an arbitrary affine inversion pole.  It is the
origin formula transported through the coordinate chart `x ↦ x -ᵥ c`. -/
theorem affineSpherePower_inversion_mul_dist_sq
    (c : P) (R r : ℝ) (a x : P) (hx : x ≠ c)
    (hδ : affineInversionDenominator c a r ≠ 0) :
    affineSpherePower a r (EuclideanGeometry.inversion c R x) * dist x c ^ 2 =
      affineInversionDenominator c a r *
        affineSpherePower (affineInvertedSphereCenter c R a r)
          (affineInvertedSphereRadius c R r a) x := by
  have hxv : x -ᵥ c ≠ 0 := vsub_ne_zero.mpr hx
  have hδv : inversionDenominator (a -ᵥ c) r ≠ 0 := by
    rwa [← affineInversionDenominator_eq_centered]
  have hI :
      EuclideanGeometry.inversion (0 : V) R (x -ᵥ c) =
        EuclideanGeometry.inversion c R x -ᵥ c := by
    simp [EuclideanGeometry.inversion, dist_eq_norm_vsub]
  have H :=
    spherePower_inversion_mul_norm_sq R r (a -ᵥ c) (x -ᵥ c) hxv hδv
  rw [hI] at H
  rw [affineSpherePower_eq_centered c a _ r,
    affineSpherePower_eq_centered c
      (affineInvertedSphereCenter c R a r) x _,
    affineInversionDenominator_eq_centered c a r,
    affineInvertedSphereCenter_vsub,
    affineInvertedSphereRadius,
    affineInversionDenominator_eq_centered c a r,
    dist_eq_norm_vsub]
  exact H

/-- Arbitrary-pole version of the singular formula.  When the original
boundary passes through `c`, its image equation is affine in the translated
coordinates `x -ᵥ c`. -/
theorem affineSpherePower_inversion_mul_dist_sq_of_denominator_zero
    (c : P) (R r : ℝ) (a x : P) (hx : x ≠ c)
    (hδ : affineInversionDenominator c a r = 0) :
    affineSpherePower a r (EuclideanGeometry.inversion c R x) * dist x c ^ 2 =
      R ^ 2 * (R ^ 2 - 2 * inner ℝ (x -ᵥ c) (a -ᵥ c)) := by
  have hxv : x -ᵥ c ≠ 0 := vsub_ne_zero.mpr hx
  have hδv : inversionDenominator (a -ᵥ c) r = 0 := by
    rwa [← affineInversionDenominator_eq_centered]
  have hI :
      EuclideanGeometry.inversion (0 : V) R (x -ᵥ c) =
        EuclideanGeometry.inversion c R x -ᵥ c := by
    simp [EuclideanGeometry.inversion, dist_eq_norm_vsub]
  have H :=
    spherePower_inversion_mul_norm_sq_of_denominator_zero
      R r (a -ᵥ c) (x -ᵥ c) hxv hδv
  rw [hI] at H
  rw [affineSpherePower_eq_centered c a _ r, dist_eq_norm_vsub]
  exact H

theorem affineSpherePower_neg_iff_mem_ball
    (a x : P) {r : ℝ} (hr : 0 < r) :
    affineSpherePower a r x < 0 ↔ x ∈ Metric.ball a r := by
  rw [Metric.mem_ball, affineSpherePower]
  constructor <;> intro h
  · nlinarith [(dist_nonneg : 0 ≤ dist x a)]
  · nlinarith [(dist_nonneg : 0 ≤ dist x a)]

theorem affineSpherePower_pos_iff_not_mem_closedBall
    (a x : P) {r : ℝ} (hr : 0 < r) :
    0 < affineSpherePower a r x ↔ x ∉ Metric.closedBall a r := by
  rw [Metric.mem_closedBall, affineSpherePower]
  constructor <;> intro h
  · nlinarith [(dist_nonneg : 0 ≤ dist x a)]
  · have hlt : r < dist x a := lt_of_not_ge h
    nlinarith [(dist_nonneg : 0 ≤ dist x a)]

theorem affineInversionDenominator_neg_iff_pole_mem_ball
    (c a : P) {r : ℝ} (hr : 0 < r) :
    affineInversionDenominator c a r < 0 ↔ c ∈ Metric.ball a r := by
  simpa [affineSpherePower, affineInversionDenominator, dist_comm] using
    (affineSpherePower_neg_iff_mem_ball a c hr)

theorem affineInversionDenominator_pos_iff_pole_not_mem_closedBall
    (c a : P) {r : ℝ} (hr : 0 < r) :
    0 < affineInversionDenominator c a r ↔
      c ∉ Metric.closedBall a r := by
  simpa [affineSpherePower, affineInversionDenominator, dist_comm] using
    (affineSpherePower_pos_iff_not_mem_closedBall a c hr)

/-- Fully affine ball-to-half-space theorem for a boundary sphere through the
inversion pole. -/
theorem affineInversion_mem_ball_iff_halfspace
    (c : P) (R : ℝ) {r : ℝ} (a x : P) (hx : x ≠ c)
    (hR : R ≠ 0) (hr : 0 < r)
    (hδ : affineInversionDenominator c a r = 0) :
    EuclideanGeometry.inversion c R x ∈ Metric.ball a r ↔
      R ^ 2 < 2 * inner ℝ (x -ᵥ c) (a -ᵥ c) := by
  have hn : 0 < dist x c ^ 2 :=
    sq_pos_of_ne_zero (dist_ne_zero.mpr hx)
  have hR2 : 0 < R ^ 2 := sq_pos_of_ne_zero hR
  have hid :=
    affineSpherePower_inversion_mul_dist_sq_of_denominator_zero
      c R r a x hx hδ
  rw [← affineSpherePower_neg_iff_mem_ball a _ hr]
  constructor
  · intro h
    have hl :
        affineSpherePower a r (EuclideanGeometry.inversion c R x) *
          dist x c ^ 2 < 0 :=
      mul_neg_of_neg_of_pos h hn
    rw [hid] at hl
    nlinarith
  · intro h
    have hrhs :
        R ^ 2 * (R ^ 2 - 2 * inner ℝ (x -ᵥ c) (a -ᵥ c)) < 0 :=
      mul_neg_of_pos_of_neg hR2 (sub_neg.mpr h)
    rw [← hid] at hrhs
    rcases (mul_neg_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

/-- Fully affine ball-to-ball theorem for a pole outside the original ball. -/
theorem affineInversion_mem_ball_iff_mem_ball_of_pole_outside
    (c : P) (R : ℝ) {r : ℝ} (a x : P) (hx : x ≠ c)
    (hR : R ≠ 0) (hr : 0 < r)
    (hδ : 0 < affineInversionDenominator c a r) :
    EuclideanGeometry.inversion c R x ∈ Metric.ball a r ↔
      x ∈ Metric.ball (affineInvertedSphereCenter c R a r)
        (affineInvertedSphereRadius c R r a) := by
  have hn : 0 < dist x c ^ 2 :=
    sq_pos_of_ne_zero (dist_ne_zero.mpr hx)
  have hs : 0 < affineInvertedSphereRadius c R r a := by
    unfold affineInvertedSphereRadius
    positivity
  have hid :=
    affineSpherePower_inversion_mul_dist_sq c R r a x hx hδ.ne'
  rw [← affineSpherePower_neg_iff_mem_ball a _ hr,
    ← affineSpherePower_neg_iff_mem_ball _ x hs]
  constructor
  · intro h
    have hl :
        affineSpherePower a r (EuclideanGeometry.inversion c R x) *
          dist x c ^ 2 < 0 :=
      mul_neg_of_neg_of_pos h hn
    rw [hid] at hl
    rcases (mul_neg_iff.mp hl) with hgood | hbad
    · exact hgood.2
    · exact (not_lt_of_ge hδ.le hbad.1).elim
  · intro h
    have hrhs :
        affineInversionDenominator c a r *
          affineSpherePower (affineInvertedSphereCenter c R a r)
            (affineInvertedSphereRadius c R r a) x < 0 :=
      mul_neg_of_pos_of_neg hδ h
    rw [← hid] at hrhs
    rcases (mul_neg_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

/-- Fully affine ball-to-complement theorem for a pole inside the original
ball. -/
theorem affineInversion_mem_ball_iff_not_mem_closedBall_of_pole_inside
    (c : P) (R : ℝ) {r : ℝ} (a x : P) (hx : x ≠ c)
    (hR : R ≠ 0) (hr : 0 < r)
    (hδ : affineInversionDenominator c a r < 0) :
    EuclideanGeometry.inversion c R x ∈ Metric.ball a r ↔
      x ∉ Metric.closedBall (affineInvertedSphereCenter c R a r)
        (affineInvertedSphereRadius c R r a) := by
  have hn : 0 < dist x c ^ 2 :=
    sq_pos_of_ne_zero (dist_ne_zero.mpr hx)
  have hs : 0 < affineInvertedSphereRadius c R r a := by
    unfold affineInvertedSphereRadius
    exact div_pos (mul_pos (sq_pos_of_ne_zero hR) hr)
      (abs_pos.mpr hδ.ne)
  have hid :=
    affineSpherePower_inversion_mul_dist_sq c R r a x hx hδ.ne
  rw [← affineSpherePower_neg_iff_mem_ball a _ hr,
    ← affineSpherePower_pos_iff_not_mem_closedBall _ x hs]
  constructor
  · intro h
    have hl :
        affineSpherePower a r (EuclideanGeometry.inversion c R x) *
          dist x c ^ 2 < 0 :=
      mul_neg_of_neg_of_pos h hn
    rw [hid] at hl
    rcases (mul_neg_iff.mp hl) with hbad | hgood
    · exact (not_lt_of_ge hδ.le hbad.1).elim
    · exact hgood.2
  · intro h
    have hrhs :
        affineInversionDenominator c a r *
          affineSpherePower (affineInvertedSphereCenter c R a r)
            (affineInvertedSphereRadius c R r a) x < 0 :=
      mul_neg_of_neg_of_pos hδ h
    rw [← hid] at hrhs
    rcases (mul_neg_iff.mp hrhs) with hbad | hgood
    · exact (not_lt_of_ge hn.le hbad.2).elim
    · exact hgood.1

end AffineBallFormula

end

end ConformalMereology
