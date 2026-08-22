import Mathlib.Combinatorics.Enumerative.IncidenceAlgebra
import Mathlib.Data.Fintype.WithTopBot
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Data.Set.Image
import Mathlib.Order.Heyting.Regular
import Mathlib.Order.Cover
import Mathlib.Order.Fin.Basic
import Mathlib.Order.Preorder.Finite
import Mathlib.Topology.Sets.Opens
import ConformalMereology.SphericalInversion
import ConformalMereology.RoundSideIndex

/-!
# Conformal mereology: the two Möbius operations

This file deliberately separates three layers:

1. A geometric Möbius map is represented only by the fact that it is a
   bijection of the conformal compactification. It therefore transports
   halfspaces while preserving inclusion and complement.
2. Choosing a Euclidean point at infinity selects one side of every wall.
   Moving that point flips exactly the walls separating the two points.
3. Möbius inversion on a finite poset is the inverse of its zeta transform.
   It is equivariant under every order isomorphism, not specifically under
   geometric Möbius maps.
4. Pivoting a containment chain changes the zeta kernel and its Möbius inverse
   together. A full chain pivot is order duality, which transposes the Möbius
   kernel.

The formal statements below test the genuine connection without asserting
that geometric inversion and incidence-algebra inversion are the same map.
-/

open scoped BigOperators

namespace ConformalMereology

section Halfspaces

variable {X Y : Type*}

/-- Transport a region through a bijection. Every genuine geometric Möbius
transformation of the conformal sphere supplies such a bijection. -/
def pushRegion (M : X ≃ Y) (h : Set X) : Set Y :=
  M '' h

/-- A bijection preserves the containment order on regions. -/
theorem pushRegion_subset_iff (M : X ≃ Y) (h k : Set X) :
    pushRegion M h ⊆ pushRegion M k ↔ h ⊆ k := by
  constructor
  · intro himage x hx
    have hMx : M x ∈ pushRegion M h := ⟨x, hx, rfl⟩
    obtain ⟨z, hz, hzx⟩ := himage hMx
    have : z = x := M.injective hzx
    simpa [this] using hz
  · intro hsubset y hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact ⟨x, hsubset hx, rfl⟩

/-- A bijection preserves the complementary pairing of halfspaces. -/
theorem pushRegion_compl (M : X ≃ Y) (h : Set X) :
    pushRegion M hᶜ = (pushRegion M h)ᶜ := by
  ext y
  constructor
  · rintro ⟨x, hx, rfl⟩ ⟨z, hz, hzx⟩
    exact hx (M.injective hzx.symm ▸ hz)
  · intro hy
    refine ⟨M.symm y, ?_, M.apply_symm_apply y⟩
    intro hx
    apply hy
    exact ⟨M.symm y, hx, M.apply_symm_apply y⟩

/-- The point-based orientation of a wall: `true` means that `q` lies in the
named halfspace. A principal ultrafilter is the collection of all true sides. -/
def orientationAt (q : X) (h : Set X) : Prop :=
  q ∈ h

/-- A wall separates two possible points at infinity exactly when their
orientation bits disagree. -/
def Separates (q r : X) (h : Set X) : Prop :=
  orientationAt q h ↔ ¬ orientationAt r h

/-- The side selected as "bounded/interior" relative to `q` is the side not
containing `q`. On the conformal sphere, `q` plays the role of infinity. -/
noncomputable def chosenSide (q : X) (h : Set X) : Set X := by
  classical
  exact if q ∈ h then hᶜ else h

theorem point_not_mem_chosenSide (q : X) (h : Set X) :
    q ∉ chosenSide q h := by
  classical
  by_cases hq : q ∈ h <;> simp [chosenSide, hq]

/-- Changing the point at infinity leaves a wall orientation unchanged when
the wall does not separate the old and new points. -/
theorem chosenSide_eq_of_not_separates
    (q r : X) (h : Set X) (hs : ¬ Separates q r h) :
    chosenSide q h = chosenSide r h := by
  classical
  by_cases hq : q ∈ h <;> by_cases hr : r ∈ h
  all_goals simp [chosenSide, Separates, orientationAt, hq, hr] at hs ⊢

/-- Changing the point at infinity complements exactly the separating walls.
This is the precise "inside-out" operation. -/
theorem chosenSide_eq_compl_of_separates
    (q r : X) (h : Set X) (hs : Separates q r h) :
    chosenSide r h = (chosenSide q h)ᶜ := by
  classical
  by_cases hq : q ∈ h <;> by_cases hr : r ∈ h
  all_goals simp [chosenSide, Separates, orientationAt, hq, hr] at hs ⊢

/-- Transporting both a point and a halfspace preserves its orientation bit. -/
theorem orientationAt_pushRegion (M : X ≃ Y) (q : X) (h : Set X) :
    orientationAt (M q) (pushRegion M h) ↔ orientationAt q h := by
  simp [orientationAt, pushRegion]

/-- Choosing a side and transporting a complementary set partition commute. -/
theorem pushRegion_chosenSide (M : X ≃ Y) (q : X) (h : Set X) :
    pushRegion M (chosenSide q h) = chosenSide (M q) (pushRegion M h) := by
  classical
  by_cases hq : q ∈ h
  · have hMq : M q ∈ pushRegion M h := ⟨q, hq, rfl⟩
    simp [chosenSide, hq, hMq, pushRegion_compl]
  · have hMq : M q ∉ pushRegion M h := by
      rintro ⟨x, hx, hxeq⟩
      exact hq (M.injective hxeq ▸ hx)
    simp [chosenSide, hq, hMq]

end Halfspaces

section RoundSideTransport

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The first concrete bridge between geometric spherical inversion and the
halfspace transport API: every compactified open round side is sent to
another compactified open round side, for an arbitrary inversion centre. -/
theorem isOpenRoundSide_pushRegion_extendedInversion
    (c : E) (R : ℝ) (hR : R ≠ 0) {S : Set (OnePoint E)}
    (hS : IsOpenRoundSide S) :
    IsOpenRoundSide
      (pushRegion (extendedInversionEquiv c R hR) S) := by
  simpa only [pushRegion] using hS.image_extendedInversionEquiv c R hR

end RoundSideTransport

section RegularOpenSides

open Heyting
open TopologicalSpace

/-- The Boolean algebra of regular-open regions.  Its complement is the
interior of the set-theoretic complement, so two open sides share neither
boundary points nor an arbitrary boundary assignment. -/
abbrev RegularOpen (X : Type*) [TopologicalSpace X] :=
  Heyting.Regular (Opens X)

namespace RegularOpen

variable {X : Type*} [TopologicalSpace X]

/-- The underlying point set of a regular-open region. -/
def carrier (u : RegularOpen X) : Set X :=
  (u : Opens X)

@[simp]
theorem carrier_top : carrier (⊤ : RegularOpen X) = Set.univ :=
  rfl

@[simp]
theorem carrier_bot : carrier (⊥ : RegularOpen X) = ∅ :=
  rfl

/-- Heyting complement on open sets is the interior of ordinary set
complement.  This is the complement operation needed for exact open spherical
sides. -/
theorem opens_compl_carrier (u : Opens X) :
    ((uᶜ : Opens X) : Set X) = interior ((u : Set X)ᶜ) := by
  apply Set.Subset.antisymm
  · apply interior_maximal
    · intro x hx hxu
      have hd : Disjoint (((uᶜ : Opens X) : Set X)) (u : Set X) :=
        Opens.coe_disjoint.mpr disjoint_compl_left
      exact Set.disjoint_left.1 hd hx hxu
    · exact (uᶜ : Opens X).isOpen
  · intro x hx
    have hx' : x ∈ Opens.interior ((u : Set X)ᶜ) := hx
    have hle : Opens.interior ((u : Set X)ᶜ) ≤ uᶜ := by
      apply (le_compl_iff_disjoint_right).2
      apply Opens.coe_disjoint.mp
      exact Set.disjoint_left.2 fun z hz hzu => (interior_subset hz) hzu
    exact hle hx'

@[simp]
theorem carrier_compl (u : RegularOpen X) :
    carrier (uᶜ) = interior ((carrier u)ᶜ) := by
  exact opens_compl_carrier (u : Opens X)

/-- Package a concrete compactified round side as an element of the exact
regular-open Boolean algebra. -/
def ofOpenRoundSide
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [ProperSpace E] {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) :
    RegularOpen (OnePoint E) :=
  ⟨⟨S, hS.isOpen⟩, by
    unfold Heyting.IsRegular
    apply Opens.ext
    rw [opens_compl_carrier, opens_compl_carrier]
    change interior (interior Sᶜ)ᶜ = S
    rw [← closure_eq_compl_interior_compl]
    exact hS.isRegularOpen⟩

@[simp]
theorem carrier_ofOpenRoundSide
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [ProperSpace E] {S : Set (OnePoint E)} (hS : IsOpenRoundSide S) :
    carrier (ofOpenRoundSide hS) = S :=
  rfl

/-- An anchor is admissible for an open wall when it lies strictly on one of
its two sides, rather than on the common boundary. -/
def IsOffWall (q : X) (h : RegularOpen X) : Prop :=
  q ∈ carrier h ∨ q ∈ carrier (hᶜ)

/-- The regular-open side not containing the admissible anchor. -/
noncomputable def chosenSide (q : X) (h : RegularOpen X) : RegularOpen X := by
  classical
  exact if q ∈ carrier h then hᶜ else h

theorem point_not_mem_chosenSide (q : X) (h : RegularOpen X)
    (hq : IsOffWall q h) :
    q ∉ carrier (chosenSide q h) := by
  classical
  rcases hq with hmem | hmem
  · rw [chosenSide, if_pos hmem]
    have hd : Disjoint (carrier h) (carrier (hᶜ)) :=
      Opens.coe_disjoint.mpr disjoint_compl_right
    exact fun hbad => Set.disjoint_left.1 hd hmem hbad
  · have hnmem : q ∉ carrier h := by
      intro hbad
      have hd : Disjoint (carrier h) (carrier (hᶜ)) :=
        Opens.coe_disjoint.mpr disjoint_compl_right
      exact Set.disjoint_left.1 hd hbad hmem
    rw [chosenSide, if_neg hnmem]
    exact hnmem

end RegularOpen

end RegularOpenSides

section PayloadMass

variable {X Y ι : Type*}

/-- Signed mass of the finite payload whose positions lie in `A`.

The payload consists of actual indexed items at points.  Re-anchoring changes
which region is counted, not the items or their values. -/
noncomputable def massIn (items : Finset ι) (position : ι → X)
    (value : ι → ℤ) (A : Set X) : ℤ := by
  classical
  exact ∑ i ∈ items, if position i ∈ A then value i else 0

/-- Total signed mass, independent of any region or anchor. -/
def totalMass (items : Finset ι) (value : ι → ℤ) : ℤ :=
  ∑ i ∈ items, value i

@[simp]
theorem massIn_empty (items : Finset ι) (position : ι → X)
    (value : ι → ℤ) :
    massIn items position value ∅ = 0 := by
  classical
  simp [massIn]

@[simp]
theorem massIn_univ (items : Finset ι) (position : ι → X)
    (value : ι → ℤ) :
    massIn items position value Set.univ = totalMass items value := by
  classical
  simp [massIn, totalMass]

/-- Complementation gives the honest accounting rule for signed payloads. -/
theorem massIn_compl (items : Finset ι) (position : ι → X)
    (value : ι → ℤ) (A : Set X) :
    massIn items position value Aᶜ =
      totalMass items value - massIn items position value A := by
  classical
  simp only [massIn, totalMass, Set.mem_compl_iff]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  by_cases hA : position i ∈ A <;> simp [hA]

/-- Moving positions and their region through the same bijection preserves
the mass. -/
theorem massIn_pushRegion (M : X ≃ Y) (items : Finset ι)
    (position : ι → X) (value : ι → ℤ) (A : Set X) :
    massIn items (M ∘ position) value (pushRegion M A) =
      massIn items position value A := by
  classical
  unfold massIn
  apply Finset.sum_congr rfl
  intro i hi
  simp [pushRegion]

/-- On a separating wall, changing the anchor replaces the old selected-side
total by its complement in the grand total. -/
theorem massIn_chosenSide_of_separates
    (items : Finset ι) (position : ι → X) (value : ι → ℤ)
    (q r : X) (h : Set X) (hs : Separates q r h) :
    massIn items position value (chosenSide r h) =
      totalMass items value - massIn items position value (chosenSide q h) := by
  rw [chosenSide_eq_compl_of_separates q r h hs, massIn_compl]

/-- On a nonseparating wall, changing the anchor leaves its selected-side
total unchanged. -/
theorem massIn_chosenSide_of_not_separates
    (items : Finset ι) (position : ι → X) (value : ι → ℤ)
    (q r : X) (h : Set X) (hs : ¬ Separates q r h) :
    massIn items position value (chosenSide r h) =
      massIn items position value (chosenSide q h) := by
  rw [chosenSide_eq_of_not_separates q r h hs]

/-- Signed mass in the carrier of an exact regular-open side. -/
noncomputable def regularMassIn [TopologicalSpace X]
    (items : Finset ι) (position : ι → X) (value : ι → ℤ)
    (A : RegularOpen X) : ℤ :=
  massIn items position value (RegularOpen.carrier A)

/-- Boolean complement has the familiar `total - old` accounting law when
payload positions avoid the wall boundary.  Without this hypothesis the two
open sides intentionally omit boundary payloads. -/
theorem regularMassIn_compl [TopologicalSpace X]
    (items : Finset ι) (position : ι → X) (value : ι → ℤ)
    (A : RegularOpen X)
    (hoff : ∀ i ∈ items, RegularOpen.IsOffWall (position i) A) :
    regularMassIn items position value Aᶜ =
      totalMass items value - regularMassIn items position value A := by
  classical
  simp only [regularMassIn, massIn, totalMass]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  have hd : Disjoint (RegularOpen.carrier A)
      (RegularOpen.carrier (Aᶜ)) :=
    TopologicalSpace.Opens.coe_disjoint.mpr disjoint_compl_right
  rcases hoff i hi with hA | hAc
  · have hnAc : position i ∉ RegularOpen.carrier (Aᶜ) :=
      fun hbad => Set.disjoint_left.1 hd hA hbad
    rw [if_neg hnAc, if_pos hA]
    ring
  · have hnA : position i ∉ RegularOpen.carrier A :=
      fun hbad => Set.disjoint_left.1 hd hbad hAc
    rw [if_pos hAc, if_neg hnA]
    ring

end PayloadMass

section SphericalPayloadTransport

variable {V P ι : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MetricSpace P] [NormedAddTorsor V P]

/-- Compactified spherical inversion transports the side selected relative to
old infinity to the side selected relative to its image, the inversion pole.
This is the first direct application of the abstract re-anchoring operation to
the concrete spherical-inversion equivalence. -/
theorem extendedInversion_pushRegion_chosenSide
    (c : P) (R : ℝ) (hR : R ≠ 0) (h : Set (OnePoint P)) :
    pushRegion (extendedInversionEquiv c R hR)
        (chosenSide OnePoint.infty h) =
      chosenSide (c : OnePoint P)
        (pushRegion (extendedInversionEquiv c R hR) h) := by
  simpa using
    (pushRegion_chosenSide (extendedInversionEquiv c R hR)
      (OnePoint.infty : OnePoint P) h)

/-- Actual finite payload mass is invariant when both the compactified points
and their region are transported through spherical inversion. -/
theorem massIn_extendedInversion
    (c : P) (R : ℝ) (hR : R ≠ 0) (items : Finset ι)
    (position : ι → OnePoint P) (value : ι → ℤ)
    (A : Set (OnePoint P)) :
    massIn items ((extendedInversionEquiv c R hR) ∘ position) value
        (pushRegion (extendedInversionEquiv c R hR) A) =
      massIn items position value A :=
  massIn_pushRegion (extendedInversionEquiv c R hR) items position value A

end SphericalPayloadTransport

section InversiveGram

variable {W E : Type*}

/-- Reorient a signed Gram matrix by assigning one scalar sign to every wall.
For `±1` signs this is the Layer-2 orientation action. -/
def reorientGram (sign : W → ℝ) (G : Matrix W W ℝ) : Matrix W W ℝ :=
  fun i j => sign i * G i j * sign j

/-- Reorientation is diagonal conjugation, `G_new = D * G_old * D`. -/
theorem reorientGram_eq_diagonal_mul [Fintype W] [DecidableEq W]
    (sign : W → ℝ) (G : Matrix W W ℝ) :
    reorientGram sign G =
      Matrix.diagonal sign * G * Matrix.diagonal sign := by
  ext i j
  rw [Matrix.mul_diagonal, Matrix.diagonal_mul]
  rfl

/-- Absolute Gram entries are invariant under genuine sign reorientation. -/
theorem abs_reorientGram_apply (sign : W → ℝ) (G : Matrix W W ℝ)
    (hsign : ∀ i, |sign i| = 1) (i j : W) :
    |reorientGram sign G i j| = |G i j| := by
  simp [reorientGram, abs_mul, hsign]

/-- The sign vector that flips exactly one wall. -/
def wallFlipSign [DecidableEq W] (w : W) : W → ℝ :=
  fun i => if i = w then -1 else 1

/-- A single-wall flip leaves every entry outside that wall's row and column
unchanged. -/
theorem reorientGram_singleFlip_off [DecidableEq W]
    (G : Matrix W W ℝ) (w i j : W) (hi : i ≠ w) (hj : j ≠ w) :
    reorientGram (wallFlipSign w) G i j = G i j := by
  simp [reorientGram, wallFlipSign, hi, hj]

theorem reorientGram_singleFlip_row [DecidableEq W]
    (G : Matrix W W ℝ) (w j : W) (hj : j ≠ w) :
    reorientGram (wallFlipSign w) G w j = -G w j := by
  simp [reorientGram, wallFlipSign, hj]

theorem reorientGram_singleFlip_column [DecidableEq W]
    (G : Matrix W W ℝ) (w i : W) (hi : i ≠ w) :
    reorientGram (wallFlipSign w) G i w = -G i w := by
  simp [reorientGram, wallFlipSign, hi]

/-! ### Rank-two row/column updates -/

/-- The row-supported part of a matrix at `w`. -/
def singleRowPart [DecidableEq W]
    (A : Matrix W W ℝ) (w : W) : Matrix W W ℝ :=
  Matrix.vecMulVec (Pi.single w 1) (A w)

/-- The column-supported remainder after the `(w,w)` entry has already been
included in `singleRowPart`. -/
def singleColumnRemainder [DecidableEq W]
    (A : Matrix W W ℝ) (w : W) : Matrix W W ℝ :=
  Matrix.vecMulVec (fun i => if i = w then 0 else A i w) (Pi.single w 1)

/-- Any matrix supported on one row and one column is the sum of two explicit
outer products. -/
theorem eq_singleRowPart_add_singleColumnRemainder_of_off_eq_zero
    [DecidableEq W] (A : Matrix W W ℝ) (w : W)
    (hoff : ∀ i j, i ≠ w → j ≠ w → A i j = 0) :
    A = singleRowPart A w + singleColumnRemainder A w := by
  ext i j
  by_cases hi : i = w
  · subst i
    simp [singleRowPart, singleColumnRemainder, Matrix.vecMulVec_apply]
  · by_cases hj : j = w
    · subst j
      simp [singleRowPart, singleColumnRemainder,
        Matrix.vecMulVec_apply, hi]
    · simp [singleRowPart, singleColumnRemainder,
        Matrix.vecMulVec_apply, hi, hj, hoff i j hi hj]

/-- Matrix rank is subadditive over the real field.  This derives the missing
matrix-facing form from mathlib's range and submodule finrank lemmas. -/
theorem matrix_rank_add_le [Fintype W]
    (A B : Matrix W W ℝ) : (A + B).rank ≤ A.rank + B.rank := by
  rw [Matrix.rank, Matrix.rank, Matrix.rank, Matrix.mulVecLin_add]
  calc
    Module.finrank ℝ (LinearMap.range (A.mulVecLin + B.mulVecLin)) ≤
        Module.finrank ℝ
          (LinearMap.range A.mulVecLin ⊔ LinearMap.range B.mulVecLin :
            Submodule ℝ (W → ℝ)) := by
      apply Submodule.finrank_mono
      rintro y ⟨x, rfl⟩
      apply Submodule.mem_sup.mpr
      exact ⟨A.mulVecLin x, ⟨x, rfl⟩,
        B.mulVecLin x, ⟨x, rfl⟩, rfl⟩
    _ ≤ Module.finrank ℝ (LinearMap.range A.mulVecLin) +
        Module.finrank ℝ (LinearMap.range B.mulVecLin) :=
      Submodule.finrank_add_le_finrank_add_finrank _ _

/-- A finite matrix supported on one row and one column has rank at most two. -/
theorem matrix_rank_le_two_of_off_eq_zero [Fintype W] [DecidableEq W]
    (A : Matrix W W ℝ) (w : W)
    (hoff : ∀ i j, i ≠ w → j ≠ w → A i j = 0) :
    A.rank ≤ 2 := by
  rw [eq_singleRowPart_add_singleColumnRemainder_of_off_eq_zero A w hoff]
  calc
    (singleRowPart A w + singleColumnRemainder A w).rank ≤
        (singleRowPart A w).rank + (singleColumnRemainder A w).rank :=
      matrix_rank_add_le _ _
    _ ≤ 1 + 1 := Nat.add_le_add
      (Matrix.rank_vecMulVec_le _ _) (Matrix.rank_vecMulVec_le _ _)
    _ = 2 := rfl

/-- If two finite kernels agree away from a distinguished row and column,
their update has rank at most two.  This is the generic T8 statement used by
both zeta-like relation matrices and signed Gram updates. -/
theorem matrix_rank_sub_le_two_of_eq_off [Fintype W] [DecidableEq W]
    (A B : Matrix W W ℝ) (w : W)
    (hoff : ∀ i j, i ≠ w → j ≠ w → A i j = B i j) :
    (A - B).rank ≤ 2 := by
  apply matrix_rank_le_two_of_off_eq_zero _ w
  intro i j hi hj
  rw [Matrix.sub_apply, hoff i j hi hj, sub_self]

/-- The signed-Gram update caused by flipping one wall has rank at most two. -/
theorem rank_singleWallGramFlip_sub_le_two [Fintype W] [DecidableEq W]
    (G : Matrix W W ℝ) (w : W) :
    (reorientGram (wallFlipSign w) G - G).rank ≤ 2 := by
  apply matrix_rank_sub_le_two_of_eq_off _ _ w
  intro i j hi hj
  exact reorientGram_singleFlip_off G w i j hi hj

variable [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Pairwise signed inversive distances as a finite or infinite Gram kernel. -/
noncomputable def inversiveGram (center : W → E) (radius : W → ℝ) :
    Matrix W W ℝ :=
  fun i j =>
    signedInversiveDistance (center i) (radius i) (center j) (radius j)

/-- Inverting an indexed sphere family reorients its signed inversive Gram
matrix by the denominator-sign vector. The absolute contact/crossing skeleton
is therefore unchanged. -/
theorem inversiveGram_inversion
    (R : ℝ) (hR : R ≠ 0) (center : W → E) (radius : W → ℝ)
    (hδ : ∀ i, inversionDenominator (center i) (radius i) ≠ 0) :
    inversiveGram
        (fun i => invertedSphereCenter R (center i) (radius i))
        (fun i => invertedSphereRadius R (radius i) (center i)) =
      reorientGram
        (fun i => inversionOrientationSign (center i) (radius i))
        (inversiveGram center radius) := by
  ext i j
  exact signedInversiveDistance_inversion_eq_orientationSigns
    R (radius i) (radius j) (center i) (center j) hR (hδ i) (hδ j)

/-- Entrywise absolute Gram data is invariant under inversion. -/
theorem abs_inversiveGram_inversion_apply
    (R : ℝ) (hR : R ≠ 0) (center : W → E) (radius : W → ℝ)
    (hδ : ∀ i, inversionDenominator (center i) (radius i) ≠ 0)
    (i j : W) :
    |inversiveGram
        (fun k => invertedSphereCenter R (center k) (radius k))
        (fun k => invertedSphereRadius R (radius k) (center k)) i j| =
      |inversiveGram center radius i j| := by
  rw [inversiveGram_inversion R hR center radius hδ]
  exact abs_reorientGram_apply _ _
    (fun k => abs_inversionOrientationSign _ _ (hδ k)) i j

end InversiveGram

section AbstractMobiusInversion

variable {P Q R : Type*}

/-- Relabel a function along an equivalence. -/
def transport (e : P ≃ Q) (f : P → R) : Q → R :=
  fun q => f (e.symm q)

@[simp]
theorem transport_symm_transport (e : P ≃ Q) (f : P → R) :
    transport e.symm (transport e f) = f := by
  funext p
  simp [transport]

@[simp]
theorem transport_transport_symm (e : P ≃ Q) (f : Q → R) :
    transport e (transport e.symm f) = f := by
  funext q
  simp [transport]

/-- Relabelling along a composite equivalence is successive relabelling. -/
theorem transport_trans {S : Type*} (e : P ≃ Q) (d : Q ≃ S) (f : P → R) :
    transport (e.trans d) f = transport d (transport e f) := by
  funext s
  simp [transport]

/-- Conjugate an operator on poset-labelled data through a relabelling. -/
def conjugate (e : P ≃ Q) (T : (P → R) → (P → R)) :
    (Q → R) → (Q → R) :=
  fun f => transport e (T (transport e.symm f))

/-- Inverses remain inverses after relabelling. This is the generic mechanism
behind equivariance of incidence Möbius inversion under a poset isomorphism. -/
theorem conjugate_leftInverse
    {zeta mobius : (P → R) → (P → R)}
    (h : Function.LeftInverse mobius zeta)
    (e : P ≃ Q) :
    Function.LeftInverse (conjugate e mobius) (conjugate e zeta) := by
  intro f
  funext q
  simp only [conjugate, transport_symm_transport]
  change mobius (zeta (transport e.symm f)) (e.symm q) = f q
  calc
    _ = transport e.symm f (e.symm q) :=
      congrFun (h (transport e.symm f)) (e.symm q)
    _ = f q := by simp [transport]

theorem conjugate_rightInverse
    {zeta mobius : (P → R) → (P → R)}
    (h : Function.RightInverse mobius zeta)
    (e : P ≃ Q) :
    Function.RightInverse (conjugate e mobius) (conjugate e zeta) := by
  intro f
  funext q
  simp only [conjugate, transport_symm_transport]
  change zeta (mobius (transport e.symm f)) (e.symm q) = f q
  calc
    _ = transport e.symm f (e.symm q) :=
      congrFun (h (transport e.symm f)) (e.symm q)
    _ = f q := by simp [transport]

end AbstractMobiusInversion

section ChangeOfContainmentBasis

variable {A B : Type*}

/-- Change cumulative coordinates from an old containment order to a new one:
first recover the direct labels with the old Möbius transform, then accumulate
them with the new zeta transform.  In matrix notation this is `Z_new * M_old`.
-/
def containmentPivot
    (oldMobius : A → A) (newZeta : A → B) : A → B :=
  fun oldTotals => newZeta (oldMobius oldTotals)

theorem containmentPivot_apply
    (oldMobius : A → A) (newZeta : A → B) (oldTotals : A) :
    containmentPivot oldMobius newZeta oldTotals =
      newZeta (oldMobius oldTotals) :=
  rfl

/-- The reverse change of containment coordinates is `Z_old * M_new`.
The two cancel whenever each Möbius transform inverts its own zeta transform.
-/
theorem containmentPivot_roundtrip
    (oldZeta oldMobius : A → A) (newZeta newMobius : A → A)
    (hold : Function.RightInverse oldMobius oldZeta)
    (hnew : Function.LeftInverse newMobius newZeta) :
    Function.LeftInverse
      (containmentPivot newMobius oldZeta)
      (containmentPivot oldMobius newZeta) := by
  intro oldTotals
  simp only [containmentPivot]
  rw [hnew, hold]

/-- Successive pivots telescope.  No mysterious extra operation appears:
`(Z_C * M_B) * (Z_B * M_A) = Z_C * M_A`. -/
theorem containmentPivot_trans
    (mobiusA : A → A) (zetaB mobiusB : A → A) (zetaC : A → B)
    (hB : Function.LeftInverse mobiusB zetaB)
    (totals : A) :
    containmentPivot mobiusB zetaC
        (containmentPivot mobiusA zetaB totals) =
      containmentPivot mobiusA zetaC totals := by
  change zetaC (mobiusB (zetaB (mobiusA totals))) =
    zetaC (mobiusA totals)
  rw [hB (mobiusA totals)]

/-- Honest coordinate transport includes a semantic reassignment `R` between
the old and new direct chambers. In matrix notation this is
`Z_new * R * M_old`. -/
def semanticContainmentPivot
    (oldMobius : A → A) (reassign : A → B) (newZeta : B → B) : A → B :=
  fun oldTotals => newZeta (reassign (oldMobius oldTotals))

theorem semanticContainmentPivot_apply
    (oldMobius : A → A) (reassign : A → B) (newZeta : B → B)
    (oldTotals : A) :
    semanticContainmentPivot oldMobius reassign newZeta oldTotals =
      newZeta (reassign (oldMobius oldTotals)) :=
  rfl

/-- Semantic pivots along a path compose by composing their chamber
reassignments. The intermediate zeta/Möbius pair cancels, but `R` does not. -/
theorem semanticContainmentPivot_trans {C : Type*}
    (mobiusA : A → A) (reassignAB : A → B)
    (zetaB mobiusB : B → B) (reassignBC : B → C)
    (zetaC : C → C) (hB : Function.LeftInverse mobiusB zetaB)
    (totals : A) :
    semanticContainmentPivot mobiusB reassignBC zetaC
        (semanticContainmentPivot mobiusA reassignAB zetaB totals) =
      semanticContainmentPivot mobiusA
        (fun direct => reassignBC (reassignAB direct)) zetaC totals := by
  change zetaC
      (reassignBC (mobiusB (zetaB (reassignAB (mobiusA totals))))) =
    zetaC (reassignBC (reassignAB (mobiusA totals)))
  rw [hB]

end ChangeOfContainmentBasis

section FiniteZetaTransform

variable {P Q : Type*}
  [Fintype P] [PartialOrder P] [DecidableLE P]
  [Fintype Q] [PartialOrder Q] [DecidableLE Q]

/-- The zeta transform accumulates direct labels over the containment order. -/
def zetaTransform (f : P → ℤ) (y : P) : ℤ :=
  ∑ x : P, if x ≤ y then f x else 0

/-- An order isomorphism only relabels the zeta transform. A geometric Möbius
map reaches this theorem through `pushRegion_subset_iff`. -/
theorem zetaTransform_orderIso (e : P ≃o Q) (f : P → ℤ) (y : P) :
    zetaTransform (P := Q) (transport e.toEquiv f) (e y) =
      zetaTransform (P := P) f y := by
  classical
  unfold zetaTransform
  calc
    (∑ q : Q, if q ≤ e y then transport e.toEquiv f q else 0) =
        ∑ p : P, if e p ≤ e y then transport e.toEquiv f (e p) else 0 :=
      (e.toEquiv.sum_comp _).symm
    _ = ∑ p : P, if p ≤ y then f p else 0 := by
      simp [transport]

end FiniteZetaTransform

section FiniteMobiusTransform

variable {P : Type*} [Fintype P] [PartialOrder P] [DecidableEq P]
  [DecidableLE P] [LocallyFiniteOrder P]

/-- Möbius recovery over all lower labels of a finite poset.  This definition
does not require the poset itself to have a global bottom element. -/
noncomputable def mobiusTransform (F : P → ℤ) (y : P) : ℤ :=
  ∑ x : P, IncidenceAlgebra.mu ℤ x y * F x

omit [DecidableEq P] [LocallyFiniteOrder P] in
theorem zetaTransform_eq_incidence_sum (f : P → ℤ) (y : P) :
    zetaTransform f y =
      ∑ x : P, IncidenceAlgebra.zeta ℤ x y * f x := by
  classical
  unfold zetaTransform
  apply Finset.sum_congr rfl
  intro x hx
  simp [IncidenceAlgebra.zeta_apply]

/-- Extending the interval convolution by zero from `Icc z y` to the whole
finite carrier does not change it. -/
private theorem sum_zeta_mul_mu_univ (z y : P) :
    (∑ x : P,
        IncidenceAlgebra.zeta ℤ z x * IncidenceAlgebra.mu ℤ x y) =
      (IncidenceAlgebra.zeta ℤ * IncidenceAlgebra.mu ℤ :
        IncidenceAlgebra ℤ P) z y := by
  rw [IncidenceAlgebra.mul_apply]
  symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro x hx hxIcc
  simp only [Finset.mem_univ] at hx
  rw [Finset.mem_Icc, not_and_or] at hxIcc
  rcases hxIcc with hzx | hxy
  · rw [IncidenceAlgebra.zeta_apply, if_neg hzx, zero_mul]
  · rw [IncidenceAlgebra.apply_eq_zero_of_not_le hxy, mul_zero]

/-- Finite-poset Möbius inversion with no global-bottom hypothesis.

Mathlib's `moebius_inversion_bot` assumes `OrderBot`.  For a finite chamber
poset carrying only a background top this more general statement follows
directly from `zeta * mu = 1`. -/
theorem mobiusTransform_zetaTransform (f : P → ℤ) :
    mobiusTransform (zetaTransform f) = f := by
  classical
  funext y
  simp_rw [mobiusTransform, zetaTransform_eq_incidence_sum, Finset.mul_sum]
  rw [Finset.sum_comm]
  calc
    (∑ z : P, ∑ x : P,
        IncidenceAlgebra.mu ℤ x y *
          (IncidenceAlgebra.zeta ℤ z x * f z)) =
        ∑ z : P, (∑ x : P,
          IncidenceAlgebra.zeta ℤ z x * IncidenceAlgebra.mu ℤ x y) * f z := by
      apply Finset.sum_congr rfl
      intro z hz
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro x hx
      ring
    _ = ∑ z : P, (1 : IncidenceAlgebra ℤ P) z y * f z := by
      apply Finset.sum_congr rfl
      intro z hz
      rw [sum_zeta_mul_mu_univ, IncidenceAlgebra.zeta_mul_mu]
    _ = f y := by simp

end FiniteMobiusTransform

section FiniteChamberModel

variable {X ι W : Type*} [Fintype W] [PartialOrder W] [DecidableEq W]
  [DecidableLE W] [LocallyFiniteOrder W]

/-- A finite chamber presentation of a space.  Each point has one owning
chamber; the region at `w` is reconstructed as the union of chambers at or
below `w`.  Taking `W := WithTop V` makes the top fibre the background
chamber. -/
structure ChamberModel (W X : Type*) [LE W] where
  owner : X → W

namespace ChamberModel

/-- The points directly owned by chamber `w`. -/
def chamber (model : ChamberModel W X) (w : W) : Set X :=
  {x | model.owner x = w}

/-- The cumulative region represented by `w`. -/
def region (model : ChamberModel W X) (w : W) : Set X :=
  {x | model.owner x ≤ w}

omit [Fintype W] [DecidableEq W] [DecidableLE W]
  [LocallyFiniteOrder W] in
theorem chamber_pairwise_disjoint (model : ChamberModel W X) :
    Pairwise fun v w : W => Disjoint (model.chamber v) (model.chamber w) := by
  intro v w hvw
  rw [Set.disjoint_left]
  intro x hxv hxw
  exact hvw (hxv.symm.trans hxw)

omit [Fintype W] [DecidableEq W] [DecidableLE W]
  [LocallyFiniteOrder W] in
/-- Every point belongs to exactly one direct chamber, so the chambers cover
the ambient space. -/
theorem iUnion_chamber_eq_univ (model : ChamberModel W X) :
    ⋃ w, model.chamber w = Set.univ := by
  ext x
  simp [chamber]

omit [Fintype W] [DecidableEq W] [DecidableLE W]
  [LocallyFiniteOrder W] in
/-- A cumulative region is the union of the direct chambers indexed below
it. -/
theorem region_eq_iUnion_chamber (model : ChamberModel W X) (w : W) :
    model.region w = ⋃ v : {v : W // v ≤ w}, model.chamber v.1 := by
  ext x
  constructor
  · intro hx
    exact Set.mem_iUnion_of_mem ⟨model.owner x, hx⟩ rfl
  · simp only [Set.mem_iUnion]
    rintro ⟨v, hv⟩
    change model.owner x = v.1 at hv
    change model.owner x ≤ w
    rw [hv]
    exact v.2

omit [LocallyFiniteOrder W] in
/-- Cumulative region mass is exactly the incidence zeta transform of direct
chamber mass. -/
theorem massIn_region_eq_zetaTransform (model : ChamberModel W X)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) (w : W) :
    massIn items position value (model.region w) =
      zetaTransform (fun v => massIn items position value (model.chamber v)) w := by
  classical
  simp only [massIn, zetaTransform]
  symm
  calc
    (∑ v : W, if v ≤ w then
        ∑ i ∈ items, if position i ∈ model.chamber v then value i else 0
      else 0) =
        ∑ v : W, ∑ i ∈ items,
          if v ≤ w ∧ position i ∈ model.chamber v then value i else 0 := by
      apply Finset.sum_congr rfl
      intro v hv
      by_cases hvw : v ≤ w <;> simp [hvw]
    _ = ∑ i ∈ items, ∑ v : W,
          if v ≤ w ∧ position i ∈ model.chamber v then value i else 0 := by
      rw [Finset.sum_comm]
    _ = ∑ i ∈ items,
          if position i ∈ model.region w then value i else 0 := by
      apply Finset.sum_congr rfl
      intro i hi
      by_cases how : model.owner (position i) ≤ w
      · have hregion : position i ∈ model.region w := how
        rw [if_pos hregion]
        calc
          (∑ v : W, if v ≤ w ∧ position i ∈ model.chamber v then
              value i else 0) =
              ∑ v : W, if v = model.owner (position i) then value i else 0 := by
            apply Finset.sum_congr rfl
            intro v hv
            by_cases heq : v = model.owner (position i)
            · subst v
              have hmem : position i ∈
                  model.chamber (model.owner (position i)) := rfl
              simp [how, hmem]
            · have hnmem : position i ∉ model.chamber v := by
                intro hmem
                exact heq hmem.symm
              simp [heq, hnmem]
          _ = value i := by simp
      · have hnregion : position i ∉ model.region w := how
        rw [if_neg hnregion]
        apply Finset.sum_eq_zero
        intro v hv
        by_cases heq : v = model.owner (position i)
        · subst v
          have hmem : position i ∈
              model.chamber (model.owner (position i)) := rfl
          simp [how, hmem]
        · have hnmem : position i ∉ model.chamber v := by
            intro hmem
            exact heq hmem.symm
          simp [hnmem]

/-- Incidence Möbius inversion recovers the direct geometric chamber masses
from cumulative region masses.  This is the finite geometry–incidence bridge;
it works without adding an artificial bottom to a background-extended poset. -/
theorem massIn_chamber_eq_mobiusTransform (model : ChamberModel W X)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) :
    (fun w => massIn items position value (model.chamber w)) =
      mobiusTransform (fun w => massIn items position value (model.region w)) := by
  let direct : W → ℤ :=
    fun w => massIn items position value (model.chamber w)
  have hcumulative :
      (fun w => massIn items position value (model.region w)) =
        zetaTransform direct := by
    funext w
    exact model.massIn_region_eq_zetaTransform items position value w
  rw [hcumulative, mobiusTransform_zetaTransform]

end ChamberModel

end FiniteChamberModel

section SemanticChamberReassignment

variable {X P Q S : Type*} [PartialOrder P] [PartialOrder Q] [PartialOrder S]

/-- A semantic reassignment between two chamber models is an equivalence of
indices whose matched direct chambers are the same subsets of space. It need
not preserve order: changing containment is precisely the point. -/
structure ChamberReassignment
    (oldModel : ChamberModel P X) (newModel : ChamberModel Q X) where
  toEquiv : P ≃ Q
  chamber_eq : ∀ p, oldModel.chamber p = newModel.chamber (toEquiv p)

namespace ChamberReassignment

def refl (model : ChamberModel P X) : ChamberReassignment model model where
  toEquiv := Equiv.refl P
  chamber_eq := fun _ => rfl

/-- Semantic reassignment along consecutive re-anchors composes by matching
the same physical chambers through the intermediate model. -/
def trans {middleModel : ChamberModel Q X} {newModel : ChamberModel S X}
    {oldModel : ChamberModel P X}
    (movePQ : ChamberReassignment oldModel middleModel)
    (moveQS : ChamberReassignment middleModel newModel) :
    ChamberReassignment oldModel newModel where
  toEquiv := movePQ.toEquiv.trans moveQS.toEquiv
  chamber_eq := fun p =>
    (movePQ.chamber_eq p).trans (moveQS.chamber_eq (movePQ.toEquiv p))

@[simp] theorem refl_toEquiv (model : ChamberModel P X) :
    (refl model).toEquiv = Equiv.refl P :=
  rfl

@[simp] theorem trans_toEquiv {middleModel : ChamberModel Q X}
    {newModel : ChamberModel S X} {oldModel : ChamberModel P X}
    (movePQ : ChamberReassignment oldModel middleModel)
    (moveQS : ChamberReassignment middleModel newModel) :
    (movePQ.trans moveQS).toEquiv =
      movePQ.toEquiv.trans moveQS.toEquiv :=
  rfl

variable {ι : Type*}

/-- Matching physical chambers makes their payload masses transform by the
semantic relabelling `R`. -/
theorem directMass_eq_transport {oldModel : ChamberModel P X}
    {newModel : ChamberModel Q X}
    (move : ChamberReassignment oldModel newModel)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) :
    (fun q => massIn items position value (newModel.chamber q)) =
      transport move.toEquiv
        (fun p => massIn items position value (oldModel.chamber p)) := by
  funext q
  calc
    massIn items position value (newModel.chamber q) =
        massIn items position value
          (newModel.chamber (move.toEquiv (move.toEquiv.symm q))) := by
      rw [move.toEquiv.apply_symm_apply]
    _ = massIn items position value
        (oldModel.chamber (move.toEquiv.symm q)) :=
      congrArg (massIn items position value)
        (move.chamber_eq (move.toEquiv.symm q)).symm

/-- The direct-mass reassignment for a composite path is the composite of
the direct-mass reassignments. -/
theorem directMass_trans {middleModel : ChamberModel Q X}
    {newModel : ChamberModel S X} {oldModel : ChamberModel P X}
    (movePQ : ChamberReassignment oldModel middleModel)
    (moveQS : ChamberReassignment middleModel newModel)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) :
    transport (movePQ.trans moveQS).toEquiv
        (fun p => massIn items position value (oldModel.chamber p)) =
      transport moveQS.toEquiv
        (transport movePQ.toEquiv
          (fun p => massIn items position value (oldModel.chamber p))) :=
  transport_trans movePQ.toEquiv moveQS.toEquiv _

end ChamberReassignment

namespace ChamberModel

/-- Reindex the direct chambers along any equivalence, while allowing the new
index type to carry a different containment order. -/
def reindex (model : ChamberModel P X) (e : P ≃ Q) : ChamberModel Q X where
  owner := e ∘ model.owner

theorem chamber_eq_reindex (model : ChamberModel P X) (e : P ≃ Q) (p : P) :
    model.chamber p = (model.reindex e).chamber (e p) := by
  ext x
  simp [chamber, reindex]

/-- Reindexing supplies a canonical semantic chamber reassignment. -/
def reindexReassignment (model : ChamberModel P X) (e : P ≃ Q) :
    ChamberReassignment model (model.reindex e) where
  toEquiv := e
  chamber_eq := model.chamber_eq_reindex e

end ChamberModel

section FiniteSemanticPivot

variable {ι : Type*}
  [Fintype P] [DecidableEq P] [DecidableLE P]
  [LocallyFiniteOrder P]
  [Fintype Q] [DecidableEq Q] [DecidableLE Q]

/-- The checked `Z_new * R * M_old` formula. Old cumulative region masses are
Möbius-inverted into direct chamber masses, semantically reassigned to the
same physical chambers, then accumulated in the new containment order. -/
theorem ChamberReassignment.regionMass_eq_semanticContainmentPivot
    {oldModel : ChamberModel P X} {newModel : ChamberModel Q X}
    (move : ChamberReassignment oldModel newModel)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) :
    (fun q => massIn items position value (newModel.region q)) =
      semanticContainmentPivot
        (mobiusTransform (P := P))
        (transport move.toEquiv)
        (zetaTransform (P := Q))
        (fun p => massIn items position value (oldModel.region p)) := by
  let oldDirect : P → ℤ :=
    fun p => massIn items position value (oldModel.chamber p)
  let oldTotals : P → ℤ :=
    fun p => massIn items position value (oldModel.region p)
  let newDirect : Q → ℤ :=
    fun q => massIn items position value (newModel.chamber q)
  let newTotals : Q → ℤ :=
    fun q => massIn items position value (newModel.region q)
  have hold : oldDirect = mobiusTransform oldTotals :=
    oldModel.massIn_chamber_eq_mobiusTransform items position value
  have hmove : newDirect = transport move.toEquiv oldDirect :=
    move.directMass_eq_transport items position value
  have hnew : newTotals = zetaTransform newDirect := by
    funext q
    exact newModel.massIn_region_eq_zetaTransform items position value q
  calc
    newTotals = zetaTransform newDirect := hnew
    _ = zetaTransform (transport move.toEquiv oldDirect) :=
      congrArg zetaTransform hmove
    _ = zetaTransform
        (transport move.toEquiv (mobiusTransform oldTotals)) :=
      congrArg (fun direct => zetaTransform (transport move.toEquiv direct))
        hold
    _ = semanticContainmentPivot
        (mobiusTransform (P := P))
        (transport move.toEquiv)
        (zetaTransform (P := Q)) oldTotals := rfl

end FiniteSemanticPivot

section FiniteSemanticPivotPath

variable {ι : Type*}
  [Fintype P] [DecidableEq P] [DecidableLE P]
  [LocallyFiniteOrder P]
  [Fintype Q] [DecidableEq Q] [DecidableLE Q]
  [LocallyFiniteOrder Q]
  [Fintype S] [DecidableEq S] [DecidableLE S]

omit [DecidableLE P] [DecidableEq S] in
/-- For two consecutive semantic re-anchors, the intermediate incidence
coordinates cancel. The surviving reassignment is the composite physical
chamber match. Repeating this theorem handles an arbitrary finite flip path.
-/
theorem ChamberReassignment.semanticContainmentPivot_path_two
    {oldModel : ChamberModel P X} {middleModel : ChamberModel Q X}
    {newModel : ChamberModel S X}
    (movePQ : ChamberReassignment oldModel middleModel)
    (moveQS : ChamberReassignment middleModel newModel)
    (oldTotals : P → ℤ) :
    semanticContainmentPivot
        (mobiusTransform (P := Q))
        (transport moveQS.toEquiv)
        (zetaTransform (P := S))
        (semanticContainmentPivot
          (mobiusTransform (P := P))
          (transport movePQ.toEquiv)
          (zetaTransform (P := Q)) oldTotals) =
      semanticContainmentPivot
        (mobiusTransform (P := P))
        (transport (movePQ.trans moveQS).toEquiv)
        (zetaTransform (P := S)) oldTotals := by
  rw [semanticContainmentPivot_trans
    (hB := mobiusTransform_zetaTransform (P := Q))]
  unfold semanticContainmentPivot
  rw [trans_toEquiv, transport_trans]

/-- The final geometric region masses after two re-anchors equal the
two-step semantic pivot from the original region masses. -/
theorem ChamberReassignment.regionMass_eq_semanticPivot_path_two
    {oldModel : ChamberModel P X} {middleModel : ChamberModel Q X}
    {newModel : ChamberModel S X}
    (movePQ : ChamberReassignment oldModel middleModel)
    (moveQS : ChamberReassignment middleModel newModel)
    (items : Finset ι) (position : ι → X) (value : ι → ℤ) :
    (fun s => massIn items position value (newModel.region s)) =
      semanticContainmentPivot
        (mobiusTransform (P := Q))
        (transport moveQS.toEquiv)
        (zetaTransform (P := S))
        (semanticContainmentPivot
          (mobiusTransform (P := P))
          (transport movePQ.toEquiv)
          (zetaTransform (P := Q))
          (fun p => massIn items position value (oldModel.region p))) := by
  calc
    (fun s => massIn items position value (newModel.region s)) =
        semanticContainmentPivot
          (mobiusTransform (P := P))
          (transport (movePQ.trans moveQS).toEquiv)
          (zetaTransform (P := S))
          (fun p => massIn items position value (oldModel.region p)) :=
      (movePQ.trans moveQS).regionMass_eq_semanticContainmentPivot
        items position value
    _ = semanticContainmentPivot
        (mobiusTransform (P := Q))
        (transport moveQS.toEquiv)
        (zetaTransform (P := S))
        (semanticContainmentPivot
          (mobiusTransform (P := P))
          (transport movePQ.toEquiv)
          (zetaTransform (P := Q))
          (fun p => massIn items position value (oldModel.region p))) :=
      (movePQ.semanticContainmentPivot_path_two moveQS _).symm

end FiniteSemanticPivotPath

section FiniteChainReversal

variable {X ι : Type*} {n : ℕ}

/-- Reverse every direct chamber of the canonical finite chain. If `Fin n`
includes the background as its last element, reversal necessarily moves that
background chamber too. -/
def ChamberModel.reverseFiniteChain (model : ChamberModel (Fin n) X) :
    ChamberModel (Fin n) X :=
  model.reindex Fin.revPerm

theorem ChamberModel.chamber_eq_reverseFiniteChain
    (model : ChamberModel (Fin n) X) (i : Fin n) :
    model.chamber i = model.reverseFiniteChain.chamber i.rev := by
  simpa [ChamberModel.reverseFiniteChain] using
    model.chamber_eq_reindex Fin.revPerm i

/-- The arbitrary-length finite-chain version of the honest four-chamber
regression: full reversal is `Z_rev * R_rev * M_old`, with `R_rev` acting on
all direct chambers. -/
theorem reverseFiniteChain_regionMass_eq_semanticContainmentPivot
    (model : ChamberModel (Fin n) X) (items : Finset ι)
    (position : ι → X) (value : ι → ℤ) :
    (fun i => massIn items position value
        (model.reverseFiniteChain.region i)) =
      semanticContainmentPivot
        (mobiusTransform (P := Fin n))
        (transport Fin.revPerm)
        (zetaTransform (P := Fin n))
        (fun i => massIn items position value (model.region i)) := by
  exact ChamberReassignment.regionMass_eq_semanticContainmentPivot
    (model.reindexReassignment Fin.revPerm) items position value

end FiniteChainReversal

end SemanticChamberReassignment

section LaminarChamberDerivation

variable {X W : Type*} [PartialOrder W]

/-- A finite-ready indexed laminar family. The index order is exactly region
inclusion, while a shared point forces comparability. The latter condition is
the pointwise form of laminarity and permits disjoint branches. -/
structure FiniteLaminarFamily (W X : Type*) [PartialOrder W] where
  region : W → Set X
  order_iff_subset : ∀ {v w}, v ≤ w ↔ region v ⊆ region w
  comparable_of_mem :
    ∀ {x v w}, x ∈ region v → x ∈ region w → v ≤ w ∨ w ≤ v

namespace FiniteLaminarFamily

/-- Adjoin the entire ambient space as a background region. -/
def extendedRegion (family : FiniteLaminarFamily W X) : WithTop W → Set X :=
  WithTop.recTopCoe Set.univ family.region

@[simp] theorem extendedRegion_top (family : FiniteLaminarFamily W X) :
    family.extendedRegion ⊤ = Set.univ :=
  WithTop.recTopCoe_top _ _

@[simp] theorem extendedRegion_coe (family : FiniteLaminarFamily W X) (w : W) :
    family.extendedRegion (w : WithTop W) = family.region w :=
  WithTop.recTopCoe_coe _ _ _

theorem extendedRegion_mono (family : FiniteLaminarFamily W X) :
    Monotone family.extendedRegion := by
  intro v w hvw
  induction v using WithTop.recTopCoe with
  | top =>
      have hw : w = ⊤ := top_unique hvw
      subst w
      exact Set.Subset.rfl
  | coe v =>
      induction w using WithTop.recTopCoe with
      | top => exact Set.subset_univ _
      | coe w =>
          apply family.order_iff_subset.mp
          simpa using hvw

theorem comparable_of_mem_extended (family : FiniteLaminarFamily W X)
    {x : X} {v w : WithTop W} (hv : x ∈ family.extendedRegion v)
    (hw : x ∈ family.extendedRegion w) : v ≤ w ∨ w ≤ v := by
  induction v using WithTop.recTopCoe with
  | top => exact Or.inr le_top
  | coe v =>
      induction w using WithTop.recTopCoe with
      | top => exact Or.inl le_top
      | coe w => simpa using family.comparable_of_mem hv hw

section Finite

variable [Fintype W]

/-- All extended regions containing a point. The top element ensures this
finset is never empty. -/
noncomputable def containing (family : FiniteLaminarFamily W X) (x : X) :
    Finset (WithTop W) := by
  classical
  exact Finset.univ.filter fun w => x ∈ family.extendedRegion w

@[simp] theorem mem_containing (family : FiniteLaminarFamily W X) (x : X)
    (w : WithTop W) :
    w ∈ family.containing x ↔ x ∈ family.extendedRegion w := by
  classical
  simp [containing]

/-- The containing indices form a finite chain and therefore have a unique
least element. This is the direct existence theorem behind chamber
ownership. -/
theorem exists_unique_least_containing (family : FiniteLaminarFamily W X)
    (x : X) :
    ∃! m : WithTop W,
      x ∈ family.extendedRegion m ∧
        ∀ w, x ∈ family.extendedRegion w → m ≤ w := by
  classical
  let s := family.containing x
  have htop : (⊤ : WithTop W) ∈ s := by
    simp [s]
  obtain ⟨m, hm⟩ := s.exists_minimal ⟨⊤, htop⟩
  have hmmem : x ∈ family.extendedRegion m := by
    simpa [s] using hm.1
  have hmleast : ∀ w, x ∈ family.extendedRegion w → m ≤ w := by
    intro w hw
    have hws : w ∈ s := by simpa [s] using hw
    rcases family.comparable_of_mem_extended hmmem hw with hmw | hwm
    · exact hmw
    · exact (hm.eq_of_ge hws hwm).le
  refine ⟨m, ⟨hmmem, hmleast⟩, ?_⟩
  intro n hn
  exact le_antisymm (hn.2 m hmmem) (hmleast n hn.1)

/-- The unique least indexed region containing a point. -/
noncomputable def owner (family : FiniteLaminarFamily W X) (x : X) :
    WithTop W :=
  Classical.choose (family.exists_unique_least_containing x)

theorem owner_mem (family : FiniteLaminarFamily W X) (x : X) :
    x ∈ family.extendedRegion (family.owner x) := by
  simpa [owner] using
    (Classical.choose_spec (family.exists_unique_least_containing x)).1.1

theorem owner_le_of_mem (family : FiniteLaminarFamily W X) (x : X)
    {w : WithTop W} (hw : x ∈ family.extendedRegion w) :
    family.owner x ≤ w := by
  simpa [owner] using
    (Classical.choose_spec
      (family.exists_unique_least_containing x)).1.2 w hw

/-- Geometric laminar data canonically supplies the owner map required by the
abstract chamber model. -/
noncomputable def chamberModel (family : FiniteLaminarFamily W X) :
    ChamberModel (WithTop W) X where
  owner := family.owner

theorem chamberModel_region_eq (family : FiniteLaminarFamily W X)
    (w : WithTop W) :
    family.chamberModel.region w = family.extendedRegion w := by
  ext x
  constructor
  · intro hx
    exact family.extendedRegion_mono hx (family.owner_mem x)
  · exact family.owner_le_of_mem x

/-- The geometric chamber at `w` is the part of its region left after all
strictly smaller regions are removed. At top this is precisely the ambient
background. -/
def geometricChamber (family : FiniteLaminarFamily W X)
    (w : WithTop W) : Set X :=
  family.extendedRegion w \
    ⋃ v : {v : WithTop W // v < w}, family.extendedRegion v.1

theorem chamberModel_chamber_eq (family : FiniteLaminarFamily W X)
    (w : WithTop W) :
    family.chamberModel.chamber w = family.geometricChamber w := by
  ext x
  constructor
  · intro hx
    change family.owner x = w at hx
    subst w
    refine ⟨family.owner_mem x, ?_⟩
    simp only [Set.mem_iUnion, not_exists]
    intro v hxv
    exact (not_le_of_gt v.property) (family.owner_le_of_mem x hxv)
  · rintro ⟨hxw, hxsmall⟩
    change family.owner x = w
    have how : family.owner x ≤ w := family.owner_le_of_mem x hxw
    rcases lt_or_eq_of_le how with howlt | howeq
    · exfalso
      apply hxsmall
      exact Set.mem_iUnion_of_mem ⟨family.owner x, howlt⟩
        (family.owner_mem x)
    · exact howeq

/-- A point is owned by the background exactly when it lies in none of the
original regions. -/
theorem owner_eq_top_iff (family : FiniteLaminarFamily W X) (x : X) :
    family.owner x = ⊤ ↔ ∀ w : W, x ∉ family.region w := by
  constructor
  · intro htop w hxw
    have hle := family.owner_le_of_mem x (w := (w : WithTop W)) hxw
    rw [htop] at hle
    simp at hle
  · intro houtside
    induction howner : family.owner x using WithTop.recTopCoe with
    | top => exact rfl
    | coe w =>
        exfalso
        apply houtside w
        simpa [howner] using family.owner_mem x

theorem geometricChamber_pairwise_disjoint
    (family : FiniteLaminarFamily W X) :
    Pairwise fun v w : WithTop W ↦
      Disjoint (family.geometricChamber v) (family.geometricChamber w) := by
  simpa only [← family.chamberModel_chamber_eq] using
    family.chamberModel.chamber_pairwise_disjoint

theorem iUnion_geometricChamber_eq_univ
    (family : FiniteLaminarFamily W X) :
    ⋃ w, family.geometricChamber w = Set.univ := by
  simpa only [← family.chamberModel_chamber_eq] using
    family.chamberModel.iUnion_chamber_eq_univ

/-- Every extended region is exactly the union of its lower geometric
chambers. -/
theorem extendedRegion_eq_iUnion_geometricChamber
    (family : FiniteLaminarFamily W X) (w : WithTop W) :
    family.extendedRegion w =
      ⋃ v : {v : WithTop W // v ≤ w}, family.geometricChamber v.1 := by
  rw [← family.chamberModel_region_eq]
  simpa only [family.chamberModel_chamber_eq] using
    family.chamberModel.region_eq_iUnion_chamber w

variable {ι : Type*}

section Incidence

variable [DecidableEq W] [DecidableLE W]

noncomputable local instance withTopDecidableLT : DecidableLT (WithTop W) :=
  Classical.decRel _

noncomputable local instance withTopLocallyFiniteOrder :
    LocallyFiniteOrder (WithTop W) :=
  Fintype.toLocallyFiniteOrder

/-- Payload mass in an extended laminar region is the zeta transform of the
direct geometric-chamber masses. -/
theorem massIn_extendedRegion_eq_zetaTransform_geometricChamber
    (family : FiniteLaminarFamily W X) (items : Finset ι)
    (position : ι → X) (value : ι → ℤ) (w : WithTop W) :
    massIn items position value (family.extendedRegion w) =
      zetaTransform
        (fun v => massIn items position value (family.geometricChamber v)) w := by
  rw [← family.chamberModel_region_eq]
  simpa only [family.chamberModel_chamber_eq] using
    family.chamberModel.massIn_region_eq_zetaTransform
      items position value w

/-- Möbius inversion recovers each direct geometric-chamber mass from the
cumulative masses of the extended laminar regions, including background and
without an artificial bottom. -/
theorem massIn_geometricChamber_eq_mobiusTransform
    (family : FiniteLaminarFamily W X) (items : Finset ι)
    (position : ι → X) (value : ι → ℤ) :
    (fun w => massIn items position value (family.geometricChamber w)) =
      mobiusTransform
        (fun w => massIn items position value (family.extendedRegion w)) := by
  simpa only [← family.chamberModel_chamber_eq,
    ← family.chamberModel_region_eq] using
      family.chamberModel.massIn_chamber_eq_mobiusTransform
        items position value

end Incidence

end Finite

end FiniteLaminarFamily

end LaminarChamberDerivation

section IncidenceAlgebra

variable {P : Type*} [PartialOrder P] [LocallyFiniteOrder P] [DecidableEq P]
  [DecidableLE P]

/-- Mathlib's actual incidence-algebra statement: the poset Möbius function
is a two-sided inverse of the zeta function. -/
example :
    IncidenceAlgebra.mu ℤ * IncidenceAlgebra.zeta ℤ =
      (1 : IncidenceAlgebra ℤ P) :=
  IncidenceAlgebra.mu_mul_zeta ℤ P

example :
    IncidenceAlgebra.zeta ℤ * IncidenceAlgebra.mu ℤ =
      (1 : IncidenceAlgebra ℤ P) :=
  IncidenceAlgebra.zeta_mul_mu

end IncidenceAlgebra

section PivotedKernel

variable {P : Type*} [PartialOrder P] [LocallyFiniteOrder P] [DecidableEq P]

/-- A full pivot reverses every comparison. On the incidence kernel this is
exactly transposition: the Möbius coefficient from `a` to `b` after the pivot
is the old coefficient from `b` to `a`. -/
theorem fullPivot_transposes_mu (a b : P) :
    (IncidenceAlgebra.mu ℤ :
      IncidenceAlgebra ℤ (OrderDual P)) (OrderDual.toDual a) (OrderDual.toDual b) =
        (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) b a :=
  IncidenceAlgebra.mu_toDual ℤ a b

end PivotedKernel

section ChainIntervalMobiusKernel

variable {P : Type*} [PartialOrder P] [LocallyFiniteOrder P]
  [DecidableEq P]

/-- Every closed interval is totally ordered. Forest-shaped containment
posets have this property even though unrelated branches are incomparable. -/
def HasChainIntervals (P : Type*) [PartialOrder P] : Prop :=
  ∀ a b : P, IsChain (· ≤ ·) (Set.Icc a b)

/-- The Möbius coefficient across a cover is `-1`. -/
theorem mu_of_covBy {a b : P} (h : a ⋖ b) :
    (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b = -1 := by
  rw [IncidenceAlgebra.mu_eq_neg_sum_Ioc_of_ne h.ne]
  have hIoc : Finset.Ioc a b = {b} := by
    ext x
    simpa using Set.ext_iff.mp h.Ioc_eq x
  rw [hIoc]
  simp

/-- In a poset with chain intervals, every strict non-cover Möbius
coefficient vanishes. The proof inducts on interval size: the unique maximal
point below the endpoint contributes `-1`, the endpoint contributes `1`, and
all earlier terms vanish recursively. -/
theorem mu_eq_zero_of_lt_not_covBy
    (hchain : HasChainIntervals P) {a b : P}
    (hab : a < b) (hncov : ¬ a ⋖ b) :
    (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b = 0 := by
  generalize hn : (Finset.Ioc a b).card = n
  induction n using Nat.strong_induction_on generalizing a b with
  | h n ih =>
      let below := Finset.Ico a b
      have hbelow : below.Nonempty := by
        exact ⟨a, Finset.mem_Ico.2 ⟨le_rfl, hab⟩⟩
      obtain ⟨p, hp⟩ := below.exists_maximal hbelow
      have hpIco : p ∈ Finset.Ico a b := hp.1
      have hap : a ≤ p := (Finset.mem_Ico.mp hpIco).1
      have hpb : p < b := (Finset.mem_Ico.mp hpIco).2
      have hpGreatest : ∀ x ∈ Finset.Ico a b, x ≤ p := by
        intro x hx
        have hxIcc : x ∈ Set.Icc a b :=
          ⟨(Finset.mem_Ico.mp hx).1, (Finset.mem_Ico.mp hx).2.le⟩
        have hpIcc : p ∈ Set.Icc a b := ⟨hap, hpb.le⟩
        rcases (hchain a b).total hxIcc hpIcc with hxp | hpx
        · exact hxp
        · exact hp.2 hx hpx
      have hpcov : p ⋖ b := by
        refine covBy_iff_lt_and_eq_or_eq.2 ⟨hpb, ?_⟩
        intro c hpc hcb
        by_cases hcbne : c = b
        · exact Or.inr hcbne
        · left
          apply le_antisymm
          · apply hpGreatest c
            exact Finset.mem_Ico.2
              ⟨hap.trans hpc, lt_of_le_of_ne hcb hcbne⟩
          · exact hpc
      rw [IncidenceAlgebra.mu_eq_neg_sum_Ioc_of_ne hab.ne]
      have hpIoc : p ∈ Finset.Ioc a b :=
        Finset.mem_Ioc.2
          ⟨lt_of_le_of_ne hap (fun hpa => hncov (hpa ▸ hpcov)), hpb.le⟩
      have hbIoc : b ∈ Finset.Ioc a b :=
        Finset.mem_Ioc.2 ⟨hab, le_rfl⟩
      have hterms : ∀ x ∈ Finset.Ioc a b,
          (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) x b =
            if x = p then -1 else if x = b then 1 else 0 := by
        intro x hx
        by_cases hxp : x = p
        · subst x
          simp [mu_of_covBy hpcov]
        · by_cases hxb : x = b
          · subst x
            simp [hxp]
          · rw [if_neg hxp, if_neg hxb]
            have hax : a < x := (Finset.mem_Ioc.mp hx).1
            have hxb_lt : x < b :=
              lt_of_le_of_ne (Finset.mem_Ioc.mp hx).2 hxb
            have hxIco : x ∈ Finset.Ico a b :=
              Finset.mem_Ico.2 ⟨hax.le, hxb_lt⟩
            have hxp_le : x ≤ p := hpGreatest x hxIco
            have hxp_lt : x < p := lt_of_le_of_ne hxp_le hxp
            have hnotcov : ¬ x ⋖ b :=
              not_covBy_of_lt_of_lt hxp_lt hpb
            have hsubset : Finset.Ioc x b ⊆ Finset.Ioc a b :=
              Finset.Ioc_subset_Ioc hax.le le_rfl
            have hstrict : Finset.Ioc x b ⊂ Finset.Ioc a b := by
              refine Finset.ssubset_iff_subset_ne.2 ⟨hsubset, ?_⟩
              intro heq
              have hxnew : x ∈ Finset.Ioc x b := heq ▸ hx
              exact (Finset.mem_Ioc.mp hxnew).1.false
            have hcard : (Finset.Ioc x b).card < n := by
              rw [← hn]
              exact Finset.card_lt_card hstrict
            exact ih (Finset.Ioc x b).card hcard hxb_lt hnotcov rfl
      rw [Finset.sum_congr rfl hterms]
      have hsplit :
          (∑ x ∈ Finset.Ioc a b,
              if x = p then (-1 : ℤ) else if x = b then 1 else 0) =
            (∑ x ∈ Finset.Ioc a b,
              if x = p then (-1 : ℤ) else 0) +
            ∑ x ∈ Finset.Ioc a b, if x = b then (1 : ℤ) else 0 := by
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro x hx
        by_cases hxp : x = p
        · subst x
          simp [hpcov.ne]
        · simp [hxp]
      rw [hsplit, Finset.sum_ite_eq', Finset.sum_ite_eq']
      simp [hpIoc, hbIoc]

/-- The diagonal indicator kernel. -/
def incidenceIdentityKernel (a b : P) : ℤ :=
  if a = b then 1 else 0

/-- The `0/1` indicator kernel of the cover relation. Classical decidability
keeps the theorem independent of a `Decidable CovBy` instance. -/
noncomputable def coverKernel (a b : P) : ℤ := by
  classical
  exact if a ⋖ b then 1 else 0

/-- On a chain-interval poset, the incidence Möbius kernel is exactly
`I - C`, where `C` is the cover kernel. -/
theorem mu_eq_identityKernel_sub_coverKernel
    (hchain : HasChainIntervals P) (a b : P) :
    (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b =
      incidenceIdentityKernel a b - coverKernel a b := by
  classical
  by_cases heq : a = b
  · subst b
    have hncov : ¬ a ⋖ a := fun h => h.ne rfl
    simp [incidenceIdentityKernel, coverKernel, hncov]
  · by_cases hcov : a ⋖ b
    · simpa [incidenceIdentityKernel, coverKernel, heq, hcov] using
        mu_of_covBy hcov
    · by_cases hle : a ≤ b
      · have hlt : a < b := lt_of_le_of_ne hle heq
        simpa [incidenceIdentityKernel, coverKernel, heq, hcov] using
          mu_eq_zero_of_lt_not_covBy hchain hlt hcov
      · have hzero :
            (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b = 0 :=
          IncidenceAlgebra.apply_eq_zero_of_not_le hle _
        simpa [incidenceIdentityKernel, coverKernel, heq, hcov] using hzero

section ChangedOrder

variable {Q : Type*} [PartialOrder Q] [LocallyFiniteOrder Q]
  [DecidableEq Q]

/-- Sparse-update identity for two chain-interval containment orders. After
matching their carriers by `e`, the change in the Möbius kernel is exactly the
old cover kernel minus the new one. -/
theorem mu_update_eq_cover_update
    (hP : HasChainIntervals P) (hQ : HasChainIntervals Q)
    (e : P ≃ Q) (a b : P) :
    (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ Q) (e a) (e b) -
        (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b =
      coverKernel a b - coverKernel (P := Q) (e a) (e b) := by
  rw [mu_eq_identityKernel_sub_coverKernel hQ,
    mu_eq_identityKernel_sub_coverKernel hP]
  have hid : incidenceIdentityKernel (e a) (e b) =
      incidenceIdentityKernel a b := by
    simp [incidenceIdentityKernel]
  rw [hid]
  ring

/-- Consequently, an entry of the Möbius kernel can change only where the
matched cover relation changes. -/
theorem mu_eq_of_covBy_iff
    (hP : HasChainIntervals P) (hQ : HasChainIntervals Q)
    (e : P ≃ Q) {a b : P}
    (hcover : (a ⋖ b) ↔ (e a ⋖ e b)) :
    (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ Q) (e a) (e b) =
      (IncidenceAlgebra.mu ℤ : IncidenceAlgebra ℤ P) a b := by
  apply sub_eq_zero.mp
  rw [mu_update_eq_cover_update hP hQ e]
  apply sub_eq_zero.mpr
  classical
  simp only [coverKernel]
  exact if_congr hcover rfl rfl

end ChangedOrder

end ChainIntervalMobiusKernel

section ThreeLevelExperiment

/-- Labels on a three-deep laminar containment chain. -/
structure ThreeLevel where
  inner : ℤ
  middle : ℤ
  outer : ℤ
deriving DecidableEq, Repr

@[ext]
theorem ThreeLevel.ext {a b : ThreeLevel}
    (hinner : a.inner = b.inner)
    (hmiddle : a.middle = b.middle)
    (houter : a.outer = b.outer) :
    a = b := by
  cases a
  cases b
  simp_all

/-- Cumulative containment totals: each level includes every deeper level. -/
def zeta₃ (f : ThreeLevel) : ThreeLevel where
  inner := f.inner
  middle := f.inner + f.middle
  outer := f.inner + f.middle + f.outer

/-- Incidence Möbius inversion on the three-level chain. -/
def mobius₃ (F : ThreeLevel) : ThreeLevel where
  inner := F.inner
  middle := F.middle - F.inner
  outer := F.outer - F.middle

theorem mobius₃_zeta₃ (f : ThreeLevel) : mobius₃ (zeta₃ f) = f := by
  ext <;> simp [mobius₃, zeta₃]

theorem zeta₃_mobius₃ (F : ThreeLevel) : zeta₃ (mobius₃ F) = F := by
  ext <;> simp [mobius₃, zeta₃]

/-- Reverse the labels of the three-level chain. -/
def reverse₃ (F : ThreeLevel) : ThreeLevel where
  inner := F.outer
  middle := F.middle
  outer := F.inner

/-- Reversal of cumulative coordinates on the three stored wall levels.

This algebraic operation is internally invertible, but it is not by itself
the physical re-anchoring of chamber payloads: a three-wall arrangement has a
fourth, background chamber.  See `honest_reanchor_differs_from_naive` below. -/
def pivotZeta₃ (f : ThreeLevel) : ThreeLevel :=
  reverse₃ (zeta₃ (reverse₃ f))

/-- The incidence Möbius transform of the pivoted containment order. -/
def pivotMobius₃ (F : ThreeLevel) : ThreeLevel :=
  reverse₃ (mobius₃ (reverse₃ F))

theorem reverse₃_involutive (F : ThreeLevel) : reverse₃ (reverse₃ F) = F := by
  rfl

theorem pivotMobius₃_pivotZeta₃ (f : ThreeLevel) :
    pivotMobius₃ (pivotZeta₃ f) = f := by
  simp [pivotMobius₃, pivotZeta₃, reverse₃_involutive, mobius₃_zeta₃]

theorem pivotZeta₃_pivotMobius₃ (F : ThreeLevel) :
    pivotZeta₃ (pivotMobius₃ F) = F := by
  simp [pivotMobius₃, pivotZeta₃, reverse₃_involutive, zeta₃_mobius₃]

/-- A concrete pivot: direct labels `(1,2,4)` accumulate as `(7,6,4)` in the
reversed containment order, and its pivoted Möbius transform recovers them. -/
example :
    pivotZeta₃ { inner := 1, middle := 2, outer := 4 } =
      { inner := 7, middle := 6, outer := 4 } := by
  decide

example :
    pivotMobius₃ { inner := 7, middle := 6, outer := 4 } =
      { inner := 1, middle := 2, outer := 4 } := by
  decide

end ThreeLevelExperiment

section BackgroundAwareExperiment

/-- Direct payloads in all four chambers cut out by three nested walls.
`background` is the chamber containing the old distinguished point at
infinity. -/
structure FourChamber where
  deepest : ℤ
  next : ℤ
  nextOuter : ℤ
  background : ℤ
deriving DecidableEq, Repr

/-- Reverse the extended four-chamber chain.  Unlike `reverse₃`, this
includes the background chamber. -/
def reverse₄ (f : FourChamber) : FourChamber where
  deepest := f.background
  next := f.nextOuter
  nextOuter := f.next
  background := f.deepest

/-- Wall totals after moving infinity into the old deepest chamber.  Each new
selected side is the complement of the corresponding old selected side. -/
def honestReanchoredTotals₃ (f : FourChamber) : ThreeLevel where
  inner := f.next + f.nextOuter + f.background
  middle := f.nextOuter + f.background
  outer := f.background

/-- Accumulating the fully reversed four-chamber chain and then restoring the
wall names gives exactly the honest re-anchored totals.  This is the corrected
extended-chain reversal statement. -/
theorem honestReanchoredTotals₃_eq_extended_reversal (f : FourChamber) :
    honestReanchoredTotals₃ f =
      reverse₃ (zeta₃ {
        inner := (reverse₄ f).deepest
        middle := (reverse₄ f).next
        outer := (reverse₄ f).nextOuter }) := by
  ext <;> simp [honestReanchoredTotals₃, reverse₄, reverse₃, zeta₃,
    add_comm, add_left_comm]

/-- The concrete missing-background regression test.  Reversing only the
three stored labels gives `(7,6,4)`, whereas complementing the old wall totals
inside total mass `15` gives `(14,12,8)`. -/
theorem honest_reanchor_differs_from_naive :
    let chambers : FourChamber :=
      { deepest := 1, next := 2, nextOuter := 4, background := 8 }
    let stored : ThreeLevel :=
      { inner := chambers.deepest
        middle := chambers.next
        outer := chambers.nextOuter }
    pivotZeta₃ stored = { inner := 7, middle := 6, outer := 4 } ∧
      honestReanchoredTotals₃ chambers =
        { inner := 14, middle := 12, outer := 8 } ∧
      pivotZeta₃ stored ≠ honestReanchoredTotals₃ chambers := by
  decide

end BackgroundAwareExperiment

end ConformalMereology
