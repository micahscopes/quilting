import ConformalMereology.SphericalInversion

/-!
# Conformally equivariant round-side indices

This file isolates the part of a spatial hierarchy that survives a conformal
change of chart.  Nodes carry oriented open round sides, and the finite
preorder records certified carrier containment.  There is intentionally no
distance, nearest-neighbour order, AABB, or Morton-code claim here: those are
not invariants of a general Möbius transformation.

The useful invariant layer is incidence:

* a certified `RoundSideAutomorphism` sends every carrier to another open
  round side;
* carrier containment and disjointness are reflected as well as preserved;
* testing a pushed carrier against a destination-space query is equivalent to
  testing the source carrier against the query pulled through the inverse;
* pruning a parent conservatively prunes every contained descendant.

The query is an arbitrary set.  This deliberately covers a single round side,
an intersection of pulled-back frustum sides, and a proximity region with the
same theorem.  Only the indexed bounds themselves need the round-side
certificate.
-/

open Set

namespace ConformalMereology

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A finite containment hierarchy whose node carriers are oriented open
round sides of the conformal compactification.  The preorder is the
descendant relation: `i ≤ j` means that node `i` is contained in node `j`.

The structure does not prescribe how the hierarchy is laid out in memory.
In particular, `ι` may later be a persistent or merklized node identifier. -/
structure RoundSideIndex (ι : Type*) [Fintype ι] [Preorder ι] where
  carrier : ι → Set (OnePoint E)
  isRoundSide : ∀ i, IsOpenRoundSide (carrier i)
  carrier_mono : Monotone carrier

namespace RoundSideIndex

variable {ι : Type*} [Fintype ι] [Preorder ι]

/-- Transport every bound through a certified conformal generator word while
leaving the finite hierarchy topology unchanged. -/
def map (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E) :
    RoundSideIndex (E := E) ι where
  carrier i := A.toEquiv '' I.carrier i
  isRoundSide i := A.mapsRoundSide (I.isRoundSide i)
  carrier_mono _ _ hij := Set.image_mono (I.carrier_mono hij)

@[simp]
theorem map_carrier (I : RoundSideIndex (E := E) ι)
    (A : RoundSideAutomorphism E) (i : ι) :
    (I.map A).carrier i = A.toEquiv '' I.carrier i :=
  rfl

/-- Exact carrier containment is equivariant, not merely preserved. -/
theorem map_carrier_subset_map_carrier_iff
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    (i j : ι) :
    (I.map A).carrier i ⊆ (I.map A).carrier j ↔
      I.carrier i ⊆ I.carrier j := by
  simp only [map_carrier]
  exact Set.image_subset_image_iff A.toEquiv.injective

/-- Exact disjointness of indexed carriers is equivariant.  This is the
incidence fact behind conformally stable sibling-pruning certificates. -/
theorem map_carrier_disjoint_map_carrier_iff
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    (i j : ι) :
    Disjoint ((I.map A).carrier i) ((I.map A).carrier j) ↔
      Disjoint (I.carrier i) (I.carrier j) := by
  simp only [map_carrier]
  exact Set.disjoint_image_iff A.toEquiv.injective

/-- Pull a destination-space query back to the source chart.  The query need
not itself be a single round side; for example, it may be an intersection of
the six pulled-back round sides of a transformed frustum. -/
def pullQuery (A : RoundSideAutomorphism E) (query : Set (OnePoint E)) :
    Set (OnePoint E) :=
  A.toEquiv.symm '' query

@[simp]
theorem mem_pullQuery_iff (A : RoundSideAutomorphism E)
    (query : Set (OnePoint E)) (x : OnePoint E) :
    x ∈ pullQuery A query ↔ A.toEquiv x ∈ query := by
  simp [pullQuery]

/-- Pulling a query through the inverse is exactly equivalent to pushing the
indexed carrier forward.  This is the central query law for keeping an
immutable source-space hierarchy while the view undergoes conformal motion. -/
theorem map_carrier_disjoint_iff_pullQuery
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    (i : ι) (query : Set (OnePoint E)) :
    Disjoint ((I.map A).carrier i) query ↔
      Disjoint (I.carrier i) (pullQuery A query) := by
  constructor
  · intro hpushed
    refine Set.disjoint_left.2 ?_
    intro x hxCarrier hxQuery
    exact Set.disjoint_left.1 hpushed
      (by exact ⟨x, hxCarrier, rfl⟩)
      ((mem_pullQuery_iff A query x).1 hxQuery)
  · intro hpulled
    refine Set.disjoint_left.2 ?_
    intro y hyCarrier hyQuery
    obtain ⟨x, hxCarrier, rfl⟩ := hyCarrier
    exact Set.disjoint_left.1 hpulled hxCarrier
      ((mem_pullQuery_iff A query x).2 hyQuery)

/-- The equivalent positive form of the query law: a pushed carrier hits a
destination query exactly when its source carrier hits the pulled query. -/
theorem map_carrier_hits_iff_pullQuery
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    (i : ι) (query : Set (OnePoint E)) :
    ¬Disjoint ((I.map A).carrier i) query ↔
      ¬Disjoint (I.carrier i) (pullQuery A query) := by
  rw [map_carrier_disjoint_iff_pullQuery]

/-- Conservative hierarchy pruning in the destination chart.  Once a parent
bound misses the query, every descendant certified by the source hierarchy
also misses it after the conformal transformation. -/
theorem descendant_pruned
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    {child parent : ι} (hchild : child ≤ parent)
    {query : Set (OnePoint E)}
    (hparent : Disjoint ((I.map A).carrier parent) query) :
    Disjoint ((I.map A).carrier child) query := by
  refine Set.disjoint_left.2 ?_
  intro x hxChild hxQuery
  exact Set.disjoint_left.1 hparent
    ((I.map A).carrier_mono hchild hxChild) hxQuery

/-- Source-space version of conservative pruning.  Combined with
`map_carrier_disjoint_iff_pullQuery`, this is the form used when the immutable
index stays in source space and a frustum or proximity query is pulled back. -/
theorem descendant_pruned_against_pullQuery
    (I : RoundSideIndex (E := E) ι) (A : RoundSideAutomorphism E)
    {child parent : ι} (hchild : child ≤ parent)
    {query : Set (OnePoint E)}
    (hparent : Disjoint (I.carrier parent) (pullQuery A query)) :
    Disjoint ((I.map A).carrier child) query := by
  apply (map_carrier_disjoint_iff_pullQuery I A child query).2
  refine Set.disjoint_left.2 ?_
  intro x hxChild hxQuery
  exact Set.disjoint_left.1 hparent (I.carrier_mono hchild hxChild) hxQuery

end RoundSideIndex

end

end ConformalMereology
