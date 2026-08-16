//! Finite-poset payload coordinates for chamber aggregation.
//!
//! Zeta coordinates store cumulative payloads.  Incidence Möbius inversion
//! recovers the direct payloads.  No global bottom is assumed: disconnected
//! containment forests work without adding a synthetic element.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::ops::{AddAssign, SubAssign};

/// A finite partial order represented by its complete `≤` relation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinitePoset {
    leq: Vec<Vec<bool>>,
}

impl FinitePoset {
    pub fn new(leq: Vec<Vec<bool>>) -> Result<Self, PosetError> {
        let poset = Self { leq };
        poset.validate()?;
        Ok(poset)
    }

    pub fn len(&self) -> usize {
        self.leq.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leq.is_empty()
    }

    pub fn leq(&self, lower: usize, upper: usize) -> Result<bool, PosetError> {
        self.leq
            .get(lower)
            .and_then(|row| row.get(upper))
            .copied()
            .ok_or(PosetError::UnknownElement(lower.max(upper)))
    }

    pub fn validate(&self) -> Result<(), PosetError> {
        let n = self.leq.len();
        if self.leq.iter().any(|row| row.len() != n) {
            return Err(PosetError::NonSquare);
        }
        for i in 0..n {
            if !self.leq[i][i] {
                return Err(PosetError::NotReflexive(i));
            }
            for j in 0..n {
                if i != j && self.leq[i][j] && self.leq[j][i] {
                    return Err(PosetError::NotAntisymmetric(i, j));
                }
                for k in 0..n {
                    if self.leq[i][j] && self.leq[j][k] && !self.leq[i][k] {
                        return Err(PosetError::NotTransitive(i, j, k));
                    }
                }
            }
        }
        Ok(())
    }

    /// A deterministic linear extension: strict lower elements precede upper
    /// elements.  Sorting by principal-lower-set size has this property.
    pub fn linear_extension(&self) -> Vec<usize> {
        let mut elements = (0..self.len()).collect::<Vec<_>>();
        elements.sort_by_key(|&j| {
            let lower_count = (0..self.len()).filter(|&i| self.leq[i][j]).count();
            (lower_count, j)
        });
        elements
    }

    /// Cumulative/zeta coordinates: `total[j] = sum { direct[i] | i ≤ j }`.
    pub fn zeta_transform<T>(&self, direct: &[T]) -> Result<Vec<T>, PosetError>
    where
        T: Copy + Default + AddAssign,
    {
        self.require_payload_len(direct.len())?;
        let mut totals = vec![T::default(); self.len()];
        for (j, total) in totals.iter_mut().enumerate() {
            for (i, value) in direct.iter().enumerate() {
                if self.leq[i][j] {
                    *total += *value;
                }
            }
        }
        Ok(totals)
    }

    /// Incidence Möbius inversion, implemented by triangular recovery along a
    /// linear extension.  This works for any finite poset and any additive
    /// coefficient type with subtraction.
    pub fn mobius_transform<T>(&self, totals: &[T]) -> Result<Vec<T>, PosetError>
    where
        T: Copy + Default + SubAssign,
    {
        self.require_payload_len(totals.len())?;
        let mut direct = vec![T::default(); self.len()];
        for j in self.linear_extension() {
            let mut value = totals[j];
            for (i, recovered) in direct.iter().enumerate() {
                if i != j && self.leq[i][j] {
                    value -= *recovered;
                }
            }
            direct[j] = value;
        }
        Ok(direct)
    }

    /// Integer incidence Möbius coefficients `μ(i,j)`.
    pub fn mobius_kernel(&self) -> Vec<Vec<i64>> {
        let n = self.len();
        let order = self.linear_extension();
        let mut mu = vec![vec![0_i64; n]; n];
        for i in 0..n {
            mu[i][i] = 1;
            for &j in &order {
                if i == j || !self.leq[i][j] {
                    continue;
                }
                let mut sum = 0_i64;
                for (k, coefficient) in mu[i].iter().enumerate() {
                    if k != j && self.leq[i][k] && self.leq[k][j] {
                        sum += *coefficient;
                    }
                }
                mu[i][j] = -sum;
            }
        }
        mu
    }

    /// Whether `lower ⋖ upper`: strict order with no element in between.
    pub fn covers(&self, lower: usize, upper: usize) -> Result<bool, PosetError> {
        self.leq(lower, upper)?;
        if lower == upper || !self.leq[lower][upper] {
            return Ok(false);
        }
        Ok(!(0..self.len()).any(|middle| {
            middle != lower && middle != upper && self.leq[lower][middle] && self.leq[middle][upper]
        }))
    }

    /// Every closed interval is a chain.  This is the finite runtime analogue
    /// of the laminar-forest hypothesis used by the Lean sparsity theorem.
    pub fn has_chain_intervals(&self) -> bool {
        for lower in 0..self.len() {
            for upper in 0..self.len() {
                if !self.leq[lower][upper] {
                    continue;
                }
                let interval = (0..self.len())
                    .filter(|&x| self.leq[lower][x] && self.leq[x][upper])
                    .collect::<Vec<_>>();
                for &a in &interval {
                    for &b in &interval {
                        if !self.leq[a][b] && !self.leq[b][a] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Check the sparse forest formula `μ = I - C`, where `C` is the cover
    /// indicator.  Returning `false` outside chain-interval posets is expected.
    pub fn mobius_is_identity_minus_cover(&self) -> bool {
        let mu = self.mobius_kernel();
        (0..self.len()).all(|i| {
            (0..self.len()).all(|j| {
                let expected = if i == j {
                    1
                } else if self.covers(i, j).unwrap_or(false) {
                    -1
                } else {
                    0
                };
                mu[i][j] == expected
            })
        })
    }

    fn require_payload_len(&self, actual: usize) -> Result<(), PosetError> {
        if actual == self.len() {
            Ok(())
        } else {
            Err(PosetError::PayloadLength {
                expected: self.len(),
                actual,
            })
        }
    }
}

/// The historical wall-only reversal for three stored direct labels.  It does
/// not include the background chamber and is not physical re-anchoring.
pub fn naive_three_wall_reversal([deepest, next, outer]: [i64; 3]) -> [i64; 3] {
    [deepest + next + outer, next + outer, outer]
}

/// Honest totals after moving chart infinity into the old deepest chamber.
/// Input is `[deepest, next, next_outer, background]`.
pub fn honest_three_wall_reanchor([_deepest, next, next_outer, background]: [i64; 4]) -> [i64; 3] {
    [
        next + next_outer + background,
        next_outer + background,
        background,
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosetError {
    NonSquare,
    NotReflexive(usize),
    NotAntisymmetric(usize, usize),
    NotTransitive(usize, usize, usize),
    UnknownElement(usize),
    PayloadLength { expected: usize, actual: usize },
}

impl fmt::Display for PosetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare => write!(f, "finite-poset relation matrix must be square"),
            Self::NotReflexive(i) => write!(f, "finite-poset relation is not reflexive at {i}"),
            Self::NotAntisymmetric(i, j) => {
                write!(f, "finite-poset relation is not antisymmetric at ({i},{j})")
            }
            Self::NotTransitive(i, j, k) => {
                write!(
                    f,
                    "finite-poset relation is not transitive at ({i},{j},{k})"
                )
            }
            Self::UnknownElement(i) => write!(f, "unknown finite-poset element {i}"),
            Self::PayloadLength { expected, actual } => write!(
                f,
                "payload length {actual} does not match finite-poset size {expected}"
            ),
        }
    }
}

impl Error for PosetError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn disconnected_chains() -> FinitePoset {
        // 0 < 1 and 2 < 3, with no global bottom and no comparisons across
        // components.
        FinitePoset::new(vec![
            vec![true, true, false, false],
            vec![false, true, false, false],
            vec![false, false, true, true],
            vec![false, false, false, true],
        ])
        .unwrap()
    }

    #[test]
    fn zeta_and_mobius_round_trip_without_a_global_bottom() {
        let poset = disconnected_chains();
        let direct = [2_i64, 3, 5, 7];
        let totals = poset.zeta_transform(&direct).unwrap();
        assert_eq!(totals, vec![2, 5, 5, 12]);
        assert_eq!(poset.mobius_transform(&totals).unwrap(), direct);
    }

    #[test]
    fn chain_interval_forests_have_identity_minus_cover_kernel() {
        let poset = disconnected_chains();
        assert!(poset.has_chain_intervals());
        assert!(poset.mobius_is_identity_minus_cover());
        assert_eq!(
            poset.mobius_kernel(),
            vec![
                vec![1, -1, 0, 0],
                vec![0, 1, 0, 0],
                vec![0, 0, 1, -1],
                vec![0, 0, 0, 1],
            ]
        );
    }

    #[test]
    fn diamond_overlap_has_non_cover_mobius_coefficient() {
        // 0 < 1,2 < 3.  The +1 at (0,3) is the inclusion-exclusion overlap
        // term and shows why a bare containment forest is a special case.
        let diamond = FinitePoset::new(vec![
            vec![true, true, true, true],
            vec![false, true, false, true],
            vec![false, false, true, true],
            vec![false, false, false, true],
        ])
        .unwrap();
        assert!(!diamond.has_chain_intervals());
        assert!(!diamond.mobius_is_identity_minus_cover());
        assert_eq!(diamond.mobius_kernel()[0][3], 1);
    }

    #[test]
    fn honest_background_regression_is_not_wall_only_reversal() {
        let naive = naive_three_wall_reversal([1, 2, 4]);
        let honest = honest_three_wall_reanchor([1, 2, 4, 8]);
        assert_eq!(naive, [7, 6, 4]);
        assert_eq!(honest, [14, 12, 8]);
        assert_ne!(naive, honest);
    }
}
