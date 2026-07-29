//! Small fixed-size dense linear algebra.
//!
//! Every fitter in this crate ends up solving a tiny normal-equation system
//! (`JᵀJ δ = -Jᵀr`) inside a Gauss-Newton inner loop: 4×4 for the c-estimator,
//! 8×8 for the two-weight patch fit. At those sizes a direct Gaussian
//! elimination is both fast and accurate, and it keeps a general-purpose linear
//! algebra crate out of the WASM build.

/// Solve `A x = b` for a dense `N`×`N` system by Gaussian elimination with
/// partial pivoting.
///
/// A column whose pivot is numerically zero is skipped and the corresponding
/// unknown is left at zero, so a rank-deficient system yields the solution of
/// the well-determined subsystem rather than NaNs. Callers add
/// Levenberg-Marquardt damping to the diagonal, which normally keeps the system
/// nonsingular in the first place.
pub fn solve_gauss<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    const EPS: f64 = 1e-15;

    for col in 0..N {
        // Partial pivoting: move the largest remaining entry in this column up.
        let mut max_row = col;
        for row in (col + 1)..N {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        if a[max_row][col].abs() < EPS {
            continue; // singular column
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / pivot;
            for c in col..N {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = [0.0; N];
    for col in (0..N).rev() {
        if a[col][col].abs() < EPS {
            continue;
        }
        let mut sum = b[col];
        for c in (col + 1)..N {
            sum -= a[col][c] * x[c];
        }
        x[col] = sum / a[col][col];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = solve_gauss(a, [3.0, -1.0, 7.0]);
        assert_eq!(x, [3.0, -1.0, 7.0]);
    }

    #[test]
    fn test_solve_requires_pivoting() {
        // Zero leading pivot: only solvable if rows are exchanged.
        let a = [[0.0, 2.0], [1.0, 1.0]];
        let x = solve_gauss(a, [4.0, 3.0]);
        assert!((x[0] - 1.0).abs() < 1e-12, "x = {:?}", x);
        assert!((x[1] - 2.0).abs() < 1e-12, "x = {:?}", x);
    }

    #[test]
    fn test_solve_reproduces_rhs() {
        let a = [
            [4.0, 1.0, 2.0, 0.5],
            [1.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 5.0, 1.5],
            [0.5, 1.0, 1.5, 2.0],
        ];
        let b = [1.0, -2.0, 3.0, 0.25];
        let x = solve_gauss(a, b);
        for i in 0..4 {
            let axi: f64 = (0..4).map(|j| a[i][j] * x[j]).sum();
            assert!((axi - b[i]).abs() < 1e-10, "row {}: {} vs {}", i, axi, b[i]);
        }
    }

    #[test]
    fn test_singular_column_leaves_zero() {
        // Second unknown is unconstrained; it should come back as 0, not NaN.
        let a = [[1.0, 0.0], [0.0, 0.0]];
        let x = solve_gauss(a, [2.0, 0.0]);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert_eq!(x[1], 0.0);
    }
}
