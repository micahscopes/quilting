/// Compressed Sparse Row matrix and Conjugate Gradient solver.

/// CSR sparse matrix.
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Build from (row, col, value) triplets. Duplicate entries are summed.
    pub fn from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> Self {
        // Count entries per row
        let mut row_counts = vec![0usize; n];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }

        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        let nnz = row_ptr[n];
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut offsets = row_ptr[..n].to_vec();

        for &(r, c, v) in triplets {
            let pos = offsets[r];
            col_idx[pos] = c;
            values[pos] = v;
            offsets[r] += 1;
        }

        // Sort each row by column index and merge duplicates
        let mut result = Self { n, row_ptr, col_idx, values };
        result.sort_and_merge();
        result
    }

    fn sort_and_merge(&mut self) {
        for i in 0..self.n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            if start >= end { continue; }

            // Sort by column
            let slice_len = end - start;
            let mut pairs: Vec<(usize, f64)> = (0..slice_len)
                .map(|k| (self.col_idx[start + k], self.values[start + k]))
                .collect();
            pairs.sort_by_key(|&(c, _)| c);

            // Write back (merge duplicates later if needed)
            for (k, &(c, v)) in pairs.iter().enumerate() {
                self.col_idx[start + k] = c;
                self.values[start + k] = v;
            }
        }
    }

    /// y = A * x
    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut sum = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = sum;
        }
    }

    /// Extract diagonal values.
    fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.n];
        for i in 0..self.n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                if self.col_idx[k] == i {
                    diag[i] = self.values[k];
                    break;
                }
            }
        }
        diag
    }
}

/// Jacobi-preconditioned Conjugate Gradient.
/// Solves A*x = b. Returns the number of iterations used.
pub fn solve_cg(
    a: &CsrMatrix,
    b: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
) -> usize {
    let n = a.n;
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    // Jacobi preconditioner: M = diag(A)
    let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag.iter()
        .map(|&d| if d.abs() > 1e-15 { 1.0 / d } else { 1.0 })
        .collect();

    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];

    // r = b - A*x
    a.mul_vec(x, &mut r);
    for i in 0..n { r[i] = b[i] - r[i]; }

    // z = M^-1 * r
    for i in 0..n { z[i] = inv_diag[i] * r[i]; }

    // p = z
    p.copy_from_slice(&z);

    let mut rz = dot(&r, &z);

    let b_norm = dot(b, b).sqrt();
    let threshold = if b_norm > 1e-15 { tol * b_norm } else { tol };

    for iter in 0..max_iter {
        let r_norm = dot(&r, &r).sqrt();
        if r_norm < threshold {
            return iter;
        }

        // ap = A*p
        a.mul_vec(&p, &mut ap);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 { return iter; }

        let alpha = rz / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        // z = M^-1 * r
        for i in 0..n { z[i] = inv_diag[i] * r[i]; }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz.max(1e-30);
        rz = rz_new;

        for i in 0..n { p[i] = z[i] + beta * p[i]; }
    }

    max_iter
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_system() {
        // A = I, b = [1, 2, 3], solution = [1, 2, 3]
        let triplets: Vec<(usize, usize, f64)> = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let a = CsrMatrix::from_triplets(3, &triplets);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let iters = solve_cg(&a, &b, &mut x, 100, 1e-10);
        assert!(iters <= 1);
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-8, "x[{}]={}, expected {}", i, x[i], b[i]);
        }
    }

    #[test]
    fn test_laplacian_1d() {
        // 1D Laplacian: -1, 2, -1 tridiagonal (4x4 interior of 6-point grid)
        let n = 4;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 { triplets.push((i, i - 1, -1.0)); }
            if i + 1 < n { triplets.push((i, i + 1, -1.0)); }
        }
        let a = CsrMatrix::from_triplets(n, &triplets);
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut x = vec![0.0; n];
        let iters = solve_cg(&a, &b, &mut x, 200, 1e-10);
        assert!(iters < 20, "CG should converge quickly, took {} iters", iters);

        // Verify A*x = b
        let mut check = vec![0.0; n];
        a.mul_vec(&x, &mut check);
        for i in 0..n {
            assert!((check[i] - b[i]).abs() < 1e-8, "residual at {} = {}", i, check[i] - b[i]);
        }
    }

    #[test]
    fn test_2d_laplacian() {
        // 3x3 interior grid of a 5x5 grid => 9 unknowns
        let n = 9;
        let mut triplets = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                let idx = i * 3 + j;
                triplets.push((idx, idx, 4.0));
                if j > 0 { triplets.push((idx, idx - 1, -1.0)); }
                if j < 2 { triplets.push((idx, idx + 1, -1.0)); }
                if i > 0 { triplets.push((idx, idx - 3, -1.0)); }
                if i < 2 { triplets.push((idx, idx + 3, -1.0)); }
            }
        }
        let a = CsrMatrix::from_triplets(n, &triplets);
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let iters = solve_cg(&a, &b, &mut x, 200, 1e-10);
        assert!(iters < 50, "2D Laplacian CG took {} iters", iters);

        let mut check = vec![0.0; n];
        a.mul_vec(&x, &mut check);
        for i in 0..n {
            assert!((check[i] - b[i]).abs() < 1e-8);
        }
    }
}
