/// Cubic Hermite trajectories in 4D spacetime.
///
/// A vertex traces a path through 3D space over time. We represent this as
/// a sequence of cubic Hermite segments, each defined by position and velocity
/// at its endpoints. The 4D point is (x, y, z, t).

/// A cubic Hermite segment: position + velocity at start and end.
#[derive(Debug, Clone)]
pub struct HermiteSegment {
    pub t_start: f64,
    pub t_end: f64,
    pub pos_start: [f64; 3],
    pub pos_end: [f64; 3],
    pub vel_start: [f64; 3],
    pub vel_end: [f64; 3],
}

/// A complete vertex trajectory -- sequence of cubic Hermite segments.
#[derive(Debug, Clone)]
pub struct VertexTrajectory {
    pub segments: Vec<HermiteSegment>,
}

impl HermiteSegment {
    /// Evaluate cubic Hermite at parameter u in [0,1].
    ///
    /// Uses the standard Hermite basis:
    ///   h00(u) = 2u^3 - 3u^2 + 1
    ///   h10(u) = u^3 - 2u^2 + u
    ///   h01(u) = -2u^3 + 3u^2
    ///   h11(u) = u^3 - u^2
    pub fn eval(&self, u: f64) -> [f64; 3] {
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        let dt = self.t_end - self.t_start;

        let mut result = [0.0; 3];
        for i in 0..3 {
            result[i] = h00 * self.pos_start[i]
                + h10 * dt * self.vel_start[i]
                + h01 * self.pos_end[i]
                + h11 * dt * self.vel_end[i];
        }
        result
    }

    /// Time at parameter u.
    fn time_at(&self, u: f64) -> f64 {
        self.t_start + u * (self.t_end - self.t_start)
    }

    /// Find intersections of this segment with a hyperplane.
    ///
    /// The 4D point at parameter u is (x(u), y(u), z(u), t(u)) where
    /// t(u) = t_start + u * (t_end - t_start).
    ///
    /// The hyperplane equation dot(normal, point) = offset becomes a cubic in u.
    /// We solve it using Newton's method with multiple starting points.
    pub fn intersect_hyperplane(&self, normal: [f64; 4], offset: f64) -> Vec<(f64, [f64; 3])> {
        // Build the cubic polynomial coefficients for f(u) = dot(normal, P(u)) - offset.
        //
        // The Hermite basis in monomial form:
        //   h00(u) = 1 - 3u^2 + 2u^3
        //   h10(u) = u - 2u^2 + u^3
        //   h01(u) = 3u^2 - 2u^3
        //   h11(u) = -u^2 + u^3
        //
        // For each spatial dimension i, the position component is:
        //   x_i(u) = p0_i * h00 + dt * v0_i * h10 + p1_i * h01 + dt * v1_i * h11
        //
        // The time dimension: t(u) = t_start + u * dt
        //
        // So f(u) = sum_i(n_i * x_i(u)) + n_t * t(u) - offset
        //         = c0 + c1*u + c2*u^2 + c3*u^3

        let dt = self.t_end - self.t_start;
        let n = normal;

        // Accumulate polynomial coefficients across spatial dims + time.
        let mut c0: f64 = 0.0;
        let mut c1: f64 = 0.0;
        let mut c2: f64 = 0.0;
        let mut c3: f64 = 0.0;

        for i in 0..3 {
            let p0 = self.pos_start[i];
            let p1 = self.pos_end[i];
            let v0 = self.vel_start[i] * dt;
            let v1 = self.vel_end[i] * dt;

            // Monomial coefficients for this dimension:
            // constant: p0
            // u^1: v0
            // u^2: -3*p0 - 2*v0 + 3*p1 - v1
            // u^3: 2*p0 + v0 - 2*p1 + v1
            c0 += n[i] * p0;
            c1 += n[i] * v0;
            c2 += n[i] * (-3.0 * p0 - 2.0 * v0 + 3.0 * p1 - v1);
            c3 += n[i] * (2.0 * p0 + v0 - 2.0 * p1 + v1);
        }

        // Time dimension: t(u) = t_start + u * dt
        c0 += n[3] * self.t_start;
        c1 += n[3] * dt;

        // Subtract offset
        c0 -= offset;

        // Now solve c0 + c1*u + c2*u^2 + c3*u^3 = 0 for u in [0,1]
        let roots = solve_cubic_in_unit(c0, c1, c2, c3);

        roots
            .into_iter()
            .map(|u| {
                let t = self.time_at(u);
                let pos = self.eval(u);
                (t, pos)
            })
            .collect()
    }
}

impl VertexTrajectory {
    /// Evaluate position at time t.
    ///
    /// Finds the segment containing t and evaluates it. If t is outside the
    /// trajectory range, clamps to the nearest endpoint.
    pub fn eval(&self, t: f64) -> [f64; 3] {
        if self.segments.is_empty() {
            return [0.0; 3];
        }

        // Before first segment
        let first = &self.segments[0];
        if t <= first.t_start {
            return first.pos_start;
        }

        // After last segment
        let last = &self.segments[self.segments.len() - 1];
        if t >= last.t_end {
            return last.pos_end;
        }

        // Find the right segment
        for seg in &self.segments {
            if t >= seg.t_start && t <= seg.t_end {
                let dt = seg.t_end - seg.t_start;
                let u = if dt > 0.0 { (t - seg.t_start) / dt } else { 0.0 };
                return seg.eval(u);
            }
        }

        // Shouldn't reach here, but just in case
        last.pos_end
    }

    /// Find all intersections with hyperplane dot(normal, (x,y,z,t)) = offset.
    /// Returns Vec<(t_value, position_3d)>.
    pub fn intersect_hyperplane(&self, normal: [f64; 4], offset: f64) -> Vec<(f64, [f64; 3])> {
        let mut results = Vec::new();
        for seg in &self.segments {
            results.extend(seg.intersect_hyperplane(normal, offset));
        }

        // Check trajectory endpoints explicitly — the cubic solver can miss
        // roots at the exact boundary (u=0 or u=1) due to floating point.
        if !self.segments.is_empty() {
            let eps = 1e-6;
            let first = &self.segments[0];
            let p = first.pos_start;
            let val = normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + normal[3]*first.t_start;
            if (val - offset).abs() < eps && !results.iter().any(|&(t,_)| (t - first.t_start).abs() < eps) {
                results.push((first.t_start, p));
            }

            let last = &self.segments[self.segments.len() - 1];
            let p = last.pos_end;
            let val = normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + normal[3]*last.t_end;
            if (val - offset).abs() < eps && !results.iter().any(|&(t,_)| (t - last.t_end).abs() < eps) {
                results.push((last.t_end, p));
            }
        }

        results
    }
}

/// Solve c0 + c1*u + c2*u^2 + c3*u^3 = 0 for real roots in [0, 1].
///
/// Uses Newton's method from multiple starting points. Not the most elegant
/// approach, but robust and simple.
fn solve_cubic_in_unit(c0: f64, c1: f64, c2: f64, c3: f64) -> Vec<f64> {
    let eps = 1e-10;

    // Evaluate polynomial
    let f = |u: f64| -> f64 { c0 + u * (c1 + u * (c2 + u * c3)) };
    let df = |u: f64| -> f64 { c1 + u * (2.0 * c2 + u * 3.0 * c3) };

    // If cubic coefficient is essentially zero, solve as quadratic
    if c3.abs() < eps {
        return solve_quadratic_in_unit(c0, c1, c2);
    }

    // Use analytical approach: find critical points, then check sign changes
    // in each monotone interval.
    //
    // f'(u) = c1 + 2*c2*u + 3*c3*u^2 = 0
    // This is a quadratic in u.
    let disc = 4.0 * c2 * c2 - 12.0 * c3 * c1;

    let mut intervals: Vec<(f64, f64)> = Vec::new();

    if disc > 0.0 {
        let sqrt_disc = disc.sqrt();
        let u_crit1 = (-2.0 * c2 - sqrt_disc) / (6.0 * c3);
        let u_crit2 = (-2.0 * c2 + sqrt_disc) / (6.0 * c3);
        let (u_lo, u_hi) = if u_crit1 < u_crit2 {
            (u_crit1, u_crit2)
        } else {
            (u_crit2, u_crit1)
        };

        // Build intervals [0, u_lo], [u_lo, u_hi], [u_hi, 1]
        // but only the parts inside [0, 1]
        let breakpoints: Vec<f64> = [0.0, u_lo, u_hi, 1.0]
            .iter()
            .copied()
            .filter(|&u| u >= 0.0 && u <= 1.0)
            .collect();

        for w in breakpoints.windows(2) {
            intervals.push((w[0], w[1]));
        }
        // If no critical points in [0,1], just use the whole interval
        if intervals.is_empty() {
            intervals.push((0.0, 1.0));
        }
    } else {
        // Monotone on [0,1]
        intervals.push((0.0, 1.0));
    }

    let mut roots = Vec::new();

    for &(a, b) in &intervals {
        let fa = f(a);
        let fb = f(b);

        // Check endpoints
        if fa.abs() < eps {
            let candidate = a;
            if !roots.iter().any(|&r: &f64| (r - candidate).abs() < 1e-8) {
                roots.push(candidate);
            }
            continue;
        }
        if fb.abs() < eps {
            let candidate = b;
            if !roots.iter().any(|&r: &f64| (r - candidate).abs() < 1e-8) {
                roots.push(candidate);
            }
            continue;
        }

        // Sign change means a root in this interval
        if fa * fb > 0.0 {
            continue;
        }

        // Newton's method starting from midpoint, with bisection fallback
        let mut u = (a + b) / 2.0;
        let mut lo = a;
        let mut hi = b;
        if fa > 0.0 {
            std::mem::swap(&mut lo, &mut hi);
        }

        for _ in 0..64 {
            let fv = f(u);
            let dfv = df(u);

            if fv.abs() < eps {
                break;
            }

            // Newton step
            let mut u_next = if dfv.abs() > eps {
                u - fv / dfv
            } else {
                // Bisection
                (lo + hi) / 2.0
            };

            // If Newton went out of bracket, fall back to bisection
            if u_next < a || u_next > b {
                u_next = (lo + hi) / 2.0;
            }

            // Update bracket
            if fv < 0.0 {
                lo = u;
            } else {
                hi = u;
            }

            u = u_next;
        }

        if u >= -eps && u <= 1.0 + eps {
            let u_clamped = u.clamp(0.0, 1.0);
            if !roots.iter().any(|&r: &f64| (r - u_clamped).abs() < 1e-8) {
                roots.push(u_clamped);
            }
        }
    }

    roots
}

/// Solve c0 + c1*u + c2*u^2 = 0 for real roots in [0, 1].
fn solve_quadratic_in_unit(c0: f64, c1: f64, c2: f64) -> Vec<f64> {
    let eps = 1e-10;

    if c2.abs() < eps {
        // Linear
        if c1.abs() < eps {
            return vec![];
        }
        let u = -c0 / c1;
        if u >= -eps && u <= 1.0 + eps {
            return vec![u.clamp(0.0, 1.0)];
        }
        return vec![];
    }

    let disc = c1 * c1 - 4.0 * c2 * c0;
    if disc < 0.0 {
        return vec![];
    }

    let sqrt_disc = disc.sqrt();
    let mut roots = Vec::new();
    for u in [(-c1 - sqrt_disc) / (2.0 * c2), (-c1 + sqrt_disc) / (2.0 * c2)] {
        if u >= -eps && u <= 1.0 + eps {
            let u_clamped = u.clamp(0.0, 1.0);
            if !roots.iter().any(|&r: &f64| (r - u_clamped).abs() < 1e-8) {
                roots.push(u_clamped);
            }
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_trajectory(p0: [f64; 3], p1: [f64; 3], t0: f64, t1: f64) -> VertexTrajectory {
        let dt = t1 - t0;
        let vel = if dt > 0.0 {
            [
                (p1[0] - p0[0]) / dt,
                (p1[1] - p0[1]) / dt,
                (p1[2] - p0[2]) / dt,
            ]
        } else {
            [0.0; 3]
        };
        VertexTrajectory {
            segments: vec![HermiteSegment {
                t_start: t0,
                t_end: t1,
                pos_start: p0,
                pos_end: p1,
                vel_start: vel,
                vel_end: vel,
            }],
        }
    }

    #[test]
    fn hermite_eval_endpoints() {
        let seg = HermiteSegment {
            t_start: 0.0,
            t_end: 1.0,
            pos_start: [1.0, 2.0, 3.0],
            pos_end: [4.0, 5.0, 6.0],
            vel_start: [3.0, 3.0, 3.0],
            vel_end: [3.0, 3.0, 3.0],
        };
        let p0 = seg.eval(0.0);
        let p1 = seg.eval(1.0);
        for i in 0..3 {
            assert!((p0[i] - seg.pos_start[i]).abs() < 1e-12);
            assert!((p1[i] - seg.pos_end[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_trajectory_time_slice() {
        // Vertex moves from (0,0,0) to (1,0,0) over [0, 1].
        let traj = linear_trajectory([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 1.0);

        // Time-aligned hyperplane: t = 0.5
        let normal = [0.0, 0.0, 0.0, 1.0];
        let offset = 0.5;
        let hits = traj.intersect_hyperplane(normal, offset);

        assert_eq!(hits.len(), 1);
        let (t, pos) = &hits[0];
        assert!((*t - 0.5).abs() < 1e-6, "t = {}, expected 0.5", t);
        assert!((pos[0] - 0.5).abs() < 1e-6, "x = {}, expected 0.5", pos[0]);
    }

    #[test]
    fn cubic_trajectory_multiple_intersections() {
        // Build a trajectory that oscillates, creating multiple crossings
        // with a spatial hyperplane.
        let traj = VertexTrajectory {
            segments: vec![
                HermiteSegment {
                    t_start: 0.0,
                    t_end: 1.0,
                    pos_start: [-1.0, 0.0, 0.0],
                    pos_end: [1.0, 0.0, 0.0],
                    vel_start: [4.0, 0.0, 0.0],
                    vel_end: [4.0, 0.0, 0.0],
                },
                HermiteSegment {
                    t_start: 1.0,
                    t_end: 2.0,
                    pos_start: [1.0, 0.0, 0.0],
                    pos_end: [-1.0, 0.0, 0.0],
                    vel_start: [-4.0, 0.0, 0.0],
                    vel_end: [-4.0, 0.0, 0.0],
                },
            ],
        };

        // Hyperplane: x = 0 (a spatial plane)
        let normal = [1.0, 0.0, 0.0, 0.0];
        let offset = 0.0;
        let hits = traj.intersect_hyperplane(normal, offset);

        // The trajectory oscillates through x=0, should have multiple crossings
        assert!(
            hits.len() >= 2,
            "Expected at least 2 crossings, got {}",
            hits.len()
        );

        // All intersection positions should have x near 0
        for (_, pos) in &hits {
            assert!(
                pos[0].abs() < 1e-4,
                "Intersection x should be ~0, got {}",
                pos[0]
            );
        }
    }

    #[test]
    fn no_intersection_parallel() {
        // Trajectory moves along X axis. Hyperplane is x = 100 (never reached).
        let traj = linear_trajectory([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 1.0);
        let normal = [1.0, 0.0, 0.0, 0.0];
        let offset = 100.0;
        let hits = traj.intersect_hyperplane(normal, offset);
        assert!(hits.is_empty(), "Expected no intersections, got {}", hits.len());
    }

    #[test]
    fn eval_trajectory_clamps() {
        let traj = linear_trajectory([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 1.0);
        let before = traj.eval(-1.0);
        let after = traj.eval(2.0);
        assert!((before[0] - 0.0).abs() < 1e-12);
        assert!((after[0] - 1.0).abs() < 1e-12);
    }
}
