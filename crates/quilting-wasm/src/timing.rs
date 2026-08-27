//! Bounded, query-time timing distributions for runtime diagnostics.
//!
//! Recording is constant-time and allocation-free after the window reaches
//! capacity. Percentiles are derived only when a diagnostic snapshot is
//! requested, so the render loop never sorts or serializes samples.

use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug)]
pub(crate) struct TimingDistribution<const CAPACITY: usize> {
    samples: VecDeque<f64>,
    sum_ms: f64,
    total_samples: u64,
    rejected_samples: u64,
}

impl<const CAPACITY: usize> Default for TimingDistribution<CAPACITY> {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(CAPACITY),
            sum_ms: 0.0,
            total_samples: 0,
            rejected_samples: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimingDistributionSnapshot {
    pub total_samples: u64,
    pub window_samples: usize,
    pub capacity: usize,
    pub rejected_samples: u64,
    pub last_ms: f64,
    pub mean_ms: f64,
    pub minimum_ms: f64,
    pub maximum_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

impl<const CAPACITY: usize> TimingDistribution<CAPACITY> {
    pub(crate) fn record(&mut self, elapsed_ms: f64) -> bool {
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 || CAPACITY == 0 {
            self.rejected_samples = self.rejected_samples.saturating_add(1);
            return false;
        }
        if self.samples.len() == CAPACITY {
            self.sum_ms -= self
                .samples
                .pop_front()
                .expect("a full timing window has a front sample");
        }
        self.samples.push_back(elapsed_ms);
        self.sum_ms += elapsed_ms;
        self.total_samples = self.total_samples.saturating_add(1);
        true
    }

    pub(crate) fn clear(&mut self) {
        self.samples.clear();
        self.sum_ms = 0.0;
        self.total_samples = 0;
        self.rejected_samples = 0;
    }

    pub(crate) fn snapshot(&self) -> TimingDistributionSnapshot {
        let mut sorted = self.samples.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let sample_count = sorted.len();
        let percentile = |numerator: usize, denominator: usize| {
            if sample_count == 0 {
                return 0.0;
            }
            let rank = sample_count
                .saturating_mul(numerator)
                .div_ceil(denominator)
                .max(1);
            sorted[rank.saturating_sub(1).min(sample_count - 1)]
        };
        TimingDistributionSnapshot {
            total_samples: self.total_samples,
            window_samples: sample_count,
            capacity: CAPACITY,
            rejected_samples: self.rejected_samples,
            last_ms: self.samples.back().copied().unwrap_or(0.0),
            mean_ms: if sample_count == 0 {
                0.0
            } else {
                self.sum_ms / sample_count as f64
            },
            minimum_ms: sorted.first().copied().unwrap_or(0.0),
            maximum_ms: sorted.last().copied().unwrap_or(0.0),
            p50_ms: percentile(50, 100),
            p90_ms: percentile(90, 100),
            p95_ms: percentile(95, 100),
            p99_ms: percentile(99, 100),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn bounded_window_reports_nearest_rank_percentiles() {
        let mut distribution = TimingDistribution::<5>::default();
        for sample in [1.0, 2.0, 3.0, 4.0, 100.0] {
            assert!(distribution.record(sample));
        }
        let snapshot = distribution.snapshot();
        assert_eq!(snapshot.total_samples, 5);
        assert_eq!(snapshot.window_samples, 5);
        assert_eq!(snapshot.mean_ms, 22.0);
        assert_eq!(snapshot.p50_ms, 3.0);
        assert_eq!(snapshot.p90_ms, 100.0);
        assert_eq!(snapshot.p99_ms, 100.0);
    }

    #[wasm_bindgen_test]
    fn bounded_window_evicts_old_samples_without_losing_total_count() {
        let mut distribution = TimingDistribution::<3>::default();
        for sample in [10.0, 20.0, 30.0, 40.0] {
            assert!(distribution.record(sample));
        }
        let snapshot = distribution.snapshot();
        assert_eq!(snapshot.total_samples, 4);
        assert_eq!(snapshot.window_samples, 3);
        assert_eq!(snapshot.minimum_ms, 20.0);
        assert_eq!(snapshot.maximum_ms, 40.0);
        assert_eq!(snapshot.mean_ms, 30.0);
        assert_eq!(snapshot.last_ms, 40.0);
    }

    #[wasm_bindgen_test]
    fn invalid_samples_are_rejected_and_clear_resets_the_window() {
        let mut distribution = TimingDistribution::<2>::default();
        assert!(!distribution.record(f64::NAN));
        assert!(!distribution.record(-1.0));
        assert!(distribution.record(3.0));
        assert_eq!(distribution.snapshot().rejected_samples, 2);
        distribution.clear();
        assert_eq!(
            distribution.snapshot(),
            TimingDistributionSnapshot {
                total_samples: 0,
                window_samples: 0,
                capacity: 2,
                rejected_samples: 0,
                last_ms: 0.0,
                mean_ms: 0.0,
                minimum_ms: 0.0,
                maximum_ms: 0.0,
                p50_ms: 0.0,
                p90_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
            }
        );
    }
}
