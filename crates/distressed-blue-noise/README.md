# distressed-blue-noise

Deterministic variable-density Poisson-disk sampling for rectangles and an
equilateral triangle domain. Quilting uses it to distribute tessellation
samples, but the crate has no dependency on the rest of the workspace.

The density callback returns the local minimum point spacing. Values must be
positive and finite. A fixed seed produces byte-for-byte identical point
coordinates for the same configuration and callback.

```rust
use distressed_blue_noise::{Domain, PoissonSampler, SamplerConfig};

let sampler = PoissonSampler::new(SamplerConfig {
    k_candidates: 30,
    seed: 42,
    domain: Domain::Rectangle {
        width: 1.0,
        height: 1.0,
    },
});

// Denser on the left, progressively sparser toward the right.
let points = sampler.sample(|[x, _]| 0.03 + 0.12 * x);
assert!(points.iter().all(|[x, y]| {
    (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y)
}));
```

`with_seed_points` preserves valid caller-supplied points before sampling.
`sample_jittered` provides a faster jittered hex grid for a genuinely constant
spacing field; it evaluates the callback once at the domain center and should
not be used for variable density.

Licensed under either Apache-2.0 or MIT, at your option.
