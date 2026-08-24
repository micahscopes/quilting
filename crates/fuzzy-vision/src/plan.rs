use core::iter::FusedIterator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShaderId {
    Init,
    Step,
    Firmness,
    BlurDir,
    WeightRadial,
    WeightConformal,
    ReduceMinmax,
    WeightConformalNorm,
    WeightKawase,
    Passthrough,
    MipComposite,
    GaussDown,
}

impl ShaderId {
    pub(super) const ALL: [Self; 12] = [
        Self::Init,
        Self::Step,
        Self::Firmness,
        Self::BlurDir,
        Self::WeightRadial,
        Self::WeightConformal,
        Self::ReduceMinmax,
        Self::WeightConformalNorm,
        Self::WeightKawase,
        Self::Passthrough,
        Self::MipComposite,
        Self::GaussDown,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Extent2d {
    width: u32,
    height: u32,
}

impl Extent2d {
    pub(super) const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub(super) const fn width(self) -> u32 {
        self.width
    }

    pub(super) const fn height(self) -> u32 {
        self.height
    }

    const fn primary_step_count(self) -> u32 {
        let max_dimension = if self.width > self.height {
            self.width
        } else {
            self.height
        };
        if max_dimension <= 1 {
            0
        } else {
            u32::BITS - (max_dimension - 1).leading_zeros()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PingPong {
    Ping,
    Pong,
}

impl PingPong {
    const fn other(self) -> Self {
        match self {
            Self::Ping => Self::Pong,
            Self::Pong => Self::Ping,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JfaPropagationStep {
    pub(super) step: u32,
    pub(super) source: PingPong,
    pub(super) destination: PingPong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JfaPropagationPlan {
    extent: Extent2d,
}

impl JfaPropagationPlan {
    const CLEANUP_STEP_COUNT: u8 = 2;

    pub(super) const fn new(extent: Extent2d) -> Self {
        Self { extent }
    }

    pub(super) const fn extent(self) -> Extent2d {
        self.extent
    }

    pub(super) const fn initial_buffer(self) -> PingPong {
        PingPong::Ping
    }

    pub(super) const fn primary_step_count(self) -> u32 {
        self.extent.primary_step_count()
    }

    pub(super) fn primary_steps(self) -> PrimaryJfaSteps {
        let remaining = self.primary_step_count();
        let next_step = if remaining == 0 {
            0
        } else {
            1 << (remaining - 1)
        };
        PrimaryJfaSteps {
            next_step,
            remaining,
            source: self.initial_buffer(),
        }
    }

    pub(super) fn cleanup_steps(self) -> CleanupJfaSteps {
        CleanupJfaSteps {
            remaining: Self::CLEANUP_STEP_COUNT,
            source: self.buffer_after_primary_steps(),
        }
    }

    pub(super) const fn final_buffer(self) -> PingPong {
        // Exactly two cleanup passes preserve the primary sequence's parity.
        self.buffer_after_primary_steps()
    }

    const fn buffer_after_primary_steps(self) -> PingPong {
        if self.primary_step_count() & 1 == 0 {
            self.initial_buffer()
        } else {
            self.initial_buffer().other()
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PrimaryJfaSteps {
    next_step: u32,
    remaining: u32,
    source: PingPong,
}

impl Iterator for PrimaryJfaSteps {
    type Item = JfaPropagationStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let destination = self.source.other();
        let pass = JfaPropagationStep {
            step: self.next_step,
            source: self.source,
            destination,
        };
        self.next_step >>= 1;
        self.remaining -= 1;
        self.source = destination;
        Some(pass)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PrimaryJfaSteps {}
impl FusedIterator for PrimaryJfaSteps {}

#[derive(Debug, Clone)]
pub(super) struct CleanupJfaSteps {
    remaining: u8,
    source: PingPong,
}

impl Iterator for CleanupJfaSteps {
    type Item = JfaPropagationStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let destination = self.source.other();
        let pass = JfaPropagationStep {
            step: 1,
            source: self.source,
            destination,
        };
        self.remaining -= 1;
        self.source = destination;
        Some(pass)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CleanupJfaSteps {}
impl FusedIterator for CleanupJfaSteps {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_steps_are_exact_descending_powers_of_two() {
        let plan = JfaPropagationPlan::new(Extent2d::new(17, 5));
        assert_eq!(plan.extent(), Extent2d::new(17, 5));
        assert_eq!(
            plan.primary_steps().collect::<Vec<_>>(),
            vec![
                JfaPropagationStep {
                    step: 16,
                    source: PingPong::Ping,
                    destination: PingPong::Pong
                },
                JfaPropagationStep {
                    step: 8,
                    source: PingPong::Pong,
                    destination: PingPong::Ping
                },
                JfaPropagationStep {
                    step: 4,
                    source: PingPong::Ping,
                    destination: PingPong::Pong
                },
                JfaPropagationStep {
                    step: 2,
                    source: PingPong::Pong,
                    destination: PingPong::Ping
                },
                JfaPropagationStep {
                    step: 1,
                    source: PingPong::Ping,
                    destination: PingPong::Pong
                },
            ],
        );
    }

    #[test]
    fn cleanup_is_exactly_two_unit_steps_after_primary() {
        let plan = JfaPropagationPlan::new(Extent2d::new(17, 5));
        let mut cleanup = plan.cleanup_steps();
        assert_eq!(cleanup.len(), 2);
        assert_eq!(
            cleanup.next(),
            Some(JfaPropagationStep {
                step: 1,
                source: PingPong::Pong,
                destination: PingPong::Ping,
            }),
        );
        assert_eq!(cleanup.len(), 1);
        assert_eq!(
            cleanup.next(),
            Some(JfaPropagationStep {
                step: 1,
                source: PingPong::Ping,
                destination: PingPong::Pong,
            }),
        );
        assert_eq!(cleanup.next(), None);
        assert_eq!(cleanup.next(), None);
    }

    #[test]
    fn final_buffer_matches_primary_step_parity() {
        for (extent, expected_steps, expected_buffer) in [
            (Extent2d::new(1, 1), 0, PingPong::Ping),
            (Extent2d::new(2, 1), 1, PingPong::Pong),
            (Extent2d::new(256, 128), 8, PingPong::Ping),
            (Extent2d::new(257, 1), 9, PingPong::Pong),
        ] {
            let plan = JfaPropagationPlan::new(extent);
            assert_eq!(plan.primary_steps().len(), expected_steps);
            assert_eq!(plan.final_buffer(), expected_buffer);
            assert_eq!(
                plan.cleanup_steps().last().unwrap().destination,
                expected_buffer
            );
        }
    }
}
