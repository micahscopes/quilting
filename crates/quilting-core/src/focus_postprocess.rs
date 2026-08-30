//! Backend-neutral scheduling for focus-aware image composition.
//!
//! WebGL2 and WebGPU own different texture and pipeline handles, but they must
//! not independently reinterpret the focus policy. This module freezes the
//! pass topology, scratch-buffer parity, and blur normalization without
//! depending on either graphics API.

use crate::render::{FocusPostprocessMode, FocusPostprocessPacket, RenderContractError};
use core::iter::FusedIterator;

pub const FOCUS_JFA_DOWNSAMPLE: u32 = 2;
pub const FOCUS_JFA_CLEANUP_STEPS: u8 = 2;
pub const FOCUS_BLUR_DIRECTIONS: [[f32; 2]; 3] = [[1.0, 0.0], [0.5, 0.866], [-0.5, 0.866]];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusExtent2d {
    pub width: u32,
    pub height: u32,
}

impl FocusExtent2d {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    const fn primary_step_count(self) -> u32 {
        let maximum = if self.width > self.height {
            self.width
        } else {
            self.height
        };
        if maximum <= 1 {
            0
        } else {
            u32::BITS - (maximum - 1).leading_zeros()
        }
    }

    fn downsample(self, divisor: u32) -> Self {
        let divisor = divisor.max(1);
        Self {
            width: (self.width / divisor).max(1),
            height: (self.height / divisor).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPingPong {
    Ping,
    Pong,
}

impl FocusPingPong {
    const fn other(self) -> Self {
        match self {
            Self::Ping => Self::Pong,
            Self::Pong => Self::Ping,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusJfaPropagationStep {
    pub step: u32,
    pub source: FocusPingPong,
    pub destination: FocusPingPong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusJfaPropagationPlan {
    extent: FocusExtent2d,
}

impl FocusJfaPropagationPlan {
    pub const fn new(extent: FocusExtent2d) -> Self {
        Self { extent }
    }

    pub const fn extent(self) -> FocusExtent2d {
        self.extent
    }

    pub const fn initial_buffer(self) -> FocusPingPong {
        FocusPingPong::Ping
    }

    pub const fn primary_step_count(self) -> u32 {
        self.extent.primary_step_count()
    }

    pub fn primary_steps(self) -> PrimaryFocusJfaSteps {
        let remaining = self.primary_step_count();
        let next_step = if remaining == 0 {
            0
        } else {
            1 << (remaining - 1)
        };
        PrimaryFocusJfaSteps {
            next_step,
            remaining,
            source: self.initial_buffer(),
        }
    }

    pub fn cleanup_steps(self) -> CleanupFocusJfaSteps {
        CleanupFocusJfaSteps {
            remaining: FOCUS_JFA_CLEANUP_STEPS,
            source: self.buffer_after_primary_steps(),
        }
    }

    pub const fn final_buffer(self) -> FocusPingPong {
        // Two cleanup passes preserve the primary sequence's parity.
        self.buffer_after_primary_steps()
    }

    const fn buffer_after_primary_steps(self) -> FocusPingPong {
        if self.primary_step_count() & 1 == 0 {
            self.initial_buffer()
        } else {
            self.initial_buffer().other()
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrimaryFocusJfaSteps {
    next_step: u32,
    remaining: u32,
    source: FocusPingPong,
}

impl Iterator for PrimaryFocusJfaSteps {
    type Item = FocusJfaPropagationStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let destination = self.source.other();
        let pass = FocusJfaPropagationStep {
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

impl ExactSizeIterator for PrimaryFocusJfaSteps {}
impl FusedIterator for PrimaryFocusJfaSteps {}

#[derive(Debug, Clone)]
pub struct CleanupFocusJfaSteps {
    remaining: u8,
    source: FocusPingPong,
}

impl Iterator for CleanupFocusJfaSteps {
    type Item = FocusJfaPropagationStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let destination = self.source.other();
        let pass = FocusJfaPropagationStep {
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

impl ExactSizeIterator for CleanupFocusJfaSteps {}
impl FusedIterator for CleanupFocusJfaSteps {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusBlurSurface {
    Scene,
    Ping,
    Pong,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusDirectionalBlurPass {
    pub gaussian_pass: u8,
    pub direction_index: u8,
    pub direction: [f32; 2],
    pub source: FocusBlurSurface,
    pub destination: FocusBlurSurface,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
pub struct FocusDirectionalBlurPasses {
    next: u16,
    total: u16,
}

impl Iterator for FocusDirectionalBlurPasses {
    type Item = FocusDirectionalBlurPass;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.total {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let is_final = self.next == self.total;
        let source = match index {
            0 => FocusBlurSurface::Scene,
            value if value & 1 == 1 => FocusBlurSurface::Ping,
            _ => FocusBlurSurface::Pong,
        };
        let destination = if is_final {
            FocusBlurSurface::Output
        } else if index & 1 == 0 {
            FocusBlurSurface::Ping
        } else {
            FocusBlurSurface::Pong
        };
        let direction_index = (index % FOCUS_BLUR_DIRECTIONS.len() as u16) as u8;
        Some(FocusDirectionalBlurPass {
            gaussian_pass: (index / FOCUS_BLUR_DIRECTIONS.len() as u16) as u8,
            direction_index,
            direction: FOCUS_BLUR_DIRECTIONS[direction_index as usize],
            source,
            destination,
            is_final,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::from(self.total - self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FocusDirectionalBlurPasses {}
impl FusedIterator for FocusDirectionalBlurPasses {}

/// Immutable schedule shared by WebGL2 and WebGPU focus pipelines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusPostprocessSchedule {
    full_extent: FocusExtent2d,
    jfa_extent: FocusExtent2d,
    packet: FocusPostprocessPacket,
}

impl FocusPostprocessSchedule {
    pub fn build(
        viewport: [u32; 2],
        packet: FocusPostprocessPacket,
    ) -> Result<Self, RenderContractError> {
        packet.validate()?;
        if viewport[0] == 0 || viewport[1] == 0 {
            return Err(RenderContractError::InvalidView);
        }
        let full_extent = FocusExtent2d::new(viewport[0], viewport[1]);
        Ok(Self {
            full_extent,
            jfa_extent: full_extent.downsample(FOCUS_JFA_DOWNSAMPLE),
            packet,
        })
    }

    pub const fn full_extent(self) -> FocusExtent2d {
        self.full_extent
    }

    pub const fn jfa_extent(self) -> FocusExtent2d {
        self.jfa_extent
    }

    pub const fn packet(self) -> FocusPostprocessPacket {
        self.packet
    }

    pub const fn uses_jfa(self) -> bool {
        !matches!(self.packet.mode, FocusPostprocessMode::Spheroidal)
    }

    pub fn jfa_plan(self) -> Option<FocusJfaPropagationPlan> {
        self.uses_jfa()
            .then_some(FocusJfaPropagationPlan::new(self.jfa_extent))
    }

    pub fn directional_blur_passes(self) -> FocusDirectionalBlurPasses {
        FocusDirectionalBlurPasses {
            next: 0,
            total: u16::from(self.packet.gaussian_passes) * FOCUS_BLUR_DIRECTIONS.len() as u16,
        }
    }

    pub fn per_subpass_blur_strength(self) -> f32 {
        focus_per_subpass_blur_strength(
            self.packet.blur_strength,
            u32::from(self.packet.gaussian_passes),
        )
    }

    pub fn kawase_offset(self, pass: u8) -> Option<f32> {
        (pass < self.packet.kawase_passes)
            .then_some(self.packet.kawase_offset * (f32::from(pass) + 1.0))
    }
}

/// Normalize each of the three directional passes so adding quality passes
/// does not also increase the requested total blur variance.
pub fn focus_per_subpass_blur_strength(blur_strength: f32, gaussian_passes: u32) -> f32 {
    let pass_count = gaussian_passes.max(1);
    blur_strength / (pass_count as f32 * FOCUS_BLUR_DIRECTIONS.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(mode: FocusPostprocessMode) -> FocusPostprocessPacket {
        FocusPostprocessPacket {
            mode,
            blur_radius_pixels: 11,
            blur_strength: 3.0,
            focus_coordinate: 0.62,
            bandwidth: 0.1,
            normalize_range: false,
            stretch_range: [0.5, 0.5],
            gaussian_passes: 2,
            kawase_passes: 3,
            kawase_offset: 1.5,
        }
    }

    #[test]
    fn primary_steps_are_exact_descending_powers_of_two() {
        let plan = FocusJfaPropagationPlan::new(FocusExtent2d::new(17, 5));
        assert_eq!(plan.extent(), FocusExtent2d::new(17, 5));
        assert_eq!(
            plan.primary_steps().collect::<Vec<_>>(),
            vec![
                FocusJfaPropagationStep {
                    step: 16,
                    source: FocusPingPong::Ping,
                    destination: FocusPingPong::Pong,
                },
                FocusJfaPropagationStep {
                    step: 8,
                    source: FocusPingPong::Pong,
                    destination: FocusPingPong::Ping,
                },
                FocusJfaPropagationStep {
                    step: 4,
                    source: FocusPingPong::Ping,
                    destination: FocusPingPong::Pong,
                },
                FocusJfaPropagationStep {
                    step: 2,
                    source: FocusPingPong::Pong,
                    destination: FocusPingPong::Ping,
                },
                FocusJfaPropagationStep {
                    step: 1,
                    source: FocusPingPong::Ping,
                    destination: FocusPingPong::Pong,
                },
            ],
        );
    }

    #[test]
    fn cleanup_preserves_primary_parity_even_for_one_pixel_fields() {
        for (extent, expected_steps, expected_buffer) in [
            (FocusExtent2d::new(1, 1), 0, FocusPingPong::Ping),
            (FocusExtent2d::new(2, 1), 1, FocusPingPong::Pong),
            (FocusExtent2d::new(256, 128), 8, FocusPingPong::Ping),
            (FocusExtent2d::new(257, 1), 9, FocusPingPong::Pong),
        ] {
            let plan = FocusJfaPropagationPlan::new(extent);
            assert_eq!(plan.primary_steps().len(), expected_steps);
            let cleanup = plan.cleanup_steps().collect::<Vec<_>>();
            assert_eq!(cleanup.len(), usize::from(FOCUS_JFA_CLEANUP_STEPS));
            assert_eq!(cleanup.last().unwrap().destination, expected_buffer);
            assert_eq!(plan.final_buffer(), expected_buffer);
        }
    }

    #[test]
    fn schedule_freezes_downsample_bypass_kawase_and_blur_ping_pong() {
        let schedule =
            FocusPostprocessSchedule::build([1919, 1079], packet(FocusPostprocessMode::Spheroidal))
                .unwrap();
        assert_eq!(schedule.full_extent(), FocusExtent2d::new(1919, 1079));
        assert_eq!(schedule.jfa_extent(), FocusExtent2d::new(959, 539));
        assert!(!schedule.uses_jfa());
        assert!(schedule.jfa_plan().is_none());
        assert_eq!(schedule.kawase_offset(0), Some(1.5));
        assert_eq!(schedule.kawase_offset(2), Some(4.5));
        assert_eq!(schedule.kawase_offset(3), None);

        let passes = schedule.directional_blur_passes().collect::<Vec<_>>();
        assert_eq!(passes.len(), 6);
        assert_eq!(passes[0].source, FocusBlurSurface::Scene);
        assert_eq!(passes[0].destination, FocusBlurSurface::Ping);
        assert_eq!(passes[1].source, FocusBlurSurface::Ping);
        assert_eq!(passes[1].destination, FocusBlurSurface::Pong);
        assert_eq!(passes[4].source, FocusBlurSurface::Pong);
        assert_eq!(passes[4].destination, FocusBlurSurface::Ping);
        assert_eq!(passes[5].source, FocusBlurSurface::Ping);
        assert_eq!(passes[5].destination, FocusBlurSurface::Output);
        assert!(passes[5].is_final);
        assert_eq!(passes[5].gaussian_pass, 1);
        assert_eq!(passes[5].direction_index, 2);
        assert!((schedule.per_subpass_blur_strength() - (3.0 / 6.0_f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn non_spheroidal_schedule_uses_the_shared_jfa_plan() {
        let schedule = FocusPostprocessSchedule::build(
            [1920, 1080],
            packet(FocusPostprocessMode::ConformalStretch),
        )
        .unwrap();
        let plan = schedule.jfa_plan().unwrap();
        assert_eq!(plan.extent(), FocusExtent2d::new(960, 540));
        assert_eq!(plan.primary_steps().len(), 10);
        assert_eq!(plan.cleanup_steps().len(), 2);
    }

    #[test]
    fn schedule_rejects_zero_extent_or_invalid_policy() {
        assert_eq!(
            FocusPostprocessSchedule::build(
                [0, 1080],
                packet(FocusPostprocessMode::ConformalStretch),
            ),
            Err(RenderContractError::InvalidView),
        );
        let mut invalid = packet(FocusPostprocessMode::ConformalStretch);
        invalid.blur_strength = f32::NAN;
        assert_eq!(
            FocusPostprocessSchedule::build([1920, 1080], invalid),
            Err(RenderContractError::InvalidFocusPostprocess),
        );
    }
}
