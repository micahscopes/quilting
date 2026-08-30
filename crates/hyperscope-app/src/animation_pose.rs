//! Renderer-independent coordination for asynchronous animation-pose jobs.
//!
//! Animation evaluation is a platform effect, but revision allocation,
//! continuity epochs, latest-only coalescing, and stale completion decisions
//! are application semantics. This coordinator keeps at most one physical job
//! in flight and one newest pending sample without owning worker or GPU state.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationPoseStamp {
    pub clip_time_seconds: f64,
    pub sample_time_seconds: f64,
    pub revision: u32,
    pub continuity_epoch: u32,
}

impl AnimationPoseStamp {
    pub fn validate(self) -> Result<Self, AnimationPoseScheduleError> {
        if !self.clip_time_seconds.is_finite() || !self.sample_time_seconds.is_finite() {
            return Err(AnimationPoseScheduleError::NonFiniteTime);
        }
        if self.revision == 0 {
            return Err(AnimationPoseScheduleError::ZeroRevision);
        }
        if self.continuity_epoch == 0 {
            return Err(AnimationPoseScheduleError::ZeroContinuityEpoch);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationPoseRequestDisposition {
    Dispatch,
    Coalesced,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationPoseRequest {
    pub disposition: AnimationPoseRequestDisposition,
    pub stamp: AnimationPoseStamp,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationPoseCompletion {
    /// False means this completion did not name the physical in-flight job and
    /// therefore changed no scheduler state.
    pub matched_in_flight: bool,
    /// A matched job from an earlier continuity epoch is retired even though
    /// its physical worker call still had to settle.
    pub retired: bool,
    /// The newest coalesced sample to dispatch after the matched job settled.
    pub next: Option<AnimationPoseStamp>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationPoseScheduleSnapshot {
    pub continuity_epoch: u32,
    pub last_revision: u32,
    pub in_flight: Option<AnimationPoseStamp>,
    pub pending: Option<AnimationPoseStamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimationPoseScheduleError {
    NonFiniteTime,
    ZeroRevision,
    ZeroContinuityEpoch,
    ContinuityEpochExhausted,
}

impl fmt::Display for AnimationPoseScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteTime => "animation pose clocks must be finite",
            Self::ZeroRevision => "animation pose revision must be nonzero",
            Self::ZeroContinuityEpoch => "animation pose continuity epoch must be nonzero",
            Self::ContinuityEpochExhausted => "animation pose continuity epoch is exhausted",
        })
    }
}

impl Error for AnimationPoseScheduleError {}

#[derive(Debug, Clone)]
pub struct AnimationPoseScheduler {
    continuity_epoch: u32,
    last_revision: u32,
    in_flight: Option<AnimationPoseStamp>,
    pending: Option<AnimationPoseStamp>,
}

impl Default for AnimationPoseScheduler {
    fn default() -> Self {
        Self {
            continuity_epoch: 1,
            last_revision: 0,
            in_flight: None,
            pending: None,
        }
    }
}

impl AnimationPoseScheduler {
    pub fn snapshot(&self) -> AnimationPoseScheduleSnapshot {
        AnimationPoseScheduleSnapshot {
            continuity_epoch: self.continuity_epoch,
            last_revision: self.last_revision,
            in_flight: self.in_flight,
            pending: self.pending,
        }
    }

    /// Retire the current continuity epoch. A physical in-flight effect is
    /// retained until it completes, preventing overlapping worker jobs; its
    /// result cannot become current because its epoch no longer matches.
    pub fn rebase(&mut self) -> Result<u32, AnimationPoseScheduleError> {
        self.continuity_epoch = self
            .continuity_epoch
            .checked_add(1)
            .ok_or(AnimationPoseScheduleError::ContinuityEpochExhausted)?;
        self.last_revision = 0;
        self.pending = None;
        Ok(self.continuity_epoch)
    }

    pub fn request(
        &mut self,
        clip_time_seconds: f64,
        sample_time_seconds: f64,
    ) -> Result<AnimationPoseRequest, AnimationPoseScheduleError> {
        if !clip_time_seconds.is_finite() || !sample_time_seconds.is_finite() {
            return Err(AnimationPoseScheduleError::NonFiniteTime);
        }
        if self.last_revision == u32::MAX {
            self.rebase()?;
        }
        self.last_revision += 1;
        let stamp = AnimationPoseStamp {
            clip_time_seconds,
            sample_time_seconds,
            revision: self.last_revision,
            continuity_epoch: self.continuity_epoch,
        };
        let disposition = if self.in_flight.is_none() {
            self.in_flight = Some(stamp);
            AnimationPoseRequestDisposition::Dispatch
        } else {
            self.pending = Some(stamp);
            AnimationPoseRequestDisposition::Coalesced
        };
        Ok(AnimationPoseRequest { disposition, stamp })
    }

    /// Settle the exact physical in-flight job. A mismatched completion is
    /// stale and leaves both the real job and newest pending request intact.
    /// When `evaluation_available` is false, pending work is discarded rather
    /// than dispatching against resources the adapter has already retired.
    pub fn complete(
        &mut self,
        completed: AnimationPoseStamp,
        evaluation_available: bool,
    ) -> Result<AnimationPoseCompletion, AnimationPoseScheduleError> {
        let completed = completed.validate()?;
        if self.in_flight != Some(completed) {
            return Ok(AnimationPoseCompletion {
                matched_in_flight: false,
                retired: completed.continuity_epoch != self.continuity_epoch,
                next: None,
            });
        }
        self.in_flight = None;
        let retired = completed.continuity_epoch != self.continuity_epoch;
        if !evaluation_available {
            self.pending = None;
        }
        let next = self.pending.take().filter(|pending| {
            evaluation_available && pending.continuity_epoch == self.continuity_epoch
        });
        if let Some(next) = next {
            self.in_flight = Some(next);
        }
        Ok(AnimationPoseCompletion {
            matched_in_flight: true,
            retired,
            next,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(time: f64, revision: u32, epoch: u32) -> AnimationPoseStamp {
        AnimationPoseStamp {
            clip_time_seconds: time,
            sample_time_seconds: time * 10.0,
            revision,
            continuity_epoch: epoch,
        }
    }

    #[test]
    fn rapid_requests_keep_one_physical_job_and_only_the_latest_followup() {
        let mut scheduler = AnimationPoseScheduler::default();
        let first = scheduler.request(0.1, 1.0).unwrap();
        assert_eq!(first.disposition, AnimationPoseRequestDisposition::Dispatch);
        assert_eq!(first.stamp, stamp(0.1, 1, 1));
        assert_eq!(
            scheduler.request(0.2, 2.0).unwrap().disposition,
            AnimationPoseRequestDisposition::Coalesced,
        );
        let latest = scheduler.request(0.3, 3.0).unwrap();
        assert_eq!(
            latest.disposition,
            AnimationPoseRequestDisposition::Coalesced
        );
        assert_eq!(scheduler.snapshot().pending, Some(stamp(0.3, 3, 1)));

        let completed = scheduler.complete(first.stamp, true).unwrap();
        assert!(completed.matched_in_flight);
        assert!(!completed.retired);
        assert_eq!(completed.next, Some(latest.stamp));
        assert_eq!(scheduler.snapshot().in_flight, Some(latest.stamp));
        assert_eq!(scheduler.snapshot().pending, None);
    }

    #[test]
    fn stale_completion_cannot_release_or_replace_the_real_job() {
        let mut scheduler = AnimationPoseScheduler::default();
        let first = scheduler.request(0.1, 1.0).unwrap();
        let latest = scheduler.request(0.2, 2.0).unwrap();
        let stale = scheduler.complete(stamp(9.0, 99, 1), true).unwrap();
        assert!(!stale.matched_in_flight);
        assert_eq!(scheduler.snapshot().in_flight, Some(first.stamp));
        assert_eq!(scheduler.snapshot().pending, Some(latest.stamp));
    }

    #[test]
    fn rebase_retires_inflight_work_without_starting_a_second_worker() {
        let mut scheduler = AnimationPoseScheduler::default();
        let old = scheduler.request(0.1, 1.0).unwrap().stamp;
        assert_eq!(scheduler.rebase().unwrap(), 2);
        let current = scheduler.request(0.5, 5.0).unwrap();
        assert_eq!(
            current.disposition,
            AnimationPoseRequestDisposition::Coalesced
        );
        assert_eq!(current.stamp, stamp(0.5, 1, 2));

        let completed = scheduler.complete(old, true).unwrap();
        assert!(completed.matched_in_flight);
        assert!(completed.retired);
        assert_eq!(completed.next, Some(current.stamp));
    }

    #[test]
    fn unavailable_evaluator_drops_coalesced_work_atomically() {
        let mut scheduler = AnimationPoseScheduler::default();
        let first = scheduler.request(0.1, 1.0).unwrap().stamp;
        scheduler.request(0.2, 2.0).unwrap();
        let completed = scheduler.complete(first, false).unwrap();
        assert!(completed.matched_in_flight);
        assert_eq!(completed.next, None);
        assert_eq!(scheduler.snapshot().in_flight, None);
        assert_eq!(scheduler.snapshot().pending, None);
    }

    #[test]
    fn invalid_times_and_stamps_are_rejected_without_state_changes() {
        let mut scheduler = AnimationPoseScheduler::default();
        let before = scheduler.snapshot();
        assert_eq!(
            scheduler.request(f64::NAN, 0.0),
            Err(AnimationPoseScheduleError::NonFiniteTime),
        );
        assert_eq!(scheduler.snapshot(), before);
        assert_eq!(
            scheduler.complete(stamp(0.0, 0, 1), true),
            Err(AnimationPoseScheduleError::ZeroRevision),
        );
        assert_eq!(scheduler.snapshot(), before);
    }
}
