//! Bounded, restart-aware delivery memory for the optional local peer relay.
//!
//! The relay deliberately stores opaque JSON. It assigns delivery cursors but
//! never validates application commands, allocates projection revisions, or
//! interprets arrival order as authored causality.

use serde::Serialize;
use serde_json::value::RawValue;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

pub const DEFAULT_RELAY_CAPACITY: usize = 4_096;
pub const DEFAULT_POLL_LIMIT: usize = 256;
pub const MAX_POLL_LIMIT: usize = 1_024;
pub const MAX_FRAME_BYTES: usize = 256 * 1_024;

#[derive(Debug, Clone)]
struct StoredFrame {
    cursor: u64,
    frame: Box<RawValue>,
}

/// In-memory delivery state for one process generation.
#[derive(Debug)]
pub struct LocalPeerRelay {
    generation: String,
    capacity: usize,
    latest_cursor: u64,
    frames: VecDeque<StoredFrame>,
}

impl LocalPeerRelay {
    pub fn new(generation: impl Into<String>, capacity: usize) -> Result<Self, RelayError> {
        let generation = generation.into();
        if generation.is_empty()
            || generation.len() > 128
            || !generation
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(RelayError::InvalidGeneration);
        }
        if capacity == 0 {
            return Err(RelayError::InvalidCapacity);
        }
        Ok(Self {
            generation,
            capacity,
            latest_cursor: 0,
            frames: VecDeque::with_capacity(capacity.min(1_024)),
        })
    }

    pub fn generation(&self) -> &str {
        &self.generation
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn latest_cursor(&self) -> u64 {
        self.latest_cursor
    }

    /// Retain one opaque JSON frame and return its transport-only cursor.
    pub fn append_json(&mut self, frame_json: &str) -> Result<u64, RelayError> {
        if frame_json.len() > MAX_FRAME_BYTES {
            return Err(RelayError::FrameTooLarge {
                bytes: frame_json.len(),
                maximum: MAX_FRAME_BYTES,
            });
        }
        let frame = RawValue::from_string(frame_json.to_owned())
            .map_err(|error| RelayError::InvalidJson(error.to_string()))?;
        let cursor = self
            .latest_cursor
            .checked_add(1)
            .ok_or(RelayError::CursorOverflow)?;
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(StoredFrame { cursor, frame });
        self.latest_cursor = cursor;
        Ok(cursor)
    }

    /// Return retained frames after `after`. A gap never fabricates repair:
    /// clients receive the retained suffix and must surface degraded status or
    /// ask the future HHHS lane to reconcile durable state.
    pub fn poll(&self, after: u64, limit: usize) -> Result<RelayBatch, RelayError> {
        if limit == 0 || limit > MAX_POLL_LIMIT {
            return Err(RelayError::InvalidPollLimit {
                limit,
                maximum: MAX_POLL_LIMIT,
            });
        }
        let oldest_cursor = self.frames.front().map(|frame| frame.cursor);
        let gap = after > self.latest_cursor
            || oldest_cursor.is_some_and(|oldest| after < oldest.saturating_sub(1));
        let resume_after =
            oldest_cursor.map_or(self.latest_cursor, |oldest| oldest.saturating_sub(1));
        let mut matching = self.frames.iter().filter(|frame| frame.cursor > after);
        let frames = matching
            .by_ref()
            .take(limit)
            .map(|frame| RelayDelivery {
                cursor: frame.cursor.to_string(),
                frame: frame.frame.clone(),
            })
            .collect();
        let has_more = matching.next().is_some();
        Ok(RelayBatch {
            generation: self.generation.clone(),
            requested_after: after.to_string(),
            resume_after: resume_after.to_string(),
            oldest_cursor: oldest_cursor.map(|cursor| cursor.to_string()),
            latest_cursor: self.latest_cursor.to_string(),
            gap,
            has_more,
            frames,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelayBatch {
    pub generation: String,
    pub requested_after: String,
    pub resume_after: String,
    pub oldest_cursor: Option<String>,
    pub latest_cursor: String,
    pub gap: bool,
    pub has_more: bool,
    pub frames: Vec<RelayDelivery>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RelayDelivery {
    pub cursor: String,
    pub frame: Box<RawValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayError {
    InvalidGeneration,
    InvalidCapacity,
    InvalidPollLimit { limit: usize, maximum: usize },
    FrameTooLarge { bytes: usize, maximum: usize },
    InvalidJson(String),
    CursorOverflow,
}

impl fmt::Display for RelayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeneration => {
                formatter.write_str("relay generation must be 1..128 URL-safe ASCII characters")
            }
            Self::InvalidCapacity => formatter.write_str("relay capacity must be positive"),
            Self::InvalidPollLimit { limit, maximum } => {
                write!(
                    formatter,
                    "relay poll limit {limit} is outside 1..={maximum}"
                )
            }
            Self::FrameTooLarge { bytes, maximum } => {
                write!(
                    formatter,
                    "relay frame has {bytes} bytes; maximum is {maximum}"
                )
            }
            Self::InvalidJson(message) => {
                write!(formatter, "relay frame is invalid JSON: {message}")
            }
            Self::CursorOverflow => formatter.write_str("relay delivery cursor overflow"),
        }
    }
}

impl Error for RelayError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_delivery_reports_gaps_without_claiming_repair() {
        let mut relay = LocalPeerRelay::new("generation-a", 2).unwrap();
        for value in 1..=3 {
            relay
                .append_json(&format!(r#"{{"opaque":{value}}}"#))
                .unwrap();
        }
        let batch = relay.poll(0, DEFAULT_POLL_LIMIT).unwrap();
        assert!(batch.gap);
        assert_eq!(batch.resume_after, "1");
        assert_eq!(batch.oldest_cursor.as_deref(), Some("2"));
        assert_eq!(batch.latest_cursor, "3");
        assert_eq!(
            batch
                .frames
                .iter()
                .map(|delivery| delivery.cursor.as_str())
                .collect::<Vec<_>>(),
            ["2", "3"],
        );
    }

    #[test]
    fn opaque_frames_preserve_future_fields_and_exact_integers() {
        let mut relay = LocalPeerRelay::new("generation-a", 4).unwrap();
        let source = r#"{"lane":"future","opaque":18446744073709551615}"#;
        relay.append_json(source).unwrap();
        let batch = relay.poll(0, 1).unwrap();
        assert_eq!(batch.frames[0].frame.get(), source);
        assert!(serde_json::to_string(&batch)
            .unwrap()
            .contains(r#""opaque":18446744073709551615"#));
    }

    #[test]
    fn polling_is_paginated_and_restart_aware() {
        let mut relay = LocalPeerRelay::new("generation-a", 4).unwrap();
        relay.append_json("null").unwrap();
        relay.append_json("true").unwrap();
        let first = relay.poll(0, 1).unwrap();
        assert!(!first.gap);
        assert!(first.has_more);
        assert_eq!(first.frames[0].cursor, "1");
        let second = relay.poll(1, 1).unwrap();
        assert!(!second.has_more);
        assert_eq!(second.frames[0].cursor, "2");

        let restarted = LocalPeerRelay::new("generation-b", 4).unwrap();
        let stale_cursor = restarted.poll(2, 1).unwrap();
        assert!(stale_cursor.gap);
        assert_eq!(stale_cursor.generation, "generation-b");
        assert_eq!(stale_cursor.resume_after, "0");
    }

    #[test]
    fn invalid_inputs_leave_delivery_state_unchanged() {
        assert_eq!(
            LocalPeerRelay::new("bad generation", 1).unwrap_err(),
            RelayError::InvalidGeneration,
        );
        assert_eq!(
            LocalPeerRelay::new("valid", 0).unwrap_err(),
            RelayError::InvalidCapacity,
        );
        let mut relay = LocalPeerRelay::new("valid", 1).unwrap();
        assert!(matches!(
            relay.append_json("{"),
            Err(RelayError::InvalidJson(_)),
        ));
        assert!(matches!(
            relay.poll(0, 0),
            Err(RelayError::InvalidPollLimit { .. }),
        ));
        assert_eq!(relay.latest_cursor(), 0);
        assert!(relay.is_empty());
    }

    #[test]
    fn cursor_overflow_is_atomic() {
        let mut relay = LocalPeerRelay::new("valid", 1).unwrap();
        relay.latest_cursor = u64::MAX;
        assert_eq!(relay.append_json("null"), Err(RelayError::CursorOverflow));
        assert!(relay.is_empty());
    }
}
