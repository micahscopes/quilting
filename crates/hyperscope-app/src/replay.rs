//! Versioned, adapter-independent application replay traces.
//!
//! A trace records semantic inputs, reducer outcomes, and compact committed
//! state. It deliberately excludes DOM events, device reports, renderer
//! resources, and wall clocks so native tools, browsers, Blender adapters, and
//! future render backends can consume the same oracle.

use crate::{
    AppCommit, AppEffect, AppEvent, AppStore, CommitDisposition, FrameTick,
    NavigationSynchronization, PresentationAction, SemanticAction, Timed,
};
use hyperscape::{AuthoredCamera, AuthoredFocus, Presentation, SphereReflectionState};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

pub const APP_REPLAY_VERSION: &str = "hyperscope-app-replay/0.1";
pub const APP_REPLAY_FINGERPRINT_ALGORITHM: &str = "fnv1a-128-json";
const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayScript {
    pub version: String,
    pub events: Vec<AppReplayEvent>,
}

impl AppReplayScript {
    pub fn new(events: Vec<AppReplayEvent>) -> Self {
        Self {
            version: APP_REPLAY_VERSION.to_owned(),
            events,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppReplayEvent {
    LoadPresentation {
        presentation: Presentation,
    },
    SynchronizeNavigation {
        camera: AuthoredCamera,
        focus: AuthoredFocus,
    },
    Present {
        sequence: u64,
        at_seconds: f64,
        action: ReplayPresentationAction,
    },
    Frame {
        elapsed_seconds: f64,
        delta_seconds: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayPresentationAction {
    Start,
    Advance,
    Reverse,
    JumpToCue { cue: Uuid },
    Clear,
}

impl From<ReplayPresentationAction> for PresentationAction {
    fn from(action: ReplayPresentationAction) -> Self {
        match action {
            ReplayPresentationAction::Start => Self::Start,
            ReplayPresentationAction::Advance => Self::Advance,
            ReplayPresentationAction::Reverse => Self::Reverse,
            ReplayPresentationAction::JumpToCue { cue } => Self::JumpToCue(cue),
            ReplayPresentationAction::Clear => Self::Clear,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayTrace {
    pub version: String,
    pub records: Vec<AppReplayRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayRecord {
    pub ordinal: usize,
    pub event: AppReplayEvent,
    pub outcome: AppReplayOutcome,
    pub state: AppReplayState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AppReplayOutcome {
    Committed {
        revision: u64,
        disposition: ReplayCommitDisposition,
        published_ui: bool,
        effects: Vec<AppReplayEffect>,
    },
    Rejected {
        error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCommitDisposition {
    Applied,
    IgnoredStale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppReplayEffect {
    FetchAsset {
        request_id: String,
        asset_id: String,
        uri: String,
    },
    CancelAssetLoad {
        request_id: String,
        asset_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayState {
    pub revision: u64,
    pub elapsed_seconds: f64,
    pub active_cue: Option<Uuid>,
    pub active_scene: Option<Uuid>,
    pub active_view: Option<Uuid>,
    pub reflection: ReplayReflection,
    pub camera: AppReplayCameraState,
    pub focus: AppReplayFocusState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayReflection {
    Identity,
    SphereReflection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayCameraState {
    pub eye: [f64; 3],
    pub orientation_wxyz: [f64; 4],
    pub control_distance: f64,
    pub semantic_target: Option<[f64; 3]>,
    pub vertical_fov_radians: f64,
    pub near: f64,
    pub far: f64,
    pub transition_remaining_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayFocusState {
    pub center: [f64; 3],
    pub radius: f64,
    pub anchored: bool,
    pub focus_enabled: bool,
    pub inversion_enabled: bool,
    pub coordinate: f64,
    pub angular_aperture: f64,
    pub transition_remaining_seconds: Option<f64>,
}

pub fn run_app_replay(script: &AppReplayScript) -> Result<AppReplayTrace, AppReplayError> {
    if script.version != APP_REPLAY_VERSION {
        return Err(AppReplayError::UnsupportedVersion(script.version.clone()));
    }
    let store = AppStore::default();
    let mut records = Vec::with_capacity(script.events.len());
    for (ordinal, event) in script.events.iter().cloned().enumerate() {
        let outcome = replay_event(&store, &event);
        records.push(AppReplayRecord {
            ordinal,
            event,
            outcome,
            state: replay_state(&store),
        });
    }
    Ok(AppReplayTrace {
        version: APP_REPLAY_VERSION.to_owned(),
        records,
    })
}

/// Builds a deterministic successful walkthrough of every cue in a
/// presentation. The initial navigation synchronization makes the starting
/// state explicit instead of depending on a platform adapter's defaults.
pub fn presentation_walkthrough_replay(presentation: Presentation) -> AppReplayScript {
    let cues = presentation
        .cues
        .iter()
        .map(|cue| (cue.id, cue.transition.duration_seconds))
        .collect::<Vec<_>>();
    let mut events = vec![
        AppReplayEvent::LoadPresentation { presentation },
        AppReplayEvent::SynchronizeNavigation {
            camera: AuthoredCamera::default(),
            focus: AuthoredFocus::default(),
        },
    ];
    let Some((_, first_duration)) = cues.first().copied() else {
        return AppReplayScript::new(events);
    };

    let mut elapsed_seconds = 0.0;
    events.push(AppReplayEvent::Present {
        sequence: 1,
        at_seconds: elapsed_seconds,
        action: ReplayPresentationAction::Start,
    });
    elapsed_seconds += first_duration;
    events.push(AppReplayEvent::Frame {
        elapsed_seconds,
        delta_seconds: first_duration,
    });

    for (sequence, (cue, duration)) in (2_u64..).zip(cues.into_iter().skip(1)) {
        events.push(AppReplayEvent::Present {
            sequence,
            at_seconds: elapsed_seconds,
            action: ReplayPresentationAction::JumpToCue { cue },
        });
        elapsed_seconds += duration;
        events.push(AppReplayEvent::Frame {
            elapsed_seconds,
            delta_seconds: duration,
        });
    }
    AppReplayScript::new(events)
}

pub fn app_replay_fingerprint(trace: &AppReplayTrace) -> Result<String, serde_json::Error> {
    let encoded = serde_json::to_vec(trace)?;
    let mut fingerprint = FNV1A_128_OFFSET;
    for byte in encoded {
        fingerprint ^= u128::from(byte);
        fingerprint = fingerprint.wrapping_mul(FNV1A_128_PRIME);
    }
    Ok(format!("{fingerprint:032x}"))
}

fn replay_event(store: &AppStore, event: &AppReplayEvent) -> AppReplayOutcome {
    let result: Result<AppCommit, String> = match event {
        AppReplayEvent::LoadPresentation { presentation } => store
            .dispatch(AppEvent::PresentationLoaded(presentation.clone()))
            .map_err(|error| error.to_string()),
        AppReplayEvent::SynchronizeNavigation { camera, focus } => camera
            .to_camera_rig()
            .and_then(|camera| {
                focus
                    .to_focus_navigation()
                    .map(|focus| NavigationSynchronization { camera, focus })
            })
            .map_err(|error| error.to_string())
            .and_then(|synchronization| {
                store
                    .dispatch(AppEvent::NavigationSynchronized(synchronization))
                    .map_err(|error| error.to_string())
            }),
        AppReplayEvent::Present {
            sequence,
            at_seconds,
            action,
        } => store
            .dispatch(AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::Present((*action).into()),
            }))
            .map_err(|error| error.to_string()),
        AppReplayEvent::Frame {
            elapsed_seconds,
            delta_seconds,
        } => store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: *elapsed_seconds,
                delta_seconds: *delta_seconds,
            }))
            .map_err(|error| error.to_string()),
    };
    match result {
        Ok(commit) => AppReplayOutcome::Committed {
            revision: commit.revision,
            disposition: match commit.disposition {
                CommitDisposition::Applied => ReplayCommitDisposition::Applied,
                CommitDisposition::IgnoredStale => ReplayCommitDisposition::IgnoredStale,
            },
            published_ui: commit.published_ui,
            effects: commit.effects.iter().map(replay_effect).collect(),
        },
        Err(error) => AppReplayOutcome::Rejected { error },
    }
}

fn replay_effect(effect: &AppEffect) -> AppReplayEffect {
    match effect {
        AppEffect::FetchAsset { request_id, asset } => AppReplayEffect::FetchAsset {
            request_id: request_id.to_string(),
            asset_id: asset.id.to_string(),
            uri: asset.uri.clone(),
        },
        AppEffect::CancelAssetLoad {
            request_id,
            asset_id,
        } => AppReplayEffect::CancelAssetLoad {
            request_id: request_id.to_string(),
            asset_id: asset_id.to_string(),
        },
    }
}

fn replay_state(store: &AppStore) -> AppReplayState {
    let frame = store.frame_snapshot();
    let presentation = store.presentation_snapshot();
    let active = presentation
        .as_ref()
        .and_then(|presentation| presentation.active.as_ref());
    let focus_transition_remaining = frame
        .focus
        .transition
        .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
    AppReplayState {
        revision: frame.revision,
        elapsed_seconds: frame.elapsed_seconds,
        active_cue: active.map(|snapshot| snapshot.cue_id),
        active_scene: active.map(|snapshot| snapshot.scene_id),
        active_view: active.map(|snapshot| snapshot.view_id),
        reflection: match frame.reflection {
            SphereReflectionState::Identity => ReplayReflection::Identity,
            SphereReflectionState::Sphere(_) => ReplayReflection::SphereReflection,
        },
        camera: AppReplayCameraState {
            eye: frame.camera.eye,
            orientation_wxyz: [
                frame.camera.orientation.w,
                frame.camera.orientation.x,
                frame.camera.orientation.y,
                frame.camera.orientation.z,
            ],
            control_distance: frame.camera.control_distance,
            semantic_target: frame.camera.semantic_target,
            vertical_fov_radians: frame.camera.lens.vertical_fov_radians,
            near: frame.camera.lens.near,
            far: frame.camera.lens.far,
            transition_remaining_seconds: frame.camera_transition_remaining,
        },
        focus: AppReplayFocusState {
            center: frame.focus.sphere.center,
            radius: frame.focus.sphere.radius,
            anchored: frame.focus.anchor.is_some(),
            focus_enabled: frame.focus.focus_enabled,
            inversion_enabled: frame.focus.inversion_enabled,
            coordinate: frame.focus.focus_coordinate,
            angular_aperture: frame.focus.angular_aperture,
            transition_remaining_seconds: focus_transition_remaining,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppReplayError {
    UnsupportedVersion(String),
}

impl fmt::Display for AppReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported app replay version {version:?}")
            }
        }
    }
}

impl Error for AppReplayError {}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../examples/hacker-night.presentation.json");
    const GOLDEN: &str = include_str!("../../../examples/hacker-night.replay.fingerprint");
    const UNKNOWN_CUE: &str = "f0000000-0000-4000-8000-000000000099";

    fn fixture() -> Presentation {
        Presentation::from_json(FIXTURE).unwrap()
    }

    fn assert_semantic_state_eq(left: &AppReplayState, right: &AppReplayState) {
        assert_eq!(left.active_cue, right.active_cue);
        assert_eq!(left.active_scene, right.active_scene);
        assert_eq!(left.active_view, right.active_view);
        assert_eq!(left.reflection, right.reflection);
        assert_eq!(left.camera, right.camera);
        assert_eq!(left.focus, right.focus);
    }

    #[test]
    fn walkthrough_trace_is_deterministic_and_json_roundtrips() {
        let script = presentation_walkthrough_replay(fixture());
        let encoded_script = serde_json::to_string(&script).unwrap();
        let decoded_script: AppReplayScript = serde_json::from_str(&encoded_script).unwrap();
        assert_eq!(decoded_script, script);

        let first = run_app_replay(&script).unwrap();
        let second = run_app_replay(&decoded_script).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.records.len(), 14);
        assert!(first.records.iter().all(|record| matches!(
            record.outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::Applied,
                ..
            }
        )));
        assert_eq!(
            first.records.last().unwrap().state.active_cue,
            fixture().cues.last().map(|cue| cue.id)
        );

        let encoded_trace = serde_json::to_string(&first).unwrap();
        let decoded_trace: AppReplayTrace = serde_json::from_str(&encoded_trace).unwrap();
        assert_eq!(decoded_trace, first);
        assert_eq!(
            app_replay_fingerprint(&first).unwrap(),
            app_replay_fingerprint(&second).unwrap()
        );
        assert_eq!(
            format!(
                "{APP_REPLAY_FINGERPRINT_ALGORITHM}:{}",
                app_replay_fingerprint(&first).unwrap()
            ),
            GOLDEN.trim()
        );
    }

    #[test]
    fn rejected_event_is_recorded_without_mutating_committed_state() {
        let presentation = fixture();
        let mut script = AppReplayScript::new(vec![
            AppReplayEvent::LoadPresentation { presentation },
            AppReplayEvent::Present {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayPresentationAction::Start,
            },
        ]);
        script.events.push(AppReplayEvent::Present {
            sequence: 2,
            at_seconds: 0.0,
            action: ReplayPresentationAction::JumpToCue {
                cue: Uuid::parse_str(UNKNOWN_CUE).unwrap(),
            },
        });

        let trace = run_app_replay(&script).unwrap();
        assert!(matches!(
            trace.records.last().unwrap().outcome,
            AppReplayOutcome::Rejected { .. }
        ));
        let before = &trace.records[trace.records.len() - 2].state;
        let after = &trace.records.last().unwrap().state;
        assert_eq!(after.revision, before.revision);
        assert_semantic_state_eq(after, before);
    }

    #[test]
    fn completed_transition_is_cadence_independent_in_replay() {
        let presentation = fixture();
        let destination = presentation.cues[5].id;
        let prefix = vec![
            AppReplayEvent::LoadPresentation { presentation },
            AppReplayEvent::Present {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayPresentationAction::Start,
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.7,
                delta_seconds: 0.7,
            },
            AppReplayEvent::Present {
                sequence: 2,
                at_seconds: 0.7,
                action: ReplayPresentationAction::JumpToCue { cue: destination },
            },
        ];
        let mut single = prefix.clone();
        single.push(AppReplayEvent::Frame {
            elapsed_seconds: 1.9,
            delta_seconds: 1.2,
        });
        let mut partitioned = prefix;
        for step in 1..=12 {
            partitioned.push(AppReplayEvent::Frame {
                elapsed_seconds: 0.7 + f64::from(step) * 0.1,
                delta_seconds: 0.1,
            });
        }

        let single = run_app_replay(&AppReplayScript::new(single)).unwrap();
        let partitioned = run_app_replay(&AppReplayScript::new(partitioned)).unwrap();
        let single = &single.records.last().unwrap().state;
        let partitioned = &partitioned.records.last().unwrap().state;
        assert!((single.elapsed_seconds - partitioned.elapsed_seconds).abs() < 1.0e-12);
        assert_semantic_state_eq(single, partitioned);
    }

    #[test]
    fn unsupported_trace_version_fails_before_dispatch() {
        let script = AppReplayScript {
            version: "hyperscope-app-replay/9.9".to_owned(),
            events: vec![AppReplayEvent::LoadPresentation {
                presentation: fixture(),
            }],
        };
        assert_eq!(
            run_app_replay(&script),
            Err(AppReplayError::UnsupportedVersion(script.version))
        );
    }
}
