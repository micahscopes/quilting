use super::{
    project_animation_clip_control, project_animation_control, project_animation_timeline,
    seek_animation_timeline, select_animation_clip, toggle_animation_playback,
    AnimationClipControlCommit, AnimationClipControlViewModel, AnimationClipJobEffect,
    AnimationControlViewModel, AnimationTimelineViewModel,
};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function, Object, Reflect};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount a permanent Leptos CSR playback control over the AppStore's compact
/// animation signal. The button dispatches directly through the reducer;
/// browser callbacks receive only committed adaptation or rejection effects.
pub fn mount_animation_control(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (control, set_control) = signal(project_animation_control(&store.animation_snapshot()));
        let updates = store.animation_signal().for_each(move |animation| {
            set_control.set(project_animation_control(&animation));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AnimationControl control store on_commit on_error /> }
    })
    .forget();
}

/// Mount the throttled Rust animation timeline. Playback frames publish only
/// the compact animation read model; seeking dispatches directly through the
/// reducer and returns an already committed authored sample time.
pub fn mount_animation_timeline(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (timeline, set_timeline) =
            signal(project_animation_timeline(&store.animation_snapshot()));
        let updates = store.animation_signal().for_each(move |animation| {
            set_timeline.set(project_animation_timeline(&animation));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AnimationTimeline timeline store on_commit on_error /> }
    })
    .forget();
}

/// Mount the explicit Rust-authority installed-clip selector. The summary
/// signal is used as a revision fence; each notification samples the installed
/// catalog and active/pending clip projections as one committed view.
pub fn mount_animation_clip_control(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (control, set_control) = signal(project_animation_clip_control(
            store.installed_primary_scene_snapshot().as_ref(),
            &store.animation_clip_selection_snapshot(),
        ));
        let projection_store = store.clone();
        let updates = store.summary_signal().for_each(move |_| {
            set_control.set(project_animation_clip_control(
                projection_store.installed_primary_scene_snapshot().as_ref(),
                &projection_store.animation_clip_selection_snapshot(),
            ));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AnimationClipControl control store on_commit on_error /> }
    })
    .forget();
}

#[component]
fn AnimationControl(
    control: ReadSignal<AnimationControlViewModel>,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    view! {
        <div class="toggle-row" id="animation-control-rust-view">
            <button
                type="button"
                class="toggle"
                class:on=move || control.read().playing
                role="switch"
                aria-checked=move || control.read().playing.to_string()
                aria-label=move || control.read().action_label
                title=move || control.read().state_label
                on:click=move |_| toggle_playback(&store, &on_commit, &on_error)
            ></button>
            <span class="toggle-label">"Auto-animate"</span>
        </div>
    }
}

#[component]
fn AnimationTimeline(
    timeline: ReadSignal<AnimationTimelineViewModel>,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    view! {
        <div id="animation-timeline-rust-view">
            <label for="animation-time-rust-input">"Animation Time"</label>
            <div class="sr">
                <span style="font-size:10px;color:#8b949e;width:50px">"Time"</span>
                <input
                    id="animation-time-rust-input"
                    type="range"
                    min=move || timeline.read().minimum_seconds
                    max=move || timeline.read().maximum_seconds
                    step="any"
                    disabled=move || timeline.read().disabled
                    aria-label=move || timeline.read().status_label.clone()
                    prop:value=move || timeline.read().sample_time_seconds
                    on:input=move |event| {
                        let Ok(value) = event_target_value(&event).parse::<f64>() else {
                            return;
                        };
                        match seek_animation_timeline(&store, value) {
                            Ok(committed) => {
                                let arguments = Array::new();
                                arguments.push(&JsValue::from_f64(committed.sample_time_seconds));
                                arguments.push(&JsValue::from(committed.sequence));
                                arguments.push(&JsValue::from(committed.revision));
                                let _ = on_commit.apply(&JsValue::UNDEFINED, &arguments);
                            }
                            Err(error) => {
                                let _ = on_error.call1(
                                    &JsValue::UNDEFINED,
                                    &JsValue::from_str(&error.to_string()),
                                );
                            }
                        }
                    }
                />
                <span class="v">{move || format!("{:.2}", timeline.read().sample_time_seconds)}</span>
            </div>
        </div>
    }
}

#[component]
fn AnimationClipControl(
    control: ReadSignal<AnimationClipControlViewModel>,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    view! {
        <div id="animation-clip-control-rust-view">
            <label for="animation-clip-select-rust">"Animation"</label>
            <select
                id="animation-clip-select-rust"
                disabled=move || control.read().disabled
                aria-busy=move || control.read().pending_index.is_some().to_string()
                aria-label=move || control.read().status_label.clone()
                prop:value=move || control
                    .read()
                    .selected_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "-1".to_owned())
                on:change=move |event| {
                    if let Ok(index) = event_target_value(&event).parse::<u32>() {
                        choose_clip(&store, index, &on_commit, &on_error);
                    }
                }
            >
                <For
                    each=move || control.read().options.clone()
                    key=|option| option.index
                    children=move |option| view! {
                        <option value=option.index.to_string()>{option.label}</option>
                    }
                />
            </select>
            <div class="lab-note" aria-live="polite">{move || control.read().status_label.clone()}</div>
        </div>
    }
}

fn toggle_playback(store: &AppStore, callback: &Function, error_callback: &Function) {
    let committed = match toggle_animation_playback(store) {
        Ok(committed) => committed,
        Err(error) => {
            let _ =
                error_callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(&error.to_string()));
            return;
        }
    };
    let arguments = Array::new();
    arguments.push(&JsValue::from_bool(committed.playing));
    arguments.push(&JsValue::from(committed.sequence));
    arguments.push(&JsValue::from(committed.revision));
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}

fn choose_clip(store: &AppStore, index: u32, callback: &Function, error_callback: &Function) {
    match select_animation_clip(store, index) {
        Ok(committed) => emit_clip_commit(callback, &committed),
        Err(error) => {
            let _ =
                error_callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(&error.to_string()));
        }
    }
}

fn emit_clip_commit(callback: &Function, committed: &AnimationClipControlCommit) {
    let arguments = Array::new();
    arguments.push(&JsValue::from_f64(f64::from(committed.requested_index)));
    arguments.push(&JsValue::from_str(&committed.sequence.to_string()));
    arguments.push(&JsValue::from_str(&committed.revision.to_string()));
    arguments.push(
        &committed
            .selection
            .as_ref()
            .map(|effect| clip_effect_to_js("select_animation_clip", effect))
            .unwrap_or(JsValue::NULL),
    );
    let cancellations = Array::new();
    for effect in &committed.cancellations {
        cancellations.push(&clip_effect_to_js(
            "cancel_animation_clip_selection",
            effect,
        ));
    }
    arguments.push(&cancellations);
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}

fn clip_effect_to_js(effect_type: &str, effect: &AnimationClipJobEffect) -> JsValue {
    let object = Object::new();
    for (key, value) in [
        ("type", JsValue::from_str(effect_type)),
        ("job_id", JsValue::from_str(&effect.job_id.to_string())),
        (
            "scene_request_id",
            JsValue::from_str(&effect.scene_request_id),
        ),
        ("asset_id", JsValue::from_str(&effect.asset_id)),
        (
            "clip_index",
            JsValue::from_f64(f64::from(effect.clip_index)),
        ),
    ] {
        let _ = Reflect::set(&object, &JsValue::from_str(key), &value);
    }
    object.into()
}
