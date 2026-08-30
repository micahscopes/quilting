use super::{
    project_navigation_controls, set_navigation_controls, NavigationControlCommit,
    NavigationControlIntent, NavigationControlsViewModel,
};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount the Rust-authoritative low-rate navigation settings view. Camera
/// integration remains on its direct frame snapshot and never waits for DOM
/// publication.
pub fn mount_navigation_controls(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (controls, set_controls) = signal(project_navigation_controls(
            &store.navigation_settings_snapshot(),
        ));
        let updates = store
            .navigation_settings_signal()
            .for_each(move |settings| {
                set_controls.set(project_navigation_controls(&settings));
                async {}
            });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <NavigationControls controls store on_commit on_error /> }
    })
    .forget();
}

#[component]
fn NavigationControls(
    controls: ReadSignal<NavigationControlsViewModel>,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    view! {
        <div id="navigation-controls-rust-view">
            <NavigationSlider
                label="Focus/navigation transition"
                domain=move || controls.read().transition
                value=move || controls.read().value.settings().transition_seconds
                display=move || format!("{:.2} s", controls.read().value.settings().transition_seconds)
                update=move |current, value| current.with_transition_seconds(value)
                store=store.clone() on_commit=on_commit.clone() on_error=on_error.clone()
            />
            <NavigationSlider
                label="Walk frame smoothing"
                domain=move || controls.read().smoothing
                value=move || controls.read().value.settings().surface_walk.smoothing_seconds
                display=move || format!("{:.2} s", controls.read().value.settings().surface_walk.smoothing_seconds)
                update=move |current, value| current.with_smoothing_seconds(value)
                store=store.clone() on_commit=on_commit.clone() on_error=on_error.clone()
            />
            <NavigationSlider
                label="Walk frame pull"
                domain=move || controls.read().tangent_pull
                value=move || controls.read().value.settings().surface_walk.tangent_pull_fraction
                display=move || format!("{:.0}%", controls.read().value.settings().surface_walk.tangent_pull_fraction * 100.0)
                update=move |current, value| current.with_tangent_pull_fraction(value)
                store=store.clone() on_commit=on_commit.clone() on_error=on_error.clone()
            />
            <NavigationSlider
                label="Walk speed"
                domain=move || controls.read().speed
                value=move || controls.read().value.settings().surface_walk.speed_octave_steps
                display=move || {
                    let walk = controls.read().value.settings().surface_walk;
                    format!("{:.2} R/s", walk.base_radii_per_second * 2.0_f64.powf(walk.speed_octave_steps / 100.0))
                }
                update=move |current, value| current.with_speed_octave_steps(value)
                store=store.clone() on_commit=on_commit.clone() on_error=on_error.clone()
            />
            <NavigationSlider
                label="Walk scale"
                domain=move || controls.read().body_scale
                value=move || controls.read().value.settings().surface_walk.body_scale_octave_steps
                display=move || format!("{:.2}×", 2.0_f64.powf(controls.read().value.settings().surface_walk.body_scale_octave_steps / 100.0))
                update=move |current, value| current.with_body_scale_octave_steps(value)
                store=store.clone() on_commit=on_commit.clone() on_error=on_error.clone()
            />
            <NavigationSlider
                label="Walk eye height"
                domain=move || controls.read().eye_height
                value=move || controls.read().value.settings().surface_walk.eye_height_octave_steps
                display=move || format!("{:.2}×", 2.0_f64.powf(controls.read().value.settings().surface_walk.eye_height_octave_steps / 100.0))
                update=move |current, value| current.with_eye_height_octave_steps(value)
                store on_commit on_error
            />
        </div>
    }
}

#[component]
fn NavigationSlider<D, V, F, U>(
    label: &'static str,
    domain: D,
    value: V,
    display: F,
    update: U,
    store: AppStore,
    on_commit: SendWrapper<Function>,
    on_error: SendWrapper<Function>,
) -> impl IntoView
where
    D: Fn() -> crate::controls::NumericControlViewDomain + Copy + Send + Sync + 'static,
    V: Fn() -> f64 + Send + Sync + 'static,
    F: Fn() -> String + Send + Sync + 'static,
    U: Fn(NavigationControlIntent, f64) -> NavigationControlIntent + Copy + Send + Sync + 'static,
{
    view! {
        <label>{label}</label>
        <div class="sr">
            <input
                type="range"
                min=move || domain().minimum
                max=move || domain().maximum
                step=move || domain().step
                prop:value=value
                on:input=move |event| {
                    if let Ok(value) = event_target_value(&event).parse::<f64>() {
                        submit_intent(&store, &on_commit, &on_error, |current| {
                            update(current, value)
                        });
                    }
                }
            />
            <span class="v">{display}</span>
        </div>
    }
}

fn submit_intent(
    store: &AppStore,
    callback: &Function,
    error_callback: &Function,
    update: impl FnOnce(NavigationControlIntent) -> NavigationControlIntent,
) {
    let intent = update(project_navigation_controls(&store.navigation_settings_snapshot()).value);
    let committed = match set_navigation_controls(store, intent) {
        Ok(committed) => committed,
        Err(error) => {
            let _ =
                error_callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(&error.to_string()));
            return;
        }
    };
    emit_committed(callback, &committed);
}

fn emit_committed(callback: &Function, committed: &NavigationControlCommit) {
    let settings = committed.value.settings();
    let arguments = Array::new();
    arguments.push(&JsValue::from_f64(settings.transition_seconds));
    arguments.push(&JsValue::from_f64(settings.surface_walk.smoothing_seconds));
    arguments.push(&JsValue::from_f64(
        settings.surface_walk.tangent_pull_fraction,
    ));
    arguments.push(&JsValue::from_f64(settings.surface_walk.speed_octave_steps));
    arguments.push(&JsValue::from_f64(
        settings.surface_walk.body_scale_octave_steps,
    ));
    arguments.push(&JsValue::from_f64(
        settings.surface_walk.eye_height_octave_steps,
    ));
    arguments.push(&JsValue::from(committed.sequence));
    arguments.push(&JsValue::from(committed.revision));
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}
