use super::{project_animation_control, toggle_animation_playback, AnimationControlViewModel};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount a permanent Leptos CSR playback control over the AppStore's low-rate
/// summary signal. The button dispatches directly through the reducer; browser
/// callbacks receive only committed renderer adaptation or rejection effects.
pub fn mount_animation_control(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (control, set_control) = signal(project_animation_control(&store.summary_snapshot()));
        let updates = store.summary_signal().for_each(move |summary| {
            set_control.set(project_animation_control(&summary));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AnimationControl control store on_commit on_error /> }
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
