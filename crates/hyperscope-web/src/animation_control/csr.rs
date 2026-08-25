use super::{project_animation_control, AnimationControlViewModel};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::Function;
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount a permanent Leptos CSR playback control over the AppStore's low-rate
/// summary signal. The button emits only semantic toggle intent through the
/// temporary browser effect adapter.
pub fn mount_animation_control(parent: web_sys::HtmlElement, store: AppStore, on_action: Function) {
    mount_to(parent, move || {
        let (control, set_control) = signal(project_animation_control(&store.summary_snapshot()));
        let updates = store.summary_signal().for_each(move |summary| {
            set_control.set(project_animation_control(&summary));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AnimationControl control on_action /> }
    })
    .forget();
}

#[component]
fn AnimationControl(
    control: ReadSignal<AnimationControlViewModel>,
    on_action: Function,
) -> impl IntoView {
    let on_action = SendWrapper::new(on_action);
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
                on:click=move |_| invoke_action(&on_action, "toggle")
            ></button>
            <span class="toggle-label">"Auto-animate"</span>
        </div>
    }
}

fn invoke_action(callback: &Function, action: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(action));
}
