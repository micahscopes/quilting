use super::{project_presentation_card, PresentationCardViewModel};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::Function;
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount a permanent Leptos CSR presentation card over the AppStore's
/// low-rate presentation projection. Cue buttons emit semantic intent to the
/// temporary browser effect adapter; the component owns no application state.
pub fn mount_presentation_card(parent: web_sys::HtmlElement, store: AppStore, on_action: Function) {
    mount_to(parent, move || {
        let (card, set_card) = signal(project_presentation_card(
            store.presentation_snapshot().as_ref(),
        ));
        let updates = store.presentation_signal().for_each(move |presentation| {
            set_card.set(project_presentation_card(presentation.as_ref()));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <PresentationCardView card on_action /> }
    })
    .forget();
}

#[component]
fn PresentationCardView(
    card: ReadSignal<Option<PresentationCardViewModel>>,
    on_action: Function,
) -> impl IntoView {
    let on_action = SendWrapper::new(on_action);
    view! {
        <section
            class="presentation-card"
            class:active=move || card.read().is_some()
            id="presentation-card-rust-view"
            aria-live="polite"
        >
            {move || {
                card.get().map(|card| {
                    let reverse = on_action.clone();
                    let advance = on_action.clone();
                    let status = card.adapter_status();
                    view! {
                        <div class="presentation-eyebrow">{card.eyebrow}</div>
                        <h2 class="presentation-heading">{card.heading}</h2>
                        <p class="presentation-body">{card.body}</p>
                        <div class="presentation-footer">
                            <button
                                type="button"
                                aria-label="Previous cue"
                                disabled=!card.can_reverse
                                on:click=move |_| invoke_action(&reverse, "reverse")
                            >
                                "←"
                            </button>
                            <button
                                type="button"
                                aria-label="Next cue"
                                disabled=!card.can_advance
                                on:click=move |_| invoke_action(&advance, "advance")
                            >
                                "→"
                            </button>
                            <span class="presentation-progress">{card.progress}</span>
                            <span class="presentation-adapter-status">{status}</span>
                        </div>
                    }
                })
            }}
        </section>
    }
}

fn invoke_action(callback: &Function, action: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(action));
}
