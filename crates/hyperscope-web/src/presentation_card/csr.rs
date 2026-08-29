use super::{
    activate_presentation_card, project_presentation_card, PresentationAnimationClipEffect,
    PresentationCardAction, PresentationCardViewModel,
};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function, Object, Reflect};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount a permanent Leptos CSR presentation card over the AppStore's
/// low-rate presentation projection. The preparation callback synchronizes
/// platform camera/focus state; Rust then commits the cue action and emits only
/// committed adaptation or rejection effects.
pub fn mount_presentation_card(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_prepare: Function,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (card, set_card) = signal(project_presentation_card(
            store.presentation_snapshot().as_ref(),
        ));
        let updates = store.presentation_signal().for_each(move |presentation| {
            set_card.set(project_presentation_card(presentation.as_ref()));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <PresentationCardView card store on_prepare on_commit on_error /> }
    })
    .forget();
}

#[component]
fn PresentationCardView(
    card: ReadSignal<Option<PresentationCardViewModel>>,
    store: AppStore,
    on_prepare: Function,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_prepare = SendWrapper::new(on_prepare);
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    view! {
        <section
            class="presentation-card"
            class:active=move || card.read().is_some()
            id="presentation-card-rust-view"
            aria-live="polite"
        >
            {move || {
                card.get().map(|card| {
                    let reverse_store = store.clone();
                    let reverse_prepare = on_prepare.clone();
                    let reverse_commit = on_commit.clone();
                    let reverse_error = on_error.clone();
                    let advance_store = store.clone();
                    let advance_prepare = on_prepare.clone();
                    let advance_commit = on_commit.clone();
                    let advance_error = on_error.clone();
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
                                on:click=move |_| activate(
                                    &reverse_store,
                                    &reverse_prepare,
                                    &reverse_commit,
                                    &reverse_error,
                                    PresentationCardAction::Reverse,
                                )
                            >
                                "←"
                            </button>
                            <button
                                type="button"
                                aria-label="Next cue"
                                disabled=!card.can_advance
                                on:click=move |_| activate(
                                    &advance_store,
                                    &advance_prepare,
                                    &advance_commit,
                                    &advance_error,
                                    PresentationCardAction::Advance,
                                )
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

fn activate(
    store: &AppStore,
    prepare_callback: &Function,
    commit_callback: &Function,
    error_callback: &Function,
    action: PresentationCardAction,
) {
    let action_name = action.wire_name();
    if let Err(error) = prepare_callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(action_name))
    {
        emit_error(
            error_callback,
            &error
                .as_string()
                .unwrap_or_else(|| "presentation preparation failed".to_owned()),
        );
        return;
    }
    let committed = match activate_presentation_card(store, action) {
        Ok(committed) => committed,
        Err(error) => {
            emit_error(error_callback, &error.to_string());
            return;
        }
    };
    let arguments = Array::new();
    arguments.push(&JsValue::from_str(action_name));
    arguments.push(&JsValue::from(committed.sequence));
    arguments.push(&JsValue::from(committed.revision));
    arguments.push(
        &committed
            .selection
            .as_ref()
            .map(|effect| presentation_clip_effect_to_js("select_animation_clip", effect))
            .unwrap_or(JsValue::NULL),
    );
    let cancellations = Array::new();
    for effect in &committed.cancellations {
        cancellations.push(&presentation_clip_effect_to_js(
            "cancel_animation_clip_selection",
            effect,
        ));
    }
    arguments.push(&cancellations);
    let _ = commit_callback.apply(&JsValue::UNDEFINED, &arguments);
}

fn presentation_clip_effect_to_js(
    effect_type: &str,
    effect: &PresentationAnimationClipEffect,
) -> JsValue {
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

fn emit_error(callback: &Function, message: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(message));
}
