use super::{InteractionStatusViewModel, project_interaction_status};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use leptos::mount::mount_to;
use leptos::prelude::*;

/// Mount a permanent, read-only selection view over the AppStore's throttled
/// navigation frame. The component owns no renderer handle or interaction
/// state and emits no actions.
pub fn mount_interaction_status(parent: web_sys::HtmlElement, store: AppStore) {
    mount_to(parent, move || {
        let (status, set_status) = signal(project_interaction_status(&store.navigation_snapshot()));
        let updates = store.navigation_signal().for_each(move |navigation| {
            set_status.set(project_interaction_status(&navigation));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <InteractionStatus status /> }
    })
    .forget();
}

#[component]
fn InteractionStatus(status: ReadSignal<InteractionStatusViewModel>) -> impl IntoView {
    view! {
        <div
            class="interaction-status"
            id="interaction-status-rust-view"
            role="status"
            aria-live="polite"
        >
            <span>{move || status.read().status_label()}</span>
            {move || status.read().selection.as_ref().map(|selection| view! {
                <span>{format!(" · {}", selection.geometry_label())}</span>
            })}
        </div>
    }
}
