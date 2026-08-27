use super::{project_navigation_status, NavigationStatusViewModel};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use leptos::mount::mount_to;
use leptos::prelude::*;

/// Mount a permanent Leptos CSR status view over the AppStore's throttled
/// navigation projection. The component emits no actions and owns no state.
pub fn mount_navigation_status(parent: web_sys::HtmlElement, store: AppStore) {
    mount_to(parent, move || {
        let (status, set_status) = signal(project_navigation_status(&store.navigation_snapshot()));
        let updates = store.navigation_signal().for_each(move |navigation| {
            set_status.set(project_navigation_status(&navigation));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <NavigationStatus status /> }
    })
    .forget();
}

#[component]
fn NavigationStatus(status: ReadSignal<NavigationStatusViewModel>) -> impl IntoView {
    view! {
        <div
            class="stats navigation-status"
            id="navigation-status-rust-view"
            role="status"
            aria-live="polite"
        >
            <strong>"Navigation"</strong>
            <span>{move || format!(" · {} · {} · {}",
                status.read().anchor_label(),
                status.read().chart_label(),
                status.read().focus_label(),
            )}</span>
            <br />
            <span>{move || format!(
                "sphere r {:.3} · shell {:.3} · aperture {:.3} · FOV {:.1}°",
                status.read().sphere_radius,
                status.read().focus_coordinate,
                status.read().angular_aperture,
                status.read().vertical_fov_degrees,
            )}</span>
        </div>
    }
}
