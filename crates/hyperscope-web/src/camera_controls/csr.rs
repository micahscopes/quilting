use super::{project_camera_lens_control, queue_camera_lens_control, CameraLensControlViewModel};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

/// Mount the camera-lens control over the AppStore navigation projection. The
/// queue callback is diagnostic/platform scheduling only: it receives the
/// validated Rust request, never an independently interpreted browser value.
pub fn mount_camera_lens_control(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_queue: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (lens, set_lens) = signal(project_camera_lens_control(&store.navigation_snapshot()));
        let updates = store.navigation_signal().for_each(move |navigation| {
            set_lens.set(project_camera_lens_control(&navigation));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <CameraLensControl lens store on_queue on_error /> }
    })
    .forget();
}

#[component]
fn CameraLensControl(
    lens: ReadSignal<CameraLensControlViewModel>,
    store: AppStore,
    on_queue: Function,
    on_error: Function,
) -> impl IntoView {
    let on_queue = SendWrapper::new(on_queue);
    let on_error = SendWrapper::new(on_error);
    view! {
        <div id="camera-lens-control-rust-view">
            <label style="margin-top:5px" for="camera-fov-rust-input">
                "Vertical field of view"
            </label>
            <div class="sr">
                <input
                    id="camera-fov-rust-input"
                    type="range"
                    min=move || lens.read().domain.minimum
                    max=move || lens.read().domain.maximum
                    step=move || lens.read().domain.step
                    prop:value=move || lens.read().vertical_fov_degrees
                    on:input=move |event| {
                        let Ok(value) = event_target_value(&event).parse::<f64>() else {
                            return;
                        };
                        match queue_camera_lens_control(&store, value) {
                            Ok(queued) => {
                                let arguments = Array::new();
                                arguments.push(&JsValue::from_f64(
                                    queued.requested_lens.vertical_fov_radians.to_degrees(),
                                ));
                                arguments.push(&JsValue::from(queued.sequence));
                                arguments.push(&JsValue::from(queued.queue_revision));
                                let _ = on_queue.apply(&JsValue::UNDEFINED, &arguments);
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
                <span class="v">{move || format!("{:.0}°", lens.read().vertical_fov_degrees)}</span>
            </div>
        </div>
    }
}
