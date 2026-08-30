use super::{
    project_render_controls, set_render_controls, RenderControlCommit, RenderControlIntent,
    RenderControlsViewModel,
};
use crate::effect_js::patch_lab_effect_to_js;
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

const RENDER_STYLES: &[(&str, &str)] = &[
    ("pbr", "PBR"),
    ("matcap", "Matcap"),
    ("lod", "LOD"),
    ("wire", "Wire"),
    ("matcap_wire", "Both"),
    ("normals", "Normals"),
    ("stretch", "Stretch"),
];

/// Mount the explicit Rust-authority render controls over the committed
/// AppStore render signal. Every user edit dispatches one complete replacement
/// value directly through the reducer; the platform callback only receives the
/// resulting committed projection for renderer adaptation.
pub fn mount_render_controls(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (controls, set_controls) = signal(project_render_controls(&store.render_snapshot()));
        let updates = store.render_signal().for_each(move |render| {
            set_controls.set(project_render_controls(&render));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <RenderControls controls store on_commit on_error /> }
    })
    .forget();
}

#[component]
fn RenderControls(
    controls: ReadSignal<RenderControlsViewModel>,
    store: AppStore,
    on_commit: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_error = SendWrapper::new(on_error);
    let style_buttons = RENDER_STYLES
        .iter()
        .map(|&(style, label)| {
            let store = store.clone();
            let on_commit = on_commit.clone();
            let on_error = on_error.clone();
            view! {
                <button
                    type="button"
                    class:a=move || controls.read().value.style == style
                    aria-pressed=move || (controls.read().value.style == style).to_string()
                    on:click=move |_| submit_intent(&store, &on_commit, &on_error, |current| {
                        current.with_style(style)
                    })
                >{label}</button>
            }
        })
        .collect_view();

    view! {
        <div id="render-controls-rust-view">
            <label>"Render"</label>
            <div class="btns">{style_buttons}</div>

            <label>"Resolution (0=auto)"</label>
            <div class="sr">
                <input
                    type="range"
                    min=move || controls.read().resolution.minimum
                    max=move || controls.read().resolution.maximum
                    step=move || controls.read().resolution.step
                    prop:value=move || controls.read().value.resolution_level
                    on:input={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<u8>() {
                            submit_intent(&store, &on_commit, &on_error, |current| {
                                current.with_resolution(value)
                            });
                        }
                    }
                />
                <span class="v">{move || {
                    let level = controls.read().value.resolution_level;
                    if level == 0 { "auto".to_owned() } else { format!("{}×", 1_u16 << level) }
                }}</span>
            </div>

            <label>"Tess density"</label>
            <div class="sr">
                <input
                    type="range"
                    min=move || controls.read().density.minimum
                    max=move || controls.read().density.maximum
                    step=move || controls.read().density.step
                    prop:value=move || controls.read().value.density
                    on:input={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<f64>() {
                            submit_intent(&store, &on_commit, &on_error, |current| {
                                current.with_density(value)
                            });
                        }
                    }
                />
                <span class="v">{move || format!("{:.0}", controls.read().value.density)}</span>
            </div>

            <div class="toggle-row">
                <input
                    type="checkbox"
                    role="switch"
                    prop:checked=move || controls.read().value.screen_attenuation
                    on:change={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| submit_intent(&store, &on_commit, &on_error, |current| {
                            current.with_screen_attenuation(event_target_checked(&event))
                        })
                    }
                />
                <span class="toggle-label">"Screen-space attenuation"</span>
            </div>

            <label>"Pixel floor per sub-edge"</label>
            <div class="sr">
                <input
                    type="range"
                    min=move || controls.read().pixel_floor.minimum
                    max=move || controls.read().pixel_floor.maximum
                    step=move || controls.read().pixel_floor.step
                    prop:value=move || controls.read().value.min_pixels_per_subdivision
                    on:input={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<f64>() {
                            submit_intent(&store, &on_commit, &on_error, |current| {
                                current.with_pixel_floor(value)
                            });
                        }
                    }
                />
                <span class="v">{move || format!(
                    "{:.1}", controls.read().value.min_pixels_per_subdivision,
                )}</span>
            </div>

            <label>"Atlas resolution"</label>
            <div class="sr">
                <input
                    type="range"
                    min=move || controls.read().atlas.minimum
                    max=move || controls.read().atlas.maximum
                    step=move || controls.read().atlas.step
                    prop:value=move || controls.read().value.atlas_exponent
                    on:change={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<u8>() {
                            submit_intent(&store, &on_commit, &on_error, |current| {
                                current.with_atlas(value)
                            });
                        }
                    }
                />
                <span class="v">{move || controls.read().value.atlas_exponent}</span>
            </div>

            <label>"Within-face grading"</label>
            <div class="btns">
                {[2_u8, 4_u8].into_iter().map(|ratio| {
                    let store = store.clone();
                    let on_commit = on_commit.clone();
                    let on_error = on_error.clone();
                    view! {
                        <button
                            type="button"
                            class:a=move || controls.read().value.max_face_edge_ratio == ratio
                            aria-pressed=move || {
                                (controls.read().value.max_face_edge_ratio == ratio).to_string()
                            }
                            on:click=move |_| submit_intent(&store, &on_commit, &on_error, |current| {
                                current.with_grading(ratio)
                            })
                        >{format!("{ratio}:1")}</button>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

fn submit_intent(
    store: &AppStore,
    callback: &Function,
    error_callback: &Function,
    update: impl FnOnce(RenderControlsViewModel) -> RenderControlIntent,
) {
    // Read the reducer directly at the event boundary. The asynchronously
    // published Leptos signal is a view projection and may legitimately lag a
    // rapid sequence of input events by one microtask.
    let intent = update(project_render_controls(&store.render_snapshot()));
    let committed = match set_render_controls(store, intent) {
        Ok(committed) => committed,
        Err(error) => {
            emit_error(error_callback, &error.to_string());
            return;
        }
    };
    emit_committed(callback, &committed);
}

fn emit_error(callback: &Function, message: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(message));
}

fn emit_committed(callback: &Function, committed: &RenderControlCommit) {
    let intent = committed.value;
    let arguments = Array::new();
    arguments.push(&JsValue::from_str(intent.style));
    arguments.push(&JsValue::from_f64(f64::from(intent.resolution_level)));
    arguments.push(&JsValue::from_f64(intent.density));
    arguments.push(&JsValue::from_bool(intent.screen_attenuation));
    arguments.push(&JsValue::from_f64(intent.min_pixels_per_subdivision));
    arguments.push(&JsValue::from_f64(f64::from(intent.atlas_exponent)));
    arguments.push(&JsValue::from_f64(f64::from(intent.max_face_edge_ratio)));
    let focus = intent.focus_postprocess;
    arguments.push(&JsValue::from_bool(focus.enabled));
    arguments.push(&JsValue::from_f64(f64::from(focus.mode.wire_index())));
    arguments.push(&JsValue::from_f64(f64::from(focus.blur_radius_pixels)));
    arguments.push(&JsValue::from_f64(focus.blur_strength));
    arguments.push(&JsValue::from_f64(focus.focus_coordinate));
    arguments.push(&JsValue::from_f64(focus.bandwidth));
    arguments.push(&JsValue::from_bool(focus.normalize_range));
    arguments.push(&JsValue::from_f64(f64::from(focus.gaussian_passes)));
    arguments.push(&JsValue::from_f64(f64::from(focus.kawase_passes)));
    arguments.push(&JsValue::from_f64(focus.kawase_offset));
    arguments.push(&JsValue::from(committed.sequence));
    let effects = Array::new();
    for effect in &committed.patch_lab_effects {
        effects.push(&patch_lab_effect_to_js(effect));
    }
    arguments.push(&effects);
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}
