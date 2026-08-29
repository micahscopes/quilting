use super::{project_render_controls, RenderControlIntent, RenderControlsViewModel};
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
/// AppStore render signal. Every user edit emits one complete replacement
/// value through the temporary platform callback.
pub fn mount_render_controls(parent: web_sys::HtmlElement, store: AppStore, on_action: Function) {
    mount_to(parent, move || {
        let (controls, set_controls) = signal(project_render_controls(&store.render_snapshot()));
        let updates = store.render_signal().for_each(move |render| {
            set_controls.set(project_render_controls(&render));
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <RenderControls controls on_action /> }
    })
    .forget();
}

#[component]
fn RenderControls(
    controls: ReadSignal<RenderControlsViewModel>,
    on_action: Function,
) -> impl IntoView {
    let on_action = SendWrapper::new(on_action);
    let style_buttons = RENDER_STYLES
        .iter()
        .map(|&(style, label)| {
            let on_action = on_action.clone();
            view! {
                <button
                    type="button"
                    class:a=move || controls.read().value.style == style
                    aria-pressed=move || (controls.read().value.style == style).to_string()
                    on:click=move |_| emit(&on_action, controls.get_untracked().with_style(style))
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
                    step="1"
                    prop:value=move || controls.read().value.resolution_level
                    on:input={
                        let on_action = on_action.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<u8>() {
                            emit(&on_action, controls.get_untracked().with_resolution(value));
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
                    step="1"
                    prop:value=move || controls.read().value.density
                    on:input={
                        let on_action = on_action.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<f64>() {
                            emit(&on_action, controls.get_untracked().with_density(value));
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
                        let on_action = on_action.clone();
                        move |event| emit(
                            &on_action,
                            controls.get_untracked().with_screen_attenuation(event_target_checked(&event)),
                        )
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
                    step="0.1"
                    prop:value=move || controls.read().value.min_pixels_per_subdivision
                    on:input={
                        let on_action = on_action.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<f64>() {
                            emit(&on_action, controls.get_untracked().with_pixel_floor(value));
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
                    step="1"
                    prop:value=move || controls.read().value.atlas_exponent
                    on:change={
                        let on_action = on_action.clone();
                        move |event| if let Ok(value) = event_target_value(&event).parse::<u8>() {
                            emit(&on_action, controls.get_untracked().with_atlas(value));
                        }
                    }
                />
                <span class="v">{move || controls.read().value.atlas_exponent}</span>
            </div>

            <label>"Within-face grading"</label>
            <div class="btns">
                {[2_u8, 4_u8].into_iter().map(|ratio| {
                    let on_action = on_action.clone();
                    view! {
                        <button
                            type="button"
                            class:a=move || controls.read().value.max_face_edge_ratio == ratio
                            aria-pressed=move || {
                                (controls.read().value.max_face_edge_ratio == ratio).to_string()
                            }
                            on:click=move |_| emit(
                                &on_action,
                                controls.get_untracked().with_grading(ratio),
                            )
                        >{format!("{ratio}:1")}</button>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

fn emit(callback: &Function, intent: RenderControlIntent) {
    let arguments = Array::new();
    arguments.push(&JsValue::from_str(intent.style));
    arguments.push(&JsValue::from_f64(f64::from(intent.resolution_level)));
    arguments.push(&JsValue::from_f64(intent.density));
    arguments.push(&JsValue::from_bool(intent.screen_attenuation));
    arguments.push(&JsValue::from_f64(intent.min_pixels_per_subdivision));
    arguments.push(&JsValue::from_f64(f64::from(intent.atlas_exponent)));
    arguments.push(&JsValue::from_f64(f64::from(intent.max_face_edge_ratio)));
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}
