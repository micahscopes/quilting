use super::{
    project_patch_lab_controls, set_patch_lab_session, PatchLabControlCommit,
    PatchLabControlsViewModel,
};
use crate::effect_js::patch_lab_effect_to_js;
use futures_signals::signal::SignalExt as _;
use hyperscope_app::{AppStore, PatchLabField, PatchLabSessionIntent, PatchLabShape};
use js_sys::{Array, Function};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;

const SHAPES: &[(PatchLabShape, &str)] = &[
    (PatchLabShape::Triangle, "Tri patch"),
    (PatchLabShape::Plane, "Plane"),
    (PatchLabShape::Cube, "Cube"),
];

const FIELDS: &[(PatchLabField, &str)] = &[
    (PatchLabField::ManualEdges, "Three edge controls"),
    (PatchLabField::Wave, "Traveling wave"),
    (PatchLabField::Radial, "Radial rings"),
    (PatchLabField::Sweep, "Moving detail band"),
    (PatchLabField::Uniform, "Uniform"),
];

const EDGE_LABELS: [&str; 3] = ["BC", "CA", "AB"];

/// Mount the Patch Lab as a permanent FRP view over one `AppStore`.
///
/// The commit callback receives `(sequence, revision, effects)`. Effects are
/// backend-neutral job descriptions; the platform performs worker/renderer
/// IO and returns completions through `HyperscopeAppShadow`. The platform
/// action callback temporarily owns the three host-only actions
/// `surface_wire`, `lod_colors`, and `exit` until route/render lifecycle is
/// wholly Rust-authoritative.
pub fn mount_patch_lab_controls(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_commit: Function,
    on_platform_action: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let (control, set_control) = signal(project_patch_lab_controls(
            &store.patch_lab_snapshot(),
            &store.render_snapshot(),
        ));

        let patch_projection_store = store.clone();
        let patch_updates = store.patch_lab_signal().for_each(move |patch_lab| {
            set_control.set(project_patch_lab_controls(
                &patch_lab,
                &patch_projection_store.render_snapshot(),
            ));
            async {}
        });
        wasm_bindgen_futures::spawn_local(patch_updates);

        let render_projection_store = store.clone();
        let render_updates = store.render_signal().for_each(move |render| {
            set_control.set(project_patch_lab_controls(
                &render_projection_store.patch_lab_snapshot(),
                &render,
            ));
            async {}
        });
        wasm_bindgen_futures::spawn_local(render_updates);

        view! {
            <PatchLabControls
                control
                store
                on_commit
                on_platform_action
                on_error
            />
        }
    })
    .forget();
}

#[component]
fn PatchLabControls(
    control: ReadSignal<PatchLabControlsViewModel>,
    store: AppStore,
    on_commit: Function,
    on_platform_action: Function,
    on_error: Function,
) -> impl IntoView {
    let on_commit = SendWrapper::new(on_commit);
    let on_platform_action = SendWrapper::new(on_platform_action);
    let on_error = SendWrapper::new(on_error);

    let shape_buttons = SHAPES
        .iter()
        .copied()
        .map(|(shape, label)| {
            let store = store.clone();
            let on_commit = on_commit.clone();
            let on_error = on_error.clone();
            view! {
                <button
                    type="button"
                    class:a=move || {
                        let control = control.read();
                        control.value.active && control.value.controls.shape == shape
                    }
                    aria-pressed=move || {
                        let control = control.read();
                        (control.value.active && control.value.controls.shape == shape).to_string()
                    }
                    on:click=move |_| submit_intent(
                        &store,
                        &on_commit,
                        &on_error,
                        |current| Ok(current.activate_shape(shape)),
                    )
                >{label}</button>
            }
        })
        .collect_view();

    let field_options = FIELDS
        .iter()
        .copied()
        .map(|(field, label)| {
            view! {
                <option
                    value=field.wire_name()
                    disabled=move || field == PatchLabField::ManualEdges
                        && control.read().value.controls.shape != PatchLabShape::Triangle
                >{label}</option>
            }
        })
        .collect_view();

    let edge_controls = EDGE_LABELS
        .into_iter()
        .enumerate()
        .map(|(edge, label)| {
            let store = store.clone();
            let on_commit = on_commit.clone();
            let on_error = on_error.clone();
            view! {
                <div class="sr">
                    <span class="patch-lab-edge-label">{label}</span>
                    <input
                        type="range"
                        aria-label=format!("{label} requested exponent")
                        min=move || control.read().exponent.minimum
                        max=move || control.read().exponent.maximum
                        step=move || control.read().exponent.step
                        prop:value=move || control.read().value.controls.manual_edge_exponents[edge]
                        on:input=move |event| {
                            if let Ok(exponent) = event_target_value(&event).parse::<u8>() {
                                submit_intent(
                                    &store,
                                    &on_commit,
                                    &on_error,
                                    |current| current.with_manual_edge(edge, exponent),
                                );
                            }
                        }
                    />
                    <span class="v">{move || edge_subdivision_label(&control.read(), edge)}</span>
                </div>
            }
        })
        .collect_view();

    view! {
        <div id="patch-lab-rust-view">
            <label>"Patch Lab"</label>
            <div class="btns">{shape_buttons}</div>

            <label for="patch-lab-field-rust" style="margin-top:6px">"LOD function"</label>
            <select
                id="patch-lab-field-rust"
                prop:value=move || control.read().value.controls.field.wire_name()
                on:change={
                    let store = store.clone();
                    let on_commit = on_commit.clone();
                    let on_error = on_error.clone();
                    move |event| {
                        let value = event_target_value(&event);
                        if let Some(field) = PatchLabField::from_wire_name(&value) {
                            submit_intent(
                                &store,
                                &on_commit,
                                &on_error,
                                |current| Ok(current.with_field(field)),
                            );
                        }
                    }
                }
            >{field_options}</select>

            <div hidden=move || !control.read().manual_edges_visible() style="margin-top:5px">
                {edge_controls}
            </div>

            <div hidden=move || !control.read().field_controls_visible()>
                <label style="margin-top:5px">"Requested exponent range"</label>
                <ExponentSlider
                    label="Min"
                    control
                    store=store.clone()
                    on_commit=on_commit.clone()
                    on_error=on_error.clone()
                    minimum=true
                />
                <ExponentSlider
                    label="Max"
                    control
                    store=store.clone()
                    on_commit=on_commit.clone()
                    on_error=on_error.clone()
                    minimum=false
                />
                <div class="sr">
                    <span class="patch-lab-edge-label">"Phase"</span>
                    <input
                        type="range"
                        aria-label="LOD field phase"
                        min=move || control.read().phase.minimum
                        max=move || control.read().phase.maximum
                        step=move || control.read().phase.step
                        prop:value=move || control.read().value.controls.phase_radians()
                        on:input={
                            let store = store.clone();
                            let on_commit = on_commit.clone();
                            let on_error = on_error.clone();
                            move |event| if let Ok(phase) = event_target_value(&event).parse::<f64>() {
                                submit_intent(
                                    &store,
                                    &on_commit,
                                    &on_error,
                                    |current| current.with_phase_radians(phase),
                                );
                            }
                        }
                    />
                    <span class="v">{move || format!(
                        "{:.2}",
                        control.read().value.controls.phase_radians(),
                    )}</span>
                </div>
            </div>

            <div hidden=move || !control.read().bend_visible() style="margin-top:5px">
                <label>"QB corner-weight bend"</label>
                <div class="sr">
                    <input
                        type="range"
                        aria-label="QB corner-weight bend"
                        min=move || control.read().bend.minimum
                        max=move || control.read().bend.maximum
                        step=move || control.read().bend.step
                        prop:value=move || control.read().value.controls.bend_percent
                        on:change={
                            let store = store.clone();
                            let on_commit = on_commit.clone();
                            let on_error = on_error.clone();
                            move |event| if let Ok(bend) = event_target_value(&event).parse::<u8>() {
                                submit_intent(
                                    &store,
                                    &on_commit,
                                    &on_error,
                                    |current| Ok(current.with_bend_percent(bend)),
                                );
                            }
                        }
                    />
                    <span class="v">{move || format!(
                        "{:.2}",
                        f64::from(control.read().value.controls.bend_percent) / 100.0,
                    )}</span>
                </div>
            </div>

            <div hidden=move || !control.read().grid_visible() style="margin-top:5px">
                <label>"Plane grid"</label>
                <div class="sr">
                    <input
                        type="range"
                        aria-label="Plane grid width"
                        min=move || control.read().grid.minimum
                        max=move || control.read().grid.maximum
                        step=move || control.read().grid.step
                        prop:value=move || control.read().value.controls.grid
                        on:change={
                            let store = store.clone();
                            let on_commit = on_commit.clone();
                            let on_error = on_error.clone();
                            move |event| if let Ok(grid) = event_target_value(&event).parse::<u8>() {
                                submit_intent(
                                    &store,
                                    &on_commit,
                                    &on_error,
                                    |current| Ok(current.with_grid(grid)),
                                );
                            }
                        }
                    />
                    <span class="v">{move || format!(
                        "{}²",
                        control.read().value.controls.grid,
                    )}</span>
                </div>
            </div>

            <label class="toggle-row" style="margin-top:6px">
                <input
                    type="checkbox"
                    role="switch"
                    prop:checked=move || control.read().value.controls.animate
                    on:change={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        move |event| submit_intent(
                            &store,
                            &on_commit,
                            &on_error,
                            |current| Ok(current.with_animation(event_target_checked(&event))),
                        )
                    }
                />
                <span class="toggle-label">"Animate LOD function"</span>
            </label>

            <div class="btns" style="margin-top:5px">
                <PlatformActionButton
                    action="surface_wire"
                    label="Surface + wire"
                    callback=on_platform_action.clone()
                />
                <PlatformActionButton
                    action="lod_colors"
                    label="LOD colors"
                    callback=on_platform_action.clone()
                />
                <button
                    type="button"
                    disabled=move || !control.read().value.active
                    on:click={
                        let store = store.clone();
                        let on_commit = on_commit.clone();
                        let on_error = on_error.clone();
                        let on_platform_action = on_platform_action.clone();
                        move |_| {
                            submit_intent(
                                &store,
                                &on_commit,
                                &on_error,
                                |current| Ok(current.deactivate()),
                            );
                            emit_platform_action(&on_platform_action, "exit");
                        }
                    }
                >"Exit lab"</button>
            </div>

            <div class="lab-note">{move || policy_label(&control.read())}</div>
            <div class="lab-stats" aria-live="polite">{move || status_label(&control.read())}</div>
        </div>
    }
}

#[component]
fn ExponentSlider(
    label: &'static str,
    control: ReadSignal<PatchLabControlsViewModel>,
    store: AppStore,
    on_commit: SendWrapper<Function>,
    on_error: SendWrapper<Function>,
    minimum: bool,
) -> impl IntoView {
    view! {
        <div class="sr">
            <span class="patch-lab-edge-label">{label}</span>
            <input
                type="range"
                aria-label=format!("{label} requested exponent")
                min=move || control.read().exponent.minimum
                max=move || control.read().exponent.maximum
                step=move || control.read().exponent.step
                prop:value=move || if minimum {
                    control.read().value.controls.min_exponent
                } else {
                    control.read().value.controls.max_exponent
                }
                on:input=move |event| {
                    if let Ok(exponent) = event_target_value(&event).parse::<u8>() {
                        submit_intent(&store, &on_commit, &on_error, |current| Ok(if minimum {
                            current.with_min_exponent(exponent)
                        } else {
                            current.with_max_exponent(exponent)
                        }));
                    }
                }
            />
            <span class="v">{move || {
                let exponent = if minimum {
                    control.read().value.controls.min_exponent
                } else {
                    control.read().value.controls.max_exponent
                };
                (1_u32 << exponent).to_string()
            }}</span>
        </div>
    }
}

#[component]
fn PlatformActionButton(
    action: &'static str,
    label: &'static str,
    callback: SendWrapper<Function>,
) -> impl IntoView {
    view! {
        <button type="button" on:click=move |_| emit_platform_action(&callback, action)>{label}</button>
    }
}

fn submit_intent(
    store: &AppStore,
    callback: &Function,
    error_callback: &Function,
    update: impl FnOnce(PatchLabControlsViewModel) -> Result<PatchLabSessionIntent, &'static str>,
) {
    let current = project_patch_lab_controls(&store.patch_lab_snapshot(), &store.render_snapshot());
    let intent = match update(current) {
        Ok(intent) => intent,
        Err(error) => {
            emit_error(error_callback, error);
            return;
        }
    };
    match set_patch_lab_session(store, intent) {
        Ok(committed) => emit_commit(callback, &committed),
        Err(error) => emit_error(error_callback, &error.to_string()),
    }
}

fn emit_commit(callback: &Function, committed: &PatchLabControlCommit) {
    let arguments = Array::new();
    arguments.push(&JsValue::from_str(&committed.sequence.to_string()));
    arguments.push(&JsValue::from_str(&committed.revision.to_string()));
    let effects = Array::new();
    for effect in &committed.effects {
        effects.push(&patch_lab_effect_to_js(effect));
    }
    arguments.push(&effects);
    let _ = callback.apply(&JsValue::UNDEFINED, &arguments);
}

fn emit_error(callback: &Function, message: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(message));
}

fn emit_platform_action(callback: &Function, action: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(action));
}

fn edge_subdivision_label(control: &PatchLabControlsViewModel, edge: usize) -> String {
    let exponent = control.value.controls.manual_edge_exponents[edge];
    let requested = 1_u32 << exponent;
    let resident = control
        .state
        .latest_lod
        .as_ref()
        .and_then(|lod| lod.resident_first_face)
        .map(|edges| edges[edge]);
    match resident.filter(|resident| *resident != requested) {
        Some(resident) => format!("{requested} → {resident}"),
        None => requested.to_string(),
    }
}

fn policy_label(control: &PatchLabControlsViewModel) -> String {
    format!(
        "Each edge is a request. The resident atlas reconciles shared edges and grades each face to {}:1; labels show request → resident.",
        control.max_face_edge_ratio,
    )
}

fn status_label(control: &PatchLabControlsViewModel) -> String {
    if let Some(error) = &control.state.last_error {
        return format!("{}: {}", error.code, error.message);
    }
    if let Some(job) = control.state.pending_geometry_job {
        return format!("Building geometry job {job}…");
    }
    if let Some(job) = control.state.pending_lod_job {
        return format!("Evaluating LOD job {job}…");
    }
    let Some(lod) = &control.state.latest_lod else {
        return if control.value.active {
            "Waiting for Patch Lab geometry…".to_owned()
        } else {
            "Choose a tri patch, plane, or cube.".to_owned()
        };
    };
    let promotion = if lod.promoted_edges == 0 {
        "no reconciliation promotions".to_owned()
    } else {
        format!(
            "{} edge promotion(s) on {} face(s)",
            lod.promoted_edges, lod.promoted_faces,
        )
    };
    format!(
        "{} shared-edge mismatches · observed max {}:1 · {} · {} rendered triangles · {} resident pattern(s)",
        lod.shared_edge_mismatches,
        lod.max_face_edge_ratio,
        promotion,
        lod.rendered_triangles,
        lod.histogram.len(),
    )
}
