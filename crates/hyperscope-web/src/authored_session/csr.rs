use super::{
    project_authored_session, proposal_role_wire_name, AuthoredSessionPhase,
    AuthoredSessionViewModel,
};
use futures_signals::signal::SignalExt as _;
use hyperscope_app::AppStore;
use js_sys::{Array, Function, Promise};
use leptos::mount::mount_to;
use leptos::prelude::*;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

/// Mount a thin CSR view over reducer-owned authored-session lifecycle.
///
/// `on_open(project_id, proposal_role)` and `on_close()` own browser resources
/// and must return promise-like values. The open callback enters the generated
/// Rust durability adapter, which alone commits the semantic intent. The new
/// project callback supplies browser entropy but does not open a session.
pub fn mount_authored_session_control(
    parent: web_sys::HtmlElement,
    store: AppStore,
    on_open: Function,
    on_close: Function,
    on_new_project: Function,
    on_error: Function,
) {
    mount_to(parent, move || {
        let initial = project_authored_session(&store.authored_session_snapshot());
        let initial_project = initial.project_id().unwrap_or_default();
        let initial_role = proposal_role_wire_name(initial.proposal_role()).to_owned();
        let (session, set_session) = signal(initial);
        let (project_id, set_project_id) = signal(initial_project);
        let (proposal_role, set_proposal_role) = signal(initial_role);
        let (platform_pending, set_platform_pending) = signal(false);
        let updates = store.authored_session_signal().for_each(move |snapshot| {
            let projected = project_authored_session(&snapshot);
            if let Some(intent) = projected.intent {
                set_project_id.set(intent.project_id.to_string());
                set_proposal_role.set(proposal_role_wire_name(intent.proposal_role).to_owned());
            }
            set_session.set(projected);
            async {}
        });
        wasm_bindgen_futures::spawn_local(updates);
        view! {
            <AuthoredSessionControl
                session
                project_id
                set_project_id
                proposal_role
                set_proposal_role
                platform_pending
                set_platform_pending
                on_open
                on_close
                on_new_project
                on_error
            />
        }
    })
    .forget();
}

#[component]
fn AuthoredSessionControl(
    session: ReadSignal<AuthoredSessionViewModel>,
    project_id: ReadSignal<String>,
    set_project_id: WriteSignal<String>,
    proposal_role: ReadSignal<String>,
    set_proposal_role: WriteSignal<String>,
    platform_pending: ReadSignal<bool>,
    set_platform_pending: WriteSignal<bool>,
    on_open: Function,
    on_close: Function,
    on_new_project: Function,
    on_error: Function,
) -> impl IntoView {
    let on_open = SendWrapper::new(on_open);
    let on_close = SendWrapper::new(on_close);
    let on_new_project = SendWrapper::new(on_new_project);
    let on_error = SendWrapper::new(on_error);
    let new_project_callback = on_new_project.clone();
    let new_project_error = on_error.clone();
    let primary_open = on_open.clone();
    let primary_close = on_close.clone();
    let primary_error = on_error.clone();
    view! {
        <div id="authored-session-rust-view">
            <label for="authored-project-rust">"Authored project UUID"</label>
            <input
                class="peer-input"
                id="authored-project-rust"
                type="text"
                autocomplete="off"
                spellcheck="false"
                placeholder="Shared project UUID"
                disabled=move || session.read().inputs_locked || platform_pending.get()
                prop:value=move || project_id.get()
                on:input=move |event| set_project_id.set(event_target_value(&event))
            />
            <label for="authored-role-rust">"Raw proposal role"</label>
            <select
                id="authored-role-rust"
                disabled=move || session.read().inputs_locked || platform_pending.get()
                prop:value=move || proposal_role.get()
                on:change=move |event| {
                    let value = event_target_value(&event);
                    if matches!(value.as_str(), "replica" | "admission_authority") {
                        set_proposal_role.set(value);
                    }
                }
            >
                <option value="replica">"Replica (authorized records only)"</option>
                <option value="admission_authority">"Admission authority (promote Blender proposals)"</option>
            </select>
            <div class="btns">
                <button
                    type="button"
                    disabled=move || session.read().inputs_locked || platform_pending.get()
                    on:click=move |_| {
                        match new_project_callback.call0(&JsValue::UNDEFINED) {
                            Ok(value) => match value.as_string() {
                                Some(value) if !value.trim().is_empty() => set_project_id.set(value),
                                _ => emit_error(
                                    &new_project_error,
                                    "browser did not supply a new project UUID",
                                ),
                            },
                            Err(error) => emit_js_error(&new_project_error, error),
                        }
                    }
                >"New project ID"</button>
                <button
                    type="button"
                    class:a=move || session.read().phase == AuthoredSessionPhase::Active
                    disabled=move || {
                        let projected = session.read();
                        projected.primary_disabled
                            || platform_pending.get()
                            || (projected.phase != AuthoredSessionPhase::Active
                                && project_id.read().trim().is_empty())
                    }
                    on:click=move |_| {
                        let projected = session.read();
                        let callback = if projected.phase == AuthoredSessionPhase::Active {
                            &*primary_close
                        } else {
                            &*primary_open
                        };
                        let arguments = Array::new();
                        if projected.phase != AuthoredSessionPhase::Active {
                            arguments.push(&JsValue::from_str(project_id.read().trim()));
                            arguments.push(&JsValue::from_str(proposal_role.read().as_str()));
                        }
                        invoke_platform(
                            callback,
                            &arguments,
                            set_platform_pending,
                            primary_error.clone(),
                        );
                    }
                >{move || session.read().primary_label}</button>
            </div>
            <div
                class="lab-note"
                class:hs-warn=move || session.read().status_is_error
                role="status"
                aria-live="polite"
            >{move || session.read().status_label.clone()}</div>
            <div class="lab-note">
                "Replica is safe by default. Select admission authority on exactly one peer that may promote raw Blender edits. Relay URL and bearer token remain runtime-only browser resources."
            </div>
        </div>
    }
}

fn invoke_platform(
    callback: &Function,
    arguments: &Array,
    set_pending: WriteSignal<bool>,
    on_error: SendWrapper<Function>,
) {
    let returned = match callback.apply(&JsValue::UNDEFINED, arguments) {
        Ok(returned) => returned,
        Err(error) => {
            emit_js_error(&on_error, error);
            return;
        }
    };
    set_pending.set(true);
    let promise = Promise::resolve(&returned);
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = JsFuture::from(promise).await {
            emit_js_error(&on_error, error);
        }
        set_pending.set(false);
    });
}

fn emit_js_error(callback: &Function, error: JsValue) {
    let message = error.as_string().unwrap_or_else(|| format!("{error:?}"));
    emit_error(callback, &message);
}

fn emit_error(callback: &Function, message: &str) {
    let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(message));
}
