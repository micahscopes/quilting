use super::{compact_credit_text, external_credit_url, project_asset_credits, AssetCredit};
use futures_signals::signal::SignalExt as _;
use futures_signals::signal_vec::SignalVecExt as _;
use hyperscope_app::AppStore;
use leptos::mount::mount_to;
use leptos::prelude::*;

/// Mount a permanent Leptos CSR island over the AppStore's read-only asset
/// signal. The reducer remains the only mutation boundary; this task merely
/// projects each committed vector into DOM state.
pub fn mount_asset_credits(parent: web_sys::HtmlElement, store: AppStore) {
    mount_to(parent, move || {
        let (credits, set_credits) = signal(project_asset_credits(&store.asset_snapshot()));
        let updates = store
            .asset_signal_vec()
            .to_signal_cloned()
            .for_each(move |assets| {
                set_credits.set(project_asset_credits(&assets));
                async {}
            });
        wasm_bindgen_futures::spawn_local(updates);
        view! { <AssetCredits credits /> }
    })
    .forget();
}

#[component]
fn AssetCredits(credits: ReadSignal<Vec<AssetCredit>>) -> impl IntoView {
    view! {
        <div class="asset-credit-footer" aria-live="polite">
            {move || {
                credits
                    .get()
                    .into_iter()
                    .filter_map(|credit| compact_credit_text(&credit))
                    .map(|credit| view! { <span>{credit}</span> })
                    .collect_view()
            }}
        </div>
        <details class="hs-diag" id="asset-credits-rust-view">
            <summary>
                "Asset credits "
                {move || {
                    let count = credits.read().len();
                    (count > 0).then(|| format!("({count})"))
                }}
            </summary>
            {move || {
                let credits = credits.get();
                if credits.is_empty() {
                    view! {
                        <div class="asset-credit-note">
                            "No embedded attribution metadata is available for the loaded asset."
                        </div>
                    }
                        .into_any()
                } else {
                    credits
                        .into_iter()
                        .map(|credit| view! { <AssetCreditEntry credit /> })
                        .collect_view()
                        .into_any()
                }
            }}
            <div class="asset-credit-note">
                <a href="ASSET_ATTRIBUTION.md" target="_blank" rel="noreferrer">
                    "Bundled asset attribution and redistribution notes"
                </a>
            </div>
        </details>
    }
}

#[component]
fn AssetCreditEntry(credit: AssetCredit) -> impl IntoView {
    let metadata = credit.metadata;
    let source = metadata.source.clone();
    view! {
        <div class="asset-credit" data-asset-id=credit.asset_id.to_string()>
            <strong>{credit.display_name}</strong>
            {credit_line("Author", metadata.author)}
            {credit_line("License", metadata.license)}
            {credit_line("Copyright", metadata.copyright)}
            {credit_line("Generator", metadata.generator)}
            {source.map(|source| {
                let link = external_credit_url(&source).map(str::to_owned);
                match link {
                    Some(link) => view! {
                        <div class="asset-credit-line"><a href=link target="_blank" rel="noreferrer">"Source"</a></div>
                    }
                        .into_any(),
                    None => view! {
                        <div class="asset-credit-line">{format!("Source: {source}")}</div>
                    }
                        .into_any(),
                }
            })}
        </div>
    }
}

fn credit_line(label: &'static str, value: Option<String>) -> impl IntoView {
    value.map(move |value| {
        view! { <div class="asset-credit-line">{format!("{label}: {value}")}</div> }
    })
}
