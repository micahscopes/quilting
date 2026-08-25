//! Read-only asset-credit projection for browser views.
//!
//! This module deliberately receives committed [`hyperscope_app`] read
//! models. It does not parse glTF, mutate application state, or infer a legal
//! conclusion from authored metadata.

use hyperscape_protocol::AssetId;
use hyperscope_app::{AssetMetadata, AssetReadModel, AssetStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetCredit {
    pub asset_id: AssetId,
    pub uri: String,
    pub display_name: String,
    pub metadata: AssetMetadata,
}

impl AssetCredit {
    fn from_ready_asset(asset: &AssetReadModel) -> Option<Self> {
        let AssetStatus::Ready { metadata, .. } = &asset.status else {
            return None;
        };
        if metadata.is_empty() {
            return None;
        }
        Some(Self {
            asset_id: asset.descriptor.id,
            uri: asset.descriptor.uri.clone(),
            display_name: metadata
                .title
                .clone()
                .unwrap_or_else(|| asset_basename(&asset.descriptor.uri).to_owned()),
            metadata: metadata.clone(),
        })
    }
}

/// Project sorted AppStore assets into the subset that carries authored
/// provenance. Input ordering is retained so the AppStore remains the sole
/// ordering authority for the view.
pub fn project_asset_credits(assets: &[AssetReadModel]) -> Vec<AssetCredit> {
    assets
        .iter()
        .filter_map(AssetCredit::from_ready_asset)
        .collect()
}

/// Only HTTP(S) metadata may become a clickable browser link. Other source
/// strings remain displayable as plain text by the view.
pub fn external_credit_url(source: &str) -> Option<&str> {
    let remainder = source
        .strip_prefix("https://")
        .or_else(|| source.strip_prefix("http://"))?;
    (!remainder.is_empty()
        && !source
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace()))
    .then_some(source)
}

fn asset_basename(uri: &str) -> &str {
    uri.rsplit('/')
        .find(|component| !component.is_empty())
        .unwrap_or(uri)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape_protocol::AssetDescriptor;

    fn asset(id: u128, uri: &str, status: AssetStatus) -> AssetReadModel {
        AssetReadModel {
            descriptor: AssetDescriptor {
                id: AssetId::from_u128(id).unwrap(),
                uri: uri.to_owned(),
                media_type: Some("model/gltf-binary".to_owned()),
                content_digest: None,
            },
            status,
        }
    }

    fn ready(metadata: AssetMetadata) -> AssetStatus {
        AssetStatus::Ready {
            byte_length: 42,
            content_digest: None,
            metadata,
        }
    }

    #[test]
    fn projection_keeps_ready_credited_assets_in_application_order() {
        let assets = vec![
            asset(
                1,
                "/models/credited.glb",
                ready(AssetMetadata {
                    title: Some("Authored title".to_owned()),
                    author: Some("Example Artist".to_owned()),
                    license: Some("CC-BY-4.0".to_owned()),
                    source: Some("https://example.test/model".to_owned()),
                    ..AssetMetadata::default()
                }),
            ),
            asset(
                2,
                "pending.glb",
                AssetStatus::Loading {
                    request_id: hyperscape_protocol::RequestId::from_u128(2).unwrap(),
                },
            ),
            asset(
                3,
                "/models/generated.glb",
                ready(AssetMetadata {
                    generator: Some("Example exporter".to_owned()),
                    ..AssetMetadata::default()
                }),
            ),
            asset(4, "empty.glb", ready(AssetMetadata::default())),
        ];

        let credits = project_asset_credits(&assets);
        assert_eq!(credits.len(), 2);
        assert_eq!(credits[0].display_name, "Authored title");
        assert_eq!(
            credits[0].metadata.author.as_deref(),
            Some("Example Artist")
        );
        assert_eq!(credits[1].display_name, "generated.glb");
        assert_eq!(
            credits[1].metadata.generator.as_deref(),
            Some("Example exporter")
        );
    }

    #[test]
    fn source_links_admit_only_nonempty_http_urls_without_whitespace() {
        assert_eq!(
            external_credit_url("https://example.test/model"),
            Some("https://example.test/model")
        );
        assert_eq!(
            external_credit_url("http://localhost/model"),
            Some("http://localhost/model")
        );
        for source in [
            "javascript:alert(1)",
            "file:///tmp/model.glb",
            "https://",
            "https://example.test/bad path",
            "",
        ] {
            assert_eq!(external_credit_url(source), None, "accepted {source:?}");
        }
    }
}
