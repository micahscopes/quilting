//! Filesystem-only release preflight for an offline Hyperscope bundle.
//!
//! This deliberately reuses [`crate::Presentation`] validation. It does not
//! start a browser or claim that WebGL2/WebHID runtime capabilities are
//! available; those checks belong to the rehearsal documented in the runbook.

use crate::Presentation;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

const REQUIRED_RUNTIME_FILES: &[&str] = &[
    "index.html",
    "hyperscope_worker.js",
    "spacemouse.mjs",
    "hyperscope_focus.mjs",
    "pkg/quilting_atlas_wasm.js",
    "pkg/quilting_atlas_wasm_bg.wasm",
    "pkg/quilting_worker_wasm.js",
    "pkg/quilting_worker_wasm_bg.wasm",
    "pkg/quilting_wasm.js",
    "pkg/quilting_wasm_bg.wasm",
    "envmaps/rosendal_plains_1_1k.hdr",
    "envmaps/rogland_clear_night_2k.hdr",
    "envmaps/ticknock_04_1k.hdr",
    "ant.glb",
    "ASSET_ATTRIBUTION.md",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];

const BUILD_INPUT_ROOT_FILES: &[&str] = &[
    "Cargo.lock",
    "Cargo.toml",
    "Trunk.toml",
    "hyperscope.html",
    "hyperscope_worker.js",
    "hyperscope_focus.mjs",
    "spacemouse.mjs",
    "horse.glb",
    "ant.glb",
    "examples/hyperscape-blender-demo.glb",
    "examples/hacker-night.presentation.json",
    "ASSET_ATTRIBUTION.md",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];
const BUILD_INPUT_ROOT_DIRS: &[&str] = &["crates", "envmaps"];
const BUILD_RECEIPT_VERSION: &str = "hyperscope-build-inputs/0.1";
const BUILD_FINGERPRINT_ALGORITHM: &str = "fnv1a-128-path-length-content";
const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

const RETIRED_DISTRIBUTION_FILES: &[&str] = &[
    "matcaps/aqua.png",
    "matcaps/citric-acid.png",
    "matcaps/golden-soft.png",
    "matcaps/soft-studio.png",
];

const HORSE_RESTRICTED_WARNING: &str = "horse.glb is Mirada's ROME model under CC BY-NC-SA 3.0; explicitly choose --distribution-policy noncommercial-mixed or replace/exclude it before a permissive-only or commercial-use release";
const HORSE_NONCOMMERCIAL_NOTE: &str = "distribution policy explicitly admits horse.glb under CC BY-NC-SA 3.0; retain its Mirada/ROME attribution and license notice, keep the bundle noncommercial, and apply ShareAlike to adaptations";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistributionPolicy {
    #[default]
    PermissiveOnly,
    NoncommercialMixed,
}

impl DistributionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PermissiveOnly => "permissive-only",
            Self::NoncommercialMixed => "noncommercial-mixed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OfflinePreflightOptions {
    pub source_root: PathBuf,
    pub source_manifest: PathBuf,
    pub dist_dir: PathBuf,
    pub distribution_policy: DistributionPolicy,
}

impl Default for OfflinePreflightOptions {
    fn default() -> Self {
        Self {
            source_root: PathBuf::from("."),
            source_manifest: PathBuf::from("examples/hacker-night.presentation.json"),
            dist_dir: PathBuf::from("dist"),
            distribution_policy: DistributionPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BundleFileCheck {
    pub path: String,
    pub bytes: u64,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OfflinePreflightReport {
    pub ok: bool,
    pub source_manifest: String,
    pub dist_dir: String,
    pub distribution_policy: DistributionPolicy,
    pub presentation_title: Option<String>,
    pub cue_count: usize,
    pub asset_count: usize,
    pub essential_bytes: u64,
    pub source_build_fingerprint: Option<String>,
    pub bundle_build_fingerprint: Option<String>,
    pub files: Vec<BundleFileCheck>,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
    pub errors: Vec<String>,
}

impl OfflinePreflightReport {
    fn new(options: &OfflinePreflightOptions) -> Self {
        Self {
            ok: false,
            source_manifest: options.source_manifest.display().to_string(),
            dist_dir: options.dist_dir.display().to_string(),
            distribution_policy: options.distribution_policy,
            presentation_title: None,
            cue_count: 0,
            asset_count: 0,
            essential_bytes: 0,
            source_build_fingerprint: None,
            bundle_build_fingerprint: None,
            files: Vec::new(),
            warnings: Vec::new(),
            notes: vec![
                "Filesystem preflight cannot certify WebGL2, animation, picking, audio/video, or optional WebHID capabilities; rehearse those in the target browser".to_owned(),
            ],
            errors: Vec::new(),
        }
    }

    fn record_file(&mut self, relative: &Path, bytes: u64, kind: &str) {
        self.essential_bytes = self.essential_bytes.saturating_add(bytes);
        self.files.push(BundleFileCheck {
            path: path_for_report(relative),
            bytes,
            kind: kind.to_owned(),
        });
    }
}

/// Deterministic receipt for the exact source inputs consumed by the release
/// build. This detects stale local bundles; it is not a cryptographic signature
/// and makes no tamper-resistance claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HyperscopeBuildReceipt {
    pub version: String,
    pub algorithm: String,
    pub fingerprint: String,
    pub files: usize,
    pub bytes: u64,
}

/// Fingerprint the checked-in runtime sources and copied release assets.
pub fn hyperscope_build_receipt(source_root: &Path) -> Result<HyperscopeBuildReceipt, String> {
    let mut paths = BUILD_INPUT_ROOT_FILES
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    for directory in BUILD_INPUT_ROOT_DIRS {
        collect_build_input_files(source_root, Path::new(directory), &mut paths)?;
    }
    paths.sort();
    paths.dedup();

    let mut fingerprint = FNV1A_128_OFFSET;
    let mut bytes = 0_u64;
    update_fingerprint(&mut fingerprint, BUILD_RECEIPT_VERSION.as_bytes());
    for relative in &paths {
        let path = source_root.join(relative);
        let content = fs::read(&path)
            .map_err(|error| format!("cannot read build input {}: {error}", path.display()))?;
        let relative = path_for_report(relative);
        update_build_input_fingerprint(&mut fingerprint, &relative, &content);
        bytes = bytes.saturating_add(content.len() as u64);
    }
    Ok(HyperscopeBuildReceipt {
        version: BUILD_RECEIPT_VERSION.to_owned(),
        algorithm: BUILD_FINGERPRINT_ALGORITHM.to_owned(),
        fingerprint: format!("{fingerprint:032x}"),
        files: paths.len(),
        bytes,
    })
}

/// Write a pretty, deterministic receipt for Trunk to copy into the bundle.
pub fn write_hyperscope_build_receipt(
    source_root: &Path,
    output: &Path,
) -> Result<HyperscopeBuildReceipt, String> {
    let receipt = hyperscope_build_receipt(source_root)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create build receipt directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut encoded = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("cannot serialize build receipt: {error}"))?;
    encoded.push('\n');
    fs::write(output, encoded)
        .map_err(|error| format!("cannot write build receipt {}: {error}", output.display()))?;
    Ok(receipt)
}

fn collect_build_input_files(
    source_root: &Path,
    relative: &Path,
    paths: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let directory = source_root.join(relative);
    let mut entries = fs::read_dir(&directory)
        .map_err(|error| {
            format!(
                "cannot read build input directory {}: {error}",
                directory.display()
            )
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            format!(
                "cannot enumerate build input directory {}: {error}",
                directory.display()
            )
        })?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type().map_err(|error| {
            format!(
                "cannot inspect build input {}: {error}",
                entry.path().display()
            )
        })?;
        let child = relative.join(entry.file_name());
        if file_type.is_dir() {
            collect_build_input_files(source_root, &child, paths)?;
        } else if file_type.is_file() {
            paths.push(child);
        } else {
            return Err(format!(
                "build input {} is neither a regular file nor directory",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

fn update_fingerprint(fingerprint: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *fingerprint ^= u128::from(*byte);
        *fingerprint = fingerprint.wrapping_mul(FNV1A_128_PRIME);
    }
}

fn update_build_input_fingerprint(fingerprint: &mut u128, path: &str, content: &[u8]) {
    update_fingerprint(fingerprint, &(path.len() as u64).to_le_bytes());
    update_fingerprint(fingerprint, path.as_bytes());
    update_fingerprint(fingerprint, &(content.len() as u64).to_le_bytes());
    update_fingerprint(fingerprint, content);
}

/// Validate a completed Trunk distribution without starting a browser.
pub fn run_offline_preflight(options: &OfflinePreflightOptions) -> OfflinePreflightReport {
    let mut report = OfflinePreflightReport::new(options);
    let source_bytes = match fs::read(&options.source_manifest) {
        Ok(bytes) => bytes,
        Err(error) => {
            report.errors.push(format!(
                "cannot read source manifest {}: {error}",
                options.source_manifest.display()
            ));
            return report;
        }
    };
    let presentation = match std::str::from_utf8(&source_bytes)
        .map_err(|error| error.to_string())
        .and_then(|json| Presentation::from_json(json).map_err(|error| error.to_string()))
    {
        Ok(presentation) => presentation,
        Err(error) => {
            report.errors.push(format!(
                "source presentation {} is invalid: {error}",
                options.source_manifest.display()
            ));
            return report;
        }
    };
    report.presentation_title = Some(presentation.title.clone());
    report.cue_count = presentation.cues.len();
    report.asset_count = presentation.assets.len();

    if !options.dist_dir.is_dir() {
        report.errors.push(format!(
            "distribution directory {} does not exist; run `trunk build --release` first",
            options.dist_dir.display()
        ));
        return report;
    }

    let mut checked = BTreeSet::new();
    let copied_manifest = match options.source_manifest.file_name() {
        Some(name) => PathBuf::from(name),
        None => {
            report.errors.push(format!(
                "source manifest {} has no filename",
                options.source_manifest.display()
            ));
            return report;
        }
    };
    let copied_manifest_bytes = check_bundle_file(
        &options.dist_dir,
        &copied_manifest,
        "presentation manifest",
        &mut report,
        &mut checked,
    );
    if let Some(bytes) = copied_manifest_bytes {
        if bytes != source_bytes {
            report.errors.push(format!(
                "{} is stale: it differs from {}",
                options.dist_dir.join(&copied_manifest).display(),
                options.source_manifest.display()
            ));
        }
    }

    check_build_receipt(options, &mut report, &mut checked);

    for relative in REQUIRED_RUNTIME_FILES {
        let relative = Path::new(relative);
        if let Some(bytes) = check_bundle_file(
            &options.dist_dir,
            relative,
            "runtime",
            &mut report,
            &mut checked,
        ) {
            if relative
                .extension()
                .is_some_and(|extension| extension == "glb")
            {
                validate_glb_bytes(relative, &bytes, &mut report);
            }
        }
    }

    for asset in &presentation.assets {
        match local_uri_to_relative_path(&asset.uri) {
            Ok(relative) => {
                if let Some(bytes) = check_bundle_file(
                    &options.dist_dir,
                    &relative,
                    "presentation asset",
                    &mut report,
                    &mut checked,
                ) {
                    if relative
                        .extension()
                        .is_some_and(|extension| extension == "glb")
                    {
                        validate_glb_bytes(&relative, &bytes, &mut report);
                    }
                }
            }
            Err(error) => report.errors.push(format!(
                "presentation asset {:?} has a non-offline URI {:?}: {error}",
                asset.name, asset.uri
            )),
        }
    }

    check_generated_trunk_pair(&options.dist_dir, &mut report, &mut checked);

    let local_glbs = options.dist_dir.join("local-glbs");
    if local_glbs.is_dir() {
        let (count, bytes) = directory_payload(&local_glbs);
        if count > 0 {
            report.warnings.push(format!(
                "dist/local-glbs contains {count} untracked file(s), {} bytes; exclude them from a public archive unless each asset is intentionally licensed",
                bytes
            ));
        }
    }
    let retired_files = RETIRED_DISTRIBUTION_FILES
        .iter()
        .filter(|relative| options.dist_dir.join(relative).is_file())
        .copied()
        .collect::<Vec<_>>();
    if !retired_files.is_empty() {
        report.warnings.push(format!(
            "dist contains retired unlicensed matcap image(s): {}; make a clean build before release",
            retired_files.join(", ")
        ));
    }
    match options.distribution_policy {
        DistributionPolicy::PermissiveOnly => {
            report.warnings.push(HORSE_RESTRICTED_WARNING.to_owned());
        }
        DistributionPolicy::NoncommercialMixed => {
            report.notes.push(HORSE_NONCOMMERCIAL_NOTE.to_owned());
        }
    }

    report
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    report.ok = report.errors.is_empty();
    report
}

fn check_build_receipt(
    options: &OfflinePreflightOptions,
    report: &mut OfflinePreflightReport,
    checked: &mut BTreeSet<PathBuf>,
) {
    let source_receipt = match hyperscope_build_receipt(&options.source_root) {
        Ok(receipt) => {
            report.source_build_fingerprint = Some(receipt.fingerprint.clone());
            receipt
        }
        Err(error) => {
            report
                .errors
                .push(format!("cannot fingerprint current build inputs: {error}"));
            return;
        }
    };
    let relative = Path::new("pkg/hyperscope-build.json");
    let Some(bytes) = check_bundle_file(
        &options.dist_dir,
        relative,
        "build receipt",
        report,
        checked,
    ) else {
        return;
    };
    let bundle_receipt: HyperscopeBuildReceipt = match serde_json::from_slice(&bytes) {
        Ok(receipt) => receipt,
        Err(error) => {
            report.errors.push(format!(
                "bundle build receipt {} is invalid: {error}",
                options.dist_dir.join(relative).display()
            ));
            return;
        }
    };
    report.bundle_build_fingerprint = Some(bundle_receipt.fingerprint.clone());
    if bundle_receipt.version != BUILD_RECEIPT_VERSION
        || bundle_receipt.algorithm != BUILD_FINGERPRINT_ALGORITHM
    {
        report.errors.push(format!(
            "bundle build receipt uses unsupported version {:?} or algorithm {:?}",
            bundle_receipt.version, bundle_receipt.algorithm,
        ));
        return;
    }
    if bundle_receipt != source_receipt {
        report.errors.push(format!(
            "bundle is stale: build fingerprint {} does not match current source {}",
            bundle_receipt.fingerprint, source_receipt.fingerprint,
        ));
    }
}

fn check_bundle_file(
    dist_dir: &Path,
    relative: &Path,
    kind: &str,
    report: &mut OfflinePreflightReport,
    checked: &mut BTreeSet<PathBuf>,
) -> Option<Vec<u8>> {
    if !checked.insert(relative.to_owned()) {
        return fs::read(dist_dir.join(relative)).ok();
    }
    let path = dist_dir.join(relative);
    let canonical_root = fs::canonicalize(dist_dir).ok();
    match (canonical_root, fs::canonicalize(&path)) {
        (Some(root), Ok(resolved)) if !resolved.starts_with(&root) => {
            report.errors.push(format!(
                "required bundle path {} resolves outside {}",
                path.display(),
                dist_dir.display()
            ));
            return None;
        }
        (_, Err(error)) => {
            report.errors.push(format!(
                "required bundle file {} is unavailable: {error}",
                path.display()
            ));
            return None;
        }
        _ => {}
    }
    match fs::read(&path) {
        Ok(bytes) if bytes.is_empty() => {
            report
                .errors
                .push(format!("required bundle file {} is empty", path.display()));
            None
        }
        Ok(bytes) => {
            report.record_file(relative, bytes.len() as u64, kind);
            Some(bytes)
        }
        Err(error) => {
            report.errors.push(format!(
                "cannot read required bundle file {}: {error}",
                path.display()
            ));
            None
        }
    }
}

fn check_generated_trunk_pair(
    dist_dir: &Path,
    report: &mut OfflinePreflightReport,
    checked: &mut BTreeSet<PathBuf>,
) {
    for (suffix, kind) in [(".js", "Trunk bootstrap"), ("_bg.wasm", "Trunk bootstrap")] {
        let matches = fs::read_dir(dist_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                (name.starts_with("trunk-stub-") && name.ends_with(suffix))
                    .then_some(PathBuf::from(name))
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            report.errors.push(format!(
                "expected exactly one generated trunk-stub-*{suffix} file in {}, found {}",
                dist_dir.display(),
                matches.len()
            ));
            continue;
        }
        check_bundle_file(dist_dir, &matches[0], kind, report, checked);
    }
}

fn local_uri_to_relative_path(uri: &str) -> Result<PathBuf, &'static str> {
    let uri = uri.trim();
    if uri.is_empty() {
        return Err("URI is empty");
    }
    if uri.contains("://") || uri.starts_with("//") {
        return Err("network URLs are not offline assets");
    }
    if uri.contains(['?', '#', '\\']) {
        return Err("query, fragment, and backslash components are not supported");
    }
    let path = Path::new(uri.trim_start_matches('/'));
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("path must contain only normal components");
    }
    Ok(path.to_owned())
}

fn validate_glb_bytes(relative: &Path, bytes: &[u8], report: &mut OfflinePreflightReport) {
    if let Err(error) = validate_glb_header(bytes) {
        report.errors.push(format!(
            "bundle asset {} is not a valid glTF 2 GLB: {error}",
            relative.display()
        ));
    }
}

fn validate_glb_header(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 12 {
        return Err("file is shorter than the 12-byte GLB header".to_owned());
    }
    if &bytes[0..4] != b"glTF" {
        return Err("magic is not `glTF`".to_owned());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("four-byte slice"));
    if version != 2 {
        return Err(format!("version is {version}, expected 2"));
    }
    let declared = u32::from_le_bytes(bytes[8..12].try_into().expect("four-byte slice")) as usize;
    if declared != bytes.len() {
        return Err(format!(
            "header declares {declared} bytes but file contains {}",
            bytes.len()
        ));
    }
    Ok(())
}

fn directory_payload(directory: &Path) -> (usize, u64) {
    let mut count = 0;
    let mut bytes = 0_u64;
    let Ok(entries) = fs::read_dir(directory) else {
        return (count, bytes);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let nested = directory_payload(&path);
            count += nested.0;
            bytes = bytes.saturating_add(nested.1);
        } else if entry.file_name() == ".gitkeep" {
            continue;
        } else if let Ok(metadata) = entry.metadata() {
            count += 1;
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    (count, bytes)
}

fn path_for_report(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::{
        local_uri_to_relative_path, update_build_input_fingerprint, validate_glb_header,
        DistributionPolicy, FNV1A_128_OFFSET,
    };
    use std::path::PathBuf;

    #[test]
    fn offline_uri_accepts_root_relative_and_relative_assets() {
        assert_eq!(
            local_uri_to_relative_path("/horse.glb"),
            Ok(PathBuf::from("horse.glb"))
        );
        assert_eq!(
            local_uri_to_relative_path("models/horse.glb"),
            Ok(PathBuf::from("models/horse.glb"))
        );
    }

    #[test]
    fn offline_uri_rejects_escaping_and_network_paths() {
        for uri in [
            "../horse.glb",
            "/models/../horse.glb",
            "https://example.test/horse.glb",
            "//example.test/horse.glb",
            "horse.glb?rev=1",
            "models\\horse.glb",
        ] {
            assert!(local_uri_to_relative_path(uri).is_err(), "accepted {uri:?}");
        }
    }

    #[test]
    fn build_fingerprint_commits_to_paths_lengths_and_content() {
        let fingerprint = |path: &str, content: &[u8]| {
            let mut value = FNV1A_128_OFFSET;
            update_build_input_fingerprint(&mut value, path, content);
            value
        };
        let baseline = fingerprint("crates/example.rs", b"same bytes");
        assert_eq!(baseline, fingerprint("crates/example.rs", b"same bytes"));
        assert_ne!(baseline, fingerprint("crates/renamed.rs", b"same bytes"));
        assert_ne!(baseline, fingerprint("crates/example.rs", b"changed bytes"));
        assert_ne!(
            fingerprint("ab", b"c"),
            fingerprint("a", b"bc"),
            "length prefixes must prevent path/content boundary ambiguity",
        );
    }

    #[test]
    fn glb_header_requires_magic_version_and_exact_length() {
        let mut valid = Vec::from(*b"glTF\x02\0\0\0\x0c\0\0\0");
        assert_eq!(validate_glb_header(&valid), Ok(()));
        valid[0] = b'B';
        assert!(validate_glb_header(&valid).is_err());

        let wrong_length = b"glTF\x02\0\0\0\x10\0\0\0";
        assert!(validate_glb_header(wrong_length).is_err());
    }

    #[test]
    fn distribution_policy_names_are_stable() {
        assert_eq!(DistributionPolicy::default().as_str(), "permissive-only");
        assert_eq!(
            DistributionPolicy::NoncommercialMixed.as_str(),
            "noncommercial-mixed"
        );
    }
}
