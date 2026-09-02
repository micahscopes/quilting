//! Release-only standards HTML precompiler for Quilting's Fe browser demos.
//!
//! The upstream `fe web` host currently constructs its compiler database with
//! the development profile. This small adapter keeps the standards-based
//! `application/fe` page contract while making the release profile an explicit
//! invariant. It contains no renderer or demo-specific JavaScript.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::{WebBuildOptions, WebBundle};
use fe_html_precompile::{
    precompile_html_with_render_lane, RenderBundleArtifact, RenderShaderArtifact,
    RenderSupportArtifact,
};
use hir::hir_def::HirIngot;
use salsa::Setter;
use url::Url;

fn main() {
    if let Err(error) = run() {
        eprintln!("precompile-quilting-fe-web: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let html_path = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    let output = arguments.next().map(PathBuf::from).ok_or_else(usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }

    let html_path = html_path
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", html_path.display()))?;
    let document_url = Url::from_file_path(&html_path)
        .map_err(|()| format!("cannot represent {} as a file URL", html_path.display()))?;
    let html = fs::read_to_string(&html_path)
        .map_err(|error| format!("cannot read {}: {error}", html_path.display()))?;

    let publication = precompile_html_with_render_lane(
        document_url.as_str(),
        &html,
        fe_codegen::render_runtime_js(),
        load_text,
        compile_release_render_bundle,
    )
    .map_err(|error| error.to_string())?;

    publish(&output, publication.html.as_bytes(), publication.assets)?;
    println!("published release Fe site at {}", output.display());
    Ok(())
}

fn usage() -> String {
    "usage: precompile-quilting-fe-web <index.html> <output-directory>".to_owned()
}

fn load_text(url: &Url) -> Result<String, String> {
    let path = url
        .to_file_path()
        .map_err(|()| format!("unsupported non-file source URL: {url}"))?;
    fs::read_to_string(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn compile_release_render_bundle(
    url: &Url,
    entry: Option<&str>,
) -> Result<Option<RenderBundleArtifact>, String> {
    let Ok(path) = url.to_file_path() else {
        return Ok(None);
    };
    if !path.is_dir() {
        return Ok(None);
    }
    let entry = entry.ok_or_else(|| {
        format!(
            "release render source {} requires data-fe-entry",
            path.display()
        )
    })?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("cannot resolve render ingot {}: {error}", path.display()))?;
    let ingot_url = Url::from_directory_path(&canonical)
        .map_err(|()| format!("cannot represent {} as an ingot URL", canonical.display()))?;

    let mut db = DriverDataBase::default();
    db.compilation_settings()
        .set_profile(&mut db)
        .to("release".into());
    if driver::init_ingot(&mut db, &ingot_url) {
        return Err(format!(
            "render ingot initialization failed for {}",
            canonical.display()
        ));
    }
    let top_mod = db
        .workspace()
        .containing_ingot(&db, ingot_url)
        .ok_or_else(|| format!("no initialized ingot contains {}", canonical.display()))?
        .root_mod(&db);
    let diagnostics = db.run_on_top_mod(top_mod).format_diags(&db);
    if !diagnostics.is_empty() {
        return Err(format!(
            "release render diagnostics for {}:\n{diagnostics}",
            canonical.display()
        ));
    }

    let source_id = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("quilting-fe-render")
        .to_owned();
    let bundle = WebBundle::compile(
        &db,
        top_mod,
        WebBuildOptions::render(entry, Some(source_id)),
    )
    .map_err(|error| error.to_string())?;
    let manifest_json = bundle.manifest_json().map_err(|error| error.to_string())?;
    let materialized = bundle
        .materialized_files()
        .map_err(|error| error.to_string())?;
    let support_files = materialized
        .iter()
        .filter(|file| {
            file.path() == "interface.js"
                || file.path() == "interface.d.ts"
                || file.path().starts_with("runtime/")
        })
        .map(|file| RenderSupportArtifact {
            path: file.path().to_owned(),
            bytes: file.bytes().to_vec(),
        })
        .collect();
    let scoped_task_files = materialized
        .iter()
        .filter_map(|file| {
            file.path()
                .strip_prefix("tasks/")
                .map(|path| RenderSupportArtifact {
                    path: path.to_owned(),
                    bytes: file.bytes().to_vec(),
                })
        })
        .collect();
    let pass_wgsl = bundle
        .pass_wgsl
        .into_iter()
        .map(|shader| RenderShaderArtifact {
            path: shader.path,
            bytes: shader.source.into_bytes(),
        })
        .collect();

    Ok(Some(RenderBundleArtifact {
        wasm: (!bundle.wasm.is_empty()).then_some(bundle.wasm),
        wgsl: bundle.wgsl.into_bytes(),
        pass_wgsl,
        support_files,
        scoped_task_files,
        manifest_json,
        source_dependencies: None,
    }))
}

fn publish(
    output: &Path,
    html: &[u8],
    assets: std::collections::BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    fs::create_dir_all(output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    for (relative, bytes) in assets {
        let destination = output.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        if destination.exists() {
            let existing = fs::read(&destination)
                .map_err(|error| format!("cannot read {}: {error}", destination.display()))?;
            if existing != bytes {
                return Err(format!(
                    "content-addressed asset collision at {}",
                    destination.display()
                ));
            }
        } else {
            fs::write(&destination, bytes)
                .map_err(|error| format!("cannot write {}: {error}", destination.display()))?;
        }
    }

    let mut staged = tempfile::NamedTempFile::new_in(output)
        .map_err(|error| format!("cannot stage index.html in {}: {error}", output.display()))?;
    staged
        .write_all(html)
        .and_then(|()| staged.as_file().sync_all())
        .map_err(|error| format!("cannot stage index.html: {error}"))?;
    staged
        .persist(output.join("index.html"))
        .map_err(|error| format!("cannot publish index.html: {}", error.error))?;
    Ok(())
}
