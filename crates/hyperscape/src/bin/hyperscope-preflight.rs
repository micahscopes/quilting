use hyperscape::{run_offline_preflight, OfflinePreflightOptions, OfflinePreflightReport};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut options = OfflinePreflightOptions::default();
    let mut json = false;
    let mut strict = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--manifest" => {
                let Some(path) = args.next() else {
                    return usage_error("--manifest requires a path");
                };
                options.source_manifest = PathBuf::from(path);
            }
            "--dist" => {
                let Some(path) = args.next() else {
                    return usage_error("--dist requires a path");
                };
                options.dist_dir = PathBuf::from(path);
            }
            "--json" => json = true,
            "--strict" => strict = true,
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            unknown => return usage_error(&format!("unknown argument {unknown:?}")),
        }
    }

    let report = run_offline_preflight(&options);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(output) => println!("{output}"),
            Err(error) => {
                eprintln!("failed to serialize preflight report: {error}");
                return ExitCode::from(2);
            }
        }
    } else {
        print_human_report(&report, strict);
    }

    if !report.ok || (strict && !report.warnings.is_empty()) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn print_human_report(report: &OfflinePreflightReport, strict: bool) {
    let status = if report.ok && (!strict || report.warnings.is_empty()) {
        "PASS"
    } else {
        "FAIL"
    };
    println!("Hyperscope offline preflight: {status}");
    if let Some(title) = &report.presentation_title {
        println!("Presentation: {title}");
    }
    println!(
        "Manifest: {} cue(s), {} asset(s)",
        report.cue_count, report.asset_count
    );
    println!(
        "Bundle: {} checked file(s), {}",
        report.files.len(),
        format_bytes(report.essential_bytes)
    );
    for warning in &report.warnings {
        println!("WARN: {warning}");
    }
    for note in &report.notes {
        println!("NOTE: {note}");
    }
    for error in &report.errors {
        println!("ERROR: {error}");
    }
    if strict && !report.warnings.is_empty() {
        println!("ERROR: --strict treats distribution warnings as failures");
    }
}

fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.2} MiB", bytes as f64 / MIB)
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!("error: {message}");
    print_usage();
    ExitCode::from(2)
}

fn print_usage() {
    println!(
        "Usage: hyperscope-preflight [--manifest PATH] [--dist PATH] [--json] [--strict]\n\
         Defaults: --manifest examples/hacker-night.presentation.json --dist dist\n\
         --strict also fails on uncleared distribution assets and local GLBs"
    );
}
