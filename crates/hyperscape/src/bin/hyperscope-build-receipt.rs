use hyperscape::write_hyperscope_build_receipt;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments = env::args().skip(1);
    let output = arguments
        .next()
        .map_or_else(|| PathBuf::from("pkg/hyperscope-build.json"), PathBuf::from);
    if arguments.next().is_some() {
        eprintln!("usage: hyperscope-build-receipt [OUTPUT]");
        return ExitCode::from(2);
    }
    match write_hyperscope_build_receipt(Path::new("."), &output) {
        Ok(receipt) => {
            println!(
                "Hyperscope build inputs: {} files, {} bytes, {}",
                receipt.files, receipt.bytes, receipt.fingerprint,
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
