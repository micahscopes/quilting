use hyperscape::Presentation;
use hyperscope_app::{
    app_replay_fingerprint, presentation_walkthrough_replay, run_app_replay, AppReplayScript,
    APP_REPLAY_FINGERPRINT_ALGORITHM,
};
use std::env;
use std::fs;
use std::process::ExitCode;

const HACKER_NIGHT_PRESENTATION: &str = hyperscape::HACKER_NIGHT_PRESENTATION_JSON;
const HACKER_NIGHT_GOLDEN: &str =
    include_str!("../../fixtures/hacker-night.replay.fingerprint");
const NAVIGATION_REPLAY: &str = include_str!("../../fixtures/navigation.app-replay.json");
const NAVIGATION_GOLDEN: &str = include_str!("../../fixtures/navigation.replay.fingerprint");
const ORCHESTRATION_REPLAY: &str =
    include_str!("../../fixtures/orchestration.app-replay.json");
const ORCHESTRATION_GOLDEN: &str =
    include_str!("../../fixtures/orchestration.replay.fingerprint");

#[derive(Debug, Default)]
struct Options {
    source: Option<Source>,
    fingerprint_only: bool,
    check: bool,
}

#[derive(Debug)]
enum Source {
    NavigationOracle,
    OrchestrationOracle,
    Script(String),
    Presentation(String),
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hyperscope-replay: {error}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<(), String> {
    let Some(options) = parse_options()? else {
        return Ok(());
    };
    let expected = options.check.then(|| match options.source.as_ref() {
        None => Ok(HACKER_NIGHT_GOLDEN.trim()),
        Some(Source::NavigationOracle) => Ok(NAVIGATION_GOLDEN.trim()),
        Some(Source::OrchestrationOracle) => Ok(ORCHESTRATION_GOLDEN.trim()),
        Some(Source::Script(_) | Source::Presentation(_)) => {
            Err("--check only applies to an embedded oracle".to_owned())
        }
    });
    let expected = expected.transpose()?;
    let script = load_script(options.source)?;
    let trace = run_app_replay(&script).map_err(|error| error.to_string())?;
    let fingerprint = app_replay_fingerprint(&trace).map_err(|error| error.to_string())?;
    let qualified = format!("{APP_REPLAY_FINGERPRINT_ALGORITHM}:{fingerprint}");

    if options.check {
        let expected = expected.expect("check mode selected an embedded oracle");
        if qualified != expected {
            return Err(format!(
                "golden replay mismatch: expected {expected}, observed {qualified}"
            ));
        }
        println!("PASS {qualified}");
    } else if options.fingerprint_only {
        println!("{qualified}");
    } else {
        eprintln!("{qualified}");
        println!(
            "{}",
            serde_json::to_string_pretty(&trace).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn parse_options() -> Result<Option<Options>, String> {
    let mut options = Options::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--script" => set_source(
                &mut options,
                Source::Script(
                    arguments
                        .next()
                        .ok_or_else(|| "--script requires a path".to_owned())?,
                ),
            )?,
            "--presentation" => set_source(
                &mut options,
                Source::Presentation(
                    arguments
                        .next()
                        .ok_or_else(|| "--presentation requires a path".to_owned())?,
                ),
            )?,
            "--navigation" => set_source(&mut options, Source::NavigationOracle)?,
            "--orchestration" => set_source(&mut options, Source::OrchestrationOracle)?,
            "--fingerprint" => options.fingerprint_only = true,
            "--check" => options.check = true,
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            unknown => return Err(format!("unknown argument {unknown:?}; use --help")),
        }
    }
    if options.check && options.fingerprint_only {
        return Err("--check and --fingerprint are mutually exclusive".to_owned());
    }
    Ok(Some(options))
}

fn set_source(options: &mut Options, source: Source) -> Result<(), String> {
    if options.source.is_some() {
        Err(
            "choose at most one replay source: --navigation, --orchestration, --script, or --presentation"
                .to_owned(),
        )
    } else {
        options.source = Some(source);
        Ok(())
    }
}

fn load_script(source: Option<Source>) -> Result<AppReplayScript, String> {
    match source {
        Some(Source::NavigationOracle) => serde_json::from_str(NAVIGATION_REPLAY)
            .map_err(|error| format!("embedded navigation replay is invalid: {error}")),
        Some(Source::OrchestrationOracle) => serde_json::from_str(ORCHESTRATION_REPLAY)
            .map_err(|error| format!("embedded orchestration replay is invalid: {error}")),
        Some(Source::Script(path)) => {
            let json = fs::read_to_string(&path)
                .map_err(|error| format!("could not read replay script {path:?}: {error}"))?;
            serde_json::from_str(&json)
                .map_err(|error| format!("could not parse replay script {path:?}: {error}"))
        }
        Some(Source::Presentation(path)) => {
            let json = fs::read_to_string(&path)
                .map_err(|error| format!("could not read presentation {path:?}: {error}"))?;
            let presentation = Presentation::from_json(&json)
                .map_err(|error| format!("could not parse presentation {path:?}: {error}"))?;
            Ok(presentation_walkthrough_replay(presentation))
        }
        None => Presentation::from_json(HACKER_NIGHT_PRESENTATION)
            .map(presentation_walkthrough_replay)
            .map_err(|error| format!("embedded presentation is invalid: {error}")),
    }
}

fn print_help() {
    println!(
        "Usage: hyperscope-replay [--navigation | --orchestration | --script PATH | --presentation PATH] [--fingerprint | --check]\n\
         \n\
         Replays versioned semantic application events without a browser or renderer.\n\
         With no source, walks the embedded hacker-night presentation.\n\
         \n\
           --navigation         Replay the embedded semantic navigation oracle\n\
           --orchestration      Replay the embedded effects/presence/authored oracle\n\
           --script PATH        Replay a serialized AppReplayScript\n\
           --presentation PATH  Build a complete cue walkthrough from a presentation\n\
           --fingerprint        Print only the qualified deterministic fingerprint\n\
           --check              Check the selected embedded oracle against its golden"
    );
}
