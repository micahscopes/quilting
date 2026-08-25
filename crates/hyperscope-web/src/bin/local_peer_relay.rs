use hyperscope_web::local_peer_relay::{
    LocalPeerRelay, RelayError, DEFAULT_POLL_LIMIT, DEFAULT_RELAY_CAPACITY, MAX_FRAME_BYTES,
};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::env;
use std::io::Read;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::thread;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use uuid::Uuid;

const DEFAULT_BIND: &str = "127.0.0.1:42117";
const DEFAULT_WORKERS: usize = 4;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hyperscope local peer relay: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "--help" | "-h"))
    {
        println!("{}", usage());
        return Ok(());
    }
    let config = Arc::new(Config::parse(arguments.into_iter())?);
    validate_bind(config.bind, config.allow_non_loopback)?;
    let server = Arc::new(Server::http(config.bind).map_err(|error| error.to_string())?);
    let generation = Uuid::new_v4().to_string();
    let relay = Arc::new(Mutex::new(
        LocalPeerRelay::new(generation.clone(), config.capacity)
            .map_err(|error| error.to_string())?,
    ));

    println!(
        "Hyperscope local peer relay listening on http://{}",
        server.server_addr()
    );
    println!("generation: {generation}");
    println!("bearer token: {}", config.token);
    println!(
        "allowed browser origins: {}",
        config
            .origins
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("delivery only: no persistence, repair, or projection authority");

    let mut workers = Vec::with_capacity(config.workers);
    for _ in 0..config.workers {
        let server = Arc::clone(&server);
        let relay = Arc::clone(&relay);
        let config = Arc::clone(&config);
        workers.push(thread::spawn(move || loop {
            match server.recv() {
                Ok(request) => handle_request(request, &relay, &config),
                Err(error) => {
                    eprintln!("local peer relay receive error: {error}");
                    break;
                }
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| "local peer relay worker panicked".to_owned())?;
    }
    Ok(())
}

#[derive(Debug)]
struct Config {
    bind: SocketAddr,
    capacity: usize,
    workers: usize,
    token: String,
    origins: BTreeSet<String>,
    allow_non_loopback: bool,
}

impl Config {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut bind = DEFAULT_BIND
            .parse::<SocketAddr>()
            .expect("the default relay address is valid");
        let mut capacity = DEFAULT_RELAY_CAPACITY;
        let mut workers = DEFAULT_WORKERS;
        let mut token = Uuid::new_v4().to_string();
        let mut origins = [
            "http://localhost:8888".to_owned(),
            "http://127.0.0.1:8888".to_owned(),
            "http://10.0.0.1:8888".to_owned(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut allow_non_loopback = false;
        let mut arguments = arguments;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--bind" => {
                    bind = next_value(&mut arguments, "--bind")?
                        .parse()
                        .map_err(|error| format!("invalid --bind address: {error}"))?;
                }
                "--capacity" => {
                    capacity =
                        parse_positive(&next_value(&mut arguments, "--capacity")?, "--capacity")?;
                }
                "--workers" => {
                    workers =
                        parse_positive(&next_value(&mut arguments, "--workers")?, "--workers")?;
                    if workers > 64 {
                        return Err("--workers must be at most 64".to_owned());
                    }
                }
                "--token" => {
                    token = next_value(&mut arguments, "--token")?;
                    validate_token(&token)?;
                }
                "--origin" => {
                    let origin = next_value(&mut arguments, "--origin")?;
                    validate_origin(&origin)?;
                    origins.insert(origin);
                }
                "--allow-non-loopback" => allow_non_loopback = true,
                _ => return Err(format!("unknown argument {argument:?}\n{}", usage())),
            }
        }
        validate_token(&token)?;
        Ok(Self {
            bind,
            capacity,
            workers,
            token,
            origins,
            allow_non_loopback,
        })
    }
}

fn next_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{flag} must be a positive integer"))
}

fn validate_token(token: &str) -> Result<(), String> {
    if token.is_empty()
        || token.len() > 256
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~'))
    {
        return Err("--token must be 1..256 URL-safe ASCII characters".to_owned());
    }
    Ok(())
}

fn validate_bind(bind: SocketAddr, allow_non_loopback: bool) -> Result<(), String> {
    if !bind.ip().is_loopback() && !allow_non_loopback {
        return Err(format!(
            "{bind} is not loopback; pass --allow-non-loopback only on a trusted network"
        ));
    }
    Ok(())
}

fn validate_origin(origin: &str) -> Result<(), String> {
    let authority = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    if origin.len() > 512
        || authority.is_none_or(|authority| {
            authority.is_empty()
                || authority.contains(['/', '?', '#', '@'])
                || authority.chars().any(char::is_whitespace)
        })
        || origin.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err("--origin must be one HTTP(S) origin without a path or credentials".to_owned());
    }
    Ok(())
}

fn usage() -> String {
    format!(
        "usage: hyperscope-local-peer-relay [--bind {DEFAULT_BIND}] \
         [--capacity {DEFAULT_RELAY_CAPACITY}] [--workers {DEFAULT_WORKERS}] \
         [--token TOKEN] [--origin ORIGIN] [--allow-non-loopback]"
    )
}

fn handle_request(mut request: Request, relay: &Arc<Mutex<LocalPeerRelay>>, config: &Config) {
    let origin = request_header(&request, "Origin");
    if origin
        .as_ref()
        .is_some_and(|origin| !config.origins.contains(origin))
    {
        respond_error(request, 403, "browser origin is not allowed", None);
        return;
    }
    let cors_origin = origin.as_deref();
    if request.method() == &Method::Options {
        respond_options(request, cors_origin);
        return;
    }
    let authorized = request_header(&request, "Authorization")
        .is_some_and(|value| value == format!("Bearer {}", config.token));
    if !authorized {
        respond_error(request, 401, "relay bearer token is required", cors_origin);
        return;
    }

    let url = request.url().to_owned();
    let (path, query) = url.split_once('?').unwrap_or((url.as_str(), ""));
    match (request.method(), path) {
        (&Method::Get, "/v1/health") => {
            let relay = match relay.lock() {
                Ok(relay) => relay,
                Err(_) => {
                    respond_error(request, 500, "relay state lock is poisoned", cors_origin);
                    return;
                }
            };
            respond_json(
                request,
                200,
                &json!({
                    "generation": relay.generation(),
                    "latestCursor": relay.latest_cursor().to_string(),
                    "retainedFrames": relay.len(),
                    "durable": false,
                }),
                cors_origin,
            );
        }
        (&Method::Get, "/v1/frames") => {
            let after = match query_integer(query, "after", 0) {
                Ok(value) => value,
                Err(error) => {
                    respond_error(request, 400, &error, cors_origin);
                    return;
                }
            };
            let limit =
                match query_integer(query, "limit", DEFAULT_POLL_LIMIT as u64).and_then(|value| {
                    usize::try_from(value).map_err(|_| "limit is too large".to_owned())
                }) {
                    Ok(value) => value,
                    Err(error) => {
                        respond_error(request, 400, &error, cors_origin);
                        return;
                    }
                };
            let batch = match relay
                .lock()
                .map_err(|_| "relay state lock is poisoned".to_owned())
                .and_then(|relay| relay.poll(after, limit).map_err(|error| error.to_string()))
            {
                Ok(batch) => batch,
                Err(error) => {
                    let status = if error.contains("poll limit") {
                        400
                    } else {
                        500
                    };
                    respond_error(request, status, &error, cors_origin);
                    return;
                }
            };
            respond_json(request, 200, &batch, cors_origin);
        }
        (&Method::Post, "/v1/frame") => {
            if !request_header(&request, "Content-Type")
                .is_some_and(|value| value.to_ascii_lowercase().starts_with("application/json"))
            {
                respond_error(
                    request,
                    415,
                    "Content-Type must be application/json",
                    cors_origin,
                );
                return;
            }
            if request
                .body_length()
                .is_some_and(|length| length > MAX_FRAME_BYTES)
            {
                respond_error(
                    request,
                    413,
                    "relay frame exceeds the byte limit",
                    cors_origin,
                );
                return;
            }
            let mut body = Vec::new();
            if let Err(error) = request
                .as_reader()
                .take((MAX_FRAME_BYTES + 1) as u64)
                .read_to_end(&mut body)
            {
                respond_error(
                    request,
                    400,
                    &format!("could not read request body: {error}"),
                    cors_origin,
                );
                return;
            }
            if body.len() > MAX_FRAME_BYTES {
                respond_error(
                    request,
                    413,
                    "relay frame exceeds the byte limit",
                    cors_origin,
                );
                return;
            }
            let body = match String::from_utf8(body) {
                Ok(body) => body,
                Err(_) => {
                    respond_error(request, 400, "relay frame must be UTF-8 JSON", cors_origin);
                    return;
                }
            };
            let (generation, cursor) = match relay.lock() {
                Ok(mut relay) => match relay.append_json(&body) {
                    Ok(cursor) => (relay.generation().to_owned(), cursor),
                    Err(error) => {
                        let status = match error {
                            RelayError::FrameTooLarge { .. } => 413,
                            RelayError::InvalidJson(_) => 400,
                            _ => 500,
                        };
                        respond_error(request, status, &error.to_string(), cors_origin);
                        return;
                    }
                },
                Err(_) => {
                    respond_error(request, 500, "relay state lock is poisoned", cors_origin);
                    return;
                }
            };
            respond_json(
                request,
                202,
                &json!({
                    "generation": generation,
                    "cursor": cursor.to_string(),
                }),
                cors_origin,
            );
        }
        (_, "/v1/health" | "/v1/frames" | "/v1/frame") => {
            respond_error(request, 405, "method is not allowed", cors_origin);
        }
        _ => respond_error(request, 404, "route does not exist", cors_origin),
    }
}

fn request_header(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_owned())
}

fn query_integer(query: &str, name: &str, default: u64) -> Result<u64, String> {
    let Some(value) = query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == name).then_some(value)
    }) else {
        return Ok(default);
    };
    value
        .parse::<u64>()
        .map_err(|_| format!("query parameter {name} must be an unsigned integer"))
}

fn respond_options(request: Request, cors_origin: Option<&str>) {
    let mut response = Response::from_string("").with_status_code(StatusCode(204));
    add_common_headers(&mut response, cors_origin);
    response.add_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"));
    response.add_header(header(
        "Access-Control-Allow-Headers",
        "Authorization, Content-Type",
    ));
    response.add_header(header("Access-Control-Max-Age", "600"));
    if let Err(error) = request.respond(response) {
        eprintln!("local peer relay response error: {error}");
    }
}

fn respond_error(request: Request, status: u16, message: &str, cors_origin: Option<&str>) {
    respond_json(request, status, &json!({ "error": message }), cors_origin);
}

fn respond_json(request: Request, status: u16, value: &impl Serialize, cors_origin: Option<&str>) {
    let body = serde_json::to_string(value)
        .unwrap_or_else(|_| r#"{"error":"response serialization failed"}"#.to_owned());
    let mut response = Response::from_string(body).with_status_code(StatusCode(status));
    response.add_header(header("Content-Type", "application/json; charset=utf-8"));
    add_common_headers(&mut response, cors_origin);
    if let Err(error) = request.respond(response) {
        eprintln!("local peer relay response error: {error}");
    }
}

fn add_common_headers<R: Read>(response: &mut Response<R>, cors_origin: Option<&str>) {
    response.add_header(header("Cache-Control", "no-store"));
    response.add_header(header("X-Content-Type-Options", "nosniff"));
    if let Some(origin) = cors_origin {
        response.add_header(header("Access-Control-Allow-Origin", origin));
        response.add_header(header("Vary", "Origin"));
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes())
        .expect("validated response header must be ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_loopback_authenticated_and_bounded() {
        let config = Config::parse(std::iter::empty()).unwrap();
        assert!(config.bind.ip().is_loopback());
        assert_eq!(config.capacity, DEFAULT_RELAY_CAPACITY);
        assert_eq!(config.workers, DEFAULT_WORKERS);
        validate_token(&config.token).unwrap();
        assert!(config.origins.contains("http://localhost:8888"));
    }

    #[test]
    fn unsafe_cli_values_are_rejected_before_listening() {
        assert!(Config::parse(["--token".to_owned(), "bad token".to_owned()].into_iter()).is_err());
        assert!(Config::parse(
            [
                "--origin".to_owned(),
                "http://localhost:8888/path".to_owned()
            ]
            .into_iter(),
        )
        .is_err());
        assert!(Config::parse(["--workers".to_owned(), "65".to_owned()].into_iter()).is_err());
        assert!(validate_bind("0.0.0.0:42117".parse().unwrap(), false).is_err());
        validate_bind("0.0.0.0:42117".parse().unwrap(), true).unwrap();
    }

    #[test]
    fn poll_query_accepts_only_unsigned_decimal_values() {
        assert_eq!(query_integer("after=7&limit=4", "after", 0).unwrap(), 7);
        assert_eq!(query_integer("", "after", 3).unwrap(), 3);
        assert!(query_integer("after=-1", "after", 0).is_err());
        assert!(query_integer("after=1.5", "after", 0).is_err());
    }
}
