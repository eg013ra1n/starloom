//! The command's logs (P2-R19): a `tracing` registry with a stderr layer filtered by `-v`, and a JSON layer into
//! `<work>/logs/<UTC stamp>-<command>.jsonl` that always takes `debug` for starloom and its two engines.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use tracing_subscriber::filter::{EnvFilter, LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// The folder of the logs under the working folder.
pub const LOGS_DIR: &str = "logs";

/// The stderr filter for `-v` given `verbose` times: `warn`, `info`, `debug`.
pub fn stderr_level(verbose: u8) -> &'static str {
    match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    }
}

/// The stderr filter directives: the level, and the two targets whose events are lines already printed on stderr (a
/// stage's user warning, which its progress sink prints, and the command's own `error:`, `note:` and `warning:`).
fn stderr_filter(verbose: u8) -> String {
    format!(
        "{},{}=off,{}=off",
        stderr_level(verbose),
        crate::pipeline::USER_WARNING_TARGET,
        crate::cli::out::PRINTED_TARGET
    )
}

/// The log file name for a command started at `started`: `<%Y%m%dT%H%M%S%.3fZ>-<command>.jsonl`, with `-<n>` before the
/// extension for the n-th attempt after a clash (`n` 2 and up).
fn file_name(started: DateTime<Utc>, command: &str, attempt: u32) -> String {
    let stamp = started.format("%Y%m%dT%H%M%S%.3fZ");
    if attempt <= 1 {
        format!("{stamp}-{command}.jsonl")
    } else {
        format!("{stamp}-{command}-{attempt}.jsonl")
    }
}

/// Creates the command's log file in `<work>/logs/` (made when missing), opened with `create_new`: a clash with another
/// command started in the same millisecond adds `-2`, `-3`, ... Returns the open file and its absolute path.
pub fn open_log(work_dir: &Path, command: &str, started: DateTime<Utc>) -> Result<(File, PathBuf)> {
    let dir = work_dir.join(LOGS_DIR);
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    for attempt in 1..=1000 {
        let path = dir.join(file_name(started, command, attempt));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((file, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).with_context(|| format!("cannot create the log file {}", path.display())),
        }
    }
    bail!("cannot find a free log file name in {}", dir.display())
}

/// Installs the process's `tracing` subscriber: a compact `fmt` layer on stderr (`EnvFilter` from `-v`), and, when
/// `file` is given, a JSON layer into it at `debug` for the targets `starloom`, `astroimage` and `solvemyastro`. A
/// second call in the same process leaves the first subscriber in place.
pub fn init(verbose: u8, file: Option<File>) {
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_target(false)
        .without_time()
        .with_filter(EnvFilter::new(stderr_filter(verbose)));
    let json = file.map(|f| {
        tracing_subscriber::fmt::layer()
            .json()
            .with_writer(Mutex::new(f))
            .with_filter(
                Targets::new()
                    .with_target("starloom", LevelFilter::DEBUG)
                    .with_target("astroimage", LevelFilter::DEBUG)
                    .with_target("solvemyastro", LevelFilter::DEBUG),
            )
    });
    let _ = tracing_subscriber::registry().with(stderr).with(json).try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn the_name_has_milliseconds_and_a_clash_adds_a_counter() {
        let t = Utc.with_ymd_and_hms(2026, 11, 25, 10, 9, 8).unwrap() + chrono::Duration::milliseconds(45);
        assert_eq!(file_name(t, "measure", 1), "20261125T100908.045Z-measure.jsonl");
        assert_eq!(file_name(t, "measure", 2), "20261125T100908.045Z-measure-2.jsonl");
        assert_eq!(file_name(t, "measure", 3), "20261125T100908.045Z-measure-3.jsonl");
    }

    #[test]
    fn two_commands_in_one_millisecond_get_two_files() {
        let dir = tempfile::tempdir().unwrap();
        let t = Utc::now();
        let (_a, pa) = open_log(dir.path(), "stack", t).unwrap();
        let (_b, pb) = open_log(dir.path(), "stack", t).unwrap();
        assert_ne!(pa, pb);
        assert!(pb.file_name().unwrap().to_string_lossy().ends_with("-stack-2.jsonl"));
    }

    #[test]
    fn stderr_leaves_out_the_lines_it_already_printed() {
        let filter = stderr_filter(1);
        assert!(filter.starts_with("info,"), "{filter}");
        assert!(filter.contains("starloom::user_warning=off") && filter.contains("starloom::cli::printed=off"), "{filter}");
        assert!(EnvFilter::try_new(&filter).is_ok());
    }

    #[test]
    fn the_stderr_level_follows_v() {
        assert_eq!([0, 1, 2, 5].map(stderr_level), ["warn", "info", "debug", "debug"]);
    }
}
