//! The command's stdout and stderr (P2-R13): the one place in starloom that writes to either.
//!
//! stdout carries command products only (`facts`, `config show`); progress, warnings, notes and the summary line go to
//! stderr. Both go through `std::io::Write` handles, and a closed pipe is not an error (`starloom facts | head`).

use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::pipeline::Progress;

/// Writes one line of a command's product to stdout.
pub fn product_line(text: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{text}");
}

/// Writes one line to stderr.
pub fn stderr_line(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{text}");
}

/// The target of the events that record a line the command has just printed on stderr (an `error:`, a `note:`, a
/// `warning:`). The stderr layer drops this target, so the line shows once there and the log file holds the event.
pub const PRINTED_TARGET: &str = "starloom::cli::printed";

/// Prints `error: <error>` on stderr and logs the same text (the whole chain) as an `error` event: the log of a refused
/// or failed command says why it ended.
pub fn error_line(error: &anyhow::Error) {
    let text = format!("{error:#}");
    stderr_line(&format!("error: {text}"));
    tracing::error!(target: PRINTED_TARGET, text = %text, "command error");
}

/// Prints `note: <text>` on stderr and logs it.
pub fn note_line(text: &str) {
    stderr_line(&format!("note: {text}"));
    tracing::info!(target: PRINTED_TARGET, text, "command note");
}

/// Prints `warning: <text>` on stderr and logs it, for a warning the command itself raises (the stages' go through
/// their progress sink).
pub fn warning_line(text: &str) {
    stderr_line(&format!("warning: {text}"));
    tracing::warn!(target: PRINTED_TARGET, text, "command warning");
}

/// How often a stage's progress line is printed at most, besides its first and last.
const PROGRESS_EVERY: Duration = Duration::from_secs(1);

struct Last {
    stage: String,
    done: usize,
    at: Instant,
}

/// The [`Progress`] of the command line: a stage's `done/total` on stderr (at most once a second, plus the first and
/// the last step of a stage), and its warnings as `warning:` lines.
pub struct StderrProgress {
    last: Mutex<Option<Last>>,
}

impl StderrProgress {
    /// A sink that has printed nothing yet.
    pub fn new() -> StderrProgress {
        StderrProgress { last: Mutex::new(None) }
    }
}

impl Default for StderrProgress {
    fn default() -> Self {
        StderrProgress::new()
    }
}

impl Progress for StderrProgress {
    fn step(&self, stage: &str, done: usize, total: usize) {
        let mut last = self.last.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let print = match last.as_ref() {
            None => true,
            // A new stage, or a counter that started over (the drizzle runs under `stack` with its own count).
            Some(l) => l.stage != stage || done < l.done || done >= total || now.duration_since(l.at) >= PROGRESS_EVERY,
        };
        if print {
            stderr_line(&format!("starloom: {stage} {done}/{total}"));
            *last = Some(Last { stage: stage.to_string(), done, at: now });
        } else if let Some(l) = last.as_mut() {
            l.done = done;
        }
    }

    fn warn(&self, message: &str) {
        stderr_line(&format!("warning: {message}"));
    }
}
