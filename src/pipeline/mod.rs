//! The working-folder pipeline: the stages a command line drives (`master`, `calibrate`, `measure`, `register`,
//! `normalize`, `stack`, `run`), the inputs they take, the config levels that drive them and the versioned files
//! they leave in a working folder.
//!
//! This module is the skeleton the stages hang on:
//! - [`Ctx`] and [`Progress`]: what a stage runs with (the caller's rayon pool, a cancel flag, a progress sink, a
//!   memory budget, the working folder). The pipeline never builds a thread pool (SL-9) and writes nothing to stdout
//!   or stderr: logs are `tracing`, and the caller's [`Progress`] carries what a person should see.
//! - [`inputs`]: file, folder and glob arguments expanded the same way on every platform, the frame order a stack
//!   uses, the calibration plan file.
//! - [`config`]: the effective [`AlgorithmConfig`](crate::stacking::config::AlgorithmConfig) (preset, file, `--set`),
//!   the stage hashes and the whole-config hash.
//! - [`cards`]: the `SLM_*` cards every file the pipeline writes carries.
//! - [`formats`]: the manifest (`starloom.json`, with the command history) and the measure, transforms and weights
//!   reports, each tagged and versioned, plus the rules that keep a half-written stage from looking done.
//!
//! # The working folder
//!
//! A stage writes its products into `<work>/.tmp/` and renames them into place, then writes its manifest entry: an
//! entry exists only for products that are complete. Every command that writes clears `<work>/.tmp/` when it
//! starts ([`formats::begin_writing`]), so a killed process leaves nothing that looks finished.

pub mod cards;
pub mod config;
pub mod formats;
pub mod inputs;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub use cards::{common_cards, SLM_CFG, SLM_PLV, SLM_VER};
pub use config::{config_hash, effective_config, stage_hash};
pub use formats::{
    append_history, begin_writing, check_upstream, commit_stage, frames_or_upstream, Entry, FileId, HistoryRow, Manifest,
};
pub use inputs::{
    expand_inputs, frame_stem, order_lights, parse_date_obs, read_plan, CalGroup, CalSource, FlatPrecalSource, InputKind,
};

/// The version of the pipeline as a whole: what the cards `SLM_PLV` records, and what a golden row of the whole
/// chain is keyed on. A change to the pipeline's own glue that moves a product's bytes without moving a stage version
/// (the order frames are taken in, the cards a product carries) bumps it.
pub const PIPELINE_VERSION: u32 = 1;

/// The stages that leave an entry in the manifest, in the order a chain runs them. Drizzle is part of `Stack`, and
/// `master` leaves its masters beside the calibrated frames without an entry of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Stage {
    /// Light calibration: raw lights to calibrated frames.
    Calibrate,
    /// Star measurement, weights and frame selection.
    Measure,
    /// Registration of every frame onto one reference.
    Register,
    /// Local normalization.
    Normalize,
    /// Rejection, integration and drizzle.
    Stack,
}

impl Stage {
    /// The stage's command name, as the manifest and the "re-run `starloom <stage>`" messages spell it.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Calibrate => "calibrate",
            Stage::Measure => "measure",
            Stage::Register => "register",
            Stage::Normalize => "normalize",
            Stage::Stack => "stack",
        }
    }

    /// The name of the config subtree the stage's hash is made of, as a message names it ("another measurement
    /// config").
    pub fn subtree_name(self) -> &'static str {
        match self {
            Stage::Calibrate => "calibration",
            Stage::Measure => "measurement",
            Stage::Register => "registration",
            Stage::Normalize => "normalization",
            Stage::Stack => "stack",
        }
    }
}

/// Where a stage reports what it is doing. The caller decides what that is: the command line prints to stderr, a
/// library caller may ignore it. It is `Sync` because the stages call it from pool threads.
pub trait Progress: Sync {
    /// `done` of `total` units of `stage` are finished (a frame, a plane, a step). `stage` is the command's name.
    fn step(&self, stage: &str, done: usize, total: usize);
    /// Something a person should know that does not stop the work: a skipped frame, a mismatch that only warns.
    fn warn(&self, message: &str);
}

/// A [`Progress`] that discards everything, for a caller that wants none.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl Progress for NoProgress {
    fn step(&self, _stage: &str, _done: usize, _total: usize) {}
    fn warn(&self, _message: &str) {}
}

/// What a stage runs with. The pool is the caller's (SL-9): a stage installs work on it and never builds one.
pub struct Ctx<'a> {
    /// The rayon pool every parallel step runs on.
    pub pool: Arc<rayon::ThreadPool>,
    /// Set by the caller to stop the stage; a stage checks it between frames ([`Ctx::check_cancelled`]) and leaves
    /// no manifest entry when it stops.
    pub cancel: &'a AtomicBool,
    /// Where progress and warnings go.
    pub progress: &'a dyn Progress,
    /// How many bytes of working memory the stage may plan for, which bounds how many frames it holds at once.
    pub memory_budget_bytes: u64,
    /// The working folder: the manifest, the stages' products and `.tmp/` live under it.
    pub work_dir: &'a Path,
}

impl Ctx<'_> {
    /// `Err` when the caller has asked to stop. Stages call it between frames and propagate the error with `?`.
    pub fn check_cancelled(&self) -> anyhow::Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cancelled_context_stops_the_stage_with_an_error() {
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
        let cancel = AtomicBool::new(false);
        let dir = tempfile::tempdir().unwrap();
        let ctx = Ctx {
            pool,
            cancel: &cancel,
            progress: &NoProgress,
            memory_budget_bytes: 1 << 30,
            work_dir: dir.path(),
        };
        assert!(ctx.check_cancelled().is_ok());
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(ctx.check_cancelled().unwrap_err().to_string(), "cancelled");
    }

    #[test]
    fn stage_names_are_the_command_names() {
        let names: Vec<&str> = [Stage::Calibrate, Stage::Measure, Stage::Register, Stage::Normalize, Stage::Stack]
            .into_iter()
            .map(Stage::name)
            .collect();
        assert_eq!(names, ["calibrate", "measure", "register", "normalize", "stack"]);
        assert_eq!(serde_json::to_string(&Stage::Normalize).unwrap(), "\"normalize\"");
    }
}
