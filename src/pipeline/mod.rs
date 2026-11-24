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
//! - [`config`]: the effective [`AlgorithmConfig`](crate::stacking::config::AlgorithmConfig) (preset, file, `--set`,
//!   then Athenaeum's resolution), the keys a file held that it does not read, the stage hashes, the whole-config hash
//!   and [`ln_runs`].
//! - [`versions`]: every version constant by one name, as entries and reports record them.
//! - [`cards`]: the `SLM_*` cards every file the pipeline writes carries.
//! - [`master`] and [`calibrate`](mod@calibrate): the first two stages. [`build_master`] integrates raw bias, dark or
//!   flat frames into a master; [`calibrate()`](calibrate::calibrate) turns raw lights into calibrated frames, refusing
//!   a mismatch from the headers before any pixel is read ([`preflight`]). Raw frames may be compressed.
//! - [`measure`](mod@measure): the third stage. [`measure()`](measure::measure) measures every calibrated frame (two passes under the
//!   `auto` star profile, with a group β), writes `measure.json`, `measure.csv` and the `.athf` star fits, and [`finish`]
//!   turns stored measurements into weights and a selection under any config.
//! - [`register`](mod@register): the fourth stage. [`register()`](register::register) aligns every included frame onto one
//!   reference (a file, or the best-weighted frame), writes `transforms.json`, then reweighs ([`current_weights`]) and
//!   writes `weights.json`; with a flag it also writes each frame resampled into the reference's grid.
//! - [`formats`]: the manifest (`starloom.json`, with the command history) and the measure, transforms and weights
//!   reports, each tagged and versioned, plus the rules that keep a half-written stage from looking done.
//!
//! # Manual mode
//!
//! The stages are not linked (P2-R24). The CLI never compares one stage's products with another's and never refuses a
//! product as stale: the user decides what to re-run and what to clean between stages. A later command uses the
//! products it is given, or else those it finds in the working folder (a command given no frames takes the previous
//! stage's from the manifest, P2-R20). A missing input (a frame with no measurement, transform or `.athln`) is an
//! ordinary error that names the frame and the command that makes it. The manifest is a journal of what each command
//! did (products, sources, config hash, versions, history) for the user to read; nothing checks it.
//!
//! # The working folder
//!
//! A stage builds each product beside its destination ([`formats::part_path`]) and renames it into place, so an
//! output folder on another volume works; then it writes its manifest entry ([`formats::commit_stage`]), so an
//! interrupted stage leaves no entry and nothing named like a finished product. Every command that writes clears the
//! scratch folder `<work>/.tmp/` when it starts ([`formats::begin_writing`]). One command runs in a working folder at
//! a time: nothing locks the folder, and two at once would interleave their manifest writes.

pub mod calibrate;
pub mod cards;
pub mod config;
pub mod formats;
pub mod inputs;
pub mod master;
pub mod measure;
pub mod normalize;
pub mod register;
pub mod versions;
pub mod weights;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub use calibrate::{calibrate, preflight, preflight_with, CalibrateReport, CalibratedFrameReport};
pub use cards::{common_cards, SLM_CFG, SLM_PLV, SLM_VER};
pub use config::{config_hash, effective_config, ln_runs, stage_hash};
pub use formats::{
    append_history, begin_writing, commit_stage, frames_or_upstream, part_path, source_facts, Entry, HistoryRow, Manifest,
    MeasureFrame, MeasureReport,
};
pub use inputs::{
    expand_calibration_frames, expand_inputs, frame_stem, order_lights, parse_date_obs, read_plan, CalGroup, CalSource,
    FlatPrecalSource, InputKind,
};
pub use master::{build_master, MasterKind, MasterReport};
pub use register::{register, register_dir, registered_path, ReferenceChoice, RegisterReport, REGISTER_DIR, REGISTERED_DIR, SLM_REF};
pub use normalize::{athln_path, ln_dir, normalize, LN_REFERENCE_FILE, LN_REFERENCE_SATURATION_FILE, NORMALIZE_DIR};
pub use weights::{current_weights, subset_measure};
pub use measure::{athf_path, finish, measure, measure_dir, pass1_path, MEASURE_CSV_FILE, MEASURE_DIR};
pub use versions::{all_versions, recorded_versions, VERSION_CARDS};

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
    /// The stage's command name, as the manifest and the "run `starloom <stage>`" messages spell it.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Calibrate => "calibrate",
            Stage::Measure => "measure",
            Stage::Register => "register",
            Stage::Normalize => "normalize",
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
