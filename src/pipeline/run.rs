//! `run`: the stages one after another over all the groups' lights, into one working folder.
//!
//! It is a sequence of the stage functions, not a second implementation (P2-R3): `calibrate` (which builds the masters
//! of every `CalSource::Raw` first), `measure`, `register`, `normalize` (only when [`ln_runs`]) and `stack` (with the
//! drizzle). Nothing is checked between the stages and nothing is cached (P2-R24): `run` makes one master from all the
//! lights, and a plan's groups are calibration groups only (P2-R12). The calibrated frames go to `<work>/calibrated/`
//! as plain FITS, which is what the later stages read (P2-R11); `format` is the master's container.
//!
//! The caller clears `<work>/.tmp/` ([`begin_writing`](crate::pipeline::begin_writing)) as it does for any writing
//! command.
//!
//! `keep_mosaic` is set where Athenaeum's calibration stage sets it: when [`wants_cfa_mosaic`] says the group's drizzle
//! deposits each colour's own samples, with `osc` true when a light's facts carry a CFA pattern. Without it a Bayer
//! drizzle would find no mosaic.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::fits_writer::OutputFormat;
use crate::pipeline::calibrate::{calibrate, CalibrateReport};
use crate::pipeline::formats::{MeasureReport, NormalizeReport};
use crate::pipeline::inputs::CalGroup;
use crate::pipeline::master::frame_facts;
use crate::pipeline::measure::measure;
use crate::pipeline::normalize::normalize;
use crate::pipeline::register::{register, ReferenceChoice, RegisterReport};
use crate::pipeline::stack::{stack, StackReport};
use crate::pipeline::{ln_runs, Ctx};
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::run_helpers::wants_cfa_mosaic;

/// The folder under the working folder the calibrated frames (and their mosaics) go to.
pub const CALIBRATED_DIR: &str = "calibrated";

/// What every stage of a [`run`] left.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// The calibrated frames and the masters built for them.
    pub calibrate: CalibrateReport,
    /// The measurements, weights and selection.
    pub measure: MeasureReport,
    /// The transforms and the reweigh.
    pub register: RegisterReport,
    /// Local normalization's report; `None` when [`ln_runs`] is false and the stage was not run.
    pub normalize: Option<NormalizeReport>,
    /// The master, its maps and the drizzle.
    pub stack: StackReport,
}

/// `<work>/calibrated`: where [`run`] puts the calibrated frames.
pub fn calibrated_dir(work_dir: &Path) -> PathBuf {
    work_dir.join(CALIBRATED_DIR)
}

/// Whether the calibration keeps the calibrated CFA mosaic beside each debayered frame: [`wants_cfa_mosaic`] with `osc`
/// true when a light's facts carry a CFA pattern. The lights' headers are read only when the other three terms already
/// hold; a header that cannot be read is `calibrate`'s refusal to make, not this one's.
fn keep_mosaic(groups: &[CalGroup], cfg: &AlgorithmConfig) -> bool {
    if !wants_cfa_mosaic(&cfg.drizzle, &cfg.calibration, true) {
        return false;
    }
    let osc = groups.iter().flat_map(|g| &g.lights).any(|light| match frame_facts(light) {
        Ok(facts) => facts.bayerpat.as_deref().map(str::trim).is_some_and(|p| !p.is_empty()),
        Err(e) => {
            tracing::debug!(path = %light.display(), error = %format!("{e:#}"), "keep_mosaic: the light's header is unreadable");
            false
        }
    });
    wants_cfa_mosaic(&cfg.drizzle, &cfg.calibration, osc)
}

/// Runs master, calibrate, measure, register, normalize and stack over every group's lights ([module docs](self)).
///
/// `reference` is what `register` aligns onto; `out_dir` and `format` are where the master (and the drizzle) go.
/// A stage's failure or refusal fails the run with that stage's error; the products of the stages before it stay.
pub fn run(
    ctx: &Ctx<'_>,
    groups: &[CalGroup],
    cfg: &AlgorithmConfig,
    reference: &ReferenceChoice,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<RunReport> {
    let mut opts = cfg.calibration.clone();
    opts.keep_mosaic = keep_mosaic(groups, cfg);
    let calibrated = calibrate(ctx, cfg, groups, &opts, &calibrated_dir(ctx.work_dir), OutputFormat::Fits)?;
    let frames: Vec<PathBuf> = calibrated.frames.iter().map(|f| f.output.clone()).collect();
    let measured = measure(ctx, cfg, &frames, None, false)?;
    let registered = register(ctx, &frames, &measured, reference, cfg, false)?;
    let normalized = if ln_runs(cfg) { Some(normalize(ctx, &frames, &measured, &registered, cfg)?) } else { None };
    let stacked = stack(ctx, cfg, &frames, &measured, &registered, normalized.as_ref(), out_dir, format)?;
    Ok(RunReport { calibrate: calibrated, measure: measured, register: registered, normalize: normalized, stack: stacked })
}
