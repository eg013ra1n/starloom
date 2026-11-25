//! The `stack` stage: integration with rejection, weights and local normalization, and the master light.
//!
//! This mirrors Athenaeum's Output stage (`process_group_output`) for one group, up to the master write, without the
//! catalog, the project grid, the crop and the artifact cache (P2-R3):
//!
//! 1. the weights and the selection come from [`current_weights`] under the caller's config, over the stored
//!    measurements and transforms (P2-R21); fewer than 3 included frames is refused, as Athenaeum skips such a group;
//! 2. the members are the included, measured, registered frames, in frame order; with local normalization on
//!    ([`ln_runs`]) each member's grids come from its `.athln` (`<work>/ln/<stem>.athln`) by the status `normalize`
//!    gave it in `ln/normalize.json`: a frame `normalize` excluded is left out, one it kept on the global normalization
//!    stacks without grids, and a report whose status is `off` stacks the whole group on the global normalization, as
//!    Athenaeum's run falls back;
//! 3. the normalization anchor is the member that covers the reference rectangle and ranks first by weight over the
//!    square root of its sky ([`pick_ln_anchor`](crate::stacking::run_helpers::pick_ln_anchor)), picked over the
//!    members that are left;
//! 4. the group is integrated ([`integrate_group`]) in the registration reference's grid with the frame-common star
//!    set the reweigh found, the rejection-bitmap set Athenaeum creates for large-scale rejection (and for drizzle), and
//!    the rejection maps when `integration.writeRejectionMaps` is set and they fit a quarter of the machine's memory;
//! 5. the master light is written with the standard cards of Athenaeum's `build_master_light_cards` (the anchor's
//!    copy-through cards, `NCOMBINE`, the weighted `EXPTIME`, `DATE-OBS`/`DATE-END`), the registration reference's
//!    world-coordinate cards, and starloom's own in place of every `ATH_*` card (P2-R6): `SLM_VER`, `SLM_CFG`,
//!    `SLM_PLV`, `SLM_REF` (the registration reference's file name) and the versions `SLM_MSV`, `SLM_MV`, `SLM_ALV`,
//!    `SLM_LNV`, `SLM_INV`.
//!
//! 6. with `drizzle.enabled`, the drizzle runs on the integrated frames while the rejection bitmaps exist and writes
//!    the drizzled master (and its weight map); see [`drizzle`]. It never fails the stack: a skip or a failure is a
//!    warning and the master stands. A Bayer drizzle with no frame's mosaic is refused before any pixel is read.
//!
//! Products: the master light `<out>/master.fits` (or `.xisf`), the rejection maps `<out>/master_rejlow.fits` and
//! `<out>/master_rejhigh.fits` (always FITS, as Athenaeum writes them), and `<work>/stack/weights.json`, the weights the
//! stage stacked with (a frame local normalization excluded is marked excluded there, with `normalize`'s reason, as
//! Athenaeum persists the exclusion). Each is built beside its destination and renamed into place with the manifest entry.
//!
//! **Before any pixel is read**, every included frame must have its transform and, with local normalization on, its
//! `.athln`; a missing one is an error naming the frame and the command that makes it (`starloom register`, `starloom
//! normalize`). Nothing compares the products with one another (P2-R24): a `normalize` run under another config is used
//! as it is. A frame is warped on the fly through its transform: the registered copies `register --write-registered`
//! may have written hold the same pixels and are not read. A frame's id inside the stage is its index among the frames
//! it was given, which stands for the catalog id Athenaeum breaks ties on (P2-R10). Everything that decodes a pixel,
//! warps or integrates runs on the context's pool (SL-9).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::{header_key_map, header_text, FrameFacts};

use crate::fits_writer::card::format_card;
use crate::fits_writer::{write_fits_f32, write_image_f32, Card, CardValue, FitsWriteError, OutputFormat, XisfBounds};
use crate::integration::engine::EngineProgress;
use crate::integration::plane_reader::PlaneReader;
use crate::integration::IntegrationError;
use crate::pipeline::calibrate::{card_from_kv, encode_ident, file_name_of};
use crate::pipeline::cards::version_card;
use crate::pipeline::drizzle::{self, DrizzleReport};
use crate::pipeline::formats::{
    commit_stage, part_path, Entry, FrameStatus, MeasureReport, NormalizeFrame, NormalizeReport, NormalizeStatus,
    TransformFrame, WeightSettings, WeightsReport, NORMALIZE_FILE, WEIGHTS_FILE,
};
use crate::pipeline::master::{io_policy, scratch_dir};
use crate::pipeline::normalize::Ranking;
use crate::pipeline::register::{header_cards, is_wcs, RegisterReport, SLM_REF};
use crate::pipeline::weights::{current_weights, subset_measure};
use crate::pipeline::{
    athln_path, common_cards, config_hash, frame_stem, ln_dir, ln_runs, parse_date_obs, source_facts, warn_user, Ctx,
    Manifest, Progress, Stage,
};
use crate::orientation::row_order_is_bottom_up;
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::integrate::{
    included_after_min_weight, integrate_group, large_scale_passes, GroupInput, GroupOutput, GroupProgress, GroupStats,
    StackFrame,
};
use crate::stacking::ln::LnFrameGrids;
use crate::stacking::rej::RejBitmapSet;
use crate::stacking::weights::FrameWeight;

/// The stage's folder under the working folder: `weights.json` lives there.
pub const STACK_DIR: &str = "stack";
/// The master light's file stem in the output folder; the rejection maps add `_rejlow` and `_rejhigh`.
pub const MASTER_STEM: &str = "master";
/// The scratch folder, under `<work>/.tmp/`, the rejection-bitmap set is written to while the stage runs.
const REJ_SCRATCH_DIR: &str = "stack-rej";

/// The cards a master light copies from its anchor's header: Athenaeum's `REGISTERED_COPY_THROUGH`, the list
/// `build_master_light_cards` filters the reference's cards with.
const MASTER_COPY_THROUGH: [&str; 26] = [
    "EXPTIME", "GAIN", "OFFSET", "EGAIN", "XBINNING", "YBINNING", "XPIXSZ", "YPIXSZ", "CCD-TEMP", "SET-TEMP", "INSTRUME",
    "TELESCOP", "FOCALLEN", "APTDIA", "FILTER", "DATE-OBS", "OBJECT", "OBJCTRA", "OBJCTDEC", "RA", "DEC", "SITELAT",
    "SITELONG", "SITEELEV", "OBSERVER", "ROWORDER",
];

/// The version cards a master light carries after the common ones and `SLM_REF` (P2-R6), each one of
/// [`VERSION_CARDS`](crate::pipeline::VERSION_CARDS).
pub const MASTER_VERSION_CARDS: [&str; 5] = ["SLM_MSV", "SLM_MV", "SLM_ALV", "SLM_LNV", "SLM_INV"];

/// The stage's folder: `<work>/stack`.
pub fn stack_dir(work_dir: &Path) -> PathBuf {
    work_dir.join(STACK_DIR)
}

/// Where `stack` writes the master light in `out_dir`: `<out>/master.fits`, or `master.xisf` for
/// [`OutputFormat::Xisf`].
pub fn master_path(out_dir: &Path, format: OutputFormat) -> PathBuf {
    out_dir.join(format!("{MASTER_STEM}.{}", format.extension()))
}

/// Where `stack` writes the rejection map `suffix` (`rejlow` or `rejhigh`) beside the master: always FITS.
fn rejection_map_path(out_dir: &Path, suffix: &str) -> PathBuf {
    out_dir.join(format!("{MASTER_STEM}_{suffix}.fits"))
}

/// What `stack` made.
#[derive(Debug, Clone)]
pub struct StackReport {
    /// The master light.
    pub master: PathBuf,
    /// The low-side rejection map, when `integration.writeRejectionMaps` was set and the maps fit in memory.
    pub rejection_low: Option<PathBuf>,
    /// The high-side rejection map, as `rejection_low`.
    pub rejection_high: Option<PathBuf>,
    /// `<work>/stack/weights.json`: the weights the stage stacked with.
    pub weights_file: PathBuf,
    /// The weights and the selection the stage stacked with, as `weights.json` holds them: a frame local normalization
    /// excluded is marked excluded, with the reason `normalize` gave it.
    pub weights: WeightsReport,
    /// The frames integrated, in the integration's order (after the `integration.minWeight` floor).
    pub frames: Vec<PathBuf>,
    /// The normalization anchor: the frame whose measurement normalizes the others and whose header the master copies.
    pub anchor: PathBuf,
    /// The registration reference, whose grid the master is in (`SLM_REF`).
    pub reference: PathBuf,
    /// The integration's summary: counts, rejection fractions, the master's noise and star figures, the SNR gain.
    pub stats: GroupStats,
    /// The drizzled master, when `drizzle.enabled` was set and the drizzle ran; `None` when it is off, was skipped
    /// (the rejection bitmaps were unavailable) or failed (a warning says which): the master stands either way.
    pub drizzle: Option<DrizzleReport>,
    /// Set when the weights were made under other weight settings (mode, formula, keyword, selection) than the ones
    /// `measure` ran with: the stage re-weighed from the stored measurements and says so (P2-R21).
    pub reweighed: Option<String>,
    /// Everything the stage sent to the context's [`Progress`] as a warning, in order: its own and the ones the weights
    /// and the anchor pick raise.
    pub warnings: Vec<String>,
}

/// Integrates `frames` (calibrated plain-FITS frames `measure` has rows for and `register` has transforms for) into one
/// master light in `out_dir`, and writes `<work>/stack/weights.json` ([module docs](self)).
///
/// `measure` holds the stored measurements, `register` the transforms; the weights and the selection are computed again
/// under `cfg` ([`current_weights`], P2-R21). Under [`ln_runs`], `normalize` is the report of the `normalize` run to
/// stack with (`None` reads `<work>/ln/normalize.json`): a frame it excluded is left out, a frame it kept on the global
/// normalization stacks without grids, every other member's `.athln` is read, and a report whose status is
/// [`NormalizeStatus::Off`] stacks the whole group on the global normalization, as Athenaeum's run falls back. Without
/// local normalization `normalize` is not read.
///
/// Refused before any pixel is read: a frame with no row in `measure` (naming `starloom measure`), an included frame
/// with no transform (naming `starloom register`), with local normalization on no normalize report or an included
/// frame with no `.athln` (naming `starloom normalize`), and fewer than 3 included frames. The products are built beside their destinations and
/// renamed into place with the manifest entry ([`commit_stage`]), so a failure or a cancel leaves no entry and no
/// half-written product. Everything that decodes, warps or integrates runs on the context's pool (SL-9).
#[allow(clippy::too_many_arguments)]
pub fn stack(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    frames: &[PathBuf],
    measure: &MeasureReport,
    register: &RegisterReport,
    normalize: Option<&NormalizeReport>,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<StackReport> {
    if frames.is_empty() {
        refuse!("no frames to stack");
    }
    let work = std::path::absolute(ctx.work_dir).with_context(|| format!("cannot make {} absolute", ctx.work_dir.display()))?;
    let out_dir = std::path::absolute(out_dir).with_context(|| format!("cannot make {} absolute", out_dir.display()))?;
    let sub = subset_measure(measure, frames)?;
    ctx.check_cancelled()?;
    // The warnings the weights and the anchor pick send to the context's progress sink go to the report too.
    let tee = WarningTee { inner: ctx.progress, warnings: Mutex::new(Vec::new()) };
    let teed = Ctx {
        pool: ctx.pool.clone(),
        cancel: ctx.cancel,
        progress: &tee,
        memory_budget_bytes: ctx.memory_budget_bytes,
        work_dir: ctx.work_dir,
    };
    // Everything below decodes, warps or integrates on the caller's pool, never rayon's global one (SL-9).
    let mut report = ctx.pool.install(|| run(&teed, cfg, &work, &sub, register, normalize, &out_dir, format))?;
    report.warnings = tee.take();
    Ok(report)
}

/// A [`Progress`] that passes everything to the caller's and keeps the warnings for the report.
struct WarningTee<'a> {
    inner: &'a dyn Progress,
    warnings: Mutex<Vec<String>>,
}

impl WarningTee<'_> {
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.warnings.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }
}

impl Progress for WarningTee<'_> {
    fn step(&self, stage: &str, done: usize, total: usize) {
        self.inner.step(stage, done, total);
    }

    fn warn(&self, message: &str) {
        self.warnings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(message.to_string());
        self.inner.warn(message);
    }
}

/// Sends a warning to the log and the context (whose sink the report's warnings are taken from).
fn warn(ctx: &Ctx<'_>, message: String) {
    warn_user(ctx, &message);
}

/// A config value's serde name (`psfSnr`, `noise`, ...), for a message.
fn serde_name<T: serde::Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        Ok(other) => other.to_string(),
        Err(_) => "?".to_string(),
    }
}

/// The note of a stack whose weight settings are not the ones `measure` ran with (P2-R21); `None` when they are.
fn reweigh_note(measured_with: &AlgorithmConfig, used: &WeightSettings) -> Option<String> {
    let measured = WeightSettings {
        weight_mode: measured_with.measurement.weight_mode,
        formula: measured_with.measurement.formula.clone(),
        keyword: measured_with.measurement.keyword.clone(),
        selection: measured_with.selection,
    };
    (measured != *used).then(|| {
        format!(
            "re-weighed from the stored measurements without measuring again: weight mode {} (measure ran with {}){}",
            serde_name(&used.weight_mode),
            serde_name(&measured.weight_mode),
            if measured.weight_mode == used.weight_mode { ", other formula, keyword or selection" } else { "" }
        )
    })
}

/// The zero weight Athenaeum's output stage gives a member it has no weight for.
fn no_weight() -> FrameWeight {
    FrameWeight { channels: Vec::new(), normalized: Vec::new(), mean: 0.0, normalized_mean: 0.0, missing: None }
}

/// `DATE-OBS` as Athenaeum's catalog stores it (the parsed time, RFC 3339), which the master's date cards copy.
fn stored_date(facts: Option<&FrameFacts>) -> Option<String> {
    let facts = facts?;
    parse_date_obs(facts.date_obs.as_deref()?, facts.time_obs.as_deref()).map(|d| d.to_rfc3339())
}

/// A list of frames for an error: the first three, then how many more.
fn listed(items: &[String]) -> String {
    let shown = items.iter().take(3).cloned().collect::<Vec<_>>().join("; ");
    if items.len() > 3 {
        format!("{shown} and {} more", items.len() - 3)
    } else {
        shown
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    work: &Path,
    sub: &MeasureReport,
    register: &RegisterReport,
    normalize: Option<&NormalizeReport>,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<StackReport> {
    let files: Vec<PathBuf> = sub.frames.iter().map(|f| f.file.clone()).collect();

    // The weights under this config; an included frame with no transform is refused here, naming `starloom register`.
    let mut weights = current_weights(ctx, sub, register, cfg)?;
    let reweighed = reweigh_note(&sub.config, &weights.settings);
    if let Some(note) = &reweighed {
        tracing::info!("{note}");
    }
    let included_count = weights.frames.iter().filter(|f| f.included).count();
    if included_count < 3 {
        refuse!(
            "fewer than 3 included frames: {included_count} of {} are included under this config; a stack needs at least 3",
            files.len()
        );
    }

    // The members: included, measured, registered (`GroupMember`s of Athenaeum's output stage), in frame order.
    let manifest = Manifest::load(ctx.work_dir)?;
    let sources: BTreeMap<PathBuf, FrameFacts> = source_facts(&manifest, &files)?;
    let rows: BTreeMap<&Path, &TransformFrame> = register.transforms.frames.iter().map(|t| (t.file.as_path(), t)).collect();
    let mut members: Vec<StackFrame> = Vec::with_capacity(included_count);
    for (index, w) in weights.frames.iter().enumerate() {
        if !w.included {
            continue;
        }
        let (Some(measurement), Some(map)) = (w.measurement.as_ref(), rows.get(w.file.as_path()).and_then(|t| t.map.as_ref())) else {
            continue;
        };
        let facts = sources.get(&w.file);
        members.push(StackFrame {
            path: w.file.clone(),
            map: map.clone(),
            measurement: measurement.clone(),
            weight: w.weight.clone().unwrap_or_else(no_weight),
            exposure_s: facts.and_then(|f| f.exptime_s).unwrap_or(0.0),
            date_obs: stored_date(facts),
            frame_id: index as i64,
            // The warp is applied on the fly: a registered copy on disk is the same pixels, and nothing says it is still
            // the one these transforms made.
            registered_path: None,
        });
    }
    if members.len() < 3 {
        refuse!("{} of {included_count} included frames are registered; at least 3 are needed", members.len());
    }

    // Local normalization: each member's grids, read before any pixel is; a frame `normalize` excluded is left out.
    let ln = if ln_runs(cfg) { ln_grids(ctx, work, &members, normalize)? } else { LnPlan::default() };
    let mut ln_by_member = ln.grids;
    if !ln.dropped.is_empty() {
        // Athenaeum persists the exclusion on the frame's row: `weights.json` says the frame was not stacked.
        for (k, reason) in &ln.dropped {
            if let Some(row) = weights.frames.iter_mut().find(|f| f.file == members[*k].path) {
                row.included = false;
                row.reason = Some(reason.clone());
            }
        }
        let keep: Vec<bool> = (0..members.len()).map(|k| !ln.dropped.iter().any(|(d, _)| *d == k)).collect();
        let mut k = 0;
        members.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        if let Some(grids) = ln_by_member.as_mut() {
            let mut k = 0;
            grids.retain(|_| {
                k += 1;
                keep[k - 1]
            });
        }
        if members.len() < 3 {
            refuse!("local normalization left {} frames; at least 3 are needed", members.len());
        }
    }

    // The registration reference's grid, and the normalization anchor over the members that are left.
    let reference = register.transforms.reference.clone();
    let (width, height) = {
        let reader = PlaneReader::open(&reference)
            .map_err(|e| anyhow!("cannot read the registration reference {}: {e}", reference.display()))?;
        (reader.width(), reader.height())
    };
    let anchor = {
        let refs: Vec<&StackFrame> = members.iter().collect();
        Ranking::of(&refs, width, height).anchor(ctx).ok_or_else(|| anyhow!("no included frame to anchor normalization"))?
    };
    let channels = members[anchor].measurement.channels.len();

    // The drizzle's run-wide checks, before any pixel is read: its scale, and for a Bayer drizzle the mosaics.
    let drizzle_settings = if cfg.drizzle.enabled { Some(drizzle::settings(ctx, &cfg.drizzle)?) } else { None };
    let mosaics = if drizzle_settings.is_some() && drizzle::bayer_wanted(cfg, channels) {
        let sources = drizzle::mosaic_sources(&members);
        if sources.iter().all(Option::is_none) {
            refuse!(
                "Bayer drizzle needs each frame's calibrated CFA mosaic, and none of the {} frames has one beside it; \
                 run `starloom calibrate --keep-mosaic` on the raw lights (with the debayer on), or turn drizzle.bayer off",
                members.len()
            );
        }
        Some(sources)
    } else {
        None
    };

    // The rejection maps, off for this group with a warning when they would not fit a quarter of the memory.
    let mut integration = cfg.integration.clone();
    if integration.write_rejection_maps {
        let maps_bytes = 3u64 * channels as u64 * width as u64 * height as u64 * 4;
        // Unknown memory is taken as not fitting, as Athenaeum's run does.
        let fits = crate::sys::total_ram_bytes().is_some_and(|total| maps_bytes <= total / 4);
        if !fits {
            integration.write_rejection_maps = false;
            warn(ctx, "rejection maps disabled: they would exceed the memory budget".to_string());
        }
    }

    // The rejection-bitmap set large-scale rejection derives its structures from, and drizzle reads (Athenaeum creates
    // it before the integration, sized and ordered to what the weight floor keeps).
    let bitmaps_for_drizzle = cfg.drizzle.enabled && cfg.drizzle.use_rejection;
    let bitmaps_for_large_scale = integration.large_scale.enabled;
    let bitmap_consumers = match (bitmaps_for_drizzle, bitmaps_for_large_scale) {
        (true, true) => "drizzle and large-scale rejection",
        (true, false) => "drizzle",
        _ => "large-scale rejection",
    };
    // The scratch folder's guard is declared first, so it drops (and removes the folder) after the set.
    let mut _rej_cleanup: Option<RemoveOnDrop> = None;
    let mut rej_set: Option<RejBitmapSet> = None;
    let mut rej_set_failure: Option<String> = None;
    if bitmaps_for_drizzle || bitmaps_for_large_scale {
        let rej_dir = scratch_dir(ctx)?.join(REJ_SCRATCH_DIR);
        remove_dir_if_present(&rej_dir)?;
        _rej_cleanup = Some(RemoveOnDrop(rej_dir.clone()));
        let stems: Vec<String> = included_after_min_weight(&members, anchor, integration.min_weight)
            .iter()
            .map(|&i| frame_stem(&members[i].path).to_string())
            .collect();
        match RejBitmapSet::create(&rej_dir, &stems, width, height, channels) {
            Ok(set) => rej_set = Some(set),
            Err(e) => {
                let msg = format!("failed to create the rejection-bitmap set: {e:#}");
                warn(ctx, format!("{bitmap_consumers} skipped: {msg}"));
                rej_set_failure = Some(msg);
            }
        }
    }

    // The integration, plane by plane, with the frame-common star set the reweigh found.
    let common_stars: Option<&[Option<Vec<(f64, f64)>>]> =
        (!weights.common_stars.is_empty()).then_some(weights.common_stars.as_slice());
    let input = GroupInput {
        frames: &members,
        reference: anchor,
        width,
        height,
        channels,
        interpolation: cfg.registration.interpolation,
        clamping: cfg.registration.clamping_threshold,
        integration: &integration,
        normalization: &cfg.normalization,
        ln: ln_by_member.as_deref(),
        rej: rej_set.as_ref(),
        second_pass_bitmaps: bitmaps_for_drizzle,
        coverage_map: false,
        stats_on_covered: false,
        common_stars,
    };
    let planes_total = channels * large_scale_passes(&integration, rej_set.is_some());
    let on_plane = |plane: usize, _planes: usize| ctx.progress.step(Stage::Stack.name(), plane, planes_total);
    let on_band = |_band: usize, _bands: usize, _done: u64, _total: u64| {};
    let on_combine = |_rows: usize, _rows_total: usize, _done: u64, _total: u64| {};
    let progress = GroupProgress { on_plane: &on_plane, engine: EngineProgress { on_band: &on_band, on_combine: &on_combine } };
    let measure_opts = cfg.measurement.measure_options(cfg.normalization.scale_estimator);
    let output: GroupOutput = {
        let _span = tracing::debug_span!("integrate_group").entered();
        integrate_group(&input, &measure_opts, &ctx.pool, ctx.cancel, &progress, io_policy(ctx).with_local_prefetch())
    }
    .map_err(|e| match e {
        IntegrationError::Cancelled => anyhow!("cancelled"),
        other => anyhow!("group integration failed: {other}"),
    })?;
    ctx.check_cancelled()?;
    ctx.progress.step(Stage::Stack.name(), planes_total, planes_total);

    // A bitmap write fault keeps the master; what it costs is said here, as is a large-scale pass that did not run.
    if let Some(reason) = rej_set.as_ref().and_then(RejBitmapSet::failure) {
        warn(ctx, format!("{bitmap_consumers} skipped: rejection bitmaps could not be written: {reason}"));
        rej_set_failure = Some(reason);
    }
    if bitmaps_for_large_scale && output.stats.large_scale_rejected_fraction.is_none() && rej_set_failure.is_none() {
        warn(
            ctx,
            "large-scale rejection did not run: the master is the per-pixel result (see the log for the reason)".to_string(),
        );
    }

    // The large-scale second pass ran and left no bitmap set of its own: nothing describes this master's rejected
    // samples, so drizzle is skipped rather than pointed at the first pass's bits.
    if drizzle::second_pass_unavailable(
        bitmaps_for_drizzle,
        output.stats.large_scale_rejected_fraction.is_some(),
        output.second_pass_rej_ok,
        rej_set_failure.as_deref(),
    ) {
        let msg = "the second pass's rejection bitmaps are unavailable".to_string();
        warn(ctx, format!("drizzle skipped: {msg}"));
        rej_set_failure = Some(msg);
    }

    // The master and its maps, built beside their destinations; then `weights.json` and the entry.
    let cards = master_cards(cfg, &members, anchor, &reference, &output)?;
    // XISF has no row order: the master's rows are its frames' order, and a viewer shows a bottom-up (or unstated) set
    // flipped against the FITS master. The pixels are not flipped, so the user is told (Athenaeum warns once per group).
    if format == OutputFormat::Xisf && cards_row_order_is_bottom_up(&cards) {
        warn(
            ctx,
            "XISF output keeps the frames' bottom-up row order; XISF viewers will show it flipped relative to the FITS master"
                .to_string(),
        );
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;
    let master = master_path(out_dir, format);
    let mut moves: Vec<(PathBuf, PathBuf)> = Vec::new();
    let written = (|| -> Result<()> {
        let built = part_path(&master);
        write_image_f32(&built, output.width, output.height, output.channels, &output.data, &cards, format, XisfBounds::Unit)
            .with_context(|| format!("cannot write {}", master.display()))?;
        moves.push((built, master.clone()));
        for (suffix, label, data) in
            [("rejlow", "Low", output.rejection_low.as_deref()), ("rejhigh", "High", output.rejection_high.as_deref())]
        {
            let Some(data) = data else { continue };
            let dest = rejection_map_path(out_dir, suffix);
            let built = part_path(&dest);
            write_fits_f32(&built, output.width, output.height, output.channels, data, &rejection_map_cards(label, &cards)?)
                .with_context(|| format!("cannot write {}", dest.display()))?;
            moves.push((built, dest));
        }
        let weights_dest = stack_dir(work).join(WEIGHTS_FILE);
        let built = part_path(&weights_dest);
        weights.write(&built)?;
        moves.push((built, weights_dest));
        Ok(())
    })();
    if let Err(e) = written {
        remove_built(&moves);
        return Err(e);
    }

    // The drizzle, while the bitmaps exist. It never fails the stack: a skip or a failure is a warning and the master
    // stays; a cancel is an error.
    let mut drizzle_report: Option<DrizzleReport> = None;
    if let Some(settings) = drizzle_settings {
        if let Some(reason) = &rej_set_failure {
            tracing::warn!(error = %reason, "drizzle skipped for this stack; the master is unaffected");
            if !bitmaps_for_drizzle {
                warn(ctx, format!("drizzle skipped: rejection bitmaps unavailable ({reason}); the master is unaffected"));
            }
        } else {
            let attempt = drizzle::Inputs {
                cfg,
                settings,
                members: &members,
                output: &output,
                ln: ln_by_member.as_deref(),
                rej: rej_set.as_ref(),
                mosaics: mosaics.as_deref(),
                width,
                height,
                channels,
                master_cards: &cards,
                out_dir: &out_dir,
                format,
            };
            match drizzle::drizzle(ctx, &attempt) {
                Ok(built) => {
                    moves.extend(built.moves);
                    drizzle_report = Some(built.report);
                }
                Err(drizzle::Failure::Cancelled) => {
                    remove_built(&moves);
                    bail!("cancelled");
                }
                Err(drizzle::Failure::Other(msg)) => {
                    tracing::warn!(error = %msg, "drizzle failed; the master is unaffected");
                    warn(ctx, format!("drizzle failed: {msg}; the master is unaffected"));
                }
            }
        }
    }
    let products: Vec<PathBuf> = moves.iter().map(|(_, dest)| dest.clone()).collect();
    let entry = Entry::new(Stage::Stack, cfg, files, products.clone());
    if let Err(e) = commit_stage(ctx.work_dir, &moves, entry) {
        remove_built(&moves);
        return Err(e);
    }
    // Maps and drizzled outputs an earlier stack left beside the master describe another master.
    let earlier = ["rejlow", "rejhigh"].map(|suffix| rejection_map_path(out_dir, suffix)).into_iter().chain(drizzle::possible_products(out_dir, format));
    for stale in earlier {
        if !products.contains(&stale) && stale.is_file() {
            if let Err(e) = std::fs::remove_file(&stale) {
                tracing::warn!(path = %stale.display(), error = %e, "cannot remove an earlier stack's output");
            }
        }
    }

    let frames_integrated: Vec<PathBuf> = output.included.iter().map(|&i| members[i].path.clone()).collect();
    tracing::info!(
        frames = output.stats.included,
        dropped_below_min_weight = output.stats.dropped_below_min_weight,
        ln_frames = output.stats.ln_frames,
        anchor = %members[anchor].path.display(),
        reference = %reference.display(),
        path = %master.display(),
        "stack stage written"
    );
    Ok(StackReport {
        rejection_low: output.rejection_low.is_some().then(|| rejection_map_path(out_dir, "rejlow")),
        rejection_high: output.rejection_high.is_some().then(|| rejection_map_path(out_dir, "rejhigh")),
        master,
        weights_file: stack_dir(work).join(WEIGHTS_FILE),
        weights,
        frames: frames_integrated,
        anchor: members[anchor].path.clone(),
        reference,
        stats: output.stats,
        drizzle: drizzle_report,
        reweighed,
        warnings: Vec::new(),
    })
}

/// The local-normalization grids of the members, aligned with them, and the members a failed normalization leaves out.
#[derive(Default)]
struct LnPlan {
    /// `None` when local normalization does not apply to the stack (off, or `normalize` could not run for the group).
    grids: Option<Vec<Option<LnFrameGrids>>>,
    /// Indices into the members, each with the reason `normalize` gave.
    dropped: Vec<(usize, String)>,
}

/// Each member's `.athln` grids under [`ln_runs`], by the status `normalize` gave the frame in its report (`normalize`
/// when the caller passes one, else `<work>/ln/normalize.json`):
/// - a report whose status is [`NormalizeStatus::Off`] (local normalization could not run for the group): no grids at
///   all, the whole stack on the global normalization, with a warning, as Athenaeum's run falls back;
/// - `excluded` (its own normalization failed while local normalization drove the output): the frame is dropped;
/// - `keptGlobal` (it failed while local normalization drove rejection only) and `globalLowOverlap` (too little overlap
///   with the LN reference): no grid, the frame stacks on the global normalization;
/// - `normalized`, or no row (the frame was not a member when `normalize` ran): its `.athln` must be readable.
///
/// No report at all, a report of a `normalize` that had nothing to do, and a missing `.athln` are errors naming the
/// frames and `starloom normalize` (P2-R24). Nothing in the report is compared with this stack's config.
fn ln_grids(
    ctx: &Ctx<'_>,
    work: &Path,
    members: &[StackFrame],
    normalize: Option<&NormalizeReport>,
) -> Result<LnPlan> {
    let names = || listed(&members.iter().map(|m| m.path.display().to_string()).collect::<Vec<_>>());
    let stored;
    let report = match normalize {
        Some(r) => r,
        None => {
            let path = ln_dir(work).join(NORMALIZE_FILE);
            if !path.is_file() {
                refuse!(
                    "local normalization is on and {} is missing, so no frame has its .athln ({}); run `starloom normalize` with this config",
                    path.display(),
                    names()
                );
            }
            stored = NormalizeReport::read(&path)?;
            &stored
        }
    };
    match report.status {
        NormalizeStatus::Ran => {}
        NormalizeStatus::NotRequested => refuse!(
            "local normalization is on, and the normalize report says it was not asked for when normalize ran, so no frame \
             has its .athln ({}); run `starloom normalize` with this config",
            names()
        ),
        NormalizeStatus::Off => {
            let why = report.note.as_deref().unwrap_or("normalize could not run for the group");
            warn(ctx, format!("local normalization is off for this stack ({why}); the stack uses the global normalization"));
            return Ok(LnPlan::default());
        }
    }
    let rows: BTreeMap<&Path, &NormalizeFrame> = report.frames.iter().map(|f| (f.file.as_path(), f)).collect();
    let mut grids: Vec<Option<LnFrameGrids>> = Vec::with_capacity(members.len());
    let mut dropped: Vec<(usize, String)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for (k, member) in members.iter().enumerate() {
        let row = rows.get(member.path.as_path()).copied();
        match row.map(|r| r.status) {
            Some(FrameStatus::Excluded) => {
                let reason = row.and_then(|r| r.reason.as_deref()).unwrap_or("its local normalization failed");
                warn(ctx, format!("{}: {reason}; the frame is excluded (local normalization)", member.path.display()));
                dropped.push((k, reason.to_string()));
                grids.push(None);
            }
            Some(FrameStatus::KeptGlobal | FrameStatus::GlobalLowOverlap) => grids.push(None),
            Some(FrameStatus::Normalized) | None => {
                let path = row.and_then(|r| r.athln.clone()).unwrap_or_else(|| athln_path(work, &member.path));
                if !path.is_file() {
                    missing.push(member.path.display().to_string());
                    grids.push(None);
                    continue;
                }
                let read = LnFrameGrids::read(&path).map_err(|e| {
                    anyhow!(
                        "{}: its local normalization {} cannot be read ({e:#}); run `starloom normalize`",
                        member.path.display(),
                        path.display()
                    )
                })?;
                grids.push(Some(read));
            }
        }
    }
    if !missing.is_empty() {
        refuse!(
            "{} frame(s) have no .athln while local normalization is on: {}; run `starloom normalize` with this config",
            missing.len(),
            listed(&missing)
        );
    }
    Ok(LnPlan { grids: Some(grids), dropped })
}

/// A string card that keeps `comment` only when the whole card fits one 80-column record (Athenaeum's
/// `card_with_optional_comment`).
fn card_with_optional_comment(keyword: &str, value: &str, comment: &str) -> Result<Card> {
    let with = Card::new(keyword, CardValue::Str(value.to_string()))?.with_comment(comment);
    match format_card(&with) {
        Ok(records) if records.len() == 1 => Ok(with),
        Ok(_) | Err(FitsWriteError::CommentTooLong(_)) => Ok(Card::new(keyword, CardValue::Str(value.to_string()))?),
        Err(e) => Err(e.into()),
    }
}

/// The anchor's copy-through cards ([`MASTER_COPY_THROUGH`]), read from its header and sorted by keyword, as
/// Athenaeum's `source_cards_from_file` reads them.
fn copy_through_cards(path: &Path) -> Result<Vec<Card>> {
    let (header_format, text) = header_text(path).with_context(|| format!("cannot read the header of {}", path.display()))?;
    let map = header_key_map(header_format, &text).into_legacy_map();
    let mut cards: Vec<Card> =
        MASTER_COPY_THROUGH.iter().filter_map(|keyword| map.get(*keyword).and_then(|v| card_from_kv(keyword, v))).collect();
    cards.sort_by(|a, b| a.keyword.cmp(&b.keyword));
    Ok(cards)
}

/// The master light's cards (P2-R6), in the order of Athenaeum's `build_master_light_cards`: `IMAGETYP`, `SWCREATE`, the
/// anchor's copy-through cards (but its `EXPTIME`, and its `DATE-OBS` when the frames have dates), `NCOMBINE`, the
/// weighted `EXPTIME`, `DATE-OBS` and `DATE-END` over the integrated frames, the registration reference's
/// world-coordinate cards (the master is in its grid), then starloom's in place of every `ATH_*` card: `SLM_VER`,
/// `SLM_CFG`, `SLM_PLV`, `SLM_REF` and [`MASTER_VERSION_CARDS`].
fn master_cards(cfg: &AlgorithmConfig, members: &[StackFrame], anchor: usize, reference: &Path, output: &GroupOutput) -> Result<Vec<Card>> {
    let date_obs_first = output.included.iter().filter_map(|&i| members[i].date_obs.as_deref()).min();
    let date_obs_last = output.included.iter().filter_map(|&i| members[i].date_obs.as_deref()).max();
    let mut cards = vec![
        Card::new("IMAGETYP", CardValue::Str("Master Light".to_string()))?.with_comment("type of image"),
        Card::new("SWCREATE", CardValue::Str(format!("starloom {}", env!("CARGO_PKG_VERSION"))))?
            .with_comment("software that created this file"),
    ];
    cards.extend(
        copy_through_cards(&members[anchor].path)?
            .into_iter()
            .filter(|c| c.keyword != "EXPTIME")
            .filter(|c| date_obs_first.is_none() || c.keyword != "DATE-OBS"),
    );
    cards.push(Card::new("NCOMBINE", CardValue::Integer(output.stats.included as i64))?.with_comment("frames combined"));
    cards.push(Card::new("EXPTIME", CardValue::Real(output.stats.weighted_exposure_s))?.with_comment("weighted total exposure, s"));
    if let Some(first) = date_obs_first {
        cards.push(Card::new("DATE-OBS", CardValue::Str(first.to_string()))?);
    }
    if let Some(last) = date_obs_last {
        cards.push(Card::new("DATE-END", CardValue::Str(last.to_string()))?);
    }
    cards.extend(header_cards(reference)?.into_iter().filter(|c| is_wcs(&c.keyword)));
    cards.extend(common_cards(&config_hash(cfg))?);
    cards.push(card_with_optional_comment(SLM_REF, &encode_ident(&file_name_of(reference)), "registration reference")?);
    for card in MASTER_VERSION_CARDS {
        cards.push(version_card(card)?);
    }
    Ok(cards)
}

/// The cards of one rejection map (`Low` or `High`): its `IMAGETYP`, `BUNIT`, the master's `ROWORDER` (the map is the
/// master's grid, row for row) and the master's own `SLM_*` trace cards, as Athenaeum's `rejection_map_cards` copies
/// its trace cards.
fn rejection_map_cards(label: &str, master: &[Card]) -> Result<Vec<Card>> {
    let mut cards = vec![
        Card::new("IMAGETYP", CardValue::Str(format!("Rejection Map {label}")))?,
        Card::new("BUNIT", CardValue::Str("count".to_string()))?.with_comment("rejected-sample count"),
    ];
    for keyword in ["ROWORDER", crate::pipeline::SLM_VER, crate::pipeline::SLM_CFG, crate::pipeline::SLM_PLV, SLM_REF] {
        if let Some(c) = master.iter().find(|c| c.keyword == keyword) {
            cards.push(c.clone());
        }
    }
    Ok(cards)
}

/// Whether the master's cards say its rows run bottom-up, an absent `ROWORDER` included (Athenaeum's
/// `cards_row_order_is_bottom_up`).
fn cards_row_order_is_bottom_up(cards: &[Card]) -> bool {
    let value = cards.iter().find(|c| c.keyword == "ROWORDER").and_then(|c| match c.value.as_ref() {
        Some(CardValue::Str(s)) => Some(s.as_str()),
        _ => None,
    });
    row_order_is_bottom_up(value)
}

/// Removes the products built so far after a failure.
fn remove_built(moves: &[(PathBuf, PathBuf)]) {
    for (built, _) in moves {
        if let Err(e) = std::fs::remove_file(built) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %built.display(), error = %e, "cannot remove a half-written stack product");
            }
        }
    }
}

/// Removes a folder a failed run left, when there is one.
fn remove_dir_if_present(dir: &Path) -> Result<()> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("cannot clear {}", dir.display())),
    }
}

/// Removes the rejection-bitmap scratch folder however the stage ends.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %self.0.display(), error = %e, "cannot remove the rejection-bitmap scratch folder");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_master_copies_athenaeums_registered_copy_through_list() {
        // Athenaeum's `REGISTERED_COPY_THROUGH`, spelled out.
        let athenaeum = "EXPTIME GAIN OFFSET EGAIN XBINNING YBINNING XPIXSZ YPIXSZ CCD-TEMP SET-TEMP INSTRUME TELESCOP \
                         FOCALLEN APTDIA FILTER DATE-OBS OBJECT OBJCTRA OBJCTDEC RA DEC SITELAT SITELONG SITEELEV OBSERVER \
                         ROWORDER";
        let listed: Vec<&str> = athenaeum.split_whitespace().collect();
        assert_eq!(MASTER_COPY_THROUGH.as_slice(), listed.as_slice());
        // Every version card the master carries is one the pipeline knows, and none is a drizzle or calibration one.
        for card in MASTER_VERSION_CARDS {
            assert!(version_card(card).is_ok(), "{card}");
        }
        assert!(!MASTER_VERSION_CARDS.contains(&"SLM_DZV") && !MASTER_VERSION_CARDS.contains(&"SLM_LCV"));
    }

    #[test]
    fn a_long_reference_name_drops_the_comment_and_keeps_the_value() {
        let long = "x".repeat(60);
        let card = card_with_optional_comment(SLM_REF, &long, "registration reference").unwrap();
        assert!(card.comment.is_none());
        assert!(matches!(card.value, Some(CardValue::Str(ref v)) if *v == long));
        let short = card_with_optional_comment(SLM_REF, "c_Light_0001.fits", "registration reference").unwrap();
        assert_eq!(short.comment.as_deref(), Some("registration reference"));
    }

    #[test]
    fn a_reweigh_note_is_given_only_when_the_weight_settings_moved() {
        let cfg = crate::stacking::config::algorithm_preset(crate::stacking::config::StackingPreset::Default);
        let same = WeightSettings {
            weight_mode: cfg.measurement.weight_mode,
            formula: cfg.measurement.formula.clone(),
            keyword: cfg.measurement.keyword.clone(),
            selection: cfg.selection,
        };
        assert_eq!(reweigh_note(&cfg, &same), None);
        let other = WeightSettings { weight_mode: crate::stacking::weights::WeightMode::PsfSnr, ..same };
        let note = reweigh_note(&cfg, &other).unwrap();
        assert!(note.contains("psfSnr") && note.contains("psfSignalWeight"), "{note}");
    }
}
