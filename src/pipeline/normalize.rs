//! The `normalize` stage: the anchor, the LN reference, its background model, the seeds calibration and one `.athln`
//! per frame.
//!
//! This mirrors Athenaeum's Normalize stage (`run_group_normalization`) for one group, without the catalog, the
//! artifact cache and the project grid (P2-R3). Under `normalization.local.enabled`, or whenever the rejection
//! normalization is `local` ([`ln_runs`]), it does what the run does:
//!
//! 1. the weights come from [`current_weights`] under the caller's config (P2-R21), and the stack's members are the
//!    frames it includes that have a transform;
//! 2. the normalization anchor is the member that covers the reference rectangle and ranks first by weight over the
//!    square root of its sky ([`pick_ln_anchor`]); the LN reference is integrated from the best `referenceFrames`
//!    members in that same order, with the anchor's measurement normalizing the others;
//! 3. the reference's background model is built once per plane, its stars are detected and fitted once, at the group β
//!    under `psfModel = auto` (Measure's fits were made at it) or at the profile's own β;
//! 4. the seeds calibration `k` is measured over a stratified sample of the members, the registration reference first;
//! 5. every member is normalized against the reference, with Measure's star fits as the seeds of its scale, and its
//!    grids are written as `<work>/ln/<stem>.athln`.
//!
//! Products, under `<work>/ln/`: `reference.fits`, `reference-saturation.fits`, `<stem>.athln` for every frame that was
//! normalized, and `normalize.json` ([`NormalizeReport`]). A frame whose own normalization failed is excluded when
//! local normalization drives the output and keeps the global normalization otherwise, as in Athenaeum's run; the
//! report says which, and `stack` reads it.
//!
//! When local normalization cannot run for the group (fewer than 3 frames to build the reference from, a refused
//! background model, or a failed reference build) Athenaeum's run warns and stacks the group on the global
//! normalization. Here that outcome is written down: `<work>/ln/` is replaced by a folder holding only `normalize.json`,
//! with status [`NormalizeStatus::Off`] and the reason, and its manifest entry, so that no earlier run's sidecar is left
//! for `stack` to pick up and `stack` stacks on the global normalization.
//!
//! The stage works from the unwarped calibrated frames and their transforms: the registered copies `register
//! --write-registered` may have written are not read, because they hold the very pixels the warp gives and a stale
//! copy would be used without anyone noticing (P2-R24). A frame's id inside the stage is its index among the frames it
//! was given (the pipeline order), which stands for the catalog id Athenaeum breaks ties on (P2-R10).
//!
//! Known divergences from Athenaeum's run: there is no project grid, so the masked reference never applies, and with it
//! the low-overlap rule (it reads a masked reference's overlap), so this stage never writes a
//! [`FrameStatus::GlobalLowOverlap`] row; a group of fewer than 3 members is an error, where the run skips it; the
//! anchor tie and the seeds calibration's tie break on the frame's index in the pipeline order, where the run breaks
//! them on the catalog id (P2-R3, P2-R10). Everything that decodes a pixel or fits a star runs on the context's
//! pool (SL-9).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::FrameFacts;

use crate::integration::plane_reader::PlaneReader;
use crate::integration::IntegrationError;
use crate::pipeline::formats::{
    commit_stage, part_path, Entry, FrameStatus, MeasureReport, NormalizeFrame, NormalizeReport, NormalizeStatus,
    TransformFrame, NORMALIZE_FILE, NORMALIZE_FORMAT, NORMALIZE_REPORT_VERSION,
};
use crate::pipeline::master::io_policy;
use crate::pipeline::measure::fan_out;
use crate::pipeline::register::RegisterReport;
use crate::pipeline::versions::recorded_versions;
use crate::pipeline::weights::{current_weights, subset_measure};
use crate::pipeline::{athf_path, common_cards, config_hash, frame_stem, ln_runs, source_facts, Ctx, Manifest, Stage};
use crate::resample::Interpolation;
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::fits_artifact::read_fits;
use crate::stacking::integrate::{GroupInput, StackFrame};
use crate::stacking::ln::{
    build_reference, build_reference_saturation, effective_scale_mode, measure_seeds_calibration, normalize_frame,
    resolve_scale, select_calibration_frames, write_reference, write_reference_saturation, BackgroundModel, ChannelOutcome,
    LnError, LnFrameGrids, LnFrameOutcome, LnReference, LnReferenceForDetection, LnScaleSeeds, SeedsCalibration, TileTable,
    CLEANING, LN_MIN_OVERLAP, SEEDS_CALIBRATION_FRAMES,
};
use crate::stacking::psf_signal::{PsfModel, StarFit};
use crate::stacking::run_helpers::{pick_ln_anchor, AnchorPick};
use crate::stacking::weights::{reference_coverage, sky_penalized_order, FrameWeight, MIN_REFERENCE_COVERAGE};

/// The stage's folder under the working folder.
pub const NORMALIZE_DIR: &str = "ln";
/// The LN reference, under [`NORMALIZE_DIR`].
pub const LN_REFERENCE_FILE: &str = "reference.fits";
/// The reference-frames saturation map, under [`NORMALIZE_DIR`].
pub const LN_REFERENCE_SATURATION_FILE: &str = "reference-saturation.fits";
/// The scratch folder the seeds calibration writes its throwaway sidecars in, inside the stage's own folder while it is
/// being built (the measurement removes it).
const CALIBRATION_SCRATCH_DIR: &str = "calibration";

/// Planes of one channel resident while a worker normalizes a frame (Athenaeum's `LN_PLANES_RESIDENT`, measured there).
const LN_PLANES_RESIDENT: u64 = 8;

/// The stage's folder: `<work>/ln`.
pub fn ln_dir(work_dir: &Path) -> PathBuf {
    work_dir.join(NORMALIZE_DIR)
}

/// Where `normalize` leaves the grids of `frame`: `<work>/ln/<stem>.athln`. The file is missing for a frame that was not
/// normalized (it kept the global normalization, was excluded, or was not a member).
pub fn athln_path(work_dir: &Path, frame: &Path) -> PathBuf {
    ln_dir(work_dir).join(format!("{}.athln", frame_stem(frame)))
}

/// The star profile the LN reference is fitted with: under `auto` the group's β (Measure's fits were made at it, so
/// fitting the reference at anything else would bring back the β mismatch the group β removes), and any explicit
/// profile as it is (`moffat4`, `{"moffat": β}`).
fn reference_psf(model: PsfModel, group_beta: f64) -> PsfModel {
    match model {
        PsfModel::Auto => PsfModel::Fixed(group_beta),
        other => other,
    }
}

/// The β a profile fits at, for the report; `None` for `auto`, which [`reference_psf`] has already resolved.
fn beta_of(model: PsfModel) -> Option<f64> {
    match model {
        PsfModel::Moffat4 => Some(4.0),
        PsfModel::Fixed(beta) | PsfModel::Moffat(beta) => Some(beta),
        PsfModel::Auto => None,
    }
}

/// A member's measurements, as the anchor pick and the reference ranking read them (`stack` picks its anchor with it
/// too).
pub(crate) struct Ranking {
    weights: Vec<FrameWeight>,
    star_counts: Vec<usize>,
    sky: Vec<f64>,
    coverage: Vec<f64>,
}

impl Ranking {
    pub(crate) fn of(members: &[&StackFrame], ref_w: usize, ref_h: usize) -> Ranking {
        Ranking {
            weights: members.iter().map(|m| m.weight.clone()).collect(),
            star_counts: members.iter().map(|m| m.measurement.min_stars()).collect(),
            sky: members.iter().map(|m| m.measurement.mean_median()).collect(),
            coverage: members
                .iter()
                .map(|m| reference_coverage(&m.map, m.measurement.width, m.measurement.height, ref_w, ref_h))
                .collect(),
        }
    }

    /// The anchor over these members (`pick_reference_idx` of Athenaeum's run), with the two relaxations the pick made
    /// reported through `ctx` and the log.
    pub(crate) fn anchor(&self, ctx: &Ctx<'_>) -> Option<usize> {
        let (pick, coverage_dropped) = pick_ln_anchor(&self.weights, &self.star_counts, &self.sky, &self.coverage);
        if let Some(count) = coverage_dropped {
            tracing::warn!(count, min_coverage = MIN_REFERENCE_COVERAGE, "reference-coverage filter dropped for the anchor: fewer than 3 members pass");
            ctx.progress.warn(&format!(
                "reference-coverage filter dropped for normalization anchoring: only {count} member(s) cover the reference geometry (need >= 3)"
            ));
        }
        let Some(AnchorPick { idx, sky_fallback, .. }) = pick else {
            tracing::warn!("no admissible member has a usable measured background; no anchor");
            return None;
        };
        if let Some(count) = sky_fallback {
            tracing::warn!(count, "no admissible member has a usable measured background; anchoring by weight alone");
            ctx.progress.warn(&format!(
                "no admissible member has a usable measured background: anchored by weight alone among {count} candidate(s)"
            ));
        }
        tracing::info!(weight = self.weights[idx].normalized_mean, background = self.sky[idx], "normalization anchor chosen");
        Some(idx)
    }
}

/// A frame's own star fits, one list per plane, for the seeds of its scale. A missing file is a plane with no seeds (the
/// scale then falls back to detection); an unreadable one is logged, at `warn` where `loud` (the calibration's frames).
fn load_fits(work: &Path, file: &Path, planes: usize, loud: bool) -> Vec<Vec<StarFit>> {
    (0..planes)
        .map(|plane| {
            let path = athf_path(work, file, plane);
            if !path.exists() {
                tracing::debug!(path = %path.display(), plane, "ln: no fits artifact; falling back to detection for this plane");
                return Vec::new();
            }
            read_fits(&path).unwrap_or_else(|e| {
                if loud {
                    tracing::warn!(path = %path.display(), error = %e, "ln calibration: fits artifact unreadable; this plane contributes no stars");
                } else {
                    tracing::debug!(path = %path.display(), error = %e, "ln: fits artifact unreadable; falling back to detection for this plane");
                }
                Vec::new()
            })
        })
        .collect()
}

/// Normalizes `frames` (calibrated plain-FITS frames `measure` has rows for and `register` has transforms for) against
/// an LN reference built from the best of them, and writes `<work>/ln/` ([module docs](self)).
///
/// `measure` holds the stored measurements, `register` the transforms; the weights and the selection are computed again
/// under `cfg` ([`current_weights`], P2-R21). A frame the config includes that has no transform is an error naming it
/// and `starloom register`; a frame with no row in `measure` is one naming `starloom measure` (P2-R24). Fewer than 3
/// members is an error too.
///
/// When local normalization is not asked for ([`ln_runs`]) the stage does nothing: the report's status is
/// [`NormalizeStatus::NotRequested`], no file is written and no manifest entry is left. When it is asked for and cannot
/// run for the group (fewer than 3 frames to build the reference from, `referenceFrames` under 3, its background model
/// refused, or the reference's build failed) the status is [`NormalizeStatus::Off`], with a warning worded as
/// Athenaeum's run words it, and the outcome is written: `<work>/ln/` is replaced by a folder holding only
/// `normalize.json` (so no earlier run's sidecars are left), with its manifest entry, and `stack` then stacks the group
/// on the global normalization, as Athenaeum's run does. A cancel is an error, never `off`.
///
/// The products are built in `<work>/ln.part/` and renamed into place together with the manifest entry
/// ([`commit_stage`]), so a failure or a cancel leaves no entry and no half-written folder. Everything that decodes a
/// pixel or fits a star runs on the context's pool (SL-9).
pub fn normalize(
    ctx: &Ctx<'_>,
    frames: &[PathBuf],
    measure: &MeasureReport,
    register: &RegisterReport,
    cfg: &AlgorithmConfig,
) -> Result<NormalizeReport> {
    if frames.is_empty() {
        bail!("no frames to normalize");
    }
    if !ln_runs(cfg) {
        let note = "local normalization is not asked for (normalization.local.enabled is off and normalization.rejection is not local); nothing to do".to_string();
        tracing::info!("{note}");
        return Ok(NormalizeReport::idle(NormalizeStatus::NotRequested, note, cfg.clone(), recorded_versions()));
    }
    let work = std::path::absolute(ctx.work_dir).with_context(|| format!("cannot make {} absolute", ctx.work_dir.display()))?;
    let sub = subset_measure(measure, frames)?;
    ctx.check_cancelled()?;
    // Everything below decodes or detects on the caller's pool, never rayon's global one (SL-9).
    ctx.pool.install(|| run(ctx, &work, &sub, register, cfg))
}

/// Why local normalization cannot run for a group: Athenaeum's `LnFallback`, without the group key.
enum Fallback<'a> {
    /// `referenceFrames` is under 3.
    ReferenceFramesBelowThree { reference_frames: u32 },
    /// Fewer than 3 members can build the reference.
    TooFewMembers { count: usize },
    /// The reference's background model is refused on a plane.
    BackgroundRefused { reason: &'a str },
    /// Building the reference or its products failed.
    BuildFailed { msg: &'a str },
}

/// The warning for a fallback, worded as Athenaeum's `ln_fallback_warning` (`rejection_only` when local normalization
/// was not asked for and only drives the rejection).
fn fallback_warning(rejection_only: bool, what: &Fallback<'_>) -> String {
    const TAIL: &str = "rejection used the global normalization, which can clip star cores where seeing varies";
    match (rejection_only, what) {
        (false, Fallback::BuildFailed { msg }) => format!("local normalization failed: {msg}"),
        (true, Fallback::BuildFailed { msg }) => {
            format!("local rejection normalization could not be computed ({msg}); {TAIL}")
        }
        (false, Fallback::ReferenceFramesBelowThree { reference_frames }) => {
            format!("local normalization off \u{2014} referenceFrames ({reference_frames}) < 3; using global normalization")
        }
        (true, Fallback::ReferenceFramesBelowThree { reference_frames }) => {
            format!("local rejection normalization could not be computed \u{2014} referenceFrames ({reference_frames}) < 3; {TAIL}")
        }
        (false, Fallback::TooFewMembers { count }) => format!(
            "local normalization off \u{2014} only {count} member(s) can build the LN reference (need >= 3); using global normalization"
        ),
        (true, Fallback::TooFewMembers { count }) => format!(
            "local rejection normalization could not be computed \u{2014} only {count} member(s) can build the reference (need >= 3); {TAIL}"
        ),
        (false, Fallback::BackgroundRefused { reason }) => format!(
            "local normalization off \u{2014} the LN reference's background model is refused ({reason}); using global normalization"
        ),
        (true, Fallback::BackgroundRefused { reason }) => format!(
            "local rejection normalization could not be computed \u{2014} the reference frame's background model is refused ({reason}); {TAIL}"
        ),
    }
}

/// The outcome of a group local normalization cannot run for: the warning, and `<work>/ln/` replaced by a folder that
/// holds only `normalize.json` with status [`NormalizeStatus::Off`] (so an earlier run's sidecars, reference and entry
/// are gone), committed with the stage's manifest entry.
fn off(ctx: &Ctx<'_>, work: &Path, cfg: &AlgorithmConfig, files: &[PathBuf], what: Fallback<'_>) -> Result<NormalizeReport> {
    let note = fallback_warning(!cfg.normalization.local.enabled, &what);
    tracing::warn!("{note}");
    ctx.progress.warn(&note);
    let report = NormalizeReport::idle(NormalizeStatus::Off, note, cfg.clone(), recorded_versions());
    let final_dir = ln_dir(work);
    let part_dir = part_path(&final_dir);
    remove_dir_if_present(&part_dir)?;
    std::fs::create_dir_all(&part_dir).with_context(|| format!("cannot create {}", part_dir.display()))?;
    report.write(&part_dir.join(NORMALIZE_FILE)).inspect_err(|_| remove_part(&part_dir))?;
    let entry = Entry::new(Stage::Normalize, cfg, files.to_vec(), vec![final_dir.join(NORMALIZE_FILE)]);
    if let Err(e) = commit_stage(ctx.work_dir, &[(part_dir.clone(), final_dir)], entry) {
        remove_part(&part_dir);
        return Err(e);
    }
    Ok(report)
}

fn run(ctx: &Ctx<'_>, work: &Path, sub: &MeasureReport, register: &RegisterReport, cfg: &AlgorithmConfig) -> Result<NormalizeReport> {
    let ln_cfg = cfg.normalization.local.clone();
    let files: Vec<PathBuf> = sub.frames.iter().map(|f| f.file.clone()).collect();
    let weights = current_weights(ctx, sub, register, cfg)?;
    let manifest = Manifest::load(ctx.work_dir)?;
    let sources: BTreeMap<PathBuf, FrameFacts> = source_facts(&manifest, &files)?;

    // The members: included, measured, registered (`GroupMember`s of Athenaeum's output stage), in frame order.
    let rows: BTreeMap<&Path, &TransformFrame> = register.transforms.frames.iter().map(|t| (t.file.as_path(), t)).collect();
    let mut stack_frames: Vec<StackFrame> = Vec::new();
    for (index, w) in weights.frames.iter().enumerate() {
        if !w.included {
            continue;
        }
        let (Some(measurement), Some(map)) = (w.measurement.as_ref(), rows.get(w.file.as_path()).and_then(|t| t.map.as_ref())) else {
            continue;
        };
        let facts = sources.get(&w.file);
        stack_frames.push(StackFrame {
            path: w.file.clone(),
            map: map.clone(),
            measurement: measurement.clone(),
            weight: w.weight.clone().unwrap_or(FrameWeight {
                channels: Vec::new(),
                normalized: Vec::new(),
                mean: 0.0,
                normalized_mean: 0.0,
                missing: None,
            }),
            exposure_s: facts.and_then(|f| f.exptime_s).unwrap_or(0.0),
            date_obs: facts.and_then(|f| f.date_obs.clone()),
            frame_id: index as i64,
            // The warp is applied on the fly: a registered copy on disk is the same pixels, and nothing says it is
            // still the one this config's transforms made.
            registered_path: None,
        });
    }
    if stack_frames.len() < 3 {
        bail!(
            "local normalization needs at least 3 included, registered frames; {} of {} qualify under this config",
            stack_frames.len(),
            files.len()
        );
    }

    // The reference geometry: the registration reference's grid.
    let reference_file = &register.transforms.reference;
    let (ref_w, ref_h) = {
        let reader = PlaneReader::open(reference_file)
            .map_err(|e| anyhow!("cannot read the registration reference {}: {e}", reference_file.display()))?;
        (reader.width(), reader.height())
    };
    let frame_refs: Vec<&StackFrame> = stack_frames.iter().collect();
    let ranking = Ranking::of(&frame_refs, ref_w, ref_h);

    // Spec §2.3 (L-10): the one scale rule, on this group's LN geometry.
    let scale_px = resolve_scale(&ln_cfg, ref_w, ref_h);
    tracing::info!(
        ln_scale_px = scale_px,
        ln_scale_mode = effective_scale_mode(&ln_cfg).as_str(),
        ln_flux_estimator = ln_cfg.flux_estimator.as_str(),
        width = ref_w,
        height = ref_h,
        "ln scale resolved"
    );

    // The anchor, then the LN reference's members: the same sky-penalized ranking over the members that cover the
    // reference rectangle (the filter is dropped, with a warning, for a group where fewer than 3 pass).
    let Some(anchor) = ranking.anchor(ctx) else {
        bail!("no included frame can anchor the normalization");
    };
    let coverage_mask: Vec<bool> = ranking.coverage.iter().map(|&c| c >= MIN_REFERENCE_COVERAGE).collect();
    let admissible = coverage_mask.iter().filter(|&&ok| ok).count();
    let included_mask = if admissible >= 3 {
        coverage_mask
    } else {
        let message = format!(
            "reference-coverage filter dropped for the LN reference's member list: only {admissible} member(s) cover the reference geometry (need >= 3)"
        );
        tracing::warn!(count = admissible, min_coverage = MIN_REFERENCE_COVERAGE, "reference-coverage filter dropped for this group: fewer than 3 members pass");
        ctx.progress.warn(&message);
        vec![true; stack_frames.len()]
    };
    let ranked = sky_penalized_order(&ranking.weights, &included_mask, &ranking.star_counts, &ranking.sky);
    let n = (ln_cfg.reference_frames as usize).min(ranked.len());
    if n < 3 {
        let what = if (ln_cfg.reference_frames as usize) < 3 {
            Fallback::ReferenceFramesBelowThree { reference_frames: ln_cfg.reference_frames }
        } else {
            Fallback::TooFewMembers { count: ranked.len() }
        };
        return off(ctx, work, cfg, &files, what);
    }
    let reference_members: Vec<usize> = ranked[..n].to_vec();
    let member_file = |k: usize| stack_frames[k].path.clone();

    let final_dir = ln_dir(work);
    let part_dir = part_path(&final_dir);
    remove_dir_if_present(&part_dir)?;
    std::fs::create_dir_all(&part_dir).with_context(|| format!("cannot create {}", part_dir.display()))?;
    let built = build(
        ctx,
        work,
        cfg,
        &part_dir,
        &stack_frames,
        anchor,
        &reference_members,
        n,
        (ref_w, ref_h),
        scale_px,
        sub.resolved_group_beta(),
        reference_file,
    );
    let outcome = match built {
        Ok(outcome) => outcome,
        Err(e) => {
            remove_part(&part_dir);
            // A cancel stays an error; any other failure of the build is Athenaeum's "warn and stack on global".
            ctx.check_cancelled()?;
            let msg = format!("{e:#}");
            tracing::warn!(error = %msg, "local normalization failed for this group; falling back to global normalization");
            return off(ctx, work, cfg, &files, Fallback::BuildFailed { msg: &msg });
        }
    };
    let (mut report, normalized) = match outcome {
        Built::Off(reason) => {
            remove_part(&part_dir);
            return off(ctx, work, cfg, &files, Fallback::BackgroundRefused { reason: &reason });
        }
        Built::Ran { report, normalized } => (report, normalized),
    };
    report.reference_members = reference_members.iter().map(|&k| member_file(k)).collect();
    report.anchor = Some(member_file(anchor));

    // The anchor over what the exclusions left (Athenaeum's run picks it again from the survivors).
    let excluded = report.frames.iter().filter(|f| f.status == FrameStatus::Excluded).count();
    if excluded > 0 {
        let survivors: Vec<usize> =
            (0..stack_frames.len()).filter(|&k| report.frames[k].status != FrameStatus::Excluded).collect();
        if survivors.len() < 3 {
            let message = format!(
                "local normalization left {} frame(s); a stack needs at least 3",
                survivors.len()
            );
            tracing::warn!("{message}");
            ctx.progress.warn(&message);
        } else {
            let refs: Vec<&StackFrame> = survivors.iter().map(|&k| &stack_frames[k]).collect();
            report.anchor_after_exclusions = Ranking::of(&refs, ref_w, ref_h).anchor(ctx).map(|i| member_file(survivors[i]));
        }
    }

    let mut products = vec![final_dir.join(LN_REFERENCE_FILE), final_dir.join(LN_REFERENCE_SATURATION_FILE)];
    products.extend(normalized.iter().map(|&k| athln_path(work, &member_file(k))));
    products.push(final_dir.join(NORMALIZE_FILE));
    report.reference = Some(final_dir.join(LN_REFERENCE_FILE));
    report.reference_saturation = Some(final_dir.join(LN_REFERENCE_SATURATION_FILE));
    report.write(&part_dir.join(NORMALIZE_FILE)).inspect_err(|_| remove_part(&part_dir))?;
    let entry = Entry::new(Stage::Normalize, cfg, files.clone(), products);
    if let Err(e) = commit_stage(ctx.work_dir, &[(part_dir.clone(), final_dir)], entry) {
        remove_part(&part_dir);
        return Err(e);
    }
    tracing::info!(
        frames = report.frames.len(),
        normalized = normalized.len(),
        reference_frames = report.reference_members.len(),
        ln_scale_px = scale_px,
        "normalize stage written"
    );
    Ok(report)
}

/// What the build left: the report (without the paths and lists [`run`] fills in) and the members that have a sidecar,
/// or the reason the reference's background model was refused (local normalization cannot run for the group).
enum Built {
    Off(String),
    Ran { report: NormalizeReport, normalized: Vec<usize> },
}

/// The LN reference, its saturation map and background models, the seeds calibration and every member's sidecar, built
/// into `part_dir`.
#[allow(clippy::too_many_arguments)]
fn build(
    ctx: &Ctx<'_>,
    work: &Path,
    cfg: &AlgorithmConfig,
    part_dir: &Path,
    stack_frames: &[StackFrame],
    anchor: usize,
    reference_members: &[usize],
    n: usize,
    (ref_w, ref_h): (usize, usize),
    scale_px: u32,
    group_beta: Option<f64>,
    registration_reference: &Path,
) -> Result<Built> {
    let ln_cfg = cfg.normalization.local.clone();
    let ln_drives_output = ln_cfg.enabled;
    let channels = stack_frames[anchor].measurement.channels.len();
    let interpolation: Interpolation = cfg.registration.interpolation;
    let clamping = cfg.registration.clamping_threshold;
    let measure_opts = cfg.measurement.measure_options(cfg.normalization.scale_estimator);
    let input = GroupInput {
        frames: stack_frames,
        reference: anchor,
        width: ref_w,
        height: ref_h,
        channels,
        interpolation,
        clamping,
        integration: &cfg.integration,
        normalization: &cfg.normalization,
        ln: None,
        rej: None,
        second_pass_bitmaps: false,
        coverage_map: false,
        stats_on_covered: false,
        common_stars: None,
    };
    let io = io_policy(ctx);
    let cards = common_cards(&config_hash(cfg))?;

    // The reference: the best `n` members, integrated; then its saturation map, written beside it.
    let integration_error = |what: &str, e: IntegrationError| match e {
        IntegrationError::Cancelled => anyhow!("cancelled"),
        other => anyhow!("{what}: {other}"),
    };
    let ln_reference: LnReference = build_reference(&input, reference_members, n, &ctx.pool, ctx.cancel, io, &|_, _| {}, false)
        .map_err(|e| integration_error("LN reference build failed", e))?;
    write_reference(&ln_reference, &part_dir.join(LN_REFERENCE_FILE), &cards)?;
    let mut maps = build_reference_saturation(&input, &ln_reference.frames_used, &ctx.pool, ctx.cancel)
        .map_err(|e| integration_error("LN reference saturation map failed", e))?;
    let saturation = ln_cfg.star_selection.saturation;
    let masks = write_reference_saturation(
        &mut maps,
        ln_reference.width,
        ln_reference.height,
        &part_dir.join(LN_REFERENCE_SATURATION_FILE),
        &cards,
        saturation,
    )?;
    ctx.check_cancelled()?;

    // The reference's background, once per plane. A plane the model refuses makes no frame's sidecar trustworthy.
    let mut ref_models: Vec<BackgroundModel> = Vec::with_capacity(ln_reference.planes.len());
    for (p, plane) in ln_reference.planes.iter().enumerate() {
        let table = TileTable::measure(plane, ln_reference.width, ln_reference.height, &CLEANING, Some(&ctx.pool));
        match BackgroundModel::fit(table, scale_px) {
            Ok(m) => {
                tracing::debug!(
                    channel = p,
                    ln_samples = m.samples,
                    ln_samples_rejected = m.samples_rejected,
                    ln_cells_unsupported = m.cells_unsupported,
                    "ln reference background modelled"
                );
                ref_models.push(m);
            }
            Err(r) => {
                tracing::debug!(channel = p, error = %r, "ln reference background refused");
                return Ok(Built::Off(r.to_string()));
            }
        }
    }

    // The reference's stars, detected and fitted once, at the group β under `auto`.
    let beta = group_beta.unwrap_or(4.0);
    let psf = reference_psf(ln_cfg.psf_model, beta);
    let reference_for_detection = {
        let _span = tracing::debug_span!("ln_reference").entered();
        LnReferenceForDetection::build(&ln_reference, psf, measure_opts.max_stars, Some(&ctx.pool), Some(masks))
    };
    ctx.check_cancelled()?;

    // How many frames run at once: the stage's own working set per worker, within the context's budget.
    let working_set = LN_PLANES_RESIDENT * ref_w as u64 * ref_h as u64 * 4;
    let concurrency = usize::try_from(ctx.memory_budget_bytes / working_set.max(1))
        .unwrap_or(usize::MAX)
        .clamp(1, ctx.pool.current_num_threads().max(1));

    // The seeds calibration `k` (ruling C-14), over the registration reference and a stratified sample of the rest.
    let reference_frame_id = stack_frames
        .iter()
        .find(|f| same_file(&f.path, registration_reference))
        .map_or(-1, |f| f.frame_id);
    let calibration_weights: Vec<(i64, f64)> = stack_frames.iter().map(|f| (f.frame_id, f.weight.normalized_mean)).collect();
    let calibration_indices = select_calibration_frames(reference_frame_id, &calibration_weights, SEEDS_CALIBRATION_FRAMES);
    let total_units = stack_frames.len() + calibration_indices.len();
    let done = AtomicUsize::new(0);
    let tick = || ctx.progress.step("normalize", done.fetch_add(1, Ordering::Relaxed) + 1, total_units);
    let seeds_calibration: Option<SeedsCalibration> = if calibration_indices.is_empty() {
        None
    } else {
        let load = |i: usize| load_fits(work, &stack_frames[i].path, stack_frames[i].measurement.channels.len(), true);
        let run_jobs = |count: usize, job: &(dyn Fn(usize) -> Option<LnFrameOutcome> + Sync)| -> Vec<Option<LnFrameOutcome>> {
            fan_out(ctx, count, concurrency, job).into_iter().map(Option::flatten).collect()
        };
        let on_progress = |_done: usize, _total: usize, _frame_id: i64| tick();
        let measurement = measure_seeds_calibration(
            &ln_reference,
            &reference_for_detection,
            &ref_models,
            stack_frames,
            &calibration_indices,
            &load,
            &ln_cfg,
            scale_px,
            &measure_opts,
            interpolation,
            clamping,
            &part_dir.join(CALIBRATION_SCRATCH_DIR),
            Some(&ctx.pool),
            ctx.cancel,
            &on_progress,
            &run_jobs,
        );
        // A cancelled measurement returns `None` silently; the stage's own cancel check is what stops it.
        ctx.check_cancelled()?;
        measurement.map(|m| {
            report_calibration(&m.channels);
            m.calibration()
        })
    };

    // Every member against the reference.
    let results = fan_out(ctx, stack_frames.len(), concurrency, |k| {
        let frame = &stack_frames[k];
        let fits = load_fits(work, &frame.path, frame.measurement.channels.len(), false);
        let sidecar = part_dir.join(format!("{}.athln", frame_stem(&frame.path)));
        let out = normalize_frame(
            &ln_reference,
            &reference_for_detection,
            &ref_models,
            frame,
            LnScaleSeeds::Measured(&fits),
            &ln_cfg,
            scale_px,
            &measure_opts,
            interpolation,
            clamping,
            &sidecar,
            seeds_calibration.as_ref(),
            Some(&ctx.pool),
            ctx.cancel,
        );
        tick();
        out
    });
    ctx.check_cancelled()?;

    let mut report = NormalizeReport {
        format: NORMALIZE_FORMAT.to_string(),
        version: NORMALIZE_REPORT_VERSION,
        status: NormalizeStatus::Ran,
        note: None,
        config: cfg.clone(),
        versions: recorded_versions(),
        local_drives_output: ln_drives_output,
        anchor: None,
        anchor_after_exclusions: None,
        reference_members: Vec::new(),
        calibration_frames: calibration_indices.iter().map(|&i| stack_frames[i].path.clone()).collect(),
        k: seeds_calibration.as_ref().map(|c| c.k.clone()).unwrap_or_default(),
        reference_beta: beta_of(psf),
        scale_px: Some(scale_px),
        reference: None,
        reference_saturation: None,
        frames: Vec::with_capacity(stack_frames.len()),
    };
    let mut normalized: Vec<usize> = Vec::new();
    let (mut aperture_shortfalls, mut low_overlap) = (0usize, 0usize);
    for (k, res) in results.into_iter().enumerate() {
        let file = stack_frames[k].path.clone();
        let mut row = NormalizeFrame {
            file: file.clone(),
            status: FrameStatus::Normalized,
            athln: None,
            scale: None,
            matches: None,
            scale_source: None,
            overlap: None,
            reason: None,
        };
        // A frame whose own normalization failed: excluded when local normalization drives the output, else kept on the
        // global normalization with a warning.
        let fail = |row: &mut NormalizeFrame, reason: String| {
            if ln_drives_output {
                tracing::warn!(path = %file.display(), reason = %reason, "frame excluded: local normalization");
                ctx.progress.warn(&format!("{}: {reason}; the frame is excluded", file.display()));
                row.status = FrameStatus::Excluded;
            } else {
                tracing::warn!(path = %file.display(), reason = %reason, "local normalization failed; frame keeps global normalization");
                ctx.progress.warn(&format!(
                    "{}: local rejection normalization could not be computed ({reason}); this frame's rejection used the global normalization",
                    file.display()
                ));
                row.status = FrameStatus::KeptGlobal;
            }
            row.reason = Some(reason);
        };
        match res.ok_or_else(|| anyhow!("cancelled"))? {
            Ok(outcome) => {
                // Read the sidecar straight back: a file that was written and cannot be read is no normalization.
                match LnFrameGrids::read(&outcome.sidecar) {
                    Ok(_) => {
                        tracing::debug!(
                            path = %file.display(),
                            ln_scale = outcome.scale,
                            ln_matches = outcome.matches,
                            ln_samples = outcome.samples,
                            ln_samples_rejected = outcome.samples_rejected,
                            ln_samples_shared = outcome.samples_shared,
                            ln_cells_unsupported = outcome.cells_unsupported,
                            ln_scale_source = outcome.ln_scale_source,
                            "ln frame normalized"
                        );
                        row.athln = Some(athln_path(work, &file));
                        row.scale = Some(outcome.scale);
                        row.matches = Some(outcome.matches);
                        row.scale_source = Some(outcome.ln_scale_source.to_string());
                        normalized.push(k);
                    }
                    Err(e) => {
                        if let Err(rm) = std::fs::remove_file(&outcome.sidecar) {
                            tracing::warn!(path = %outcome.sidecar.display(), error = %rm, "cannot remove an unreadable sidecar");
                        }
                        fail(&mut row, format!("local normalization: writing .athln sidecar: reading it back failed: {e:#}"));
                    }
                }
            }
            Err(LnError::LowOverlap { fraction }) => {
                tracing::debug!(path = %file.display(), overlap = fraction, "ln: frame keeps global normalization (little overlap with the LN reference)");
                row.status = FrameStatus::GlobalLowOverlap;
                row.overlap = Some(fraction);
                low_overlap += 1;
            }
            Err(e) => {
                if ln_drives_output && matches!(e, LnError::CommonApertureShortfall { .. }) {
                    aperture_shortfalls += 1;
                }
                fail(&mut row, e.to_string());
            }
        }
        report.frames.push(row);
    }
    if aperture_shortfalls > 0 {
        let message = format!(
            "{aperture_shortfalls} frame(s) were left out of local normalization because the common aperture could not measure enough stars on them; Star flux: PSF signal may keep them"
        );
        tracing::warn!(count = aperture_shortfalls, "local normalization excluded frames the common aperture could not measure");
        ctx.progress.warn(&message);
    }
    if low_overlap > 0 {
        let percent = LN_MIN_OVERLAP * 100.0;
        let message = if ln_drives_output {
            format!("local normalization kept {low_overlap} frame(s) on global normalization: under {percent:.0} % overlap with the LN reference")
        } else {
            format!("{low_overlap} frame(s) used the global normalization for rejection: under {percent:.0} % overlap with the reference frame")
        };
        tracing::warn!(count = low_overlap, "local normalization kept frames with little overlap on global normalization");
        ctx.progress.warn(&message);
    }
    Ok(Built::Ran { report, normalized })
}

/// The log lines of a seeds calibration, one per plane that has something to say (ruling C-14 item 1).
fn report_calibration(channels: &[crate::stacking::ln::ChannelCalibration]) {
    let short = channels.iter().filter(|c| c.outcome == ChannelOutcome::TooFewRatios).map(|c| c.samples).min();
    if let Some(count) = short {
        tracing::warn!(count, "ln seeds calibration: too few usable calibration frames; the seeds path runs uncalibrated for this group");
    }
    for (channel, c) in channels.iter().enumerate() {
        match c.outcome {
            ChannelOutcome::Calibrated => {
                tracing::info!(channel, count = c.samples, ln_scale = c.k, "ln seeds calibration");
            }
            ChannelOutcome::OutOfBand { measured } => {
                tracing::warn!(channel, count = c.samples, ln_scale = measured, "ln seeds calibration: the measured factor left its band; this channel runs uncalibrated");
            }
            ChannelOutcome::TooFewRatios => {}
        }
    }
}

/// Whether two paths name one file: equal as given, or as the file system resolves them.
fn same_file(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}

/// Removes a folder a failed run left, when there is one.
fn remove_dir_if_present(dir: &Path) -> Result<()> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("cannot clear {}", dir.display())),
    }
}

fn remove_part(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %dir.display(), error = %e, "cannot remove a half-written stage folder");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback sentences are Athenaeum's (`ln_fallback_warning`), without the group key.
    #[test]
    fn fallback_warnings_are_worded_as_the_run_words_them() {
        let cases = [
            Fallback::BuildFailed { msg: "boom" },
            Fallback::ReferenceFramesBelowThree { reference_frames: 2 },
            Fallback::TooFewMembers { count: 2 },
            Fallback::BackgroundRefused { reason: "no background sample" },
        ];
        let tail = "rejection used the global normalization, which can clip star cores where seeing varies";
        let on = [
            "local normalization failed: boom".to_string(),
            "local normalization off \u{2014} referenceFrames (2) < 3; using global normalization".to_string(),
            "local normalization off \u{2014} only 2 member(s) can build the LN reference (need >= 3); using global normalization".to_string(),
            "local normalization off \u{2014} the LN reference's background model is refused (no background sample); using global normalization".to_string(),
        ];
        let rejection_only = [
            format!("local rejection normalization could not be computed (boom); {tail}"),
            format!("local rejection normalization could not be computed \u{2014} referenceFrames (2) < 3; {tail}"),
            format!("local rejection normalization could not be computed \u{2014} only 2 member(s) can build the reference (need >= 3); {tail}"),
            format!("local rejection normalization could not be computed \u{2014} the reference frame's background model is refused (no background sample); {tail}"),
        ];
        for (k, what) in cases.iter().enumerate() {
            assert_eq!(fallback_warning(false, what), on[k], "on {k}");
            assert_eq!(fallback_warning(true, what), rejection_only[k], "rejection only {k}");
        }
    }
}
