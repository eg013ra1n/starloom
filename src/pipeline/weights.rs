//! The weights a stage stacks with: Athenaeum's reweigh, over the stored measurements and the stored transforms.
//!
//! Measure weighs every frame on its own measurement. Under the PSF family the weight that counts is the one read on
//! the stars every member has in common, which needs the registration maps: each member's `.athf` star fits are
//! matched through its own map, no pixel is read. [`current_weights`] is that step (`reweigh_groups` in Athenaeum's
//! run) for one group: it takes the stored measurements through [`finish`] under the caller's config, checks that
//! every frame the config includes has a transform, and reweighs. `register`, `normalize` and `stack` each call it,
//! so a weight mode changed between two commands re-weighs without measuring again (P2-R21); `weights.json` is the
//! report of what was used and never an input.
//!
//! A frame the config includes that `register` did not register (it was out of the selection then) is refused,
//! naming the frame and `starloom register` (P2-R24: nothing is compared, a missing input is an ordinary error).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use astroimage::header::FrameFacts;

use crate::geometry::PixelMap;
use crate::pipeline::formats::{
    MeasureFrame, MeasureReport, TransformFrame, TransformsReport, WeightFrame, WeightSettings, WeightsReport,
    REGISTRATION_FAILED,
};
use crate::pipeline::register::RegisterReport;
use crate::pipeline::{athf_path, finish, source_facts, Ctx, Manifest};
use crate::stacking::common_set;
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::fits_artifact::read_fits;
use crate::stacking::psf_signal::{self, StarFit};
use crate::stacking::run_helpers::map_scale;
use crate::stacking::weights::{compute_weights, defers_weight_gate, weight_gate, FrameWeight, WeightInput};

/// `reason` of a fallback names "under 20 common stars"; the count it names is the module's.
const _: () = assert!(common_set::COMMON_MIN_STARS == 20);

/// The frames of `measure` that are among `frames`, in `measure`'s order (the pipeline's frame order, P2-R10): the
/// report a command works on when it was given a subset. A frame with no row in `measure` is an error naming it and
/// `starloom measure`.
pub fn subset_measure(measure: &MeasureReport, frames: &[PathBuf]) -> Result<MeasureReport> {
    let mut wanted: HashSet<PathBuf> = HashSet::with_capacity(frames.len());
    for frame in frames {
        let abs = std::path::absolute(frame).map_err(|e| anyhow::anyhow!("cannot make {} absolute: {e}", frame.display()))?;
        if !measure.frames.iter().any(|f| f.file == abs) {
            bail!("{}: no measurement of this frame in measure.json; run `starloom measure` on it", abs.display());
        }
        wanted.insert(abs);
    }
    let mut out = measure.clone();
    out.frames.retain(|f| wanted.contains(&f.file));
    Ok(out)
}

/// Why one plane keeps its Measure-time weight inputs (ruling PD-6). Nothing here is per frame.
#[derive(Debug, Clone, PartialEq)]
enum Fallback {
    /// Fewer than `COMMON_MIN_STARS` stars are common to the members.
    TooFewStars(usize),
    /// This member has no readable `.athf` for the plane (or no such plane or map).
    MissingFits(usize),
    /// No member at or above the median star count has a usable FWHM, so there is no anchor.
    NoAnchor,
    /// The set is big enough, but the anchor's aperture sums over it add to zero or less (PD-30).
    AnchorFluxNotPositive(usize),
    /// The match table did not fit the candidates: a wiring bug of this module, never a property of the sky.
    MalformedTable,
}

impl Fallback {
    /// The text of the fallback warning.
    fn reason(&self) -> &'static str {
        match self {
            Fallback::TooFewStars(_) => "under 20 common stars",
            Fallback::MissingFits(_) => "member without a readable .athf",
            Fallback::NoAnchor => "no anchor",
            Fallback::AnchorFluxNotPositive(_) => "anchor flux not positive",
            Fallback::MalformedTable => "match table malformed",
        }
    }

    /// The set size when one was counted, else 0.
    fn common_stars(&self) -> usize {
        match self {
            Fallback::TooFewStars(n) | Fallback::AnchorFluxNotPositive(n) => *n,
            _ => 0,
        }
    }
}

/// One plane after the match: the set and every member's flux over it, or why there is none.
enum PlaneOutcome {
    Measured {
        positions: Vec<(f64, f64)>,
        per_member: Vec<common_set::CommonFlux>,
        /// The anchor's candidate stars the set was drawn from: the denominator of the anchor-map warning (PD-39).
        candidates: usize,
    },
    Fallback(Fallback),
}

/// A member's readable `.athf` for `plane`; `None` when it is missing or unreadable (an unreadable one is logged).
fn read_member_fits(work: &Path, file: &Path, plane: usize) -> Option<Vec<StarFit>> {
    let path = athf_path(work, file, plane);
    match read_fits(&path) {
        Ok(fits) => Some(fits),
        Err(e) => {
            tracing::warn!(path = %path.display(), plane, error = %e, "weights: fits artifact unreadable");
            None
        }
    }
}

/// The common set of one plane and every member's flux over it: the anchor's candidates, then each member's match
/// through its own stored map, one member's fits resident at a time (`plane_outcome` of Athenaeum's reweigh, reading
/// the `.athf` files of the working folder).
fn plane_outcome(
    ctx: &Ctx<'_>,
    frames: &[MeasureFrame],
    members: &[usize],
    maps: &[&PixelMap],
    plane: usize,
    planes: usize,
    anchor: usize,
) -> Result<PlaneOutcome> {
    let plane_of = |k: usize| frames[members[k]].measurement.as_ref()?.channels.get(plane);
    let missing = |k: usize| PlaneOutcome::Fallback(Fallback::MissingFits(members[k]));

    let cands = {
        let Some(ch) = plane_of(anchor) else {
            return Ok(missing(anchor));
        };
        let Some(fits) = read_member_fits(ctx.work_dir, &frames[members[anchor]].file, plane) else {
            return Ok(missing(anchor));
        };
        common_set::candidates(&common_set::MemberPlane {
            fits: &fits,
            map: maps[anchor],
            fwhm_px: ch.fwhm_px,
            noise: ch.noise,
            c_stand_in: ch.aperture_correction_group.unwrap_or(1.0),
            planes,
        })
    };
    let mut usable = Vec::with_capacity(members.len());
    for k in 0..members.len() {
        // Between members: a group's planes x members `.athf` reads are the longest stretch of the reweigh.
        ctx.check_cancelled()?;
        let Some(ch) = plane_of(k) else {
            return Ok(missing(k));
        };
        let Some(fits) = read_member_fits(ctx.work_dir, &frames[members[k]].file, plane) else {
            return Ok(missing(k));
        };
        usable.push(common_set::match_member(
            &cands,
            &common_set::MemberPlane {
                fits: &fits,
                map: maps[k],
                fwhm_px: ch.fwhm_px,
                noise: ch.noise,
                c_stand_in: ch.aperture_correction_group.unwrap_or(1.0),
                planes,
            },
        ));
    }
    Ok(match common_set::common_flux(&cands, &usable, anchor, planes) {
        common_set::CommonOutcome::Measured { positions, per_member } => {
            PlaneOutcome::Measured { positions, per_member, candidates: cands.len() }
        }
        common_set::CommonOutcome::TooFewStars { stars } => PlaneOutcome::Fallback(Fallback::TooFewStars(stars)),
        common_set::CommonOutcome::NoAnchorFlux { stars } => PlaneOutcome::Fallback(Fallback::AnchorFluxNotPositive(stars)),
        common_set::CommonOutcome::MalformedTable => {
            tracing::error!(plane, candidates = cands.len(), members = usable.len(), "weights: common-set match table malformed");
            PlaneOutcome::Fallback(Fallback::MalformedTable)
        }
    })
}

/// The weights `cfg` gives the frames of `measure` once they sit on `register`'s transforms: [`finish`] over the stored
/// measurements (the fold, the weights, the selection), then Athenaeum's reweigh. `measure` holds the stored
/// measurements as `measure.json` has them; its frames are the group, so a caller with a subset passes
/// [`subset_measure`]. The raw lights' facts come from the manifest of the working folder (the exposure weight reads
/// them), and the `.athf` star fits from `<work>/measure/fits/`.
///
/// Under the PSF family every member's weight is recomputed on the stars the members have in common and the weight
/// floor that Measure deferred is applied to it. In every mode the common set is built and returned in the report, for
/// the SNR gain. The reweigh never re-admits an excluded frame. A frame that registration failed on is excluded under
/// `selection.exclude_on_registration_failure` and is an error otherwise.
///
/// A frame the config includes that has no transform is an error that names it and `starloom register`: `register`
/// left it out because the selection it ran with excluded it (P2-R21, P2-R24). Nothing else is compared.
pub fn current_weights(
    ctx: &Ctx<'_>,
    measure: &MeasureReport,
    register: &RegisterReport,
    cfg: &AlgorithmConfig,
) -> Result<WeightsReport> {
    let manifest = Manifest::load(ctx.work_dir)?;
    let files: Vec<PathBuf> = measure.frames.iter().map(|f| f.file.clone()).collect();
    let sources = source_facts(&manifest, &files)?;
    let mut finished = finish(measure, cfg, &sources);
    let named = register.transforms.reference_pinned.then_some(register.transforms.reference.as_path());
    keep_named_reference(ctx, &mut finished, named)?;
    weigh(ctx, finished, &register.transforms, &sources, cfg)
}

/// A manual pin stays in the stack: when `named` (the reference the user named, when they named one) is a frame of
/// `finished` that the selection left out, it is included again, with no reason and a warning, as stage 4 of
/// Athenaeum's run forces its manual reference in (`run/reference.rs`: `included = true`, `reason = None`, "kept despite
/// selection"). `register` and [`current_weights`] both call it right after [`finish`], so the common set, the members'
/// weights and the stack see the same frames. A frame with no measurement cannot be weighed, so it cannot be kept: that
/// is an error naming it, as Athenaeum refuses a manual reference whose calibrated file is missing.
pub(crate) fn keep_named_reference(ctx: &Ctx<'_>, finished: &mut MeasureReport, named: Option<&Path>) -> Result<()> {
    let Some(named) = named else { return Ok(()) };
    let Some(frame) = finished.frames.iter_mut().find(|f| f.file == named) else { return Ok(()) };
    if frame.included {
        return Ok(());
    }
    let reason = frame.reason.clone().unwrap_or_else(|| "excluded".to_string());
    if frame.measurement.is_none() {
        bail!("the named reference frame {} is excluded: {reason}", frame.file.display());
    }
    frame.included = true;
    frame.reason = None;
    tracing::warn!(path = %frame.file.display(), note = %reason, "named reference kept despite selection");
    ctx.progress.warn(&format!("named reference frame {} kept despite selection: {reason}", frame.file.display()));
    Ok(())
}

/// [`current_weights`] after [`finish`]: `finished` is `finish`'s result under `cfg`, `transforms` the registration.
pub(crate) fn weigh(
    ctx: &Ctx<'_>,
    mut finished: MeasureReport,
    transforms: &TransformsReport,
    sources: &BTreeMap<PathBuf, FrameFacts>,
    cfg: &AlgorithmConfig,
) -> Result<WeightsReport> {
    // Every frame the config includes needs a map; a frame that failed to register is out.
    let rows: BTreeMap<&Path, &TransformFrame> = transforms.frames.iter().map(|t| (t.file.as_path(), t)).collect();
    let mut lacking: Vec<String> = Vec::new();
    for f in finished.frames.iter_mut().filter(|f| f.included) {
        match rows.get(f.file.as_path()) {
            Some(t) if t.ok && t.map.is_some() => {}
            Some(t) if t.error.as_deref().is_some_and(|e| e.starts_with(REGISTRATION_FAILED)) => {
                let reason = t.error.clone().unwrap_or_default();
                if !cfg.selection.exclude_on_registration_failure {
                    bail!("{}: {reason}", f.file.display());
                }
                tracing::warn!(path = %f.file.display(), reason = %reason, "frame excluded");
                f.included = false;
                f.reason = Some(reason);
            }
            Some(t) => lacking.push(format!("{} ({})", f.file.display(), t.error.as_deref().unwrap_or("no transform"))),
            None => lacking.push(format!("{} (no row in transforms.json)", f.file.display())),
        }
    }
    if !lacking.is_empty() {
        let shown = lacking.iter().take(3).cloned().collect::<Vec<_>>().join("; ");
        let more = if lacking.len() > 3 { format!(" and {} more", lacking.len() - 3) } else { String::new() };
        bail!(
            "{} frame(s) are included under this config but have no transform: {shown}{more}; run `starloom register` \
             with this config",
            lacking.len()
        );
    }

    let settings = WeightSettings {
        weight_mode: cfg.measurement.weight_mode,
        formula: cfg.measurement.formula.clone(),
        keyword: cfg.measurement.keyword.clone(),
        selection: cfg.selection,
    };
    let mut report = WeightsReport::new(settings);

    // The members: included, measured, weighed, registered.
    let members: Vec<usize> = finished
        .frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.included && f.measurement.is_some() && f.weight.is_some())
        .map(|(i, _)| i)
        .collect();
    if !members.is_empty() {
        reweigh(ctx, &mut finished, &members, &rows, transforms, sources, cfg, &mut report)?;
    }
    report.frames = finished
        .frames
        .into_iter()
        .map(|f| WeightFrame { file: f.file, measurement: f.measurement, weight: f.weight, included: f.included, reason: f.reason })
        .collect();
    Ok(report)
}

/// `reweigh_group` for the one group: the anchor, each plane's common set, the PSF family's weights over it and the
/// deferred floor. The frames of `finished` are updated in place and the sets land in `report`.
#[allow(clippy::too_many_arguments)]
fn reweigh(
    ctx: &Ctx<'_>,
    finished: &mut MeasureReport,
    members: &[usize],
    rows: &BTreeMap<&Path, &TransformFrame>,
    transforms: &TransformsReport,
    sources: &BTreeMap<PathBuf, FrameFacts>,
    cfg: &AlgorithmConfig,
    report: &mut WeightsReport,
) -> Result<()> {
    // PD-19 b: the set is built in every mode; the channel writes and the weights (phases 2 and 3 below) are the PSF
    // family's.
    let psf_family = defers_weight_gate(cfg.measurement.weight_mode);

    // Phase 1, on shared borrows: the anchor and each plane's outcome.
    let maps: Vec<&PixelMap> = members
        .iter()
        .map(|&i| rows.get(finished.frames[i].file.as_path()).and_then(|t| t.map.as_ref()).expect("a member has a transform"))
        .collect();
    let (anchor, outcomes) = {
        let frames = finished.frames.as_slice();
        let meas = |k: usize| frames[members[k]].measurement.as_ref().expect("a member is measured");
        let n = members.len();
        let nch = (0..n).map(|k| meas(k).channels.len()).max().unwrap_or(0);
        let mean_fwhm: Vec<f64> = (0..n).map(|k| meas(k).mean_fwhm_px()).collect();
        let unsaturated: Vec<usize> = (0..n).map(|k| meas(k).channels.iter().map(|c| c.stars_unsaturated).sum()).collect();
        // The frame's place in the pipeline's order stands for the catalog id the tie breaks on.
        let frame_ids: Vec<i64> = members.iter().map(|&i| i as i64).collect();
        let anchor = common_set::pick_anchor(&mean_fwhm, &unsaturated, &frame_ids);

        // The match radius is built from frame-pixel FWHMs and applied in group pixels: a map whose scale is not 1 is
        // not corrected for, only logged.
        for k in 0..n {
            let scale = map_scale(maps[k], meas(k).width, meas(k).height);
            if scale.is_finite() && (scale - 1.0).abs() > 0.05 {
                tracing::debug!(path = %frames[members[k]].file.display(), map_scale = scale, "weights: registration map scale differs from 1");
            }
        }

        let mut outcomes = Vec::with_capacity(nch);
        for plane in 0..nch {
            outcomes.push(match anchor {
                None => PlaneOutcome::Fallback(Fallback::NoAnchor),
                Some(a) => plane_outcome(ctx, frames, members, &maps, plane, nch, a)?,
            });
        }
        (anchor, outcomes)
    };
    let anchor_file = anchor.map(|a| finished.frames[members[a]].file.clone());

    // Phase 2: record each plane's outcome.
    let any_measured = outcomes.iter().any(|o| matches!(o, PlaneOutcome::Measured { .. }));
    let nch = outcomes.len();
    let mut paired_by_plane: Vec<Option<Vec<usize>>> = vec![None; nch];
    let mut sets: Vec<Option<Vec<(f64, f64)>>> = Vec::with_capacity(nch);
    for (plane, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            PlaneOutcome::Measured { positions, per_member, candidates } => {
                let stars = positions.len();
                if psf_family {
                    for (&i, cf) in members.iter().zip(&per_member) {
                        let Some(ch) = finished.frames[i].measurement.as_mut().and_then(|m| m.channels.get_mut(plane)) else {
                            continue;
                        };
                        ch.common_tflux = Some(cf.tflux);
                        ch.common_tmean_flux = Some(cf.tmean_flux);
                        ch.common_stars = Some(stars);
                        // All native units: the scale-invariant estimators read the channel's own noise and M*.
                        ch.psf_signal_weight = psf_signal::psf_signal_weight(cf.tflux, cf.tmean_flux, ch.noise, ch.m_star);
                        ch.psf_snr = psf_signal::psf_snr(cf.tflux, ch.noise);
                    }
                }
                paired_by_plane[plane] = Some(per_member.iter().map(|cf| cf.paired).collect());
                tracing::info!(
                    plane,
                    anchor = anchor_file.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
                    common_stars = stars,
                    candidates,
                    count = members.len(),
                    "weights: frame-common star set"
                );
                // PD-39: candidates enter group coordinates through the anchor's one registration map, so an error of
                // that map larger than the match radius takes a star out of every member at once.
                if common_set::set_is_small_against_candidates(stars, candidates) {
                    tracing::warn!(
                        plane,
                        common_stars = stars,
                        candidates,
                        "weights: common set far under the anchor's candidates; the anchor's registration map may be wrong"
                    );
                }
                sets.push(Some(positions));
            }
            PlaneOutcome::Fallback(f) => {
                tracing::warn!(
                    plane,
                    common_stars = f.common_stars(),
                    fallback = "measure_time",
                    reason = f.reason(),
                    "weights: no frame-common star set"
                );
                // Outside the PSF family the plane has no weights to keep (they are Measure's either way): what it
                // loses is the SNR gain.
                let consequence = if psf_family { "its weights keep the Measure-time values" } else { "its SNR gain is not measured" };
                let named = match &f {
                    Fallback::MissingFits(i) => format!(" ({}; {})", f.reason(), finished.frames[*i].file.display()),
                    _ => format!(" ({})", f.reason()),
                };
                ctx.progress.warn(&format!("no frame-common star set on plane {plane}{named}; {consequence}"));
                sets.push(None);
            }
        }
    }
    // The record stays whenever this phase ran: a group whose every plane fell back reaches integrate as
    // `Some([None, ...])`, a fallback that warns, so an empty list means "never reweighed" and nothing else.
    report.common_stars = sets;
    if !psf_family {
        // PD-19 b: Measure's weights and its floor stand as they are (PD-5).
        return Ok(());
    }

    // Phase 3: the weights over the members, and the floor that Measure deferred. A group whose every plane fell back
    // keeps the weights Measure gave it, on its one scale (PD-6); the floor runs on them. Otherwise the members' weights
    // are recomputed, planes at their new or their Measure-time inputs, and normalized over the members.
    let recomputed: Option<Vec<FrameWeight>> = any_measured.then(|| {
        let inputs: Vec<WeightInput> = members
            .iter()
            .map(|&i| WeightInput {
                measurement: finished.frames[i].measurement.as_ref().expect("a member is measured"),
                exposure_s: sources.get(&finished.frames[i].file).and_then(|s| s.exptime_s),
                keyword_value: None,
            })
            .collect();
        let excluded = vec![false; inputs.len()];
        compute_weights(&inputs, cfg.measurement.weight_mode, &cfg.measurement.formula, &excluded)
    });
    let weights: Vec<FrameWeight> = match recomputed {
        Some(w) => w,
        None => members.iter().map(|&i| finished.frames[i].weight.clone().expect("a member is weighed")).collect(),
    };
    // PD-17: only a named reference stays under the floor (Athenaeum forces a manual pin in despite the selection); an
    // auto reference is the geometry's origin and nothing more, and is excluded like any frame.
    let pinned: Option<&Path> = transforms.reference_pinned.then_some(transforms.reference.as_path());
    for (k, w) in weights.into_iter().enumerate() {
        let i = members[k];
        for (plane, paired) in paired_by_plane.iter().enumerate() {
            if let (Some(paired), Some(weight)) = (paired, w.normalized.get(plane)) {
                tracing::debug!(path = %finished.frames[i].file.display(), plane, paired_stars = paired[k], weight = *weight, "frame reweighed");
            }
        }
        let gate = weight_gate(&w, &cfg.selection);
        let frame = &mut finished.frames[i];
        frame.weight = Some(w);
        if let Some(reason) = gate {
            if pinned == Some(frame.file.as_path()) {
                tracing::warn!(path = %frame.file.display(), reason = %reason, "weights: named reference kept despite the weight floor");
                ctx.progress.warn(&format!("the reference frame {} is kept despite its weight: {reason}", frame.file.display()));
            } else {
                tracing::warn!(path = %frame.file.display(), reason = %reason, "frame excluded");
                frame.included = false;
                frame.reason = Some(reason);
            }
        }
    }
    // One scale per weights table (PD-6): a frame the reweigh did not weigh carries no weight.
    if any_measured {
        let member_set: HashSet<usize> = members.iter().copied().collect();
        for (i, f) in finished.frames.iter_mut().enumerate() {
            if !member_set.contains(&i) {
                f.weight = None;
            }
        }
    }
    Ok(())
}
