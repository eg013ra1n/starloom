//! The `register` stage: every frame onto one reference, then the reweigh.
//!
//! This mirrors Athenaeum's Register stage (`register_group_pass`, then `reweigh_groups`) for one group, without the
//! catalog, the two-pass reference pick, the plate-solve seed and the project grid (P2-R3). The reference is a file
//! the user names (a frame of the stack, or any calibrated FITS) or the best-weighted frame ([`ReferenceChoice`]).
//! Each included frame's stars are detected (a single-plane frame reuses Measure's `.athf` fits instead of detecting
//! again, as Athenaeum's mono path does) and aligned onto the reference's; the reference itself, when it is one of the
//! frames, gets the identity.
//!
//! Products, under `<work>/register/`:
//! - `transforms.json`: one row per frame, with its map ([`TransformsReport`]);
//! - `weights.json`: the weights the stage reweighed to ([`WeightsReport`], see [`current_weights`](crate::pipeline::current_weights));
//! - with `write_registered`, `registered/<stem>.fits`: every registered frame resampled into the reference's grid
//!   (NaN outside its coverage) by the warp Athenaeum stacks with.
//!
//! A named reference that the selection left out is kept in the stack with a warning (a manual pin, as in Athenaeum's
//! stage 4) and stays under the weight floor; an auto pick is neither. A registered file carries starloom's own cards
//! (`SLM_VER`, `SLM_CFG`, `SLM_PLV`, `SLM_REF`) and none of its source's `ATH_*` or checksum cards (P2-R6, SL-8).
//!
//! A frame that fails to register is excluded when `selection.exclude_on_registration_failure` is set (its row says
//! why) and refuses the run otherwise, as Athenaeum's run does. A frame that Measure's selection left out is not
//! registered; its row says so, and a later command that includes it under another config asks for this command again
//! (P2-R24). Everything that decodes a pixel or detects a star runs on the context's pool (SL-9).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::{header_key_map, header_text, read_frame_facts, FrameFacts};

use crate::fits_writer::{write_fits_f32_streaming_with, Card, CardValue, Durability, FitsWriteError};
use crate::geometry::header_pixel_scale_arcsec;
use crate::integration::plane_reader::PlaneReader;
use crate::pipeline::calibrate::{card_from_kv, encode_ident, file_name_of};
use crate::pipeline::formats::{
    commit_stage, part_path, Entry, MeasureFrame, MeasureReport, TransformFrame, TransformsReport, WeightsReport, NOT_REGISTERED,
    REGISTRATION_FAILED, TRANSFORMS_FILE, WEIGHTS_FILE,
};
use crate::pipeline::measure::fan_out;
use crate::pipeline::versions::recorded_versions;
use crate::pipeline::weights::{keep_named_reference, subset_measure, weigh};
use crate::pipeline::measure::athf_in;
use crate::pipeline::{common_cards, config_hash, frame_stem, finish, source_facts, warn_user, Ctx, Manifest, Stage, SLM_CFG, SLM_PLV, SLM_VER};
use crate::resample::{warp_rows, Plane};
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::fits_artifact::read_fits;
use crate::stacking::psf_signal::StarFit;
use crate::stacking::register::align::{model_name, Alignment, SeedPolicy, MIN_INLIERS};
use crate::stacking::register::detect::stars_from_fits;
use crate::stacking::register::frame::{
    detect_frame_stars, identity_registration, reference_stars, register_detected, FrameRegistration, ReferenceStars,
};
use crate::stacking::register::wcs_seed::ratio_wants_seed;
use crate::stacking::register::{low_inlier_frames, scale_gate_for, scale_ratio_for};
use crate::stacking::weights::{best_by_weight, FrameWeight};

/// The stage's folder under the working folder.
pub const REGISTER_DIR: &str = "register";
/// The folder of the registered frames, under [`REGISTER_DIR`].
pub const REGISTERED_DIR: &str = "registered";
/// The card naming the reference a registered frame was resampled onto (P2-R6).
pub const SLM_REF: &str = "SLM_REF";

/// Planes of one reference-geometry channel resident while a worker registers a mono frame (Athenaeum's
/// `REGISTER_PLANES_RESIDENT`, measured there).
const REGISTER_PLANES_RESIDENT: u64 = 4;
/// The same for a three-plane frame (`REGISTER_PLANES_RESIDENT_OSC`).
const REGISTER_PLANES_RESIDENT_OSC: u64 = 6;

/// What a frame is registered onto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceChoice {
    /// A calibrated plain-FITS file the user names: one of the frames, or any other file (which then supplies the
    /// stars and the grid only, and no frame gets the identity).
    File(PathBuf),
    /// The best-weighted frame among those the selection includes and the Measure-time weight floor leaves in
    /// (`best_by_weight`, ties by star count), picked from the weights `register` computes under its config.
    Auto,
}

/// What `register` leaves, as the stages after it read it.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterReport {
    /// Where every frame went (`transforms.json`).
    pub transforms: TransformsReport,
    /// The weights the stage reweighed to (`weights.json`).
    pub weights: WeightsReport,
    /// The registered frames written, in frame order; empty unless `write_registered` was set.
    pub registered: Vec<PathBuf>,
}

impl RegisterReport {
    /// Reads the products of a `register` run from the working folder `work_dir`: `register/transforms.json`,
    /// `register/weights.json`, and the registered frames (the `.fits` products of the Register manifest entry, in the
    /// order the entry lists them, which is frame order). A folder with no Register entry is an error.
    pub fn read(work_dir: &Path) -> Result<RegisterReport> {
        let dir = register_dir(work_dir);
        let transforms = TransformsReport::read(&dir.join(TRANSFORMS_FILE))?;
        let weights = WeightsReport::read(&dir.join(WEIGHTS_FILE))?;
        let manifest = Manifest::load(work_dir)?;
        let entry = manifest
            .entry(Stage::Register)
            .ok_or_else(|| anyhow!("{} has no register entry in its manifest; run `starloom register`", work_dir.display()))?;
        let registered: Vec<PathBuf> = entry.products.iter().filter(|p| p.extension().is_some_and(|e| e == "fits")).cloned().collect();
        Ok(RegisterReport { transforms, weights, registered })
    }
}

/// The stage's folder: `<work>/register`.
pub fn register_dir(work_dir: &Path) -> PathBuf {
    work_dir.join(REGISTER_DIR)
}

/// Where `register --write-registered` leaves the registered copy of `frame`: `<work>/register/registered/<stem>.fits`.
pub fn registered_path(work_dir: &Path, frame: &Path) -> PathBuf {
    register_dir(work_dir).join(REGISTERED_DIR).join(format!("{}.fits", frame_stem(frame)))
}

// ---------------------------------------------------------------------------------------------------------------
// The reference
// ---------------------------------------------------------------------------------------------------------------

/// A copy of Athenaeum's `MeasuredFrame::in_reference_pick`: included, and not under the Measure-time floor that the PSF
/// family defers.
fn in_reference_pick(f: &MeasureFrame) -> bool {
    f.included && !f.measure_floored
}

/// The best-weighted frame a reference pick may see (`best_included_idx`): a frame whose weight never computed ranks
/// last rather than dropping out.
fn pick_auto(finished: &MeasureReport) -> Result<usize> {
    let zero = FrameWeight { channels: Vec::new(), normalized: Vec::new(), mean: 0.0, normalized_mean: 0.0, missing: None };
    let weights: Vec<FrameWeight> = finished.frames.iter().map(|f| f.weight.clone().unwrap_or_else(|| zero.clone())).collect();
    let included: Vec<bool> = finished.frames.iter().map(in_reference_pick).collect();
    let star_counts: Vec<usize> = finished.frames.iter().map(|f| f.measurement.as_ref().map_or(0, |m| m.min_stars())).collect();
    best_by_weight(&weights, &included, &star_counts)
        .ok_or_else(|| anyhow!("no frame can be the reference: none is included and clear of the weight floor; name one with --reference FILE"))
}

/// Whether two paths name one file: equal as given, or as the file system resolves them.
fn same_file(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}

/// A single-plane frame's own Measure fits, for detection to reuse (`resolve_register_fits`): `None` when the file is
/// missing or unreadable, or when the cuts leave too few stars to align on (detection then runs, as in Athenaeum).
fn frame_fits(fits_root: &Path, file: &Path, cfg: &AlgorithmConfig) -> Option<Vec<StarFit>> {
    let path = athf_in(fits_root, frame_stem(file), 0);
    let fits = match read_fits(&path) {
        Ok(fits) => fits,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "register: no readable fits artifact; detecting");
            return None;
        }
    };
    // The same cut and truncation detection's own fits branch will run, judged here rather than discovered after
    // RANSAC.
    let usable = stars_from_fits(&fits, &cfg.registration.detection, cfg.registration.max_stars).len();
    if usable < MIN_INLIERS {
        tracing::warn!(path = %file.display(), count = usable, "register: fits artifact has too few usable stars after cuts; detecting");
        return None;
    }
    Some(fits)
}

// ---------------------------------------------------------------------------------------------------------------
// The registered frames
// ---------------------------------------------------------------------------------------------------------------

/// The keywords a FITS or XISF header text lists, in order, once each.
fn header_keywords(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let key = line.split('=').next().unwrap_or("").trim();
        if key.is_empty() || key.contains(' ') || !seen.insert(key.to_string()) {
            continue;
        }
        out.push(key.to_string());
    }
    out
}

/// The header's cards in header order, typed as `card_from_kv` types them. Long and unwritable cards are dropped.
pub(crate) fn header_cards(path: &Path) -> Result<Vec<Card>> {
    let (format, text) = header_text(path).with_context(|| format!("cannot read the header of {}", path.display()))?;
    let map = header_key_map(format, &text).into_legacy_map();
    Ok(header_keywords(&text).iter().filter_map(|k| map.get(k).and_then(|v| card_from_kv(k, v))).collect())
}

/// Structure of the file, which the writer makes itself, and keywords that describe the pixels of the source and not
/// of its resampled copy.
fn is_structural(key: &str) -> bool {
    matches!(
        key,
        "SIMPLE" | "BITPIX" | "EXTEND" | "BZERO" | "BSCALE" | "BLANK" | "PCOUNT" | "GCOUNT" | "XTENSION" | "END" | "COMMENT" | "HISTORY"
            | "CONTINUE" | "DATAMIN" | "DATAMAX" | "ZIMAGE"
    ) || key.starts_with("NAXIS")
}

/// A world-coordinate keyword: the axes' `CTYPEn`, `CRVALn`, `CRPIXn`, `CDELTn`, `CROTAn`, `CUNITn`, the `CDi_j`,
/// `PCi_j` and `PVi_m` matrices, the SIP terms and the pole and system cards.
pub(crate) fn is_wcs(key: &str) -> bool {
    const AXIS: [&str; 8] = ["CTYPE", "CRVAL", "CRPIX", "CDELT", "CROTA", "CUNIT", "CRDER", "CSYER"];
    const MATRIX: [&str; 3] = ["CD", "PC", "PV"];
    if matches!(key, "WCSAXES" | "LONPOLE" | "LATPOLE" | "RADESYS" | "EQUINOX" | "EPOCH" | "WCSNAME") {
        return true;
    }
    if AXIS.iter().any(|p| key.strip_prefix(p).is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))) {
        return true;
    }
    if MATRIX.iter().any(|p| key.strip_prefix(p).is_some_and(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_digit()) && rest.contains('_'))) {
        return true;
    }
    ["A_", "B_", "AP_", "BP_"].iter().any(|p| key.starts_with(p))
}

/// The cards a registered frame does not keep from its source: the mosaic's CFA description (the frame is resampled, so
/// it has none); every `ATH_*` card and `CALSTAT` (Athenaeum's, and a registered file is starloom's own, P2-R6, SL-8);
/// the checksums, which describe the source's bytes and would fail on the copy; and the source's own stamps (`SLM_VER`,
/// `SLM_CFG`, `SLM_PLV`, which name the stage that wrote it, and `SLM_REF` of an earlier registration).
fn is_dropped_from_source(key: &str) -> bool {
    matches!(key, "BAYERPAT" | "XBAYROFF" | "YBAYROFF" | "CALSTAT" | "CHECKSUM" | "DATASUM")
        || key.starts_with("ATH_")
        || matches!(key, SLM_VER | SLM_CFG | SLM_PLV | SLM_REF)
}

/// What a registered copy of one frame is made from besides the frame: the folder it goes to, the reference's grid
/// cards and name, and the hash of the config `register` ran with.
struct RegisteredOut<'a> {
    dir: PathBuf,
    wcs: &'a [Card],
    roworder: Option<&'a Card>,
    reference_name: &'a str,
    cfg_hash: &'a str,
}

/// The cards of one registered frame: the source's non-structural cards minus the world-coordinate ones and the
/// foreign ones ([`is_dropped_from_source`]), the reference's world-coordinate cards when it has them (the grid is its
/// grid), then the file's own: `SLM_VER`, `SLM_CFG`, `SLM_PLV` ([`common_cards`]) and `SLM_REF` (P2-R6). `ROWORDER` is
/// the reference's when it has one, because the registered array is the reference's grid.
fn registered_cards(source: &Path, to: &RegisteredOut<'_>) -> Result<Vec<Card>> {
    let mut cards: Vec<Card> = header_cards(source)?
        .into_iter()
        .filter(|c| !is_structural(&c.keyword) && !is_wcs(&c.keyword) && !is_dropped_from_source(&c.keyword))
        .collect();
    if let Some(order) = to.roworder {
        cards.retain(|c| c.keyword != "ROWORDER");
        cards.push(order.clone());
    }
    cards.extend(to.wcs.iter().cloned());
    cards.extend(common_cards(to.cfg_hash)?);
    cards.push(Card::new(SLM_REF, CardValue::Str(encode_ident(to.reference_name)))?.with_comment("registration reference"));
    Ok(cards)
}

/// Resamples every plane of `source` through `map` (frame to reference) into the reference's `ref_w` x `ref_h` grid, NaN
/// outside the frame's coverage, and writes the 32-bit float FITS plane by plane (`write_registered_frame` of
/// Athenaeum's run). Runs on the current rayon pool.
#[allow(clippy::too_many_arguments)]
fn write_registered(
    source: &Path,
    map: &crate::geometry::PixelMap,
    ref_w: usize,
    ref_h: usize,
    cfg: &AlgorithmConfig,
    cards: &[Card],
    out: &Path,
) -> Result<()> {
    // This frame's displacement grid goes back when the function returns, however it returns.
    let _release = map.release_grids_on_drop();
    let reader = PlaneReader::open(source).map_err(|e| anyhow!("{e}"))?;
    let (w, h, channels) = (reader.width(), reader.height(), reader.channels());
    let plane_len = ref_w * ref_h;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let interp = cfg.registration.interpolation;
    let clamping = cfg.registration.clamping_threshold;
    write_fits_f32_streaming_with(out, ref_w, ref_h, channels, cards, Durability::Durable, |plane| {
        let data = reader.read_plane(plane).map_err(|e| FitsWriteError::Io(std::io::Error::other(e.to_string())))?;
        let src = Plane::full(&data, w, h);
        let mut dst = vec![f32::NAN; plane_len];
        warp_rows(&src, map, ref_w, 0, ref_h, interp, clamping, &mut dst);
        Ok(dst)
    })
    .with_context(|| format!("cannot write {}", out.display()))
}

// ---------------------------------------------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------------------------------------------

/// What a worker leaves of one frame: the registration (or why it failed) and whether its registered copy failed.
struct Outcome {
    registration: std::result::Result<FrameRegistration, String>,
    write_error: Option<String>,
}

/// The row of a frame that was aligned.
fn aligned_row(file: &Path, a: &Alignment) -> TransformFrame {
    TransformFrame {
        file: file.to_path_buf(),
        ok: true,
        error: None,
        model: model_name(a.model, a.distortion, a.seed),
        map: Some(a.map.clone()),
        inliers: a.inliers,
        inlier_ratio: a.inlier_ratio,
        rms_px: a.rms_px,
        peak_px: a.peak_px,
        scale: a.scale,
        rotation_deg: a.rotation_deg,
        flipped: a.flipped,
        overlap: a.overlap,
        coverage: a.coverage,
        warnings: a.warnings.clone(),
    }
}

/// The row of a frame with no map: `error` says why.
fn unmapped_row(file: &Path, error: String) -> TransformFrame {
    TransformFrame {
        file: file.to_path_buf(),
        ok: false,
        error: Some(error),
        model: String::new(),
        map: None,
        inliers: 0,
        inlier_ratio: 0.0,
        rms_px: 0.0,
        peak_px: (0.0, 0.0),
        scale: 0.0,
        rotation_deg: 0.0,
        flipped: false,
        overlap: 0.0,
        coverage: 0.0,
        warnings: Vec::new(),
    }
}

/// Registers `frames` (calibrated plain-FITS frames `measure` has rows for) onto `reference`, reweighs, and writes
/// `<work>/register/` ([module docs](self)).
///
/// `measure` holds the stored measurements as `measure.json` has them; the weights, the selection and the Measure-time
/// floor are computed again from it under `cfg` ([`finish`], P2-R21), and only the frames that selection includes are
/// registered (plus the reference when it is one of the frames). A frame with no row in `measure` refuses the run,
/// naming it and `starloom measure` (P2-R24). `write_registered` also writes every registered frame's resampled copy.
///
/// The products are built in `<work>/register.part/` and renamed into place together with the manifest entry
/// ([`commit_stage`]), so a failure or a cancel leaves no entry and no half-written folder. A group with fewer than
/// three included frames is warned about, as Athenaeum's run skips such a group.
pub fn register(
    ctx: &Ctx<'_>,
    frames: &[PathBuf],
    measure: &MeasureReport,
    reference: &ReferenceChoice,
    cfg: &AlgorithmConfig,
    write_registered: bool,
) -> Result<RegisterReport> {
    if frames.is_empty() {
        refuse!("no frames to register");
    }
    let work = std::path::absolute(ctx.work_dir).with_context(|| format!("cannot make {} absolute", ctx.work_dir.display()))?;
    let sub = subset_measure(measure, frames)?;
    let reference = match reference {
        ReferenceChoice::File(path) => {
            ReferenceChoice::File(std::path::absolute(path).with_context(|| format!("cannot make {} absolute", path.display()))?)
        }
        ReferenceChoice::Auto => ReferenceChoice::Auto,
    };
    ctx.check_cancelled()?;
    // Everything below decodes or detects on the caller's pool, never rayon's global one (SL-9).
    ctx.pool.install(|| run(ctx, &work, &sub, &reference, cfg, write_registered))
}

fn run(
    ctx: &Ctx<'_>,
    work: &Path,
    sub: &MeasureReport,
    reference: &ReferenceChoice,
    cfg: &AlgorithmConfig,
    write_registered_files: bool,
) -> Result<RegisterReport> {
    let manifest = Manifest::load(ctx.work_dir)?;
    let files: Vec<PathBuf> = sub.frames.iter().map(|f| f.file.clone()).collect();
    let sources = source_facts(&manifest, &files)?;
    let mut finished = finish(sub, cfg, &sources);

    // The reference: its file, whether the user named it, and which frame it is, if any.
    let (ref_file, pinned, ref_idx) = match reference {
        ReferenceChoice::File(path) => {
            let idx = files.iter().position(|f| same_file(f, path));
            (idx.map_or_else(|| path.clone(), |i| files[i].clone()), true, idx)
        }
        ReferenceChoice::Auto => {
            let idx = pick_auto(&finished)?;
            (files[idx].clone(), false, Some(idx))
        }
    };
    // A named reference that the selection left out stays in the stack, as a manual pin does in Athenaeum.
    keep_named_reference(ctx, &mut finished, ref_idx.filter(|_| pinned).map(|i| files[i].as_path()))?;
    // Measure's fits seed a frame's stars only when every frame of the group is one plane (Athenaeum reuses them when
    // the group's stage-one planes are 1); a reference outside the stack gets fresh detection and supplies stars and
    // grid only (P2-R3).
    let reuse_fits = finished.frames.iter().filter_map(|f| f.measurement.as_ref()).all(|m| m.channels.len() == 1);
    let ref_planes = ref_idx.and_then(|i| finished.frames[i].measurement.as_ref()).map(|m| m.channels.len());
    let fits_root = finished.fits_root(ctx.work_dir);
    let ref_fits = if reuse_fits && ref_planes == Some(1) { frame_fits(&fits_root, &ref_file, cfg) } else { None };
    let ref_stars: ReferenceStars = reference_stars(&ref_file, &cfg.registration, Some(&ctx.pool), ref_fits.as_deref())
        .map_err(|e| anyhow!("reference star detection failed for {}: {e}", ref_file.display()))?;
    let (ref_w, ref_h) = (ref_stars.width, ref_stars.height);
    let ref_facts: Option<FrameFacts> = match ref_idx {
        Some(i) => sources.get(&files[i]).cloned(),
        None => match read_frame_facts(&ref_file) {
            Ok(facts) => Some(facts),
            Err(e) => {
                tracing::warn!(path = %ref_file.display(), error = %format!("{e:#}"), "register: the reference's header could not be read; no pixel scale");
                None
            }
        },
    };
    let ref_scale = ref_facts.as_ref().and_then(|s| header_pixel_scale_arcsec(s.xpixsz_um, s.focallen_mm));

    // The frames to register: the included ones, and the reference when it is a frame.
    let members: Vec<usize> = (0..files.len()).filter(|&i| finished.frames[i].included || Some(i) == ref_idx).collect();
    if members.is_empty() {
        refuse!("no frame is included under this config; there is nothing to register");
    }
    let included = finished.frames.iter().filter(|f| f.included).count();
    if included < 3 {
        let message = format!(
            "only {included} of {} frames are included; a stack needs at least 3 (Athenaeum's run skips such a group)",
            files.len()
        );
        warn_user(ctx, &message);
    }

    let planes = members.iter().filter_map(|&i| finished.frames[i].measurement.as_ref()).map(|m| m.channels.len()).max().unwrap_or(1);
    let resident = if planes >= 3 { REGISTER_PLANES_RESIDENT_OSC } else { REGISTER_PLANES_RESIDENT };
    let working_set = resident * ref_w as u64 * ref_h as u64 * 4;
    let concurrency = usize::try_from(ctx.memory_budget_bytes / working_set.max(1)).unwrap_or(usize::MAX).clamp(1, ctx.pool.current_num_threads().max(1));

    // The cards every registered frame shares.
    let reference_header = if write_registered_files { header_cards(&ref_file)? } else { Vec::new() };
    let reference_wcs: Vec<Card> = reference_header.iter().filter(|c| is_wcs(&c.keyword)).cloned().collect();
    let reference_roworder: Option<Card> = reference_header.iter().find(|c| c.keyword == "ROWORDER").cloned();
    let reference_name = file_name_of(&ref_file);
    let cfg_hash = config_hash(cfg);

    let final_dir = register_dir(work);
    let part_dir = part_path(&final_dir);
    remove_dir_if_present(&part_dir)?;
    std::fs::create_dir_all(&part_dir).with_context(|| format!("cannot create {}", part_dir.display()))?;
    let built = (|| -> Result<(TransformsReport, WeightsReport, Vec<PathBuf>)> {
        let done = std::sync::atomic::AtomicUsize::new(0);
        let total = members.len();
        let results = fan_out(ctx, members.len(), concurrency, |k| {
            let i = members[k];
            let f = &finished.frames[i];
            let outcome = register_one(
                ctx,
                cfg,
                &fits_root,
                &ref_stars,
                ref_scale,
                &sources,
                f,
                Some(i) == ref_idx,
                reuse_fits,
                write_registered_files.then(|| RegisteredOut {
                    dir: part_dir.join(REGISTERED_DIR),
                    wcs: &reference_wcs,
                    roworder: reference_roworder.as_ref(),
                    reference_name: reference_name.as_str(),
                    cfg_hash: cfg_hash.as_str(),
                }),
            );
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            ctx.progress.step("register", n, total);
            outcome
        });
        ctx.check_cancelled()?;

        let mut transforms = TransformsReport::new(ref_file.clone(), cfg.clone(), recorded_versions());
        transforms.reference_pinned = pinned;
        let mut by_frame: std::collections::BTreeMap<usize, Outcome> = std::collections::BTreeMap::new();
        for (k, res) in results.into_iter().enumerate() {
            by_frame.insert(members[k], res.ok_or_else(|| anyhow!("cancelled"))?);
        }
        let mut registered: Vec<PathBuf> = Vec::new();
        let mut inlier_rows: Vec<(i64, usize)> = Vec::new();
        for (i, f) in finished.frames.iter().enumerate() {
            let Some(outcome) = by_frame.remove(&i) else {
                let why = f.reason.clone().unwrap_or_else(|| "excluded".to_string());
                transforms.frames.push(unmapped_row(&f.file, format!("{NOT_REGISTERED}: {why}")));
                continue;
            };
            if let Some(e) = outcome.write_error {
                bail!("{}: {e}", f.file.display());
            }
            match outcome.registration.and_then(|r| r.outcome.map_err(|e| format!("{REGISTRATION_FAILED}: {e}"))) {
                Ok(alignment) => {
                    for note in &alignment.warnings {
                        warn_user(ctx, &format!("{}: {note}", f.file.display()));
                    }
                    if Some(i) != ref_idx {
                        inlier_rows.push((i as i64, alignment.inliers));
                    }
                    if write_registered_files {
                        registered.push(registered_path(work, &f.file));
                    }
                    transforms.frames.push(aligned_row(&f.file, &alignment));
                }
                Err(reason) => {
                    // A registration that failed is excluded or refuses the run, as Athenaeum's does.
                    let reason = if reason.starts_with(REGISTRATION_FAILED) { reason } else { format!("{REGISTRATION_FAILED}: {reason}") };
                    if !cfg.selection.exclude_on_registration_failure {
                        refuse!("{}: {reason}", f.file.display());
                    }
                    warn_user(ctx, &format!("{}: {reason}; the frame is excluded", f.file.display()));
                    transforms.frames.push(unmapped_row(&f.file, reason));
                }
            }
        }
        if let Some((median, flagged)) = low_inlier_frames(&inlier_rows) {
            for (i, inliers) in flagged {
                tracing::warn!(
                    path = %files[i as usize].display(),
                    inliers,
                    median_inliers = median,
                    "registration kept under half its group's median inliers"
                );
            }
        }

        let weights = weigh(ctx, finished.clone(), &transforms, &sources, cfg)?;
        transforms.write(&part_dir.join(TRANSFORMS_FILE))?;
        weights.write(&part_dir.join(WEIGHTS_FILE))?;
        ctx.check_cancelled()?;
        Ok((transforms, weights, registered))
    })();
    let (transforms, weights, registered) = match built {
        Ok(v) => v,
        Err(e) => {
            remove_part(&part_dir);
            return Err(e);
        }
    };

    let mut products = vec![final_dir.join(TRANSFORMS_FILE), final_dir.join(WEIGHTS_FILE)];
    products.extend(registered.iter().cloned());
    let entry = Entry::new(Stage::Register, cfg, files, products);
    if let Err(e) = commit_stage(ctx.work_dir, &[(part_dir.clone(), final_dir)], entry) {
        remove_part(&part_dir);
        return Err(e);
    }
    tracing::info!(
        frames = transforms.frames.len(),
        registered = transforms.frames.iter().filter(|t| t.ok).count(),
        reference = %transforms.reference.display(),
        "register stage written"
    );
    Ok(RegisterReport { transforms, weights, registered })
}

/// Registers one frame: its stars (Measure's fits for a single-plane frame, else detection), the alignment onto the
/// reference, and its registered copy when asked. The reference itself gets the identity.
#[allow(clippy::too_many_arguments)]
fn register_one(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    fits_root: &Path,
    ref_stars: &ReferenceStars,
    ref_scale: Option<f64>,
    sources: &std::collections::BTreeMap<PathBuf, FrameFacts>,
    frame: &MeasureFrame,
    is_reference: bool,
    reuse_fits: bool,
    registered_out: Option<RegisteredOut<'_>>,
) -> Outcome {
    let registration = if is_reference {
        Ok(identity_registration(ref_stars))
    } else {
        (|| {
            let facts = sources.get(&frame.file);
            let frame_scale = facts.and_then(|s| header_pixel_scale_arcsec(s.xpixsz_um, s.focallen_mm));
            // The gate and the order of Athenaeum's `registration_gate_and_hint`, with no plate-solve hint.
            let gate = scale_gate_for(frame_scale, ref_scale);
            let policy = if ratio_wants_seed(scale_ratio_for(frame_scale, ref_scale)) { SeedPolicy::WcsFirst } else { SeedPolicy::QuadFirst };
            let planes = frame.measurement.as_ref().map_or(0, |m| m.channels.len());
            let fits = if reuse_fits && planes == 1 { frame_fits(fits_root, &frame.file, cfg) } else { None };
            let detected = detect_frame_stars(&frame.file, &cfg.registration, Some(&ctx.pool), fits.as_deref())
                .map_err(|e| format!("{REGISTRATION_FAILED}: {e}"))?;
            if ctx.check_cancelled().is_err() {
                return Err("registration cancelled".to_string());
            }
            Ok(register_detected(ref_stars, &detected, &cfg.registration, None, policy, gate))
        })()
    };
    let mut write_error = None;
    if let (Ok(reg), Some(to)) = (&registration, registered_out) {
        if let Ok(alignment) = &reg.outcome {
            let out = to.dir.join(format!("{}.fits", frame_stem(&frame.file)));
            let written = registered_cards(&frame.file, &to)
                .and_then(|cards| write_registered(&frame.file, &alignment.map, ref_stars.width, ref_stars.height, cfg, &cards, &out));
            if let Err(e) = written {
                write_error = Some(format!("cannot write the registered frame: {e:#}"));
            }
        }
    }
    Outcome { registration, write_error }
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
