//! The `measure` stage: stars, noise and background of every calibrated frame, then the weights and the selection.
//!
//! This mirrors Athenaeum's Measure stage (`stage_measure`) for one group, without the catalog. Under the default star
//! profile (`psfModel = auto`) a frame is measured in two passes: pass 1 gives every frame its own Moffat β and is spilled
//! to `<work>/measure/pass1/`; the group β is the lower median of every plane's own β; pass 2 fits every frame at that one
//! β, so no frame's flux is compared across the group at a β of its own. A fixed profile (`moffat4`, `moffat=<β>`)
//! measures a frame in one pass. Both leave the same products under `<work>/measure/`:
//!
//! - `measure.json`: the frames' measurements as the module made them (before the group's aperture-correction fold),
//!   the weights, the selection and the group β ([`MeasureReport`]);
//! - `measure.csv`: one row per frame and channel with the folded values ([`MEASURE_CSV_COLUMNS`]);
//! - `fits/<stem>.p<plane>.athf`: every plane's accepted star fits, the file registration and normalization reuse;
//! - `pass1/<stem>.athm`: the first-pass spills (the `auto` profile only);
//! - with `stars_csv`, `stars/<stem>.p<plane>.csv`: every [`StarFit`] field of every accepted star.
//!
//! [`finish`] is the second half of Athenaeum's `measure_group_finish`: the fold, the weights and the selection, computed
//! from the stored measurements. `measure` runs it once; a later command that runs under another weight mode runs it
//! again over the stored values (P2-R21) and never measures again. The fold lives in `finish` only: reading
//! `measure.json` returns the values before it, and a report `finish` has folded says so ([`MeasureReport::folded`]).
//!
//! Measure reads plain FITS only, as Athenaeum's run does: an XISF or compressed frame is refused with the reason, and
//! `calibrate` writes plain FITS unless `--format xisf` was given (P2-R11). Everything that decodes a pixel or fits a
//! star runs on the context's pool (SL-9).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::{normalize_keyword, read_header_key_map, FrameFacts, HeaderKeyMap};

use crate::calibration_library::flat_cards::slm_cscl;
use crate::geometry::header_pixel_scale_arcsec;
use crate::integration::plane_reader::PlaneReader;
use crate::integration::IntegrationError;
use crate::pipeline::formats::{
    commit_stage, part_path, source_facts, Entry, GroupC, Manifest, MeasureFrame, MeasureReport, MEASURE_CSV_COLUMNS,
    MEASURE_REPORT_FILE,
};
use crate::pipeline::versions::recorded_versions;
use crate::pipeline::{frame_stem, order_lights, warn_user, Ctx, Stage};
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::fits_artifact::write_fits;
use crate::stacking::measure::{
    channel_c_inputs, channel_c_summary, measure_frame_full, measure_frame_pass1_gain, measure_frame_pass2,
    substitute_group_c, ChannelMeasurement, FrameMeasurement, FramePass1, MeasureOptions, SeedOverride,
};
use crate::stacking::pass1_artifact::{read_pass1, write_pass1};
use crate::stacking::psf_signal::{group_beta, PsfModel, StarFit};
use crate::stacking::weights::{
    compute_weights, defers_weight_gate, select_frames, select_frames_deferring_weight_gate, weight_gate, WeightInput,
    WeightMode,
};

/// The stage's folder under the working folder.
pub const MEASURE_DIR: &str = "measure";
/// The folder of a stage's `.athf` star-fit files, under [`MEASURE_DIR`].
const FITS_DIR: &str = "fits";
/// The folder of the first-pass spills, under [`MEASURE_DIR`].
const PASS1_DIR: &str = "pass1";
/// The folder of the optional per-star CSVs, under [`MEASURE_DIR`].
const STARS_DIR: &str = "stars";
/// The `measure.csv` file name, under [`MEASURE_DIR`].
pub const MEASURE_CSV_FILE: &str = "measure.csv";

/// Planes of one channel resident while a frame is measured (Athenaeum's `MEASURE_PLANES_RESIDENT`, measured on a real
/// OSC frame): the working set of one measurement is this many planes of the frame's size, in bytes of `f32`.
const MEASURE_PLANES_RESIDENT: u64 = 8;

/// The reason prefix of a measurement that failed (Athenaeum's text).
const FAILED: &str = "measurement failed";
/// The reason of a frame whose location or scale is not finite (Athenaeum's text).
const NON_FINITE: &str = "measurement invalid: non-finite location/scale";

/// The stage's folder: `<work>/measure`.
pub fn measure_dir(work_dir: &Path) -> PathBuf {
    work_dir.join(MEASURE_DIR)
}

/// Where `measure` leaves the star fits of one plane of `frame`: `<work>/measure/fits/<stem>.p<plane>.athf`
/// ([`read_fits`](crate::stacking::fits_artifact::read_fits) reads it). The file is missing for a frame that was not
/// measured, and when its write failed (that run said so).
pub fn athf_path(work_dir: &Path, frame: &Path, plane: usize) -> PathBuf {
    athf_in(&measure_dir(work_dir), frame_stem(frame), plane)
}

/// Where `measure` leaves the first-pass spill of `frame`: `<work>/measure/pass1/<stem>.athm`. Only the `auto` profile
/// writes one.
pub fn pass1_path(work_dir: &Path, frame: &Path) -> PathBuf {
    pass1_in(&measure_dir(work_dir), frame_stem(frame))
}

pub(crate) fn athf_in(dir: &Path, stem: &str, plane: usize) -> PathBuf {
    dir.join(FITS_DIR).join(format!("{stem}.p{plane}.athf"))
}

fn pass1_in(dir: &Path, stem: &str) -> PathBuf {
    dir.join(PASS1_DIR).join(format!("{stem}.athm"))
}

fn stars_in(dir: &Path, stem: &str, plane: usize) -> PathBuf {
    dir.join(STARS_DIR).join(format!("{stem}.p{plane}.csv"))
}

// ---------------------------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------------------------

/// What `calibrate` does about an XISF frame.
const XISF_HINT: &str = "`starloom calibrate` writes plain FITS unless --format xisf is given, so run it again without that option";
/// What to do about a compressed frame.
const COMPRESSED_HINT: &str = "give measure the plain FITS frames `starloom calibrate` writes, or decompress this one first";

/// The refusal of a frame Measure cannot read (P2-R11): an XISF file or a compressed one, by name or by magic bytes,
/// as `(what it is, what to do)`. `None` for anything else, which the plane reader then judges.
fn unsupported_format(path: &Path) -> Option<(&'static str, &'static str)> {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    if ext.as_deref() == Some("xisf") {
        return Some(("an XISF file", XISF_HINT));
    }
    if matches!(ext.as_deref(), Some("fz" | "gz")) {
        return Some(("a compressed file", COMPRESSED_HINT));
    }
    let mut magic = [0u8; 8];
    let read = std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic));
    match read {
        Ok(()) if magic[..2] == [0x1f, 0x8b] => Some(("a gzip-compressed file", COMPRESSED_HINT)),
        Ok(()) if &magic == b"XISF0100" => Some(("an XISF file", XISF_HINT)),
        _ => None,
    }
}

/// A header card's number by the key as written (Athenaeum reads `FitsHeader::get_f64` with the configured key and does
/// not normalise it): the key map holds the cards' keywords trimmed and upper-cased, as `FitsHeader` does, so a key
/// that normalisation would change (`ssweight`, `SSWEIGHT `) matches no card, exactly as it matches none there. The one
/// divergence is a mixed-case `HIERARCH` keyword, which `FitsHeader` keeps as written and the map upper-cases.
fn number_as_written(map: &HeaderKeyMap, key: &str) -> Option<f64> {
    if normalize_keyword(key).as_deref() != Some(key) {
        return None;
    }
    map.number(key)
}

/// A calibrated frame's gain in e- per calibrated unit (Athenaeum's `gain_from_header`, P2-R7): `EGAIN` times
/// `SLM_CSCL`, when both cards are present and the product is a positive number; otherwise none, and the SNR has no shot
/// term. An unreadable header is logged and gives none.
fn header_gain(path: &Path) -> Option<f64> {
    let map = match read_header_key_map(path) {
        Ok(map) => map,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %format!("{e:#}"), "snr: the header could not be read; no gain");
            return None;
        }
    };
    let gain = number_as_written(&map, "EGAIN").zip(slm_cscl(&map)).map(|(egain, cscl)| egain * cscl);
    if gain.is_none() {
        tracing::debug!(path = %path.display(), "snr: no EGAIN and SLM_CSCL in the header; the shot term is omitted");
    }
    gain.filter(|g| g.is_finite() && *g > 0.0)
}

/// One input frame on its way through the stage.
struct Member {
    /// The frame, absolute.
    file: PathBuf,
    /// The frame's file stem, which names its products.
    stem: String,
    /// Why the frame is out before any pixel is read (unreadable); `None` when it will be measured.
    unreadable: Option<String>,
    /// `(width, height)`, from the plane reader.
    size: (usize, usize),
    /// The gain its measurement runs with.
    gain: Option<f64>,
}

/// What a worker leaves of one measured frame: the measurement and the files it wrote (relative to the stage folder).
struct Measured {
    measurement: FrameMeasurement,
    files: Vec<PathBuf>,
    warnings: Vec<String>,
}

/// What a first-pass worker hands back: the frame's own β per plane, and whether the spill was written (the records
/// themselves are dropped, so one frame's first pass is resident per worker).
struct Pass1Done {
    own: Vec<f64>,
    written: bool,
}

/// The failure of a second pass: the text, and whether the worker had written the spill (it ran pass 1 here).
struct Pass2Failed {
    message: String,
    written: bool,
}

/// Runs `work(0)`, `work(1)`, ... `work(jobs - 1)` on the current rayon pool, at most `concurrency` at once, and returns
/// the results by index. A job not started when the caller's cancel flag was raised is `None`.
pub(crate) fn fan_out<T: Send>(
    ctx: &Ctx<'_>,
    jobs: usize,
    concurrency: usize,
    work: impl Fn(usize) -> T + Sync,
) -> Vec<Option<T>> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..jobs).map(|_| None).collect());
    rayon::scope(|scope| {
        for _ in 0..concurrency.clamp(1, jobs.max(1)) {
            scope.spawn(|_| loop {
                if ctx.check_cancelled().is_err() {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs {
                    return;
                }
                let out = work(i);
                results.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[i] = Some(out);
            });
        }
    });
    results.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------------------------
// The products of a measured frame
// ---------------------------------------------------------------------------------------------------------------

/// A number as the CSV files hold it: shortest round-trip text, empty when it is not finite.
fn num(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        String::new()
    }
}

/// A CSV field, quoted when it holds a separator, a quote or a line break.
fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// The columns of a per-star CSV, in order: every [`StarFit`] field (`flags` as the `.athf` byte, then its names).
const STAR_COLUMNS: [&str; 22] = [
    "x",
    "y",
    "background",
    "amplitude",
    "fwhm_x",
    "fwhm_y",
    "fwtm_x",
    "fwtm_y",
    "theta",
    "beta",
    "residual",
    "signal",
    "area",
    "peak5",
    "sky_ring",
    "sky_ring_sigma",
    "sky_ring_n",
    "snr",
    "ap_sum",
    "hfr",
    "flags",
    "flag_names",
];

fn write_stars_csv(path: &Path, fits: &[StarFit]) -> std::io::Result<()> {
    let mut text = STAR_COLUMNS.join(",");
    text.push('\n');
    for f in fits {
        let row = [
            num(f.x),
            num(f.y),
            num(f.background),
            num(f.amplitude),
            num(f.fwhm_x),
            num(f.fwhm_y),
            num(f.fwtm_x),
            num(f.fwtm_y),
            num(f.theta),
            num(f.beta),
            num(f.residual),
            num(f.signal),
            num(f.area),
            num(f.peak5),
            num(f.sky_ring),
            num(f.sky_ring_sigma),
            f.sky_ring_n.to_string(),
            num(f.snr),
            num(f.ap_sum),
            num(f.hfr),
            f.flags.0.to_string(),
            f.flags.names().join("|"),
        ];
        text.push_str(&row.join(","));
        text.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)
}

/// Writes one measured frame's `.athf` file per plane (and its star CSV when asked) into `dir`. A write that fails
/// degrades, as Athenaeum's does: a consumer falls back to detection, and the failure is a warning here, never silent.
fn persist(dir: &Path, stem: &str, fits: &[Vec<StarFit>], stars_csv: bool) -> (Vec<PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut warnings = Vec::new();
    for (plane, plane_fits) in fits.iter().enumerate() {
        let target = athf_in(dir, stem, plane);
        match write_fits(&target, plane_fits) {
            Ok(()) => files.push(target.strip_prefix(dir).map(Path::to_path_buf).unwrap_or(target)),
            Err(e) => {
                tracing::warn!(path = %target.display(), error = %e, "fits artifact write failed; consumers will fall back");
                warnings.push(format!("{}: the star fits of plane {plane} could not be written ({e})", stem));
            }
        }
        if stars_csv {
            let target = stars_in(dir, stem, plane);
            match write_stars_csv(&target, plane_fits) {
                Ok(()) => files.push(target.strip_prefix(dir).map(Path::to_path_buf).unwrap_or(target)),
                Err(e) => {
                    tracing::warn!(path = %target.display(), error = %e, "star csv write failed");
                    warnings.push(format!("{}: the star list of plane {plane} could not be written ({e})", stem));
                }
            }
        }
    }
    (files, warnings)
}

/// The first pass of one frame, spilled to `spill` best-effort: a failed write is a warning (the first pass then re-runs
/// when the second needs it).
fn run_pass1_and_spill(
    ctx: &Ctx<'_>,
    member: &Member,
    spill: &Path,
    opts: &MeasureOptions,
) -> Result<(FramePass1, bool), IntegrationError> {
    let pass1 = measure_frame_pass1_gain(&member.file, opts, Some(&ctx.pool), ctx.cancel, member.gain)?;
    let written = match write_pass1(spill, &pass1) {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(
                path = %spill.display(),
                error = %e,
                "measure: first-pass artifact write failed; the first pass re-runs when it is needed"
            );
            false
        }
    };
    Ok((pass1, written))
}

// ---------------------------------------------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------------------------------------------

/// Measures `frames` (calibrated plain-FITS frames) and writes `<work>/measure/` ([module docs](self)).
///
/// `cfg` is the effective config; the stage reads its `measurement` (the profile, the detection settings, the weight
/// mode, formula and keyword), `selection` and the scale estimator of `normalization`. The frames are taken in the order
/// of P2-R10 (`DATE-OBS` of the raw light, then input position), and the manifest entry records that order.
///
/// `gain` is `--gain`: the e- per calibrated unit every frame is measured with. Without it a frame's gain is `EGAIN`
/// times `SLM_CSCL` from its header (P2-R7), and a run with frames that have none says so once. `stars_csv` adds a CSV of
/// every accepted star per plane.
///
/// A frame's exposure and date come from the raw light's facts the `calibrate` entry recorded for it, else from the
/// frame's own header (P2-R3); the weight keyword is read from the calibrated frame's header. `fwhm_arcsec` in
/// `measure.csv` needs the pixel size and focal length of those facts and is empty without them.
///
/// A frame whose measurement fails, or whose header cannot be read, is excluded with the reason and a warning and the
/// run goes on; one that is XISF or compressed refuses the whole run (P2-R11), and so does a run in which no frame could
/// be measured (nothing is written then). A group with fewer than 3 included frames is warned about: Athenaeum's run
/// skips such a group, and a stack needs the 3. The returned report is what `measure.json` holds: measurements before the fold, weights and selection
/// after it. The products are built in `<work>/measure.part/` and renamed into place together with the manifest entry
/// ([`commit_stage`]), so a failure or a cancel leaves no entry and no half-written folder. Everything that decodes a
/// pixel or fits a star runs on the context's pool (SL-9).
pub fn measure(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    frames: &[PathBuf],
    gain: Option<f64>,
    stars_csv: bool,
) -> Result<MeasureReport> {
    if frames.is_empty() {
        refuse!("no frames to measure");
    }
    if let Some(g) = gain {
        if !g.is_finite() || g <= 0.0 {
            refuse!("--gain must be a positive number, got {g}");
        }
    }
    let work = std::path::absolute(ctx.work_dir).with_context(|| format!("cannot make {} absolute", ctx.work_dir.display()))?;
    let mut files: Vec<PathBuf> = Vec::with_capacity(frames.len());
    for frame in frames {
        let abs = std::path::absolute(frame).with_context(|| format!("cannot make {} absolute", frame.display()))?;
        if let Some((what, hint)) = unsupported_format(&abs) {
            refuse!("{}: measure reads plain FITS only, and this is {what}; {hint}", abs.display());
        }
        files.push(abs);
    }
    // Two frames with one stem would write one product twice.
    let mut seen: BTreeMap<String, &Path> = BTreeMap::new();
    for file in &files {
        if let Some(first) = seen.insert(frame_stem(file).to_lowercase(), file) {
            refuse!("{} and {} have the same name apart from the extension; their products would overwrite each other", first.display(), file.display());
        }
    }
    ctx.check_cancelled()?;

    // Everything below decodes or fits on the caller's pool, never rayon's global one (SL-9).
    ctx.pool.install(|| run(ctx, cfg, &work, files, gain, stars_csv))
}

fn run(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    work: &Path,
    files: Vec<PathBuf>,
    gain: Option<f64>,
    stars_csv: bool,
) -> Result<MeasureReport> {
    // The raw lights' facts: exposure and date (P2-R3), and the pixel scale.
    let manifest = Manifest::load(ctx.work_dir)?;
    // A frame whose header cannot be read has no facts; the plane reader then excludes it with its own reason, as
    // Athenaeum excludes a calibrated frame it cannot read, and the run goes on.
    let mut sources: BTreeMap<PathBuf, FrameFacts> = BTreeMap::new();
    for file in &files {
        match source_facts(&manifest, std::slice::from_ref(file)) {
            Ok(one) => sources.extend(one),
            Err(e) => {
                tracing::warn!(path = %file.display(), error = %format!("{e:#}"), "frame header facts could not be read; the frame has none");
                sources.insert(file.clone(), FrameFacts::default());
            }
        }
    }
    let facts: Vec<FrameFacts> = files.iter().map(|f| sources[f].clone()).collect();
    let order = order_lights(&files, &facts)?;
    let files: Vec<PathBuf> = order.iter().map(|&i| files[i].clone()).collect();

    let opts = cfg.measurement.measure_options(cfg.normalization.scale_estimator);
    let auto = matches!(cfg.measurement.psf_model, PsfModel::Auto);

    // What each frame is: its size and planes, and its gain.
    let mut members: Vec<Member> = Vec::with_capacity(files.len());
    for file in &files {
        let (unreadable, size) = match PlaneReader::open(file) {
            Ok(r) if r.channels() > 0 => (None, (r.width(), r.height())),
            Ok(_) => (Some("calibrated frame has no planes".to_string()), (0, 0)),
            Err(e) => (Some(format!("calibrated frame unreadable: {e}")), (0, 0)),
        };
        let frame_gain = if unreadable.is_none() { gain.or_else(|| header_gain(file)) } else { None };
        members.push(Member { file: file.clone(), stem: frame_stem(file).to_string(), unreadable, size, gain: frame_gain });
    }
    // One user-facing warning: the sentence to the progress sink and the log, once ([`warn_user`]).
    let warn = |message: &str| warn_user(ctx, message);
    for m in members.iter().filter(|m| m.unreadable.is_some()) {
        let reason = m.unreadable.as_deref().unwrap_or_default();
        warn(&format!("{}: {reason}; the frame is excluded", m.file.display()));
    }
    let measurable: Vec<usize> = (0..members.len()).filter(|&i| members[i].unreadable.is_none()).collect();
    if gain.is_none() {
        let missing = measurable.iter().filter(|&&i| members[i].gain.is_none()).count();
        if missing > 0 {
            warn(&format!(
                "{missing} of {} frames have no gain (EGAIN and SLM_CSCL are not both in their headers), so their SNR has no \
                 shot term; give --gain to set one",
                measurable.len()
            ));
        }
    }

    let final_dir = measure_dir(work);
    let part_dir = part_path(&final_dir);
    remove_dir_if_present(&part_dir)?;
    std::fs::create_dir_all(&part_dir).with_context(|| format!("cannot create {}", part_dir.display()))?;
    let built = build(ctx, cfg, &opts, auto, &members, &measurable, &sources, &part_dir, stars_csv, &warn);
    let (report, csv, mut produced) = match built {
        Ok(v) => v,
        Err(e) => {
            if let Err(clean) = std::fs::remove_dir_all(&part_dir) {
                tracing::warn!(path = %part_dir.display(), error = %clean, "cannot remove a half-written stage folder");
            }
            return Err(e);
        }
    };
    let finished = (|| -> Result<()> {
        report.write(&part_dir.join(MEASURE_REPORT_FILE))?;
        std::fs::write(part_dir.join(MEASURE_CSV_FILE), csv).with_context(|| format!("cannot write {}", MEASURE_CSV_FILE))?;
        ctx.check_cancelled()
    })();
    if let Err(e) = finished {
        if let Err(clean) = std::fs::remove_dir_all(&part_dir) {
            tracing::warn!(path = %part_dir.display(), error = %clean, "cannot remove a half-written stage folder");
        }
        return Err(e);
    }

    let mut products = vec![final_dir.join(MEASURE_REPORT_FILE), final_dir.join(MEASURE_CSV_FILE)];
    produced.sort();
    products.extend(produced.into_iter().map(|rel| final_dir.join(rel)));
    let entry = Entry::new(Stage::Measure, cfg, files, products);
    if let Err(e) = commit_stage(ctx.work_dir, &[(part_dir.clone(), final_dir)], entry) {
        if let Err(clean) = std::fs::remove_dir_all(&part_dir) {
            if clean.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %part_dir.display(), error = %clean, "cannot remove a half-written stage folder");
            }
        }
        return Err(e);
    }
    tracing::info!(
        frames = report.frames.len(),
        included = report.frames.iter().filter(|f| f.included).count(),
        group_beta = report.group_beta,
        "measure stage written"
    );
    Ok(report)
}

/// Removes a folder a killed run left, when there is one.
fn remove_dir_if_present(dir: &Path) -> Result<()> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("cannot clear {}", dir.display())),
    }
}

/// The measurement proper: every measurable frame measured into `dir`, then the weights and selection. Returns the report
/// `measure.json` holds, the text of `measure.csv` and the files written (relative to the stage folder).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    opts: &MeasureOptions,
    auto: bool,
    members: &[Member],
    measurable: &[usize],
    sources: &BTreeMap<PathBuf, FrameFacts>,
    dir: &Path,
    stars_csv: bool,
    warn: &dyn Fn(&str),
) -> Result<(MeasureReport, String, Vec<PathBuf>)> {
    let mut report = MeasureReport::new(cfg.clone(), recorded_versions());
    report.frames = members
        .iter()
        .map(|m| MeasureFrame {
            file: m.file.clone(),
            measurement: None,
            weight: None,
            included: false,
            reason: m.unreadable.clone(),
            measure_floored: false,
        })
        .collect();
    let mut produced: Vec<PathBuf> = Vec::new();

    let (max_w, max_h) = measurable.iter().fold((0u64, 0u64), |(w, h), &i| (w.max(members[i].size.0 as u64), h.max(members[i].size.1 as u64)));
    let working_set = MEASURE_PLANES_RESIDENT * max_w * max_h * 4;
    let concurrency = usize::try_from(ctx.memory_budget_bytes / working_set.max(1))
        .unwrap_or(usize::MAX)
        .clamp(1, ctx.pool.current_num_threads().max(1));
    let total = measurable.len() * if auto { 2 } else { 1 };
    let done = AtomicUsize::new(0);
    let tick = |units: usize| {
        let n = done.fetch_add(units, Ordering::Relaxed) + units;
        ctx.progress.step("measure", n, total);
    };
    let exclude = |report: &mut MeasureReport, i: usize, reason: String| {
        tracing::warn!(path = %members[i].file.display(), reason = %reason, "frame excluded");
        report.frames[i].included = false;
        report.frames[i].reason = Some(reason);
    };

    if !auto {
        let results = fan_out(ctx, measurable.len(), concurrency, |k| {
            let m = &members[measurable[k]];
            let out = measure_frame_full(&m.file, opts, Some(&ctx.pool), ctx.cancel, SeedOverride::Fast, m.gain)
                .map_err(|e| format!("{FAILED}: {e}"))
                .map(|(measurement, fits)| {
                    let (files, warnings) = persist(dir, &m.stem, &fits, stars_csv);
                    Measured { measurement, files, warnings }
                });
            tick(1);
            out
        });
        ctx.check_cancelled()?;
        for (k, res) in results.into_iter().enumerate() {
            let i = measurable[k];
            match res.ok_or_else(|| anyhow!("cancelled"))? {
                Ok(done) => take(&mut report, i, done, &mut produced, warn),
                Err(message) => exclude(&mut report, i, message),
            }
        }
    } else {
        // Pass 1: every frame's own β, the first pass spilled and dropped.
        let first = fan_out(ctx, measurable.len(), concurrency, |k| {
            let m = &members[measurable[k]];
            let out = run_pass1_and_spill(ctx, m, &pass1_in(dir, &m.stem), opts)
                .map(|(pass1, written)| Pass1Done { own: pass1.beta_own().to_vec(), written })
                .map_err(|e| format!("{FAILED}: {e}"));
            tick(1);
            out
        });
        ctx.check_cancelled()?;
        let mut own: Vec<Option<Vec<f64>>> = vec![None; members.len()];
        let mut spilled: Vec<bool> = vec![false; members.len()];
        let mut failed = 0usize;
        for (k, res) in first.into_iter().enumerate() {
            let i = measurable[k];
            match res.ok_or_else(|| anyhow!("cancelled"))? {
                Ok(p1) => {
                    own[i] = Some(p1.own);
                    spilled[i] = p1.written;
                }
                Err(message) => {
                    failed += 1;
                    exclude(&mut report, i, message);
                }
            }
        }
        // A frame that failed pass 1 has no second half; the bar still lands on its total.
        if failed > 0 {
            tick(failed);
        }

        // The group β, once: the lower median of every measured plane's own β (a frame whose pass 2 then fails counts).
        let own_all: Vec<f64> = own.iter().flatten().flatten().copied().collect();
        if !own_all.is_empty() {
            let gb = group_beta(&own_all);
            let (lo, hi) = own_all.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &b| (lo.min(b), hi.max(b)));
            tracing::info!(beta = gb, count = own.iter().flatten().count(), beta_own_min = lo, beta_own_max = hi, "measure: group beta resolved; every member is fitted at it");
            report.group_beta = Some(gb);

            // Pass 2: every frame fitted at the group β, its first pass read back from the spill (or re-run here).
            let second_jobs: Vec<usize> = measurable.iter().copied().filter(|&i| own[i].is_some()).collect();
            let second = fan_out(ctx, second_jobs.len(), concurrency, |k| {
                let i = second_jobs[k];
                let m = &members[i];
                let spill = pass1_in(dir, &m.stem);
                let from_spill = if spilled[i] {
                    match read_pass1(&spill, &m.file, opts) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            tracing::warn!(path = %spill.display(), error = %e, "measure: first-pass artifact refused on read; pass 1 re-runs");
                            None
                        }
                    }
                } else {
                    None
                };
                let first = match from_spill {
                    Some(p) => Ok((p, spilled[i])),
                    None => run_pass1_and_spill(ctx, m, &spill, opts),
                };
                let out = match first {
                    Err(e) => Err(Pass2Failed { message: format!("{FAILED}: {e}"), written: false }),
                    Ok((p, written)) => {
                        let betas = vec![gb; p.beta_own().len()];
                        match measure_frame_pass2(&m.file, p, &betas, opts, Some(&ctx.pool), ctx.cancel) {
                            Ok((measurement, fits)) => {
                                let (files, warnings) = persist(dir, &m.stem, &fits, stars_csv);
                                Ok((Measured { measurement, files, warnings }, written))
                            }
                            Err(e) => Err(Pass2Failed { message: format!("{FAILED}: {e}"), written }),
                        }
                    }
                };
                tick(1);
                out
            });
            ctx.check_cancelled()?;
            for (k, res) in second.into_iter().enumerate() {
                let i = second_jobs[k];
                match res.ok_or_else(|| anyhow!("cancelled"))? {
                    Ok((done, written)) => {
                        spilled[i] = spilled[i] || written;
                        take(&mut report, i, done, &mut produced, warn);
                    }
                    Err(failure) => {
                        spilled[i] = spilled[i] || failure.written;
                        exclude(&mut report, i, failure.message);
                    }
                }
            }
        }
        for &i in measurable {
            if spilled[i] {
                produced.push(PathBuf::from(PASS1_DIR).join(format!("{}.athm", members[i].stem)));
            }
        }
    }

    // A measure in which no frame was measured has nothing to store: it fails, naming why each frame is out.
    if report.frames.iter().all(|f| f.measurement.is_none()) {
        let reasons: Vec<String> = report
            .frames
            .iter()
            .map(|f| format!("{}: {}", f.file.display(), f.reason.as_deref().unwrap_or("not measured")))
            .collect();
        bail!("no frame could be measured: {}", reasons.join("; "));
    }

    // Weights and selection over what was measured; the stored measurements stay as the module made them.
    let finished = finish(&report, cfg, sources);
    for (stored, done) in report.frames.iter_mut().zip(&finished.frames) {
        stored.weight = done.weight.clone();
        stored.included = done.included;
        stored.reason = done.reason.clone();
        stored.measure_floored = done.measure_floored;
    }
    report.group_c = finished.group_c.clone();
    let included = report.frames.iter().filter(|f| f.included).count();
    if included < 3 {
        warn(&format!("only {included} of {} frames are included; a stack needs at least 3", report.frames.len()));
    }
    let csv = measure_csv(&finished, sources);
    Ok((report, csv, produced))
}

/// Records a measured frame in the report and its files in `produced`; the worker's warnings are raised here.
fn take(report: &mut MeasureReport, i: usize, done: Measured, produced: &mut Vec<PathBuf>, warn: &dyn Fn(&str)) {
    report.frames[i].measurement = Some(done.measurement);
    produced.extend(done.files);
    for w in &done.warnings {
        warn(w);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Weights and selection
// ---------------------------------------------------------------------------------------------------------------

/// The weight keyword's value in a calibrated frame's header, by the key as written ([`number_as_written`]); `None`
/// when the header is unreadable (logged: the frame then has no weight) or the card is missing.
fn keyword_value(path: &Path, keyword: &str) -> Option<f64> {
    let map: HeaderKeyMap = match read_header_key_map(path) {
        Ok(map) => map,
        Err(e) => {
            tracing::warn!(path = %path.display(), keyword, error = %format!("{e:#}"), "weight keyword: the header could not be read; no value");
            return None;
        }
    };
    number_as_written(&map, keyword)
}

/// The second half of Athenaeum's `measure_group_finish`, over the stored measurements: the group's aperture-correction
/// fold, the weights of `cfg.measurement.weight_mode`, and the selection of `cfg.selection`. Returns a copy of `report`
/// with `config` set to `cfg`, every measured frame's measurement **folded** (a plane at the fit-call floor takes its
/// channel's median aperture correction), and its `weight`, `included`, `reason` and `measure_floored` recomputed; a frame
/// with no measurement keeps its reason. `group_c` holds the per-plane inputs of the fold.
///
/// `report` holds the module's values, as `measure.json` stores them. The result is marked
/// [`folded`](MeasureReport::folded); a report that is already marked is not folded again (a second fold would scale
/// the substituted planes a second time), its weights and selection are recomputed over the values it holds and its
/// `group_c` is kept. `sources` gives each frame's raw-light facts by the frame's path: the exposure
/// the exposure weight reads comes from there (P2-R3; a frame with no entry has none), and a caller takes the map from
/// the manifest ([`source_facts`]). The keyword weight reads the calibrated frame's header, as Athenaeum's run does.
pub fn finish(report: &MeasureReport, cfg: &AlgorithmConfig, sources: &BTreeMap<PathBuf, FrameFacts>) -> MeasureReport {
    let mut out = report.clone();
    out.config = cfg.clone();
    let already_folded = out.folded;
    out.folded = true;
    if !already_folded {
        out.group_c.clear();
    }
    for f in &mut out.frames {
        if f.measurement.is_some() {
            f.weight = None;
            f.included = false;
            f.reason = None;
            f.measure_floored = false;
        }
    }

    // The fold: one channel at a time, over every measured member (never narrowed to the included ones).
    let channels = if already_folded {
        0
    } else {
        out.frames.iter().filter_map(|f| f.measurement.as_ref()).map(|m| m.channels.len()).max().unwrap_or(0)
    };
    for plane in 0..channels {
        let kept = channel_c_inputs(out.frames.iter().filter_map(|f| f.measurement.as_ref()).filter_map(|m| m.channels.get(plane)));
        let c = channel_c_summary(&kept);
        out.group_c.push(GroupC { plane, planes: c.planes, count: c.count, median: c.median, min: c.min, max: c.max });
        match (c.median, c.min, c.max) {
            (Some(median), Some(min), Some(max)) => tracing::info!(
                plane,
                planes = c.planes,
                count = c.count,
                aperture_correction = median,
                aperture_correction_min = min,
                aperture_correction_max = max,
                "measure: aperture correction resolved per channel"
            ),
            _ => tracing::info!(plane, planes = c.planes, count = c.count, "measure: aperture correction resolved per channel"),
        }
        let mut at_plane: Vec<&mut ChannelMeasurement> = out
            .frames
            .iter_mut()
            .filter_map(|f| f.measurement.as_mut())
            .filter_map(|m| m.channels.get_mut(plane))
            .collect();
        if let Some((m, count)) = substitute_group_c(&mut at_plane) {
            tracing::info!(plane, count, aperture_correction = m, "measure: floored planes took the group channel's median aperture correction");
        }
    }

    // A non-finite location or scale takes the frame out of the weighing.
    let mut weigh: Vec<usize> = Vec::new();
    for (i, f) in out.frames.iter_mut().enumerate() {
        let Some(m) = &f.measurement else { continue };
        if m.channels.iter().any(|c| !c.location.is_finite() || !c.scale.is_finite()) {
            f.included = false;
            f.reason = Some(NON_FINITE.to_string());
            continue;
        }
        weigh.push(i);
    }
    if weigh.is_empty() {
        return out;
    }

    let mode = cfg.measurement.weight_mode;
    if mode == WeightMode::Keyword && normalize_keyword(&cfg.measurement.keyword).as_deref() != Some(cfg.measurement.keyword.as_str()) {
        tracing::warn!(
            keyword = %cfg.measurement.keyword,
            "weight keyword is not a card name as written (upper case, no surrounding space); no frame has a value"
        );
    }
    let keywords: Vec<Option<f64>> = weigh
        .iter()
        .map(|&i| if mode == WeightMode::Keyword { keyword_value(&out.frames[i].file, &cfg.measurement.keyword) } else { None })
        .collect();
    let (weights, reasons) = {
        let inputs: Vec<WeightInput> = weigh
            .iter()
            .zip(&keywords)
            .filter_map(|(&i, &keyword_value)| {
                let f = &out.frames[i];
                Some(WeightInput {
                    measurement: f.measurement.as_ref()?,
                    exposure_s: sources.get(&f.file).and_then(|s| s.exptime_s),
                    keyword_value,
                })
            })
            .collect();
        let w = compute_weights(&inputs, mode, &cfg.measurement.formula, &vec![false; inputs.len()]);
        let manual = vec![false; inputs.len()];
        // The PSF family's 5 % floor is applied to the frame-common weight after registration; here only a weight of zero
        // is refused.
        let sel = if defers_weight_gate(mode) {
            select_frames_deferring_weight_gate(&inputs, &w, &manual, &cfg.selection)
        } else {
            select_frames(&inputs, &w, &manual, &cfg.selection)
        };
        (w, sel)
    };
    for (k, &i) in weigh.iter().enumerate() {
        let f = &mut out.frames[i];
        match &reasons[k] {
            Some(reason) => {
                f.included = false;
                f.reason = Some(reason.clone());
                tracing::warn!(path = %f.file.display(), reason = %reason, "frame excluded");
            }
            None => {
                f.included = true;
                // The deferred floor will judge the frame after registration; the reference pick must not see it.
                f.measure_floored = defers_weight_gate(mode) && weight_gate(&weights[k], &cfg.selection).is_some();
            }
        }
        tracing::debug!(path = %f.file.display(), weight = weights[k].normalized_mean, included = f.included, "frame measured");
        f.weight = Some(weights[k].clone());
    }
    out
}

/// `measure.csv`: one row per frame and channel with the folded values of `finished`, in [`MEASURE_CSV_COLUMNS`] order.
/// A frame with no measurement gets one row with its reason. `fwhm_arcsec` is `fwhm_px` times the pixel scale of the
/// raw light's pixel size and focal length, empty without them; `weight` is the channel's normalized weight; a
/// non-finite number is empty.
fn measure_csv(finished: &MeasureReport, sources: &BTreeMap<PathBuf, FrameFacts>) -> String {
    let mut text = MEASURE_CSV_COLUMNS.join(",");
    text.push('\n');
    for f in &finished.frames {
        let file = csv_field(&f.file.to_string_lossy());
        let reason = csv_field(f.reason.as_deref().unwrap_or(""));
        let scale = sources.get(&f.file).and_then(|s| header_pixel_scale_arcsec(s.xpixsz_um, s.focallen_mm));
        let tail = [f.included.to_string(), reason];
        let Some(m) = &f.measurement else {
            let mut row = vec![file, String::new()];
            row.extend(std::iter::repeat(String::new()).take(MEASURE_CSV_COLUMNS.len() - 4));
            row.extend(tail);
            text.push_str(&row.join(","));
            text.push('\n');
            continue;
        };
        for (ch, c) in m.channels.iter().enumerate() {
            let weight = f.weight.as_ref().and_then(|w| w.normalized.get(ch)).copied();
            let row = [
                file.clone(),
                ch.to_string(),
                num(c.fwhm_px),
                scale.map(|s| num(c.fwhm_px * s)).unwrap_or_default(),
                num(c.eccentricity),
                c.stars_detected.to_string(),
                c.stars_fitted.to_string(),
                num(c.snr_weight),
                num(c.psf_signal_weight),
                num(c.psf_snr),
                num(c.noise),
                num(c.median),
                num(c.mad),
                num(c.location),
                num(c.scale),
                weight.map(num).unwrap_or_default(),
                tail[0].clone(),
                tail[1].clone(),
            ];
            text.push_str(&row.join(","));
            text.push('\n');
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header that cannot be read is logged, never dropped silently: the weight keyword and the gain both say so.
    #[test]
    fn an_unreadable_header_is_logged_and_gives_no_keyword_value_and_no_gain() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.fits");
        let (value, logs) = crate::test_support::capture_logs(|| keyword_value(&missing, "SSWEIGHT"));
        assert_eq!(value, None);
        assert!(logs.contains("weight keyword: the header could not be read"), "{logs}");
        let (gain, logs) = crate::test_support::capture_logs(|| header_gain(&missing));
        assert_eq!(gain, None);
        assert!(logs.contains("snr: the header could not be read"), "{logs}");
    }
}
