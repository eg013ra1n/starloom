//! The `master` stage: raw bias, dark or flat frames integrated into one master calibration frame.
//!
//! This is Athenaeum's master build (`run_build`) without the catalog. The frames may be plain FITS, XISF, or
//! compressed (`.fz`, `.fits.gz`, an extension HDU): the integration engine reads a plain 2-D FITS frame directly and
//! spills every other file through the image reader (P2-R22). The master is raw sensor DN (`XisfBounds::Adu16`), the
//! header is the standard card set of an Athenaeum master with `SLM_*` cards in place of its own provenance cards (P2-R6), and a flat carries
//! its normalization constants (`SLM_FNRM`, and for a CFA flat `SLM_FNR`/`SLM_FNG`/`SLM_FNB`).
//!
//! Every mismatch a master can be refused for (a frame with more than one plane, frames of different sizes, a
//! pre-calibration master of another size) is found from the headers, before any pixel is read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use astroimage::header::{read_frame_facts, FrameFacts};
use chrono::{DateTime, Utc};

use crate::calibration_library::flat_cards::{SLM_FNB, SLM_FNG, SLM_FNR, SLM_FNRM};
use crate::fits_writer::bayer::Bayer;
use crate::fits_writer::{write_image_f32, Card, CardValue, OutputFormat, XisfBounds};
use crate::integration::band_budget::{MAX_BUDGET_BYTES, MIN_BUDGET_BYTES};
use crate::integration::banded::{BandPlanes, BandSource};
use crate::integration::cfa::{flat_channel_norms, CfaGeometry};
use crate::integration::combine::{Combination, IntegrationRecipe, Rejection};
use crate::integration::engine::{integrate_bias_like, integrate_flat, EngineProgress, FlatPrecal};
use crate::integration::io_policy::IoPolicy;
use crate::integration::recipe::resolve_recipe;
use crate::pipeline::cards::{common_cards, SLM_N, SLM_REC, SLM_TMAX, SLM_TMIN};
use crate::pipeline::config::config_hash;
use crate::pipeline::formats::TMP_DIR;
use crate::pipeline::inputs::{parse_date_obs, FlatPrecalSource};
use crate::pipeline::Ctx;
use crate::stacking::config::AlgorithmConfig;
use crate::storage_class::{read_concurrency, StorageClass};

/// The fewest frames a master is built from (Athenaeum's `MIN_MASTER_FRAMES`).
pub const MIN_MASTER_FRAMES: usize = 3;

/// Which calibration frame a master is built from. Dark flats are built as [`MasterKind::Dark`] or
/// [`MasterKind::Bias`]: the stage only needs to know whether the frames are flats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterKind {
    /// Bias frames (zero-length exposures).
    Bias,
    /// Dark frames.
    Dark,
    /// Flat frames, optionally pre-calibrated with a bias, a dark flat or a constant.
    Flat,
}

impl MasterKind {
    /// The frame type the recipe rule is keyed on (`resolve_recipe`'s `imagetyp`).
    fn frame_type(self) -> &'static str {
        match self {
            MasterKind::Bias => "Bias",
            MasterKind::Dark => "Dark",
            MasterKind::Flat => "Flat",
        }
    }

    /// The master's `IMAGETYP` card, spelled as Athenaeum's master builder spells it.
    pub fn imagetyp(self) -> &'static str {
        match self {
            MasterKind::Bias => "Master Bias",
            MasterKind::Dark => "Master Dark",
            MasterKind::Flat => "Master Flat",
        }
    }

    /// The lower-case word messages and file names use.
    pub fn name(self) -> &'static str {
        match self {
            MasterKind::Bias => "bias",
            MasterKind::Dark => "dark",
            MasterKind::Flat => "flat",
        }
    }
}

/// What a master build did.
#[derive(Debug, Clone)]
pub struct MasterReport {
    /// The master file.
    pub path: PathBuf,
    /// What was built.
    pub kind: MasterKind,
    /// How many frames were integrated.
    pub frames: usize,
    /// The recipe used: the one given, else the automatic one for this kind and frame count.
    pub recipe: IntegrationRecipe,
    /// The recipe as `SLM_REC` spells it.
    pub recipe_text: String,
    /// The master's width, in pixels.
    pub width: usize,
    /// The master's height, in pixels.
    pub height: usize,
    /// The fraction of all input samples that rejection dropped, in `[0, 1]`.
    pub rejected_fraction: f64,
    /// A flat's global normalization constant (`SLM_FNRM`), in DN.
    pub flat_norm: Option<f64>,
    /// A CFA flat's per-channel constants (`SLM_FNR`/`SLM_FNG`/`SLM_FNB`), in DN.
    pub channel_norms: Option<[f64; 3]>,
    /// Warnings the build raised (undefined samples excluded, constants left out); also sent to the context's
    /// [`Progress`](crate::pipeline::Progress).
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// Shared helpers (the calibrate stage uses them too)
// ---------------------------------------------------------------------------------------------------------------

/// The scratch folder (`<work>/.tmp/`) a stage spills decoded frames into, created when missing. The command clears it
/// when it starts ([`begin_writing`](crate::pipeline::begin_writing)); a stage never clears it.
pub(crate) fn scratch_dir(ctx: &Ctx<'_>) -> Result<PathBuf> {
    let dir = ctx.work_dir.join(TMP_DIR);
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    Ok(dir)
}

/// The catalog-rule facts of a frame, read from its header alone. The error names the file.
pub(crate) fn frame_facts(path: &Path) -> Result<FrameFacts> {
    read_frame_facts(path).with_context(|| format!("cannot read the header of {}", path.display()))
}

/// The refusal of a frame with more than one plane, in the words the integration reader uses for the same frame.
pub(crate) fn planes_refusal(path: &Path, planes: i64) -> String {
    format!(
        "{}: {planes}-channel image; frames must be 1-channel for calibration (CFA mosaics stay 1-channel; debayered files cannot be calibrated)",
        path.display()
    )
}

/// A frame's `(width, height)` when its header records both.
pub(crate) fn size_of(facts: &FrameFacts) -> Option<(i64, i64)> {
    facts.width.zip(facts.height)
}

/// A frame's mosaic phase for per-channel flat scaling and the debayer, or `None` when it declares no CFA pattern
/// (mono) or one [`Bayer::parse`] cannot vouch for. A missing offset reads as 0: the guess only decides which pixels
/// are grouped, whereas writing it into `XBAYROFF` would be a claim every later debayer acts on.
pub(crate) fn cfa_of(facts: &FrameFacts) -> Option<CfaGeometry> {
    let raw = facts.bayerpat.as_deref().map(str::trim).filter(|s| !s.is_empty())?;
    let Some(pattern) = Bayer::parse(raw) else {
        tracing::warn!(bayerpat = %raw, "unrecognized bayer pattern; per-channel flat scaling and the debayer are not applied");
        return None;
    };
    if facts.xbayroff.is_none() || facts.ybayroff.is_none() {
        tracing::debug!("cfa phase not declared; the missing offset is taken as 0");
    }
    Some(CfaGeometry { pattern, xoff: facts.xbayroff.unwrap_or(0), yoff: facts.ybayroff.unwrap_or(0) })
}

/// The I/O settings of a run on the context's memory budget: the band budget is the stage's budget within the
/// engine's own floor and ceiling, and the reads in flight are the pool's width.
fn io_policy(ctx: &Ctx<'_>) -> IoPolicy {
    let budget = usize::try_from(ctx.memory_budget_bytes).unwrap_or(usize::MAX);
    IoPolicy {
        band_budget_bytes: budget.clamp(MIN_BUDGET_BYTES, MAX_BUDGET_BYTES),
        read_concurrency: read_concurrency(StorageClass::Local, 0, ctx.pool.current_num_threads()),
        storage: StorageClass::Local,
        prefetch_local: false,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The master header
// ---------------------------------------------------------------------------------------------------------------

/// The value most frames agree on (the smallest wins a tie), and whether any frame disagreed.
fn majority<T: Ord + Clone>(values: impl Iterator<Item = T>) -> (Option<T>, bool) {
    let mut counts: BTreeMap<T, usize> = BTreeMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let disagreed = counts.len() > 1;
    let mut best: Option<(&T, usize)> = None;
    for (v, &n) in &counts {
        if best.is_none_or(|(_, m)| n > m) {
            best = Some((v, n));
        }
    }
    (best.map(|(v, _)| v.clone()), disagreed)
}

/// What a master's header says, consolidated from the facts of its member frames the way Athenaeum consolidates a
/// calibration set: the CFA geometry by majority, temperatures by mean, the rest from the first member that has it.
struct Consolidated {
    exptime: Option<f64>,
    gain: Option<f64>,
    offset: Option<f64>,
    xbinning: Option<i64>,
    ybinning: Option<i64>,
    xpixsz: Option<f64>,
    ypixsz: Option<f64>,
    instrume: Option<String>,
    telescop: Option<String>,
    focallen: Option<f64>,
    filter: Option<String>,
    bayerpat: Option<String>,
    xbayroff: Option<i64>,
    ybayroff: Option<i64>,
    roworder: Option<String>,
    temp_mean: Option<f64>,
    temp_span: Option<(f64, f64)>,
    date_midpoint: Option<DateTime<Utc>>,
    origin: Option<(i64, i64)>,
}

impl Consolidated {
    fn from_facts(facts: &[FrameFacts]) -> Consolidated {
        let first = |f: &dyn Fn(&FrameFacts) -> Option<String>| facts.iter().find_map(f);
        let first_f = |f: &dyn Fn(&FrameFacts) -> Option<f64>| facts.iter().find_map(f);
        let text = |s: &Option<String>| s.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_ascii_uppercase);
        let (bayerpat, dis_pat) = majority(facts.iter().filter_map(|f| text(&f.bayerpat)));
        let (xbayroff, dis_x) = majority(facts.iter().filter_map(|f| f.xbayroff));
        let (ybayroff, dis_y) = majority(facts.iter().filter_map(|f| f.ybayroff));
        let (roworder, dis_row) = majority(facts.iter().filter_map(|f| text(&f.roworder)));
        for (field, disagreed) in [("bayerpat", dis_pat), ("xbayroff", dis_x), ("ybayroff", dis_y), ("roworder", dis_row)] {
            if disagreed {
                tracing::warn!(field, "bayer metadata disagrees across the master's frames; using the majority");
            }
        }
        let temps: Vec<f64> = facts.iter().filter_map(|f| f.ccd_temp).filter(|t| t.is_finite()).collect();
        let temp_mean = (!temps.is_empty()).then(|| temps.iter().sum::<f64>() / temps.len() as f64);
        let temp_span = temps.iter().copied().reduce(f64::min).zip(temps.iter().copied().reduce(f64::max));
        let dates: Vec<DateTime<Utc>> =
            facts.iter().filter_map(|f| f.date_obs.as_deref().and_then(|d| parse_date_obs(d, f.time_obs.as_deref()))).collect();
        let date_midpoint = match (dates.iter().min(), dates.iter().max()) {
            (Some(&a), Some(&b)) => Some(a + (b - a) / 2),
            _ => None,
        };
        let origins: Vec<Option<(i64, i64)>> = facts.iter().map(|f| f.origin).collect();
        let origin = match origins.first().copied().flatten() {
            Some(o) if origins.iter().all(|x| *x == Some(o)) => Some(o),
            Some(_) => {
                tracing::warn!("master origin not stamped: the frames disagree or some record none");
                None
            }
            None => None,
        };
        Consolidated {
            exptime: first_f(&|f| f.exptime_s),
            gain: first_f(&|f| f.gain),
            offset: first_f(&|f| f.offset),
            xbinning: facts.iter().filter_map(|f| f.xbinning).max(),
            ybinning: facts.iter().filter_map(|f| f.ybinning).max(),
            xpixsz: facts.iter().filter_map(|f| f.xpixsz_um).reduce(f64::max),
            ypixsz: facts.iter().filter_map(|f| f.ypixsz_um).reduce(f64::max),
            instrume: first(&|f| f.instrume.clone()),
            telescop: first(&|f| f.telescop.clone()),
            focallen: first_f(&|f| f.focallen_mm),
            filter: first(&|f| f.filter.clone()),
            bayerpat,
            xbayroff,
            ybayroff,
            roworder,
            temp_mean,
            temp_span,
            date_midpoint,
            origin,
        }
    }
}

fn card(keyword: &str, value: CardValue, comment: &str) -> Result<Card> {
    Ok(Card::new(keyword, value)?.with_comment(comment))
}

/// The recipe as `SLM_REC` spells it: the combination, the rejection and its numbers, lower case and ASCII
/// (`average winsorized 3 3`).
pub fn recipe_text(recipe: &IntegrationRecipe) -> String {
    let combination = match recipe.combination {
        Combination::Average => "average",
        Combination::Median => "median",
    };
    let rejection = match recipe.rejection {
        Rejection::None => "none".to_string(),
        Rejection::PercentileClip { low, high } => format!("percentile {low} {high}"),
        Rejection::SigmaClip { sigma_low, sigma_high } => format!("sigma {sigma_low} {sigma_high}"),
        Rejection::WinsorizedSigma { sigma_low, sigma_high } => format!("winsorized {sigma_low} {sigma_high}"),
        Rejection::LinearFitClip { sigma_low, sigma_high } => format!("linearfit {sigma_low} {sigma_high}"),
        Rejection::MinMax { low, high } => format!("minmax {low} {high}"),
        Rejection::Esd { outliers_fraction, alpha, low_relaxation } => {
            format!("esd {outliers_fraction} {alpha} {low_relaxation}")
        }
        Rejection::Rcr { limit } => format!("rcr {limit}"),
    };
    format!("{combination} {rejection}")
}

/// The standard cards of a master, in the order Athenaeum's master builder writes its own, with `SLM_*` cards in place of
/// its provenance cards: the frame type, the creator, the acquisition facts the frames agree on, the CFA geometry, the
/// sub-frame origin, then the build's own cards (`SLM_N`, `SLM_REC`, the three common cards, the temperature span) and,
/// for a flat, the normalization constants. The two provenance cards that name catalog rows (the source set and the
/// member hash) have no counterpart.
fn master_cards(
    kind: MasterKind,
    c: &Consolidated,
    n: usize,
    recipe: &str,
    cfg_hash: &str,
    flat_norm: Option<f64>,
    channel_norms: Option<[f64; 3]>,
) -> Result<Vec<Card>> {
    let mut cards = vec![
        card("IMAGETYP", CardValue::Str(kind.imagetyp().to_string()), "type of image")?,
        card(
            "SWCREATE",
            CardValue::Str(format!("starloom {}", env!("CARGO_PKG_VERSION"))),
            "software that created this file",
        )?,
    ];
    if let Some(v) = c.exptime {
        cards.push(card("EXPTIME", CardValue::Real(v), "[s] exposure duration")?);
    }
    if let Some(t) = c.date_midpoint {
        cards.push(card(
            "DATE-OBS",
            CardValue::Str(t.format("%Y-%m-%dT%H:%M:%S%.3f").to_string()),
            "UTC observation start/midpoint",
        )?);
    }
    if let Some(t) = c.temp_mean {
        cards.push(card("CCD-TEMP", CardValue::Real(t), "[degC] sensor temperature")?);
    }
    if let Some(g) = c.gain {
        cards.push(card("GAIN", CardValue::Integer(g.round() as i64), "camera gain setting")?);
    }
    if let Some(o) = c.offset {
        cards.push(card("OFFSET", CardValue::Integer(o.round() as i64), "camera offset setting")?);
    }
    if let Some((x, y)) = c.xbinning.zip(c.ybinning) {
        cards.push(card("XBINNING", CardValue::Integer(x), "binning factor X")?);
        cards.push(card("YBINNING", CardValue::Integer(y), "binning factor Y")?);
    }
    if let Some((x, y)) = c.xpixsz.zip(c.ypixsz) {
        cards.push(card("XPIXSZ", CardValue::Real(x), "[um] pixel width after binning")?);
        cards.push(card("YPIXSZ", CardValue::Real(y), "[um] pixel height after binning")?);
    }
    if let Some(v) = &c.instrume {
        cards.push(card("INSTRUME", CardValue::Str(v.clone()), "camera")?);
    }
    if let Some(v) = &c.telescop {
        cards.push(card("TELESCOP", CardValue::Str(v.clone()), "telescope")?);
    }
    if let Some(v) = c.focallen {
        cards.push(card("FOCALLEN", CardValue::Real(v), "[mm] focal length")?);
    }
    if let Some(v) = &c.filter {
        cards.push(card("FILTER", CardValue::Str(v.clone()), "filter name")?);
    }
    if let Some(p) = &c.bayerpat {
        match Bayer::parse(p) {
            Some(b) => {
                cards.push(card("BAYERPAT", CardValue::Str(b.as_str().to_string()), "Bayer color pattern")?);
                // The offsets are the CFA phase of that pattern, and only when the frames declared one: a missing
                // offset stays missing, because XBAYROFF = 0 is a claim about the mosaic and a wrong one swaps R and B.
                for (keyword, value, comment) in
                    [("XBAYROFF", c.xbayroff, "Bayer X offset"), ("YBAYROFF", c.ybayroff, "Bayer Y offset")]
                {
                    if let Some(v) = value {
                        if !(0..=1).contains(&v) {
                            tracing::warn!(field = keyword, value = v, "bayer offset outside 0/1; emitting verbatim");
                        }
                        cards.push(card(keyword, CardValue::Integer(v), comment)?);
                    }
                }
            }
            None => tracing::warn!(bayerpat = %p, "unrecognized bayer pattern; bayer cards omitted from the master"),
        }
    }
    if let Some(r) = &c.roworder {
        match r.trim_matches('\'').trim() {
            v @ ("TOP-DOWN" | "BOTTOM-UP") => cards.push(card("ROWORDER", CardValue::Str(v.to_string()), "image row order")?),
            _ => tracing::warn!(roworder = %r, "unrecognized row order; ROWORDER omitted from the master"),
        }
    }
    if let Some((x, y)) = c.origin {
        cards.push(card("XORGSUBF", CardValue::Integer(x), "Subframe X position")?);
        cards.push(card("YORGSUBF", CardValue::Integer(y), "Subframe Y position")?);
    }
    cards.push(card(SLM_N, CardValue::Integer(n as i64), "number of integrated frames")?);
    cards.push(card(SLM_REC, CardValue::Str(recipe.to_string()), "integration recipe")?);
    cards.extend(common_cards(cfg_hash)?);
    if let Some((min, max)) = c.temp_span {
        cards.push(card(SLM_TMIN, CardValue::Real(min), "[degC] min member CCD-TEMP")?);
        cards.push(card(SLM_TMAX, CardValue::Real(max), "[degC] max member CCD-TEMP")?);
    }
    if let Some(norm) = flat_norm {
        cards.push(card(SLM_FNRM, CardValue::Real(norm), "central-third mean of this master flat")?);
        // The per-channel constants ride with the global one: same window, same units.
        if let Some(ch) = channel_norms {
            for (keyword, v) in [(SLM_FNR, ch[0]), (SLM_FNG, ch[1]), (SLM_FNB, ch[2])] {
                cards.push(card(keyword, CardValue::Real(v), "per-channel flat normalization constant")?);
            }
        }
    }
    Ok(cards)
}

/// The per-channel constants of a freshly integrated flat: the central-third means of the plane under the frames'
/// CFA geometry (a missing phase reads as 0, as the build assumes), `None` for a mono flat or degenerate constants,
/// which are divisors and so worse than absent.
fn measure_channel_norms(c: &Consolidated, data: &[f32], width: usize, height: usize) -> Option<[f64; 3]> {
    let pattern = Bayer::parse(c.bayerpat.as_deref()?)?;
    let geom = CfaGeometry { pattern, xoff: c.xbayroff.unwrap_or(0), yoff: c.ybayroff.unwrap_or(0) };
    match flat_channel_norms(data, width, height, geom) {
        Ok(means) => Some(means),
        Err(means) => {
            tracing::warn!(value = ?means, "per-channel flat constants degenerate; omitted");
            None
        }
    }
}

/// Reads the whole plane of a pre-calibration master into RAM, through the same reader the integration engine uses.
fn load_precal_master(path: &Path, scratch: &Path) -> Result<FlatPrecal> {
    let src = BandSource::open(&[path.to_path_buf()], scratch, 1)
        .map_err(|e| anyhow::anyhow!("pre-calibration master {} is unreadable: {e}", path.display()))?;
    let (width, height) = (src.width(), src.height());
    let mut planes = BandPlanes::new(&src);
    src.read_band(0, height, &mut planes, 1).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let mut data = vec![0f32; width * height];
    planes.decode_frame_into(0, &mut data);
    Ok(FlatPrecal::MasterFrame { data, width, height })
}

/// Checks the frames of a master from their headers: at least [`MIN_MASTER_FRAMES`], one plane each and one size.
/// Returns their facts, in order.
fn check_frames(frames: &[PathBuf]) -> Result<Vec<FrameFacts>> {
    if frames.len() < MIN_MASTER_FRAMES {
        bail!("{} frame(s) given; at least {MIN_MASTER_FRAMES} are required to build a master", frames.len());
    }
    let mut facts = Vec::with_capacity(frames.len());
    let mut first: Option<(&Path, (i64, i64))> = None;
    for path in frames {
        let f = frame_facts(path)?;
        if f.planes.is_some_and(|p| p > 1) {
            bail!("{}", planes_refusal(path, f.planes.unwrap_or(0)));
        }
        if let Some(size) = size_of(&f) {
            match first {
                None => first = Some((path, size)),
                Some((other, other_size)) if other_size != size => bail!(
                    "frames of different sizes: {} is {}×{}, {} is {}×{}",
                    other.display(),
                    other_size.0,
                    other_size.1,
                    path.display(),
                    size.0,
                    size.1
                ),
                Some(_) => {}
            }
        }
        facts.push(f);
    }
    Ok(facts)
}

/// Builds one master calibration frame from raw `frames` and writes it to `out` (atomically, in `format`).
///
/// `precal` is how a flat's frames are pre-calibrated before they are integrated (a master of the flats' size, or a
/// constant in DN); it is refused for a bias or a dark. `recipe` `None` takes the automatic one for the kind and the
/// frame count ([`resolve_recipe`]). `cfg` is the effective config whose hash the master's `SLM_CFG` card records.
///
/// Refused before any pixel is read: fewer than [`MIN_MASTER_FRAMES`] frames, a frame with more than one plane, frames
/// of different sizes and a pre-calibration master of another size, each naming the files. The raw frames may be
/// compressed; the master is always plain DN.
///
/// A master leaves no manifest entry: the caller decides where it goes and what to record. The scratch folder is
/// `<work>/.tmp/`.
pub fn build_master(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    kind: MasterKind,
    frames: &[PathBuf],
    precal: Option<&FlatPrecalSource>,
    recipe: Option<IntegrationRecipe>,
    out: &Path,
    format: OutputFormat,
) -> Result<MasterReport> {
    if precal.is_some() && kind != MasterKind::Flat {
        bail!("a pre-calibration applies to flats only, not to {} frames", kind.name());
    }
    let facts = check_frames(frames)?;
    if let (Some(FlatPrecalSource::Master(p)), Some(size)) = (precal, facts.iter().find_map(size_of)) {
        let pf = frame_facts(p)?;
        if pf.planes.is_some_and(|n| n > 1) {
            bail!("{}", planes_refusal(p, pf.planes.unwrap_or(0)));
        }
        if let Some(psize) = size_of(&pf).filter(|s| *s != size) {
            bail!(
                "pre-calibration master {} is {}×{}, the flats are {}×{}",
                p.display(),
                psize.0,
                psize.1,
                size.0,
                size.1
            );
        }
    }
    ctx.check_cancelled()?;

    let scratch = scratch_dir(ctx)?;
    let io = io_policy(ctx);
    let recipe = resolve_recipe(recipe, kind.frame_type(), frames.len() as i64);
    let recipe_text = recipe_text(&recipe);
    tracing::info!(kind = kind.name(), frames = frames.len(), recipe = %recipe_text, "master build started");

    let on_band = |_band: usize, _bands: usize, done: u64, total: u64| {
        ctx.progress.step("master", usize::try_from(done).unwrap_or(usize::MAX), usize::try_from(total).unwrap_or(usize::MAX));
    };
    let progress = EngineProgress { on_band: &on_band, on_combine: &|_, _, _, _| {} };
    let integrated = if kind == MasterKind::Flat {
        let precal = match precal {
            Some(FlatPrecalSource::Master(p)) => load_precal_master(p, &scratch)?,
            Some(FlatPrecalSource::SyntheticBias(b)) => FlatPrecal::SyntheticBias(*b),
            None => FlatPrecal::None,
        };
        integrate_flat(frames, &precal, recipe, &ctx.pool, &scratch, ctx.cancel, progress, io)
    } else {
        integrate_bias_like(frames, recipe, &ctx.pool, &scratch, ctx.cancel, progress, io)
    }
    .map_err(|e| anyhow::anyhow!("integrating the {} frames failed: {e}", kind.name()))?;
    ctx.check_cancelled()?;

    // Undefined (non-finite) samples are dropped from the stack, never a failure; the person hears about the loss.
    let mut warnings = Vec::new();
    let bad_total: usize = integrated.bad_samples_per_frame.iter().sum();
    if bad_total > 0 || integrated.all_bad_pixels > 0 {
        for (path, &count) in frames.iter().zip(&integrated.bad_samples_per_frame) {
            if count > 0 {
                tracing::warn!(path = %path.display(), count, "non-finite samples excluded from integration");
            }
        }
        let touched = integrated.bad_samples_per_frame.iter().filter(|&&c| c > 0).count();
        let mut message = format!("{bad_total} undefined sample(s) excluded across {touched} frame(s)");
        if integrated.all_bad_pixels > 0 {
            message.push_str(&format!("; {} pixel(s) had no valid data", integrated.all_bad_pixels));
        }
        warnings.push(message);
    }

    let consolidated = Consolidated::from_facts(&facts);
    let channel_norms = if integrated.flat_norm.is_some() {
        measure_channel_norms(&consolidated, &integrated.data, integrated.width, integrated.height)
    } else {
        None
    };
    let cards = master_cards(
        kind,
        &consolidated,
        frames.len(),
        &recipe_text,
        &config_hash(cfg),
        integrated.flat_norm,
        channel_norms,
    )?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    write_image_f32(out, integrated.width, integrated.height, 1, &integrated.data, &cards, format, XisfBounds::Adu16)
        .with_context(|| format!("cannot write {}", out.display()))?;
    for w in &warnings {
        ctx.progress.warn(w);
    }
    tracing::info!(path = %out.display(), kind = kind.name(), "master written");
    Ok(MasterReport {
        path: out.to_path_buf(),
        kind,
        frames: frames.len(),
        recipe,
        recipe_text,
        width: integrated.width,
        height: integrated.height,
        rejected_fraction: integrated.rejected_fraction,
        flat_norm: integrated.flat_norm,
        channel_norms,
        warnings,
    })
}
