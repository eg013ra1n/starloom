//! The `calibrate` stage: raw lights to calibrated frames, and the checks that refuse a mismatch before any pixel is
//! read.
//!
//! This mirrors Athenaeum's calibrated-light generation (`resolve_generation_mapped` and `execute_generation`) without
//! the catalog. For each group: a dark, flat or bias given as raw frames is first built into a master under
//! `<work>/masters/` ([`build_master`]); then each light is calibrated (dark, else bias; then flat), its hot pixels are
//! replaced by the master dark's map, an OSC light is debayered, and the frame is written beside the others in the
//! output folder as plain FITS (or XISF when asked). The lights and the calibration frames may be compressed
//! (P2-R22).
//!
//! The header of a calibrated light is the light's copy-through cards ([`COPY_THROUGH_KEYWORDS`]), `IMAGETYP`,
//! `CALSTAT` and the `SLM_*` cards of P2-R6; no card of Athenaeum's own is read or written (SL-8).
//!
//! [`preflight`] is the whole of the checking, from headers alone (SL-20): a size mismatch and a frame with more than
//! one plane are refused, another sub-frame origin and a flat of another CFA phase are warned about. [`calibrate`]
//! runs it first, so a mismatch fails before the first pixel and leaves nothing in the output folder.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::{header_key_map, header_text, FrameFacts, HeaderFormat};
use rayon::prelude::*;

use crate::calibration_library::cosmetic::{apply_hot_pixel_correction, hot_pixel_map_from_dark, HotPixelMapOutcome};
use crate::calibration_library::debayer::{compute_calstat, debayer_plane, drop_mosaic_cards};
use crate::calibration_library::flat_cards::{SlmFlatCards, SLM_CSCL};
use crate::calibration_library::light_cal::{
    calibrate_light_compute_with_cards, preload_master_plane, scale_divisor_for_bitpix, BiasFallback, LightCalInputs,
    PreloadedMasters,
};
use crate::calibration_library::options::CalibratedLightOptions;
use crate::calibration_library::COPY_THROUGH_KEYWORDS;
use crate::fits_writer::{write_image_f32_with, Card, CardValue, Durability, OutputFormat, XisfBounds};
use crate::integration::banded::probe_bitpix;
use crate::integration::cfa::CfaGeometry;
use crate::pipeline::cards::{common_cards, version_card, SLM_CDBM, SLM_CHPX};
use crate::pipeline::config::config_hash;
use crate::pipeline::formats::{commit_stage, part_path, Entry};
use crate::pipeline::inputs::{frame_stem, order_lights, CalGroup, CalSource, FlatPrecalSource};
use crate::pipeline::master::{
    build_master, cfa_of, frame_facts, planes_refusal, scratch_dir, size_of, MasterKind, MasterReport,
};
use crate::pipeline::{Ctx, Stage};
use crate::stacking::config::AlgorithmConfig;

/// What one calibrated frame came from and what was done to it.
#[derive(Debug, Clone)]
pub struct CalibratedFrameReport {
    /// The raw light.
    pub source: PathBuf,
    /// The calibrated frame (absolute).
    pub output: PathBuf,
    /// What was applied: `BDF`-style flags (`BD`, `B`, `F`, ...).
    pub calstat: String,
    /// Whether the frame was debayered (its name then ends in `_d`).
    pub debayered: bool,
    /// How many hot pixels the cosmetic pass replaced; `None` when the pass did not run.
    pub hot_pixels_replaced: Option<u64>,
    /// The per-channel flat constants applied, when the flat was normalized per CFA channel.
    pub flat_channel_divisors: Option<[f64; 3]>,
    /// How many pixels were at or below zero right after the dark or bias was subtracted (a counter only).
    pub non_positive_pixels: u64,
    /// The pixels that counter looked at; 0 when no dark or bias was subtracted.
    pub pixels: u64,
}

/// What a `calibrate` run did.
#[derive(Debug, Clone, Default)]
pub struct CalibrateReport {
    /// The calibrated frames, in the order every later stage keeps (P2-R10: `DATE-OBS`, then input position).
    pub frames: Vec<CalibratedFrameReport>,
    /// The masters built from raw calibration frames, under `<work>/masters/`.
    pub masters: Vec<MasterReport>,
    /// Every warning the run raised (also sent to the context's progress sink): the preflight's, a refused hot-pixel
    /// map, a build's.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------------------------------------------

/// The facts of every file preflight read, and its warnings.
struct Checked {
    warnings: Vec<String>,
    facts: BTreeMap<PathBuf, FrameFacts>,
}

fn facts_of<'a>(cache: &'a mut BTreeMap<PathBuf, FrameFacts>, path: &Path) -> Result<&'a FrameFacts> {
    if !cache.contains_key(path) {
        cache.insert(path.to_path_buf(), frame_facts(path)?);
    }
    Ok(&cache[path])
}

/// A geometry as Athenaeum's messages spell it: `6384×4258`, with ` @ x,y` when the frame records an origin.
fn label(facts: &FrameFacts) -> String {
    let size = match size_of(facts) {
        Some((w, h)) => format!("{w}×{h}"),
        None => "size not recorded".to_string(),
    };
    match facts.origin {
        Some((x, y)) => format!("{size} @ {x},{y}"),
        None => size,
    }
}

fn cfa_label(cfa: Option<CfaGeometry>) -> String {
    match cfa {
        Some(g) => format!("{} @ {},{}", g.pattern.as_str(), g.xoff, g.yoff),
        None => "no CFA pattern".to_string(),
    }
}

fn same_cfa(a: Option<CfaGeometry>, b: Option<CfaGeometry>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.same_phase(b),
        _ => false,
    }
}

/// The files a calibration source reads.
fn source_paths(source: &CalSource) -> Vec<&Path> {
    match source {
        CalSource::Master(p) => vec![p.as_path()],
        CalSource::Raw(frames) => frames.iter().map(PathBuf::as_path).collect(),
    }
}

fn check(groups: &[CalGroup]) -> Result<Checked> {
    let mut cache: BTreeMap<PathBuf, FrameFacts> = BTreeMap::new();
    let mut warnings = Vec::new();
    if groups.is_empty() {
        bail!("no groups to calibrate");
    }
    for (gi, group) in groups.iter().enumerate() {
        let n = gi + 1;
        if group.lights.is_empty() {
            bail!("group {n} has no lights");
        }
        if group.dark.is_none() && group.flat.is_none() && group.bias.is_none() {
            bail!("group {n} has no calibration frames (give a dark, a flat or a bias)");
        }
        let mut lights: Vec<(&Path, FrameFacts)> = Vec::with_capacity(group.lights.len());
        for light in &group.lights {
            let facts = facts_of(&mut cache, light)?.clone();
            if let Some(planes) = facts.planes.filter(|p| *p > 1) {
                bail!("{}", planes_refusal(light, planes));
            }
            lights.push((light.as_path(), facts));
        }
        // The bias is read only when there is no dark (the raw-master-dark convention), so only then is it checked.
        let bias = if group.dark.is_some() { None } else { group.bias.as_ref() };
        for (role, source) in [("dark", group.dark.as_ref()), ("flat", group.flat.as_ref()), ("bias", bias)] {
            let Some(source) = source else { continue };
            let mut origin_warned = false;
            let mut cfa_warned = false;
            for path in source_paths(source) {
                let cal = facts_of(&mut cache, path)?.clone();
                if let Some(planes) = cal.planes.filter(|p| *p > 1) {
                    bail!("{}", planes_refusal(path, planes));
                }
                for (light_path, light) in &lights {
                    if let (Some(a), Some(b)) = (size_of(&cal), size_of(light)) {
                        if a != b {
                            bail!(
                                "Calibration does not fit the frame size — {role} {} is {}×{}, light {} is {}×{}",
                                path.display(),
                                a.0,
                                a.1,
                                light_path.display(),
                                b.0,
                                b.1
                            );
                        }
                        if !origin_warned && cal.origin.zip(light.origin).is_some_and(|(c, l)| c != l) {
                            origin_warned = true;
                            warnings.push(format!(
                                "Calibration at another sub-frame origin — {role} {} is {}, light {} is {}",
                                path.display(),
                                label(&cal),
                                light_path.display(),
                                label(light)
                            ));
                        }
                    }
                    if role == "flat" && !cfa_warned && !same_cfa(cfa_of(&cal), cfa_of(light)) {
                        cfa_warned = true;
                        warnings.push(format!(
                            "Flat {} has another CFA pattern or phase than light {} ({} against {}); \
                             its per-channel constants are measured again from its pixels",
                            path.display(),
                            light_path.display(),
                            cfa_label(cfa_of(&cal)),
                            cfa_label(cfa_of(light))
                        ));
                    }
                }
            }
        }
        // A flat pre-calibration master is subtracted from raw flats, so it must be their size.
        if let (Some(FlatPrecalSource::Master(p)), Some(CalSource::Raw(flats))) = (&group.flat_precal, &group.flat) {
            let pre = facts_of(&mut cache, p)?.clone();
            if let Some(planes) = pre.planes.filter(|n| *n > 1) {
                bail!("{}", planes_refusal(p, planes));
            }
            for flat in flats {
                let f = facts_of(&mut cache, flat)?;
                if let Some((a, b)) = size_of(&pre).zip(size_of(f)).filter(|(a, b)| a != b) {
                    bail!(
                        "Calibration does not fit the frame size — flat pre-calibration {} is {}×{}, flat {} is {}×{}",
                        p.display(),
                        a.0,
                        a.1,
                        flat.display(),
                        b.0,
                        b.1
                    );
                }
            }
        }
    }
    Ok(Checked { warnings, facts: cache })
}

/// Checks every group's lights and calibration frames against each other from their headers alone, before any pixel is
/// read (SL-20), and returns the warnings.
///
/// Refused, naming the files:
/// - a calibration frame whose size differs from a light's: "Calibration does not fit the frame size — dark `file` is
///   W×H, light `file` is W×H";
/// - a light or a calibration frame with more than one plane (the reason the integration reader gives);
/// - a group with no calibration frame at all, and a flat pre-calibration master of another size than its flats.
///
/// Warned about:
/// - another sub-frame origin: "Calibration at another sub-frame origin — ...";
/// - a flat whose CFA pattern or phase differs from the light's (its per-channel constants are then measured again
///   from its pixels).
///
/// Exposure, temperature, filter and gain are not checked. Only the bias a light is calibrated with (none when there
/// is a dark) is checked. A compressed frame is read for its logical header, so its size is the image's; a gzip file
/// is inflated to find it, which takes seconds per frame (P2-R22), and a truncated one fails here, naming it.
pub fn preflight(groups: &[CalGroup]) -> Result<Vec<String>> {
    check(groups).map(|c| c.warnings)
}

// ---------------------------------------------------------------------------------------------------------------
// The cards of a calibrated light
// ---------------------------------------------------------------------------------------------------------------

/// A typed card from a `KEYWORD -> value` pair: an integer, else a real, else a string (a keyword the writer rejects
/// is dropped), as the copied-through cards keep their type.
fn card_from_kv(keyword: &str, value: &str) -> Option<Card> {
    let cv = if let Ok(i) = value.parse::<i64>() {
        CardValue::Integer(i)
    } else if let Ok(f) = value.parse::<f64>() {
        CardValue::Real(f)
    } else {
        CardValue::Str(value.to_string())
    };
    Card::new(keyword, cv).ok()
}

/// The light's cards that survive calibration: [`COPY_THROUGH_KEYWORDS`] read from its header, then the four CFA cards
/// a header can lack while the facts hold them (an XISF that declares its CFA only through its
/// `<ColorFilterArray>` element), sorted by keyword so the order is the same on every run.
fn source_cards(light: &Path, facts: &FrameFacts) -> Result<Vec<Card>> {
    let (format, text) = header_text(light).with_context(|| format!("cannot read the header of {}", light.display()))?;
    let map = header_key_map(format, &text).into_legacy_map();
    let mut cards: Vec<Card> = COPY_THROUGH_KEYWORDS
        .iter()
        .filter_map(|keyword| map.get(*keyword).and_then(|v| card_from_kv(keyword, v)))
        .collect();
    let trimmed = |v: &Option<String>| v.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let derived = [
        trimmed(&facts.bayerpat).map(|v| ("BAYERPAT", CardValue::Str(v))),
        facts.xbayroff.map(|v| ("XBAYROFF", CardValue::Integer(v))),
        facts.ybayroff.map(|v| ("YBAYROFF", CardValue::Integer(v))),
        trimmed(&facts.roworder).map(|v| ("ROWORDER", CardValue::Str(v))),
    ];
    for (keyword, value) in derived.into_iter().flatten() {
        if !cards.iter().any(|c| c.keyword == keyword) {
            cards.push(Card::new(keyword, value)?);
        }
    }
    cards.sort_by(|a, b| a.keyword.cmp(&b.keyword));
    Ok(cards)
}

// ---------------------------------------------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------------------------------------------

/// One light of a group, resolved.
struct Job<'a> {
    light: &'a Path,
    facts: &'a FrameFacts,
    cfa: Option<CfaGeometry>,
    debayer: bool,
    part: PathBuf,
    dest: PathBuf,
}

/// The masters a group's lights are calibrated with, as files.
struct Masters {
    dark: Option<PathBuf>,
    bias: Option<PathBuf>,
    flat: Option<PathBuf>,
}

/// What one calibrated frame left behind.
struct Done {
    report: CalibratedFrameReport,
    part: PathBuf,
    dest: PathBuf,
}

/// A calibration source as one master file: a master is used as it is, raw frames are built into `masters_dir`.
#[allow(clippy::too_many_arguments)]
fn resolve_master(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    source: &CalSource,
    kind: MasterKind,
    precal: Option<&FlatPrecalSource>,
    masters_dir: &Path,
    group_number: usize,
    built: &mut Vec<MasterReport>,
) -> Result<PathBuf> {
    match source {
        CalSource::Master(path) => Ok(path.clone()),
        CalSource::Raw(frames) => {
            let out = masters_dir.join(format!("group-{group_number:02}-{}.fits", kind.name()));
            let report = build_master(ctx, cfg, kind, frames, precal, None, &out, OutputFormat::Fits)
                .with_context(|| format!("building the {} master of group {group_number}", kind.name()))?;
            built.push(report);
            Ok(out)
        }
    }
}

/// Calibrates one light and writes it to its part file.
fn calibrate_one(
    ctx: &Ctx<'_>,
    cfg_hash: &str,
    job: &Job<'_>,
    masters: &Masters,
    opts: &CalibratedLightOptions,
    format: OutputFormat,
    scratch: &Path,
    preloaded: &PreloadedMasters,
    hot_maps: &HashMap<PathBuf, Arc<HotPixelMapOutcome>>,
) -> Result<Done> {
    let name = |e: &dyn std::fmt::Display| anyhow!("{}: {e}", job.light.display());
    // The scale divisor of the light's own bit depth. A compressed frame's depth is its logical one, from its header.
    let bitpix = probe_bitpix(job.light)
        .or_else(|| (job.facts.format == HeaderFormat::Fits).then_some(job.facts.bitpix).flatten().and_then(|b| i32::try_from(b).ok()));
    let scale_divisor = scale_divisor_for_bitpix(bitpix);

    let dark_applied = masters.dark.is_some();
    let bias_applied = !dark_applied && masters.bias.is_some();
    let calstat = compute_calstat(dark_applied, bias_applied, masters.flat.is_some());

    let inputs = LightCalInputs {
        light_path: job.light.to_path_buf(),
        dark_path: masters.dark.clone(),
        // The bias is read only when there is no dark.
        bias_path: if dark_applied { None } else { masters.bias.clone() },
        flat_path: masters.flat.clone(),
        flat_norm: opts.flat_norm,
        flat_norm_mode: opts.flat_norm_mode,
        cfa_geometry: job.cfa,
        params: opts.params,
        scale_divisor,
        output_path: PathBuf::new(),
        cards: Vec::new(),
        scratch_dir: scratch.to_path_buf(),
    };
    let (mut frame, outcome) =
        calibrate_light_compute_with_cards(&inputs, Some(preloaded), &SlmFlatCards, ctx.cancel).map_err(|e| name(&e))?;

    // The cosmetic pass: only a master dark can say which pixels are defective.
    let mut hot_pixels_replaced = None;
    if let (true, Some(dark)) = (opts.hot_pixel_correction, &masters.dark) {
        if let Some(outcome) = hot_maps.get(dark) {
            if let HotPixelMapOutcome::Map(map) = &**outcome {
                hot_pixels_replaced = Some(apply_hot_pixel_correction(&mut frame.data, frame.width, frame.height, map, job.cfa));
            }
        }
    }
    ctx.check_cancelled()?;

    let (data, channels) = match (job.debayer, job.cfa) {
        (true, Some(geom)) => (debayer_plane(&frame.data, frame.width, frame.height, geom, Some(&ctx.pool)), 3usize),
        _ => (std::mem::take(&mut frame.data), 1usize),
    };

    let mut cards = vec![
        Card::new("IMAGETYP", CardValue::Str("Light Frame".to_string()))?.with_comment("type of image"),
        Card::new("CALSTAT", CardValue::Str(calstat.clone()))?.with_comment("calibration state (B/D/F)"),
    ];
    cards.extend(source_cards(job.light, job.facts)?);
    cards.extend(common_cards(cfg_hash)?);
    cards.push(Card::new(SLM_CSCL, CardValue::Real(scale_divisor))?.with_comment("numeric-scale divisor applied"));
    cards.push(version_card("SLM_LCV")?);
    if channels == 3 {
        drop_mosaic_cards(&mut cards);
        cards.push(Card::new(SLM_CDBM, CardValue::Str("VNG".to_string()))?);
    }
    if let Some(n) = hot_pixels_replaced {
        cards.push(Card::new(SLM_CHPX, CardValue::Integer(n as i64))?.with_comment("hot pixels replaced"));
    }

    let durability = if opts.skip_fsync { Durability::Volatile } else { Durability::Durable };
    write_image_f32_with(&job.part, frame.width, frame.height, channels, &data, &cards, format, XisfBounds::Unit, durability)
        .map_err(|e| anyhow!("writing {}: {e}", job.dest.display()))?;
    tracing::debug!(
        src = %job.light.display(),
        dest = %job.dest.display(),
        calstat = %calstat,
        debayered = channels == 3,
        flat_norm_divisor = outcome.flat_norm_divisor,
        "light calibrated"
    );
    Ok(Done {
        report: CalibratedFrameReport {
            source: job.light.to_path_buf(),
            output: job.dest.clone(),
            calstat,
            debayered: channels == 3,
            hot_pixels_replaced,
            flat_channel_divisors: outcome.flat_channel_divisors,
            non_positive_pixels: outcome.non_positive_pixels,
            pixels: outcome.pixels,
        },
        part: job.part.clone(),
        dest: job.dest.clone(),
    })
}

/// Deletes the part files of a stage that failed (nothing named like a finished product is left).
fn remove_parts(parts: impl IntoIterator<Item = PathBuf>) {
    for part in parts {
        if let Err(e) = std::fs::remove_file(&part) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %part.display(), error = %e, "cannot remove a part file");
            }
        }
    }
}

/// Calibrates the lights of `groups` into `out_dir`, one `c_<name>.fits` per light (`c_<name>_d.fits` when debayered;
/// `.xisf` for [`OutputFormat::Xisf`]), and records the stage in the manifest of the working folder.
///
/// `opts` says how (flat normalization, hot-pixel pass, debayer: normally `cfg.calibration`); `cfg` is the effective
/// config whose hash every product and the manifest entry record. Raw calibration frames are first built into masters
/// under `<work>/masters/` ([`build_master`], automatic recipe, plain FITS).
///
/// [`preflight`] runs first and its warnings go to the context's progress sink and the report; a refusal fails the
/// run before the output folder is created. The products are built beside their destinations and moved into place
/// together with the entry ([`commit_stage`]), so a failure (a truncated gzip light, a cancel) leaves no product and no
/// entry. The entry's products are the calibrated frames in the frame order of P2-R10 (`DATE-OBS`, then input
/// position, across all groups), and its `sources` hold each raw light's facts, keyed by the product.
///
/// Lights are calibrated on the context's pool, as many at once as its memory budget allows. The command clears the
/// scratch folder (`<work>/.tmp/`) when it starts ([`begin_writing`](crate::pipeline::begin_writing)). `keep_mosaic` is
/// not honoured here: no mosaic is written beside a debayered frame.
pub fn calibrate(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    groups: &[CalGroup],
    opts: &CalibratedLightOptions,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<CalibrateReport> {
    let checked = check(groups)?;
    let mut report = CalibrateReport { warnings: checked.warnings.clone(), ..CalibrateReport::default() };
    for w in &checked.warnings {
        ctx.progress.warn(w);
    }
    for (gi, g) in groups.iter().enumerate() {
        if g.dark.is_none() && g.bias.is_some() && opts.params.bias_fallback == BiasFallback::SkipFrame {
            bail!("group {}: no dark master (bias fallback disabled)", gi + 1);
        }
    }
    let out_dir = std::path::absolute(out_dir).with_context(|| format!("cannot make {} absolute", out_dir.display()))?;
    ctx.check_cancelled()?;

    // The frame order every stage keeps, across all groups; and the names the products will have.
    let all: Vec<(usize, &PathBuf)> = groups.iter().enumerate().flat_map(|(gi, g)| g.lights.iter().map(move |l| (gi, l))).collect();
    let all_paths: Vec<PathBuf> = all.iter().map(|(_, l)| (*l).clone()).collect();
    let all_facts: Vec<FrameFacts> = all.iter().map(|(_, l)| checked.facts[*l].clone()).collect();
    let order = order_lights(&all_paths, &all_facts)?;
    let ext = format.extension();
    let mut dests: Vec<PathBuf> = Vec::with_capacity(all.len());
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (i, light) in all_paths.iter().enumerate() {
        let debayer = opts.debayer_osc && cfa_of(&all_facts[i]).is_some();
        let file = format!("c_{}{}.{ext}", frame_stem(light), if debayer { "_d" } else { "" });
        if let Some(first) = seen.insert(file.to_lowercase(), i) {
            bail!(
                "{} and {} would both become {file}; give lights with distinct names",
                all_paths[first].display(),
                light.display()
            );
        }
        dests.push(out_dir.join(file));
    }

    std::fs::create_dir_all(&out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;
    let scratch = scratch_dir(ctx)?;
    let masters_dir = ctx.work_dir.join("masters");
    let cfg_hash = config_hash(cfg);
    let total = all.len();
    let done_count = AtomicUsize::new(0);
    let threads = ctx.pool.current_num_threads().max(1);
    let budget = usize::try_from(ctx.memory_budget_bytes).unwrap_or(usize::MAX);

    let mut finished: Vec<(usize, Done)> = Vec::with_capacity(total);
    let run = |finished: &mut Vec<(usize, Done)>, report: &mut CalibrateReport| -> Result<()> {
        let mut index = 0usize;
        for (gi, group) in groups.iter().enumerate() {
            let n = gi + 1;
            let first_index = index;
            index += group.lights.len();
            let mut masters = Masters { dark: None, bias: None, flat: None };
            if let Some(source) = &group.dark {
                masters.dark = Some(resolve_master(ctx, cfg, source, MasterKind::Dark, None, &masters_dir, n, &mut report.masters)?);
            } else if let Some(source) = &group.bias {
                masters.bias = Some(resolve_master(ctx, cfg, source, MasterKind::Bias, None, &masters_dir, n, &mut report.masters)?);
            }
            if let Some(source) = &group.flat {
                let precal = group.flat_precal.as_ref();
                masters.flat = Some(resolve_master(ctx, cfg, source, MasterKind::Flat, precal, &masters_dir, n, &mut report.masters)?);
            }
            ctx.check_cancelled()?;

            // The master dark's hot pixels, measured once for the group; only the dark can say which pixels are defective.
            let mut hot_maps: HashMap<PathBuf, Arc<HotPixelMapOutcome>> = HashMap::new();
            if let (true, Some(dark)) = (opts.hot_pixel_correction, &masters.dark) {
                let measured = hot_pixel_map_from_dark(dark, &scratch).unwrap_or_else(|e| {
                    tracing::warn!(path = %dark.display(), error = %format!("{e:#}"), "measuring the master dark failed; hot-pixel correction refused");
                    HotPixelMapOutcome::Refused(format!("measuring it failed: {e:#}"))
                });
                if let HotPixelMapOutcome::Refused(reason) = &measured {
                    let message = format!("Hot-pixel correction skipped for {}: {reason}", dark.display());
                    ctx.progress.warn(&message);
                    report.warnings.push(message);
                }
                hot_maps.insert(dark.clone(), Arc::new(measured));
            }

            // The group's masters decoded once, shared by every light (a failure is not fatal: the engine reads the
            // file itself).
            let mut preloaded = PreloadedMasters { planes: HashMap::new() };
            let subtrahend = masters.dark.as_ref().or(masters.bias.as_ref());
            for path in subtrahend.into_iter().chain(masters.flat.as_ref()) {
                match preload_master_plane(path, &scratch, ctx.cancel) {
                    Ok(plane) => {
                        preloaded.planes.insert(path.clone(), plane);
                    }
                    Err(e) => tracing::warn!(path = %path.display(), error = %e, "master preload failed; the engine reads it per frame"),
                }
            }

            let jobs: Vec<Job<'_>> = (0..group.lights.len())
                .map(|k| {
                    let i = first_index + k;
                    let cfa = cfa_of(&all_facts[i]);
                    Job {
                        light: group.lights[k].as_path(),
                        facts: &all_facts[i],
                        cfa,
                        debayer: opts.debayer_osc && cfa.is_some(),
                        part: part_path(&dests[i]),
                        dest: dests[i].clone(),
                    }
                })
                .collect();

            // How many lights at once: what the budget holds after the resident masters, one frame's planes each.
            let px = all_facts[first_index..index].iter().filter_map(|f| size_of(f).map(|(w, h)| (w * h) as usize)).max().unwrap_or(0);
            let osc = jobs.iter().any(|j| j.debayer);
            let per_frame = px.saturating_mul(4).saturating_mul(if osc { 8 } else { 3 }).max(1);
            let resident = preloaded.planes.len().saturating_mul(px).saturating_mul(4);
            let concurrency = (budget.saturating_sub(resident) / per_frame).clamp(1, threads);

            for (chunk_no, chunk) in jobs.chunks(concurrency).enumerate() {
                ctx.check_cancelled()?;
                let results: Vec<Result<Done>> = ctx.pool.install(|| {
                    chunk
                        .par_iter()
                        .map(|job| {
                            let r = calibrate_one(ctx, &cfg_hash, job, &masters, opts, format, &scratch, &preloaded, &hot_maps);
                            ctx.progress.step("calibrate", done_count.fetch_add(1, Ordering::Relaxed) + 1, total);
                            r
                        })
                        .collect()
                });
                let mut first_err = None;
                for (k, r) in results.into_iter().enumerate() {
                    match r {
                        Ok(done) => finished.push((first_index + chunk_no * concurrency + k, done)),
                        Err(e) => {
                            first_err.get_or_insert(e);
                        }
                    }
                }
                if let Some(e) = first_err {
                    return Err(e);
                }
            }
        }
        Ok(())
    };
    if let Err(e) = run(&mut finished, &mut report) {
        remove_parts(finished.iter().map(|(_, d)| d.part.clone()).chain(dests.iter().map(|d| part_path(d))));
        return Err(e);
    }

    // The products in the order every later stage keeps.
    let mut by_index: Vec<Option<Done>> = (0..total).map(|_| None).collect();
    for (i, done) in finished {
        by_index[i] = Some(done);
    }
    let mut moves = Vec::with_capacity(total);
    let mut entry = Entry::new(
        Stage::Calibrate,
        cfg,
        order.iter().map(|&i| all_paths[i].clone()).collect(),
        order.iter().map(|&i| dests[i].clone()).collect(),
    );
    for &i in &order {
        let done = by_index[i].take().ok_or_else(|| anyhow!("{} was not calibrated", all_paths[i].display()))?;
        moves.push((done.part, done.dest.clone()));
        entry.sources.insert(done.dest, all_facts[i].clone());
        report.frames.push(done.report);
    }
    if let Err(e) = commit_stage(ctx.work_dir, &moves, entry) {
        remove_parts(moves.iter().map(|(part, _)| part.clone()));
        return Err(e);
    }
    tracing::info!(frames = total, masters = report.masters.len(), "calibrate stage written");
    Ok(report)
}
