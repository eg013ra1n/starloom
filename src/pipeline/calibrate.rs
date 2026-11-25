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
//! The header of a calibrated light is Athenaeum's (`build_light_cal_cards`, then the debayer and hot-pixel cards
//! `execute_generation` appends) with `SLM_*` cards in place of every `ATH_*` card but `ATH_CSRC`: `IMAGETYP`,
//! `CALSTAT`, the light's copy-through cards ([`COPY_THROUGH_KEYWORDS`]), the three common cards, then `SLM_CSRN`
//! (the light's file name), `SLM_CDRK` / `SLM_CFLT` / `SLM_CBIA` (the file name of the master applied, no catalog
//! id), `SLM_CSCL`, `SLM_CFNM`, `SLM_CPED`, `SLM_LCV`, the per-channel flat cards and `SLM_CTRM` when they apply, and
//! last `SLM_CDBM` and `SLM_CHPX`. No card of Athenaeum's own is read or written (SL-8).
//!
//! [`preflight`] is the whole of the checking, from headers alone (SL-20): a size mismatch, a frame with more than
//! one plane and a raw calibration set too small for a master are refused, another sub-frame origin and a flat of
//! another CFA phase are warned about, and a group whose lights cannot be calibrated at all (no master, or a bias
//! alone with the fallback off) has them excluded with Athenaeum's reason, as its run excludes them. [`calibrate`]
//! runs it first, so a mismatch fails before the first pixel and leaves nothing in the output folder. Every step that
//! decodes a pixel runs on the context's pool (SL-9).

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::{header_key_map, header_text, FrameFacts, HeaderFormat};
use rayon::prelude::*;

use crate::calibration_library::cosmetic::{apply_hot_pixel_correction, hot_pixel_map_from_dark, HotPixelMapOutcome};
use crate::calibration_library::debayer::{compute_calstat, debayer_plane, drop_mosaic_cards};
use crate::calibration_library::flat_cards::{has_compressed_name, SlmFlatCards, SLM_CSCL};
use crate::calibration_library::light_cal::{
    calibrate_light_compute_resolved, calibrate_light_compute_with_cards, preload_master_plane,
    resolve_flat_norm_divisor_with, scale_divisor_for_bitpix, BiasFallback, FlatNormDivisor, LightCalInputs,
    PreloadedMasters,
};
use crate::calibration_library::options::CalibratedLightOptions;
use crate::calibration_library::params::FlatNormMode;
use crate::calibration_library::COPY_THROUGH_KEYWORDS;
use crate::fits_writer::{
    write_fits_f32_with, write_image_f32_with, Card, CardValue, Durability, OutputFormat, XisfBounds,
};
use crate::integration::banded::probe_bitpix;
use crate::integration::cfa::CfaGeometry;
use crate::pipeline::cards::{
    common_cards, version_card, SLM_CBIA, SLM_CCFA, SLM_CDBM, SLM_CDRK, SLM_CFLT, SLM_CFNB, SLM_CFNG, SLM_CFNM, SLM_CFNR,
    SLM_CHPX, SLM_CPED, SLM_CSRN, SLM_CTRM,
};
use crate::pipeline::config::config_hash;
use crate::pipeline::formats::{commit_stage, part_path, Entry};
use crate::pipeline::inputs::{frame_stem, order_lights, CalGroup, CalSource, FlatPrecalSource};
use crate::pipeline::master::{
    build_master, cfa_of, frame_facts, planes_refusal, scratch_dir, size_of, too_few_frames, MasterKind, MasterReport,
    MIN_MASTER_FRAMES,
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
    /// Per group: why its lights are excluded from the run (`None`: they are calibrated).
    excluded: Vec<Option<&'static str>>,
}

/// The reason a light with no calibration frame at all is excluded (Athenaeum's `check_applicable`).
const NO_MASTERS: &str = "no calibration masters available (dark/flat/bias unbuilt or unlinked)";
/// The reason a light with a bias but no dark is excluded when the bias fallback is off (Athenaeum's `check_applicable`).
const NO_DARK_MASTER: &str = "no dark master (bias fallback disabled)";

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

/// Whether a flat is normalized per CFA channel for a light of this mosaic phase: the options ask for it, the light
/// is a mosaic and the statistic is the central third (the whole-frame trimmed mean has no per-channel form).
fn per_channel_applies(opts: &CalibratedLightOptions, light: Option<CfaGeometry>) -> bool {
    opts.flat_norm && opts.params.cfa_flat_scaling && opts.flat_norm_mode != FlatNormMode::TrimmedMeanWhole && light.is_some()
}

fn check(groups: &[CalGroup], opts: &CalibratedLightOptions) -> Result<Checked> {
    let mut cache: BTreeMap<PathBuf, FrameFacts> = BTreeMap::new();
    let mut warnings = Vec::new();
    let mut excluded: Vec<Option<&'static str>> = Vec::with_capacity(groups.len());
    if groups.is_empty() {
        bail!("no groups to calibrate");
    }
    for (gi, group) in groups.iter().enumerate() {
        let n = gi + 1;
        if group.lights.is_empty() {
            bail!("group {n} has no lights");
        }
        // Athenaeum's run excludes these lights (one reason each) and calibrates the rest, so nothing here is read.
        let reason = if group.dark.is_none() && group.flat.is_none() && group.bias.is_none() {
            Some(NO_MASTERS)
        } else if group.dark.is_none() && group.bias.is_some() && opts.params.bias_fallback == BiasFallback::SkipFrame {
            Some(NO_DARK_MASTER)
        } else {
            None
        };
        excluded.push(reason);
        if let Some(reason) = reason {
            for light in &group.lights {
                warnings.push(format!("{}: {reason}", light.display()));
            }
            continue;
        }
        // The bias is read only when there is no dark (the raw-master-dark convention), so only then is it checked.
        let bias = if group.dark.is_some() { None } else { group.bias.as_ref() };
        let roles = [("dark", group.dark.as_ref()), ("flat", group.flat.as_ref()), ("bias", bias)];
        // A raw set too small for a master is refused here, before group 1's lights are calibrated and group 2's
        // build fails.
        for (role, source) in roles {
            if let Some(CalSource::Raw(frames)) = source {
                if frames.len() < MIN_MASTER_FRAMES {
                    bail!("group {n} {role}: {}", too_few_frames(frames.len()));
                }
            }
        }
        if let (Some(CalSource::Master(flat)), Some(_)) = (&group.flat, &group.flat_precal) {
            warnings.push(format!(
                "group {n}: the flat pre-calibration is ignored, the flat {} is already a master",
                flat.display()
            ));
        }
        let mut lights: Vec<(&Path, FrameFacts)> = Vec::with_capacity(group.lights.len());
        for light in &group.lights {
            let facts = facts_of(&mut cache, light)?.clone();
            if let Some(planes) = facts.planes.filter(|p| *p > 1) {
                bail!("{}", planes_refusal(light, planes));
            }
            lights.push((light.as_path(), facts));
        }
        for (role, source) in roles {
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
                        // The constants are measured again only where they are per channel.
                        let again = if per_channel_applies(opts, cfa_of(light)) {
                            "; its per-channel constants are measured again from its pixels"
                        } else {
                            ""
                        };
                        warnings.push(format!(
                            "Flat {} has another CFA pattern or phase than light {} ({} against {}){again}",
                            path.display(),
                            light_path.display(),
                            cfa_label(cfa_of(&cal)),
                            cfa_label(cfa_of(light)),
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
    Ok(Checked { warnings, facts: cache, excluded })
}

/// Checks every group's lights and calibration frames against each other from their headers alone, before any pixel is
/// read (SL-20), and returns the warnings; [`preflight_with`] with the default options.
pub fn preflight(groups: &[CalGroup]) -> Result<Vec<String>> {
    preflight_with(groups, &CalibratedLightOptions::default())
}

/// [`preflight`] under the options a run would use (the bias fallback, the flat normalization and its per-channel
/// switch decide which lights are excluded and what a CFA mismatch means).
///
/// Refused, naming the files:
/// - a calibration frame whose size differs from a light's: "Calibration does not fit the frame size — dark `file` is
///   W×H, light `file` is W×H";
/// - a light or a calibration frame with more than one plane (the reason the integration reader gives);
/// - a raw calibration set with fewer than [`MIN_MASTER_FRAMES`] frames, naming the group and the role ("calibration
///   set has 2 frames — at least 3 are required to build a master");
/// - a flat pre-calibration master of another size than its flats.
///
/// Warned about:
/// - another sub-frame origin: "Calibration at another sub-frame origin — ...";
/// - a flat whose CFA pattern or phase differs from the light's (its per-channel constants are then measured again
///   from its pixels, where they are per channel);
/// - a flat pre-calibration given with a master flat (it is ignored);
/// - the lights of a group Athenaeum's run would exclude, one warning each with its reason: a group with no dark,
///   flat or bias at all ("no calibration masters available (dark/flat/bias unbuilt or unlinked)"), and one with a bias
///   alone when [`BiasFallback::SkipFrame`] is set ("no dark master (bias fallback disabled)"). Nothing of such a
///   group is read.
///
/// Exposure, temperature, filter and gain are not checked. Only the bias a light is calibrated with (none when there
/// is a dark) is checked. A compressed frame is read for its logical header, so its size is the image's; a gzip file
/// is inflated to find it, which takes seconds per frame (P2-R22), and a truncated one fails here, naming it.
pub fn preflight_with(groups: &[CalGroup], opts: &CalibratedLightOptions) -> Result<Vec<String>> {
    check(groups, opts).map(|c| c.warnings)
}

// ---------------------------------------------------------------------------------------------------------------
// The cards of a calibrated light
// ---------------------------------------------------------------------------------------------------------------

/// A typed card from a `KEYWORD -> value` pair: an integer, else a real, else a string (a keyword the writer rejects
/// is dropped), as the copied-through cards keep their type.
pub(crate) fn card_from_kv(keyword: &str, value: &str) -> Option<Card> {
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

/// A name as Athenaeum writes it into a header card (`encode_ident`): every byte outside printable ASCII, the space
/// and `%` itself as `%XX`, so a non-ASCII file name survives the writer's ASCII-only rule and reads back unchanged.
pub(crate) fn encode_ident(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if (0x21..=0x7E).contains(&b) && b != b'%' {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The file name of a path, as the identity cards carry it.
pub(crate) fn file_name_of(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// Whether the logical header of a light says what its depth is, not the physical one the direct reader probes: a
/// compressed light (`.fz`, gzip by name or by its first bytes) or one whose image sits in an extension behind an empty
/// primary (`NAXIS = 0`).
fn depth_is_logical(path: &Path) -> bool {
    if has_compressed_name(path) {
        return true;
    }
    let mut block = [0u8; 2880];
    let read = std::fs::File::open(path).and_then(|mut f| f.read(&mut block)).unwrap_or(0);
    if read >= 2 && block[0] == 0x1f && block[1] == 0x8b {
        return true;
    }
    block[..read].chunks(80).take_while(|card| !card.starts_with(b"END")).any(|card| {
        card.starts_with(b"NAXIS   =")
            && std::str::from_utf8(&card[9..]).ok().and_then(|v| v.split('/').next()).and_then(|v| v.trim().parse::<i64>().ok()) == Some(0)
    })
}

/// The divisor a light's pixels are scaled by (`SLM_CSCL`). Athenaeum's rule (`light_bitpix`, then
/// `scale_divisor_for_bitpix`): the depth `probe_bitpix` reads off a plain 2-D primary HDU, and the historic 65535 for
/// everything it cannot read (a 3-D one-plane file, a header over 64 blocks, an XISF) whatever the real depth. A
/// compressed or extension-HDU light is not a file Athenaeum catalogues and the direct reader cannot probe it; it
/// takes the depth `probe_bitpix` would give the same image stored plain, its logical `BITPIX` (P2-R22).
fn scale_divisor_of(light: &Path, facts: &FrameFacts) -> f64 {
    let bitpix = probe_bitpix(light).or_else(|| {
        (facts.format == HeaderFormat::Fits && depth_is_logical(light))
            .then_some(facts.bitpix)
            .flatten()
            .and_then(|b| i32::try_from(b).ok())
    });
    scale_divisor_for_bitpix(bitpix)
}

/// What the header of one calibrated light says about how it was made.
struct LightCardInputs<'a> {
    light: &'a Path,
    facts: &'a FrameFacts,
    cfg_hash: &'a str,
    calstat: &'a str,
    /// The master applied, by file name: the dark; the bias only when no dark applies; the flat.
    dark: Option<&'a Path>,
    flat: Option<&'a Path>,
    bias: Option<&'a Path>,
    scale_divisor: f64,
    flat_norm_divisor: f64,
    flat_channel_divisors: Option<[f64; 3]>,
    pedestal_dn: f64,
    trim_fraction: Option<f64>,
}

/// The calibrated light's header before the debayer and hot-pixel cards: Athenaeum's `build_light_cal_cards` with
/// `SLM_*` in place of every `ATH_*` card but `ATH_CSRC`, in its order (`IMAGETYP`, `CALSTAT`, the copy-through cards,
/// the common cards, then the twins of `ATH_CSRN`, `ATH_CDRK`, `ATH_CFLT`, `ATH_CBIA`, `ATH_CSCL`, `ATH_CFNM`,
/// `ATH_CPED`, `ATH_CVER`, `ATH_CCFA` / `ATH_CFNR` / `ATH_CFNG` / `ATH_CFNB`, `ATH_CTRM`). The master references hold
/// the master's file name, with no catalog id; string cards carry no comment, as there.
fn light_cards(i: &LightCardInputs<'_>) -> Result<Vec<Card>> {
    let mut cards = vec![
        Card::new("IMAGETYP", CardValue::Str("Light Frame".to_string()))?.with_comment("type of image"),
        Card::new("CALSTAT", CardValue::Str(i.calstat.to_string()))?.with_comment("calibration state (B/D/F)"),
    ];
    cards.extend(source_cards(i.light, i.facts)?);
    cards.extend(common_cards(i.cfg_hash)?);
    cards.push(Card::new(SLM_CSRN, CardValue::Str(encode_ident(&file_name_of(i.light))))?);
    for (keyword, master) in [(SLM_CDRK, i.dark), (SLM_CFLT, i.flat), (SLM_CBIA, i.bias)] {
        if let Some(master) = master {
            cards.push(Card::new(keyword, CardValue::Str(encode_ident(&file_name_of(master))))?);
        }
    }
    cards.push(Card::new(SLM_CSCL, CardValue::Real(i.scale_divisor))?.with_comment("numeric-scale divisor applied"));
    cards.push(Card::new(SLM_CFNM, CardValue::Real(i.flat_norm_divisor))?.with_comment("flat-norm divisor (1.0 = off)"));
    cards.push(Card::new(SLM_CPED, CardValue::Real(i.pedestal_dn))?.with_comment("output pedestal (DN)"));
    cards.push(version_card("SLM_LCV")?);
    if let Some(ch) = i.flat_channel_divisors {
        cards.push(Card::new(SLM_CCFA, CardValue::Logical(true))?.with_comment("flat normalized per CFA channel"));
        for (keyword, v) in [(SLM_CFNR, ch[0]), (SLM_CFNG, ch[1]), (SLM_CFNB, ch[2])] {
            cards.push(Card::new(keyword, CardValue::Real(v))?.with_comment("per-channel flat-norm divisor"));
        }
    }
    if let Some(trim) = i.trim_fraction {
        cards.push(Card::new(SLM_CTRM, CardValue::Real(trim))?.with_comment("flat-norm trim fraction per tail"));
    }
    Ok(cards)
}

// ---------------------------------------------------------------------------------------------------------------
// The dark guard's sentence
// ---------------------------------------------------------------------------------------------------------------

/// The share of a frame's pixels at or below zero right after the dark or bias was subtracted from which Athenaeum's
/// dark guard calls the subtraction lost (`DARK_SUBTRACTION_LOST_BAR`).
const DARK_SUBTRACTION_LOST_BAR: f64 = 0.9;

/// One header number as a warning shows it (Athenaeum's `card_number`).
fn card_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// A header card's values as text for a warning: one value, a range, or "not recorded" (Athenaeum's `card_text`).
fn card_text(name: &str, values: &[Option<f64>]) -> String {
    let mut recorded: Vec<f64> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    recorded.sort_by(f64::total_cmp);
    recorded.dedup();
    let partial = if values.iter().any(Option::is_none) { " (not recorded on some)" } else { "" };
    match recorded.as_slice() {
        [] => format!("{name} not recorded"),
        [v] => format!("{name} {}{partial}", card_number(*v)),
        [lo, .., hi] => format!("{name} {}\u{2013}{}{partial}", card_number(*lo), card_number(*hi)),
    }
}

/// The warning for a group whose dark or bias left most pixels at or below zero in `lost` of the `counted` frames that
/// had one subtracted, in Athenaeum's words (`subtraction_run_sentence`); `None` when no frame is lost.
fn subtraction_lost_warning(group: usize, counted: usize, lost: &[(&FrameFacts, &str)]) -> Option<String> {
    if lost.is_empty() {
        return None;
    }
    let (mut lost_dark, mut lost_bias) = (false, false);
    for (_, calstat) in lost {
        if calstat.starts_with("BD") {
            lost_dark = true;
        } else if calstat.starts_with('B') {
            lost_bias = true;
        }
    }
    let subtrahend = match (lost_dark, lost_bias) {
        (true, false) => "dark",
        (false, true) => "bias",
        _ => "dark or bias",
    };
    let offsets: Vec<Option<f64>> = lost.iter().map(|(f, _)| f.offset).collect();
    let gains: Vec<Option<f64>> = lost.iter().map(|(f, _)| f.gain).collect();
    Some(format!(
        "group {group}: dark subtraction left most pixels at or below zero in {} of {counted} frames ({}, {}) \u{2014} the {subtrahend} is brighter than the lights; check that its gain and offset match those of the lights",
        lost.len(),
        card_text("OFFSET", &offsets),
        card_text("GAIN", &gains),
    ))
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
    /// Where the calibrated CFA mosaic is kept (`keep_mosaic`, a frame being debayered): its part file and destination.
    mosaic: Option<(PathBuf, PathBuf)>,
    /// The flat-normalization divisor, resolved once per (flat, mosaic phase) for the run; `Some` when a flat is
    /// applied with normalization on.
    divisor: Option<FlatNormDivisor>,
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
    mosaic: Option<(PathBuf, PathBuf)>,
}

/// The key a flat-normalization divisor is memoized under: the flat and the light's mosaic phase folded as
/// [`CfaGeometry::same_phase`] folds it (pattern, offsets modulo 2), the only part of the geometry the divisor depends
/// on (Athenaeum's `DivisorKey`).
type DivisorKey = (PathBuf, Option<(&'static str, i64, i64)>);

fn divisor_key(flat: &Path, cfa: Option<CfaGeometry>) -> DivisorKey {
    (flat.to_path_buf(), cfa.map(|g| (g.pattern.as_str(), g.xoff.rem_euclid(2), g.yoff.rem_euclid(2))))
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

/// Calibrates one light and writes it (and, when asked, its CFA mosaic) to its part files.
#[allow(clippy::too_many_arguments)]
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
    let scale_divisor = scale_divisor_of(job.light, job.facts);

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
    let (mut frame, outcome) = match job.divisor {
        Some(divisor) => calibrate_light_compute_resolved(&inputs, Some(preloaded), &SlmFlatCards, divisor, ctx.cancel),
        None => calibrate_light_compute_with_cards(&inputs, Some(preloaded), &SlmFlatCards, ctx.cancel),
    }
    .map_err(|e| name(&e))?;

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

    // The header up to the debayer: the same cards head the mosaic and the debayered frame.
    let base = light_cards(&LightCardInputs {
        light: job.light,
        facts: job.facts,
        cfg_hash,
        calstat: &calstat,
        dark: if dark_applied { masters.dark.as_deref() } else { None },
        flat: masters.flat.as_deref(),
        bias: if bias_applied { masters.bias.as_deref() } else { None },
        scale_divisor,
        flat_norm_divisor: outcome.flat_norm_divisor,
        flat_channel_divisors: outcome.flat_channel_divisors,
        pedestal_dn: opts.params.pedestal_dn,
        trim_fraction: (masters.flat.is_some() && opts.flat_norm && opts.flat_norm_mode == FlatNormMode::TrimmedMeanWhole)
            .then_some(opts.params.trim_fraction),
    })?;
    let hot_card = match hot_pixels_replaced {
        Some(n) => Some(Card::new(SLM_CHPX, CardValue::Integer(n as i64))?.with_comment("hot pixels replaced")),
        None => None,
    };
    let durability = if opts.skip_fsync { Durability::Volatile } else { Durability::Durable };

    // The calibrated CFA mosaic, a second output of this generation: written before the debayer, from the buffer the
    // debayer is about to read. Its header keeps the mosaic cards and carries no `SLM_CDBM`; it is always plain FITS.
    if let Some((part, _)) = &job.mosaic {
        let mut mosaic_cards = base.clone();
        mosaic_cards.extend(hot_card.clone());
        write_fits_f32_with(part, frame.width, frame.height, 1, &frame.data, &mosaic_cards, durability)
            .map_err(|e| anyhow!("writing {}: {e}", part.display()))?;
    }

    let (data, channels) = match (job.debayer, job.cfa) {
        (true, Some(geom)) => (debayer_plane(&frame.data, frame.width, frame.height, geom, Some(&ctx.pool)), 3usize),
        _ => (std::mem::take(&mut frame.data), 1usize),
    };
    let mut cards = base;
    if channels == 3 {
        drop_mosaic_cards(&mut cards);
        cards.push(Card::new(SLM_CDBM, CardValue::Str("VNG".to_string()))?);
    }
    cards.extend(hot_card);

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
        mosaic: job.mosaic.clone(),
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
/// [`preflight_with`] runs first and its warnings go to the context's progress sink and the report; a refusal fails
/// the run before the output folder is created. The lights of a group Athenaeum's run would exclude (no dark, flat or
/// bias; a bias alone with [`BiasFallback::SkipFrame`]) are excluded with its reason as a warning each and nothing is
/// written for them; when no light is left the run fails. The products are built beside their destinations and moved
/// into place together with the entry ([`commit_stage`]), so a failure (a truncated gzip light, a cancel) leaves no
/// product and no entry. The entry's products are the calibrated frames in the frame order of P2-R10 (`DATE-OBS`, then
/// input position, across the calibrated lights of all groups), and its `sources` hold each raw light's facts, keyed by
/// the product.
///
/// With `opts.keep_mosaic`, an OSC light that is debayered also leaves its calibrated, hot-pixel-corrected CFA mosaic
/// beside it as `c_<name>.fits` (the mono spelling of the same name, always plain FITS, the mosaic's cards, no
/// `SLM_CDBM`), as Athenaeum's `execute_generation` does for a Bayer drizzle; the mosaic is no product of the entry.
/// Without `keep_mosaic`, a `c_<name>.fits` an earlier calibrate left beside a debayered frame this call wrote is
/// removed once the stage is committed: the Bayer drizzle ties a mosaic to its frame by file name, and that mosaic
/// belongs to another calibration (the same command removes its own stale product, P2-R24).
///
/// Lights are calibrated on the context's pool, as many at once as its memory budget allows; everything that decodes
/// a pixel runs there (SL-9). A hot-pixel map is measured once per master dark and a flat-normalization divisor
/// resolved once per (flat, mosaic phase) for the whole run. The command clears the scratch folder (`<work>/.tmp/`)
/// when it starts ([`begin_writing`](crate::pipeline::begin_writing)).
pub fn calibrate(
    ctx: &Ctx<'_>,
    cfg: &AlgorithmConfig,
    groups: &[CalGroup],
    opts: &CalibratedLightOptions,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<CalibrateReport> {
    let checked = check(groups, opts)?;
    let mut report = CalibrateReport { warnings: checked.warnings.clone(), ..CalibrateReport::default() };
    for w in &checked.warnings {
        ctx.progress.warn(w);
    }
    for (gi, g) in groups.iter().enumerate() {
        if let Some(reason) = checked.excluded[gi] {
            for light in &g.lights {
                tracing::warn!(src = %light.display(), error = %reason, "calibration failed; frame excluded");
            }
        }
    }
    let out_dir = std::path::absolute(out_dir).with_context(|| format!("cannot make {} absolute", out_dir.display()))?;
    ctx.check_cancelled()?;

    // The frame order every stage keeps, across the calibrated groups; and the names the products will have.
    let all: Vec<(usize, &PathBuf)> = groups
        .iter()
        .enumerate()
        .filter(|(gi, _)| checked.excluded[*gi].is_none())
        .flat_map(|(gi, g)| g.lights.iter().map(move |l| (gi, l)))
        .collect();
    if all.is_empty() {
        bail!("no light could be calibrated: every light was excluded ({})", checked.excluded.iter().flatten().next().copied().unwrap_or(NO_MASTERS));
    }
    let all_paths: Vec<PathBuf> = all.iter().map(|(_, l)| (*l).clone()).collect();
    let all_facts: Vec<FrameFacts> = all.iter().map(|(_, l)| checked.facts[*l].clone()).collect();
    let order = order_lights(&all_paths, &all_facts)?;
    let ext = format.extension();
    let mut dests: Vec<PathBuf> = Vec::with_capacity(all.len());
    let mut mosaics: Vec<Option<PathBuf>> = Vec::with_capacity(all.len());
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (i, light) in all_paths.iter().enumerate() {
        let debayer = opts.debayer_osc && cfa_of(&all_facts[i]).is_some();
        let stem = frame_stem(light);
        let file = format!("c_{stem}{}.{ext}", if debayer { "_d" } else { "" });
        // The mosaic is the same stem under the mono spelling, always FITS; no other product may take its name.
        let mosaic = (opts.keep_mosaic && debayer).then(|| format!("c_{stem}.{}", OutputFormat::Fits.extension()));
        for name in std::iter::once(&file).chain(mosaic.as_ref()) {
            if let Some(first) = seen.insert(name.to_lowercase(), i) {
                bail!(
                    "{} and {} would both become {name}; give lights with distinct names",
                    all_paths[first].display(),
                    light.display()
                );
            }
        }
        dests.push(out_dir.join(file));
        mosaics.push(mosaic.map(|m| out_dir.join(m)));
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
    // Per run, as Athenaeum caches them: a hot-pixel map per master dark (a dark shared by two groups is measured and
    // warned about once) and a flat-normalization divisor per (flat, mosaic phase).
    let mut hot_maps: HashMap<PathBuf, Arc<HotPixelMapOutcome>> = HashMap::new();
    let mut divisors: HashMap<DivisorKey, FlatNormDivisor> = HashMap::new();
    let mut run = |finished: &mut Vec<(usize, Done)>, report: &mut CalibrateReport| -> Result<()> {
        let mut index = 0usize;
        for (gi, group) in groups.iter().enumerate() {
            if checked.excluded[gi].is_some() {
                continue;
            }
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

            // The master dark's hot pixels, measured once per dark for the run; only the dark can say which pixels are
            // defective.
            if let (true, Some(dark)) = (opts.hot_pixel_correction, &masters.dark) {
                if !hot_maps.contains_key(dark) {
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

            let mut jobs: Vec<Job<'_>> = (0..group.lights.len())
                .map(|k| {
                    let i = first_index + k;
                    let cfa = cfa_of(&all_facts[i]);
                    let debayer = opts.debayer_osc && cfa.is_some();
                    Job {
                        light: group.lights[k].as_path(),
                        facts: &all_facts[i],
                        cfa,
                        debayer,
                        part: part_path(&dests[i]),
                        dest: dests[i].clone(),
                        mosaic: mosaics[i].as_ref().map(|m| (part_path(m), m.clone())),
                        divisor: None,
                    }
                })
                .collect();

            // The flat's normalization divisor, resolved once per (flat, mosaic phase): a flat without stamped
            // constants is read once, not once per light.
            if let (Some(flat), true) = (&masters.flat, opts.flat_norm) {
                for job in &mut jobs {
                    let key = divisor_key(flat, job.cfa);
                    let divisor = match divisors.get(&key) {
                        Some(d) => *d,
                        None => {
                            let d = resolve_flat_norm_divisor_with(flat, &scratch, opts.flat_norm_mode, &opts.params, job.cfa, &SlmFlatCards)
                                .map_err(|e| anyhow!("{}: {e}", flat.display()))?;
                            divisors.insert(key, d);
                            d
                        }
                    };
                    job.divisor = Some(divisor);
                }
            }

            // How many lights at once: what the budget holds after the resident masters, one frame's planes each.
            let px = all_facts[first_index..index].iter().filter_map(|f| size_of(f).map(|(w, h)| (w * h) as usize)).max().unwrap_or(0);
            let osc = jobs.iter().any(|j| j.debayer);
            let per_frame = px.saturating_mul(4).saturating_mul(if osc { 8 } else { 3 }).max(1);
            let resident = preloaded.planes.len().saturating_mul(px).saturating_mul(4);
            let concurrency = (budget.saturating_sub(resident) / per_frame).clamp(1, threads);

            let group_start = finished.len();
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

            // The dark guard: a group whose dark or bias left most pixels at or below zero says so once.
            let counted: Vec<&(usize, Done)> = finished[group_start..].iter().filter(|(_, d)| d.report.pixels > 0).collect();
            let lost: Vec<(&FrameFacts, &str)> = counted
                .iter()
                .filter(|(_, d)| d.report.non_positive_pixels as f64 / d.report.pixels as f64 >= DARK_SUBTRACTION_LOST_BAR)
                .map(|(i, d)| (&all_facts[*i], d.report.calstat.as_str()))
                .collect();
            if let Some(message) = subtraction_lost_warning(n, counted.len(), &lost) {
                tracing::warn!(group = n, count = lost.len(), "dark subtraction left most pixels at or below zero");
                ctx.progress.warn(&message);
                report.warnings.push(message);
            }
        }
        Ok(())
    };
    // Everything below decodes on the caller's pool, never rayon's global one (SL-9).
    if let Err(e) = ctx.pool.install(|| run(&mut finished, &mut report)) {
        remove_parts(
            finished
                .iter()
                .flat_map(|(_, d)| std::iter::once(d.part.clone()).chain(d.mosaic.as_ref().map(|(p, _)| p.clone())))
                .chain(dests.iter().map(|d| part_path(d)))
                .chain(mosaics.iter().flatten().map(|m| part_path(m))),
        );
        return Err(e);
    }

    // The products in the order every later stage keeps.
    let mut by_index: Vec<Option<Done>> = (0..total).map(|_| None).collect();
    for (i, done) in finished {
        by_index[i] = Some(done);
    }
    let mut moves = Vec::with_capacity(total);
    let mut mosaic_moves = Vec::new();
    let mut entry = Entry::new(
        Stage::Calibrate,
        cfg,
        order.iter().map(|&i| all_paths[i].clone()).collect(),
        order.iter().map(|&i| dests[i].clone()).collect(),
    );
    for &i in &order {
        let done = by_index[i].take().ok_or_else(|| anyhow!("{} was not calibrated", all_paths[i].display()))?;
        moves.push((done.part, done.dest.clone()));
        mosaic_moves.extend(done.mosaic);
        entry.sources.insert(done.dest, all_facts[i].clone());
        report.frames.push(done.report);
    }
    moves.extend(mosaic_moves);
    if let Err(e) = commit_stage(ctx.work_dir, &moves, entry) {
        remove_parts(moves.iter().map(|(part, _)| part.clone()));
        return Err(e);
    }
    // A mosaic an earlier calibrate left beside a debayered frame this run wrote describes another calibration: a Bayer
    // drizzle would deposit its pixels next to this run's frame by file name alone. The command that writes the frame
    // removes the product it no longer makes, as `stack` does for its own outputs (P2-R24).
    for (i, dest) in dests.iter().enumerate() {
        let debayered = opts.debayer_osc && cfa_of(&all_facts[i]).is_some();
        if !debayered || mosaics[i].is_some() {
            continue;
        }
        let stale = out_dir.join(format!("c_{}.{}", frame_stem(&all_paths[i]), OutputFormat::Fits.extension()));
        // A name another product of this run took is that product, not a stale mosaic.
        let is_product = dests.iter().any(|d| d.to_string_lossy().to_lowercase() == stale.to_string_lossy().to_lowercase());
        if is_product || !stale.is_file() {
            continue;
        }
        match std::fs::remove_file(&stale) {
            Ok(()) => tracing::info!(path = %stale.display(), frame = %dest.display(), "stale cfa mosaic removed"),
            Err(e) => {
                let message = format!(
                    "{} is a calibrated CFA mosaic of an earlier calibration and could not be removed ({e}); a Bayer drizzle would deposit it beside {}",
                    stale.display(),
                    dest.display()
                );
                tracing::warn!(path = %stale.display(), error = %e, "cannot remove a stale cfa mosaic");
                ctx.progress.warn(&message);
                report.warnings.push(message);
            }
        }
    }
    tracing::info!(frames = total, masters = report.masters.len(), "calibrate stage written");
    Ok(report)
}
