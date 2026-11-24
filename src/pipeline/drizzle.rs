//! Drizzle inside `stack`: Athenaeum's drizzle block of the Output stage, for one group, without the catalog, the
//! project grid and the crop (P2-R3).
//!
//! `stack` runs it after the integration, while the rejection bitmaps still exist (`<work>/.tmp/stack-rej/`, removed
//! when `stack` returns), when `drizzle.enabled` is set. What it does, as Athenaeum's block does:
//!
//! - The drop is shrunk by `drizzle.dropShrink`, clamped to `[0.5, 1.0]` with a warning; a `drizzle.scale` outside
//!   1 to 3 is refused before any pixel is read ([`settings`]).
//! - It deposits the frames the integration kept (`GroupOutput::included`, engine order) with their normalized weights,
//!   the output normalization pairs the integration computed, and the rejection bitmaps the master was built with: the
//!   second pass's own set when large-scale rejection ran ([`rejection_paths`]). It follows local normalization only
//!   when local normalization drives the output (`normalization.local.enabled`): grids computed for the rejection
//!   alone never normalize the drizzled master.
//! - It is skipped, with the master untouched, when the rejection bitmaps are unavailable: the set could not be created
//!   or written, or the large-scale second pass left no set of its own ([`second_pass_unavailable`]).
//! - A drizzle that fails (a read, the memory refusal, a write) is a warning: the master and its entry stay. A cancel
//!   is an error.
//! - Bayer drizzle (`drizzle.bayer`, three planes, the debayer on) deposits each colour from the frame's calibrated CFA
//!   mosaic, `c_<stem>.fits` beside the debayered `c_<stem>_d.fits` that `calibrate --keep-mosaic` writes
//!   ([`mosaic_sources`]). A frame with no mosaic is deposited from its debayered planes, with a warning; **a stack in
//!   which no frame has one is refused before any pixel is read**, naming the flag (Athenaeum falls back for every frame
//!   in that case: its calibration stage always keeps the mosaics, and a CLI user may not have asked it to).
//!
//! Products: `<out>/master_drizzle<s>x.fits` (or `.xisf`, as the master), and with `drizzle.writeWeightMap`
//! `<out>/master_drizzle<s>x_weight.fits` (the master's container too, as Athenaeum writes it). Their cards are the
//! master's with the world-coordinate block scaled by the drizzle factor (`CRPIX`, `CD`, `CDELT` and the SIP terms, as
//! Athenaeum's `scale_plate_solve` scales the solve) and `SLM_DRZ`, `SLM_DRZP`, `SLM_DRZK` and `SLM_DZV` (P2-R6) in place
//! of `ATH_DRZ`, `ATH_DRZP` and `ATH_DRZK`. A drizzle coverage map is not written: the pipeline has no coverage-map
//! setting, and the master has none.
//!
//! Everything that decodes, deposits or writes runs on the context's pool (SL-9): `stack` calls this inside it.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use astroimage::BayerPattern;

use crate::calibration_library::debayer::bayer_for;
use crate::fits_writer::{write_image_f32, Card, CardValue, OutputFormat, XisfBounds};
use crate::pipeline::cards::version_card;
use crate::pipeline::formats::part_path;
use crate::pipeline::master::{cfa_of, frame_facts};
use crate::pipeline::register::is_wcs;
use crate::pipeline::stack::MASTER_STEM;
use crate::pipeline::{frame_stem, Ctx, Stage};
use crate::stacking::config::{AlgorithmConfig, DrizzleConfig};
use crate::stacking::drizzle::{
    drizzle_group, CfaSource, DrizzleError, DrizzleFrame, DrizzleInput, DrizzleProgress, DrizzleStats,
};
use crate::stacking::integrate::{GroupOutput, StackFrame};
use crate::stacking::ln::LnFrameGrids;
use crate::stacking::rej::RejBitmapSet;
use crate::stacking::run_helpers::wants_cfa_mosaic;

/// `SLM_DRZ`: the drizzle's output scale.
pub const SLM_DRZ: &str = "SLM_DRZ";
/// `SLM_DRZP`: the drop shrink (`pixfrac`) the drizzle used.
pub const SLM_DRZP: &str = "SLM_DRZP";
/// `SLM_DRZK`: the drop kernel the drizzle used.
pub const SLM_DRZK: &str = "SLM_DRZK";
/// `SLM_DZV`: the drizzle kernel's version ([`DRIZZLE_KERNEL_VERSION`](crate::stacking::config::DRIZZLE_KERNEL_VERSION)).
pub const SLM_DZV: &str = "SLM_DZV";

/// What the drizzle made: the drizzled master, its weight map when one was asked for, and the pass's statistics.
#[derive(Debug, Clone)]
pub struct DrizzleReport {
    /// The drizzled master, `<out>/master_drizzle<s>x.fits` (`.xisf` for an XISF master).
    pub master: PathBuf,
    /// The weight map beside it (`drizzle.writeWeightMap`).
    pub weight_map: Option<PathBuf>,
    /// The pass's statistics: the output grid, the frames deposited, what was applied, the drizzled planes' FWHM,
    /// eccentricity and noise.
    pub stats: DrizzleStats,
}

/// The drizzle settings after the run-wide checks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    /// The output scale, 1 to 3.
    pub scale: u32,
    /// The drop shrink, within `[0.5, 1.0]`.
    pub drop_shrink: f64,
}

/// Why a drizzle that was attempted made nothing.
#[derive(Debug)]
pub(crate) enum Failure {
    /// The caller cancelled; the stage fails.
    Cancelled,
    /// Anything else; the stage warns and keeps the master.
    Other(String),
}

impl From<DrizzleError> for Failure {
    fn from(e: DrizzleError) -> Self {
        match e {
            DrizzleError::Cancelled => Failure::Cancelled,
            other => Failure::Other(other.to_string()),
        }
    }
}

/// The run-wide checks of the drizzle settings, made before any pixel is read: a `scale` outside 1 to 3 is refused (the
/// plan blocker of Athenaeum's run), a `dropShrink` outside `[0.5, 1.0]` is clamped with a warning.
pub(crate) fn settings(ctx: &Ctx<'_>, drizzle: &DrizzleConfig) -> Result<Settings> {
    if !(1..=3).contains(&drizzle.scale) {
        bail!("drizzle scale must be 1, 2 or 3, got {}", drizzle.scale);
    }
    let configured = drizzle.drop_shrink;
    if !configured.is_finite() {
        bail!("drizzle dropShrink must be a number within [0.5, 1.0], got {configured}");
    }
    let drop_shrink = if (0.5..=1.0).contains(&configured) {
        configured
    } else {
        let clamped = configured.clamp(0.5, 1.0);
        let message = format!("drizzle drop shrink {configured} out of [0.5, 1.0]; clamped to {clamped}");
        tracing::warn!(drop_shrink = configured, clamped, "drizzle drop shrink out of [0.5, 1.0]; clamped");
        ctx.progress.warn(&message);
        clamped
    };
    Ok(Settings { scale: drizzle.scale, drop_shrink })
}

/// Whether the stack is a Bayer drizzle: the configuration asks for it and the group has three planes.
pub(crate) fn bayer_wanted(cfg: &AlgorithmConfig, channels: usize) -> bool {
    wants_cfa_mosaic(&cfg.drizzle, &cfg.calibration, channels == 3) && channels == 3
}

/// The calibrated CFA mosaic that goes with the debayered frame `frame`: `c_<stem>.fits` in the frame's folder for
/// `c_<stem>_d.fits`. A frame whose name does not end in `_d` was not debayered by `calibrate`, so it has none.
pub(crate) fn mosaic_of(frame: &Path) -> Option<PathBuf> {
    let base = frame_stem(frame).strip_suffix("_d")?;
    Some(frame.with_file_name(format!("{base}.{}", OutputFormat::Fits.extension())))
}

/// Each member's mosaic and its phase-corrected pattern, `None` where the member has no readable mosaic with a CFA
/// pattern. The pattern is read from the mosaic's own header (the light's `BAYERPAT`, `XBAYROFF` and `YBAYROFF`, which
/// the mosaic keeps) through [`bayer_for`], the function the debayer itself took its pattern from, so the deposit and
/// the debayer cannot disagree about which pixel is red.
pub(crate) fn mosaic_sources(members: &[StackFrame]) -> Vec<Option<(PathBuf, BayerPattern)>> {
    members
        .iter()
        .map(|m| {
            let path = mosaic_of(&m.path)?;
            if !path.is_file() {
                return None;
            }
            match frame_facts(&path) {
                Ok(facts) => cfa_of(&facts).map(|geom| (path, bayer_for(geom))),
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %format!("{e:#}"), "bayer drizzle: reading the frame's mosaic phase failed");
                    None
                }
            }
        })
        .collect()
}

/// The rejection bitmaps the drizzle reads, per included frame: the bits the master was built with. With the
/// large-scale second pass on, that is the second pass's own set; without it, the first pass's. The processed
/// intermediates are neither. `second_pass_rej_ok` false with the second pass having run means its files are missing
/// ([`second_pass_unavailable`]), which the caller turns into a skip.
pub(crate) fn rejection_paths(set: Option<&RejBitmapSet>, included: usize, second_pass_rej_ok: bool) -> Vec<Option<PathBuf>> {
    (0..included)
        .map(|k| set.map(|s| if second_pass_rej_ok { s.second_pass_path(k) } else { s.path(k).to_path_buf() }))
        .collect()
}

/// Whether the large-scale second pass ran and left no usable bitmap set of its own, so nothing on disk describes the
/// master's rejected samples: drizzle is skipped rather than pointed at the first pass's bits, which describe the
/// integration the second pass replaced. Gated on drizzle wanting the bitmaps (`bitmaps_for_drizzle`), because the
/// degradation is drizzle's: with drizzle off the second pass deliberately writes no set of its own. `failure` is a
/// failure of the bitmap set that is already known, which this one does not repeat.
pub(crate) fn second_pass_unavailable(
    bitmaps_for_drizzle: bool,
    large_scale_ran: bool,
    second_pass_rej_ok: bool,
    failure: Option<&str>,
) -> bool {
    bitmaps_for_drizzle && large_scale_ran && !second_pass_rej_ok && failure.is_none()
}

/// Where the drizzled master of `scale` goes: `<out>/master_drizzle<scale>x.<ext>`.
pub(crate) fn drizzle_path(out_dir: &Path, scale: u32, format: OutputFormat) -> PathBuf {
    out_dir.join(format!("{MASTER_STEM}_drizzle{scale}x.{}", format.extension()))
}

/// Where the weight map of the drizzled master of `scale` goes: `<out>/master_drizzle<scale>x_weight.<ext>`.
pub(crate) fn weight_path(out_dir: &Path, scale: u32, format: OutputFormat) -> PathBuf {
    out_dir.join(format!("{MASTER_STEM}_drizzle{scale}x_weight.{}", format.extension()))
}

/// Every path a drizzle of any allowed scale may have left in `out_dir` in `format`, for the cleanup of what an earlier
/// stack left beside a master it no longer describes.
pub(crate) fn possible_products(out_dir: &Path, format: OutputFormat) -> Vec<PathBuf> {
    (1..=3).flat_map(|s| [drizzle_path(out_dir, s, format), weight_path(out_dir, s, format)]).collect()
}

/// The numeric value of a card, when it has one.
fn number(card: &Card) -> Option<f64> {
    match card.value {
        Some(CardValue::Real(v)) => Some(v),
        Some(CardValue::Integer(v)) => Some(v as f64),
        _ => None,
    }
}

/// The two indices of a SIP term keyword (`A_2_0`, `B_1_1`, `AP_0_3`, `BP_2_1`).
fn sip_term(keyword: &str) -> Option<(i32, i32)> {
    for prefix in ["AP_", "BP_", "A_", "B_"] {
        let Some(rest) = keyword.strip_prefix(prefix) else { continue };
        let mut parts = rest.split('_');
        let (Some(i), Some(j), None) = (parts.next(), parts.next(), parts.next()) else { continue };
        let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        if digits(i) && digits(j) {
            return Some((i.parse().ok()?, j.parse().ok()?));
        }
    }
    None
}

/// One world-coordinate card for an image resampled by the integer factor `s` about the pixel grid (Athenaeum's
/// `scale_plate_solve`): `CRPIX' = s·CRPIX − (s − 1)/2` (the reference pixel's centre lands between its `s × s` output
/// pixels), `CD' = CD / s` and `CDELT' = CDELT / s`, and each SIP term `A_pq' = A_pq · s^(1 − p − q)`. Every other card
/// is unchanged.
fn scaled_wcs_card(card: &Card, s: f64) -> Result<Card> {
    let Some(value) = number(card) else { return Ok(card.clone()) };
    let key = card.keyword.as_str();
    let scaled = if matches!(key, "CRPIX1" | "CRPIX2") {
        s * value - (s - 1.0) / 2.0
    } else if matches!(key, "CD1_1" | "CD1_2" | "CD2_1" | "CD2_2" | "CDELT1" | "CDELT2") {
        value / s
    } else if let Some((p, q)) = sip_term(key) {
        value * s.powi(1 - p - q)
    } else {
        return Ok(card.clone());
    };
    if !scaled.is_finite() {
        bail!("{key} is not finite after drizzle scaling: {scaled}");
    }
    let mut out = card.clone();
    out.value = Some(CardValue::Real(scaled));
    Ok(out)
}

/// The cards of the drizzled master: the master's with its world-coordinate block taken out and put back scaled by
/// `settings.scale`, after the others, then `SLM_DRZ`, `SLM_DRZP`, `SLM_DRZK` and `SLM_DZV` (P2-R6).
pub(crate) fn drizzle_cards(master: &[Card], settings: Settings, cfg: &DrizzleConfig) -> Result<Vec<Card>> {
    let s = f64::from(settings.scale);
    let mut cards: Vec<Card> = master.iter().filter(|c| !is_wcs(&c.keyword)).cloned().collect();
    for card in master.iter().filter(|c| is_wcs(&c.keyword)) {
        cards.push(scaled_wcs_card(card, s)?);
    }
    let kernel = match serde_json::to_value(cfg.kernel) {
        Ok(serde_json::Value::String(name)) => name,
        other => bail!("the drizzle kernel did not serialize to a string: {other:?}"),
    };
    cards.push(Card::new(SLM_DRZ, CardValue::Integer(i64::from(settings.scale)))?.with_comment("drizzle output scale"));
    cards.push(Card::new(SLM_DRZP, CardValue::Real(settings.drop_shrink))?.with_comment("drizzle drop shrink"));
    cards.push(Card::new(SLM_DRZK, CardValue::Str(kernel))?.with_comment("drizzle kernel"));
    cards.push(version_card(SLM_DZV)?);
    Ok(cards)
}

/// The cards of the drizzle weight map: `IMAGETYP`, `BUNIT`, the drizzled master's scaled world-coordinate block (the
/// map is its grid), and its `ROWORDER`, trace cards and drizzle cards copied through, as Athenaeum's `weight_map_cards`
/// copies its provenance cards.
pub(crate) fn weight_map_cards(drizzle: &[Card]) -> Result<Vec<Card>> {
    use crate::pipeline::register::SLM_REF;
    use crate::pipeline::{SLM_CFG, SLM_PLV, SLM_VER};
    let mut cards = vec![
        Card::new("IMAGETYP", CardValue::Str("Drizzle Weight".to_string()))?,
        Card::new("BUNIT", CardValue::Str("relative weight".to_string()))?,
    ];
    cards.extend(drizzle.iter().filter(|c| is_wcs(&c.keyword)).cloned());
    for keyword in ["ROWORDER", SLM_VER, SLM_CFG, SLM_PLV, SLM_REF, SLM_DRZ, SLM_DZV] {
        if let Some(c) = drizzle.iter().find(|c| c.keyword == keyword) {
            cards.push(c.clone());
        }
    }
    Ok(cards)
}

/// What [`drizzle`] is given: the group as `stack` integrated it.
pub(crate) struct Inputs<'a> {
    /// The effective config: the drizzle section, the local-normalization switch and the measurement options.
    pub cfg: &'a AlgorithmConfig,
    /// The settings after [`settings`].
    pub settings: Settings,
    /// The stack's members, in frame order.
    pub members: &'a [StackFrame],
    /// The integration's output: the included frames and the output normalization pairs.
    pub output: &'a GroupOutput,
    /// Each member's local-normalization grids, when local normalization applies to the stack.
    pub ln: Option<&'a [Option<LnFrameGrids>]>,
    /// The rejection-bitmap set the integration wrote, when one was created.
    pub rej: Option<&'a RejBitmapSet>,
    /// Each member's mosaic and pattern under Bayer drizzle ([`mosaic_sources`]); `None` for a plain drizzle.
    pub mosaics: Option<&'a [Option<(PathBuf, BayerPattern)>]>,
    /// The registration reference's grid: the output grid is this times the scale.
    pub width: usize,
    /// See `width`.
    pub height: usize,
    /// The number of planes.
    pub channels: usize,
    /// The master's cards (the drizzled master's are made from them).
    pub master_cards: &'a [Card],
    /// The folder the products go to.
    pub out_dir: &'a Path,
    /// The master's container.
    pub format: OutputFormat,
}

/// The products of a drizzle, built beside their destinations.
pub(crate) struct Built {
    /// `(built, destination)` pairs to rename into place with the stage's entry.
    pub moves: Vec<(PathBuf, PathBuf)>,
    /// What was made.
    pub report: DrizzleReport,
}

/// Deposits the integration's included frames onto the scaled grid and writes the drizzled master (and its weight map)
/// to part files next to their destinations. A failure removes what it wrote.
pub(crate) fn drizzle(ctx: &Ctx<'_>, input: &Inputs<'_>) -> std::result::Result<Built, Failure> {
    let Inputs { cfg, settings, members, output, .. } = *input;
    let drizzle_cfg = &cfg.drizzle;
    let included = &output.included;

    // The output pairs per included frame: the integration's are per plane, then per included frame.
    let shape_ok = output.output_pairs.len() == input.channels && output.output_pairs.iter().all(|p| p.len() == included.len());
    if !shape_ok {
        return Err(Failure::Other(format!(
            "output_pairs shape does not match {} channel(s) x {} included frame(s)",
            input.channels,
            included.len()
        )));
    }
    let pairs_by_frame: Vec<Vec<_>> = (0..included.len()).map(|k| (0..input.channels).map(|p| output.output_pairs[p][k]).collect()).collect();
    let rej_paths = rejection_paths(input.rej, included.len(), output.second_pass_rej_ok);

    // Bayer drizzle: a frame with no mosaic is deposited from its debayered planes, with a warning.
    let mosaic_of_included = |k: usize| input.mosaics.and_then(|m| m.get(included[k])).and_then(Option::as_ref);
    if input.mosaics.is_some() {
        let missing = (0..included.len()).filter(|&k| mosaic_of_included(k).is_none()).count();
        if missing > 0 {
            tracing::warn!(count = missing, "bayer drizzle: frames without a cfa mosaic fall back to the debayered deposit");
            ctx.progress.warn(&format!(
                "Bayer drizzle deposited {missing} frame(s) from their debayered planes: no calibrated CFA mosaic was available for them"
            ));
        }
    }

    let frames: Vec<DrizzleFrame<'_>> = included
        .iter()
        .enumerate()
        .map(|(k, &idx)| DrizzleFrame {
            path: members[idx].path.as_path(),
            map: &members[idx].map,
            weight: &members[idx].weight.normalized,
            output_pair: &pairs_by_frame[k],
            ln: input.ln.and_then(|grids| grids.get(idx)).and_then(Option::as_ref),
            rej: rej_paths[k].as_deref(),
            cfa: mosaic_of_included(k).map(|(path, pattern)| CfaSource { path: path.as_path(), pattern: *pattern }),
        })
        .collect();

    let measure = cfg.measurement.measure_options(cfg.normalization.scale_estimator);
    let drizzle_input = DrizzleInput {
        frames: &frames,
        width: input.width,
        height: input.height,
        channels: input.channels,
        scale: settings.scale,
        drop_shrink: settings.drop_shrink,
        kernel: drizzle_cfg.kernel,
        use_weights: drizzle_cfg.use_weights,
        use_rejection: drizzle_cfg.use_rejection && input.rej.is_some(),
        // The drizzle follows the plain master's output normalization: grids computed for the rejection alone
        // (`rejection = local` with `local.enabled` off, the shipped default) never normalize the drizzled master, or it
        // would be locally normalized while the master beside it is not.
        use_local_normalization: drizzle_cfg.use_local_normalization && cfg.normalization.local.enabled,
        write_weight_map: drizzle_cfg.write_weight_map,
        write_coverage: false,
        stats_on_covered: false,
        stats_rect: None,
        measure: &measure,
        ram_total_bytes: None,
        force_exact_overlap_for_measurement: false,
    };
    let total_units = included.len() * input.channels;
    let on_frame = |done: usize, total: usize| ctx.progress.step(Stage::Stack.name(), done, total);
    ctx.progress.step(Stage::Stack.name(), 0, total_units);
    let drz = drizzle_group(&drizzle_input, &ctx.pool, ctx.cancel, &DrizzleProgress { on_frame: &on_frame })?;

    let cards = drizzle_cards(input.master_cards, settings, drizzle_cfg).map_err(|e| Failure::Other(format!("{e:#}")))?;
    let dest = drizzle_path(input.out_dir, settings.scale, input.format);
    let mut moves: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut weight_map = None;
    let written = (|| -> Result<()> {
        let built = part_path(&dest);
        write_image_f32(&built, drz.width, drz.height, drz.channels, &drz.data, &cards, input.format, XisfBounds::Unit)
            .map_err(|e| anyhow!("cannot write {}: {e}", dest.display()))?;
        moves.push((built, dest.clone()));
        if let Some(weight) = drz.weight.as_deref() {
            let wdest = weight_path(input.out_dir, settings.scale, input.format);
            let built = part_path(&wdest);
            write_image_f32(&built, drz.width, drz.height, drz.channels, weight, &weight_map_cards(&cards)?, input.format, XisfBounds::Unit)
                .map_err(|e| anyhow!("cannot write {}: {e}", wdest.display()))?;
            moves.push((built, wdest.clone()));
            weight_map = Some(wdest);
        }
        Ok(())
    })();
    if let Err(e) = written {
        for (built, _) in &moves {
            let _ = std::fs::remove_file(built);
        }
        return Err(Failure::Other(format!("{e:#}")));
    }
    tracing::info!(
        path = %dest.display(),
        drizzle_scale = settings.scale,
        out_width = drz.stats.out_width,
        out_height = drz.stats.out_height,
        "drizzled master built"
    );
    Ok(Built { moves, report: DrizzleReport { master: dest, weight_map, stats: drz.stats } })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real(keyword: &str, v: f64) -> Card {
        Card::new(keyword, CardValue::Real(v)).unwrap()
    }

    fn value_of(cards: &[Card], keyword: &str) -> f64 {
        number(cards.iter().find(|c| c.keyword == keyword).unwrap()).unwrap()
    }

    #[test]
    fn the_world_coordinate_block_scales_as_athenaeums_scale_plate_solve_does() {
        let wcs = [
            real("CRPIX1", 100.5),
            real("CRPIX2", 50.0),
            real("CRVAL1", 83.8),
            real("CD1_1", -1.0e-3),
            real("CD1_2", 2.0e-4),
            real("CDELT1", -3.0e-3),
            real("A_0_0", 0.5),
            real("A_1_0", 0.25),
            real("A_2_0", 1.0e-5),
            real("B_1_1", 4.0e-6),
            real("AP_3_0", 8.0e-8),
            Card::new("CTYPE1", CardValue::Str("RA---TAN-SIP".into())).unwrap(),
            Card::new("A_ORDER", CardValue::Integer(3)).unwrap(),
        ];
        let scaled: Vec<Card> = wcs.iter().map(|c| scaled_wcs_card(c, 2.0).unwrap()).collect();
        // CRPIX' = s*CRPIX - (s - 1)/2 on the 1-based card: the 0-based crpix' = s*crpix + (s - 1)/2.
        assert_eq!(value_of(&scaled, "CRPIX1"), 200.5);
        assert_eq!(value_of(&scaled, "CRPIX2"), 99.5);
        assert_eq!(value_of(&scaled, "CRVAL1"), 83.8);
        assert_eq!(value_of(&scaled, "CD1_1"), -5.0e-4);
        assert_eq!(value_of(&scaled, "CD1_2"), 1.0e-4);
        assert_eq!(value_of(&scaled, "CDELT1"), -1.5e-3);
        // A_pq * s^(1 - p - q).
        assert_eq!(value_of(&scaled, "A_0_0"), 1.0);
        assert_eq!(value_of(&scaled, "A_1_0"), 0.25);
        assert_eq!(value_of(&scaled, "A_2_0"), 5.0e-6);
        assert_eq!(value_of(&scaled, "B_1_1"), 2.0e-6);
        assert_eq!(value_of(&scaled, "AP_3_0"), 2.0e-8);
        // Strings and the order are untouched.
        assert!(matches!(&scaled[11].value, Some(CardValue::Str(s)) if s == "RA---TAN-SIP"));
        assert!(matches!(scaled[12].value, Some(CardValue::Integer(3))));
    }

    #[test]
    fn a_sip_term_that_overflows_after_scaling_is_an_error_naming_the_card() {
        let huge = real("A_0_0", f64::MAX);
        let err = scaled_wcs_card(&huge, 3.0).unwrap_err().to_string();
        assert!(err.contains("A_0_0") && err.contains("not finite"), "{err}");
    }

    #[test]
    fn a_mosaic_is_the_debayered_frames_stem_without_the_d_in_the_same_folder() {
        assert_eq!(mosaic_of(Path::new("/x/y/c_Light_0001_d.fits")), Some(PathBuf::from("/x/y/c_Light_0001.fits")));
        assert_eq!(mosaic_of(Path::new("c_Light_0001_d.fits.gz")), Some(PathBuf::from("c_Light_0001.fits")));
        assert_eq!(mosaic_of(Path::new("/x/c_Light_0001.fits")), None);
    }

    #[test]
    fn the_rejection_bitmaps_are_the_ones_the_master_was_built_with() {
        let dir = tempfile::tempdir().unwrap();
        let stems = vec!["a".to_string(), "b".to_string()];
        let set = RejBitmapSet::create(dir.path(), &stems, 8, 8, 1).unwrap();
        let first = rejection_paths(Some(&set), 2, false);
        assert_eq!(first, vec![Some(set.path(0).to_path_buf()), Some(set.path(1).to_path_buf())]);
        // With the second pass's own set complete, those are the bits that describe the master.
        let second = rejection_paths(Some(&set), 2, true);
        assert_eq!(second, vec![Some(set.second_pass_path(0)), Some(set.second_pass_path(1))]);
        assert_ne!(first, second);
        assert_eq!(rejection_paths(None, 2, true), vec![None, None]);
    }

    #[test]
    fn drizzle_is_skipped_when_the_second_pass_left_no_bitmap_set_of_its_own() {
        // Drizzle wants the bitmaps, the second pass ran, its set is not usable, and nothing else failed.
        assert!(second_pass_unavailable(true, true, false, None));
        // The second pass wrote its set, or never ran, or drizzle does not read bitmaps: nothing to skip.
        assert!(!second_pass_unavailable(true, true, true, None));
        assert!(!second_pass_unavailable(true, false, false, None));
        assert!(!second_pass_unavailable(false, true, false, None));
        // A failure already known is not said twice.
        assert!(!second_pass_unavailable(true, true, false, Some("failed to create the rejection-bitmap set")));
    }

    #[test]
    fn the_drizzle_cards_carry_the_scaled_block_and_the_slm_cards_and_no_ath_card() {
        let master = vec![
            Card::new("IMAGETYP", CardValue::Str("Master Light".into())).unwrap(),
            real("CRPIX1", 10.0),
            real("CD1_1", -2.0e-3),
            Card::new("SLM_CFG", CardValue::Str("0123456789abcdef".into())).unwrap(),
            Card::new("SLM_REF", CardValue::Str("c_Light_0000.fits".into())).unwrap(),
        ];
        let cfg = DrizzleConfig::default();
        let cards = drizzle_cards(&master, Settings { scale: 3, drop_shrink: 0.8 }, &cfg).unwrap();
        let keys: Vec<&str> = cards.iter().map(|c| c.keyword.as_str()).collect();
        // The block moves after the others; the drizzle's own cards close the header.
        assert_eq!(keys, ["IMAGETYP", "SLM_CFG", "SLM_REF", "CRPIX1", "CD1_1", "SLM_DRZ", "SLM_DRZP", "SLM_DRZK", "SLM_DZV"]);
        assert_eq!(value_of(&cards, "CRPIX1"), 29.0);
        assert_eq!(value_of(&cards, "SLM_DRZP"), 0.8);
        assert_eq!(value_of(&cards, "SLM_DRZ"), 3.0);
        assert!(cards.iter().all(|c| !c.keyword.starts_with("ATH_")));
        let weight = weight_map_cards(&cards).unwrap();
        let keys: Vec<&str> = weight.iter().map(|c| c.keyword.as_str()).collect();
        assert_eq!(keys, ["IMAGETYP", "BUNIT", "CRPIX1", "CD1_1", "SLM_CFG", "SLM_REF", "SLM_DRZ", "SLM_DZV"]);
    }
}
