//! The reader of a master flat's stamped normalization constants for files starloom's own pipeline writes (P2-R8): the
//! mirror of an application's reader of its own cards, line for line, with the `SLM_*` names.
//!
//! A flat made by `starloom master flat` carries the central-third mean of its plane ([`SLM_FNRM`]) and, for a CFA
//! flat, the three per-channel constants ([`SLM_FNR`], [`SLM_FNG`], [`SLM_FNB`]). The light-calibration math asks
//! [`SlmFlatCards`] for them before it reads the flat's pixels; a flat without them (or with unusable ones, or
//! measured under another CFA phase) gets its constants recomputed from its pixels.

use std::path::Path;

use astroimage::header::{header_key_map, header_text, HeaderFormat, HeaderKeyMap};

use crate::calibration_library::light_cal::FlatCardReader;
use crate::fits_writer::bayer::Bayer;
use crate::integration::cfa::CfaGeometry;

/// The central-third mean of a master flat's plane, in the plane's own unit: the global normalization constant.
pub const SLM_FNRM: &str = "SLM_FNRM";
/// The central-third mean of a master flat's red pixels (a CFA flat only).
pub const SLM_FNR: &str = "SLM_FNR";
/// The central-third mean of a master flat's green pixels (a CFA flat only).
pub const SLM_FNG: &str = "SLM_FNG";
/// The central-third mean of a master flat's blue pixels (a CFA flat only).
pub const SLM_FNB: &str = "SLM_FNB";
/// The divisor a calibrated light was scaled by (`scale_divisor_for_bitpix` of the raw light's bit depth).
pub const SLM_CSCL: &str = "SLM_CSCL";

/// Whether a file's name says its FITS is compressed (`.fz`, `.gz`): the shape Athenaeum's reader of its own cards does
/// not read (it reads the plain FITS header text only).
pub(crate) fn has_compressed_name(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("fz") || e.eq_ignore_ascii_case("gz"))
}

/// The header key map of a plain FITS file, or `None` when it cannot be read or is not plain FITS (every caller treats
/// that as "no card"). An XISF flat and a compressed one are not read: Athenaeum's reader of its own flat cards takes
/// the FITS header text only, so such a flat has its constants recomputed from its pixels there, and so here.
fn key_map(path: &Path) -> Option<HeaderKeyMap> {
    if has_compressed_name(path) {
        return None;
    }
    let (format, text) = header_text(path).ok()?;
    if format != HeaderFormat::Fits {
        return None;
    }
    Some(header_key_map(format, &text))
}

/// A card's value as a positive finite number, the only kind that can serve as a divisor.
fn divisor(map: &HeaderKeyMap, keyword: &str) -> Option<f64> {
    map.get(keyword).and_then(|s| s.trim().parse::<f64>().ok()).filter(|n| n.is_finite() && *n > 0.0)
}

/// `SLM_CSCL` of a calibrated light's key map: the divisor its pixels were scaled by, when the card is a number.
/// A gain is `EGAIN` times this (P2-R7).
pub fn slm_cscl(map: &HeaderKeyMap) -> Option<f64> {
    map.get(SLM_CSCL).and_then(|s| s.trim().parse::<f64>().ok())
}

/// Reads a master flat's [`SLM_FNRM`] and [`SLM_FNR`]/[`SLM_FNG`]/[`SLM_FNB`] cards. Handed to
/// [`resolve_flat_norm_divisor_with`](crate::calibration_library::light_cal::resolve_flat_norm_divisor_with), which
/// asks it where the math can use a stamped constant. Any failure (unreadable header, card absent, unparsable value)
/// is `None`, so the math recomputes from the flat's pixels.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlmFlatCards;

impl FlatCardReader for SlmFlatCards {
    /// The per-channel constants stamped on the flat, only when they were measured under the light's own mosaic phase.
    ///
    /// The three constants are the means of the flat's pixels grouped by the phase that was in force when they were
    /// measured, so that phase is recoverable from the same header. A missing `XBAYROFF`/`YBAYROFF` reads as 0: that
    /// is what the master build assumed when it measured them. A flat with the constants but no recognised `BAYERPAT`
    /// cannot be verified and counts as a mismatch. Offsets compare modulo 2 ([`CfaGeometry::same_phase`]).
    fn channel_norms(&self, flat: &Path, light: CfaGeometry) -> Option<[f64; 3]> {
        let map = key_map(flat)?;
        let k = [divisor(&map, SLM_FNR)?, divisor(&map, SLM_FNG)?, divisor(&map, SLM_FNB)?];
        let offset = |keyword: &str| -> i64 { map.get(keyword).and_then(|s| s.trim().parse::<i64>().ok()).unwrap_or(0) };
        let flat_geom = map.get("BAYERPAT").and_then(Bayer::parse).map(|pattern| CfaGeometry {
            pattern,
            xoff: offset("XBAYROFF"),
            yoff: offset("YBAYROFF"),
        });
        match flat_geom {
            Some(g) if g.same_phase(light) => Some(k),
            _ => {
                tracing::warn!(
                    path = %flat.display(),
                    flat_pattern = flat_geom.map(|g| g.pattern.as_str()).unwrap_or("none"),
                    light_pattern = light.pattern.as_str(),
                    "flat card geometry disagrees with light; per-channel constants recomputed from the flat plane"
                );
                None
            }
        }
    }

    /// The global constant stamped on the flat, when it can serve as a divisor (finite and positive).
    fn fnrm(&self, flat: &Path) -> Option<f64> {
        divisor(&key_map(flat)?, SLM_FNRM)
    }
}
