//! Calibration library: the parameter types (base) and the light-calibration math (`engine`) (Tasks 4 and 5).

pub mod options;
pub mod params;

/// The version of the light-calibration math. A change to the pixel arithmetic in `light_cal` or `cosmetic` bumps
/// it; a dependent application watches it with a tripwire test (spec SL-19).
pub const LIGHT_CAL_VERSION: u32 = 1;

/// Header keywords copied verbatim from the source LIGHT frame's own
/// on-disk cards. Categorized per spec §7 ("WCS cards, optics, DATE-OBS,
/// session cards, BAYERPAT/XBAYROFF/YBAYROFF"):
///
/// - WCS: the plate-solve solution, if the source carries one.
/// - Optics/acquisition: the equivalent of what `build_master_cards`
///   derives from its structured `MasterHeaderInputs` —
///   EXPTIME, GAIN, binning, pixel size, INSTRUME, TELESCOP, FOCALLEN,
///   FILTER, … — taken here directly from the raw source cards since a
///   calibrated light has no structured equivalent input.
/// - Session/target: OBJECT, coordinates, site, observer.
/// - Bayer/CFA geometry (mosaic preserved un-debayered per §2). ROWORDER
///   belongs to this group: the CFA pattern is only interpretable together
///   with the row direction it was declared in, so dropping it flips the
///   phase (and the colors) of every consumer that debayers the output.
///
/// Anything not listed here (IMAGETYP, CALSTAT, the caller's provenance cards) is either
/// replaced or freshly derived by the caller's card builder, never copied
/// blind from the source.
pub const COPY_THROUGH_KEYWORDS: &[&str] = &[
    // WCS solution
    "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CD1_1",
    "CD1_2", "CD2_1", "CD2_2", "CROTA1", "CROTA2", "CUNIT1", "CUNIT2", "WCSAXES", "EQUINOX",
    "RADESYS", "LONPOLE", "LATPOLE", "PLTSOLVD", // Optics / acquisition
    "EXPTIME", "GAIN", "OFFSET", "EGAIN", "XBINNING", "YBINNING", "XPIXSZ", "YPIXSZ", "CCD-TEMP",
    "SET-TEMP", "INSTRUME", "TELESCOP", "FOCALLEN", "APTDIA", "FILTER",
    // Observation timestamp
    "DATE-OBS", // Session / target
    "OBJECT", "OBJCTRA", "OBJCTDEC", "RA", "DEC", "SITELAT", "SITELONG", "SITEELEV", "OBSERVER",
    // Bayer / CFA geometry
    "BAYERPAT", "XBAYROFF", "YBAYROFF", "ROWORDER",
];
#[cfg(feature = "engine")]
pub mod cosmetic;
#[cfg(feature = "engine")]
pub mod debayer;
#[cfg(feature = "engine")]
pub mod flat_cards;
#[cfg(feature = "engine")]
pub mod light_cal;
