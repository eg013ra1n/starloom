//! The debayer glue shared by a calibrated-light generator and a stacking run's Prepare: the Bayer phase a debayer
//! must see, the VNG debayer of one CFA plane, the keywords a debayered file drops and the predicted CALSTAT string.
//! The `engine` layer (it reads `rustafits`' VNG).

use std::sync::Arc;

use astroimage::processing::vng::vng_debayer_f32;
use astroimage::BayerPattern;

use crate::fits_writer::bayer::Bayer;
use crate::fits_writer::Card;
use crate::integration::cfa::CfaGeometry;

/// Keywords a debayered output must not carry: they describe a mosaic that no
/// longer exists in the file. `ROWORDER` deliberately stays — it describes the
/// row direction of the DATA, which debayering does not change, and a consumer
/// still needs it to display the frame the right way up.
pub const MOSAIC_KEYWORDS: [&str; 3] = ["BAYERPAT", "XBAYROFF", "YBAYROFF"];

/// CALSTAT computed from which masters actually apply — must match the
/// engine's own construction: a dark subtraction removes bias and dark in one
/// go (`"BD"`), else a bias subtraction (`"B"`), plus `"F"` when a flat
/// divides. Predicted here because the header is built before the engine runs.
pub fn compute_calstat(dark: bool, bias: bool, flat: bool) -> String {
    let base = if dark {
        "BD"
    } else if bias {
        "B"
    } else {
        ""
    };
    if flat {
        format!("{base}F")
    } else {
        base.to_string()
    }
}

/// The mosaic as the debayer must see it: a pattern with the frame's
/// `XBAYROFF`/`YBAYROFF` phase already folded in.
///
/// The debayer takes a pattern and nothing else — it reads the tile at
/// `(x & 1, y & 1)`, i.e. it assumes phase zero. Our geometry carries offsets,
/// and an ODD offset is not a detail: shifting RGGB one column right lands G
/// where R was, which is the pattern GRBG. Ignoring the offsets would hand the
/// debayer a mosaic one step out of phase and swap red with blue.
///
/// Both shifts are their own inverse (shifting twice returns the original), so
/// each is written as a pairing rather than a table of sixteen entries; the
/// unit test cross-checks all four patterns against
/// [`crate::integration::cfa::cfa_channel_at`], the codebase's own definition
/// of where each colour sits.
/// `pub` since M4d Task 1: the stacking run's Bayer drizzle routes a mosaic
/// pixel to its colour's plane with the SAME phase-corrected pattern the
/// debayer was given, and must read it through this one function.
pub fn bayer_for(geom: CfaGeometry) -> BayerPattern {
    // A column shift swaps the two columns of the 2x2 tile...
    let shifted = if geom.xoff.rem_euclid(2) == 1 {
        match geom.pattern {
            Bayer::Rggb => Bayer::Grbg,
            Bayer::Grbg => Bayer::Rggb,
            Bayer::Bggr => Bayer::Gbrg,
            Bayer::Gbrg => Bayer::Bggr,
        }
    } else {
        geom.pattern
    };
    // ...and a row shift swaps its two rows.
    let shifted = if geom.yoff.rem_euclid(2) == 1 {
        match shifted {
            Bayer::Rggb => Bayer::Gbrg,
            Bayer::Gbrg => Bayer::Rggb,
            Bayer::Bggr => Bayer::Grbg,
            Bayer::Grbg => Bayer::Bggr,
        }
    } else {
        shifted
    };
    match shifted {
        Bayer::Rggb => BayerPattern::Rggb,
        Bayer::Bggr => BayerPattern::Bggr,
        Bayer::Gbrg => BayerPattern::Gbrg,
        Bayer::Grbg => BayerPattern::Grbg,
    }
}

/// The VNG debayer of one CFA plane, its `XBAYROFF`/`YBAYROFF` phase folded
/// in through [`bayer_for`] — the one debayer the generator and the stacking
/// run's Prepare share (plan 6b-2 C12). Full resolution, three planes,
/// planar RGB.
pub fn debayer_plane(
    data: &[f32],
    width: usize,
    height: usize,
    geom: CfaGeometry,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Vec<f32> {
    let pattern = bayer_for(geom);
    match pool {
        Some(p) => p.install(|| vng_debayer_f32(data, width, height, pattern)),
        None => vng_debayer_f32(data, width, height, pattern),
    }
}

/// Drops the cards a debayered file must not carry ([`MOSAIC_KEYWORDS`]): they describe a mosaic that no longer exists
/// in the file. The caller adds its own provenance card for the debayer after this.
pub fn drop_mosaic_cards(cards: &mut Vec<Card>) {
    cards.retain(|c| !MOSAIC_KEYWORDS.contains(&c.keyword.as_str()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fits_writer::CardValue;
    use crate::integration::cfa::cfa_channel_at;

    /// The pattern handed to the debayer must describe the SAME mosaic the
    /// rest of the codebase reads through [`cfa_channel_at`] — including the
    /// `XBAYROFF`/`YBAYROFF` phase, which the debayer has no parameter for.
    #[test]
    fn bayer_for_folds_offset_parity_into_the_pattern() {
        // The debayer's pattern back to ours, so the comparison below runs in
        // one vocabulary.
        fn back(p: BayerPattern) -> Bayer {
            match p {
                BayerPattern::Rggb => Bayer::Rggb,
                BayerPattern::Bggr => Bayer::Bggr,
                BayerPattern::Gbrg => Bayer::Gbrg,
                BayerPattern::Grbg => Bayer::Grbg,
                BayerPattern::None => panic!("a declared mosaic never maps to None"),
            }
        }
        for pattern in [Bayer::Rggb, Bayer::Bggr, Bayer::Gbrg, Bayer::Grbg] {
            // Both parities, and one even + one negative spelling of each, so
            // the modulo folding is covered too.
            for xoff in [0i64, 1, 2, -1] {
                for yoff in [0i64, 1, 2, -1] {
                    let geom = CfaGeometry {
                        pattern,
                        xoff,
                        yoff,
                    };
                    let phase_free = CfaGeometry {
                        pattern: back(bayer_for(geom)),
                        xoff: 0,
                        yoff: 0,
                    };
                    for y in 0..4 {
                        for x in 0..4 {
                            assert_eq!(
                                cfa_channel_at(x, y, geom),
                                cfa_channel_at(x, y, phase_free),
                                "{pattern:?} xoff {xoff} yoff {yoff} at ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn calstat_follows_the_masters_that_apply() {
        assert_eq!(compute_calstat(true, true, true), "BDF");
        assert_eq!(compute_calstat(true, false, false), "BD");
        assert_eq!(compute_calstat(false, true, true), "BF");
        assert_eq!(compute_calstat(false, true, false), "B");
        assert_eq!(compute_calstat(false, false, true), "F");
        assert_eq!(compute_calstat(false, false, false), "");
    }

    #[test]
    fn drop_mosaic_cards_removes_the_mosaic_and_keeps_the_row_order() {
        let mut cards = vec![
            Card::new("EXPTIME", CardValue::Real(180.0)).unwrap(),
            Card::new("BAYERPAT", CardValue::Str("RGGB".into())).unwrap(),
            Card::new("XBAYROFF", CardValue::Integer(1)).unwrap(),
            Card::new("YBAYROFF", CardValue::Integer(0)).unwrap(),
            Card::new("ROWORDER", CardValue::Str("TOP-DOWN".into())).unwrap(),
        ];
        drop_mosaic_cards(&mut cards);
        let names: Vec<&str> = cards.iter().map(|c| c.keyword.as_str()).collect();
        assert_eq!(names, ["EXPTIME", "ROWORDER"]);
    }
}
