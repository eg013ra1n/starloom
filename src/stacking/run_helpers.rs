//! Pure helpers a stacking run is built from, cut out of the run so that a caller with another orchestration reads the
//! same rules: a registration map's scale, whether a group keeps the calibrated CFA mosaic, and the normalization
//! anchor pick.

use crate::calibration_library::options::CalibratedLightOptions;
use crate::geometry::PixelMap;
use crate::stacking::config::DrizzleConfig;
use crate::stacking::weights::{
    best_by_weight, sky_penalized_order, FrameWeight, MIN_REFERENCE_COVERAGE,
};

/// A registration map's linear scale at the frame centre: the square root of the Jacobian determinant of
/// `forward_exact`, by central differences over one pixel. `NaN` for a map that is not finite there.
pub fn map_scale(map: &PixelMap, width: usize, height: usize) -> f64 {
    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
    let (xp, yp) = map.forward_exact(cx + 1.0, cy);
    let (xm, ym) = map.forward_exact(cx - 1.0, cy);
    let (xq, yq) = map.forward_exact(cx, cy + 1.0);
    let (xn, yn) = map.forward_exact(cx, cy - 1.0);
    let (a, c) = ((xp - xm) / 2.0, (yp - ym) / 2.0);
    let (b, d) = ((xq - xn) / 2.0, (yq - yn) / 2.0);
    (a * d - b * c).abs().sqrt()
}

/// Whether a group's stage 1 also keeps the calibrated CFA mosaic beside each debayered frame (M4d Task 1, rulings
/// R-M4d-1/2): an OSC group (`osc`) whose drizzle is on, asked to deposit each colour's own samples, AND whose lights
/// are actually being debayered. A mono group ignores `drizzle.bayer` silently: there is no mosaic to keep and nothing
/// about the run changes.
///
/// `calibration.debayer_osc` is load-bearing, not belt-and-braces: a generator only writes a mosaic for a frame it is
/// DEBAYERING, because an OSC frame with the debayer off already writes the mosaic itself as its primary output under
/// that very name. Without this term, a run with `bayer` on and the debayer off would want a mosaic no generation can
/// produce, find the artifact missing every time, and recalibrate the whole group on every run for ever.
///
/// This is THE one place the condition is spelled out: the cache gate, the calibration stage and the drizzle stage all
/// read it, so they cannot disagree about whether a mosaic should exist. The per-FRAME half of the debayer decision (a
/// `BAYERPAT` the caller cannot parse) is deliberately not folded in here.
pub fn wants_cfa_mosaic(
    drizzle: &DrizzleConfig,
    calibration: &CalibratedLightOptions,
    osc: bool,
) -> bool {
    drizzle.enabled && drizzle.bayer && calibration.debayer_osc && osc
}

/// The result of one [`pick_ln_anchor`] call (ruling R-M3-17 v2 + its coverage addendum, fix round 1 Important 3):
/// `idx` is always a valid index into the slices the pick was called with: a group with `>= 3` members always gets an
/// anchor, restoring the guarantee [`best_by_weight`] gives a non-empty candidate set. `coverage_dropped` and
/// `sky_fallback` are `Some(count)` when the pick had to relax ITS OWN filter to reach that guarantee (fewer than 3
/// members covered the reference geometry; or every covering member's background was non-finite or non-positive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorPick {
    /// The anchor's index into the candidate slices.
    pub idx: usize,
    /// `Some(n)` when only `n < 3` members passed the coverage filter and it was dropped.
    pub coverage_dropped: Option<usize>,
    /// `Some(n)` when no admissible member had a usable background and the anchor is the best by weight among `n`.
    pub sky_fallback: Option<usize>,
}

/// The normalization anchor of a group (ruling R-M3-17 v2): among the members whose registered footprint covers the
/// reference rectangle (`coverage[i] >= MIN_REFERENCE_COVERAGE`), the one [`sky_penalized_order`] ranks first. The four
/// slices are parallel, one entry per member: the frame weights, the minimum star counts, the mean background of each
/// frame and the covered fraction of the reference rectangle ([`crate::stacking::weights::reference_coverage`]).
///
/// The coverage filter is dropped for a group where fewer than 3 members would pass it, rather than ranking among too
/// few candidates to mean anything. When the ranking comes back empty (every admissible background non-finite or
/// non-positive), the anchor falls back to [`best_by_weight`] over the same mask.
///
/// Returns the pick (`None` when no member can be picked) and the number of members that passed the coverage filter
/// when it was dropped (also in the pick's `coverage_dropped`, and returned here so a caller that gets no pick can
/// still report it). Nothing is logged: the caller reports both relaxations.
pub fn pick_ln_anchor(
    weights: &[FrameWeight],
    star_counts: &[usize],
    sky: &[f64],
    coverage: &[f64],
) -> (Option<AnchorPick>, Option<usize>) {
    let coverage_mask: Vec<bool> = coverage
        .iter()
        .map(|&c| c >= MIN_REFERENCE_COVERAGE)
        .collect();
    let admissible_count = coverage_mask.iter().filter(|&&ok| ok).count();
    let (included_mask, coverage_dropped) = if admissible_count >= 3 {
        (coverage_mask, None)
    } else {
        (vec![true; weights.len()], Some(admissible_count))
    };

    // `sky_penalized_order` returning empty (every admissible member's background non-finite or non-positive, e.g. a
    // narrowband set whose dark slightly over-subtracts every frame) must not fail the whole group: a group with
    // `>= 3` members always gets an anchor, as `best_by_weight` gave before the sky penalty existed. It runs over the
    // SAME `included_mask` (still coverage-filtered) the sky-penalized rank used.
    let ordered = sky_penalized_order(weights, &included_mask, star_counts, sky);
    let (idx, sky_fallback) = match ordered.into_iter().next() {
        Some(idx) => (idx, None),
        None => {
            let count = included_mask.iter().filter(|&&ok| ok).count();
            match best_by_weight(weights, &included_mask, star_counts) {
                Some(idx) => (idx, Some(count)),
                None => return (None, coverage_dropped),
            }
        }
    };
    (
        Some(AnchorPick {
            idx,
            coverage_dropped,
            sky_fallback,
        }),
        coverage_dropped,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Linear, LinearKind};

    fn fw(normalized_mean: f64) -> FrameWeight {
        FrameWeight {
            channels: Vec::new(),
            normalized: Vec::new(),
            mean: normalized_mean,
            normalized_mean,
            missing: None,
        }
    }

    #[test]
    fn map_scale_of_identity_and_a_scaled_map() {
        let identity = PixelMap::linear(Linear::identity()).unwrap();
        assert!((map_scale(&identity, 384, 288) - 1.0).abs() < 1e-12);
        let scaled = PixelMap::linear(Linear::from_flat(
            LinearKind::Similarity,
            [1.2, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0],
        ))
        .unwrap();
        assert!((map_scale(&scaled, 384, 288) - 1.2).abs() < 1e-9);
    }

    #[test]
    fn the_cfa_mosaic_is_wanted_only_for_an_osc_drizzle_that_debayers() {
        let mut drizzle = DrizzleConfig::default();
        let mut calibration = CalibratedLightOptions::default();
        assert!(!wants_cfa_mosaic(&drizzle, &calibration, true), "drizzle is off");
        drizzle.enabled = true;
        assert!(!wants_cfa_mosaic(&drizzle, &calibration, true), "bayer drizzle is off");
        drizzle.bayer = true;
        assert!(wants_cfa_mosaic(&drizzle, &calibration, true));
        assert!(!wants_cfa_mosaic(&drizzle, &calibration, false), "a mono group keeps no mosaic");
        calibration.debayer_osc = false;
        assert!(!wants_cfa_mosaic(&drizzle, &calibration, true), "no debayer, no mosaic beside it");
    }

    #[test]
    fn the_anchor_skips_a_poorly_covering_top_scorer() {
        let weights = vec![fw(1.0), fw(0.9), fw(0.6), fw(0.5)];
        let stars = [100usize; 4];
        let sky = [1.0, 4.0, 4.0, 4.0];
        let coverage = [0.5, 1.0, 1.0, 1.0];
        let (pick, dropped) = pick_ln_anchor(&weights, &stars, &sky, &coverage);
        assert_eq!(dropped, None);
        assert_eq!(
            pick,
            Some(AnchorPick { idx: 1, coverage_dropped: None, sky_fallback: None })
        );
    }

    #[test]
    fn the_coverage_filter_is_dropped_below_three_admissible_members() {
        let weights = vec![fw(1.0), fw(0.9), fw(0.6)];
        let stars = [100usize; 3];
        let sky = [1.0, 4.0, 4.0];
        let coverage = [0.5, 1.0, 0.5];
        let (pick, dropped) = pick_ln_anchor(&weights, &stars, &sky, &coverage);
        assert_eq!(dropped, Some(1));
        let pick = pick.expect("a pick");
        assert_eq!(pick.idx, 0, "the filter is off, so the best sky-penalized score wins");
        assert_eq!(pick.coverage_dropped, Some(1));
        assert_eq!(pick.sky_fallback, None);
    }

    #[test]
    fn an_unusable_background_everywhere_falls_back_to_the_best_weight() {
        let weights = vec![fw(0.4), fw(0.9), fw(0.6)];
        let stars = [100usize; 3];
        let sky = [0.0, f64::NAN, -1.0];
        let coverage = [1.0, 1.0, 1.0];
        let (pick, dropped) = pick_ln_anchor(&weights, &stars, &sky, &coverage);
        assert_eq!(dropped, None);
        let pick = pick.expect("a pick");
        assert_eq!((pick.idx, pick.sky_fallback), (1, Some(3)));
    }

    #[test]
    fn no_member_means_no_pick() {
        assert_eq!(pick_ln_anchor(&[], &[], &[], &[]), (None, Some(0)));
    }
}
