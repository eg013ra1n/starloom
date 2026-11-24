//! Plane geometry for image registration: linear transform models, the
//! two distortion layers (a SIP-style polynomial and, since M4c, a
//! thin-plate spline), a KD-tree for correspondences and a deterministic
//! RANSAC. Pure math, no I/O, ungated. Conventions: 0-based pixel
//! coordinates, integer = pixel centre; `forward` maps subject →
//! reference, `inverse` maps reference → subject.

pub mod eigen;
pub mod kdtree;
pub mod linear;
pub mod pixel_map;
pub mod polynomial;
pub mod ransac;
pub mod tps;

pub use kdtree::KdTree2;
pub use linear::{fit_affine, Linear, LinearKind, Pair};
// `TpsGrid`/`TpsGrids` are deliberately NOT re-exported (fix round 1,
// minor 10): the displacement grids are an implementation detail of
// `pixel_map`'s spline arm, reached through
// `DistortionModel::{forward,inverse}_displacement`. Nothing outside
// `geometry` names them, and `geometry::pixel_map::TpsGrid` stays
// available for anything that ever needs to.
pub use pixel_map::{
    DisplacementField, DistortionModel, ForwardEval, InverseEval, InverseMap, PixelMap,
};
pub use polynomial::{Distortion, Polynomial2D};
pub use ransac::{ransac_fit, refit_weighted, Quality, RansacConfig, RansacResult, RefitResult};
pub use tps::{select_nodes, ThinPlateSpline, TPS_GRID_PX, TPS_MAX_NODES, TPS_MIN_NODES};

/// Pixel scale from the header's focal length and pixel size (M4b, ruling
/// R-M4b-1): `206.2648 * xpixsz_um / focallen_mm`, arcsec/px — the standard
/// plate-scale formula. `xpixsz_um` is trusted as already binned-effective
/// (an ASI294MM bin-2 light's `XPIXSZ` already reads twice the sensor's
/// native pixel size, the FITS convention every camera driver in the wild
/// follows) — there is deliberately NO separate binning factor here.
/// `None` when either input is missing, non-finite, or non-positive (a
/// zero/negative `focallen` would divide by zero or invert the sign,
/// neither a usable optical setup; a non-finite value is reachable from a
/// real header — the FITS parser passes a literal `1E999` through as `inf`
/// with no finiteness filter of its own, the catalog's FITS header reader).
pub fn header_pixel_scale_arcsec(xpixsz_um: Option<f64>, focallen_mm: Option<f64>) -> Option<f64> {
    let xpixsz = xpixsz_um?;
    let focallen = focallen_mm?;
    if !xpixsz.is_finite() || !focallen.is_finite() || xpixsz <= 0.0 || focallen <= 0.0 {
        return None;
    }
    Some(206.2648 * xpixsz / focallen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M4b, ruling R-M4b-1: `206.2648 * xpixsz_um / focallen_mm`, no binning
    /// factor — `XPIXSZ` is already binned-effective by FITS convention
    /// (an ASI294MM bin-2 light's `XPIXSZ` already reads twice the native
    /// pixel size). Values hand-computed to full precision so the test
    /// actually pins the constant, not just round-trips the formula.
    #[test]
    fn header_pixel_scale() {
        let a = header_pixel_scale_arcsec(Some(3.76), Some(1000.0)).unwrap();
        assert!((a - 0.775555648).abs() < 1e-9, "{a}");

        // An ASI294MM bin-2 light: XPIXSZ already effective at 4.63 um.
        let b = header_pixel_scale_arcsec(Some(4.63), Some(1000.0)).unwrap();
        assert!((b - 0.955006024).abs() < 1e-9, "{b}");

        assert_eq!(header_pixel_scale_arcsec(None, Some(1000.0)), None);
        assert_eq!(header_pixel_scale_arcsec(Some(3.76), None), None);
        assert_eq!(header_pixel_scale_arcsec(None, None), None);
        assert_eq!(header_pixel_scale_arcsec(Some(0.0), Some(1000.0)), None);
        assert_eq!(header_pixel_scale_arcsec(Some(-1.0), Some(1000.0)), None);
        assert_eq!(header_pixel_scale_arcsec(Some(3.76), Some(0.0)), None);
        assert_eq!(header_pixel_scale_arcsec(Some(3.76), Some(-1.0)), None);

        // Fix round 1: a non-finite input (reachable from a real header —
        // the FITS parser passes `1E999` through as `inf`) must not slip
        // past the `<= 0.0` guard, which NaN/inf both fail (`NaN <= 0.0` and
        // `inf <= 0.0` are both `false`).
        assert_eq!(
            header_pixel_scale_arcsec(Some(f64::NAN), Some(1000.0)),
            None
        );
        assert_eq!(
            header_pixel_scale_arcsec(Some(f64::INFINITY), Some(1000.0)),
            None
        );
        assert_eq!(
            header_pixel_scale_arcsec(Some(f64::NEG_INFINITY), Some(1000.0)),
            None
        );
        assert_eq!(header_pixel_scale_arcsec(Some(3.76), Some(f64::NAN)), None);
        assert_eq!(
            header_pixel_scale_arcsec(Some(3.76), Some(f64::INFINITY)),
            None
        );
        assert_eq!(
            header_pixel_scale_arcsec(Some(3.76), Some(f64::NEG_INFINITY)),
            None
        );
    }
}
