//! The per-run options of a calibrated light: how a run chooses to calibrate its lights (flat normalization, hot-pixel
//! correction, debayer and the output container). Plain configuration, ungated; the calibration math that reads it is
//! the `engine` layer (`light_cal`).

use serde::{Deserialize, Serialize};

use super::params::{FlatNormMode, LightCalParams};
use crate::fits_writer::OutputFormat;

/// serde default for the three ON-by-default toggles below — the value
/// `#[serde(default = "…")]` needs a named function for. `LightCalParams`
/// carries the same helper, for the same reason.
fn default_true() -> bool {
    true
}

/// Everything a run chooses about how its lights are calibrated. One value per
/// run (an export, a transfer preparation), shared by every frame in it.
///
/// `hot_pixel_correction` and `debayer_osc` are the two stages the generator
/// adds on top of the calibration formula; both default ON, and both degrade
/// silently to "not applicable" rather than failing — a frame with no dark
/// master gets no cosmetic pass, a mono frame is never debayered.
///
/// **Every field is optional on the wire** (`#[serde(default)]`, mirroring
/// [`LightCalParams`]), each defaulting to the recommended
/// behavior. A host command that knows only some of these — or a payload
/// written before a field existed — decodes the rest to
/// [`CalibratedLightOptions::default`] instead of failing the whole request;
/// `{}` is a valid, fully-defaulted payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct CalibratedLightOptions {
    /// Normalize the master flat by its own level before dividing (spec §2).
    #[serde(default = "default_true")]
    pub flat_norm: bool,
    /// Which statistic computes that normalization constant. Plain
    /// `#[serde(default)]` resolves through [`FlatNormMode::default`]
    /// (`CentralThird`), so this tracks the enum's own default instead of
    /// restating it here.
    #[serde(default)]
    pub flat_norm_mode: FlatNormMode,
    /// Advanced per-run parameters (pedestal, trim fraction, bias fallback,
    /// per-CFA-channel flat scaling). Omitting it wholesale is the same as
    /// sending `{}` — every one of ITS fields defaults too.
    #[serde(default)]
    pub params: LightCalParams,
    /// Replace the master dark's hot pixels with a neighbourhood median.
    #[serde(default = "default_true")]
    pub hot_pixel_correction: bool,
    /// Debayer a CFA light to full-resolution planar RGB. Ignored for mono
    /// frames and for a `BAYERPAT` the catalog cannot vouch for.
    #[serde(default = "default_true")]
    pub debayer_osc: bool,
    /// Also keep the calibrated CFA mosaic beside the debayered output (M4d
    /// Task 1, ruling R-M4d-1) — the stacking run sets this for an OSC group
    /// whose drizzle deposits each colour's own samples, and nothing else
    /// does.
    ///
    /// **Run-internal, never on the wire** (`#[serde(skip)]`, which also
    /// keeps it out of the generated TypeScript): a host command has no say
    /// in it — the artifact is an implementation detail of the pipeline that
    /// asked for it, not an export option a user picks. A payload written by
    /// any host therefore decodes it as `false`, and an options value
    /// serialized by this build round-trips to `false` too.
    #[serde(skip)]
    pub keep_mosaic: bool,
    /// The container a calibrated light is written in on export and send.
    /// The stacking run ignores it for its own intermediates (they stay
    /// FITS — `PlaneReader` reads nothing else), and says so in stage 1.
    #[serde(default)]
    pub format: OutputFormat,
    /// Skip the fsync before the write's rename (perf tier 1 Task 6). The
    /// stacking run sets this for its own calibrated intermediates — the
    /// `stacking_artifacts` row keys on a config hash plus (size, mtime),
    /// so a file a crash loses or truncates is a cache miss on the next
    /// run. Not a claim that such a file can never read back as a WRONG
    /// answer in principle: on a filesystem without ordered metadata a
    /// rename that landed before its data was flushed can, in theory,
    /// reappear at the recorded size with garbage inside it — APFS and
    /// ext4 (`data=ordered`), what this app ships on, make that
    /// unreachable. Nothing else sets this field.
    ///
    /// **Run-internal, never on the wire** (`#[serde(skip)]`, same reason as
    /// `keep_mosaic` above): a host command has no say in it, so a payload
    /// written by any host decodes it as `false` (durable) and an options
    /// value serialized by this build round-trips to `false` too.
    #[serde(skip)]
    pub skip_fsync: bool,
}

impl Default for CalibratedLightOptions {
    fn default() -> Self {
        Self {
            flat_norm: true,
            flat_norm_mode: FlatNormMode::CentralThird,
            params: LightCalParams::default(),
            hot_pixel_correction: true,
            debayer_osc: true,
            keep_mosaic: false,
            format: OutputFormat::Fits,
            skip_fsync: false,
        }
    }
}

impl CalibratedLightOptions {
    /// Resolve the six per-field host arguments, each optional: an absent
    /// (or `null`) one takes this type's own default, so neither backend
    /// restates the defaults — the export commands and the summary preview on
    /// both hosts resolve the same six fields through this one place.
    pub fn resolve(
        flat_norm: Option<bool>,
        flat_norm_mode: Option<FlatNormMode>,
        params: Option<LightCalParams>,
        hot_pixel: Option<bool>,
        debayer: Option<bool>,
        format: Option<OutputFormat>,
    ) -> Self {
        let d = Self::default();
        Self {
            flat_norm: flat_norm.unwrap_or(d.flat_norm),
            flat_norm_mode: flat_norm_mode.unwrap_or(d.flat_norm_mode),
            params: params.unwrap_or(d.params),
            hot_pixel_correction: hot_pixel.unwrap_or(d.hot_pixel_correction),
            debayer_osc: debayer.unwrap_or(d.debayer_osc),
            // Not a host argument (see the field's own doc): the run that
            // wants the mosaic sets it on the resolved options itself.
            keep_mosaic: d.keep_mosaic,
            format: format.unwrap_or(d.format),
            // Not a host argument either (see the field's own doc): the
            // stacking run sets it on the resolved options itself.
            skip_fsync: d.skip_fsync,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field is optional on the wire: a host that knows only some of
    /// these - or a payload written before a field existed - must decode, not
    /// fail. `{}` is the extreme case and has to equal the documented default.
    #[test]
    fn options_decode_from_a_partial_payload() {
        let empty: CalibratedLightOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, CalibratedLightOptions::default());

        let partial: CalibratedLightOptions =
            serde_json::from_str(r#"{"debayerOsc": false}"#).unwrap();
        assert!(!partial.debayer_osc, "the sent field wins");
        assert!(partial.flat_norm, "omitted flat_norm defaults ON");
        assert!(
            partial.hot_pixel_correction,
            "omitted hot_pixel_correction defaults ON"
        );
        assert_eq!(partial.flat_norm_mode, FlatNormMode::CentralThird);
        // An omitted `params` defaults wholesale, its own fields included.
        assert_eq!(partial.params, LightCalParams::default());
        assert_eq!(
            partial,
            CalibratedLightOptions {
                debayer_osc: false,
                ..CalibratedLightOptions::default()
            }
        );

        // The camelCase spelling is the wire contract; round-tripping our own
        // serialization must land back on the same value.
        let round: CalibratedLightOptions =
            serde_json::from_str(&serde_json::to_string(&partial).unwrap()).unwrap();
        assert_eq!(round, partial);
    }

    #[test]
    fn options_format_defaults_to_fits_and_decodes_old_documents() {
        let old: CalibratedLightOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(old.format, OutputFormat::Fits);
        let r =
            CalibratedLightOptions::resolve(None, None, None, None, None, Some(OutputFormat::Xisf));
        assert_eq!(r.format, OutputFormat::Xisf);
    }
}
