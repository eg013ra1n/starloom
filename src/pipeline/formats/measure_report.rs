//! `measure.json`: what `measure` found in every frame.
//!
//! The report stores the module's values **before** the group's fold (the floored planes' stand-in aperture
//! correction, P2-R21). The fold is not applied by [`MeasureReport::read`]: [`finish`](crate::pipeline::finish) applies
//! it, as Athenaeum's run does on every run (`measure_group_finish`), so a later command recomputes the fold, the
//! weights and the selection from the stored values under its own config. `weight` and `included` are what the
//! `measure` command computed; they are a record, never an input.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{read_versioned, write_json_atomic};
use crate::stacking::config::AlgorithmConfig;
use crate::stacking::measure::FrameMeasurement;
use crate::stacking::weights::FrameWeight;

/// The file name, under the working folder's `measure/` folder.
pub const MEASURE_REPORT_FILE: &str = "measure.json";
/// The `format` tag.
pub const MEASURE_REPORT_FORMAT: &str = "starloom-measure";
/// The version. Another version is refused.
pub const MEASURE_REPORT_VERSION: u32 = 1;
/// The columns of `measure.csv`, in order: one row per frame and channel, with the folded values.
pub const MEASURE_CSV_COLUMNS: [&str; 18] = [
    "file",
    "channel",
    "fwhm_px",
    "fwhm_arcsec",
    "eccentricity",
    "stars_detected",
    "stars_fitted",
    "snr_weight",
    "psf_signal_weight",
    "psf_snr",
    "noise",
    "median",
    "mad",
    "location",
    "scale",
    "weight",
    "included",
    "reason",
];

/// One colour channel's aperture-correction inputs over the group, the numbers the group's fold reads
/// ([`channel_c_summary`](crate::stacking::measure::channel_c_summary)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroupC {
    /// The plane (colour channel).
    pub plane: usize,
    /// The frames whose plane carries a measured aperture correction.
    pub planes: usize,
    /// The frames whose plane is at the fit-call floor, with none.
    pub count: usize,
    /// The median of the measured corrections; `None` when none was measured.
    pub median: Option<f64>,
    /// The lowest measured correction.
    pub min: Option<f64>,
    /// The highest measured correction.
    pub max: Option<f64>,
}

/// One frame's row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeasureFrame {
    /// The calibrated frame, absolute.
    pub file: PathBuf,
    /// The module's measurement, before the fold; `None` for a frame that was never measured (it was unreadable or its
    /// measurement failed: `reason` says which).
    pub measurement: Option<FrameMeasurement>,
    /// The weight `measure` computed; `None` for a frame with no measurement and for one whose measurement is not
    /// finite (nothing to weigh).
    pub weight: Option<FrameWeight>,
    /// Whether `measure` kept the frame.
    pub included: bool,
    /// Why it did not, or a note on why it did.
    pub reason: Option<String>,
    /// Whether the weight floor the PSF weight modes defer (Athenaeum's PD-18) will exclude the frame: its weight at
    /// `measure` time is below `selection.min_weight_fraction` of the group's best, and the frame is kept for now
    /// because registration judges the floor on the weight over the stars the members have in common. The reference
    /// pick must not choose such a frame. Always `false` under the other weight modes (they exclude at `measure` time)
    /// and for a frame that is not included.
    pub measure_floored: bool,
}

/// The `measure` stage's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeasureReport {
    /// Always [`MEASURE_REPORT_FORMAT`].
    pub format: String,
    /// Always [`MEASURE_REPORT_VERSION`].
    pub version: u32,
    /// The effective config the stage ran with.
    pub config: AlgorithmConfig,
    /// Every version of the build that made the report, by constant name
    /// ([`all_versions`](crate::pipeline::versions::all_versions)).
    pub versions: BTreeMap<String, u32>,
    /// The group's Moffat β under the two-pass `auto` model; `None` for a fixed profile (the profile's own β is the
    /// config's: [`MeasureReport::resolved_group_beta`] reads both).
    pub group_beta: Option<f64>,
    /// The aperture-correction inputs per plane.
    pub group_c: Vec<GroupC>,
    /// One row per frame, in the pipeline's frame order (P2-R10).
    pub frames: Vec<MeasureFrame>,
    /// Whether the group's aperture-correction fold is already applied to the measurements: `false` in everything
    /// `measure` stores and returns, `true` in what [`finish`](crate::pipeline::finish) returns. `finish` folds a
    /// report once and leaves a folded one as it is (a second fold would scale the substituted planes again).
    #[serde(default, skip_serializing_if = "is_false")]
    pub folded: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl MeasureReport {
    /// An empty report for `config` and `versions`, to be filled with frames.
    pub fn new(config: AlgorithmConfig, versions: BTreeMap<String, u32>) -> MeasureReport {
        MeasureReport {
            format: MEASURE_REPORT_FORMAT.to_string(),
            version: MEASURE_REPORT_VERSION,
            config,
            versions,
            group_beta: None,
            group_c: Vec::new(),
            frames: Vec::new(),
            folded: false,
        }
    }

    /// The β every member of the group was fitted at, as Athenaeum's `measure_group_finish` resolves it: the two-pass
    /// group β under `auto`, else the lower median of the measured planes' own β (a fixed profile fits every plane at
    /// its β). `None` when no plane was measured.
    pub fn resolved_group_beta(&self) -> Option<f64> {
        if let Some(beta) = self.group_beta {
            return Some(beta);
        }
        let betas: Vec<f64> = self
            .frames
            .iter()
            .filter_map(|f| f.measurement.as_ref())
            .flat_map(|m| m.channels.iter().map(|c| c.beta))
            .collect();
        (!betas.is_empty()).then(|| crate::stacking::psf_signal::group_beta(&betas))
    }

    /// Reads a report; another version or another kind of file is refused.
    pub fn read(path: &Path) -> Result<MeasureReport> {
        read_versioned(path, MEASURE_REPORT_FORMAT, MEASURE_REPORT_VERSION)
    }

    /// Writes the report atomically to `path` (staged beside it, then renamed).
    pub fn write(&self, path: &Path) -> Result<()> {
        write_json_atomic(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plan's column list (Task 8, `measure.csv`), spelled out.
    #[test]
    fn measure_csv_has_the_plan_s_eighteen_columns() {
        let plan = "file, channel, fwhm_px, fwhm_arcsec, eccentricity, stars_detected, stars_fitted, snr_weight, \
                    psf_signal_weight, psf_snr, noise, median, mad, location, scale, weight, included, reason";
        let listed: Vec<&str> = plan.split(',').map(str::trim).collect();
        assert_eq!(MEASURE_CSV_COLUMNS.as_slice(), listed.as_slice());
    }
}
