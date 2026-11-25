//! `weights.json`: the weights a command used.
//!
//! It is a report, never an input (P2-R21): `register`, `normalize` and `stack` recompute weights and the selection
//! from `measure.json` under their own config, then write what they used here.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{read_versioned, write_json_atomic};
use crate::stacking::measure::FrameMeasurement;
use crate::stacking::weights::{FormulaWeights, FrameWeight, SelectionConfig, WeightMode};

/// The file name.
pub const WEIGHTS_FILE: &str = "weights.json";
/// The `format` tag.
pub const WEIGHTS_FORMAT: &str = "starloom-weights";
/// The version. Another version is refused.
pub const WEIGHTS_VERSION: u32 = 1;

/// The settings the weights were made with: what a change of weight mode, formula, keyword or selection moves, and
/// the measurement hash does not hold (P2-R21).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightSettings {
    /// The weight mode.
    pub weight_mode: WeightMode,
    /// The formula weights, used by the formula mode.
    pub formula: FormulaWeights,
    /// The header keyword, used by the keyword mode.
    pub keyword: String,
    /// The frame selection.
    pub selection: SelectionConfig,
}

/// One frame's row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightFrame {
    /// The calibrated frame, absolute.
    pub file: PathBuf,
    /// The frame's measurement after the group's fold and reweigh: what Athenaeum stores in
    /// `stacking_run_frames.metrics_json`, with the same keys (per channel `commonTflux`, `commonTmeanFlux`,
    /// `commonStars` and `apertureCorrectionGroup`, each present only where it was set). `None` for a frame that was
    /// never measured (`included` is false and `reason` says why).
    pub measurement: Option<FrameMeasurement>,
    /// The reweighed weight; `None` for a frame with nothing to weigh (never measured, a non-finite plane) and, as in
    /// Athenaeum's frames table, for a frame the reweigh did not weigh.
    pub weight: Option<FrameWeight>,
    /// Whether the selection keeps the frame.
    pub included: bool,
    /// Why it does not, or a note on why it does.
    pub reason: Option<String>,
}

/// The weights a command used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightsReport {
    /// Always [`WEIGHTS_FORMAT`].
    pub format: String,
    /// Always [`WEIGHTS_VERSION`].
    pub version: u32,
    /// The settings the weights were made with.
    pub settings: WeightSettings,
    /// The frame-common star set, per plane: the star positions (group pixels) the PSF-family weight is read over, or
    /// `None` for a plane that was never reweighed.
    pub common_stars: Vec<Option<Vec<(f64, f64)>>>,
    /// One row per frame, in the pipeline's frame order.
    pub frames: Vec<WeightFrame>,
}

impl WeightsReport {
    /// An empty report for `settings`, to be filled with frames.
    pub fn new(settings: WeightSettings) -> WeightsReport {
        WeightsReport {
            format: WEIGHTS_FORMAT.to_string(),
            version: WEIGHTS_VERSION,
            settings,
            common_stars: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Reads a report; another version or another kind of file is refused.
    pub fn read(path: &Path) -> Result<WeightsReport> {
        read_versioned(path, WEIGHTS_FORMAT, WEIGHTS_VERSION)
    }

    /// Writes the report atomically to `path` (staged beside it, then renamed).
    pub fn write(&self, path: &Path) -> Result<()> {
        write_json_atomic(path, self)
    }
}
