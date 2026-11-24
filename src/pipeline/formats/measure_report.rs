//! `measure.json`: what `measure` found in every frame.
//!
//! The report stores the module's values **before** the group's fold (the floored planes' stand-in aperture
//! correction, P2-R21), and the fold is applied on read, as Athenaeum's run does (`measure_group_finish`), so a later
//! command recomputes weights and the selection from it under its own config. `weight` and `included` are what the
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
    /// The module's measurement, before the fold.
    pub measurement: FrameMeasurement,
    /// The weight `measure` computed.
    pub weight: FrameWeight,
    /// Whether `measure` kept the frame.
    pub included: bool,
    /// Why it did not, or a note on why it did.
    pub reason: Option<String>,
    /// Whether a plane of the frame is at the fit-call floor.
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
    /// The versions the numbers were made at, by name.
    pub versions: BTreeMap<String, u32>,
    /// The group's Moffat β under the two-pass `auto` model; `None` for a fixed profile.
    pub group_beta: Option<f64>,
    /// The aperture-correction inputs per plane.
    pub group_c: Vec<GroupC>,
    /// One row per frame, in the pipeline's frame order (P2-R10).
    pub frames: Vec<MeasureFrame>,
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
        }
    }

    /// Reads a report; another version or another kind of file is refused.
    pub fn read(path: &Path) -> Result<MeasureReport> {
        read_versioned(path, MEASURE_REPORT_FORMAT, MEASURE_REPORT_VERSION)
    }

    /// Writes the report atomically to `path` (staged in `<work_dir>/.tmp/`).
    pub fn write(&self, work_dir: &Path, path: &Path) -> Result<()> {
        write_json_atomic(work_dir, path, self)
    }
}
