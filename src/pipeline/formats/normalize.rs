//! `normalize.json`: what `normalize` did, for the user to read.
//!
//! Nothing reads it back to compare (P2-R24). It records what Athenaeum's local-normalization hashes fold: the anchor,
//! the frames the LN reference was built from (ranked as the stage ranked them), the frames the seeds calibration was
//! measured over and its factor `k` per plane, plus the β the reference was fitted at and the LN scale. A later command
//! finds a frame's `.athln` by its name ([`athln_path`](crate::pipeline::athln_path)).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{read_versioned, write_json_atomic};
use crate::stacking::config::AlgorithmConfig;

/// The file name.
pub const NORMALIZE_FILE: &str = "normalize.json";
/// The `format` tag.
pub const NORMALIZE_FORMAT: &str = "starloom-normalize";
/// The version. Another version is refused.
pub const NORMALIZE_REPORT_VERSION: u32 = 1;

/// What the stage did with the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NormalizeStatus {
    /// The LN reference was built and every member was tried: the products are in `ln/`.
    Ran,
    /// Local normalization is not asked for (`normalization.local.enabled` is off and `normalization.rejection` is not
    /// `local`): the stage did nothing and left no entry.
    NotRequested,
    /// Local normalization was asked for and could not run for the group (fewer than 3 frames to build the reference
    /// from, or the reference's background model was refused); `note` says why. The stage left no entry, and the stack
    /// would use the global normalization, as Athenaeum's run does.
    Off,
}

/// What became of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FrameStatus {
    /// Its `.athln` was written and read back.
    Normalized,
    /// It overlaps the LN reference on too little of its footprint and keeps the global normalization (no `.athln`).
    GlobalLowOverlap,
    /// Its own normalization failed while local normalization only drives rejection: it keeps the global one (no
    /// `.athln`).
    KeptGlobal,
    /// Its own normalization failed while local normalization drives the output: the frame is left out of the stack
    /// (no `.athln`).
    Excluded,
}

/// One frame's row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizeFrame {
    /// The calibrated frame, absolute.
    pub file: PathBuf,
    /// What became of it.
    pub status: FrameStatus,
    /// Its sidecar (`<work>/ln/<stem>.athln`) when [`FrameStatus::Normalized`].
    pub athln: Option<PathBuf>,
    /// Its relative scale, the mean over planes, when normalized.
    pub scale: Option<f64>,
    /// The star pairs the scale rests on (summed over planes), when normalized.
    pub matches: Option<usize>,
    /// `seeds` (Measure's fits seeded the scale) or `detected` (at least one plane fell back to detection), when
    /// normalized.
    pub scale_source: Option<String>,
    /// The overlap with the LN reference, 0 to 1, for [`FrameStatus::GlobalLowOverlap`].
    pub overlap: Option<f64>,
    /// Why, for [`FrameStatus::KeptGlobal`] and [`FrameStatus::Excluded`].
    pub reason: Option<String>,
}

/// The `normalize` stage's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizeReport {
    /// Always [`NORMALIZE_FORMAT`].
    pub format: String,
    /// Always [`NORMALIZE_REPORT_VERSION`].
    pub version: u32,
    /// What the stage did.
    pub status: NormalizeStatus,
    /// Why the stage did nothing, for [`NormalizeStatus::NotRequested`] and [`NormalizeStatus::Off`].
    pub note: Option<String>,
    /// The effective config the stage ran with.
    pub config: AlgorithmConfig,
    /// Every version of the build that made the report, by constant name
    /// ([`all_versions`](crate::pipeline::versions::all_versions)).
    pub versions: BTreeMap<String, u32>,
    /// Whether local normalization drives the output (`normalization.local.enabled`), so that a frame whose own
    /// normalization failed is excluded, rather than driving rejection only.
    pub local_drives_output: bool,
    /// The normalization anchor the LN reference was built against: the member the sky-penalized ranking puts first
    /// among those that cover the reference rectangle.
    pub anchor: Option<PathBuf>,
    /// The anchor over the frames that were left after the exclusions, when any frame was excluded.
    pub anchor_after_exclusions: Option<PathBuf>,
    /// The frames the LN reference was integrated from, in the order the stage ranked them (`referenceFrames` of the
    /// best, sky-penalized).
    pub reference_members: Vec<PathBuf>,
    /// The frames the seeds calibration was measured over, the registration reference first.
    pub calibration_frames: Vec<PathBuf>,
    /// The seeds calibration's factor per plane (1 for a plane that was not calibrated); empty when none was measured.
    pub k: Vec<f64>,
    /// The β the LN reference was fitted at: the group β under `auto`, else the profile's own.
    pub reference_beta: Option<f64>,
    /// The LN scale the grids were made at, in pixels.
    pub scale_px: Option<u32>,
    /// The LN reference (`<work>/ln/reference.fits`).
    pub reference: Option<PathBuf>,
    /// The reference-frames saturation map (`<work>/ln/reference-saturation.fits`).
    pub reference_saturation: Option<PathBuf>,
    /// One row per member of the stack, in the pipeline's frame order (P2-R10); a frame that was not a member (it was
    /// out of the selection, or has no transform) has none.
    pub frames: Vec<NormalizeFrame>,
}

impl NormalizeReport {
    /// A report of a stage that did nothing, with `status` and `note`.
    pub fn idle(status: NormalizeStatus, note: String, config: AlgorithmConfig, versions: BTreeMap<String, u32>) -> NormalizeReport {
        NormalizeReport {
            format: NORMALIZE_FORMAT.to_string(),
            version: NORMALIZE_REPORT_VERSION,
            status,
            note: Some(note),
            local_drives_output: config.normalization.local.enabled,
            config,
            versions,
            anchor: None,
            anchor_after_exclusions: None,
            reference_members: Vec::new(),
            calibration_frames: Vec::new(),
            k: Vec::new(),
            reference_beta: None,
            scale_px: None,
            reference: None,
            reference_saturation: None,
            frames: Vec::new(),
        }
    }

    /// Reads a report; another version or another kind of file is refused.
    pub fn read(path: &Path) -> Result<NormalizeReport> {
        read_versioned(path, NORMALIZE_FORMAT, NORMALIZE_REPORT_VERSION)
    }

    /// Writes the report atomically to `path` (staged beside it, then renamed).
    pub fn write(&self, path: &Path) -> Result<()> {
        write_json_atomic(path, self)
    }
}
