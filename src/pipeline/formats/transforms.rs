//! `transforms.json`: where `register` put every frame.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Deserializer, Serialize};

use super::{read_versioned, write_json_atomic};
use crate::geometry::pixel_map::PixelMap;
use crate::stacking::config::AlgorithmConfig;

/// The file name.
pub const TRANSFORMS_FILE: &str = "transforms.json";
/// The `format` tag.
pub const TRANSFORMS_FORMAT: &str = "starloom-transforms";
/// The version. Another version is refused.
pub const TRANSFORMS_VERSION: u32 = 1;

/// A stored map is read the way [`PixelMap::from_json`] reads one: a malformed distortion layer or a singular linear
/// model is an error, and the inverse is recomputed from the linear model, never trusted.
fn checked_map<'de, D: Deserializer<'de>>(d: D) -> Result<Option<PixelMap>, D::Error> {
    let Some(map) = Option::<PixelMap>::deserialize(d)? else {
        return Ok(None);
    };
    if map.distortion.as_ref().is_some_and(|d| !d.is_well_formed()) {
        return Err(serde::de::Error::custom("malformed distortion layer in a stored map"));
    }
    let linear_inv = map
        .linear
        .inverse()
        .ok_or_else(|| serde::de::Error::custom("singular linear transform in a stored map"))?;
    Ok(Some(PixelMap {
        linear: map.linear,
        linear_inv,
        distortion: map.distortion,
    }))
}

/// One frame's registration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformFrame {
    /// The calibrated frame, absolute.
    pub file: PathBuf,
    /// Whether the frame registered.
    pub ok: bool,
    /// Why it did not.
    pub error: Option<String>,
    /// The model the frame ended with, as the registration reports it (the linear family, with its distortion layer).
    pub model: String,
    /// The map from the frame's pixels to the reference's; `None` when `ok` is false.
    #[serde(deserialize_with = "checked_map")]
    pub map: Option<PixelMap>,
    /// Refit inliers; 0 when `ok` is false (the numbers below are 0 or empty then).
    pub inliers: usize,
    /// Refit inliers over the final pairing.
    pub inlier_ratio: f64,
    /// RMS residual through the final map, in pixels.
    pub rms_px: f64,
    /// Largest absolute residual along x and along y, in pixels.
    pub peak_px: (f64, f64),
    /// The linear scale of the map.
    pub scale: f64,
    /// The rotation of the linear map, in degrees.
    pub rotation_deg: f64,
    /// Whether the map mirrors the image.
    pub flipped: bool,
    /// Convex-hull overlap of the inliers over all pairs, 0 to 1.
    pub overlap: f64,
    /// How much of the paired overlap the inliers cover, 0 to 1.
    pub coverage: f64,
    /// Warnings the registration raised for the frame.
    pub warnings: Vec<String>,
}

/// The `register` stage's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformsReport {
    /// Always [`TRANSFORMS_FORMAT`].
    pub format: String,
    /// Always [`TRANSFORMS_VERSION`].
    pub version: u32,
    /// The reference every map goes to: a frame of the stack, or a file outside it.
    pub reference: PathBuf,
    /// The effective config the stage ran with.
    pub config: AlgorithmConfig,
    /// The versions the maps were made at, by name.
    pub versions: BTreeMap<String, u32>,
    /// One row per frame, in the pipeline's frame order (P2-R10).
    pub frames: Vec<TransformFrame>,
}

impl TransformsReport {
    /// An empty report for `reference`, `config` and `versions`, to be filled with frames.
    pub fn new(reference: PathBuf, config: AlgorithmConfig, versions: BTreeMap<String, u32>) -> TransformsReport {
        TransformsReport {
            format: TRANSFORMS_FORMAT.to_string(),
            version: TRANSFORMS_VERSION,
            reference,
            config,
            versions,
            frames: Vec::new(),
        }
    }

    /// Reads a report; another version or another kind of file is refused.
    pub fn read(path: &Path) -> Result<TransformsReport> {
        read_versioned(path, TRANSFORMS_FORMAT, TRANSFORMS_VERSION)
    }

    /// Writes the report atomically to `path` (staged in `<work_dir>/.tmp/`).
    pub fn write(&self, work_dir: &Path, path: &Path) -> Result<()> {
        write_json_atomic(work_dir, path, self)
    }
}
