//! The files a working folder holds, each tagged and versioned.
//!
//! Every format has a `format` tag and a `version`. A reader refuses a file of another version, naming both, so a
//! working folder made by another starloom is never half-understood. The files are JSON and are written atomically:
//! in `<work>/.tmp/`, then renamed into place.

pub mod manifest;
pub mod measure_report;
pub mod transforms;
pub mod weights;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use manifest::{
    append_history, begin_writing, check_upstream, commit_stage, frames_or_upstream, Entry, FileId, HistoryRow, Manifest,
    MANIFEST_FILE, MANIFEST_FORMAT, MANIFEST_VERSION, TMP_DIR,
};
pub use measure_report::{
    GroupC, MeasureFrame, MeasureReport, MEASURE_REPORT_FILE, MEASURE_REPORT_FORMAT, MEASURE_REPORT_VERSION,
};
pub use transforms::{TransformFrame, TransformsReport, TRANSFORMS_FILE, TRANSFORMS_FORMAT, TRANSFORMS_VERSION};
pub use weights::{WeightFrame, WeightSettings, WeightsReport, WEIGHTS_FILE, WEIGHTS_FORMAT, WEIGHTS_VERSION};

/// The two fields every format starts with, read first so a file of another kind or version is refused before its
/// body is parsed.
#[derive(Deserialize)]
struct Head {
    format: Option<String>,
    version: Option<u64>,
}

/// Reads a tagged, versioned JSON file. It refuses a file that is not `tag` or whose version is not `version`
/// ("made by another starloom: <tag> version N, this build reads M").
pub(crate) fn read_versioned<T: DeserializeOwned>(path: &Path, tag: &str, version: u32) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let head: Head = serde_json::from_str(&text).with_context(|| format!("{} is not a starloom file", path.display()))?;
    if head.format.as_deref() != Some(tag) {
        bail!("{} is not a {tag} file (format {:?})", path.display(), head.format.as_deref().unwrap_or("none"));
    }
    match head.version {
        Some(v) if v == u64::from(version) => {}
        found => bail!(
            "{}: made by another starloom: {tag} version {}, this build reads {version}",
            path.display(),
            found.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
    }
    serde_json::from_str(&text).with_context(|| format!("{} does not read as a {tag} file", path.display()))
}

/// Writes `value` as pretty JSON to `path`: into `<work_dir>/.tmp/`, synced, then renamed into place, so a reader
/// sees the old file or the whole new one. Creates the folders it needs.
pub(crate) fn write_json_atomic<T: Serialize>(work_dir: &Path, path: &Path, value: &T) -> Result<()> {
    let tmp_dir = work_dir.join(TMP_DIR);
    std::fs::create_dir_all(&tmp_dir).with_context(|| format!("cannot create {}", tmp_dir.display()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let name = path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or("file");
    let tmp: PathBuf = tmp_dir.join(format!("{name}.{}.part", std::process::id()));
    let mut text = serde_json::to_string_pretty(value).with_context(|| format!("cannot serialize {}", path.display()))?;
    text.push('\n');
    let file = std::fs::File::create(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    std::io::Write::write_all(&mut &file, text.as_bytes()).with_context(|| format!("cannot write {}", tmp.display()))?;
    crate::fs_util::sync_best_effort(&file, &tmp).with_context(|| format!("cannot sync {}", tmp.display()))?;
    drop(file);
    std::fs::rename(&tmp, path).with_context(|| format!("cannot move {} into place", path.display()))?;
    Ok(())
}

/// A stage's frames as `file` names them in a message: the file name of `path`.
pub(crate) fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::geometry::linear::{Linear, LinearKind};
    use crate::geometry::pixel_map::PixelMap;
    use crate::pipeline::config::stage_hash;
    use crate::pipeline::Stage;
    use crate::stacking::config::{algorithm_preset, StackingPreset};
    use crate::stacking::measure::{ChannelMeasurement, FrameMeasurement};
    use crate::stacking::weights::{FormulaWeights, FrameWeight, SelectionConfig, WeightMode};

    fn cfg() -> crate::stacking::config::AlgorithmConfig {
        algorithm_preset(StackingPreset::Default)
    }

    fn measure_report(dir: &Path) -> MeasureReport {
        let mut report = MeasureReport::new(cfg(), BTreeMap::from([("measureStage".to_string(), 1)]));
        report.group_beta = Some(0.1 + 0.2);
        report.group_c = vec![GroupC { plane: 0, planes: 5, count: 1, median: Some(1.0 / 3.0), min: Some(0.9), max: None }];
        let channel = ChannelMeasurement {
            stars_detected: 120,
            stars_fitted: 100,
            fwhm_px: 2.0 / 3.0,
            eccentricity: 0.123_456_789_012_345_6,
            aperture_correction: Some(1.07),
            ..ChannelMeasurement::default()
        };
        report.frames = vec![MeasureFrame {
            file: dir.join("a.fit"),
            measurement: FrameMeasurement { width: 64, height: 48, channels: vec![channel], duration_ms: 12 },
            weight: FrameWeight { channels: vec![0.5], normalized: vec![1.0], mean: 0.5, normalized_mean: 1.0, missing: None },
            included: true,
            reason: None,
            measure_floored: false,
        }];
        report
    }

    fn transforms_report(dir: &Path) -> TransformsReport {
        let mut report = TransformsReport::new(dir.join("a.fit"), cfg(), BTreeMap::from([("align".to_string(), 3)]));
        let linear = Linear {
            kind: LinearKind::Affine,
            m: [[1.000_000_1, 1.0 / 3.0, 12.5], [-1e-7, 0.999_999_9, -7.25], [0.0, 0.0, 1.0]],
        };
        report.frames = vec![
            TransformFrame {
                file: dir.join("a.fit"),
                ok: true,
                error: None,
                model: "affine".to_string(),
                map: Some(PixelMap::linear(linear).unwrap()),
                inliers: 80,
                inlier_ratio: 0.8,
                rms_px: 0.07,
                peak_px: (0.2, 0.3),
                scale: 1.0,
                rotation_deg: 0.01,
                flipped: false,
                overlap: 0.9,
                coverage: 0.95,
                warnings: vec!["a warning".to_string()],
            },
            TransformFrame {
                file: dir.join("b.fit"),
                ok: false,
                error: Some("no match".to_string()),
                model: String::new(),
                map: None,
                inliers: 0,
                inlier_ratio: 0.0,
                rms_px: 0.0,
                peak_px: (0.0, 0.0),
                scale: 0.0,
                rotation_deg: 0.0,
                flipped: false,
                overlap: 0.0,
                coverage: 0.0,
                warnings: Vec::new(),
            },
        ];
        report
    }

    fn weights_report(dir: &Path) -> WeightsReport {
        let mut report = WeightsReport::new(WeightSettings {
            weight_mode: WeightMode::default(),
            formula: FormulaWeights::default(),
            keyword: "MYWEIGHT".to_string(),
            selection: SelectionConfig::default(),
        });
        report.common_stars = vec![Some(vec![(10.5, 20.25), (30.0, 40.125)]), None];
        report.frames = vec![WeightFrame {
            file: dir.join("a.fit"),
            weight: FrameWeight { channels: vec![0.25], normalized: vec![0.5], mean: 0.25, normalized_mean: 0.5, missing: Some("no keyword".to_string()) },
            included: false,
            reason: Some("excluded".to_string()),
        }];
        report
    }

    /// Rewrites the `version` of a file already on disk.
    fn set_version(path: &Path, version: u64) {
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        v["version"] = serde_json::json!(version);
        std::fs::write(path, serde_json::to_string(&v).unwrap()).unwrap();
    }

    fn assert_refuses_version(err: anyhow::Error, tag: &str) {
        let text = format!("{err:#}");
        assert!(text.contains(&format!("made by another starloom: {tag} version 2, this build reads 1")), "{text}");
    }

    #[test]
    fn each_format_round_trips_and_refuses_another_version() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();

        // The manifest, with a frame's facts and a history row.
        let frame = work.join("a.fit");
        std::fs::write(&frame, b"pixels").unwrap();
        let mut facts = astroimage::header::FrameFacts::default();
        facts.exptime_s = Some(300.1);
        facts.date_obs = Some("2024-01-15T20:30:45".to_string());
        facts.origin = Some((100, 200));
        let mut manifest = Manifest::new(work);
        manifest.entries.push(Entry {
            stage: Stage::Calibrate,
            stage_hash: stage_hash(Stage::Calibrate, &cfg()),
            versions: BTreeMap::from([("lightCal".to_string(), 4)]),
            inputs: vec![FileId::of(&frame).unwrap()],
            products: vec![frame.clone()],
            sources: BTreeMap::from([(frame.clone(), facts)]),
        });
        manifest.save().unwrap();
        assert_eq!(Manifest::load(work).unwrap(), manifest);
        set_version(&work.join(MANIFEST_FILE), 2);
        assert_refuses_version(Manifest::load(work).unwrap_err(), MANIFEST_FORMAT);

        // The measure report: bit-equal floats.
        let report = measure_report(work);
        let path = work.join("measure").join(MEASURE_REPORT_FILE);
        report.write(work, &path).unwrap();
        let back = MeasureReport::read(&path).unwrap();
        assert_eq!(back, report);
        assert_eq!(back.group_beta.unwrap().to_bits(), report.group_beta.unwrap().to_bits());
        assert_eq!(
            back.frames[0].measurement.channels[0].eccentricity.to_bits(),
            report.frames[0].measurement.channels[0].eccentricity.to_bits()
        );
        set_version(&path, 2);
        assert_refuses_version(MeasureReport::read(&path).unwrap_err(), MEASURE_REPORT_FORMAT);

        // The transforms: the PixelMap of `to_json` comes back bit-equal.
        let report = transforms_report(work);
        let path = work.join("register").join(TRANSFORMS_FILE);
        report.write(work, &path).unwrap();
        let back = TransformsReport::read(&path).unwrap();
        assert_eq!(back, report);
        let (a, b) = (report.frames[0].map.as_ref().unwrap(), back.frames[0].map.as_ref().unwrap());
        assert_eq!(a.to_json(), b.to_json());
        assert!(a.linear.m.iter().flatten().zip(b.linear.m.iter().flatten()).all(|(x, y)| x.to_bits() == y.to_bits()));
        set_version(&path, 2);
        assert_refuses_version(TransformsReport::read(&path).unwrap_err(), TRANSFORMS_FORMAT);

        // The weights, with the common star set.
        let report = weights_report(work);
        let path = work.join(WEIGHTS_FILE);
        report.write(work, &path).unwrap();
        assert_eq!(WeightsReport::read(&path).unwrap(), report);
        set_version(&path, 2);
        assert_refuses_version(WeightsReport::read(&path).unwrap_err(), WEIGHTS_FORMAT);

        // Another kind of file, and a file with no version at all.
        assert!(format!("{:#}", WeightsReport::read(&work.join("measure").join(MEASURE_REPORT_FILE)).unwrap_err())
            .contains("is not a starloom-weights file"));
        let bare = work.join("bare.json");
        std::fs::write(&bare, "{\"format\":\"starloom-weights\"}").unwrap();
        assert!(format!("{:#}", WeightsReport::read(&bare).unwrap_err()).contains("version none"));
        std::fs::write(&bare, "not json").unwrap();
        assert!(WeightsReport::read(&bare).is_err());
    }

    #[test]
    fn a_stored_map_with_a_singular_linear_model_is_refused_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let report = transforms_report(work);
        let path = work.join(TRANSFORMS_FILE);
        report.write(work, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["frames"][0]["map"]["linear"]["m"] = serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        std::fs::write(&path, v.to_string()).unwrap();
        let err = format!("{:#}", TransformsReport::read(&path).unwrap_err());
        assert!(err.contains("singular"), "{err}");
    }

    #[test]
    fn writes_leave_no_partial_file_in_the_work_folder() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let report = weights_report(work);
        report.write(work, &work.join(WEIGHTS_FILE)).unwrap();
        let parts: Vec<_> = std::fs::read_dir(work.join(TMP_DIR)).unwrap().collect();
        assert!(parts.is_empty(), "the staged file was renamed into place: {parts:?}");
    }
}
