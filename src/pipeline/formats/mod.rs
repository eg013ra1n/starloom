//! The files a working folder holds, each tagged and versioned.
//!
//! Every format has a `format` tag and a `version`. A reader refuses a file of another version, naming both, so a
//! working folder made by another starloom is never half-understood. The files are JSON and are written atomically:
//! beside their destination ([`part_path`]), then renamed into place.
//!
//! # Non-finite numbers
//!
//! A non-finite float (a NaN location of a plane with no valid pixel, say) is stored the way Athenaeum's metrics
//! payloads store it: serde_json writes it as `null`. A reader takes a `null` where a float belongs as NaN, so the
//! value comes back non-finite (an infinity comes back as NaN; the stages only test finiteness) and a later command
//! makes the same decision Athenaeum makes on the value it measured. A `null` where an option belongs is `None`.

pub mod manifest;
pub mod measure_report;
pub mod transforms;
pub mod weights;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::de::{DeserializeOwned, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use manifest::{
    append_history, begin_writing, commit_stage, frames_or_upstream, source_facts, Entry, HistoryRow, Manifest,
    MANIFEST_FILE, MANIFEST_FORMAT, MANIFEST_VERSION, TMP_DIR,
};
pub use measure_report::{
    GroupC, MeasureFrame, MeasureReport, MEASURE_CSV_COLUMNS, MEASURE_REPORT_FILE, MEASURE_REPORT_FORMAT,
    MEASURE_REPORT_VERSION,
};
pub use transforms::{
    TransformFrame, TransformsReport, NOT_REGISTERED, REGISTRATION_FAILED, TRANSFORMS_FILE, TRANSFORMS_FORMAT, TRANSFORMS_VERSION,
};
pub use weights::{WeightFrame, WeightSettings, WeightsReport, WEIGHTS_FILE, WEIGHTS_FORMAT, WEIGHTS_VERSION};

/// The two fields every format starts with, read first so a file of another kind or version is refused before its
/// body is parsed.
#[derive(Deserialize)]
struct Head {
    format: Option<String>,
    version: Option<u64>,
}

/// A stored JSON value read as serde_json reads it, except that a `null` asked for as a float is NaN (see the module
/// docs). Enums read as serde_json reads them.
struct NonFinite<'de>(&'de Value);

impl<'de> serde::Deserializer<'de> for NonFinite<'de> {
    type Error = serde_json::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            Value::Array(items) => visitor.visit_seq(Items(items.iter())),
            Value::Object(map) => visitor.visit_map(Fields { iter: map.iter(), value: None }),
            other => other.deserialize_any(visitor),
        }
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        if self.0.is_null() {
            visitor.visit_f64(f64::NAN)
        } else {
            self.0.deserialize_f64(visitor)
        }
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        if self.0.is_null() {
            visitor.visit_f32(f32::NAN)
        } else {
            self.0.deserialize_f32(visitor)
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        if self.0.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.0.deserialize_enum(name, variants, visitor)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 char str string bytes byte_buf unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }
}

struct Items<'de>(std::slice::Iter<'de, Value>);

impl<'de> SeqAccess<'de> for Items<'de> {
    type Error = serde_json::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error> {
        self.0.next().map(|v| seed.deserialize(NonFinite(v))).transpose()
    }
}

struct Fields<'de> {
    iter: serde_json::map::Iter<'de>,
    value: Option<&'de Value>,
}

impl<'de> MapAccess<'de> for Fields<'de> {
    type Error = serde_json::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error> {
        let Some((key, value)) = self.iter.next() else {
            return Ok(None);
        };
        self.value = Some(value);
        seed.deserialize(serde::de::value::BorrowedStrDeserializer::new(key)).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Self::Error> {
        let value = self.value.take().ok_or_else(|| serde::de::Error::custom("a value asked for before its key"))?;
        seed.deserialize(NonFinite(value))
    }
}

/// Reads a tagged, versioned JSON file. It refuses a file that is not `tag` or whose version is not `version`
/// ("made by another starloom: <tag> version N, this build reads M"). A `null` float reads as NaN (module docs).
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
    let value: Value = serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
    T::deserialize(NonFinite(&value)).with_context(|| format!("{} does not read as a {tag} file", path.display()))
}

/// Where a product bound for `dest` is built: `<dest>.part`, in `dest`'s own folder, so the rename that puts it in
/// place never crosses a volume. A name ending in `.part` never looks like a finished product, and the next run of the
/// same stage overwrites what a killed one left there.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(std::ffi::OsStr::to_os_string).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// Writes `value` as pretty JSON to `path`: into [`part_path`]`(path)`, synced, then renamed into place, so a reader
/// sees the old file or the whole new one. Creates the folder it needs. A non-finite float is written as `null`
/// (module docs).
pub(crate) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = part_path(path);
    let mut text = serde_json::to_string_pretty(value).with_context(|| format!("cannot serialize {}", path.display()))?;
    text.push('\n');
    let file = std::fs::File::create(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    std::io::Write::write_all(&mut &file, text.as_bytes()).with_context(|| format!("cannot write {}", tmp.display()))?;
    crate::fs_util::sync_best_effort(&file, &tmp).with_context(|| format!("cannot sync {}", tmp.display()))?;
    drop(file);
    std::fs::rename(&tmp, path).with_context(|| format!("cannot move {} into place", path.display()))?;
    Ok(())
}

/// A file as a message names it: the file name of `path`. (Only the tests use it until a stage names a frame.)
#[cfg(test)]
pub(crate) fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::linear::{Linear, LinearKind};
    use crate::geometry::pixel_map::PixelMap;
    use crate::pipeline::versions::recorded_versions;
    use crate::pipeline::Stage;
    use crate::stacking::config::{algorithm_preset, StackingPreset};
    use crate::stacking::measure::{ChannelMeasurement, FrameMeasurement};
    use crate::stacking::weights::{FormulaWeights, FrameWeight, SelectionConfig, WeightMode};

    fn cfg() -> crate::stacking::config::AlgorithmConfig {
        algorithm_preset(StackingPreset::Default)
    }

    fn measurement() -> FrameMeasurement {
        let channel = ChannelMeasurement {
            stars_detected: 120,
            stars_fitted: 100,
            fwhm_px: 2.0 / 3.0,
            eccentricity: 0.123_456_789_012_345_6,
            aperture_correction: Some(1.07),
            ..ChannelMeasurement::default()
        };
        FrameMeasurement { width: 64, height: 48, channels: vec![channel], duration_ms: 12 }
    }

    fn measure_report(dir: &Path) -> MeasureReport {
        let mut report = MeasureReport::new(cfg(), recorded_versions());
        report.group_beta = Some(0.1 + 0.2);
        report.group_c = vec![GroupC { plane: 0, planes: 5, count: 1, median: Some(1.0 / 3.0), min: Some(0.9), max: None }];
        report.frames = vec![MeasureFrame {
            file: dir.join("a.fit"),
            measurement: Some(measurement()),
            weight: Some(FrameWeight { channels: vec![0.5], normalized: vec![1.0], mean: 0.5, normalized_mean: 1.0, missing: None }),
            included: true,
            reason: None,
            measure_floored: false,
        }];
        report
    }

    fn transforms_report(dir: &Path) -> TransformsReport {
        let mut report = TransformsReport::new(dir.join("a.fit"), cfg(), recorded_versions());
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
        let mut reweighed = measurement();
        reweighed.channels[0].common_tflux = Some(1234.5);
        reweighed.channels[0].common_tmean_flux = Some(12.25);
        reweighed.channels[0].common_stars = Some(88);
        reweighed.channels[0].aperture_correction_group = Some(1.05);
        report.frames = vec![WeightFrame {
            file: dir.join("a.fit"),
            measurement: Some(reweighed),
            weight: Some(FrameWeight { channels: vec![0.25], normalized: vec![0.5], mean: 0.25, normalized_mean: 0.5, missing: Some("no keyword".to_string()) }),
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
        let mut entry = Entry::new(Stage::Calibrate, &cfg(), vec![work.join("raw").join("a.fit")], vec![frame.clone()]);
        entry.sources.insert(frame.clone(), facts);
        manifest.entries.push(entry);
        manifest.save().unwrap();
        assert_eq!(Manifest::load(work).unwrap(), manifest);
        set_version(&work.join(MANIFEST_FILE), 2);
        assert_refuses_version(Manifest::load(work).unwrap_err(), MANIFEST_FORMAT);

        // The measure report: bit-equal floats.
        let report = measure_report(work);
        let path = work.join("measure").join(MEASURE_REPORT_FILE);
        report.write(&path).unwrap();
        let back = MeasureReport::read(&path).unwrap();
        assert_eq!(back, report);
        assert_eq!(back.group_beta.unwrap().to_bits(), report.group_beta.unwrap().to_bits());
        assert_eq!(
            back.frames[0].measurement.as_ref().unwrap().channels[0].eccentricity.to_bits(),
            report.frames[0].measurement.as_ref().unwrap().channels[0].eccentricity.to_bits()
        );
        set_version(&path, 2);
        assert_refuses_version(MeasureReport::read(&path).unwrap_err(), MEASURE_REPORT_FORMAT);

        // The transforms: the PixelMap of `to_json` comes back bit-equal.
        let report = transforms_report(work);
        let path = work.join("register").join(TRANSFORMS_FILE);
        report.write(&path).unwrap();
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
        report.write(&path).unwrap();
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

    /// I7: a frame's row in `weights.json` holds its reweighed measurement under Athenaeum's
    /// `stacking_run_frames.metrics_json` keys: the same type, serialized the same way.
    #[test]
    fn weights_rows_carry_the_common_set_terms_under_athenaeum_s_metrics_keys() {
        let dir = tempfile::tempdir().unwrap();
        let report = weights_report(dir.path());
        let row = serde_json::to_value(&report.frames[0]).unwrap();
        let metrics_json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(report.frames[0].measurement.as_ref().unwrap()).unwrap()).unwrap();
        assert_eq!(row["measurement"], metrics_json);
        let plane = &row["measurement"]["channels"][0];
        assert_eq!(plane["commonTflux"], serde_json::json!(1234.5));
        assert_eq!(plane["commonTmeanFlux"], serde_json::json!(12.25));
        assert_eq!(plane["commonStars"], serde_json::json!(88));
        assert_eq!(plane["apertureCorrectionGroup"], serde_json::json!(1.05));
    }

    /// M5: a NaN or an infinity is written as Athenaeum's metrics payloads hold it (`null`) and reads back non-finite;
    /// a `null` option stays `None`.
    #[test]
    fn non_finite_floats_are_stored_as_null_and_read_back_non_finite() {
        let dir = tempfile::tempdir().unwrap();
        let mut report = measure_report(dir.path());
        let plane = &mut report.frames[0].measurement.as_mut().unwrap().channels[0];
        plane.location = f64::NAN;
        plane.scale = f64::INFINITY;
        plane.psf_snr = f64::NEG_INFINITY;
        plane.aperture_correction = Some(f64::NAN);
        report.group_beta = Some(f64::NAN);
        report.frames[0].weight.as_mut().unwrap().channels = vec![f64::NAN];
        let path = dir.path().join(MEASURE_REPORT_FILE);
        report.write(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let payload = serde_json::to_string(&report.frames[0].measurement).unwrap();
        assert!(payload.contains("\"location\":null") && payload.contains("\"scale\":null"), "{payload}");
        assert!(text.contains("\"location\": null"), "the report stores what the metrics payload stores");

        let back = MeasureReport::read(&path).unwrap();
        let (b, f) = (
            &back.frames[0].measurement.as_ref().unwrap().channels[0],
            &report.frames[0].measurement.as_ref().unwrap().channels[0],
        );
        assert!(b.location.is_nan() && !b.scale.is_finite() && !b.psf_snr.is_finite());
        assert_eq!(b.aperture_correction, None, "an option written as null is None");
        assert_eq!(back.group_beta, None);
        assert!(back.frames[0].weight.as_ref().unwrap().channels[0].is_nan());
        // Every finite number is untouched.
        assert_eq!(b.fwhm_px.to_bits(), f.fwhm_px.to_bits());
        assert_eq!(b.eccentricity.to_bits(), f.eccentricity.to_bits());
        assert_eq!(back.frames[0].file, report.frames[0].file);
        assert_eq!(back.config, report.config);
        // A file written by serde_json itself reads the same way.
        let plain = serde_json::to_string(&report).unwrap();
        std::fs::write(&path, plain).unwrap();
        assert!(MeasureReport::read(&path).unwrap().frames[0].measurement.as_ref().unwrap().channels[0].location.is_nan());
    }

    #[test]
    fn a_stored_map_with_a_singular_linear_model_is_refused_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let report = transforms_report(work);
        let path = work.join(TRANSFORMS_FILE);
        report.write(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["frames"][0]["map"]["linear"]["m"] = serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        std::fs::write(&path, v.to_string()).unwrap();
        let err = format!("{:#}", TransformsReport::read(&path).unwrap_err());
        assert!(err.contains("singular"), "{err}");
    }

    #[test]
    fn writes_leave_no_partial_file_beside_the_destination() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let path = work.join("reports").join(WEIGHTS_FILE);
        weights_report(work).write(&path).unwrap();
        let names: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| display_name(&e.unwrap().path()))
            .collect();
        assert_eq!(names, [WEIGHTS_FILE], "the staged file was renamed into place");
        assert_eq!(display_name(&part_path(&path)), "weights.json.part");
        assert_eq!(part_path(&path).parent(), path.parent());
    }
}
