//! The config levels and the hashes built from them.
//!
//! A command's effective config is built in three levels, each over the one before: a built-in preset, a config file,
//! then `--set` overrides. The hashes are what the manifest compares: a stage's hash is the hash of its subtree
//! (P2-R9), so a product is reusable exactly when Athenaeum's own cache would reuse it.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::pipeline::Stage;
use crate::stacking::config::{
    algorithm_preset, calibration_subtree, measurement_subtree, normalization_subtree, registration_subtree,
    AlgorithmConfig, StackingPreset,
};

/// The seven sections of an [`AlgorithmConfig`]: the only top-level keys a config file contributes.
const SECTIONS: [&str; 7] = [
    "calibration",
    "measurement",
    "selection",
    "registration",
    "normalization",
    "integration",
    "drizzle",
];

/// Keys that exist but are left out of the serialized config while unset, so a `--set` may still name them:
/// `(the section's path, the key)`.
const ABSENT_WHILE_UNSET: [(&str, &str); 1] = [("measurement", "fitStars")];

fn xxh3_hex(bytes: &[u8]) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(bytes))
}

/// The hash of one stage's config subtree: xxh3-64 of `serde_json::to_string` of the subtree the moved builders
/// ([`calibration_subtree`], [`measurement_subtree`], [`registration_subtree`], [`normalization_subtree`]) write, as
/// 16 lowercase hex digits. Athenaeum pins the same value per preset, so a stage hash here changes exactly when
/// Athenaeum's subtree does. (Athenaeum's stored stage hashes also fold upstream hashes and sources; the two are never
/// compared, only their subtrees.)
///
/// [`Stage::Stack`] has no Athenaeum subtree of its own: its hash covers what the stack stage reads and no other stage
/// does, the `integration`, `drizzle` and `selection` sections and the weight settings.
pub fn stage_hash(stage: Stage, cfg: &AlgorithmConfig) -> String {
    let subtree = match stage {
        Stage::Calibrate => calibration_subtree(cfg),
        Stage::Measure => measurement_subtree(cfg),
        Stage::Register => registration_subtree(cfg),
        Stage::Normalize => normalization_subtree(cfg),
        Stage::Stack => serde_json::json!({
            "integration": cfg.integration,
            "drizzle": cfg.drizzle,
            "selection": cfg.selection,
            "weightMode": cfg.measurement.weight_mode,
            "formula": cfg.measurement.formula,
            "keyword": cfg.measurement.keyword,
        }),
    };
    xxh3_hex(serde_json::to_string(&subtree).expect("a subtree serializes").as_bytes())
}

/// The hash of the whole [`AlgorithmConfig`]: xxh3-64 of its JSON, as 16 lowercase hex digits. It is what the
/// `SLM_CFG` card records.
pub fn config_hash(cfg: &AlgorithmConfig) -> String {
    xxh3_hex(serde_json::to_string(cfg).expect("an AlgorithmConfig serializes").as_bytes())
}

/// Reads a config file as a JSON value: TOML for `.toml`, JSON for `.json`.
fn read_config_file(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read config {}", path.display()))?;
    let ext = path.extension().and_then(std::ffi::OsStr::to_str).map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("toml") => toml::from_str(&text).with_context(|| format!("config {} is not valid TOML", path.display())),
        Some("json") => serde_json::from_str(&text).with_context(|| format!("config {} is not valid JSON", path.display())),
        _ => bail!("config {} must be a .toml or .json file", path.display()),
    }
}

/// `--set`'s value: JSON when it parses as JSON (`0.2`, `true`, `{"moffat":8}`), else the text itself as a string
/// (`moffat4`).
fn parse_set_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn valid_keys(at: &str, obj: &serde_json::Map<String, Value>) -> String {
    let mut keys: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    keys.extend(ABSENT_WHILE_UNSET.iter().filter(|(p, _)| *p == at).map(|(_, k)| *k));
    keys.into_iter().collect::<Vec<_>>().join(", ")
}

/// Sets the value at the dotted camelCase `path`. A key the config does not have is an error that lists the keys valid
/// at that level.
fn set_path(root: &mut Value, path: &str, value: Value) -> Result<()> {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.iter().any(|s| s.is_empty()) {
        bail!("`{path}` is not a dotted path of keys");
    }
    let mut node = root;
    let mut at = String::new();
    for (i, key) in segments.iter().enumerate() {
        let last = i + 1 == segments.len();
        let level = if at.is_empty() { "the top level".to_string() } else { format!("`{at}`") };
        let obj = node.as_object_mut().ok_or_else(|| {
            anyhow!("cannot set `{path}`: `{at}` is a value, not a section with keys")
        })?;
        let known = obj.contains_key(*key) || ABSENT_WHILE_UNSET.iter().any(|(p, k)| *p == at && k == key);
        if !known {
            bail!("unknown key `{key}` at {level}; valid keys: {}", valid_keys(&at, obj));
        }
        if last {
            obj.insert((*key).to_string(), value);
            return Ok(());
        }
        node = obj.entry((*key).to_string()).or_insert(Value::Null);
        if !at.is_empty() {
            at.push('.');
        }
        at.push_str(key);
    }
    unreachable!("a path has at least one segment")
}

/// The effective config of a command, and the top-level keys of its file that no section reads.
///
/// 1. `preset` (the default preset when `None`);
/// 2. `file`, a JSON or TOML document (by extension). It may be a whole Athenaeum stacking config: only the seven
///    algorithm sections are read, each **replacing** the preset's section whole (a section the file leaves out keeps
///    the preset's; a field a section leaves out takes the field's default, not the preset's). Every other top-level
///    key is returned, sorted, so the caller can say it was ignored;
/// 3. `sets`: `(dotted camelCase path, value)` pairs, applied in order. A value that parses as JSON is that value
///    (`0.2`, `true`, `{"moffat":8}`); a bare word is a JSON string (`moffat4`). A path naming no key is an error that
///    lists the keys valid at that level.
pub fn effective_config(
    preset: Option<StackingPreset>,
    file: Option<&Path>,
    sets: &[(String, String)],
) -> Result<(AlgorithmConfig, Vec<String>)> {
    let base = algorithm_preset(preset.unwrap_or(StackingPreset::Default));
    if file.is_none() && sets.is_empty() {
        return Ok((base, Vec::new()));
    }
    let mut value = serde_json::to_value(&base).expect("an AlgorithmConfig serializes");
    let mut ignored = Vec::new();
    if let Some(path) = file {
        let doc = read_config_file(path)?;
        let Value::Object(doc) = doc else {
            bail!("config {} must hold a table of sections at its top level", path.display());
        };
        let target = value.as_object_mut().expect("an AlgorithmConfig is an object");
        for (key, section) in doc {
            if SECTIONS.contains(&key.as_str()) {
                target.insert(key, section);
            } else {
                ignored.push(key);
            }
        }
        ignored.sort();
    }
    for (path, raw) in sets {
        set_path(&mut value, path, parse_set_value(raw)).with_context(|| format!("--set {path}={raw}"))?;
    }
    let cfg: AlgorithmConfig = serde_json::from_value(value).context("the effective config is not a valid config")?;
    tracing::debug!(
        preset = ?preset,
        file = file.map(|p| p.display().to_string()).as_deref().unwrap_or(""),
        sets = sets.len(),
        ignored = ignored.len(),
        "effective config built"
    );
    Ok((cfg, ignored))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_preset() -> [(&'static str, StackingPreset); 3] {
        [
            ("Default", StackingPreset::Default),
            ("FastPreview", StackingPreset::FastPreview),
            ("MaximumQuality", StackingPreset::MaximumQuality),
        ]
    }

    /// Task 6's pins (`every_preset_stage_subtree_is_pinned` in Athenaeum's core), copied: calibration, measurement,
    /// registration, normalization. A change to a subtree moves Athenaeum's hashes, so it moves these too.
    #[test]
    fn the_stage_hash_of_every_preset_equals_the_pinned_subtree() {
        let expected = [
            ("Default", "0f3177998fbce0da", "3e6550bbf1c99012", "0a4a72469833a578", "135f0143c278b3e6"),
            ("FastPreview", "0f3177998fbce0da", "3e6550bbf1c99012", "e8bf9794120c4ab5", "06ab9d4517b16ba5"),
            ("MaximumQuality", "0f3177998fbce0da", "3e6550bbf1c99012", "dd272ff14698a2da", "eed22cdb5e2a2d25"),
        ];
        for ((name, p), (pinned_name, cal, mea, reg, nrm)) in every_preset().into_iter().zip(expected) {
            assert_eq!(name, pinned_name);
            let cfg = algorithm_preset(p);
            assert_eq!(stage_hash(Stage::Calibrate, &cfg), cal, "{name}: calibration subtree moved");
            assert_eq!(stage_hash(Stage::Measure, &cfg), mea, "{name}: measurement subtree moved");
            assert_eq!(stage_hash(Stage::Register, &cfg), reg, "{name}: registration subtree moved");
            assert_eq!(stage_hash(Stage::Normalize, &cfg), nrm, "{name}: normalization subtree moved");
        }
    }

    #[test]
    fn the_stack_hash_follows_the_stack_sections_and_the_config_hash_follows_everything() {
        let cfg = algorithm_preset(StackingPreset::Default);
        let mut other = cfg.clone();
        other.integration.min_weight += 0.05;
        assert_ne!(stage_hash(Stage::Stack, &cfg), stage_hash(Stage::Stack, &other));
        assert_eq!(stage_hash(Stage::Measure, &cfg), stage_hash(Stage::Measure, &other));
        assert_ne!(config_hash(&cfg), config_hash(&other));
        assert_eq!(config_hash(&cfg), config_hash(&cfg.clone()));
        assert_eq!(config_hash(&cfg).len(), 16);
        assert_ne!(config_hash(&cfg), config_hash(&algorithm_preset(StackingPreset::MaximumQuality)));
    }

    #[test]
    fn effective_config_applies_preset_file_and_sets_in_that_order() {
        let dir = tempfile::tempdir().unwrap();
        // No file and no sets: the preset, exactly.
        for (name, p) in every_preset() {
            let (cfg, ignored) = effective_config(Some(p), None, &[]).unwrap();
            assert_eq!(cfg, algorithm_preset(p), "{name}");
            assert!(ignored.is_empty());
        }
        let (cfg, _) = effective_config(None, None, &[]).unwrap();
        assert_eq!(cfg, AlgorithmConfig::default());

        // A TOML file over a preset: the file's sections replace the preset's, the others stay.
        let toml_path = dir.path().join("my config.toml");
        std::fs::write(
            &toml_path,
            "version = 3\n\n[grouping]\nkey = 1\n\n[integration]\nminWeight = 0.1\n\n[drizzle]\nenabled = true\nscale = 3\n",
        )
        .unwrap();
        let (cfg, ignored) = effective_config(Some(StackingPreset::FastPreview), Some(&toml_path), &[]).unwrap();
        assert_eq!(ignored, ["grouping", "version"]);
        assert_eq!(cfg.integration.min_weight, 0.1);
        assert!(cfg.drizzle.enabled && cfg.drizzle.scale == 3);
        // The integration section came from the file whole: the preset's sigma clip is gone, the field default is back.
        assert_eq!(cfg.integration.rejection, AlgorithmConfig::default().integration.rejection);
        // A section the file leaves out keeps the preset's.
        assert_eq!(cfg.registration, algorithm_preset(StackingPreset::FastPreview).registration);

        // A set wins over the file; a bare word is a string, anything else is JSON.
        let sets = vec![
            ("integration.minWeight".to_string(), "0.2".to_string()),
            ("measurement.psfModel".to_string(), "moffat4".to_string()),
            ("normalization.local.enabled".to_string(), "true".to_string()),
        ];
        let (cfg, _) = effective_config(Some(StackingPreset::FastPreview), Some(&toml_path), &sets).unwrap();
        assert_eq!(cfg.integration.min_weight, 0.2);
        assert_eq!(serde_json::to_value(&cfg.measurement.psf_model).unwrap(), serde_json::json!("moffat4"));
        assert!(cfg.normalization.local.enabled);
        let (cfg, _) = effective_config(
            None,
            None,
            &[("measurement.psfModel".to_string(), "{\"moffat\":8.0}".to_string())],
        )
        .unwrap();
        assert_eq!(serde_json::to_value(&cfg.measurement.psf_model).unwrap(), serde_json::json!({"moffat": 8.0}));
        // The key an unset option leaves out of the serialized config can still be set.
        let (cfg, _) = effective_config(None, None, &[("measurement.fitStars".to_string(), "40".to_string())]).unwrap();
        assert_eq!(cfg.measurement.fit_stars, Some(40));

        // A JSON file works the same, and the same file in both formats is the same config.
        let json_path = dir.path().join("c.json");
        std::fs::write(&json_path, "{\"integration\":{\"minWeight\":0.1},\"reference\":{}}").unwrap();
        let (cfg, ignored) = effective_config(None, Some(&json_path), &[]).unwrap();
        assert_eq!(cfg.integration.min_weight, 0.1);
        assert_eq!(ignored, ["reference"]);
    }

    #[test]
    fn a_set_naming_no_key_lists_the_keys_valid_at_that_level() {
        let err = format!(
            "{:#}",
            effective_config(None, None, &[("integration.nope".to_string(), "1".to_string())]).unwrap_err()
        );
        assert!(err.contains("integration.nope=1") && err.contains("unknown key `nope`"), "{err}");
        assert!(err.contains("minWeight"), "{err}");
        let err = format!("{:#}", effective_config(None, None, &[("nope.x".to_string(), "1".to_string())]).unwrap_err());
        assert!(err.contains("integration") && err.contains("drizzle"), "{err}");
        // Through a value, and a value of the wrong type.
        let err = format!(
            "{:#}",
            effective_config(None, None, &[("integration.minWeight.x".to_string(), "1".to_string())]).unwrap_err()
        );
        assert!(err.contains("not a section"), "{err}");
        let err = format!(
            "{:#}",
            effective_config(None, None, &[("integration.minWeight".to_string(), "[1]".to_string())]).unwrap_err()
        );
        assert!(err.contains("not a valid config"), "{err}");
        let err = format!("{:#}", effective_config(None, None, &[("integration..x".to_string(), "1".to_string())]).unwrap_err());
        assert!(err.contains("dotted path"), "{err}");
    }

    #[test]
    fn a_config_file_must_be_a_toml_or_json_table() {
        let dir = tempfile::tempdir().unwrap();
        let bad_ext = dir.path().join("c.yaml");
        std::fs::write(&bad_ext, "").unwrap();
        assert!(format!("{:#}", effective_config(None, Some(&bad_ext), &[]).unwrap_err()).contains(".toml or .json"));
        let array = dir.path().join("c.json");
        std::fs::write(&array, "[1]").unwrap();
        assert!(format!("{:#}", effective_config(None, Some(&array), &[]).unwrap_err()).contains("table of sections"));
        let missing = dir.path().join("absent.json");
        assert!(effective_config(None, Some(&missing), &[]).is_err());
    }
}
