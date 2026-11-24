//! The manifest (`starloom.json`): which stage made which products from which inputs, and the history of the commands
//! run in the working folder.
//!
//! An entry says "this stage, under this config, made these products from these inputs". A later stage trusts a
//! product only after [`check_upstream`] has compared the entry with what it is about to read. Products are written in
//! `<work>/.tmp/` and renamed into place, and [`commit_stage`] writes the entry after them, so an interrupted stage
//! leaves no entry and the files it did move are not trusted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::FrameFacts;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{display_name, read_versioned, write_json_atomic};
use crate::integration::stats::RejectionNormalization;
use crate::pipeline::config::stage_hash;
use crate::pipeline::Stage;
use crate::stacking::config::AlgorithmConfig;

/// The manifest's file name, directly in the working folder.
pub const MANIFEST_FILE: &str = "starloom.json";
/// The manifest's `format` tag.
pub const MANIFEST_FORMAT: &str = "starloom-work";
/// The manifest's version. Another version is refused.
pub const MANIFEST_VERSION: u32 = 1;
/// The folder under the working folder where a writing command builds its products before moving them into place.
pub const TMP_DIR: &str = ".tmp";

/// A file as a stage saw it: its absolute path, its size and its sampled content hash
/// ([`sampled_xxh3`](crate::fs_util::sampled_xxh3)). Two ids are equal when the file at the path is the same size
/// with the same sampled bytes, so a frame rewritten with identical content is the same input and a modification time
/// is never consulted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileId {
    /// The absolute path.
    pub path: PathBuf,
    /// The size in bytes.
    pub size: u64,
    /// The sampled content hash, 16 lowercase hex digits.
    pub sampled_hash: String,
}

impl FileId {
    /// Reads the id of the file at `path` (made absolute with `std::path::absolute`).
    pub fn of(path: &Path) -> Result<FileId> {
        let path = std::path::absolute(path).with_context(|| format!("cannot make {} absolute", path.display()))?;
        let size = std::fs::metadata(&path).with_context(|| format!("cannot stat {}", path.display()))?.len();
        let sampled_hash = crate::fs_util::sampled_xxh3(&path).with_context(|| format!("cannot hash {}", path.display()))?;
        Ok(FileId { path, size, sampled_hash })
    }
}

/// What one stage made, and what it made it from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    /// The stage.
    pub stage: Stage,
    /// The stage hash ([`stage_hash`]) of the config the stage ran with.
    pub stage_hash: String,
    /// The versions the products were made at (a stage's version constants by name), so a build with another version
    /// can tell the products are not its own.
    pub versions: BTreeMap<String, u32>,
    /// The files the stage read. For `measure`, `register` and `normalize` they are the frames, in the order the stage
    /// kept them (P2-R10), and `frames_or_upstream` hands that order to the next command; other inputs (a reference
    /// outside the frames) may follow them.
    pub inputs: Vec<FileId>,
    /// The files the stage made, absolute. They exist when the entry does.
    pub products: Vec<PathBuf>,
    /// For `calibrate`: each calibrated product's raw light and that light's facts. `measure` takes a frame's
    /// exposure and date from here (P2-R3), because the calibrated file no longer carries them.
    #[serde(default)]
    pub sources: BTreeMap<PathBuf, FrameFacts>,
}

impl Entry {
    /// The name a message uses for the entry's files: its first product's file name, or the stage's name.
    fn file_label(&self) -> String {
        self.products.first().map_or_else(|| self.stage.name().to_string(), |p| display_name(p))
    }
}

/// One command run in the working folder (P2-R19). A row is appended when the command ends, also when it fails or is
/// refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryRow {
    /// The command's name (`calibrate`, `stack`, ...).
    pub command: String,
    /// The command's arguments as given.
    pub args: Vec<String>,
    /// When it started.
    pub started: DateTime<Utc>,
    /// When it ended.
    pub finished: DateTime<Utc>,
    /// The exit code: 0 done, 2 refused, 1 failed.
    pub exit_code: i32,
    /// The version of starloom that ran it.
    pub starloom_version: String,
    /// The hash of the whole effective config ([`config_hash`](crate::pipeline::config_hash)).
    pub config_hash: String,
    /// The command's log file.
    pub log_file: PathBuf,
    /// The summary line the command printed.
    pub summary: String,
}

/// The manifest of a working folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Always [`MANIFEST_FORMAT`].
    pub format: String,
    /// Always [`MANIFEST_VERSION`].
    pub version: u32,
    /// One entry per stage that has run, in stage order.
    pub entries: Vec<Entry>,
    /// Every command run in the folder, oldest first.
    pub history: Vec<HistoryRow>,
    /// The working folder the manifest was loaded from (not stored).
    #[serde(skip)]
    work_dir: PathBuf,
}

impl Manifest {
    /// An empty manifest for `work_dir`.
    pub fn new(work_dir: &Path) -> Manifest {
        Manifest {
            format: MANIFEST_FORMAT.to_string(),
            version: MANIFEST_VERSION,
            entries: Vec::new(),
            history: Vec::new(),
            work_dir: work_dir.to_path_buf(),
        }
    }

    /// The manifest of `work_dir`: an empty one when the folder has none, an error when the file is of another version
    /// or is not a manifest.
    pub fn load(work_dir: &Path) -> Result<Manifest> {
        let path = work_dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(Manifest::new(work_dir));
        }
        let mut manifest: Manifest = read_versioned(&path, MANIFEST_FORMAT, MANIFEST_VERSION)?;
        manifest.work_dir = work_dir.to_path_buf();
        Ok(manifest)
    }

    /// Writes the manifest atomically.
    pub fn save(&self) -> Result<()> {
        write_json_atomic(&self.work_dir, &self.work_dir.join(MANIFEST_FILE), self)
    }

    /// The working folder.
    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    /// The entry of `stage`, if it has run.
    pub fn entry(&self, stage: Stage) -> Option<&Entry> {
        self.entries.iter().find(|e| e.stage == stage)
    }

    /// Puts `entry` in place of the stage's earlier one, keeping stage order.
    fn replace_entry(&mut self, entry: Entry) {
        self.entries.retain(|e| e.stage != entry.stage);
        self.entries.push(entry);
        self.entries.sort_by_key(|e| e.stage);
    }
}

/// Appends a history row to the manifest of `work_dir`, creating the manifest when the folder has none.
pub fn append_history(work_dir: &Path, row: HistoryRow) -> Result<()> {
    let mut manifest = Manifest::load(work_dir)?;
    manifest.history.push(row);
    manifest.save()
}

/// Starts a writing command: clears `<work>/.tmp/` (what a killed process left) and makes it, returning its path. Every
/// command that writes calls it first.
pub fn begin_writing(work_dir: &Path) -> Result<PathBuf> {
    let tmp = work_dir.join(TMP_DIR);
    match std::fs::remove_dir_all(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("cannot clear {}", tmp.display())),
    }
    std::fs::create_dir_all(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    Ok(tmp)
}

/// Finishes a stage: moves each `(built, final)` pair of `moves` into place, then writes `entry`.
///
/// The stage's earlier entry is removed first, and the manifest saved without it, so a failure while the products are
/// moving leaves no entry that a mix of old and new files could satisfy. After the moves every path in
/// `entry.products` must exist; only then is the entry written. A file or folder already at a destination is replaced.
pub fn commit_stage(work_dir: &Path, moves: &[(PathBuf, PathBuf)], entry: Entry) -> Result<()> {
    let mut manifest = Manifest::load(work_dir)?;
    if manifest.entry(entry.stage).is_some() {
        manifest.entries.retain(|e| e.stage != entry.stage);
        manifest.save()?;
    }
    for (built, dest) in moves {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
        }
        if dest.is_dir() {
            std::fs::remove_dir_all(dest).with_context(|| format!("cannot replace {}", dest.display()))?;
        }
        std::fs::rename(built, dest)
            .with_context(|| format!("cannot move {} to {}", built.display(), dest.display()))?;
    }
    if let Some(missing) = entry.products.iter().find(|p| !p.exists()) {
        bail!("{} stage product {} does not exist; no entry written", entry.stage.name(), missing.display());
    }
    tracing::debug!(stage = entry.stage.name(), products = entry.products.len(), inputs = entry.inputs.len(), "stage committed");
    manifest.replace_entry(entry);
    manifest.save()
}

/// The stages whose products `stage` reads and so must still be the ones the given frames came from.
fn upstream_stages(stage: Stage, cfg: &AlgorithmConfig) -> Vec<Stage> {
    match stage {
        Stage::Calibrate => Vec::new(),
        Stage::Measure => vec![Stage::Calibrate],
        Stage::Register => vec![Stage::Measure],
        Stage::Normalize => vec![Stage::Measure, Stage::Register],
        Stage::Stack => {
            let mut v = vec![Stage::Measure, Stage::Register];
            // Without local normalization there is no `.athln` to be stale.
            if cfg.normalization.rejection == RejectionNormalization::Local {
                v.push(Stage::Normalize);
            }
            v
        }
    }
}

/// Refuses to run `stage` on frames, or on products, that the stages before it did not make under the current config.
///
/// Every stage upstream of `stage` must have an entry, and the entry must agree with `frames` and `cfg`:
/// - the entry is there; otherwise "run `starloom <stage>` first";
/// - every given frame is one the entry recorded (a `calibrate` entry's products, the others' inputs) with the same
///   [`FileId`]: "<file> was made from other frames (N differ, first <name>); re-run `starloom <stage>`";
/// - the entry's stage hash equals the current config's: "<file> was made with another <subtree> config; re-run
///   `starloom <stage>`".
///
/// A change that no subtree hash holds, such as the weight mode (P2-R21), passes. Stages are checked in order, so the
/// message names the earliest stage to re-run.
pub fn check_upstream(manifest: &Manifest, stage: Stage, frames: &[PathBuf], cfg: &AlgorithmConfig) -> Result<()> {
    for up in upstream_stages(stage, cfg) {
        let entry = manifest.entry(up).ok_or_else(|| {
            anyhow!(
                "{} has no `{}` entry; run `starloom {}` first",
                manifest.work_dir().display(),
                up.name(),
                up.name()
            )
        })?;
        let file = entry.file_label();
        let mut differing = Vec::new();
        if up == Stage::Calibrate {
            let made: Vec<&PathBuf> = entry.products.iter().collect();
            for frame in frames {
                let abs = std::path::absolute(frame).with_context(|| format!("cannot make {} absolute", frame.display()))?;
                if !made.iter().any(|p| **p == abs) {
                    differing.push(abs);
                }
            }
        } else {
            let recorded: BTreeMap<&Path, &FileId> = entry.inputs.iter().map(|id| (id.path.as_path(), id)).collect();
            for frame in frames {
                let now = FileId::of(frame)?;
                if recorded.get(now.path.as_path()) != Some(&&now) {
                    differing.push(now.path);
                }
            }
        }
        if let Some(first) = differing.first() {
            bail!(
                "{file} was made from other frames ({} differ, first {}); re-run `starloom {}`",
                differing.len(),
                display_name(first),
                up.name()
            );
        }
        if entry.stage_hash != stage_hash(up, cfg) {
            bail!("{file} was made with another {} config; re-run `starloom {}`", up.subtree_name(), up.name());
        }
    }
    Ok(())
}

/// The frames a command works on: `given` when the command line named any (they win), else the frames the previous
/// stage left in the manifest, in the order it recorded (P2-R20): `measure` takes `calibrate`'s products; `register`,
/// `normalize` and `stack` take `measure`'s frames.
pub fn frames_or_upstream(manifest: &Manifest, stage: Stage, given: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !given.is_empty() {
        return Ok(given);
    }
    let up = match stage {
        Stage::Calibrate => bail!("no frames given; give LIGHTS or a plan"),
        Stage::Measure => Stage::Calibrate,
        Stage::Register | Stage::Normalize | Stage::Stack => Stage::Measure,
    };
    let entry = manifest.entry(up).ok_or_else(|| {
        anyhow!(
            "no frames given and {} has no `{}` entry; give FRAMES or run `starloom {}`",
            manifest.work_dir().display(),
            up.name(),
            up.name()
        )
    })?;
    Ok(if up == Stage::Calibrate {
        entry.products.clone()
    } else {
        entry.inputs.iter().map(|id| id.path.clone()).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stacking::config::{algorithm_preset, StackingPreset};
    use crate::stacking::weights::WeightMode;

    fn frame(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn ids(frames: &[PathBuf]) -> Vec<FileId> {
        frames.iter().map(|f| FileId::of(f).unwrap()).collect()
    }

    fn entry(stage: Stage, cfg: &AlgorithmConfig, inputs: Vec<FileId>, products: Vec<PathBuf>) -> Entry {
        Entry {
            stage,
            stage_hash: stage_hash(stage, cfg),
            versions: BTreeMap::new(),
            inputs,
            products,
            sources: BTreeMap::new(),
        }
    }

    /// A working folder with a `calibrate` entry (two calibrated frames) and a `measure` entry over them.
    fn chain(dir: &Path, cfg: &AlgorithmConfig) -> (Manifest, Vec<PathBuf>) {
        let cal = dir.join("calibrated");
        std::fs::create_dir_all(&cal).unwrap();
        let frames = vec![frame(&cal, "a.fit", b"alpha"), frame(&cal, "b.fit", b"bravo")];
        let report = frame(dir, "measure.json", b"{}");
        let mut m = Manifest::new(dir);
        m.entries.push(entry(Stage::Calibrate, cfg, Vec::new(), frames.clone()));
        m.entries.push(entry(Stage::Measure, cfg, ids(&frames), vec![report]));
        (m, frames)
    }

    #[test]
    fn check_upstream_refuses_products_made_from_other_frames_or_another_config() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = algorithm_preset(StackingPreset::Default);
        let (mut m, frames) = chain(dir.path(), &cfg);
        m.entries.push(entry(Stage::Register, &cfg, ids(&frames), vec![frame(dir.path(), "transforms.json", b"{}")]));
        m.entries.push(entry(Stage::Normalize, &cfg, ids(&frames), vec![frame(dir.path(), "ln.fit", b"{}")]));
        for stage in [Stage::Measure, Stage::Register, Stage::Normalize, Stage::Stack] {
            check_upstream(&m, stage, &frames, &cfg).unwrap_or_else(|e| panic!("{stage:?}: {e:#}"));
        }
        // A frame given twice or a subset is fine: every given frame is one the entry recorded.
        check_upstream(&m, Stage::Stack, &frames[..1], &cfg).unwrap();

        // The user re-ran `calibrate` with another dark: the calibrated frame has new content.
        std::fs::write(&frames[1], b"bravo recalibrated").unwrap();
        let err = check_upstream(&m, Stage::Stack, &frames, &cfg).unwrap_err().to_string();
        assert_eq!(err, "measure.json was made from other frames (1 differ, first b.fit); re-run `starloom measure`");
        std::fs::write(&frames[1], b"bravo").unwrap();
        check_upstream(&m, Stage::Stack, &frames, &cfg).unwrap();

        // A frame that is not among the recorded ones.
        let stranger = frame(dir.path(), "z.fit", b"zulu");
        let err = check_upstream(&m, Stage::Register, &[frames[0].clone(), stranger], &cfg).unwrap_err().to_string();
        assert!(err.contains("(1 differ, first z.fit)") && err.contains("re-run `starloom measure`"), "{err}");

        // Another measurement config: the measurement subtree moved, so does the stored hash's comparison.
        let mut other = cfg.clone();
        other.measurement.max_stars += 10;
        let err = check_upstream(&m, Stage::Register, &frames, &other).unwrap_err().to_string();
        assert_eq!(err, "measure.json was made with another measurement config; re-run `starloom measure`");
        // Another registration config reaches `normalize` and `stack` through the register entry.
        let mut other = cfg.clone();
        other.registration.max_stars += 10;
        let err = check_upstream(&m, Stage::Stack, &frames, &other).unwrap_err().to_string();
        assert_eq!(err, "transforms.json was made with another registration config; re-run `starloom register`");
        // `measure` checks the calibrate entry: the frames must be its products, under the calibration config it has.
        let err = check_upstream(&m, Stage::Measure, &[stranger_path(dir.path())], &cfg).unwrap_err().to_string();
        assert!(err.contains("re-run `starloom calibrate`"), "{err}");
        let mut other = cfg.clone();
        other.calibration.hot_pixel_correction = !other.calibration.hot_pixel_correction;
        let err = check_upstream(&m, Stage::Measure, &frames, &other).unwrap_err().to_string();
        assert!(err.contains("another calibration config; re-run `starloom calibrate`"), "{err}");

        // A missing entry.
        let empty = Manifest::new(dir.path());
        let err = check_upstream(&empty, Stage::Register, &frames, &cfg).unwrap_err().to_string();
        assert!(err.contains("has no `measure` entry; run `starloom measure` first"), "{err}");
        // Normalize is only required when the rejection normalization is local.
        let mut no_ln = m.clone();
        no_ln.entries.retain(|e| e.stage != Stage::Normalize);
        assert!(check_upstream(&no_ln, Stage::Stack, &frames, &cfg).unwrap_err().to_string().contains("`normalize`"));
        let mut skip_ln = cfg.clone();
        skip_ln.normalization.rejection = RejectionNormalization::ScaleZeroOffset;
        check_upstream(&no_ln, Stage::Stack, &frames, &skip_ln).unwrap();
    }

    fn stranger_path(dir: &Path) -> PathBuf {
        dir.join("calibrated").join("elsewhere.fit")
    }

    #[test]
    fn check_upstream_lets_a_weight_mode_change_through() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = algorithm_preset(StackingPreset::Default);
        let (mut m, frames) = chain(dir.path(), &cfg);
        m.entries.push(entry(Stage::Register, &cfg, ids(&frames), vec![frame(dir.path(), "transforms.json", b"{}")]));
        m.entries.push(entry(Stage::Normalize, &cfg, ids(&frames), vec![frame(dir.path(), "ln.fit", b"{}")]));
        let mut other = cfg.clone();
        other.measurement.weight_mode = WeightMode::PsfSnr;
        other.measurement.keyword = "MYWEIGHT".to_string();
        other.measurement.formula.fwhm += 0.25;
        assert_ne!(cfg, other);
        for stage in [Stage::Register, Stage::Normalize, Stage::Stack] {
            check_upstream(&m, stage, &frames, &other).unwrap_or_else(|e| panic!("{stage:?}: {e:#}"));
        }
    }

    #[test]
    fn frames_or_upstream_takes_the_previous_stage_s_frames_and_names_what_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = algorithm_preset(StackingPreset::Default);
        let empty = Manifest::new(dir.path());
        let err = frames_or_upstream(&empty, Stage::Measure, Vec::new()).unwrap_err().to_string();
        assert!(
            err.contains("no frames given and") && err.contains("has no `calibrate` entry; give FRAMES or run `starloom calibrate`"),
            "{err}"
        );
        let err = frames_or_upstream(&empty, Stage::Stack, Vec::new()).unwrap_err().to_string();
        assert!(err.contains("has no `measure` entry; give FRAMES or run `starloom measure`"), "{err}");
        assert!(frames_or_upstream(&empty, Stage::Calibrate, Vec::new()).is_err());

        let (m, frames) = chain(dir.path(), &cfg);
        assert_eq!(frames_or_upstream(&m, Stage::Measure, Vec::new()).unwrap(), frames);
        for stage in [Stage::Register, Stage::Normalize, Stage::Stack] {
            assert_eq!(frames_or_upstream(&m, stage, Vec::new()).unwrap(), frames, "{stage:?}");
        }
        // Frames on the command line win, in the order given.
        let given = vec![frames[1].clone(), frames[0].clone()];
        assert_eq!(frames_or_upstream(&m, Stage::Stack, given.clone()).unwrap(), given);
    }

    #[test]
    fn a_manifest_entry_follows_its_products() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let cfg = algorithm_preset(StackingPreset::Default);
        let tmp = begin_writing(work).unwrap();
        let built = frame(&tmp, "measure.json", b"{}");
        let dest = work.join("measure").join("measure.json");
        let e = entry(Stage::Measure, &cfg, Vec::new(), vec![dest.clone()]);

        // The process is killed after the products were built and before the commit: nothing is in the manifest.
        assert!(Manifest::load(work).unwrap().entries.is_empty());
        assert!(!dest.exists());

        // A commit whose product cannot be moved writes no entry, and drops the stage's earlier one.
        let mut old = Manifest::load(work).unwrap();
        old.entries.push(entry(Stage::Measure, &cfg, Vec::new(), vec![built.clone()]));
        old.save().unwrap();
        let err = commit_stage(work, &[(tmp.join("never built.json"), dest.clone())], e.clone()).unwrap_err();
        assert!(format!("{err:#}").contains("never built.json"), "{err:#}");
        assert!(Manifest::load(work).unwrap().entry(Stage::Measure).is_none(), "an old entry must not outlive a failed rewrite");

        // A commit whose entry names a product that was never made writes no entry either.
        let mut ghost = e.clone();
        ghost.products.push(work.join("ghost.json"));
        let err = commit_stage(work, &[(built.clone(), dest.clone())], ghost).unwrap_err().to_string();
        assert!(err.contains("ghost.json"), "{err}");
        assert!(Manifest::load(work).unwrap().entries.is_empty());

        // The good commit: the product is in place, the entry follows, `.tmp` no longer holds the file.
        let built = frame(&tmp, "measure.json", b"{}");
        commit_stage(work, &[(built.clone(), dest.clone())], e.clone()).unwrap();
        assert!(dest.exists() && !built.exists());
        assert_eq!(Manifest::load(work).unwrap().entry(Stage::Measure), Some(&e));

        // Starting the next writing command clears what a killed one left.
        frame(&tmp, "half.part", b"x");
        begin_writing(work).unwrap();
        assert!(!tmp.join("half.part").exists() && tmp.is_dir());
    }

    #[test]
    fn a_commit_replaces_the_stage_entry_and_keeps_stage_order() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        let cfg = algorithm_preset(StackingPreset::Default);
        for (stage, name) in [(Stage::Register, "t.json"), (Stage::Measure, "m.json"), (Stage::Measure, "m2.json")] {
            let tmp = begin_writing(work).unwrap();
            let built = frame(&tmp, name, b"{}");
            let dest = work.join(name);
            commit_stage(work, &[(built, dest.clone())], entry(stage, &cfg, Vec::new(), vec![dest])).unwrap();
        }
        let m = Manifest::load(work).unwrap();
        let stages: Vec<Stage> = m.entries.iter().map(|e| e.stage).collect();
        assert_eq!(stages, [Stage::Measure, Stage::Register]);
        assert_eq!(display_name(&m.entry(Stage::Measure).unwrap().products[0]), "m2.json");
    }

    #[test]
    fn history_rows_append_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("fresh work");
        let t0 = DateTime::parse_from_rfc3339("2026-11-24T20:00:00Z").unwrap().with_timezone(&Utc);
        for (i, command) in ["calibrate", "measure", "stack"].into_iter().enumerate() {
            let row = HistoryRow {
                command: command.to_string(),
                args: vec!["--set".to_string(), format!("integration.minWeight=0.{i}")],
                started: t0 + chrono::Duration::seconds(i as i64 * 60),
                finished: t0 + chrono::Duration::seconds(i as i64 * 60 + 30),
                exit_code: if i == 2 { 2 } else { 0 },
                starloom_version: "0.2.0-dev".to_string(),
                config_hash: "0123456789abcdef".to_string(),
                log_file: work.join("logs").join(format!("{command}.jsonl")),
                summary: format!("starloom {command}: done"),
            };
            append_history(&work, row).unwrap();
        }
        let m = Manifest::load(&work).unwrap();
        let commands: Vec<&str> = m.history.iter().map(|r| r.command.as_str()).collect();
        assert_eq!(commands, ["calibrate", "measure", "stack"]);
        assert_eq!(m.history[2].exit_code, 2);
        assert!(m.history[0].finished < m.history[1].started);
        assert!(m.entries.is_empty());
    }

    #[test]
    fn a_manifest_of_another_version_is_refused_by_load_and_append() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(MANIFEST_FILE), "{\"format\":\"starloom-work\",\"version\":2,\"entries\":[],\"history\":[]}").unwrap();
        let err = Manifest::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("made by another starloom: starloom-work version 2, this build reads 1"), "{err}");
    }

    #[test]
    fn file_ids_follow_the_content_and_not_the_time() {
        let dir = tempfile::tempdir().unwrap();
        let p = frame(dir.path(), "a.fit", b"same bytes");
        let a = FileId::of(&p).unwrap();
        assert!(a.path.is_absolute());
        // Rewriting the same bytes later is the same input.
        std::fs::write(&p, b"same bytes").unwrap();
        assert_eq!(FileId::of(&p).unwrap(), a);
        std::fs::write(&p, b"other bytes").unwrap();
        assert_ne!(FileId::of(&p).unwrap(), a);
        assert!(FileId::of(&dir.path().join("absent.fit")).is_err());
    }

    /// A drive-letter path with spaces is stored as the text it is and read back as the same path, on every platform.
    #[test]
    fn windows_style_paths_round_trip_through_the_manifest_text() {
        let id = FileId {
            path: PathBuf::from(r"C:\Users\me\My Frames\Light 0001.FIT"),
            size: 7,
            sampled_hash: "0123456789abcdef".to_string(),
        };
        let text = serde_json::to_string(&id).unwrap();
        assert!(text.contains(r"C:\\Users\\me"), "{text}");
        assert_eq!(serde_json::from_str::<FileId>(&text).unwrap(), id);
    }
}
