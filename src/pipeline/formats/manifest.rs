//! The manifest (`starloom.json`): a journal of what each stage made in the working folder, and of the commands run
//! there.
//!
//! Manual mode (P2-R24): the manifest is for the user to read, and nothing checks it. An entry says "this stage, under
//! a config with these hashes and a build with these versions, made these products from these inputs"; a later
//! command never compares it with anything and never refuses a product as stale. The user decides what to re-run and
//! what to clean. Two things are read from it: the frames a command takes when it is given none
//! ([`frames_or_upstream`], P2-R20), and a calibrated frame's raw-light facts ([`source_facts`]).
//!
//! Each product is staged beside its destination ([`part_path`](super::part_path)) and renamed into place, and
//! [`commit_stage`] writes the entry after them, so an interrupted stage leaves no entry.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::FrameFacts;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{read_versioned, write_json_atomic};
use crate::pipeline::config::{config_hash, stage_hash};
use crate::pipeline::versions::recorded_versions;
use crate::pipeline::Stage;
use crate::stacking::config::AlgorithmConfig;

/// The manifest's file name, directly in the working folder.
pub const MANIFEST_FILE: &str = "starloom.json";
/// The manifest's `format` tag.
pub const MANIFEST_FORMAT: &str = "starloom-work";
/// The manifest's version. Another version is refused.
pub const MANIFEST_VERSION: u32 = 1;
/// The working folder's scratch folder: a writing command clears it when it starts ([`begin_writing`]) and may keep
/// files there that are not products.
pub const TMP_DIR: &str = ".tmp";

/// What one stage made, and what it made it from. Every field is a record; nothing compares it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    /// The stage.
    pub stage: Stage,
    /// The stage hash ([`stage_hash`]) of the config the stage ran with.
    pub stage_hash: String,
    /// The hash of the whole config the stage ran with ([`config_hash`]), the value its products' `SLM_CFG` holds.
    pub config_hash: String,
    /// Every version of the build that ran the stage, by constant name
    /// ([`all_versions`](crate::pipeline::versions::all_versions)).
    pub versions: BTreeMap<String, u32>,
    /// The files the stage read, absolute. For `measure`, `register` and `normalize` they are the frames, in the order
    /// the stage kept them (P2-R10), and [`frames_or_upstream`] hands that order to the next command.
    pub inputs: Vec<PathBuf>,
    /// The files the stage made, absolute. They exist when the entry is written.
    pub products: Vec<PathBuf>,
    /// For `calibrate`: each calibrated product's raw light's facts, keyed by the product. `measure` takes a frame's
    /// exposure and date from here (P2-R3), because the calibrated file no longer carries them.
    #[serde(default)]
    pub sources: BTreeMap<PathBuf, FrameFacts>,
}

impl Entry {
    /// An entry for `stage` under `cfg`: its stage hash, the config hash and every version of this build, with
    /// `inputs` and `products` as given and no `sources` (a `calibrate` stage fills them in).
    pub fn new(stage: Stage, cfg: &AlgorithmConfig, inputs: Vec<PathBuf>, products: Vec<PathBuf>) -> Entry {
        Entry {
            stage,
            stage_hash: stage_hash(stage, cfg),
            config_hash: config_hash(cfg),
            versions: recorded_versions(),
            inputs,
            products,
            sources: BTreeMap::new(),
        }
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
    /// The hash of the whole effective config ([`config_hash`]).
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
    /// One entry per stage that has run, in stage order: the latest run of each.
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
        write_json_atomic(&self.work_dir.join(MANIFEST_FILE), self)
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

/// Finishes a stage: renames each `(built, final)` pair of `moves` into place, then writes `entry`.
///
/// A product is built beside its destination ([`part_path`](super::part_path)), so the rename never crosses a volume
/// (an `-o` folder on another drive or a network share works); a `built` file in another folder is refused. The
/// stage's earlier entry is removed first, and the manifest saved without it, so a failure while the products are
/// moving leaves no entry. After the moves every path in `entry.products` must exist; only then is the entry written.
/// A file or folder already at a destination is replaced.
pub fn commit_stage(work_dir: &Path, moves: &[(PathBuf, PathBuf)], entry: Entry) -> Result<()> {
    if let Some((built, dest)) = moves.iter().find(|(built, dest)| built.parent() != dest.parent()) {
        bail!(
            "{} stage product {} must be built beside its destination {}",
            entry.stage.name(),
            built.display(),
            dest.display()
        );
    }
    let mut manifest = Manifest::load(work_dir)?;
    if manifest.entry(entry.stage).is_some() {
        manifest.entries.retain(|e| e.stage != entry.stage);
        manifest.save()?;
    }
    for (built, dest) in moves {
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
    Ok(if up == Stage::Calibrate { entry.products.clone() } else { entry.inputs.clone() })
}

/// The raw-light facts of each frame, keyed by the frame's absolute path: the facts the `calibrate` entry recorded for
/// it when the manifest has them, else the frame's own header facts (a frame calibrated elsewhere, or given from
/// another folder). `measure` takes exposure and date from here (P2-R3), and `finish` takes the map as its `sources`.
pub fn source_facts(manifest: &Manifest, frames: &[PathBuf]) -> Result<BTreeMap<PathBuf, FrameFacts>> {
    let recorded = manifest.entry(Stage::Calibrate).map(|e| &e.sources);
    let mut out = BTreeMap::new();
    for frame in frames {
        let abs = std::path::absolute(frame).with_context(|| format!("cannot make {} absolute", frame.display()))?;
        let facts = match recorded.and_then(|s| s.get(&abs)) {
            Some(facts) => facts.clone(),
            None => astroimage::header::read_frame_facts(&abs)
                .with_context(|| format!("cannot read the header facts of {}", abs.display()))?,
        };
        out.insert(abs, facts);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fits_writer::{write_fits_f32, Card, CardValue};
    use crate::pipeline::formats::{display_name, part_path};
    use crate::stacking::config::{algorithm_preset, StackingPreset};

    fn frame(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn cfg() -> AlgorithmConfig {
        algorithm_preset(StackingPreset::Default)
    }

    /// A working folder with a `calibrate` entry (two calibrated frames) and a `measure` entry over them.
    fn chain(dir: &Path) -> (Manifest, Vec<PathBuf>) {
        let cal = dir.join("calibrated");
        std::fs::create_dir_all(&cal).unwrap();
        let frames = vec![frame(&cal, "a.fit", b"alpha"), frame(&cal, "b.fit", b"bravo")];
        let report = frame(dir, "measure.json", b"{}");
        let mut m = Manifest::new(dir);
        m.entries.push(Entry::new(Stage::Calibrate, &cfg(), Vec::new(), frames.clone()));
        m.entries.push(Entry::new(Stage::Measure, &cfg(), frames.clone(), vec![report]));
        (m, frames)
    }

    #[test]
    fn an_entry_records_the_hashes_and_every_version_of_this_build() {
        let mut other = cfg();
        other.integration.min_weight += 0.1;
        let e = Entry::new(Stage::Measure, &other, vec![PathBuf::from("a.fit")], Vec::new());
        assert_eq!(e.stage_hash, stage_hash(Stage::Measure, &other));
        assert_eq!(e.config_hash, config_hash(&other));
        assert_ne!(e.config_hash, config_hash(&cfg()));
        assert_eq!(e.versions, recorded_versions());
        assert_eq!(e.versions["MEASURE_VERSION"], crate::star_measure::MEASURE_VERSION);
        assert!(e.sources.is_empty());
    }

    #[test]
    fn frames_or_upstream_takes_the_previous_stage_s_frames_and_names_what_to_run() {
        let dir = tempfile::tempdir().unwrap();
        let empty = Manifest::new(dir.path());
        let err = frames_or_upstream(&empty, Stage::Measure, Vec::new()).unwrap_err().to_string();
        assert!(
            err.contains("no frames given and") && err.contains("has no `calibrate` entry; give FRAMES or run `starloom calibrate`"),
            "{err}"
        );
        let err = frames_or_upstream(&empty, Stage::Stack, Vec::new()).unwrap_err().to_string();
        assert!(err.contains("has no `measure` entry; give FRAMES or run `starloom measure`"), "{err}");
        assert!(frames_or_upstream(&empty, Stage::Calibrate, Vec::new()).is_err());

        let (m, frames) = chain(dir.path());
        assert_eq!(frames_or_upstream(&m, Stage::Measure, Vec::new()).unwrap(), frames);
        for stage in [Stage::Register, Stage::Normalize, Stage::Stack] {
            assert_eq!(frames_or_upstream(&m, stage, Vec::new()).unwrap(), frames, "{stage:?}");
        }
        // Frames on the command line win, in the order given, whatever the manifest holds.
        let stranger = frame(dir.path(), "z.fit", b"zulu");
        let given = vec![stranger, frames[0].clone()];
        assert_eq!(frames_or_upstream(&m, Stage::Stack, given.clone()).unwrap(), given);
    }

    fn tiny_fits(path: &Path, exptime: f64, date: &str) {
        let cards = vec![
            Card::new("EXPTIME", CardValue::Real(exptime)).unwrap(),
            Card::new("DATE-OBS", CardValue::Str(date.to_string())).unwrap(),
        ];
        write_fits_f32(path, 2, 2, 1, &[0.0; 4], &cards).unwrap();
    }

    /// I1: a frame the manifest's `calibrate` entry made takes its raw light's facts from the entry; a frame calibrated
    /// elsewhere is taken all the same, with its own header's facts.
    #[test]
    fn source_facts_come_from_the_calibrate_entry_else_from_the_frame_itself() {
        let dir = tempfile::tempdir().unwrap();
        let ours = dir.path().join("ours.fits");
        let elsewhere = dir.path().join("calibrated elsewhere.fits");
        tiny_fits(&ours, 1.0, "2024-01-01T00:00:00");
        tiny_fits(&elsewhere, 120.0, "2024-01-02T03:04:05");
        let mut raw = FrameFacts::default();
        raw.exptime_s = Some(300.0);
        raw.date_obs = Some("2024-01-15T20:30:45".to_string());
        let mut cal = Entry::new(Stage::Calibrate, &cfg(), Vec::new(), vec![ours.clone()]);
        cal.sources.insert(ours.clone(), raw.clone());
        let mut m = Manifest::new(dir.path());
        m.entries.push(cal);

        let facts = source_facts(&m, &[ours.clone(), elsewhere.clone()]).unwrap();
        assert_eq!(facts[&ours], raw, "the raw light's facts, not the calibrated file's");
        assert_eq!(facts[&elsewhere].exptime_s, Some(120.0));
        assert_eq!(facts[&elsewhere].date_obs.as_deref(), Some("2024-01-02T03:04:05"));
        // No calibrate entry at all: every frame's own facts.
        let facts = source_facts(&Manifest::new(dir.path()), &[ours.clone()]).unwrap();
        assert_eq!(facts[&ours].exptime_s, Some(1.0));
        // A frame that cannot be read names itself.
        let err = format!("{:#}", source_facts(&m, &[dir.path().join("absent.fits")]).unwrap_err());
        assert!(err.contains("absent.fits"), "{err}");
    }

    #[test]
    fn a_manifest_entry_follows_its_products() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        begin_writing(work).unwrap();
        let dest = work.join("measure").join("measure.json");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let built = frame(dest.parent().unwrap(), &display_name(&part_path(&dest)), b"{}");
        assert_eq!(built, part_path(&dest));
        let e = Entry::new(Stage::Measure, &cfg(), Vec::new(), vec![dest.clone()]);

        // The process is killed after the products were built and before the commit: nothing is in the manifest.
        assert!(Manifest::load(work).unwrap().entries.is_empty());
        assert!(!dest.exists());

        // A commit whose product cannot be moved writes no entry, and drops the stage's earlier one.
        let mut old = Manifest::load(work).unwrap();
        old.entries.push(Entry::new(Stage::Measure, &cfg(), Vec::new(), vec![built.clone()]));
        old.save().unwrap();
        let never = dest.with_file_name("never built.json.part");
        let err = commit_stage(work, &[(never, dest.clone())], e.clone()).unwrap_err();
        assert!(format!("{err:#}").contains("never built.json"), "{err:#}");
        assert!(Manifest::load(work).unwrap().entry(Stage::Measure).is_none(), "an old entry must not outlive a failed rewrite");

        // A commit whose entry names a product that was never made writes no entry either.
        let mut ghost = e.clone();
        ghost.products.push(work.join("ghost.json"));
        let err = commit_stage(work, &[(built.clone(), dest.clone())], ghost).unwrap_err().to_string();
        assert!(err.contains("ghost.json"), "{err}");
        assert!(Manifest::load(work).unwrap().entries.is_empty());

        // The good commit: the product is in place, the entry follows, the staged file is gone.
        let built = frame(dest.parent().unwrap(), &display_name(&part_path(&dest)), b"{}");
        commit_stage(work, &[(built.clone(), dest.clone())], e.clone()).unwrap();
        assert!(dest.exists() && !built.exists());
        assert_eq!(Manifest::load(work).unwrap().entry(Stage::Measure), Some(&e));

        // Starting the next writing command clears what a killed one left in the scratch folder.
        let tmp = work.join(TMP_DIR);
        frame(&tmp, "half.part", b"x");
        begin_writing(work).unwrap();
        assert!(!tmp.join("half.part").exists() && tmp.is_dir());
    }

    /// I6: a product is staged in its destination's folder, so an output folder outside the working folder (on any
    /// volume) is committed by a rename inside that folder; a product staged anywhere else is refused.
    #[test]
    fn a_product_outside_the_work_folder_is_staged_and_renamed_in_its_own_folder() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let out = dir.path().join("elsewhere").join("masters");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&out).unwrap();
        let tmp = begin_writing(&work).unwrap();
        let dest = out.join("master light.fits");
        let staged = part_path(&dest);
        assert_eq!(staged.parent(), dest.parent());
        std::fs::write(&staged, b"pixels").unwrap();
        let e = Entry::new(Stage::Stack, &cfg(), Vec::new(), vec![dest.clone()]);
        commit_stage(&work, &[(staged.clone(), dest.clone())], e.clone()).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"pixels");
        assert!(!staged.exists());

        let in_tmp = frame(&tmp, "master light.fits", b"pixels");
        let err = commit_stage(&work, &[(in_tmp, dest.clone())], e).unwrap_err().to_string();
        assert!(err.contains("must be built beside its destination"), "{err}");
        assert!(Manifest::load(&work).unwrap().entry(Stage::Stack).is_some(), "a refused commit touches nothing");
    }

    #[test]
    fn a_commit_replaces_the_stage_entry_and_keeps_stage_order() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path();
        for (stage, name) in [(Stage::Register, "t.json"), (Stage::Measure, "m.json"), (Stage::Measure, "m2.json")] {
            begin_writing(work).unwrap();
            let dest = work.join(name);
            let built = frame(work, &display_name(&part_path(&dest)), b"{}");
            commit_stage(work, &[(built, dest.clone())], Entry::new(stage, &cfg(), Vec::new(), vec![dest])).unwrap();
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

    /// A drive-letter path with spaces is stored as the text it is and read back as the same path, on every platform.
    #[test]
    fn windows_style_paths_round_trip_through_the_manifest_text() {
        let mut e = Entry::new(Stage::Measure, &cfg(), vec![PathBuf::from(r"C:\Users\me\My Frames\Light 0001.FIT")], Vec::new());
        e.sources.insert(PathBuf::from(r"D:\cal\Light 0001.fits"), FrameFacts::default());
        let text = serde_json::to_string(&e).unwrap();
        assert!(text.contains(r"C:\\Users\\me"), "{text}");
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), e);
    }
}
