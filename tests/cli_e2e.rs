//! The `starloom` command end to end: the built binary, run on small synthetic frames in a temporary folder.
//!
//! Every test names its files with letters, digits, `_` and `-`, builds paths with `Path::join`, and never compares a
//! path with log text, so it passes on macOS, Linux and Windows.
#![cfg(feature = "cli")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use starloom::fits_writer::write_fits_f32;

const BIN: &str = env!("CARGO_BIN_EXE_starloom");

// ---------------------------------------------------------------------------------------------------------------
// Running the command
// ---------------------------------------------------------------------------------------------------------------

/// Runs `starloom` with `args` on two threads.
fn starloom<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    Command::new(BIN).args(args).args(["--threads", "2"]).output().expect("the binary runs")
}

/// Runs `starloom -w <work>` with the arguments `args`.
fn in_work<S: AsRef<std::ffi::OsStr>>(work: &Path, args: &[S]) -> Output {
    Command::new(BIN).arg("-w").arg(work).args(args).args(["--threads", "2"]).output().expect("the binary runs")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("the command exited")
}

/// Asserts the command succeeded, showing its stderr when it did not.
fn ok(o: &Output) {
    assert_eq!(code(o), 0, "stderr:\n{}", stderr(o));
}

/// The LN-free config the shorter chains run with: the default rejection normalization is local, which needs `normalize`.
const NO_LN: [&str; 2] = ["--set", "normalization.rejection=scaleZeroOffset"];

fn manifest(work: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(work.join("starloom.json")).expect("the manifest exists")).unwrap()
}

fn history(work: &Path) -> Vec<Value> {
    manifest(work)["history"].as_array().cloned().unwrap_or_default()
}

fn log_files(work: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(work.join("logs"))
        .map(|d| d.map(|e| e.unwrap().path()).collect())
        .unwrap_or_default();
    files.sort();
    files
}

// ---------------------------------------------------------------------------------------------------------------
// Synthetic frames
// ---------------------------------------------------------------------------------------------------------------

fn noise(seed: u64, i: usize) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % 10_000) as f32 / 10_000.0
}

/// The 150-star layout of the pipeline's fixtures: a jittered 15 x 10 grid of a 768 x 512 plane.
fn grid_stars() -> Vec<(f64, f64, f64)> {
    (0..150)
        .map(|n| {
            let (i, j) = ((n % 15) as f64, (n / 15) as f64);
            (
                40.0 + i * 48.0 + (f64::from(noise(1, n)) - 0.5) * 16.0,
                40.0 + j * 48.0 + (f64::from(noise(2, n)) - 0.5) * 16.0,
                0.1 + 0.2 * f64::from(noise(3, n)),
            )
        })
        .collect()
}

const SHIFTS: [(f64, f64); 6] = [(0.0, 0.0), (3.37, -2.81), (-4.2, 5.55), (1.25, 1.9), (-2.6, -3.3), (5.1, 0.45)];

/// The star field moved by `shift`, with its own noise, as a float plane.
fn field(k: usize) -> Vec<f32> {
    let (dx, dy) = SHIFTS[k];
    let moved: Vec<(f64, f64, f64)> = grid_stars().iter().map(|&(x, y, a)| (x + dx, y + dy, a)).collect();
    let mut data = starloom::test_support::gaussian_field(768, 512, &moved, 1.8, 0.08);
    starloom::test_support::add_noise(&mut data, 0.002, 40 + k as u64);
    data
}

/// `n` calibrated frames (float FITS, `c_Light_0000.fits` ...) in `dir`.
fn calibrated_frames(dir: &Path, n: usize) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|k| {
            let path = dir.join(format!("c_Light_{k:04}.fits"));
            write_fits_f32(&path, 768, 512, 1, &field(k), &[]).unwrap();
            path
        })
        .collect()
}

/// A 16-bit FITS (BITPIX 16, BZERO 32768) the way a camera writes it.
fn write_u16(path: &Path, w: usize, h: usize, values: &[u16], extra: &[(&str, String)]) -> PathBuf {
    let mut cards: Vec<String> = vec![
        format!("{:<8}= {:>20}", "SIMPLE", "T"),
        format!("{:<8}= {:>20}", "BITPIX", 16),
        format!("{:<8}= {:>20}", "NAXIS", 2),
        format!("{:<8}= {:>20}", "NAXIS1", w),
        format!("{:<8}= {:>20}", "NAXIS2", h),
        format!("{:<8}= {:>20}", "BZERO", 32768),
        format!("{:<8}= {:>20}", "BSCALE", 1),
    ];
    for (k, v) in extra {
        cards.push(format!("{k:<8}= {v}"));
    }
    cards.push("END".to_string());
    let mut bytes = Vec::new();
    for c in cards {
        bytes.extend_from_slice(format!("{c:<80}").as_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for &v in values {
        bytes.extend_from_slice(&((i32::from(v) - 32768) as i16).to_be_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    std::fs::write(path, bytes).unwrap();
    path.to_path_buf()
}

/// `n` raw 16-bit lights of the star field in `dir/lights`, and eight raw darks in `dir/darks`.
fn raw_set(dir: &Path, n: usize) -> (PathBuf, PathBuf) {
    let (lights, darks) = (dir.join("lights"), dir.join("darks"));
    std::fs::create_dir_all(&lights).unwrap();
    std::fs::create_dir_all(&darks).unwrap();
    for k in 0..n {
        let values: Vec<u16> = field(k).iter().map(|&v| (100.0 + f64::from(v.clamp(0.0, 1.0)) * 50_000.0).round() as u16).collect();
        let extra = [("EXPTIME", "300.0".to_string()), ("DATE-OBS", format!("'2024-01-15T20:{:02}:00'", 10 + k))];
        write_u16(&lights.join(format!("Light_{k:04}.fits")), 768, 512, &values, &extra);
    }
    for k in 0..8 {
        let v: Vec<u16> = (0..768 * 512).map(|i| 100 + (6.0 * noise(k as u64 + 1, i)) as u16).collect();
        let extra = [("EXPTIME", "300.0".to_string()), ("IMAGETYP", "'Dark'".to_string())];
        write_u16(&darks.join(format!("Dark_{k:04}.fits")), 768, 512, &v, &extra);
    }
    (lights, darks)
}

/// `measure`, then `register` onto the first frame, in `work`, on `frames` given explicitly; asserts both succeeded.
fn measured_and_registered(work: &Path, frames: &[PathBuf]) {
    let mut args: Vec<std::ffi::OsString> = vec!["measure".into()];
    args.extend(frames.iter().map(|f| f.clone().into_os_string()));
    ok(&in_work(work, &args));
    let mut args: Vec<std::ffi::OsString> = vec!["register".into(), "--reference".into(), frames[0].clone().into_os_string()];
    args.extend(frames.iter().map(|f| f.clone().into_os_string()));
    ok(&in_work(work, &args));
}

// ---------------------------------------------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn config_accepts_an_athenaeum_stacking_config_and_names_the_ignored_keys() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("stacking-config-default.json");
    let out = starloom(&["config".as_ref(), "show".as_ref(), "--config".as_ref(), fixture.as_os_str()]);
    ok(&out);
    assert!(stderr(&out).contains("grouping, output, paths, reference, version"), "stderr:\n{}", stderr(&out));
    let shown: Value = serde_json::from_str(&stdout(&out)).expect("config show prints JSON");
    let stored: Value = serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
    let sections = ["calibration", "measurement", "selection", "registration", "normalization", "integration", "drizzle"];
    assert_eq!(shown.as_object().unwrap().len(), sections.len(), "config show prints the seven algorithm subtrees");
    for section in sections {
        assert_eq!(shown[section], stored[section], "section {section}");
    }
}

#[test]
fn facts_and_config_show_make_no_work_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 1);
    let work = tmp.path().join("no-work");
    ok(&in_work(&work, &["facts".as_ref(), frames[0].as_os_str()]));
    ok(&in_work(&work, &["config", "show"]));
    assert!(!work.exists(), "a read-only command makes no working folder, log or history");
}

#[test]
fn psf_flag_sets_the_measurement_model() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 3);
    let measure_with = |name: &str, psf: &str| -> Value {
        let work = tmp.path().join(name);
        let mut args: Vec<std::ffi::OsString> = vec!["measure".into(), "--psf".into(), psf.into()];
        args.extend(frames.iter().map(|f| f.clone().into_os_string()));
        ok(&in_work(&work, &args));
        let report: Value = serde_json::from_str(&std::fs::read_to_string(work.join("measure").join("measure.json")).unwrap()).unwrap();
        report["config"]["measurement"]["psfModel"].clone()
    };
    assert_eq!(measure_with("w8", "moffat=8"), serde_json::json!({"moffat": 8.0}));
    assert_eq!(measure_with("w4m", "moffat4"), serde_json::json!("moffat4"));
    assert_eq!(measure_with("w4", "moffat=4"), serde_json::json!({"moffat": 4.0}));

    let work = tmp.path().join("w1");
    let bad = in_work(&work, &["measure".as_ref(), "--psf".as_ref(), "moffat=1".as_ref(), frames[0].as_os_str()]);
    assert_eq!(code(&bad), 2);
    assert!(stderr(&bad).contains("[1.5, 100]"), "the bounds are named:\n{}", stderr(&bad));
    assert!(!work.join("measure").exists(), "nothing was measured");
}

// ---------------------------------------------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn the_chain_of_commands_writes_a_master_and_a_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let (lights, darks) = raw_set(tmp.path(), 5);
    let work = tmp.path().join("work");

    ok(&in_work(&work, &["master".as_ref(), "dark".as_ref(), darks.as_os_str()]));
    let dark = work.join("masters").join("dark.fits");
    assert!(dark.is_file(), "the master dark is in <work>/masters");

    // `calibrate` is given lights; every later command is given no frames.
    ok(&in_work(&work, &["calibrate".as_ref(), lights.as_os_str(), "--dark".as_ref(), dark.as_os_str()]));
    for stage in ["measure", "register", "normalize", "stack"] {
        ok(&in_work(&work, &[stage]));
    }

    let master = work.join("stack").join("master.fits");
    assert!(master.is_file());
    let m = manifest(&work);
    let stages: Vec<&str> = m["entries"].as_array().unwrap().iter().map(|e| e["stage"].as_str().unwrap()).collect();
    assert_eq!(stages, ["calibrate", "measure", "register", "normalize", "stack"]);
    let commands: Vec<String> = history(&work).iter().map(|h| h["command"].as_str().unwrap().to_string()).collect();
    assert_eq!(commands, ["master", "calibrate", "measure", "register", "normalize", "stack"]);
    assert!(history(&work).iter().all(|h| h["exitCode"] == 0));
}

#[test]
fn run_with_a_plan_makes_the_master_and_a_history_row() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, _) = raw_set(tmp.path(), 4);
    // The plan's paths are relative to the plan's own folder.
    std::fs::write(tmp.path().join("plan.toml"), "[[group]]\nlights = \"lights\"\ndark = \"darks\"\n").unwrap();
    let work = tmp.path().join("work");
    let out = in_work(&work, &["run".as_ref(), "--plan".as_ref(), tmp.path().join("plan.toml").as_os_str(), "--set".as_ref(), "normalization.rejection=scaleZeroOffset".as_ref()]);
    ok(&out);
    assert!(work.join("stack").join("master.fits").is_file());
    assert!(work.join("calibrated").join("c_Light_0000.fits").is_file());
    let rows = history(&work);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["command"], "run");
    assert!(stderr(&out).lines().last().unwrap().starts_with("starloom run: 4 frames"), "{}", stderr(&out));
}

#[test]
fn measure_without_frames_and_no_calibrate_names_what_to_run() {
    let tmp = tempfile::tempdir().unwrap();
    let out = in_work(&tmp.path().join("work"), &["measure"]);
    assert_eq!(code(&out), 2);
    let err = stderr(&out);
    assert!(err.contains("no frames given") && err.contains("has no `calibrate` entry") && err.contains("run `starloom calibrate`"), "{err}");
}

#[test]
fn stack_with_a_frame_never_registered_fails_naming_it_and_register() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 4);
    let work = tmp.path().join("work");
    ok(&in_work(&work, &["measure".as_ref(), frames[0].as_os_str(), frames[1].as_os_str(), frames[2].as_os_str(), frames[3].as_os_str()]));
    // Only three of the four are registered.
    ok(&in_work(&work, &["register".as_ref(), "--reference".as_ref(), frames[0].as_os_str(), frames[0].as_os_str(), frames[1].as_os_str(), frames[2].as_os_str()]));

    let mut args: Vec<std::ffi::OsString> = vec!["stack".into()];
    args.extend(frames.iter().map(|f| f.clone().into_os_string()));
    args.extend(NO_LN.iter().map(Into::into));
    let out = in_work(&work, &args);
    assert_eq!(code(&out), 2, "a frame with no transform is the input's fault: {}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("c_Light_0003.fits") && err.contains("starloom register"), "{err}");
    assert!(!work.join("stack").join("master.fits").exists());
}

#[test]
fn stack_with_another_weight_mode_re_weighs_from_the_stored_measurements() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 4);
    let work = tmp.path().join("work");
    measured_and_registered(&work, &frames);

    let mut args: Vec<std::ffi::OsString> = vec!["stack".into(), "--weights".into(), "psfSnr".into()];
    args.extend(NO_LN.iter().map(Into::into));
    let out = in_work(&work, &args);
    ok(&out);
    assert!(stderr(&out).contains("re-weighed from the stored measurements"), "{}", stderr(&out));
    let weights: Value = serde_json::from_str(&std::fs::read_to_string(work.join("stack").join("weights.json")).unwrap()).unwrap();
    assert_eq!(weights["settings"]["weightMode"], "psfSnr");
    let measured: Value = serde_json::from_str(&std::fs::read_to_string(work.join("measure").join("measure.json")).unwrap()).unwrap();
    assert_eq!(measured["config"]["measurement"]["weightMode"], "psfSignalWeight", "measure's own record is untouched");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn os(args: &[&std::ffi::OsStr]) -> Vec<std::ffi::OsString> {
    args.iter().map(|a| (*a).to_os_string()).collect()
}

#[test]
fn explicit_product_paths_override_the_working_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 4);
    let (work_a, work_b, work_c) = (tmp.path().join("a"), tmp.path().join("b"), tmp.path().join("c"));
    let named = |extra: Vec<std::ffi::OsString>| -> Vec<std::ffi::OsString> {
        extra.into_iter().chain(frames.iter().map(|f| f.clone().into_os_string())).collect()
    };
    // The chain in its own folder: the master and its weights are the reference.
    ok(&in_work(&work_a, &named(vec!["measure".into()])));
    let measure_json = work_a.join("measure").join("measure.json");
    ok(&in_work(&work_a, &named(vec!["register".into(), "--reference".into(), frames[0].clone().into_os_string()])));
    let mut args = named(vec!["stack".into()]);
    args.extend(NO_LN.iter().map(Into::into));
    ok(&in_work(&work_a, &args));
    let (master_a, weights_a) = (work_a.join("stack").join("master.fits"), work_a.join("stack").join("weights.json"));

    // `register` in another folder reads the measurement it is given, and its star fits from beside it (nothing is
    // detected again), taking the frames the report lists when it is given none.
    let out = in_work(
        &work_b,
        &os(&["register".as_ref(), "--reference".as_ref(), frames[0].as_os_str(), "--measure".as_ref(), measure_json.as_os_str()]),
    );
    ok(&out);
    assert!(!stderr(&out).contains("no readable fits artifact"), "register reused Measure's fits:\n{}", stderr(&out));
    assert!(!work_b.join("measure").exists());
    // `stack` in a third folder reads both and writes where `-o` says; the weights (which read the `.athf` files) and
    // the master are the in-folder chain's.
    let out_dir = tmp.path().join("elsewhere");
    let mut args = os(&[
        "stack".as_ref(),
        "--measure".as_ref(),
        measure_json.as_os_str(),
        "--transforms".as_ref(),
        work_b.join("register").join("transforms.json").as_os_str(),
        "-o".as_ref(),
        out_dir.as_os_str(),
    ]);
    args.extend(NO_LN.iter().map(Into::into));
    let out = in_work(&work_c, &args);
    ok(&out);
    assert!(!stderr(&out).contains("without a readable .athf"), "{}", stderr(&out));
    assert!(!work_c.join("stack").join("master.fits").exists());
    assert!(!work_c.join("measure").exists() && !work_c.join("register").exists());
    assert_eq!(std::fs::read(work_c.join("stack").join("weights.json")).unwrap(), std::fs::read(&weights_a).unwrap(), "the same weights");
    assert_eq!(std::fs::read(out_dir.join("master.fits")).unwrap(), std::fs::read(&master_a).unwrap(), "the same master");
}

#[test]
fn stack_reads_the_grids_from_the_ln_folder_it_is_given() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 4);
    let (work_a, work_b) = (tmp.path().join("a"), tmp.path().join("b"));
    measured_and_registered(&work_a, &frames);
    ok(&in_work(&work_a, &["normalize"]));
    ok(&in_work(&work_a, &["stack"]));
    let master_a = std::fs::read(work_a.join("stack").join("master.fits")).unwrap();

    // The folder is copied and the original removed: the paths `normalize.json` recorded lead nowhere.
    let copy = tmp.path().join("ln-copy");
    copy_dir(&work_a.join("ln"), &copy);
    std::fs::remove_dir_all(work_a.join("ln")).unwrap();
    let out = in_work(
        &work_b,
        &os(&[
            "stack".as_ref(),
            "--measure".as_ref(),
            work_a.join("measure").join("measure.json").as_os_str(),
            "--transforms".as_ref(),
            work_a.join("register").join("transforms.json").as_os_str(),
            "--ln".as_ref(),
            copy.as_os_str(),
        ]),
    );
    ok(&out);
    assert_eq!(std::fs::read(work_b.join("stack").join("master.fits")).unwrap(), master_a, "the same master");
}

// ---------------------------------------------------------------------------------------------------------------
// Logs, history, exit codes
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn every_command_writes_a_jsonl_log_a_history_row_and_a_summary_line() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 3);
    let work = tmp.path().join("work");
    let mut args: Vec<std::ffi::OsString> = vec!["measure".into()];
    args.extend(frames.iter().map(|f| f.clone().into_os_string()));
    let out = in_work(&work, &args);
    ok(&out);

    let logs = log_files(&work);
    assert_eq!(logs.len(), 1);
    let name = logs[0].file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.ends_with("Z-measure.jsonl") && name.len() == "20261125T101010.123Z-measure.jsonl".len(), "{name}");
    let text = std::fs::read_to_string(&logs[0]).unwrap();
    let events: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("log line is not JSON ({e}): {line}")))
        .collect();
    assert!(!events.is_empty());
    // The file always holds debug, and `--threads 2` (which the helper passes) is what the pool was built with.
    let pool = events.iter().find(|e| e["fields"]["message"] == "thread pool built").expect("a `thread pool built` event");
    assert_eq!((pool["level"].as_str(), pool["fields"]["threads"].as_u64()), (Some("DEBUG"), Some(2)), "{pool}");
    assert!(events.iter().any(|e| e["fields"]["message"] == "effective config"), "the effective config is logged at debug");

    let rows = history(&work);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["exitCode"], 0);
    assert_eq!(rows[0]["command"], "measure");
    assert_eq!(Path::new(rows[0]["logFile"].as_str().unwrap()).file_name().unwrap().to_string_lossy(), name);
    let last = stderr(&out).lines().last().unwrap().to_string();
    assert!(last.starts_with("starloom measure:") && last.contains("3 frames (0 excluded)"), "{last}");
    assert_eq!(rows[0]["summary"], last.as_str());
}

#[test]
fn a_refused_command_leaves_its_log_and_a_history_row_with_code_2() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    let out = in_work(&work, &["measure", "--set", "measurement.noSuchKey=1"]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("noSuchKey"), "{}", stderr(&out));
    assert_eq!(log_files(&work).len(), 1);
    // The log says why the command ended, not only that it did.
    let log = std::fs::read_to_string(&log_files(&work)[0]).unwrap();
    let why = log
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|e| e["level"] == "ERROR")
        .expect("an error event in the log");
    assert!(why["fields"]["text"].as_str().unwrap().contains("noSuchKey"), "{why}");
    let rows = history(&work);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["exitCode"], 2);
    assert!(stderr(&out).lines().last().unwrap().starts_with("starloom measure: refused"), "{}", stderr(&out));
}

#[test]
fn facts_json_prints_one_object_per_file() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 1);
    let gz = tmp.path().join("cal").join("copy.fits.gz");
    let mut enc = flate2::write::GzEncoder::new(std::fs::File::create(&gz).unwrap(), flate2::Compression::fast());
    enc.write_all(&std::fs::read(&frames[0]).unwrap()).unwrap();
    enc.finish().unwrap();

    let out = starloom(&["facts".as_ref(), "--json".as_ref(), frames[0].as_os_str(), gz.as_os_str()]);
    ok(&out);
    let text = stdout(&out);
    let objects: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).expect("one JSON object per line")).collect();
    assert_eq!(objects.len(), 2);
    for o in &objects {
        assert!(o.is_object() && o["file"].is_string());
        assert_eq!((o["width"].as_i64(), o["height"].as_i64()), (Some(768), Some(512)));
    }
    assert!(objects[1]["file"].as_str().unwrap().ends_with("copy.fits.gz"));
}

#[test]
fn a_refusal_exits_with_code_2_and_names_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    let missing = tmp.path().join("missing_frame.fits");
    let out = in_work(&work, &["measure".as_ref(), missing.as_os_str()]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("missing_frame.fits"), "{}", stderr(&out));

    let wrong = tmp.path().join("notes.txt");
    std::fs::write(&wrong, "not a frame").unwrap();
    let out = in_work(&work, &["measure".as_ref(), wrong.as_os_str()]);
    assert_eq!(code(&out), 2);
    assert!(stderr(&out).contains("notes.txt"), "{}", stderr(&out));
    assert_eq!(history(&work).iter().filter(|h| h["exitCode"] == 2).count(), 2);
}

#[test]
fn the_version_flag_prints_the_crate_version() {
    let out = starloom(&["--version"]);
    ok(&out);
    assert_eq!(stdout(&out).trim(), format!("starloom {}", env!("CARGO_PKG_VERSION")));
}

// ---------------------------------------------------------------------------------------------------------------
// Review round: warnings, exit codes by kind, the shell-glob refusal, merged flags
// ---------------------------------------------------------------------------------------------------------------

#[test]
fn a_warning_shows_once_on_stderr_and_once_in_the_log() {
    let tmp = tempfile::tempdir().unwrap();
    // Two frames: fewer than the three a stack needs, which `measure` warns about.
    let frames = calibrated_frames(&tmp.path().join("cal"), 2);
    let work = tmp.path().join("work");
    let mut args: Vec<std::ffi::OsString> = vec!["measure".into()];
    args.extend(frames.iter().map(|f| f.clone().into_os_string()));
    let out = in_work(&work, &args);
    ok(&out);
    let sentence = "frames are included; a stack needs at least 3";
    assert_eq!(stderr(&out).matches(sentence).count(), 1, "stderr:\n{}", stderr(&out));
    let log = std::fs::read_to_string(&log_files(&work)[0]).unwrap();
    let events: Vec<Value> = log.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let warned = events.iter().filter(|e| e["fields"]["text"].as_str().is_some_and(|t| t.contains(sentence))).count();
    assert_eq!(warned, 1, "the log holds the warning once");
}

#[test]
fn the_exit_code_follows_the_kind_of_the_error() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = calibrated_frames(&tmp.path().join("cal"), 3);
    let frame = frames[0].as_os_str();

    // A manifest that is not a starloom file is the input's fault; one that cannot be read at all is the machine's.
    let bad = tmp.path().join("bad");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("starloom.json"), "this is not json").unwrap();
    assert_eq!(code(&in_work(&bad, &["measure".as_ref(), frame])), 2);
    let unreadable = tmp.path().join("unreadable");
    std::fs::create_dir_all(unreadable.join("starloom.json")).unwrap();
    let out = in_work(&unreadable, &["measure".as_ref(), frame]);
    assert_eq!(code(&out), 1, "{}", stderr(&out));

    // A measurement report that is not one is a refusal.
    let report = tmp.path().join("measure.json");
    std::fs::write(&report, "{\"format\": \"something-else\"}").unwrap();
    let out = in_work(&tmp.path().join("w"), &["register".as_ref(), "--measure".as_ref(), report.as_os_str(), frame]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
}

#[test]
fn a_dark_given_as_a_light_names_the_shell_glob() {
    let tmp = tempfile::tempdir().unwrap();
    let (lights, darks) = raw_set(tmp.path(), 3);
    // `calibrate lights --dark darks/*.fits` as a shell passes it: `--dark` gets the first dark, the others are lights.
    let mut args: Vec<std::ffi::OsString> = vec!["calibrate".into(), lights.clone().into_os_string(), "--dark".into()];
    for k in 0..4 {
        args.push(darks.join(format!("Dark_{k:04}.fits")).into_os_string());
    }
    let work = tmp.path().join("work");
    let out = in_work(&work, &args);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("Dark_0001.fits") && err.contains("a shell expanded") && err.contains("quote it"), "{err}");
    assert!(!work.join("calibrated").exists(), "nothing was calibrated");

    // Quoted, the program expands the glob itself and the same command works.
    let glob = darks.join("*.fits");
    let out = in_work(&work, &["calibrate".as_ref(), lights.as_os_str(), "--dark".as_ref(), glob.as_os_str()]);
    ok(&out);
}

#[test]
fn set_and_v_before_and_after_the_subcommand_are_merged() {
    let out = starloom(&["--set", "measurement.weightMode=psfSnr", "config", "show", "--set", "integration.minWeight=0.25"]);
    ok(&out);
    let shown: Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(shown["measurement"]["weightMode"], "psfSnr", "the set before the subcommand is kept");
    assert_eq!(shown["integration"]["minWeight"], 0.25, "the set after it too");

    // Later wins, as for two `--set`s on one side: the one after the subcommand is the later.
    let out = starloom(&["--set", "integration.minWeight=0.1", "config", "show", "--set", "integration.minWeight=0.3"]);
    ok(&out);
    assert_eq!(serde_json::from_str::<Value>(&stdout(&out)).unwrap()["integration"]["minWeight"], 0.3);
}
