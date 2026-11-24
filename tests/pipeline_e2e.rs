//! The `master` and `calibrate` stages end to end, on small synthetic frames written to a temporary folder.
//!
//! Every test names its files with letters, digits, `_` and `-`, builds paths with `Path::join` and never compares a
//! path with log text, so it passes on macOS, Linux and Windows.
#![cfg(feature = "pipeline")]

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use astroimage::header::{header_key_map, header_text, read_frame_facts};
use starloom::calibration_library::flat_cards::{SlmFlatCards, SLM_FNB, SLM_FNG, SLM_FNR, SLM_FNRM};
use starloom::calibration_library::light_cal::FlatCardReader;
use starloom::calibration_library::options::CalibratedLightOptions;
use starloom::calibration_library::params::BiasFallback;
use starloom::fits_writer::bayer::Bayer;
use starloom::fits_writer::{write_fits_f32, Card, CardValue, OutputFormat};
use starloom::integration::banded::{BandPlanes, BandSource};
use starloom::integration::cfa::{flat_channel_norms, CfaGeometry};
use starloom::integration::combine::IntegrationRecipe;
use starloom::integration::engine::{integrate_flat, EngineProgress, FlatPrecal};
use starloom::integration::io_policy::IoPolicy;
use starloom::integration::recipe::resolve_recipe;
use starloom::storage_class::StorageClass;
use starloom::pipeline::{
    build_master, calibrate, preflight, CalGroup, CalSource, Ctx, FlatPrecalSource, Manifest, MasterKind, Progress, Stage,
};
use starloom::stacking::config::{algorithm_preset, AlgorithmConfig, StackingPreset};
use starloom::pipeline::{athf_path, finish, measure, measure_dir, pass1_path, Entry, MeasureReport};
use starloom::pipeline::formats::{commit_stage, source_facts, MEASURE_CSV_COLUMNS};
use starloom::stacking::fits_artifact::read_fits;
use starloom::stacking::psf_signal::PsfModel;
use starloom::stacking::weights::WeightMode;

const W: usize = 32;
const H: usize = 24;

/// A [`Progress`] that keeps the warnings.
#[derive(Default)]
struct Collect {
    warnings: Mutex<Vec<String>>,
}

impl Progress for Collect {
    fn step(&self, _stage: &str, _done: usize, _total: usize) {}
    fn warn(&self, message: &str) {
        self.warnings.lock().unwrap().push(message.to_string());
    }
}

impl Collect {
    fn all(&self) -> Vec<String> {
        self.warnings.lock().unwrap().clone()
    }
}

struct Env {
    pool: Arc<rayon::ThreadPool>,
    cancel: AtomicBool,
    progress: Collect,
}

impl Env {
    fn new() -> Env {
        Env {
            pool: Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap()),
            cancel: AtomicBool::new(false),
            progress: Collect::default(),
        }
    }

    fn ctx<'a>(&'a self, work: &'a Path) -> Ctx<'a> {
        Ctx { pool: self.pool.clone(), cancel: &self.cancel, progress: &self.progress, memory_budget_bytes: 1 << 30, work_dir: work }
    }
}

fn cfg() -> AlgorithmConfig {
    algorithm_preset(StackingPreset::Default)
}

/// A deterministic noise in `[0, 1)`.
fn noise(seed: u64, i: usize) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % 10_000) as f32 / 10_000.0
}

fn str_card(keyword: &str, value: &str) -> Card {
    Card::new(keyword, CardValue::Str(value.to_string())).unwrap()
}

fn int_card(keyword: &str, value: i64) -> Card {
    Card::new(keyword, CardValue::Integer(value)).unwrap()
}

/// A 32-bit float FITS plane at `path`.
fn write_f32(path: &Path, w: usize, h: usize, data: &[f32], cards: &[Card]) -> PathBuf {
    write_fits_f32(path, w, h, 1, data, cards).unwrap();
    path.to_path_buf()
}

/// A 16-bit FITS (BITPIX 16, BZERO 32768), the way a camera writes it.
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

/// A gzipped copy of `path` at `dest`.
fn gzip_copy(path: &Path, dest: &Path) -> PathBuf {
    let mut enc = flate2::write::GzEncoder::new(std::fs::File::create(dest).unwrap(), flate2::Compression::fast());
    enc.write_all(&std::fs::read(path).unwrap()).unwrap();
    enc.finish().unwrap();
    dest.to_path_buf()
}

/// The first plane of a file, through the reader the engine uses.
fn read_plane(path: &Path, scratch: &Path) -> (usize, usize, Vec<f32>) {
    let src = BandSource::open(&[path.to_path_buf()], scratch, 1).unwrap();
    let (w, h) = (src.width(), src.height());
    let mut planes = BandPlanes::new(&src);
    src.read_band(0, h, &mut planes, 1).unwrap();
    let mut data = vec![0f32; w * h];
    planes.decode_frame_into(0, &mut data);
    (w, h, data)
}

/// A file's header as the catalog's rules read it: keyword to unquoted value.
fn keys(path: &Path) -> HashMap<String, String> {
    let (format, text) = header_text(path).unwrap();
    header_key_map(format, &text)
        .into_legacy_map()
        .into_iter()
        .map(|(k, v)| (k, v.trim_matches('\'').trim().to_string()))
        .collect()
}

fn no_ath_cards(path: &Path) {
    let k = keys(path);
    let ath: Vec<&String> = k.keys().filter(|k| k.starts_with("ATH_")).collect();
    assert!(ath.is_empty(), "{} carries ATH_ cards: {ath:?}", path.display());
}

/// `n` frames of the same base value with a little noise, so a dark has a measurable spread.
fn frames(dir: &Path, prefix: &str, n: usize, base: f32, spread: f32, cards: &[Card]) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|k| {
            let data: Vec<f32> = (0..W * H).map(|i| base + spread * noise(k as u64 + 1, i)).collect();
            write_f32(&dir.join(format!("{prefix}_{k:04}.fits")), W, H, &data, cards)
        })
        .collect()
}

fn group(lights: Vec<PathBuf>, dark: Option<CalSource>, flat: Option<CalSource>, bias: Option<CalSource>) -> CalGroup {
    CalGroup { lights, dark, flat, bias, flat_precal: None }
}

fn dir_entries(dir: &Path) -> Vec<String> {
    match std::fs::read_dir(dir) {
        Ok(rd) => rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect(),
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------------------------

#[test]
fn slm_flat_cards_read_the_constants_and_refuse_another_cfa_phase() {
    let tmp = tempfile::tempdir().unwrap();
    let data = vec![1000.0f32; W * H];
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let mut cards = cfa.to_vec();
    cards.push(Card::new(SLM_FNRM, CardValue::Real(1500.0)).unwrap());
    for (k, v) in [(SLM_FNR, 1000.5), (SLM_FNG, 2000.25), (SLM_FNB, 3000.125)] {
        cards.push(Card::new(k, CardValue::Real(v)).unwrap());
    }
    let flat = write_f32(&tmp.path().join("flat_cards.fits"), W, H, &data, &cards);
    let rggb = CfaGeometry { pattern: Bayer::Rggb, xoff: 0, yoff: 0 };
    let grbg = CfaGeometry { pattern: Bayer::Grbg, xoff: 0, yoff: 0 };

    assert_eq!(SlmFlatCards.channel_norms(&flat, rggb), Some([1000.5, 2000.25, 3000.125]));
    assert_eq!(SlmFlatCards.channel_norms(&flat, CfaGeometry { xoff: 2, yoff: 4, ..rggb }), Some([1000.5, 2000.25, 3000.125]), "offsets compare modulo 2");
    assert_eq!(SlmFlatCards.channel_norms(&flat, grbg), None, "another pattern was measured under another phase");
    assert_eq!(SlmFlatCards.channel_norms(&flat, CfaGeometry { xoff: 1, ..rggb }), None, "another phase");
    assert_eq!(SlmFlatCards.fnrm(&flat), Some(1500.0));

    let bare = write_f32(&tmp.path().join("flat_bare.fits"), W, H, &data, &cfa);
    assert_eq!(SlmFlatCards.channel_norms(&bare, rggb), None, "a flat without the cards is recomputed");
    assert_eq!(SlmFlatCards.fnrm(&bare), None);
    let bad = write_f32(&tmp.path().join("flat_bad.fits"), W, H, &data, &[Card::new(SLM_FNRM, CardValue::Real(-1.0)).unwrap()]);
    assert_eq!(SlmFlatCards.fnrm(&bad), None, "a constant that cannot divide counts as absent");
    assert_eq!(SlmFlatCards.fnrm(&tmp.path().join("missing.fits")), None);
}

#[test]
fn master_dark_from_raw_frames_carries_the_recipe_and_no_ath_card() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let raw = frames(&tmp.path().join("darks"), "Dark", 16, 100.0, 4.0, &[Card::new("CCD-TEMP", CardValue::Real(-10.0)).unwrap(), Card::new("EXPTIME", CardValue::Real(300.0)).unwrap()]);
    let out = tmp.path().join("master_dark.fits");
    let report = build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw, None, None, &out, OutputFormat::Fits).unwrap();
    assert_eq!((report.width, report.height, report.frames), (W, H, 16));
    assert_eq!(report.recipe_text, format!("{} n=16", report.recipe.describe()));

    let k = keys(&out);
    assert_eq!(k["SLM_REC"], "Average | Winsorized sigma (3.0/3.0) n=16", "the value Athenaeum writes in ATH_REJ");
    assert_eq!(k["SLM_N"], "16");
    assert_eq!(k["IMAGETYP"], "Master Dark");
    assert_eq!(k["SLM_VER"], env!("CARGO_PKG_VERSION"));
    assert!(k.contains_key("SLM_CFG") && k.contains_key("SLM_PLV"));
    assert!(k.contains_key("SLM_TMIN") && k.contains_key("EXPTIME"));
    assert!(!k.contains_key("SLM_FNRM"), "only a flat carries the flat constants");
    no_ath_cards(&out);
    let (_, _, plane) = read_plane(&out, tmp.path());
    assert!(plane.iter().all(|v| (99.0..=105.0).contains(v)), "the master is the dark's level");
    assert!(Manifest::load(&work).unwrap().entry(Stage::Calibrate).is_none(), "a master leaves no manifest entry");
}

#[test]
fn master_refuses_a_bad_set_from_the_headers() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let mut raw = frames(&tmp.path().join("darks"), "Dark", 3, 100.0, 4.0, &[]);
    let odd = write_f32(&tmp.path().join("odd.fits"), W / 2, H, &vec![1.0; W / 2 * H], &[]);
    let out = tmp.path().join("m.fits");
    let two = build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw[..2], None, None, &out, OutputFormat::Fits).unwrap_err();
    assert_eq!(two.to_string(), "calibration set has 2 frames \u{2014} at least 3 are required to build a master");
    let precal = FlatPrecalSource::SyntheticBias(10.0);
    let dark_precal = build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw, Some(&precal), None, &out, OutputFormat::Fits).unwrap_err();
    assert!(dark_precal.to_string().contains("flats only"), "{dark_precal}");
    raw.push(odd.clone());
    let mixed = build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw, None, None, &out, OutputFormat::Fits).unwrap_err().to_string();
    assert!(mixed.contains("odd.fits") && mixed.contains("16×24") && mixed.contains("32×24"), "{mixed}");
    assert!(!out.exists(), "nothing was written");
}

#[test]
fn master_flat_writes_the_slm_flat_constants_and_calibrate_reads_them() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    // A CFA flat: red, green and blue pixels at different levels, with a little noise.
    let raw: Vec<PathBuf> = (0..4)
        .map(|k| {
            let data: Vec<f32> = (0..W * H)
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    let level = match (x % 2, y % 2) {
                        (0, 0) => 1000.0,
                        (1, 1) => 3000.0,
                        _ => 2000.0,
                    };
                    level + 10.0 * noise(k as u64 + 7, i)
                })
                .collect();
            write_f32(&tmp.path().join(format!("Flat_{k:04}.fits")), W, H, &data, &cfa)
        })
        .collect();
    let flat = tmp.path().join("master_flat.fits");
    let built = build_master(&env.ctx(&work), &cfg(), MasterKind::Flat, &raw, None, None, &flat, OutputFormat::Fits).unwrap();
    let norms = built.channel_norms.expect("a CFA flat has per-channel constants");
    let (w, h, plane) = read_plane(&flat, tmp.path());
    let measured = flat_channel_norms(&plane, w, h, CfaGeometry { pattern: Bayer::Rggb, xoff: 0, yoff: 0 }).unwrap();
    let k = keys(&flat);
    let stamped = [&k[SLM_FNR], &k[SLM_FNG], &k[SLM_FNB]].map(|v| v.parse::<f64>().unwrap());
    for c in 0..3 {
        assert!((stamped[c] - measured[c]).abs() <= 1e-6 * measured[c], "channel {c}: card {} measured {}", stamped[c], measured[c]);
        assert!((norms[c] - measured[c]).abs() <= 1e-6 * measured[c]);
    }
    let global: f64 = k[SLM_FNRM].parse().unwrap();
    assert!((global - built.flat_norm.unwrap()).abs() <= 1e-6 * global);
    assert_eq!(k["BAYERPAT"], "RGGB");
    assert_eq!(k["IMAGETYP"], "Master Flat");
    assert!(k["SLM_REC"].starts_with("Average | Percentile") && k["SLM_REC"].ends_with(" n=4"), "{}", k["SLM_REC"]);
    assert_eq!(k["SLM_REC"], format!("{} n=4", built.recipe.describe()), "fewer than 15 flats take the percentile recipe");
    no_ath_cards(&flat);

    // Calibrate reads the stamped constants, not new ones.
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 5000.0, 50.0, &cfa);
    let out = tmp.path().join("calibrated");
    let report = calibrate(&env.ctx(&work), &cfg(), &[group(lights, None, Some(CalSource::Master(flat.clone())), None)], &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap();
    let applied = report.frames[0].flat_channel_divisors.expect("per-channel scaling was applied");
    assert_eq!(applied, stamped, "the divisors are the stamped cards' values");
    assert_eq!(report.frames[0].calstat, "F");
}

#[test]
fn calibrate_refuses_a_dark_of_another_size_before_reading_pixels() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 2, 1000.0, 10.0, &[]);
    let dark = write_f32(&tmp.path().join("dark_small.fits"), W - 8, H, &vec![100.0; (W - 8) * H], &[]);
    let out = tmp.path().join("calibrated");
    let g = [group(lights.clone(), Some(CalSource::Master(dark)), None, None)];
    let err = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap_err().to_string();
    assert!(err.starts_with("Calibration does not fit the frame size"), "{err}");
    assert!(err.contains("dark_small.fits") && err.contains("Light_0000.fits"), "{err}");
    assert!(err.contains("24×24") && err.contains("32×24"), "{err}");
    assert!(dir_entries(&out).is_empty(), "nothing is written");
    assert!(dir_entries(&work).is_empty(), "no scratch, no master, no manifest: no pixel was read");
    assert!(preflight(&g).is_err(), "the preflight alone refuses it");
}

#[test]
fn calibrate_refuses_a_three_plane_light() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    std::fs::create_dir_all(tmp.path().join("lights")).unwrap();
    let light = tmp.path().join("lights").join("Light_rgb.fits");
    write_fits_f32(&light, W, H, 3, &vec![1000.0; W * H * 3], &[]).unwrap();
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let out = tmp.path().join("calibrated");
    let g = [group(vec![light], Some(CalSource::Master(dark)), None, None)];
    let err = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap_err().to_string();
    assert!(err.contains("Light_rgb.fits") && err.contains("3-channel"), "{err}");
    assert!(err.contains("frames must be 1-channel for calibration"), "{err}");
    assert!(dir_entries(&out).is_empty());
}

#[test]
fn calibrate_warns_on_another_subframe_origin_and_still_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 2, 1000.0, 10.0, &[int_card("XORGSUBF", 0), int_card("YORGSUBF", 0)]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[int_card("XORGSUBF", 10), int_card("YORGSUBF", 20)]);
    let out = tmp.path().join("calibrated");
    let g = [group(lights, Some(CalSource::Master(dark)), None, None)];
    let warnings = preflight(&g).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].starts_with("Calibration at another sub-frame origin"), "{}", warnings[0]);
    assert!(warnings[0].contains("32×24 @ 10,20") && warnings[0].contains("32×24 @ 0,0"), "{}", warnings[0]);
    let report = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap();
    assert_eq!(report.frames.len(), 2);
    assert!(report.warnings.iter().any(|w| w.starts_with("Calibration at another sub-frame origin")));
    assert!(env.progress.all().iter().any(|w| w.starts_with("Calibration at another sub-frame origin")), "the sink heard it too");
    for f in &report.frames {
        assert!(f.output.is_file());
    }
}

#[test]
fn calibrate_warns_on_a_flat_of_another_cfa_phase() {
    let tmp = tempfile::tempdir().unwrap();
    let cfa = |p: &str| [str_card("BAYERPAT", p), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 10.0, &cfa("RGGB"));
    let same = write_f32(&tmp.path().join("flat_same.fits"), W, H, &vec![2000.0; W * H], &cfa("RGGB"));
    let other = write_f32(&tmp.path().join("flat_other.fits"), W, H, &vec![2000.0; W * H], &cfa("GRBG"));
    let ok = preflight(&[group(lights.clone(), None, Some(CalSource::Master(same)), None)]).unwrap();
    assert!(ok.is_empty(), "{ok:?}");
    let warned = preflight(&[group(lights, None, Some(CalSource::Master(other)), None)]).unwrap();
    assert_eq!(warned.len(), 1, "{warned:?}");
    assert!(warned[0].contains("flat_other.fits") && warned[0].contains("CFA"), "{}", warned[0]);
}

#[test]
fn calibrate_uses_the_bias_only_without_a_dark() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 0.0, &[Card::new("EXPTIME", CardValue::Real(120.0)).unwrap(), str_card("DATE-OBS", "2026-07-05T20:30:00")]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let bias = write_f32(&tmp.path().join("bias.fits"), W, H, &vec![50.0; W * H], &[]);
    let opts = CalibratedLightOptions::default();

    let with_dark = [group(lights.clone(), Some(CalSource::Master(dark)), None, Some(CalSource::Master(bias.clone())))];
    let r = calibrate(&env.ctx(&work), &cfg(), &with_dark, &opts, &tmp.path().join("a"), OutputFormat::Fits).unwrap();
    assert_eq!(r.frames[0].calstat, "BD", "the dark removes the bias too");
    let (_, _, a) = read_plane(&r.frames[0].output, tmp.path());
    assert!(a.iter().all(|v| (*v - 900.0).abs() < 1e-3), "light minus dark only");
    let k = keys(&r.frames[0].output);
    assert_eq!(k["CALSTAT"], "BD");
    assert_eq!(k["IMAGETYP"], "Light Frame");
    assert_eq!(k["SLM_CSCL"].parse::<f64>().unwrap(), 1.0, "a float light is not scaled");
    assert_eq!(k["SLM_LCV"], starloom::calibration_library::LIGHT_CAL_VERSION.to_string());
    assert!(!k.contains_key("SLM_CHPX"), "a dark with no spread gives no hot-pixel map, so no card");
    assert!(r.warnings.iter().any(|w| w.starts_with("Hot-pixel correction skipped")), "{:?}", r.warnings);
    assert!(k.contains_key("EXPTIME") && k.contains_key("DATE-OBS"), "copy-through cards");
    assert!(k.contains_key("SLM_VER") && k.contains_key("SLM_CFG") && k.contains_key("SLM_PLV"));
    no_ath_cards(&r.frames[0].output);

    let bias_only = [group(lights, None, None, Some(CalSource::Master(bias)))];
    let r = calibrate(&env.ctx(&work), &cfg(), &bias_only, &opts, &tmp.path().join("b"), OutputFormat::Fits).unwrap();
    assert_eq!(r.frames[0].calstat, "B");
    let (_, _, b) = read_plane(&r.frames[0].output, tmp.path());
    assert!(b.iter().all(|v| (*v - 950.0).abs() < 1e-3), "light minus bias");
    assert!(!keys(&r.frames[0].output).contains_key("SLM_CHPX"), "no dark, no cosmetic pass");

    // The entry records the product and the raw light's facts.
    let m = Manifest::load(&work).unwrap();
    let e = m.entry(Stage::Calibrate).unwrap();
    assert_eq!(e.products, vec![r.frames[0].output.clone()]);
    assert_eq!(e.sources[&r.frames[0].output].exptime_s, Some(120.0));
}

#[test]
fn calibrated_osc_light_is_debayered_and_loses_its_mosaic_cards() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0), str_card("ROWORDER", "TOP-DOWN")];
    let lights: Vec<PathBuf> = (0..2)
        .map(|k| {
            let data: Vec<f32> = (0..W * H)
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    let base = if x % 2 == 0 && y % 2 == 0 { 5000.0 } else { 1000.0 };
                    base + 20.0 * noise(k + 3, i) + (x + y) as f32
                })
                .collect();
            write_f32(&tmp.path().join(format!("Light_{k:04}.fits")), W, H, &data, &cfa)
        })
        .collect();
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &(0..W * H).map(|i| 100.0 + 4.0 * noise(9, i)).collect::<Vec<_>>(), &[]);
    let out = tmp.path().join("calibrated");
    let g = [group(lights, Some(CalSource::Master(dark)), None, None)];
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap();
    let frame = &r.frames[0];
    assert!(frame.debayered);
    assert_eq!(frame.output.file_name().unwrap().to_str().unwrap(), "c_Light_0000_d.fits");
    let facts = read_frame_facts(&frame.output).unwrap();
    assert_eq!(facts.planes, Some(3));
    let k = keys(&frame.output);
    assert!(!k.contains_key("BAYERPAT") && !k.contains_key("XBAYROFF") && !k.contains_key("YBAYROFF"));
    assert_eq!(k["SLM_CDBM"], "VNG");
    assert_eq!(k["ROWORDER"], "TOP-DOWN", "the row order stays");
    assert!(k["SLM_CHPX"].parse::<u64>().is_ok(), "the cosmetic pass ran from the master dark");
    no_ath_cards(&frame.output);
    assert_eq!(frame.hot_pixels_replaced, k["SLM_CHPX"].parse::<u64>().ok());

    // With the debayer off the mosaic stays, and so do its cards.
    let opts = CalibratedLightOptions { debayer_osc: false, ..CalibratedLightOptions::default() };
    let mono = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join("mosaic"), OutputFormat::Fits).unwrap();
    assert!(!mono.frames[0].debayered);
    let k = keys(&mono.frames[0].output);
    assert_eq!(k["BAYERPAT"], "RGGB");
    assert!(!k.contains_key("SLM_CDBM"));
}

/// The plain run's frames and masters, from 16-bit frames with a hot pixel and noise.
fn camera_frames(dir: &Path, prefix: &str, n: usize, base: u16, hot: bool) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|k| {
            let mut v: Vec<u16> = (0..W * H).map(|i| base + (60.0 * noise(k as u64 + 11 + u64::from(base), i)) as u16 + (i % W) as u16).collect();
            if hot {
                v[5 * W + 7] = 60_000;
            }
            write_u16(&dir.join(format!("{prefix}_{:04}.fits", k + 1)), W, H, &v, &[("EXPTIME", "120.0".to_string())])
        })
        .collect()
}

#[test]
fn calibrate_reads_gzipped_lights_and_darks_and_writes_the_plain_run_s_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let plain_lights = camera_frames(&tmp.path().join("plain_lights"), "Light", 4, 3000, false);
    let plain_darks = camera_frames(&tmp.path().join("plain_darks"), "Dark", 16, 500, true);
    let gz = tmp.path().join("gz");
    std::fs::create_dir_all(&gz).unwrap();
    let gz_lights: Vec<PathBuf> = plain_lights.iter().map(|p| gzip_copy(p, &gz.join(format!("{}.gz", p.file_name().unwrap().to_str().unwrap())))).collect();
    let gz_darks: Vec<PathBuf> = plain_darks.iter().map(|p| gzip_copy(p, &gz.join(format!("{}.gz", p.file_name().unwrap().to_str().unwrap())))).collect();

    let opts = CalibratedLightOptions::default();
    let run = |name: &str, lights: Vec<PathBuf>, darks: Vec<PathBuf>| {
        let work = tmp.path().join(format!("work_{name}"));
        let g = [group(lights, Some(CalSource::Raw(darks)), None, None)];
        let r = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join(format!("out_{name}")), OutputFormat::Fits).unwrap();
        (work, r)
    };
    let (work_plain, plain) = run("plain", plain_lights, plain_darks);
    let (work_gz, zipped) = run("gz", gz_lights, gz_darks);
    assert_eq!(plain.frames.len(), 4);
    assert_eq!(zipped.frames.len(), 4);
    for (p, z) in plain.frames.iter().zip(&zipped.frames) {
        assert_eq!(p.output.file_name(), z.output.file_name(), "the stem drops the whole suffix");
        assert_eq!(p.calstat, "BD");
        let (_, _, a) = read_plane(&p.output, tmp.path());
        let (_, _, b) = read_plane(&z.output, tmp.path());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b), "calibrated data of {}", p.output.display());
        assert_eq!(p.hot_pixels_replaced, z.hot_pixels_replaced);
        assert!(p.hot_pixels_replaced.unwrap() >= 1, "the dark's hot pixel was replaced");
        let (kp, kz) = (keys(&p.output), keys(&z.output));
        assert_eq!(kp["SLM_CSCL"], kz["SLM_CSCL"]);
        assert_eq!(kp["SLM_CSCL"].parse::<f64>().unwrap(), 65535.0, "a 16-bit light, plain or compressed");
    }
    // The masters built from the plain and the compressed darks are the same data.
    let mp = read_plane(&work_plain.join("masters").join("group-01-dark.fits"), tmp.path()).2;
    let mz = read_plane(&work_gz.join("masters").join("group-01-dark.fits"), tmp.path()).2;
    assert_eq!(mp.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), mz.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
}

#[test]
fn a_truncated_gzipped_light_fails_naming_it_and_leaves_no_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = camera_frames(&tmp.path().join("lights"), "Light", 3, 3000, false);
    let gz = tmp.path().join("gz");
    std::fs::create_dir_all(&gz).unwrap();
    let mut zipped: Vec<PathBuf> = lights.iter().map(|p| gzip_copy(p, &gz.join(format!("{}.gz", p.file_name().unwrap().to_str().unwrap())))).collect();
    // The last light loses its second half.
    let bad = zipped[2].clone();
    let bytes = std::fs::read(&bad).unwrap();
    std::fs::write(&bad, &bytes[..bytes.len() / 2]).unwrap();
    zipped[2] = bad.clone();
    let dark = write_u16(&tmp.path().join("dark.fits"), W, H, &vec![500u16; W * H], &[]);
    let out = tmp.path().join("calibrated");
    let g = [group(zipped, Some(CalSource::Master(dark)), None, None)];
    let err = format!("{:#}", calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap_err());
    assert!(err.contains(bad.file_name().unwrap().to_str().unwrap()), "{err}");
    let leftovers: Vec<String> = dir_entries(&out);
    assert!(leftovers.is_empty(), "no product and no part file is left: {leftovers:?}");
    assert!(Manifest::load(&work).unwrap().entry(Stage::Calibrate).is_none(), "no manifest entry");
}

// ---------------------------------------------------------------------------------------------------------------
// The review round: Athenaeum's cards, mosaic, scale divisor, precal, the CFA rule and compressed frames
// ---------------------------------------------------------------------------------------------------------------

/// A file's cards in header order: the keyword of each card (continuation records skipped).
fn card_order(path: &Path) -> Vec<String> {
    let (_, text) = header_text(path).unwrap();
    text.lines()
        .filter_map(|l| l.split('=').next().map(|k| k.trim().to_string()))
        .filter(|k| !k.is_empty() && k != "CONTINUE" && k != "END" && !k.starts_with("XISF"))
        .filter(|k| !(["SIMPLE", "BITPIX", "EXTEND", "BZERO", "BSCALE"].contains(&k.as_str()) || k.starts_with("NAXIS")))
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A plain FITS with the given header cards and big-endian `data` (`NAXIS` and the axes from `axes`).
fn write_raw_fits(path: &Path, bitpix: i32, axes: &[usize], data: &[u8]) -> PathBuf {
    let mut cards = vec![format!("{:<8}= {:>20}", "SIMPLE", "T"), format!("{:<8}= {:>20}", "BITPIX", bitpix), format!("{:<8}= {:>20}", "NAXIS", axes.len())];
    for (i, n) in axes.iter().enumerate() {
        cards.push(format!("{:<8}= {:>20}", format!("NAXIS{}", i + 1), n));
    }
    cards.push("END".to_string());
    let mut bytes = Vec::new();
    for c in cards {
        bytes.extend_from_slice(format!("{c:<80}").as_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    std::fs::write(path, bytes).unwrap();
    path.to_path_buf()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

#[test]
fn master_cards_carry_exactly_athenaeums_values() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let raw = frames(&tmp.path().join("darks"), "Dark", 5, 100.0, 4.0, &[str_card("IMAGETYP", "Dark"), Card::new("CCD-TEMP", CardValue::Real(-10.0)).unwrap()]);
    let out = tmp.path().join("master_dark.fits");
    let report = build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw, None, None, &out, OutputFormat::Fits).unwrap();
    let k = keys(&out);
    assert_eq!(k["SLM_REC"], format!("{} n=5", report.recipe.describe()), "ATH_REJ is the recipe's description and n");
    assert_eq!(k["SLM_N"], "5");
    assert_eq!(k["SLM_TMIN"].parse::<f64>().unwrap(), -10.0);
    assert_eq!(k["SLM_TMAX"].parse::<f64>().unwrap(), -10.0);
    assert_eq!(k["IMAGETYP"], "Master Dark");
    // The cards after the build's own come in Athenaeum's order: ATH_N, ATH_REJ, ATH_VER, ATH_TMIN, ATH_TMAX.
    let order = card_order(&out);
    let at = |name: &str| order.iter().position(|k| k == name).unwrap_or_else(|| panic!("{name} missing in {order:?}"));
    assert!(at("SLM_N") < at("SLM_REC") && at("SLM_REC") < at("SLM_VER") && at("SLM_VER") < at("SLM_TMIN") && at("SLM_TMIN") < at("SLM_TMAX"));
    assert!(at("IMAGETYP") == 0 && at("SWCREATE") == 1);
}

#[test]
fn dark_flat_frames_make_a_master_dark_flat() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let raw = frames(&tmp.path().join("darkflats"), "DarkFlat", 4, 100.0, 4.0, &[str_card("IMAGETYP", "Dark Flat")]);
    let out = tmp.path().join("master_dark_flat.fits");
    build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &raw, None, None, &out, OutputFormat::Fits).unwrap();
    assert_eq!(keys(&out)["IMAGETYP"], "Master Dark Flat");
}

#[test]
fn a_master_integrates_its_frames_in_the_order_of_their_full_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    // Two nights in two folders: by file name `a_0` comes before `z_0`, by full path the night_1 folder comes first.
    let night1 = frames(&tmp.path().join("night_1"), "z", 3, 1000.0, 200.0, &[]);
    let night2 = frames(&tmp.path().join("night_2"), "a", 3, 1010.0, 300.0, &[]);
    let forward: Vec<PathBuf> = night1.iter().chain(&night2).cloned().collect();
    let mut shuffled: Vec<PathBuf> = night2.iter().rev().chain(night1.iter().rev()).cloned().collect();
    shuffled.rotate_left(2);
    let (a, b) = (tmp.path().join("a.fits"), tmp.path().join("b.fits"));
    build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &forward, None, None, &a, OutputFormat::Fits).unwrap();
    build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &shuffled, None, None, &b, OutputFormat::Fits).unwrap();
    assert_eq!(bits(&read_plane(&a, tmp.path()).2), bits(&read_plane(&b, tmp.path()).2), "the order given does not move a bit");
}

#[test]
fn a_calibrated_light_carries_the_slm_twin_of_every_ath_card_in_athenaeums_order() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 5000.0, 50.0, &[cfa[0].clone(), cfa[1].clone(), cfa[2].clone(), Card::new("EXPTIME", CardValue::Real(60.0)).unwrap()]);
    let dark = write_f32(&tmp.path().join("Master_Dark_300s.fits"), W, H, &(0..W * H).map(|i| 100.0 + 4.0 * noise(9, i)).collect::<Vec<_>>(), &[]);
    let flat = write_f32(&tmp.path().join("Master_Flat_L.fits"), W, H, &vec![2000.0; W * H], &cfa);
    let g = [group(lights.clone(), Some(CalSource::Master(dark)), Some(CalSource::Master(flat)), None)];
    let opts = CalibratedLightOptions { debayer_osc: false, ..CalibratedLightOptions::default() };
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    let out = &r.frames[0].output;
    let order = card_order(out);
    let pos = |name: &str| order.iter().position(|k| k == name).unwrap_or_else(|| panic!("{name} missing in {order:?}"));
    let expected = ["IMAGETYP", "CALSTAT"];
    for (i, name) in expected.iter().enumerate() {
        assert_eq!(pos(name), i);
    }
    let twins = ["SLM_CSRN", "SLM_CDRK", "SLM_CFLT", "SLM_CSCL", "SLM_CFNM", "SLM_CPED", "SLM_LCV", "SLM_CCFA", "SLM_CFNR", "SLM_CFNG", "SLM_CFNB"];
    let places: Vec<usize> = twins.iter().map(|t| pos(t)).collect();
    assert!(places.windows(2).all(|w| w[0] < w[1]), "twins in Athenaeum's order: {order:?}");
    assert!(pos("EXPTIME") < places[0], "the copy-through cards come first");
    assert!(order.iter().all(|k| k != "SLM_CBIA" && k != "SLM_CTRM" && k != "SLM_CSRC" && k != "SLM_CDBM"), "{order:?}");
    assert_eq!(order.last().map(String::as_str), Some("SLM_CHPX"), "the hot-pixel card closes the header: {order:?}");
    let k = keys(out);
    assert_eq!(k["SLM_CSRN"], "Light_0000.fits");
    assert_eq!(k["SLM_CDRK"], "Master_Dark_300s.fits", "the master's file name, no catalog id");
    assert_eq!(k["SLM_CFLT"], "Master_Flat_L.fits");
    assert_eq!(k["SLM_CPED"].parse::<f64>().unwrap(), 0.0);
    assert!(k["SLM_CFNM"].parse::<f64>().unwrap() > 1.0);
    assert_eq!(k["SLM_CCFA"], "T");
    no_ath_cards(out);
}

#[test]
fn a_bias_only_light_carries_slm_cbia_and_a_trimmed_mean_one_slm_ctrm() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 10.0, &[]);
    let bias = write_f32(&tmp.path().join("Master_Bias.fits"), W, H, &vec![50.0; W * H], &[]);
    let flat = write_f32(&tmp.path().join("Master_Flat.fits"), W, H, &(0..W * H).map(|i| 2000.0 + 10.0 * noise(4, i)).collect::<Vec<_>>(), &[]);
    let g = [group(lights, None, Some(CalSource::Master(flat)), Some(CalSource::Master(bias)))];
    let mut opts = CalibratedLightOptions::default();
    opts.flat_norm_mode = starloom::calibration_library::params::FlatNormMode::TrimmedMeanWhole;
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    let k = keys(&r.frames[0].output);
    assert_eq!(k["SLM_CBIA"], "Master_Bias.fits");
    assert!(!k.contains_key("SLM_CDRK"));
    assert_eq!(k["SLM_CTRM"].parse::<f64>().unwrap(), opts.params.trim_fraction);
    assert_eq!(k["CALSTAT"], "BF");
}

#[test]
fn keep_mosaic_writes_the_calibrated_mosaic_beside_the_debayered_frame() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let lights = frames(&tmp.path().join("lights"), "Light", 2, 3000.0, 100.0, &cfa);
    let darks = write_f32(&tmp.path().join("dark.fits"), W, H, &(0..W * H).map(|i| 100.0 + 4.0 * noise(9, i)).collect::<Vec<_>>(), &[]);
    let g = [group(lights, Some(CalSource::Master(darks)), None, None)];
    let opts = CalibratedLightOptions { keep_mosaic: true, ..CalibratedLightOptions::default() };
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    let out = tmp.path().join("out");
    let mut names = dir_entries(&out);
    names.sort();
    assert_eq!(names, ["c_Light_0000.fits", "c_Light_0000_d.fits", "c_Light_0001.fits", "c_Light_0001_d.fits"], "the mosaic is the mono spelling of the stem");
    // The mosaic is one plane of the corrected CFA data, with the mosaic cards and no debayer card.
    let mosaic = out.join("c_Light_0000.fits");
    assert_eq!(read_frame_facts(&mosaic).unwrap().planes, Some(1));
    let mk = keys(&mosaic);
    assert_eq!(mk["BAYERPAT"], "RGGB");
    assert!(!mk.contains_key("SLM_CDBM"));
    assert_eq!(mk["SLM_CHPX"].parse::<u64>().ok(), r.frames[0].hot_pixels_replaced);
    // It is the very frame the debayer-off run writes.
    let plain_opts = CalibratedLightOptions { debayer_osc: false, ..CalibratedLightOptions::default() };
    let plain = calibrate(&env.ctx(&work), &cfg(), &g, &plain_opts, &tmp.path().join("plain"), OutputFormat::Fits).unwrap();
    assert_eq!(bits(&read_plane(&mosaic, tmp.path()).2), bits(&read_plane(&plain.frames[0].output, tmp.path()).2));
    assert_eq!(card_order(&mosaic), card_order(&plain.frames[0].output), "the cards are the debayer-off run's");
    // The mosaic is no product of the stage: the entry names the debayered frames only.
    let entry = Manifest::load(&work).unwrap().entry(Stage::Calibrate).cloned().unwrap();
    assert_eq!(entry.products.len(), 2);
}

#[test]
fn bias_fallback_skip_frame_excludes_the_group_and_calibrates_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let l1 = frames(&tmp.path().join("l1"), "A", 2, 1000.0, 10.0, &[]);
    let l2 = frames(&tmp.path().join("l2"), "B", 2, 1000.0, 10.0, &[]);
    let bias = write_f32(&tmp.path().join("bias.fits"), W, H, &vec![50.0; W * H], &[]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let g = [group(l1.clone(), None, None, Some(CalSource::Master(bias.clone()))), group(l2, Some(CalSource::Master(dark)), None, None)];
    let mut opts = CalibratedLightOptions::default();
    opts.params.bias_fallback = BiasFallback::SkipFrame;
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &opts, &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    assert_eq!(r.frames.len(), 2, "group 2 only");
    assert!(r.frames.iter().all(|f| f.source.to_string_lossy().contains("l2")));
    let skipped: Vec<&String> = r.warnings.iter().filter(|w| w.ends_with("no dark master (bias fallback disabled)")).collect();
    assert_eq!(skipped.len(), 2, "one warning per excluded light: {:?}", r.warnings);
    assert!(skipped[0].contains("A_0000.fits"));
    assert!(env.progress.all().iter().any(|w| w.ends_with("no dark master (bias fallback disabled)")));
    // With the fallback on, the bias calibrates them.
    let on = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("on"), OutputFormat::Fits).unwrap();
    assert_eq!(on.frames.len(), 4);
    // Nothing left to calibrate fails the run, and writes nothing.
    let only = [group(l1, None, None, Some(CalSource::Master(bias)))];
    let err = calibrate(&env.ctx(&work), &cfg(), &only, &opts, &tmp.path().join("none"), OutputFormat::Fits).unwrap_err().to_string();
    assert!(err.contains("no light could be calibrated"), "{err}");
    assert!(!tmp.path().join("none").exists());
}

#[test]
fn a_group_with_no_calibration_frame_is_excluded_not_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let l1 = frames(&tmp.path().join("l1"), "A", 1, 1000.0, 10.0, &[]);
    let l2 = frames(&tmp.path().join("l2"), "B", 1, 1000.0, 10.0, &[]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let g = [group(l1, None, None, None), group(l2, Some(CalSource::Master(dark)), None, None)];
    let warnings = preflight(&g).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].ends_with("no calibration masters available (dark/flat/bias unbuilt or unlinked)"), "{}", warnings[0]);
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    assert_eq!(r.frames.len(), 1);
    assert_eq!(r.frames[0].output.file_name().unwrap().to_str().unwrap(), "c_B_0000.fits");
    assert!(r.warnings.iter().any(|w| w.contains("A_0000.fits") && w.contains("no calibration masters available")));
}

#[test]
fn the_scale_divisor_is_65535_for_a_light_the_direct_reader_cannot_probe() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let plane: Vec<f32> = (0..W * H).map(|i| 1000.0 + 10.0 * noise(2, i)).collect();
    // A plain 2-D float light: probe_bitpix reads -32, so no scaling.
    let flat2d = write_f32(&tmp.path().join("light_2d.fits"), W, H, &plane, &[]);
    // A plain 3-D light with one plane: the direct reader's probe does not take it, whatever its BITPIX says.
    let flat3d = write_raw_fits(&tmp.path().join("light_3d.fits"), -32, &[W, H, 1], &f32_bytes(&plane));
    // The same float light gzipped: a compressed file is read by its logical header (P2-R22).
    let zipped = gzip_copy(&flat2d, &tmp.path().join("light_gz.fits.gz"));
    let divisor = |light: &PathBuf| {
        let g = [group(vec![light.clone()], Some(CalSource::Master(dark.clone())), None, None)];
        let name = light.file_name().unwrap().to_str().unwrap().split('.').next().unwrap().to_string();
        let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join(format!("out_{name}")), OutputFormat::Fits).unwrap();
        keys(&r.frames[0].output)["SLM_CSCL"].parse::<f64>().unwrap()
    };
    assert_eq!(divisor(&flat2d), 1.0);
    assert_eq!(divisor(&flat3d), 65535.0, "Athenaeum keeps the historic divisor where probe_bitpix has no answer");
    assert_eq!(divisor(&zipped), 1.0, "a gzipped float light keeps its logical depth");
}

#[test]
fn a_raw_set_too_small_for_a_master_is_refused_before_group_one_is_calibrated() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let l1 = frames(&tmp.path().join("l1"), "A", 1, 1000.0, 10.0, &[]);
    let l2 = frames(&tmp.path().join("l2"), "B", 1, 1000.0, 10.0, &[]);
    let darks1 = frames(&tmp.path().join("d1"), "Dark", 4, 100.0, 4.0, &[]);
    let darks2 = frames(&tmp.path().join("d2"), "Dark", 2, 100.0, 4.0, &[]);
    let g = [group(l1, Some(CalSource::Raw(darks1)), None, None), group(l2, Some(CalSource::Raw(darks2)), None, None)];
    let err = preflight(&g).unwrap_err().to_string();
    assert!(err.contains("group 2 dark") && err.contains("calibration set has 2 frames") && err.contains("at least 3"), "{err}");
    let out = tmp.path().join("out");
    let err = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap_err().to_string();
    assert!(err.contains("group 2 dark"), "{err}");
    assert!(!out.exists(), "no light of group 1 was calibrated");
    assert!(dir_entries(&work).is_empty(), "no master of group 1 was built, no scratch made");
}

/// The engine's own flat integration of `frames` under `precal`, the master the stage must equal.
fn engine_flat(env: &Env, frames: &[PathBuf], precal: &FlatPrecal, scratch: &Path) -> Vec<f32> {
    let io = IoPolicy { band_budget_bytes: 1 << 26, read_concurrency: 2, storage: StorageClass::Local, prefetch_local: false };
    let progress = EngineProgress { on_band: &|_, _, _, _| {}, on_combine: &|_, _, _, _| {} };
    let recipe: IntegrationRecipe = resolve_recipe(None, "Flat", frames.len() as i64);
    integrate_flat(frames, precal, recipe, &env.pool, scratch, &env.cancel, progress, io).unwrap().data
}

#[test]
fn a_flat_master_precalibrated_with_a_master_bias_equals_the_engine_with_that_frame() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let flats = frames(&tmp.path().join("flats"), "Flat", 4, 3000.0, 40.0, &[]);
    let bias_plane: Vec<f32> = (0..W * H).map(|i| 200.0 + 8.0 * noise(5, i)).collect();
    let bias = write_f32(&tmp.path().join("master_bias.fits"), W, H, &bias_plane, &[]);
    let out = tmp.path().join("flat.fits");
    let precal = FlatPrecalSource::Master(bias);
    build_master(&env.ctx(&work), &cfg(), MasterKind::Flat, &flats, Some(&precal), None, &out, OutputFormat::Fits).unwrap();
    let want = engine_flat(&env, &flats, &FlatPrecal::MasterFrame { data: bias_plane, width: W, height: H }, tmp.path());
    assert_eq!(bits(&read_plane(&out, tmp.path()).2), bits(&want), "the master is the engine's, pre-calibrated with the frame");
    let bare = tmp.path().join("bare.fits");
    build_master(&env.ctx(&work), &cfg(), MasterKind::Flat, &flats, None, None, &bare, OutputFormat::Fits).unwrap();
    assert_ne!(bits(&read_plane(&bare, tmp.path()).2), bits(&want), "the pre-calibration changed the pixels");
}

#[test]
fn a_flat_master_precalibrated_with_a_constant_equals_the_engine_with_that_constant() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let flats = frames(&tmp.path().join("flats"), "Flat", 4, 3000.0, 40.0, &[]);
    let out = tmp.path().join("flat.fits");
    build_master(&env.ctx(&work), &cfg(), MasterKind::Flat, &flats, Some(&FlatPrecalSource::SyntheticBias(250.0)), None, &out, OutputFormat::Fits).unwrap();
    let want = engine_flat(&env, &flats, &FlatPrecal::SyntheticBias(250.0), tmp.path());
    assert_eq!(bits(&read_plane(&out, tmp.path()).2), bits(&want));
    let k = keys(&out);
    assert!(k["SLM_FNRM"].parse::<f64>().unwrap() < 3000.0, "the constant was subtracted before the flat was normalized");
}

#[test]
fn a_flat_precalibration_of_another_size_is_refused_before_any_pixel_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 10.0, &[]);
    let flats = frames(&tmp.path().join("flats"), "Flat", 3, 3000.0, 40.0, &[]);
    let small = write_f32(&tmp.path().join("bias_small.fits"), W - 8, H, &vec![200.0; (W - 8) * H], &[]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let g = [CalGroup { lights, dark: Some(CalSource::Master(dark)), flat: Some(CalSource::Raw(flats.clone())), bias: None, flat_precal: Some(FlatPrecalSource::Master(small.clone())) }];
    let err = preflight(&g).unwrap_err().to_string();
    assert!(err.starts_with("Calibration does not fit the frame size") && err.contains("flat pre-calibration") && err.contains("bias_small.fits"), "{err}");
    assert!(err.contains("24×24") && err.contains("32×24"), "{err}");
    let out = tmp.path().join("out");
    calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &out, OutputFormat::Fits).unwrap_err();
    assert!(!out.exists() && dir_entries(&work).is_empty(), "nothing was written and no pixel was read");
    let direct = build_master(&env.ctx(&work), &cfg(), MasterKind::Flat, &flats, Some(&FlatPrecalSource::Master(small)), None, &tmp.path().join("m.fits"), OutputFormat::Fits).unwrap_err().to_string();
    assert!(direct.contains("pre-calibration master") && direct.contains("bias_small.fits"), "{direct}");
}

#[test]
fn a_flat_precalibration_given_with_a_master_flat_is_warned_about() {
    let tmp = tempfile::tempdir().unwrap();
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 10.0, &[]);
    let flat = write_f32(&tmp.path().join("flat.fits"), W, H, &vec![2000.0; W * H], &[]);
    let g = [CalGroup { lights, dark: None, flat: Some(CalSource::Master(flat)), bias: None, flat_precal: Some(FlatPrecalSource::SyntheticBias(10.0)) }];
    let warnings = preflight(&g).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("group 1") && warnings[0].contains("pre-calibration is ignored"), "{}", warnings[0]);
}

#[test]
fn the_flat_constants_come_from_the_slm_cards_not_from_a_recompute() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 5000.0, 50.0, &cfa);
    let level = |i: usize| match ((i % W) % 2, (i / W) % 2) {
        (0, 0) => 1000.0f32,
        (1, 1) => 3000.0,
        _ => 2000.0,
    };
    let data: Vec<f32> = (0..W * H).map(level).collect();
    let run = |flat: &Path, name: &str| {
        let g = [group(lights.clone(), None, Some(CalSource::Master(flat.to_path_buf())), None)];
        calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join(name), OutputFormat::Fits).unwrap()
    };
    // Stamped constants that differ from the plane's own (1000, 2000, 3000 per channel; 1500 for the global one).
    let mut stamped = cfa.to_vec();
    for (k, v) in [(SLM_FNRM, 7777.0), (SLM_FNR, 111.0), (SLM_FNG, 222.0), (SLM_FNB, 333.0)] {
        stamped.push(Card::new(k, CardValue::Real(v)).unwrap());
    }
    let with_cards = write_f32(&tmp.path().join("flat_stamped.fits"), W, H, &data, &stamped);
    let r = run(&with_cards, "stamped");
    assert_eq!(r.frames[0].flat_channel_divisors, Some([111.0, 222.0, 333.0]), "the stamped constants are applied");
    assert_eq!(keys(&r.frames[0].output)["SLM_CFNR"].parse::<f64>().unwrap(), 111.0);
    // The same flat stamped under another CFA phase: the constants are measured again from the pixels.
    let mut other = vec![str_card("BAYERPAT", "GRBG"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    other.extend(stamped[3..].iter().cloned());
    let rephased = write_f32(&tmp.path().join("flat_rephased.fits"), W, H, &data, &other);
    let r = run(&rephased, "rephased");
    let divisors = r.frames[0].flat_channel_divisors.expect("per-channel scaling still applies");
    assert_eq!(divisors, [1000.0, 2000.0, 3000.0], "the plane's own central-third means under the light's phase");
    // A mono light takes the global constant, stamped 7777 here.
    let mono_lights = frames(&tmp.path().join("mono"), "Mono", 1, 5000.0, 50.0, &[]);
    let g = [group(mono_lights, None, Some(CalSource::Master(with_cards)), None)];
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("mono_out"), OutputFormat::Fits).unwrap();
    assert_eq!(keys(&r.frames[0].output)["SLM_CFNM"].parse::<f64>().unwrap(), 7777.0, "the stamped SLM_FNRM, not the recomputed mean");
}

#[test]
fn a_flat_without_stamped_constants_is_read_once_per_phase_not_once_per_light() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let cfa = |x: i64| [str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", x), int_card("YBAYROFF", 0)];
    let mut lights = frames(&tmp.path().join("a"), "A", 2, 5000.0, 50.0, &cfa(0));
    lights.extend(frames(&tmp.path().join("b"), "B", 2, 5000.0, 50.0, &cfa(2)));
    let flat = write_f32(&tmp.path().join("flat_bare.fits"), W, H, &(0..W * H).map(|i| 1000.0 + (i % 7) as f32).collect::<Vec<_>>(), &cfa(0));
    let g = [group(lights, None, Some(CalSource::Master(flat)), None)];
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    // Offsets 0 and 2 are one phase, so every light got the one divisor.
    let first = r.frames[0].flat_channel_divisors;
    assert!(first.is_some() && r.frames.iter().all(|f| f.flat_channel_divisors == first));
}

#[test]
fn the_cfa_rule_reads_a_missing_offset_as_zero_and_an_unreadable_pattern_as_mono() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let data: Vec<f32> = (0..W * H).map(|i| 3000.0 + 200.0 * noise(3, i)).collect();
    let bare = write_f32(&tmp.path().join("l_bare.fits"), W, H, &data, &[str_card("BAYERPAT", "RGGB")]);
    let explicit = write_f32(&tmp.path().join("l_explicit.fits"), W, H, &data, &[str_card("BAYERPAT", "RGGB"), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)]);
    let unreadable = write_f32(&tmp.path().join("l_odd.fits"), W, H, &data, &[str_card("BAYERPAT", "QQQQ")]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let one = |light: &PathBuf, name: &str| {
        let g = [group(vec![light.clone()], Some(CalSource::Master(dark.clone())), None, None)];
        calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join(name), OutputFormat::Fits).unwrap()
    };
    let a = one(&bare, "bare");
    let b = one(&explicit, "explicit");
    assert!(a.frames[0].debayered && b.frames[0].debayered, "a pattern with no offsets is a mosaic at phase 0");
    // The data blocks of the two FITS files, after their headers.
    let pixels = |p: &Path| {
        let bytes = std::fs::read(p).unwrap();
        let end = bytes.chunks(80).position(|c| c.starts_with(b"END")).unwrap();
        bytes[((end + 1) * 80).div_ceil(2880) * 2880..].to_vec()
    };
    assert_eq!(pixels(&a.frames[0].output), pixels(&b.frames[0].output), "missing offsets read as 0");
    let c = one(&unreadable, "odd");
    assert!(!c.frames[0].debayered, "an unreadable pattern is a mono frame");
    assert_eq!(c.frames[0].output.file_name().unwrap().to_str().unwrap(), "c_l_odd.fits", "no _d");
    assert_eq!(read_frame_facts(&c.frames[0].output).unwrap().planes, Some(1));
}

fn fpack_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("fpack").join(name)
}

#[test]
fn a_tile_compressed_raw_frame_calibrates_to_the_data_of_its_plain_original() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let dir = tmp.path().join("lights");
    std::fs::create_dir_all(&dir).unwrap();
    let plain = dir.join("Light_plain.fits");
    let fz = dir.join("Light_fz.fz");
    std::fs::copy(fpack_fixture("u16_stars.fits"), &plain).unwrap();
    std::fs::copy(fpack_fixture("fpack_rice_u16.fz"), &fz).unwrap();
    let dark = write_f32(&tmp.path().join("dark.fits"), 64, 64, &vec![100.0; 64 * 64], &[]);
    let run = |light: &PathBuf, name: &str| {
        let g = [group(vec![light.clone()], Some(CalSource::Master(dark.clone())), None, None)];
        calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join(name), OutputFormat::Fits).unwrap()
    };
    let (p, z) = (run(&plain, "plain"), run(&fz, "fz"));
    assert_eq!(bits(&read_plane(&p.frames[0].output, tmp.path()).2), bits(&read_plane(&z.frames[0].output, tmp.path()).2), "calibrated data of the .fz equals the plain file's");
    assert_eq!(read_frame_facts(&z.frames[0].output).unwrap().width, Some(64), "the size comes from the logical header");
    assert_eq!(keys(&z.frames[0].output)["SLM_CSCL"].parse::<f64>().unwrap(), 65535.0);
    assert_eq!(keys(&z.frames[0].output)["SLM_CSRN"], "Light_fz.fz");
    // As raw dark frames too.
    let copies = |src: &str, ext: &str| -> Vec<PathBuf> {
        (0..3)
            .map(|i| {
                let d = tmp.path().join(format!("dark_{ext}_{i}.{ext}"));
                std::fs::copy(fpack_fixture(src), &d).unwrap();
                d
            })
            .collect()
    };
    let (mp, mz) = (tmp.path().join("m_plain.fits"), tmp.path().join("m_fz.fits"));
    build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &copies("u16_stars.fits", "fits"), None, None, &mp, OutputFormat::Fits).unwrap();
    build_master(&env.ctx(&work), &cfg(), MasterKind::Dark, &copies("fpack_rice_u16.fz", "fz"), None, None, &mz, OutputFormat::Fits).unwrap();
    assert_eq!(bits(&read_plane(&mp, tmp.path()).2), bits(&read_plane(&mz, tmp.path()).2));
}

#[test]
fn a_dark_that_leaves_most_pixels_at_or_below_zero_is_warned_about_once_per_group() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let lights = frames(&tmp.path().join("lights"), "Light", 3, 1000.0, 10.0, &[int_card("GAIN", 100), int_card("OFFSET", 50)]);
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![5000.0; W * H], &[]);
    let g = [group(lights, Some(CalSource::Master(dark)), None, None)];
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    let lost: Vec<&String> = r.warnings.iter().filter(|w| w.contains("dark subtraction left most pixels at or below zero")).collect();
    assert_eq!(lost.len(), 1, "{:?}", r.warnings);
    assert_eq!(
        lost[0].as_str(),
        "group 1: dark subtraction left most pixels at or below zero in 3 of 3 frames (OFFSET 50, GAIN 100) \u{2014} the dark is brighter than the lights; check that its gain and offset match those of the lights"
    );
}

#[test]
fn a_cfa_flat_warning_claims_a_recompute_only_where_the_constants_are_per_channel() {
    let tmp = tempfile::tempdir().unwrap();
    let cfa = |p: &str| [str_card("BAYERPAT", p), int_card("XBAYROFF", 0), int_card("YBAYROFF", 0)];
    let lights = frames(&tmp.path().join("lights"), "Light", 1, 1000.0, 10.0, &cfa("RGGB"));
    let other = write_f32(&tmp.path().join("flat_other.fits"), W, H, &vec![2000.0; W * H], &cfa("GRBG"));
    let g = [group(lights, None, Some(CalSource::Master(other)), None)];
    let per_channel = preflight(&g).unwrap();
    assert!(per_channel[0].contains("measured again from its pixels"), "{}", per_channel[0]);
    let mut opts = CalibratedLightOptions::default();
    opts.flat_norm = false;
    let global = starloom::pipeline::preflight_with(&g, &opts).unwrap();
    assert_eq!(global.len(), 1);
    assert!(!global[0].contains("measured again"), "{}", global[0]);
}

#[test]
fn a_master_dark_shared_by_two_groups_is_measured_and_warned_about_once() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let l1 = frames(&tmp.path().join("l1"), "A", 1, 1000.0, 10.0, &[]);
    let l2 = frames(&tmp.path().join("l2"), "B", 1, 1000.0, 10.0, &[]);
    // A dark with no spread has a MAD of 0: no hot-pixel map, and one warning for the run.
    let dark = write_f32(&tmp.path().join("dark.fits"), W, H, &vec![100.0; W * H], &[]);
    let g = [group(l1, Some(CalSource::Master(dark.clone())), None, None), group(l2, Some(CalSource::Master(dark)), None, None)];
    let r = calibrate(&env.ctx(&work), &cfg(), &g, &CalibratedLightOptions::default(), &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    assert_eq!(r.frames.len(), 2);
    let skipped = r.warnings.iter().filter(|w| w.starts_with("Hot-pixel correction skipped")).count();
    assert_eq!(skipped, 1, "{:?}", r.warnings);
}

// ---------------------------------------------------------------------------------------------------------------
// measure
// ---------------------------------------------------------------------------------------------------------------

/// Writes a calibrated mono frame of the 150-star fixture (768 x 512) with `cards`.
fn star_frame(dir: &Path, name: &str, seed: u64, cards: &[Card]) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let (data, w, h) = starloom::test_support::fixture_field(seed, 1.0, 0.002);
    write_f32(&dir.join(name), w, h, &data, cards)
}

fn star_frames(dir: &Path, n: usize, cards: &[Card]) -> Vec<PathBuf> {
    (0..n).map(|k| star_frame(dir, &format!("c_Light_{k:04}.fits"), 7 + k as u64, cards)).collect()
}

fn real_card(keyword: &str, value: f64) -> Card {
    Card::new(keyword, CardValue::Real(value)).unwrap()
}

fn read_csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let head = lines.next().unwrap().split(',').map(str::to_string).collect();
    (head, lines.map(|l| l.split(',').map(str::to_string).collect()).collect())
}

#[test]
fn measure_writes_one_csv_row_per_frame_and_channel_with_the_fixed_columns() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let optics = [real_card("XPIXSZ", 3.76), real_card("FOCALLEN", 1000.0)];
    let mut frames = star_frames(&tmp.path().join("cal"), 3, &optics);
    // One debayered frame: three planes, three rows.
    let (a, w, h) = starloom::test_support::fixture_field(20, 1.0, 0.002);
    let (b, _, _) = starloom::test_support::fixture_field(21, 1.0, 0.002);
    let (c, _, _) = starloom::test_support::fixture_field(22, 1.0, 0.002);
    let all: Vec<f32> = [a, b, c].concat();
    let rgb = tmp.path().join("cal").join("c_Light_0003_d.fits");
    write_fits_f32(&rgb, w, h, 3, &all, &optics).unwrap();
    frames.push(rgb);

    let report = measure(&env.ctx(&work), &cfg(), &frames, None, false).unwrap();
    assert_eq!(report.frames.len(), 4);
    let dir = measure_dir(&work);
    let (head, rows) = read_csv(&dir.join("measure.csv"));
    assert_eq!(head, MEASURE_CSV_COLUMNS.to_vec());
    assert_eq!(rows.len(), 3 + 3);
    assert!(rows.iter().all(|r| r.len() == MEASURE_CSV_COLUMNS.len()));
    let channels: Vec<&str> = rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(channels, ["0", "0", "0", "0", "1", "2"]);
    // fwhm_arcsec = fwhm_px x 206.2648 x pixel size / focal length, from the frame's own facts (no manifest).
    let scale = 206.2648 * 3.76 / 1000.0;
    for r in &rows {
        let (px, arcsec): (f64, f64) = (r[2].parse().unwrap(), r[3].parse().unwrap());
        assert!((arcsec - px * scale).abs() <= 1e-9 * arcsec.abs().max(1.0), "{px} {arcsec}");
        assert_eq!(r[16], "true");
    }
    // The per-plane star fits and, under the default `auto` profile, the first-pass spills are in place.
    for f in &frames {
        assert!(athf_path(&work, f, 0).is_file());
        assert!(pass1_path(&work, f).is_file());
    }
    assert!(athf_path(&work, frames.last().unwrap(), 2).is_file());
    let manifest = Manifest::load(&work).unwrap();
    let entry = manifest.entry(Stage::Measure).unwrap();
    assert_eq!(entry.inputs.len(), 4);
    assert!(entry.products.iter().all(|p| p.exists()));
    assert!(entry.products.contains(&dir.join("measure.json")));
    assert!(!tmp.path().join("work").join("measure.part").exists());
}

#[test]
fn measure_json_round_trips_the_frame_measurements_bit_for_bit_and_refolds_on_read() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = star_frames(&tmp.path().join("cal"), 4, &[]);
    let c = cfg();
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let path = measure_dir(&work).join("measure.json");
    let back = MeasureReport::read(&path).unwrap();
    // Whatever was non-finite is null on both sides; every finite number is the same bits.
    assert_eq!(serde_json::to_string(&back).unwrap(), serde_json::to_string(&report).unwrap());
    for (a, b) in back.frames.iter().zip(&report.frames) {
        let (a, b) = (a.measurement.as_ref().unwrap(), b.measurement.as_ref().unwrap());
        for (ca, cb) in a.channels.iter().zip(&b.channels) {
            assert_eq!(ca.fwhm_px.to_bits(), cb.fwhm_px.to_bits());
            assert_eq!(ca.psf_signal_weight.to_bits(), cb.psf_signal_weight.to_bits());
            assert_eq!(ca.tflux.to_bits(), cb.tflux.to_bits());
        }
    }
    assert_eq!(back.group_beta.map(f64::to_bits), report.group_beta.map(f64::to_bits));
    assert!(report.group_beta.is_some(), "the auto profile resolves a group beta");

    let manifest = Manifest::load(&work).unwrap();
    let sources = source_facts(&manifest, &frames).unwrap();
    // Read back, the stored values give the weights and the selection `measure` computed.
    let again = finish(&back, &c, &sources);
    for (a, b) in again.frames.iter().zip(&back.frames) {
        assert_eq!(a.included, b.included);
        assert_eq!(a.reason, b.reason);
        assert_eq!(
            a.weight.as_ref().unwrap().normalized_mean.to_bits(),
            b.weight.as_ref().unwrap().normalized_mean.to_bits()
        );
    }

    // A plane at the fit-call floor (no aperture correction though stars were fitted) takes its channel's median on
    // every read; the stored value stays as the module made it.
    let mut floored = back.clone();
    let stored = floored.frames[0].measurement.as_mut().unwrap();
    assert!(stored.channels[0].stars_fitted > 0 && stored.channels[0].aperture_correction.is_some());
    stored.channels[0].aperture_correction = None;
    let tflux = stored.channels[0].tflux;
    let folded = finish(&floored, &c, &sources);
    let m = folded.group_c[0].median.expect("the other frames carry a correction");
    let took = &folded.frames[0].measurement.as_ref().unwrap().channels[0];
    assert_eq!(took.aperture_correction_group, Some(m));
    assert_eq!(took.tflux.to_bits(), (tflux * m).to_bits());
    assert_eq!(floored.frames[0].measurement.as_ref().unwrap().channels[0].tflux.to_bits(), tflux.to_bits());
    assert_eq!(folded.group_c[0].planes, 3);
    assert_eq!(folded.group_c[0].count, 1);
}

/// The text of one column of a stars CSV.
fn stars_column(path: &Path, column: &str) -> Vec<String> {
    let (head, rows) = read_csv(path);
    let at = head.iter().position(|h| h == column).unwrap();
    rows.into_iter().map(|r| r[at].clone()).collect()
}

#[test]
fn frames_without_gain_get_one_warning_and_gain_changes_snr() {
    let tmp = tempfile::tempdir().unwrap();
    let c = cfg();
    let gain_warnings = |env: &Env| env.progress.all().iter().filter(|w| w.contains("--gain")).count();

    // Two frames with EGAIN and no SLM_CSCL have no gain: one warning, and no shot term in the star SNR.
    let plain = star_frames(&tmp.path().join("plain"), 2, &[real_card("EGAIN", 2.0)]);
    let env = Env::new();
    let work_a = tmp.path().join("work_a");
    let a = measure(&env.ctx(&work_a), &c, &plain, None, true).unwrap();
    assert_eq!(gain_warnings(&env), 1, "{:?}", env.progress.all());
    assert!(env.progress.all().iter().any(|w| w.contains("2 of 2 frames have no gain")));
    let none = stars_column(&work_a.join("measure").join("stars").join("c_Light_0000.p0.csv"), "snr");
    assert!(!none.is_empty());

    // `--gain 0.5` removes the warning and gives every star an SNR.
    let env = Env::new();
    let work_b = tmp.path().join("work_b");
    let b = measure(&env.ctx(&work_b), &c, &plain, Some(0.5), true).unwrap();
    assert_eq!(gain_warnings(&env), 0, "{:?}", env.progress.all());
    let given = stars_column(&work_b.join("measure").join("stars").join("c_Light_0000.p0.csv"), "snr");
    assert_ne!(none, given, "the shot term changes every star's SNR");
    let (ca, cb) = (&a.frames[0].measurement.as_ref().unwrap().channels[0], &b.frames[0].measurement.as_ref().unwrap().channels[0]);
    // The classic `snr_weight` is the frame's background statistic and never reads the gain; the aperture correction does
    // (its star sample is cut by SNR), so the PSF flux and `psf_snr` move.
    assert_eq!(ca.snr_weight.to_bits(), cb.snr_weight.to_bits());
    assert_ne!(ca.psf_snr.to_bits(), cb.psf_snr.to_bits());

    // EGAIN x SLM_CSCL in the header is the same gain: the same SNR, with no warning. (Frames named apart.)
    let cards = [real_card("EGAIN", 2.0), real_card("SLM_CSCL", 0.25)];
    let headed = star_frames(&tmp.path().join("headed"), 2, &cards);
    let env = Env::new();
    let work_c = tmp.path().join("work_c");
    measure(&env.ctx(&work_c), &c, &headed, None, true).unwrap();
    assert_eq!(gain_warnings(&env), 0, "{:?}", env.progress.all());
    let from_header = stars_column(&work_c.join("measure").join("stars").join("c_Light_0000.p0.csv"), "snr");
    assert_eq!(from_header, given, "EGAIN 2 x SLM_CSCL 0.25 = 0.5");
}

#[test]
fn measure_takes_the_exposure_of_the_raw_light() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    // The calibrated frames carry neither EXPTIME nor EXPOSURE; the raw lights' facts are in the manifest.
    let frames = star_frames(&tmp.path().join("cal"), 3, &[]);
    let mut entry = Entry::new(Stage::Calibrate, &cfg(), Vec::new(), frames.clone());
    for (frame, exptime) in frames.iter().zip([120.0, 60.0, 30.0]) {
        let raw = tmp.path().join("raw").join(frame.file_name().unwrap());
        std::fs::create_dir_all(raw.parent().unwrap()).unwrap();
        write_f32(&raw, W, H, &vec![1.0; W * H], &[real_card("EXPOSURE", exptime)]);
        entry.sources.insert(std::path::absolute(frame).unwrap(), read_frame_facts(&raw).unwrap());
    }
    commit_stage(&work, &[], entry).unwrap();
    let mut c = cfg();
    c.measurement.weight_mode = WeightMode::Exposure;
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let weights: Vec<Vec<f64>> = report.frames.iter().map(|f| f.weight.as_ref().unwrap().channels.clone()).collect();
    assert_eq!(weights, [[120.0], [60.0], [30.0]]);

    // With no calibrate entry the frame's own header is the source.
    let own = star_frames(&tmp.path().join("own"), 3, &[real_card("EXPTIME", 45.0)]);
    let work2 = tmp.path().join("work2");
    let report = measure(&env.ctx(&work2), &c, &own, None, false).unwrap();
    assert!(report.frames.iter().all(|f| f.weight.as_ref().unwrap().channels == [45.0]));
}

#[test]
fn a_moffat_8_measure_runs_one_pass_at_beta_8() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = star_frames(&tmp.path().join("cal"), 3, &[]);
    let mut c = cfg();
    c.measurement.psf_model = PsfModel::Moffat(8.0);
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    assert_eq!(report.group_beta, None, "no group beta: one pass at a fixed profile");
    assert_eq!(report.resolved_group_beta(), Some(8.0));
    for f in &report.frames {
        assert!(f.measurement.as_ref().unwrap().channels.iter().all(|c| c.beta == 8.0));
    }
    for frame in &frames {
        let fits = read_fits(&athf_path(&work, frame, 0)).unwrap();
        assert!(fits.len() > 20 && fits.iter().all(|s| s.beta == 8.0), "{} fits", fits.len());
        assert!(!pass1_path(&work, frame).exists(), "a fixed profile spills no first pass");
    }
    assert!(!measure_dir(&work).join("pass1").exists());
}

#[test]
fn finish_with_another_weight_mode_changes_weights_and_not_measurements() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = star_frames(&tmp.path().join("cal"), 4, &[]);
    let c = cfg();
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let manifest = Manifest::load(&work).unwrap();
    let sources = source_facts(&manifest, &frames).unwrap();
    let mut other = c.clone();
    other.measurement.weight_mode = WeightMode::Noise;
    assert_ne!(c.measurement.weight_mode, other.measurement.weight_mode);
    let again = finish(&report, &other, &sources);
    assert_eq!(again.config, other);
    let first: Vec<u64> = report.frames.iter().map(|f| f.weight.as_ref().unwrap().channels[0].to_bits()).collect();
    let second: Vec<u64> = again.frames.iter().map(|f| f.weight.as_ref().unwrap().channels[0].to_bits()).collect();
    assert_ne!(first, second, "the noise weight is not the PSF signal weight");
    // The measurements are the same values (no plane was floored in this run, so the fold changed nothing).
    for (a, b) in again.frames.iter().zip(&report.frames) {
        assert_eq!(serde_json::to_string(&a.measurement).unwrap(), serde_json::to_string(&b.measurement).unwrap());
    }
    // The input is not touched.
    assert_eq!(report.config, c);
}

/// Raises the cancel flag when the first unit of work is reported done.
struct StopAfterFirst<'a>(&'a AtomicBool);

impl Progress for StopAfterFirst<'_> {
    fn step(&self, _stage: &str, done: usize, _total: usize) {
        if done >= 1 {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn warn(&self, _message: &str) {}
}

#[test]
fn an_interrupted_measure_leaves_no_manifest_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    let frames = star_frames(&tmp.path().join("cal"), 4, &[]);
    let cancel = AtomicBool::new(false);
    let progress = StopAfterFirst(&cancel);
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    let ctx = Ctx { pool, cancel: &cancel, progress: &progress, memory_budget_bytes: 1 << 30, work_dir: &work };
    let err = measure(&ctx, &cfg(), &frames, None, false).unwrap_err();
    assert_eq!(err.to_string(), "cancelled");
    assert!(Manifest::load(&work).unwrap().entry(Stage::Measure).is_none());
    assert!(!measure_dir(&work).exists());
    assert!(dir_entries(&work).iter().all(|n| !n.ends_with(".part")), "{:?}", dir_entries(&work));
}

#[test]
fn measure_refuses_an_xisf_frame_with_the_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let mut frames = star_frames(&tmp.path().join("cal"), 3, &[]);
    let xisf = tmp.path().join("cal").join("c_Light_0009.xisf");
    std::fs::write(&xisf, b"XISF0100").unwrap();
    frames.push(xisf.clone());
    let err = format!("{:#}", measure(&env.ctx(&work), &cfg(), &frames, None, false).unwrap_err());
    assert!(err.contains("c_Light_0009.xisf") && err.contains("plain FITS only") && err.contains("an XISF file"), "{err}");
    assert!(err.contains("--format xisf"), "{err}");
    assert!(Manifest::load(&work).unwrap().entry(Stage::Measure).is_none());
    assert!(!measure_dir(&work).exists());
    // A gzip frame is refused the same way, by its name.
    let gz = gzip_copy(&frames[0], &tmp.path().join("cal").join("c_Light_0010.fits.gz"));
    let err = format!("{:#}", measure(&env.ctx(&work), &cfg(), &[gz], None, false).unwrap_err());
    assert!(err.contains("a compressed file"), "{err}");
}

// ---------------------------------------------------------------------------------------------------------------
// measure: the mirror points (the review round)
// ---------------------------------------------------------------------------------------------------------------

use starloom::stacking::measure::FrameMeasurement;
use starloom::stacking::pass1_artifact::read_pass1_head;
use starloom::test_support::{add_noise, moffat_field, MoffatStar};

/// A calibrated mono frame of 30 round Moffat stars of FWHM 4 px and exponent `beta` (480 x 400), with `cards`.
fn moffat_frame(path: &Path, beta: f64, seed: u64, cards: &[Card]) -> PathBuf {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let alpha = 4.0 / (2.0 * (2f64.powf(1.0 / beta) - 1.0).sqrt());
    let stars: Vec<MoffatStar> = (0..30)
        .map(|i| MoffatStar {
            x: 60.0 + (i % 6) as f64 * 70.0 + 0.3,
            y: 60.0 + (i / 6) as f64 * 60.0 + 0.6,
            amp: 0.04 + 0.01 * i as f64,
            alpha_x: alpha,
            alpha_y: alpha,
            theta: 0.0,
        })
        .collect();
    let mut data = moffat_field(480, 400, &stars, beta, 0.08);
    add_noise(&mut data, 0.002, seed);
    write_f32(path, 480, 400, &data, cards)
}

/// The lower median of `values`, spelled out (Athenaeum's rule for the group beta).
fn lower_median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2]
}

#[test]
fn the_group_beta_is_the_lower_median_of_the_first_passes_and_every_fit_is_made_at_it() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames: Vec<PathBuf> = [2.5, 3.5, 5.0, 7.0]
        .iter()
        .enumerate()
        .map(|(k, &beta)| moffat_frame(&tmp.path().join("cal").join(format!("c_Light_{k:04}.fits")), beta, 30 + k as u64, &[]))
        .collect();
    let report = measure(&env.ctx(&work), &cfg(), &frames, None, false).unwrap();

    // Each frame's own beta, as the first pass found it.
    let own: Vec<f64> = frames.iter().flat_map(|f| read_pass1_head(&pass1_path(&work, f)).unwrap().beta_own).collect();
    assert_eq!(own.len(), 4);
    assert!(own.windows(2).all(|w| w[0] + 0.2 < w[1]), "the frames' own betas are apart: {own:?}");
    // Four values: the lower median is the second, never the upper one (the third) or their mean.
    let expected = lower_median(&own);
    assert_eq!(expected.to_bits(), own[1].to_bits());

    let gb = report.group_beta.expect("the auto profile resolves a group beta");
    assert_eq!(gb.to_bits(), expected.to_bits(), "group beta {gb}, own betas {own:?}");
    assert_eq!(report.resolved_group_beta().map(f64::to_bits), Some(expected.to_bits()));
    assert!(own.iter().any(|b| b.to_bits() != gb.to_bits()), "the group beta is not every frame's own");

    // Every channel and every accepted fit of the second pass was made at it (the `.athf` holds each number as an f32).
    for (f, frame) in report.frames.iter().zip(&frames) {
        for c in &f.measurement.as_ref().unwrap().channels {
            assert_eq!(c.beta.to_bits(), gb.to_bits(), "{}", frame.display());
        }
        let fits = read_fits(&athf_path(&work, frame, 0)).unwrap();
        assert!(fits.len() >= 15, "{} fits", fits.len());
        assert!(fits.iter().all(|s| s.beta as f32 == gb as f32), "{}", frame.display());
    }
}

/// The weight keyword of the calibrated frame as `SSWEIGHT` (one per frame).
fn weighted_frames(dir: &Path, weights: &[f64]) -> Vec<PathBuf> {
    weights
        .iter()
        .enumerate()
        .map(|(k, &w)| star_frame(dir, &format!("c_Light_{k:04}.fits"), 7 + k as u64, &[real_card("SSWEIGHT", w)]))
        .collect()
}

#[test]
fn the_keyword_weight_is_read_from_the_calibrated_header_by_the_key_as_written() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = weighted_frames(&tmp.path().join("cal"), &[1.0, 2.0, 3.0]);
    // The manifest knows the raw lights, whose exposures are not the keyword.
    let mut entry = Entry::new(Stage::Calibrate, &cfg(), Vec::new(), frames.clone());
    for (frame, exptime) in frames.iter().zip([120.0, 60.0, 30.0]) {
        let raw = tmp.path().join("raw").join(frame.file_name().unwrap());
        std::fs::create_dir_all(raw.parent().unwrap()).unwrap();
        write_f32(&raw, W, H, &vec![1.0; W * H], &[real_card("EXPOSURE", exptime), real_card("SSWEIGHT", 9.0)]);
        entry.sources.insert(std::path::absolute(frame).unwrap(), read_frame_facts(&raw).unwrap());
    }
    commit_stage(&work, &[], entry).unwrap();

    let mut c = cfg();
    c.measurement.weight_mode = WeightMode::Keyword;
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let weights: Vec<Vec<f64>> = report.frames.iter().map(|f| f.weight.as_ref().unwrap().channels.clone()).collect();
    assert_eq!(weights, [[1.0], [2.0], [3.0]], "the calibrated frames' cards, not the raw lights'");

    // The key is read as written: a configured keyword that is not a card name as the file holds it finds nothing,
    // and every frame is out with the reason.
    for key in ["ssweight", "SSWEIGHT "] {
        let mut lower = c.clone();
        lower.measurement.keyword = key.to_string();
        let manifest = Manifest::load(&work).unwrap();
        let sources = source_facts(&manifest, &frames).unwrap();
        let none = finish(&report, &lower, &sources);
        for f in &none.frames {
            assert_eq!(f.weight.as_ref().unwrap().missing.as_deref(), Some("no weight keyword value"), "{key:?}");
            assert!(!f.included, "{key:?}");
        }
    }
}

#[test]
fn gain_given_wins_over_a_header_gain() {
    let tmp = tempfile::tempdir().unwrap();
    let c = cfg();
    let snr = |work: &Path| stars_column(&work.join("measure").join("stars").join("c_Light_0000.p0.csv"), "snr");

    // The reference: no header gain, --gain 0.5.
    let plain = star_frames(&tmp.path().join("plain"), 2, &[]);
    let (env_a, work_a) = (Env::new(), tmp.path().join("work_a"));
    measure(&env_a.ctx(&work_a), &c, &plain, Some(0.5), true).unwrap();
    // The header says 2 x 0.5 = 1.0 e-/unit; --gain 0.5 must still be the gain the frames are measured with.
    let headed = star_frames(&tmp.path().join("headed"), 2, &[real_card("EGAIN", 2.0), real_card("SLM_CSCL", 0.5)]);
    let (env_b, work_b) = (Env::new(), tmp.path().join("work_b"));
    measure(&env_b.ctx(&work_b), &c, &headed, Some(0.5), true).unwrap();
    assert_eq!(snr(&work_b), snr(&work_a), "--gain 0.5 over a header gain of 1.0");
    // Without --gain the header's 1.0 is used, and the SNR is another.
    let (env_c, work_c) = (Env::new(), tmp.path().join("work_c"));
    measure(&env_c.ctx(&work_c), &c, &headed, None, true).unwrap();
    assert_ne!(snr(&work_c), snr(&work_a));
}

/// A frame of the 150-star fixture's size with only the first `n` of its stars: too few for an aperture correction.
fn sparse_frame(path: &Path, n: usize, seed: u64) -> PathBuf {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let stars: Vec<(f64, f64, f64)> = grid_stars().into_iter().take(n).collect();
    let mut data = starloom::test_support::gaussian_field(768, 512, &stars, 1.8, 0.08);
    add_noise(&mut data, 0.002, seed);
    write_f32(path, 768, 512, &data, &[])
}

/// A measured group of 4 full frames and one sparse frame (last): its plane is at the fit-call floor.
fn group_with_a_floored_plane(tmp: &Path, env: &Env, work: &Path, c: &AlgorithmConfig) -> (Vec<PathBuf>, MeasureReport) {
    let mut frames = star_frames(&tmp.join("cal"), 4, &[]);
    frames.push(sparse_frame(&tmp.join("cal").join("c_Light_0004.fits"), 6, 99));
    let report = measure(&env.ctx(work), c, &frames, None, false).unwrap();
    (frames, report)
}

#[test]
fn measure_json_holds_the_values_before_the_fold_when_a_plane_is_floored() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let c = cfg();
    let (frames, report) = group_with_a_floored_plane(tmp.path(), &env, &work, &c);
    let sparse = frames.len() - 1;

    // The sparse plane fitted stars and has no aperture correction of its own.
    let stored = report.frames[sparse].measurement.as_ref().unwrap();
    assert!(stored.channels[0].at_floor(), "stars_fitted {}", stored.channels[0].stars_fitted);
    // measure.json (and the returned report) hold it as the module made it: no stand-in, the unscaled flux.
    let back = MeasureReport::read(&measure_dir(&work).join("measure.json")).unwrap();
    assert!(!back.folded && !report.folded);
    for r in [&back, &report] {
        let ch = &r.frames[sparse].measurement.as_ref().unwrap().channels[0];
        assert_eq!(ch.aperture_correction_group, None);
        assert_eq!(ch.tflux.to_bits(), stored.channels[0].tflux.to_bits());
    }
    let tflux = back.frames[sparse].measurement.as_ref().unwrap().channels[0].tflux;

    // `finish` folds: the plane takes the group median, and the flux and the PSF weights scale by it.
    let manifest = Manifest::load(&work).unwrap();
    let sources = source_facts(&manifest, &frames).unwrap();
    let folded = finish(&back, &c, &sources);
    assert!(folded.folded);
    let m = folded.group_c[0].median.expect("the other frames carry a correction");
    let took = &folded.frames[sparse].measurement.as_ref().unwrap().channels[0];
    assert_eq!(took.aperture_correction_group, Some(m));
    assert_eq!(took.tflux.to_bits(), (tflux * m).to_bits());
    assert_eq!(folded.group_c[0].count, 1);
    // The input is untouched, and a full frame's plane is not.
    assert_eq!(back.frames[sparse].measurement.as_ref().unwrap().channels[0].tflux.to_bits(), tflux.to_bits());
    assert_eq!(folded.frames[0].measurement.as_ref().unwrap().channels[0].aperture_correction_group, None);

    // measure.csv holds the folded PSF signal weight, measure.json the unfolded one.
    let (head, rows) = read_csv(&measure_dir(&work).join("measure.csv"));
    let at = head.iter().position(|h| h == "psf_signal_weight").unwrap();
    let csv: f64 = rows[sparse][at].parse().unwrap();
    let json = back.frames[sparse].measurement.as_ref().unwrap().channels[0].psf_signal_weight;
    let fold = folded.frames[sparse].measurement.as_ref().unwrap().channels[0].psf_signal_weight;
    assert_eq!(csv.to_bits(), fold.to_bits());
    assert_ne!(json.to_bits(), fold.to_bits());
}

#[test]
fn finish_folds_a_report_once() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let c = cfg();
    let (frames, report) = group_with_a_floored_plane(tmp.path(), &env, &work, &c);
    let manifest = Manifest::load(&work).unwrap();
    let sources = source_facts(&manifest, &frames).unwrap();
    let once = finish(&report, &c, &sources);
    let twice = finish(&once, &c, &sources);
    assert!(once.folded && twice.folded);
    assert_eq!(serde_json::to_string(&once).unwrap(), serde_json::to_string(&twice).unwrap());
    // A second fold under another weight mode changes the weights and not the measurements.
    let mut other = c.clone();
    other.measurement.weight_mode = WeightMode::Noise;
    let again = finish(&once, &other, &sources);
    for (a, b) in again.frames.iter().zip(&once.frames) {
        assert_eq!(serde_json::to_string(&a.measurement).unwrap(), serde_json::to_string(&b.measurement).unwrap());
    }
    assert_eq!(again.group_c, once.group_c);
}

#[test]
fn frames_are_taken_in_the_order_of_their_dates_and_the_undated_go_last() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let dir = tmp.path().join("cal");
    let dated = |t: &str| [str_card("DATE-OBS", t)];
    // Given: undated, late, early, mid.
    let frames = vec![
        star_frame(&dir, "c_Light_0000.fits", 7, &[]),
        star_frame(&dir, "c_Light_0001.fits", 8, &dated("2026-01-01T03:00:00")),
        star_frame(&dir, "c_Light_0002.fits", 9, &dated("2026-01-01T01:00:00")),
        star_frame(&dir, "c_Light_0003.fits", 10, &dated("2026-01-01T02:00:00")),
    ];
    let name = |p: &Path| p.file_name().unwrap().to_string_lossy().into_owned();
    let expected = ["c_Light_0002.fits", "c_Light_0003.fits", "c_Light_0001.fits", "c_Light_0000.fits"];

    let report = measure(&env.ctx(&work), &cfg(), &frames, None, false).unwrap();
    assert_eq!(report.frames.iter().map(|f| name(&f.file)).collect::<Vec<_>>(), expected);
    let back = MeasureReport::read(&measure_dir(&work).join("measure.json")).unwrap();
    assert_eq!(back.frames.iter().map(|f| name(&f.file)).collect::<Vec<_>>(), expected);
    let entry = Manifest::load(&work).unwrap().entry(Stage::Measure).unwrap().clone();
    assert_eq!(entry.inputs.iter().map(|p| name(p)).collect::<Vec<_>>(), expected);
    let (_, rows) = read_csv(&measure_dir(&work).join("measure.csv"));
    let csv_files: Vec<String> = rows.iter().map(|r| name(Path::new(&r[0]))).collect();
    assert_eq!(csv_files, expected);
}

#[test]
fn a_moffat_4_measure_runs_one_pass_at_beta_4() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = star_frames(&tmp.path().join("cal"), 3, &[]);
    let mut c = cfg();
    c.measurement.psf_model = PsfModel::Moffat4;
    let report = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    assert_eq!(report.group_beta, None, "one pass at a fixed profile has no group beta");
    assert_eq!(report.resolved_group_beta(), Some(4.0));
    for f in &report.frames {
        assert!(f.measurement.as_ref().unwrap().channels.iter().all(|c| c.beta == 4.0));
    }
    for frame in &frames {
        let fits = read_fits(&athf_path(&work, frame, 0)).unwrap();
        assert!(fits.len() > 20 && fits.iter().all(|s| s.beta == 4.0), "{} fits", fits.len());
        assert!(!pass1_path(&work, frame).exists());
    }
    assert!(!measure_dir(&work).join("pass1").exists());
}

/// A measured group of 4 frames and its sources.
fn measured_four(tmp: &Path, env: &Env, c: &AlgorithmConfig) -> (MeasureReport, std::collections::BTreeMap<PathBuf, astroimage::header::FrameFacts>) {
    let work = tmp.join("work");
    let frames = star_frames(&tmp.join("cal"), 4, &[]);
    let report = measure(&env.ctx(&work), c, &frames, None, false).unwrap();
    let sources = source_facts(&Manifest::load(&work).unwrap(), &frames).unwrap();
    (report, sources)
}

#[test]
fn a_frame_with_a_non_finite_location_or_scale_is_out_of_the_weighing() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let c = cfg();
    let (mut report, sources) = measured_four(tmp.path(), &env, &c);
    report.frames[1].measurement.as_mut().unwrap().channels[0].location = f64::NAN;
    report.frames[2].measurement.as_mut().unwrap().channels[0].scale = f64::INFINITY;
    let finished = finish(&report, &c, &sources);
    for k in [1, 2] {
        let f = &finished.frames[k];
        assert!(!f.included && f.weight.is_none(), "frame {k}");
        assert_eq!(f.reason.as_deref(), Some("measurement invalid: non-finite location/scale"));
    }
    for k in [0, 3] {
        assert!(finished.frames[k].included && finished.frames[k].weight.is_some(), "frame {k}");
    }
    // A frame that was measured but not weighed still has its measurement, and the report writes and reads.
    let path = tmp.path().join("nan.json");
    finished.write(&path).unwrap();
    assert!(MeasureReport::read(&path).unwrap().frames[1].measurement.as_ref().unwrap().channels[0].location.is_nan());
}

#[test]
fn the_deferred_weight_floor_marks_the_frame_and_the_other_modes_exclude_it() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let c = cfg();
    assert_eq!(c.measurement.weight_mode, WeightMode::PsfSignalWeight);
    let (mut report, sources) = measured_four(tmp.path(), &env, &c);
    // One frame's PSF weights a thousandth of the others'.
    let weak = |m: &mut FrameMeasurement| {
        for ch in &mut m.channels {
            ch.psf_signal_weight *= 1e-3;
            ch.psf_snr *= 1e-3;
        }
    };
    weak(report.frames[3].measurement.as_mut().unwrap());
    let finished = finish(&report, &c, &sources);
    let weak_row = &finished.frames[3];
    assert!(weak_row.weight.as_ref().unwrap().normalized_mean < c.selection.min_weight_fraction);
    assert!(weak_row.included, "the PSF family defers the floor: {:?}", weak_row.reason);
    assert!(weak_row.measure_floored);
    assert!(finished.frames[..3].iter().all(|f| f.included && !f.measure_floored));

    // The PSF SNR mode defers it too; a mode that does not read these weights has no floor to mark.
    let mut snr = c.clone();
    snr.measurement.weight_mode = WeightMode::PsfSnr;
    let finished = finish(&report, &snr, &sources);
    assert!(finished.frames[3].included && finished.frames[3].measure_floored);
    let mut noise = c.clone();
    noise.measurement.weight_mode = WeightMode::None;
    assert!(finish(&report, &noise, &sources).frames.iter().all(|f| f.included && !f.measure_floored));

    // A frame already out is never marked.
    report.frames[0].measurement.as_mut().unwrap().channels[0].location = f64::NAN;
    assert!(!finish(&report, &c, &sources).frames[0].measure_floored);
}

#[test]
fn a_measure_that_measured_no_frame_fails_and_commits_no_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let dir = tmp.path().join("cal");
    std::fs::create_dir_all(&dir).unwrap();
    let junk: Vec<PathBuf> = (0..3)
        .map(|k| {
            let p = dir.join(format!("c_Light_{k:04}.fits"));
            std::fs::write(&p, b"this is not a fits file").unwrap();
            p
        })
        .collect();
    let err = format!("{:#}", measure(&env.ctx(&work), &cfg(), &junk, None, false).unwrap_err());
    assert!(err.contains("no frame could be measured") && err.contains("c_Light_0001.fits") && err.contains("unreadable"), "{err}");
    assert!(Manifest::load(&work).unwrap().entry(Stage::Measure).is_none());
    assert!(!measure_dir(&work).exists());
    assert!(dir_entries(&work).iter().all(|n| !n.ends_with(".part")), "{:?}", dir_entries(&work));
}

#[test]
fn an_unreadable_frame_outside_the_manifest_is_excluded_with_a_warning_and_the_run_goes_on() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let mut frames = star_frames(&tmp.path().join("cal"), 3, &[]);
    let junk = tmp.path().join("cal").join("c_Light_0009.fits");
    std::fs::write(&junk, b"this is not a fits file").unwrap();
    frames.insert(1, junk.clone());
    let report = measure(&env.ctx(&work), &cfg(), &frames, None, false).unwrap();
    assert_eq!(report.frames.len(), 4);
    let row = report.frames.iter().find(|f| f.file.ends_with("c_Light_0009.fits")).unwrap();
    assert!(row.measurement.is_none() && !row.included);
    assert!(row.reason.as_deref().unwrap().contains("calibrated frame unreadable"), "{:?}", row.reason);
    assert!(report.frames.iter().filter(|f| f.included).count() == 3);
    assert!(env.progress.all().iter().any(|w| w.contains("c_Light_0009.fits") && w.contains("excluded")), "{:?}", env.progress.all());
    assert!(Manifest::load(&work).unwrap().entry(Stage::Measure).is_some());
    // The row is in measure.csv with its reason.
    let (_, rows) = read_csv(&measure_dir(&work).join("measure.csv"));
    assert!(rows.iter().any(|r| r[0].ends_with("c_Light_0009.fits") && r.last().unwrap().contains("unreadable")));
}

#[test]
fn a_compressed_frame_is_not_told_to_drop_format_xisf() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let frames = star_frames(&tmp.path().join("cal"), 1, &[]);
    let gz = gzip_copy(&frames[0], &tmp.path().join("cal").join("c_Light_0010.fits.gz"));
    let err = format!("{:#}", measure(&env.ctx(&tmp.path().join("work")), &cfg(), &[gz], None, false).unwrap_err());
    assert!(err.contains("a compressed file") && !err.contains("--format xisf") && err.contains("decompress"), "{err}");
}

// ---------------------------------------------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------------------------------------------

use starloom::pipeline::{current_weights, register, register_dir, ReferenceChoice, RegisterReport};
use starloom::pipeline::formats::TransformsReport;

/// The 150-star layout of the measure fixture (a jittered 15 x 10 grid of a 768 x 512 plane), as `(x, y, amplitude)`.
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

/// 150 stars at random positions: a field with nothing in common with [`grid_stars`].
fn random_stars() -> Vec<(f64, f64, f64)> {
    (0..150)
        .map(|n| {
            (30.0 + 700.0 * f64::from(noise(11, n)), 30.0 + 450.0 * f64::from(noise(12, n)), 0.1 + 0.2 * f64::from(noise(13, n)))
        })
        .collect()
}

/// A calibrated mono frame of `stars` moved by `(dx, dy)` pixels, amplitudes times `gain`, with its own noise.
fn field_frame(path: &Path, stars: &[(f64, f64, f64)], dx: f64, dy: f64, gain: f64, seed: u64, cards: &[Card]) -> PathBuf {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let moved: Vec<(f64, f64, f64)> = stars.iter().map(|&(x, y, a)| (x + dx, y + dy, a * gain)).collect();
    let mut data = starloom::test_support::gaussian_field(768, 512, &moved, 1.8, 0.08);
    starloom::test_support::add_noise(&mut data, 0.002, seed);
    write_f32(path, 768, 512, &data, cards)
}

const SHIFTS: [(f64, f64); 6] = [(0.0, 0.0), (3.37, -2.81), (-4.2, 5.55), (1.25, 1.9), (-2.6, -3.3), (5.1, 0.45)];

fn shifted_frames(dir: &Path) -> Vec<PathBuf> {
    SHIFTS
        .iter()
        .enumerate()
        .map(|(k, &(dx, dy))| field_frame(&dir.join(format!("c_Light_{k:04}.fits")), &grid_stars(), dx, dy, 1.0, 40 + k as u64, &[]))
        .collect()
}

#[test]
fn register_recovers_the_injected_shifts() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"));
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[0].clone()), &c, false).unwrap();

    assert_eq!(report.transforms.frames.len(), 6);
    assert!(report.transforms.reference_pinned);
    assert_eq!(report.transforms.reference, frames[0]);
    let (cx, cy) = (384.0, 256.0);
    for (k, row) in report.transforms.frames.iter().enumerate() {
        assert!(row.ok, "{:?}", row.error);
        let map = row.map.as_ref().unwrap();
        // The frame's stars sit `shift` away from the reference's, so its map moves a point back by that much.
        let (x, y) = map.forward_exact(cx, cy);
        let (dx, dy) = SHIFTS[k];
        assert!((x - (cx - dx)).abs() < 0.05 && (y - (cy - dy)).abs() < 0.05, "frame {k}: ({x}, {y}) against shift ({dx}, {dy})");
    }
    // The reference's row is the identity.
    let first = &report.transforms.frames[0];
    let (x, y) = first.map.as_ref().unwrap().forward_exact(100.25, 77.5);
    assert_eq!((x, y), (100.25, 77.5));
    assert_eq!((first.inlier_ratio, first.rms_px, first.scale), (1.0, 0.0, 1.0));
    assert!(report.transforms.frames[1..].iter().all(|t| t.rms_px > 0.0));

    // The products are in place and read back; the stage left an entry and no half-written folder.
    let dir = register_dir(&work);
    assert!(dir.join("transforms.json").is_file() && dir.join("weights.json").is_file());
    assert!(!dir.join("registered").exists());
    assert!(!tmp.path().join("work").join("register.part").exists());
    let back = RegisterReport::read(&work).unwrap();
    assert_eq!(back.transforms, report.transforms);
    // (Compared as text: a plane's non-finite numbers read back as NaN, which is not equal to itself.)
    assert_eq!(serde_json::to_string(&back.weights).unwrap(), serde_json::to_string(&report.weights).unwrap());
    assert_eq!(TransformsReport::read(&dir.join("transforms.json")).unwrap().frames.len(), 6);
    let manifest = Manifest::load(&work).unwrap();
    let entry = manifest.entry(Stage::Register).unwrap();
    assert_eq!(entry.inputs.len(), 6);
    assert!(entry.products.iter().all(|p| p.exists()));
    // The reweigh ran: the common star set of the one plane is recorded, and every frame has a weight.
    assert_eq!(report.weights.common_stars.len(), 1);
    assert!(report.weights.common_stars[0].as_ref().is_some_and(|s| s.len() >= 20));
    assert!(report.weights.frames.iter().all(|f| f.weight.is_some() && f.included));
    // `current_weights` over the same inputs gives the same report.
    let again = current_weights(&env.ctx(&work), &measured, &report, &c).unwrap();
    assert_eq!(again, report.weights);
}

#[test]
fn register_accepts_a_reference_outside_the_frames() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"))[..5].to_vec();
    let wcs = [real_card("CRVAL1", 83.82), real_card("CRVAL2", -5.39), str_card("CTYPE1", "RA---TAN")];
    let outside = field_frame(&tmp.path().join("elsewhere").join("reference.fits"), &grid_stars(), 1.0, 1.0, 1.0, 99, &wcs);
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(outside.clone()), &c, true).unwrap();

    assert_eq!(report.transforms.reference, outside);
    assert_eq!(report.transforms.frames.len(), 5);
    let (cx, cy) = (384.0, 256.0);
    for (k, row) in report.transforms.frames.iter().enumerate() {
        assert!(row.ok, "{:?}", row.error);
        let (x, y) = row.map.as_ref().unwrap().forward_exact(cx, cy);
        // The reference stars sit (1, 1) away from the unshifted field.
        let (dx, dy) = (SHIFTS[k].0 - 1.0, SHIFTS[k].1 - 1.0);
        assert!((x - (cx - dx)).abs() < 0.05 && (y - (cy - dy)).abs() < 0.05, "frame {k}: ({x}, {y})");
        // No frame is the reference, so none has the identity: even the unshifted frame moves by one pixel.
        assert!(row.rms_px > 0.0 && row.inlier_ratio > 0.5 && row.inlier_ratio <= 1.0, "frame {k}: {row:?}");
        assert!(!(x == cx && y == cy), "frame {k} got the identity");
    }
    assert!(!report.transforms.frames.iter().any(|t| t.file == outside));

    // The registered frames: one per frame, in the reference's grid, with the reference's world coordinates, never the
    // frame's own, and the stars where the reference has them.
    assert_eq!(report.registered.len(), 5);
    for (k, path) in report.registered.iter().enumerate() {
        assert_eq!(path, &register_dir(&work).join("registered").join(format!("c_Light_{k:04}.fits")));
        let cards = keys(path);
        assert_eq!(cards.get("CTYPE1").map(String::as_str), Some("RA---TAN"));
        assert!(cards.contains_key("CRVAL1") && cards.contains_key("SLM_VER") && cards.get("SLM_REF").map(String::as_str) == Some("reference.fits"));
        no_ath_cards(path);
        let (w, h, data) = read_plane(path, &tmp.path().join(format!("scratch{k}")));
        assert_eq!((w, h), (768, 512));
        // The brightest star of the layout, where the reference puts it.
        let (sx, sy, _) = grid_stars().into_iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        let (cx, cy) = starloom::test_support::centroid(&data, w, sx + 1.0, sy + 1.0, 5, 0.08);
        assert!((cx - (sx + 1.0)).abs() < 0.15 && (cy - (sy + 1.0)).abs() < 0.15, "frame {k}: ({cx}, {cy}) against ({}, {})", sx + 1.0, sy + 1.0);
    }
    let manifest = Manifest::load(&work).unwrap();
    assert!(manifest.entry(Stage::Register).unwrap().products.iter().filter(|p| p.extension().is_some_and(|e| e == "fits")).count() == 5);
}

#[test]
fn register_auto_picks_the_best_weighted_frame_and_skips_floored_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    // Frame 0 is the brightest and so the best weighted; frame 3 is the next.
    let gains = [2.0, 1.0, 1.0, 1.5, 1.0, 1.0];
    let frames: Vec<PathBuf> = gains
        .iter()
        .enumerate()
        .map(|(k, &g)| field_frame(&tmp.path().join("cal").join(format!("c_Light_{k:04}.fits")), &grid_stars(), SHIFTS[k].0, SHIFTS[k].1, g, 50 + k as u64, &[]))
        .collect();
    let c = cfg();
    let mut measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();

    // Under the default config the best weighted frame is the reference, and it gets the identity.
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::Auto, &c, false).unwrap();
    // The pick reads Measure's weights (`finish`'s), not the reweighed ones the report holds.
    let sources = source_facts(&Manifest::load(&work).unwrap(), &frames).unwrap();
    let measure_weights: Vec<f64> =
        finish(&measured, &c, &sources).frames.iter().map(|f| f.weight.as_ref().unwrap().normalized_mean).collect();
    let best = measure_weights.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best, 0, "frame 0 is the brightest: {measure_weights:?}");
    assert_eq!(report.transforms.reference, frames[best]);
    assert!(!report.transforms.reference_pinned);
    let (x, y) = report.transforms.frames[best].map.as_ref().unwrap().forward_exact(12.5, 40.25);
    assert_eq!((x, y), (12.5, 40.25));

    // Take the brightest frame out of the selection (too few stars) and the next in weight wins.
    measured.frames[0].measurement.as_mut().unwrap().channels[0].stars_fitted = 2;
    let mut selective = c.clone();
    selective.selection.min_stars = Some(10);
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::Auto, &selective, false).unwrap();
    assert_eq!(report.transforms.reference, frames[3]);
    assert!(!report.transforms.frames[0].ok);
    assert!(report.transforms.frames[0].error.as_deref().unwrap().starts_with("not registered"));

    // With the floor above every remaining frame's weight, the PSF family defers it: the frames stay included, are
    // under the floor, and no reference pick sees them.
    selective.selection.min_weight_fraction = 0.999;
    let err = format!("{:#}", register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::Auto, &selective, false).unwrap_err());
    assert!(err.contains("no frame can be the reference"), "{err}");
    // Naming a reference needs no pick.
    let named = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[2].clone()), &selective, false).unwrap();
    assert_eq!(named.transforms.reference, frames[2]);
    // PD-17: it stays included under the floor it is below.
    assert!(named.weights.frames[2].included && named.weights.frames[2].reason.is_none());
}

#[test]
fn a_failed_frame_is_excluded_when_the_config_says_so_and_kept_otherwise() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let mut frames = shifted_frames(&tmp.path().join("cal"))[..4].to_vec();
    // A field with nothing in common with the others: it measures, and cannot be aligned.
    frames.push(field_frame(&tmp.path().join("cal").join("c_Light_0009.fits"), &random_stars(), 0.0, 0.0, 1.0, 77, &[]));
    let c = cfg();
    assert!(c.selection.exclude_on_registration_failure);
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    assert!(measured.frames.iter().all(|f| f.included));

    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[0].clone()), &c, false).unwrap();
    let bad = &report.transforms.frames[4];
    assert!(!bad.ok && bad.map.is_none());
    assert!(bad.error.as_deref().unwrap().starts_with("registration failed"), "{:?}", bad.error);
    assert!(report.transforms.frames[..4].iter().all(|t| t.ok));
    let row = &report.weights.frames[4];
    assert!(!row.included && row.reason.as_deref().unwrap().starts_with("registration failed"), "{:?}", row.reason);
    assert!(report.weights.frames[..4].iter().all(|f| f.included));
    assert!(env.progress.all().iter().any(|w| w.contains("c_Light_0009.fits") && w.contains("excluded")));

    let mut strict = c.clone();
    strict.selection.exclude_on_registration_failure = false;
    let err = format!("{:#}", register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[0].clone()), &strict, false).unwrap_err());
    assert!(err.contains("c_Light_0009.fits") && err.contains("registration failed"), "{err}");
    // The refused run changed nothing: the earlier stage's products are as they were.
    assert_eq!(RegisterReport::read(&work).unwrap().transforms, report.transforms);
    assert!(!tmp.path().join("work").join("register.part").exists());
}

#[test]
fn current_weights_refuses_a_newly_included_frame_without_a_transform() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"));
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();

    // `register` runs under a selection that leaves frame 2 out (too few stars there).
    let mut tampered = measured.clone();
    tampered.frames[2].measurement.as_mut().unwrap().channels[0].stars_fitted = 2;
    let mut selective = c.clone();
    selective.selection.min_stars = Some(10);
    let report = register(&env.ctx(&work), &frames, &tampered, &ReferenceChoice::File(frames[0].clone()), &selective, false).unwrap();
    assert!(!report.transforms.frames[2].ok);
    assert!(!report.weights.frames[2].included);
    // The same selection reads the same weights back.
    let same = current_weights(&env.ctx(&work), &tampered, &report, &selective).unwrap();
    assert_eq!(same, report.weights);

    // Under the plain config frame 2 is included, and it has no map.
    let err = format!("{:#}", current_weights(&env.ctx(&work), &measured, &report, &c).unwrap_err());
    assert!(err.contains("c_Light_0002.fits") && err.contains("`starloom register`"), "{err}");
    assert!(!err.contains("c_Light_0001.fits"), "{err}");
    // A frame that `measure` has no row for is refused by `register` with the other command's name.
    let stranger = field_frame(&tmp.path().join("cal").join("c_Light_0042.fits"), &grid_stars(), 0.0, 0.0, 1.0, 5, &[]);
    let err = format!("{:#}", register(&env.ctx(&work), &[stranger], &measured, &ReferenceChoice::Auto, &c, false).unwrap_err());
    assert!(err.contains("c_Light_0042.fits") && err.contains("`starloom measure`"), "{err}");
}

// ---------------------------------------------------------------------------------------------------------------
// The review round for `register`
// ---------------------------------------------------------------------------------------------------------------

fn name_of(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

/// The map of every frame at the plane's centre, for comparing two runs.
fn centre_maps(report: &RegisterReport) -> Vec<(f64, f64)> {
    report.transforms.frames.iter().map(|t| t.map.as_ref().unwrap().forward_exact(384.0, 256.0)).collect()
}

#[test]
fn a_named_reference_the_selection_cuts_is_forced_into_the_stack() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"));
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    // Frame 2 has too few stars for the selection below, and is named the reference.
    let mut tampered = measured.clone();
    tampered.frames[2].measurement.as_mut().unwrap().channels[0].stars_fitted = 2;
    let mut selective = c.clone();
    selective.selection.min_stars = Some(10);

    let report = register(&env.ctx(&work), &frames, &tampered, &ReferenceChoice::File(frames[2].clone()), &selective, false).unwrap();
    let row = &report.weights.frames[2];
    assert!(row.included && row.reason.is_none(), "{:?}", row.reason);
    // It is a member of the reweigh: weighed, and its channel carries the common-set terms.
    assert!(row.weight.is_some());
    let ch = &row.measurement.as_ref().unwrap().channels[0];
    assert!(ch.common_stars.is_some_and(|n| n >= 20) && ch.common_tflux.is_some(), "{ch:?}");
    assert!(report.weights.frames.iter().all(|f| f.included));
    assert!(env.progress.all().iter().any(|w| w.contains("c_Light_0002.fits") && w.contains("kept despite selection")), "{:?}", env.progress.all());
    // Its row in transforms is the identity, and every other frame is registered onto it.
    assert!(report.transforms.frames.iter().all(|t| t.ok));
    // `current_weights` over the same inputs forces it in the same way.
    let again = current_weights(&env.ctx(&work), &tampered, &report, &selective).unwrap();
    assert_eq!(again, report.weights);

    // An auto pick does not: the frame stays out, with its reason, and is no member.
    let auto = register(&env.ctx(&work), &frames, &tampered, &ReferenceChoice::Auto, &selective, false).unwrap();
    assert!(!auto.weights.frames[2].included && auto.weights.frames[2].weight.is_none());
    assert!(!auto.transforms.frames[2].ok);
}

#[test]
fn a_named_reference_under_the_weight_floor_stays_and_an_auto_reference_does_not() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"));
    let mut c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    // A floor that only the group's best weight clears (weights are normalized to the maximum): the reweigh excludes
    // every other member it does not pin.
    c.selection.min_weight_fraction = 0.999;
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[2].clone()), &c, false).unwrap();
    assert!(report.transforms.reference_pinned);
    let clears = |f: &starloom::pipeline::formats::WeightFrame| f.weight.as_ref().unwrap().normalized_mean >= 0.999;
    assert!(!clears(&report.weights.frames[2]), "frame 2 must be under the floor for this test to mean anything");
    for (k, f) in report.weights.frames.iter().enumerate() {
        if k == 2 || clears(f) {
            assert!(f.included && f.reason.is_none(), "frame {k}: {:?}", f.reason);
        } else {
            assert!(!f.included && f.reason.as_deref().unwrap().contains("below"), "frame {k}: {:?}", f.reason);
        }
    }
    assert!(report.weights.frames.iter().filter(|f| !f.included).count() >= 3);
    assert!(env.progress.all().iter().any(|w| w.contains("c_Light_0002.fits") && w.contains("kept despite its weight")));

    // The same stored transforms, read as an auto pick's: nothing pins the frame, and the floor excludes it too.
    let mut unpinned = report.clone();
    unpinned.transforms.reference_pinned = false;
    let again = current_weights(&env.ctx(&work), &measured, &unpinned, &c).unwrap();
    assert!(!again.frames[2].included && again.frames[2].reason.as_deref().unwrap().contains("below"), "{:?}", again.frames[2].reason);
    for (k, f) in again.frames.iter().enumerate().filter(|(k, _)| *k != 2) {
        assert_eq!(f.included, report.weights.frames[k].included, "frame {k}");
    }
}

/// Shifts every star of a frame's own `.athf` by half a pixel in x.
fn doctor_fits(work: &Path, frame: &Path) {
    let path = athf_path(work, frame, 0);
    let mut fits = read_fits(&path).unwrap();
    for f in &mut fits {
        f.x += 0.5;
    }
    starloom::stacking::fits_artifact::write_fits(&path, &fits).unwrap();
}

#[test]
fn registration_reads_measures_fits_for_a_one_plane_group_and_detects_for_any_other() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"));
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let reference = ReferenceChoice::File(frames[0].clone());
    let clean = centre_maps(&register(&env.ctx(&work), &frames, &measured, &reference, &c, false).unwrap());
    let moved = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();

    // A frame's doctored fits move its own map: the stars came from the artifact, not from the pixels.
    doctor_fits(&work, &frames[3]);
    let doctored = centre_maps(&register(&env.ctx(&work), &frames, &measured, &reference, &c, false).unwrap());
    let d = moved(clean[3], doctored[3]);
    assert!((0.3..0.7).contains(&d), "frame 3 moved {d} px");
    for k in [1, 2, 4, 5] {
        assert!(moved(clean[k], doctored[k]) < 0.05, "frame {k}");
    }

    // So do the reference's: every other frame is aligned onto the artifact's stars.
    doctor_fits(&work, &frames[3]);
    let _ = std::fs::remove_dir_all(measure_dir(&work));
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    doctor_fits(&work, &frames[0]);
    let ref_doctored = centre_maps(&register(&env.ctx(&work), &frames, &measured, &reference, &c, false).unwrap());
    let d = moved(clean[1], ref_doctored[1]);
    assert!((0.3..0.7).contains(&d), "frame 1 moved {d} px against a doctored reference");

    // One frame of the group with a second plane switches the whole group to detection (Athenaeum reuses the fits only
    // when the group's stage-one planes are 1): the doctored fits are not read, though frame 3 is still one plane.
    let _ = std::fs::remove_dir_all(measure_dir(&work));
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    doctor_fits(&work, &frames[3]);
    doctor_fits(&work, &frames[0]);
    let mut mixed = measured.clone();
    let extra = mixed.frames[5].measurement.as_ref().unwrap().channels[0].clone();
    mixed.frames[5].measurement.as_mut().unwrap().channels.push(extra);
    let by_detection = centre_maps(&register(&env.ctx(&work), &frames, &mixed, &reference, &c, false).unwrap());
    for k in 1..5 {
        assert!(moved(clean[k], by_detection[k]) < 0.05, "frame {k} moved {} px", moved(clean[k], by_detection[k]));
    }
}

#[test]
fn registered_frames_carry_their_own_cards_and_no_foreign_one() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    // Calibrated lights as Athenaeum's export leaves them, plus the checksums and the stamps of another run.
    let foreign = [
        str_card("OBJECT", "M42"),
        str_card("ATH_CSRC", "calibrated"),
        str_card("ATH_CFLT", "flat.fits"),
        str_card("CALSTAT", "BDF"),
        str_card("CHECKSUM", "abcdefghijklmnop"),
        str_card("DATASUM", "1234567890"),
        str_card("SLM_VER", "0.0.1"),
        str_card("SLM_CFG", "0123456789abcdef"),
        int_card("SLM_PLV", 99),
        str_card("SLM_CDRK", "master_dark.fits"),
    ];
    let frames: Vec<PathBuf> = SHIFTS[..4]
        .iter()
        .enumerate()
        .map(|(k, &(dx, dy))| field_frame(&tmp.path().join("cal").join(format!("c_Light_{k:04}.fits")), &grid_stars(), dx, dy, 1.0, 40 + k as u64, &foreign))
        .collect();
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[1].clone()), &c, true).unwrap();
    assert_eq!(report.registered.len(), 4);
    for path in &report.registered {
        let cards = keys(path);
        let foreign_keys: Vec<&String> =
            cards.keys().filter(|k| k.starts_with("ATH_") || matches!(k.as_str(), "CALSTAT" | "CHECKSUM" | "DATASUM")).collect();
        assert!(foreign_keys.is_empty(), "{} carries {foreign_keys:?}", path.display());
        // The file's own stamps, not the source's.
        assert_eq!(cards.get("SLM_VER").map(String::as_str), Some(env!("CARGO_PKG_VERSION")));
        assert_eq!(cards.get("SLM_CFG"), Some(&starloom::pipeline::config_hash(&c)));
        assert_eq!(cards.get("SLM_PLV"), Some(&starloom::pipeline::PIPELINE_VERSION.to_string()));
        assert_eq!(cards.get("SLM_REF").map(String::as_str), Some("c_Light_0001.fits"));
        // What describes the sky and the calibration of the frame stays.
        assert_eq!(cards.get("OBJECT").map(String::as_str), Some("M42"));
        assert_eq!(cards.get("SLM_CDRK").map(String::as_str), Some("master_dark.fits"));
        let order = card_order(path);
        let at = |k: &str| order.iter().position(|c| c == k).unwrap_or_else(|| panic!("{k} in {order:?}"));
        assert!(at("SLM_VER") < at("SLM_CFG") && at("SLM_CFG") < at("SLM_PLV") && at("SLM_PLV") < at("SLM_REF"), "{order:?}");
        assert_eq!(order.iter().filter(|c| matches!(c.as_str(), "SLM_VER" | "SLM_CFG" | "SLM_PLV" | "SLM_REF")).count(), 4, "{order:?}");
    }
}

#[test]
fn registered_pixels_are_the_warp_the_stack_applies_bit_for_bit() {
    use starloom::resample::{warp_rows, Interpolation, Plane};
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let frames = shifted_frames(&tmp.path().join("cal"))[..4].to_vec();
    let mut c = cfg();
    // Neither is the default's, so a call that drops one of them changes the pixels.
    c.registration.interpolation = Interpolation::Lanczos4;
    c.registration.clamping_threshold = 0.2;
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[0].clone()), &c, true).unwrap();
    for (k, path) in report.registered.iter().enumerate() {
        let (w, h, got) = read_plane(path, &tmp.path().join(format!("got{k}")));
        let (sw, sh, source) = read_plane(&frames[k], &tmp.path().join(format!("src{k}")));
        assert_eq!((w, h, sw, sh), (768, 512, 768, 512));
        let mut want = vec![f32::NAN; w * h];
        let map = report.transforms.frames[k].map.as_ref().unwrap();
        warp_rows(&Plane::full(&source, sw, sh), map, w, 0, h, c.registration.interpolation, c.registration.clamping_threshold, &mut want);
        let differing = got.iter().zip(&want).filter(|(a, b)| !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits()).count();
        assert_eq!(differing, 0, "frame {k}: {differing} pixels differ from a direct warp_rows");
        // The shifted frame has uncovered corners, written as NaN.
        if k > 0 {
            assert!(got.iter().any(|v| v.is_nan()), "frame {k} has no uncovered pixel");
        }
    }
}

#[test]
fn the_reweigh_writes_the_common_set_terms_and_the_weights_of_the_members_only() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    let gains = [1.0, 1.4, 0.8, 1.2, 1.0, 0.9];
    let frames: Vec<PathBuf> = gains
        .iter()
        .enumerate()
        .map(|(k, &g)| field_frame(&tmp.path().join("cal").join(format!("c_Light_{k:04}.fits")), &grid_stars(), SHIFTS[k].0, SHIFTS[k].1, g, 60 + k as u64, &[]))
        .collect();
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    // Frame 4 is out of the selection, so it is no member.
    let mut tampered = measured.clone();
    tampered.frames[4].measurement.as_mut().unwrap().channels[0].stars_fitted = 2;
    let mut selective = c.clone();
    selective.selection.min_stars = Some(10);
    let sources = source_facts(&Manifest::load(&work).unwrap(), &frames).unwrap();
    let before = finish(&tampered, &selective, &sources);
    let report = register(&env.ctx(&work), &frames, &tampered, &ReferenceChoice::File(frames[0].clone()), &selective, false).unwrap();

    let set = report.weights.common_stars[0].as_ref().unwrap().len();
    assert!(set >= 20);
    let mut changed = 0;
    for (k, f) in report.weights.frames.iter().enumerate() {
        if k == 4 {
            assert!(!f.included && f.weight.is_none(), "a non-member carries no weight");
            continue;
        }
        let ch = &f.measurement.as_ref().unwrap().channels[0];
        assert_eq!(ch.common_stars, Some(set));
        let (tflux, tmean) = (ch.common_tflux.unwrap(), ch.common_tmean_flux.unwrap());
        assert!(tflux > 0.0 && tmean > 0.0);
        // The PSF terms are the ones the common flux gives, in the channel's own units.
        assert_eq!(ch.psf_signal_weight.to_bits(), starloom::stacking::psf_signal::psf_signal_weight(tflux, tmean, ch.noise, ch.m_star).to_bits());
        assert_eq!(ch.psf_snr.to_bits(), starloom::stacking::psf_signal::psf_snr(tflux, ch.noise).to_bits());
        let (new, old) = (f.weight.as_ref().unwrap(), before.frames[k].weight.as_ref().unwrap());
        if new.normalized_mean.to_bits() != old.normalized_mean.to_bits() {
            changed += 1;
        }
    }
    assert!(changed > 0, "the reweigh left every weight at its Measure-time value");
    // The members' normalized weights are one scale over the members.
    let sum: f64 = report.weights.frames.iter().filter_map(|f| f.weight.as_ref()).map(|w| w.normalized_mean).sum();
    assert!(sum.is_finite() && sum > 0.0);

    // Under another mode the set is recorded and the weights are Measure's.
    let mut noise_cfg = selective.clone();
    noise_cfg.measurement.weight_mode = WeightMode::Noise;
    let noise_before = finish(&tampered, &noise_cfg, &sources);
    let other = register(&env.ctx(&work), &frames, &tampered, &ReferenceChoice::File(frames[0].clone()), &noise_cfg, false).unwrap();
    assert!(other.weights.common_stars[0].as_ref().is_some_and(|s| s.len() >= 20));
    for (k, f) in other.weights.frames.iter().enumerate() {
        assert_eq!(f.weight, noise_before.frames[k].weight, "frame {k}");
        assert!(f.measurement.as_ref().is_none_or(|m| m.channels[0].common_stars.is_none()), "frame {k}");
    }
}

#[test]
fn the_registered_frames_of_a_report_are_read_in_frame_order() {
    let tmp = tempfile::tempdir().unwrap();
    let env = Env::new();
    let work = tmp.path().join("work");
    // The dates run against the names, so frame order is not the order of the file names.
    let frames: Vec<PathBuf> = (0..4)
        .map(|k| {
            let stamp = format!("2026-01-01T0{}:00:00", 4 - k);
            field_frame(&tmp.path().join("cal").join(format!("c_Light_{k:04}.fits")), &grid_stars(), SHIFTS[k].0, SHIFTS[k].1, 1.0, 70 + k as u64, &[str_card("DATE-OBS", &stamp)])
        })
        .collect();
    let c = cfg();
    let measured = measure(&env.ctx(&work), &c, &frames, None, false).unwrap();
    let expected = ["c_Light_0003.fits", "c_Light_0002.fits", "c_Light_0001.fits", "c_Light_0000.fits"];
    assert_eq!(measured.frames.iter().map(|f| name_of(&f.file)).collect::<Vec<_>>(), expected);
    let report = register(&env.ctx(&work), &frames, &measured, &ReferenceChoice::File(frames[0].clone()), &c, true).unwrap();
    assert_eq!(report.registered.iter().map(|p| name_of(p)).collect::<Vec<_>>(), expected);
    let back = RegisterReport::read(&work).unwrap();
    assert_eq!(back.registered, report.registered);
    // A file in the folder that no entry lists is not a registered frame of the run.
    std::fs::write(register_dir(&work).join("registered").join("stray.fits"), b"x").unwrap();
    assert_eq!(RegisterReport::read(&work).unwrap().registered, report.registered);
}
