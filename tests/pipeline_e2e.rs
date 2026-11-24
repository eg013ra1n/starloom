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
