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
use starloom::fits_writer::bayer::Bayer;
use starloom::fits_writer::{write_fits_f32, Card, CardValue, OutputFormat};
use starloom::integration::banded::{BandPlanes, BandSource};
use starloom::integration::cfa::{flat_channel_norms, CfaGeometry};
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
    assert_eq!(report.recipe_text, "average winsorized 3 3");

    let k = keys(&out);
    assert_eq!(k["SLM_REC"], "average winsorized 3 3");
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
    assert!(two.to_string().contains("at least 3"), "{two}");
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
    assert_eq!(k["SLM_REC"], "average percentile 0.2 0.02", "fewer than 15 flats take the percentile recipe");
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
