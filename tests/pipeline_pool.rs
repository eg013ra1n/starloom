//! SL-9: every step of `master`, `calibrate` and `measure` that decodes a pixel or fits a star runs on the caller's pool.
//!
//! A compressed raw frame, a compressed pre-calibration master and a compressed light all go through the image
//! reader's spill, which decodes with rayon iterators on whatever pool is current. Outside the caller's pool that is
//! rayon's global pool, built on every core whatever the caller asked for. This test runs the whole of that on a
//! one-thread pool and then asks rayon to build its global pool: it can only succeed if nothing built it before. It
//! lives in a test binary of its own so no other test can build the global pool first.
#![cfg(feature = "pipeline")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use starloom::calibration_library::options::CalibratedLightOptions;
use starloom::fits_writer::{write_fits_f32, OutputFormat};
use starloom::pipeline::{build_master, calibrate, measure, CalGroup, CalSource, Ctx, FlatPrecalSource, MasterKind, NoProgress};
use starloom::stacking::config::{algorithm_preset, StackingPreset};

// Large enough that the reader's decode runs in parallel (it does from 131,072 samples).
const W: usize = 400;
const H: usize = 340;

fn noise(seed: u64, i: usize) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % 10_000) as f32 / 10_000.0
}

/// `n` gzipped float frames of one level with a little noise (and one hot pixel when asked).
fn gz_frames(dir: &Path, prefix: &str, n: usize, base: f32, hot: bool) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|k| {
            let mut data: Vec<f32> = (0..W * H).map(|i| base + 6.0 * noise(k as u64 + 1, i)).collect();
            if hot {
                data[5 * W + 7] = 60_000.0;
            }
            let plain = dir.join(format!("{prefix}_{k:04}.fits"));
            write_fits_f32(&plain, W, H, 1, &data, &[]).unwrap();
            let gz = dir.join(format!("{prefix}_{k:04}.fits.gz"));
            let mut enc = flate2::write::GzEncoder::new(std::fs::File::create(&gz).unwrap(), flate2::Compression::fast());
            enc.write_all(&std::fs::read(&plain).unwrap()).unwrap();
            enc.finish().unwrap();
            std::fs::remove_file(&plain).unwrap();
            gz
        })
        .collect()
}

#[test]
fn compressed_frames_decode_on_the_callers_pool_and_never_on_rayons_global_one() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let cancel = AtomicBool::new(false);
    let ctx = Ctx { pool, cancel: &cancel, progress: &NoProgress, memory_budget_bytes: 1 << 30, work_dir: &work };
    let cfg = algorithm_preset(StackingPreset::Default);

    let darks = gz_frames(&tmp.path().join("darks"), "Dark", 16, 100.0, true);
    let flats = gz_frames(&tmp.path().join("flats"), "Flat", 4, 3000.0, false);
    let lights = gz_frames(&tmp.path().join("lights"), "Light", 3, 1000.0, false);
    // A compressed pre-calibration master: `load_precal_master` decodes it.
    let precal = gz_frames(&tmp.path().join("precal"), "Bias", 1, 200.0, false).remove(0);

    // `master` on compressed raw frames and a compressed pre-calibration master, on its own.
    let master = tmp.path().join("flat_master.fits");
    build_master(&ctx, &cfg, MasterKind::Flat, &flats, Some(&FlatPrecalSource::Master(precal.clone())), None, &master, OutputFormat::Fits).unwrap();

    // `calibrate` builds the dark master from compressed raw frames, measures its hot pixels, preloads the masters and
    // calibrates compressed lights.
    let groups = [CalGroup {
        lights,
        dark: Some(CalSource::Raw(darks)),
        flat: Some(CalSource::Raw(flats)),
        bias: None,
        flat_precal: Some(FlatPrecalSource::Master(precal)),
    }];
    let report = calibrate(&ctx, &cfg, &groups, &CalibratedLightOptions::default(), &tmp.path().join("out"), OutputFormat::Fits).unwrap();
    assert_eq!(report.frames.len(), 3);
    assert!(report.frames.iter().all(|f| f.hot_pixels_replaced.is_some()), "the hot-pixel pass ran from the built master dark");

    // `measure` on star frames, two passes with a spilled first pass: the fits and the fan-out run on the caller's pool.
    let stars = tmp.path().join("stars");
    std::fs::create_dir_all(&stars).unwrap();
    let frames: Vec<PathBuf> = (0..3)
        .map(|k| {
            let (data, w, h) = starloom::test_support::fixture_field(7 + k, 1.0, 0.002);
            let path = stars.join(format!("c_Light_{k:04}.fits"));
            write_fits_f32(&path, w, h, 1, &data, &[]).unwrap();
            path
        })
        .collect();
    let report = measure(&ctx, &cfg, &frames, None, false).unwrap();
    assert!(report.group_beta.is_some() && report.frames.iter().all(|f| f.included));

    // Nothing built rayon's global pool: asking for it now succeeds (it fails once a pool has been built).
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().expect("rayon's global pool was built by a stage that ran outside the caller's pool");
}
