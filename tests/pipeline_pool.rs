//! SL-9: every step of `master`, `calibrate`, `measure`, `register`, `normalize` and `stack` that decodes a pixel or fits a star runs on the caller's pool.
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
use starloom::pipeline::{
    build_master, calibrate, measure, normalize, register, stack, CalGroup, CalSource, Ctx, FlatPrecalSource, MasterKind, NoProgress, ReferenceChoice,
};
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
    // The frames are one star field moved by a few pixels, so `register` can align them.
    let stars = tmp.path().join("stars");
    std::fs::create_dir_all(&stars).unwrap();
    let layout: Vec<(f64, f64, f64)> = (0..150)
        .map(|n| {
            let (i, j) = ((n % 15) as f64, (n / 15) as f64);
            (40.0 + i * 48.0 + f64::from(noise(1, n)) * 12.0, 40.0 + j * 48.0 + f64::from(noise(2, n)) * 12.0, 0.1 + 0.2 * f64::from(noise(3, n)))
        })
        .collect();
    let frames: Vec<PathBuf> = [(0.0, 0.0), (2.5, -1.5), (-3.25, 2.0)]
        .iter()
        .enumerate()
        .map(|(k, &(dx, dy))| {
            let moved: Vec<(f64, f64, f64)> = layout.iter().map(|&(x, y, a)| (x + dx, y + dy, a)).collect();
            let mut data = starloom::test_support::gaussian_field(768, 512, &moved, 1.8, 0.08);
            starloom::test_support::add_noise(&mut data, 0.002, 100 + k as u64);
            let path = stars.join(format!("c_Light_{k:04}.fits"));
            write_fits_f32(&path, 768, 512, 1, &data, &[]).unwrap();
            path
        })
        .collect();
    let report = measure(&ctx, &cfg, &frames, None, false).unwrap();
    assert!(report.group_beta.is_some() && report.frames.iter().all(|f| f.included));

    // `register` detects the reference's and the frames' stars (a single-plane frame reuses Measure's fits, so the
    // reference outside the stack gets detection), fits the maps, warps the registered copies and reads the `.athf`
    // files for the reweigh, all on the caller's pool.
    let outside = stars.join("reference.fits");
    {
        let mut data = starloom::test_support::gaussian_field(768, 512, &layout, 1.8, 0.08);
        starloom::test_support::add_noise(&mut data, 0.002, 999);
        write_fits_f32(&outside, 768, 512, 1, &data, &[]).unwrap();
    }
    let registered = register(&ctx, &frames, &report, &ReferenceChoice::File(outside), &cfg, true).unwrap();
    assert!(registered.transforms.frames.iter().all(|t| t.ok), "{:?}", registered.transforms.frames.iter().map(|t| t.error.clone()).collect::<Vec<_>>());
    assert_eq!(registered.registered.len(), 3);
    let in_stack = register(&ctx, &frames, &report, &ReferenceChoice::Auto, &cfg, false).unwrap();
    assert!(in_stack.transforms.frames.iter().all(|t| t.ok));

    // `normalize` builds the LN reference (an integration and its saturation map), models its background, detects and fits
    // its stars, measures the seeds calibration and normalizes every frame, all on the caller's pool.
    let normalized = normalize(&ctx, &frames, &report, &in_stack, &cfg).unwrap();
    assert!(
        normalized.frames.iter().all(|f| f.athln.is_some()),
        "{:?}",
        normalized.frames.iter().map(|f| f.reason.clone()).collect::<Vec<_>>()
    );

    // `stack` reweighs, reads the `.athln` grids, warps every frame on the fly, integrates with rejection and measures
    // the master, all on the caller's pool.
    let stacked = stack(&ctx, &cfg, &frames, &report, &in_stack, Some(&normalized), &tmp.path().join("masters"), OutputFormat::Fits).unwrap();
    assert_eq!(stacked.stats.included, 3);

    // Nothing built rayon's global pool: asking for it now succeeds (it fails once a pool has been built).
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().expect("rayon's global pool was built by a stage that ran outside the caller's pool");
}
