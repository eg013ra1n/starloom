# Changelog

All notable changes to starloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The pure items a pipeline layer is built from moved here from Athenaeum's core, byte for byte: `StackingPreset`,
  `AlgorithmConfig` (the seven algorithm sections), `algorithm_preset` and the four stage subtrees
  (`calibration_subtree`, `measurement_subtree`, `registration_subtree`, `normalization_subtree`, with their
  versioned variants) in `stacking::config`; `CalibratedLightOptions` (`calibration_library::options`);
  `resolve_recipe` (`integration::recipe`); the debayer glue and `COPY_THROUGH_KEYWORDS` (`calibration_library`);
  `flat_channel_norms` (`integration::cfa`); `sky_penalized_order`, `reference_coverage` and
  `MIN_REFERENCE_COVERAGE` (`stacking::weights`); `header_pixel_scale_arcsec` (`geometry`); the `.athf` format with
  `FITS_ARTIFACT_VERSION` 3 (`stacking::fits_artifact`); and `map_scale`, `wants_cfa_mosaic` and `pick_ln_anchor`
  (`stacking::run_helpers`). No stage output moves and no version changes; the `.athf` format gets a golden row.
- The `pipeline` layer (feature `pipeline`, on `stack`; optional `toml` and `glob`), the skeleton the working-folder
  stages hang on, and the `cli` feature (adds `clap` and `tracing-subscriber`; the binary follows). `pipeline` holds
  `Ctx` and `Progress` (the caller's pool, a cancel flag, a progress sink, a memory budget, the working folder; no
  thread pool is built); `expand_inputs` (files, folders and globs, by file name, refusing two frames with one stem),
  `parse_date_obs`, `order_lights` and `read_plan`; `effective_config` (preset, then file, then `--set`), `stage_hash`
  and `config_hash`; and the versioned working-folder formats: the manifest `starloom.json` (entries, `FileId`s, the
  per-light facts and the command history) and the `measure`, `transforms` and `weights` reports, each tagged and
  refused when made by another version, with `check_upstream` and `frames_or_upstream` and the rule that a stage's
  entry is written only after its products are in place. `fs_util::sampled_xxh3` is Athenaeum's sampled content hash.
- `[profile.release]` is `lto = "thin"`, `codegen-units = 4`, Athenaeum's release profile, for a standalone build.

## [0.1.2] — 2026-11-23

Test-only. No stage output moves; every stage version stays and the golden rows are unchanged.

### Fixed

- The drizzle test `deposit_skip_predicate_never_rejects_a_pixel_with_nonzero_clipped_area` holds
  `grid_counters::exclusive()`: it built TPS grids without it, and on a 16-thread Windows runner one of them again
  landed inside `registered_source`'s one-grid-per-frame count. It was the last such test (an `on_build` check run on
  one thread found no other).

### Changed

- This crate's own tests enforce the rule: a TPS grid built while nobody holds the lock panics in the test that built
  it, by name, instead of skewing another test's count (`cfg(test)` only; `test-support` builds are unchanged).

## [0.1.1] — 2026-11-23

Test-only. No stage output moves; every stage version stays and the golden rows are unchanged.

### Fixed

- The drizzle phase-table TPS pin and the common-set flipped-map test hold `grid_counters::exclusive()`, as every
  test that builds a TPS grid must: the counters are process-global, and on a 16-thread Windows runner one of their
  grids landed inside `registered_source`'s one-grid-per-frame count (2, not 1).

## [0.1.0] — 2026-11-23

The stacking kernel as its own repository, extracted from Athenaeum bit for bit: geometry and resampling, the
FITS/XISF writer core, light calibration, star measurement, registration, outlier-rejecting integration, drizzle and
frame weighting, in three layers (base, `engine`, `stack`) plus `ts` and `test-support`. Dependencies resolve by
version: rustafits 1.9.1 from crates.io and solvemyastro by git revision.

### Stage versions

`MEASURE_STAGE_VERSION` 1, `ALIGN_VERSION` 1, `LN_MODEL_VERSION` 4, `SNR_GAIN_VERSION` 2, `DRIZZLE_KERNEL_VERSION` 2,
`PASS1_ARTIFACT_VERSION` 1, the `.athln` `LN_GRID_VERSION` 1, `LIGHT_CAL_VERSION` 1, `INTEGRATE_VERSION` 1, `ENGINE_VERSION` 1; rustafits'
`MEASURE_VERSION` 8. The stage goldens (`tests/stage_goldens.rs`) hold these.
