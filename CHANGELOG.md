# Changelog

All notable changes to starloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The pure items a pipeline layer is built from moved here from Athenaeum's core, byte for byte: `StackingPreset`,
  `AlgorithmConfig` (the seven algorithm sections), `algorithm_preset` and the four stage subtrees
  (`calibration_subtree`, `measurement_subtree`, `registration_subtree`, `normalization_subtree`) in
  `stacking::config`; `CalibratedLightOptions` (`calibration_library::options`);
  `resolve_recipe` (`integration::recipe`); the debayer glue and `COPY_THROUGH_KEYWORDS` (`calibration_library`);
  `flat_channel_norms` (`integration::cfa`); `sky_penalized_order`, `reference_coverage` and
  `MIN_REFERENCE_COVERAGE` (`stacking::weights`); `header_pixel_scale_arcsec` (`geometry`); the `.athf` format with
  `FITS_ARTIFACT_VERSION` 3 (`stacking::fits_artifact`); and `map_scale`, `wants_cfa_mosaic` and `pick_ln_anchor`
  (`stacking::run_helpers`). No stage output moves and no version changes; the `.athf` format gets a golden row.
- `stacking::config::resolve_algorithm`, the algorithm half of Athenaeum's config resolution, moved here verbatim with
  `resolve_ln_fields`, `clamp_detection_sigma`, `clamp_tps_smoothing` and their bounds (`MIN_`/`MAX_DETECTION_SIGMA`,
  `MIN_`/`MAX_TPS_SMOOTHING`, `MIN_`/`MAX_SEED_THRESHOLD`, `MIN_FIT_STARS`): the `tpsSmoothing` and large-scale filter
  clamps, the LN fields, the seed clamps and the `fitStars` canonicalisation. Athenaeum's resolution calls it; its
  resolved configs are byte-identical (pinned before the move).
- The `pipeline` layer (feature `pipeline`, on `stack`; optional `toml` and `glob`), the skeleton the working-folder
  stages hang on, and the `cli` feature (adds `clap` and `tracing-subscriber`; the binary follows). `pipeline` holds
  `Ctx` and `Progress` (the caller's pool, a cancel flag, a progress sink, a memory budget, the working folder; no
  thread pool is built); `expand_inputs` (files, folders and globs, by file name, refusing two frames with one stem),
  `expand_calibration_frames` (no stem rule), `parse_date_obs`, `order_lights` and `read_plan` (relative paths resolve
  against the plan's folder); `effective_config` (preset, then file, then `--set` on the whole config, then
  `resolve_algorithm`; it names every key of the file it does not read, as a dotted path), `stage_hash`, `config_hash`
  (Athenaeum's canonical key order) and `ln_runs`; `versions` (`all_versions`, every version constant by its own name,
  and `VERSION_CARDS`); and the versioned working-folder formats: the manifest `starloom.json` (per stage its hashes,
  versions, inputs, products and the per-light facts, and the command history) and the `measure`, `transforms` and
  `weights` reports, each tagged and refused when made by another version, with `frames_or_upstream`, `source_facts`
  and the rule that a stage's entry is written only after its products are renamed into place beside their
  destination. Manual mode: the stages are not linked, and nothing compares one stage's products with another's. A
  non-finite float is stored as `null`, as Athenaeum's metrics payloads store it, and reads back as NaN.
- The `master` and `calibrate` stages of `pipeline`: `build_master` (a bias, dark or flat from raw frames, Athenaeum's
  master build without the catalog: the automatic recipe, the frames integrated in the order of their full paths, the
  cards of `build_master_cards` with `SLM_*` in place of its own: `SLM_REC` holds what `ATH_REJ` holds (`Average |
  Winsorized sigma (3.0/3.0) n=24`), `SLM_N`, `SLM_TMIN`/`SLM_TMAX`, the flat's `SLM_FNRM` and per-channel `SLM_FNR`/
  `SLM_FNG`/`SLM_FNB`, the common `SLM_*` cards, `Master Dark Flat` for dark-flat frames, and no `ATH_*` card),
  `calibrate` (a group's lights to calibrated frames: dark else bias, flat, hot pixels from the master dark, the VNG
  debayer of an OSC light, the CFA mosaic kept beside it with `keep_mosaic` as `execute_generation` writes it; the
  header is Athenaeum's with an `SLM_*` twin of every `ATH_*` card but `ATH_CSRC`: `SLM_CSRN`, `SLM_CDRK`/`SLM_CFLT`/
  `SLM_CBIA` (the master's file name), `SLM_CSCL`, `SLM_CFNM`, `SLM_CPED`, `SLM_LCV`, `SLM_CCFA`/`SLM_CFNR`/`SLM_CFNG`/
  `SLM_CFNB`, `SLM_CTRM`, then `SLM_CDBM` and `SLM_CHPX`; a hot-pixel map per master dark and a flat divisor per (flat,
  mosaic phase) for the run; the scale divisor 65535 where Athenaeum's probe reads no depth; a group with no master, or
  a bias alone with the fallback off, has its lights excluded with Athenaeum's reasons; the dark guard's warning; the
  manifest entry records each raw light's facts) and `preflight`/`preflight_with` (every mismatch found from headers
  alone before any pixel is read: a calibration frame of another size, a frame with more than one plane, a raw set
  with fewer than three frames and a flat pre-calibration of another size are refused, another sub-frame origin and a
  flat of another CFA phase are warned about). Raw frames may be compressed (`.fz`, `.fits.gz`, an extension HDU), and
  the products are plain FITS. Every step that decodes a pixel runs on the caller's pool (SL-9). `SlmFlatCards`
  (`calibration_library::flat_cards`, engine layer) reads a master flat's stamped constants as Athenaeum's reader of
  its own cards does (plain FITS only), CFA phase check included. `light_cal::calibrate_light_compute_resolved` takes a
  flat divisor the caller resolved once for a batch.
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
