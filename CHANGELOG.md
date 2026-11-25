# Changelog

All notable changes to starloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-11-25

starloom is now a program as well as a library. The new `starloom` command calibrates, measures, registers, normalizes
and stacks FITS and XISF frames from raw files, with no catalog and no application around it, and it writes the same
bytes as Athenaeum's own run on the same frames, masters and config. No stage output of the existing library moves:
every stage version stays and every existing golden row is unchanged.

### Added

- **The `starloom` command** (feature `cli`; `cargo install --git https://github.com/eg013ra1n/starloom --features cli
  --locked`). Commands: `facts`, `master <bias|dark|flat>`, `calibrate`, `measure`, `register`, `normalize`, `stack`,
  `run` (every stage in one go) and `config show`. Global flags: `-w/--work`, `--preset`, `--config`, `--set
  path=value`, `--threads`, `--memory`, `-v/-vv`. The short flags (`--psf`, `--weights`, `--rejection`, `--drizzle`,
  `--no-debayer`, `--no-hot-pixels`) are each one `--set`. An Athenaeum stacking config works as `--config`; the keys
  the command does not read are named on stderr.
- **A working folder holds the stages' products** (`starloom-work` by default): `masters/`, `calibrated/`, `measure/`,
  `register/`, `ln/`, `stack/`, `logs/` and the manifest `starloom.json`. A command given no frames takes them from the
  previous stage's entry there, in their recorded order, and `--measure`, `--transforms` and `--ln` name the products to
  read instead. Frames given on the command line win. A product named on the command line is read together with its own
  folder, so a stage can run in a fresh working folder.
- **Manual mode.** The stages are not linked. The command never compares one stage's products with another's and never
  refuses a product as stale: you decide what to re-run and what to clean. A missing input (a frame with no
  measurement, transform or normalization file) is an ordinary error that names the frame and the command that makes it.
  The manifest is a journal of what each command did, for you to read.
- **Logs and history.** Every command except `facts` and `config show` writes `<work>/logs/<UTC stamp>-<command>.jsonl`
  (debug-level JSON, whatever `-v` says), appends a row to the manifest's `history` (also when it fails or is
  refused) and ends with one summary line on stderr. stdout carries products only (`facts --json`, `config show`).
  Exit codes: 0 done, 1 an internal or I/O failure, 2 an input you must change (a usage error, a bad `--set` or config,
  a frame with no transform, a calibration that does not fit, a light that is a dark, flat or bias, fewer than three
  frames). There is no Ctrl-C handler: a killed command leaves only `.tmp/` and part files, which the next writing
  command clears.
- **Compressed raw frames.** `master`, `calibrate`, `run` and `facts` take `.fz`, `.fits.gz` and FITS with extension
  HDUs, besides plain FITS and XISF. The products are plain FITS (or XISF with `--format xisf`). Reading a gzip file's
  header inflates the whole file, which takes seconds per frame.
- **`--psf auto|moffat4|moffat=<beta>`.** A Moffat star profile with your own beta (1.5 to 100), in measurement and in
  local normalization. `moffat=4` measures bit for bit like `moffat4` but has its own hash.
- **Files that carry what made them.** Every file the command writes has `SLM_VER`, `SLM_CFG` and `SLM_PLV`; masters
  `SLM_REC` (and a flat's `SLM_FNRM`, `SLM_FNR`, `SLM_FNG`, `SLM_FNB`); calibrated lights an `SLM_` twin of every card
  Athenaeum writes there; master lights `SLM_REF`, `SLM_MSV`, `SLM_MV`, `SLM_ALV`, `SLM_LNV`, `SLM_INV` (and
  `SLM_DZV` on a drizzle). The command writes and reads no `ATH_` card.
- **Versioned working-folder formats**, each tagged and refused when written by another version: the manifest
  `starloom.json`, `measure.json` (with `measure.csv`), `transforms.json`, `weights.json` and `ln/normalize.json`. The
  star fits are `.athf` files and the local-normalization grids `.athln` files, as in Athenaeum. A non-finite float is
  stored as `null`.
- **The `pipeline` layer** (feature `pipeline`, on `stack`) that the command drives, usable as a library on the
  caller's rayon pool: inputs (files, folders and globs expanded by file name, refusing two frames with one stem),
  calibration plans, config levels (preset, file, `--set`, then Athenaeum's resolution), stage hashes, and the stages
  `build_master`, `calibrate`, `measure`, `register`, `normalize`, `stack` and `run`. It builds no thread pool and
  writes nothing to stdout or stderr. Every mismatch it can find from headers alone (a calibration frame of another
  size, a flat of another CFA phase, a raw set of fewer than three frames) is found before any pixel is read.
- **`stacking::config::resolve_algorithm`**, the algorithm half of Athenaeum's config resolution (the clamps, the LN
  fields, the `fitStars` canonical form), so a `--set` value resolves and hashes as it does in Athenaeum.
- A `pipeline` row in the stage goldens (a whole `run` with the drizzle) and a golden row for the `.athf` format.

### Changed

- rustafits 1.10.0 (from crates.io): `.fz`, `.fits.gz` and extension-HDU files, one public header reader, and
  `PsfModel::Moffat(f64)` (the TypeScript mirror carries it too).
- Items that were pure functions of Athenaeum's core moved here byte for byte, and core re-exports them at the old paths:
  `StackingPreset`, `AlgorithmConfig`, `algorithm_preset` and the four stage subtrees, `CalibratedLightOptions`,
  `resolve_recipe`, the debayer glue, `flat_channel_norms`, `sky_penalized_order` and `reference_coverage`,
  `header_pixel_scale_arcsec`, `map_scale`, `wants_cfa_mosaic`, `pick_ln_anchor`, `orientation::row_order_is_bottom_up`
  and the `.athf` format with `FITS_ARTIFACT_VERSION` 3. `light_cal::calibrate_light_compute_resolved` takes a flat
  divisor resolved once for a batch.
- `[profile.release]` is `lto = "thin"`, `codegen-units = 4`, Athenaeum's release profile. Equality with Athenaeum is
  claimed for that profile only.
- CI runs the `cli` layer on Linux, macOS and Windows and installs the binary on Linux.
- The stage golden rows now cover x86_64 Linux and Windows as well as macOS, each copied from its own CI run, so every platform checks its numbers and not only their determinism.

### Differences from Athenaeum's own run

On the pixel path the command mirrors Athenaeum's run step for step (measured on two lab sets, a mono and an OSC one:
equal at every stage, masters, calibrated frames, measurements, transforms, weights, normalization, stack and drizzle).
What it does differently, on purpose:

- It registers everything onto one reference, a calibrated frame you name or the best-weighted frame (`auto`); it has
  no two-pass reference pick, no plate-solve seed, no project grid and no `registration.geometry`. A frame Athenaeum
  rescues with the plate-solve seed can fail here.
- A reference outside the frames gets fresh star detection instead of the measurement's star fits.
- With fewer than three included frames, `measure` and `register` warn and go on where Athenaeum skips the group;
  `stack` still refuses to integrate fewer than three.
- An exact tie between two frames when the normalization anchor is picked breaks on the frame index, not the catalog
  id; so does an exact weight tie in local normalization's seed-calibration frame selection (it can pick another
  sample, hence another scale).
- A Bayer drizzle needs the calibrated mosaics (`calibrate --keep-mosaic`) and is refused without them, where
  Athenaeum falls back to the debayered deposit. No drizzle coverage map is written.
- `run --reference <file>` needs the file to exist before calibration: give a calibrated frame, or use `auto`.
- The frame order is by DATE-OBS, then by input position (Athenaeum's is by DATE-OBS, then catalog id).
- Frame paths are compared as given, case-sensitively, also on Windows.

### Stage versions

No stage version moves: `MEASURE_STAGE_VERSION` 1, `ALIGN_VERSION` 1, `LN_MODEL_VERSION` 4, `SNR_GAIN_VERSION` 2,
`DRIZZLE_KERNEL_VERSION` 2, `PASS1_ARTIFACT_VERSION` 1, the `.athln` `LN_GRID_VERSION` 1, `FITS_ARTIFACT_VERSION` 3
(the `.athf` version, now starloom's), `LIGHT_CAL_VERSION` 1, `INTEGRATE_VERSION` 1, `ENGINE_VERSION` 1; rustafits'
`MEASURE_VERSION` 8. New, for the working folder: `PIPELINE_VERSION`, `MANIFEST_VERSION`, `MEASURE_REPORT_VERSION`,
`TRANSFORMS_VERSION`, `WEIGHTS_VERSION` and `NORMALIZE_REPORT_VERSION`, each 1.

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
