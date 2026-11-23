# Changelog

All notable changes to starloom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-11-23

The stacking kernel as its own repository, extracted from Athenaeum bit for bit: geometry and resampling, the
FITS/XISF writer core, light calibration, star measurement, registration, outlier-rejecting integration, drizzle and
frame weighting, in three layers (base, `engine`, `stack`) plus `ts` and `test-support`. Dependencies resolve by
version: rustafits 1.9.1 from crates.io and solvemyastro by git revision.

### Stage versions

`MEASURE_STAGE_VERSION` 1, `ALIGN_VERSION` 1, `LN_MODEL_VERSION` 4, `SNR_GAIN_VERSION` 2, `DRIZZLE_KERNEL_VERSION` 2,
`PASS1_ARTIFACT_VERSION` 1, the `.athln` `LN_GRID_VERSION` 1, `LIGHT_CAL_VERSION` 1, `INTEGRATE_VERSION` 1, `ENGINE_VERSION` 1; rustafits'
`MEASURE_VERSION` 8. The stage goldens (`tests/stage_goldens.rs`) hold these.
