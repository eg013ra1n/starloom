# starloom

Starloom is the astrophotography stacking kernel behind Athenaeum: geometry and resampling, FITS/XISF writing, light calibration, star measurement, registration, outlier-rejecting integration, drizzle and frame weighting, as a library that runs on the caller's rayon pool. It is built in layers, and each layer is a cargo feature, so a consumer pays only for what it uses.

| Feature | Layer | Adds | Pulls in |
| ---- | ---- | ---- | ---- |
| none | base | geometry, resampling, the FITS/XISF writer core, the platform memory probe, light-calibration parameter types | nothing heavy |
| `engine` | 2 | integration, light calibration, cosmetic correction | `rustafits` |
| `stack` | 3 | registration, measurement, rejection, drizzle, frame weighting and the algorithm config | `engine`, `solvemyastro`, `postcard` |
| `pipeline` | 4 | the working-folder stages a command line drives (inputs, config levels, versioned folder formats, `master` to `stack`, `run`) | `stack`, `toml`, `glob` |
| `cli` | 5 | the `starloom` command, with its logs and history | `pipeline`, `clap`, `tracing-subscriber` |
| `ts` | — | ts-rs type derives (on by default) | `ts-rs` |
| `test-support` | — | test hooks and fixtures that dependent crates' tests read; never in default | nothing |

`default = ["stack", "ts"]`. The crate is standalone: its manifest has no `[workspace]` table. Licence: Apache-2.0.

## The `starloom` command

The `cli` feature builds a program, `starloom`, that calibrates, measures, registers, normalizes and stacks FITS and XISF frames from raw files, with no catalog and no application around it. On the same frames, masters and config it writes the same bytes as Athenaeum's own run, stage by stage (checked on two real sets, one mono and one OSC). It is a do-it-yourself tool: each stage is a command, and you decide what to run and what to clean between them.

### Install

```
cargo install --git https://github.com/eg013ra1n/starloom --features cli --locked
```

There are no prebuilt binaries yet. Build with the release profile (`lto = "thin"`, `codegen-units = 4`, set in `Cargo.toml`): it is Athenaeum's release profile, and equality with Athenaeum is claimed for that profile only. `starloom --version` and `starloom <command> --help` describe everything below in full.

### Commands

| Command | What it does | Main products |
| ---- | ---- | ---- |
| `facts FILES` | prints the header facts of frames, one JSON object per line with `--json` | stdout only |
| `master <bias\|dark\|flat> FILES` | integrates raw frames into a master (`--precal` or `--synthetic-bias` for flats, `--recipe`, `-o`) | `masters/<kind>.fits` |
| `calibrate [LIGHTS]` | calibrates raw lights with `--dark`, `--flat`, `--bias` (a master file, or raw frames, a folder or a glob) or a `--plan` file; hot pixels, debayer of an OSC light | `calibrated/c_<stem>.fits` |
| `measure [FRAMES]` | measures the stars of calibrated frames and weighs them (`--gain`, `--psf`, `--stars`) | `measure/measure.json`, `measure.csv`, `measure/fits/*.athf` |
| `register [FRAMES]` | registers every frame onto one reference (`--reference <file>\|auto`, `--write-registered`), then reweighs | `register/transforms.json`, `register/weights.json` |
| `normalize [FRAMES]` | builds the local normalization of every frame, when the config asks for it | `ln/normalize.json`, `ln/*.athln` |
| `stack [FRAMES]` | integrates the frames with rejection, weights and local normalization (`--rejection`, `--weights`, `--drizzle`, `--format`, `-o`) | `stack/master.fits`, rejection maps, drizzle, `stack/weights.json` |
| `run [LIGHTS]` | every stage in one go: calibrate, measure, register, normalize, stack | all of the above |
| `config show` | prints the effective config (preset, file, `--set`) as JSON | stdout only |

Global flags: `-w/--work DIR` (the working folder, `starloom-work` by default), `--preset default|fast-preview|maximum-quality`, `--config FILE` (`.toml` or `.json`; an Athenaeum stacking config works, and the keys the command does not read are named on stderr), `--set path=value` (one config value by dotted camelCase path, JSON when it parses as JSON, repeatable; `--set integration.minWeight=0.01`), `--threads N` (the one thread pool; default rayon's, which honours `RAYON_NUM_THREADS`), `--memory SIZE` and `-v`/`-vv` (more on stderr). The short flags (`--psf`, `--weights`, `--rejection`, `--drizzle`, `--no-debayer`, `--no-hot-pixels`) are each one `--set`. A glob must be quoted (`--dark 'darks/*.fits'`), so the command expands it the same way on every platform; an unquoted one is expanded by the shell, and `calibrate` would take the other files for lights (it refuses a light whose `IMAGETYP` is Dark, Flat or Bias, and says so).

### The working folder

```
starloom-work/
  starloom.json          the manifest: what each command made, with its config hash and versions, and the history
  masters/               master bias, dark, flat
  calibrated/            c_<stem>.fits (c_<stem>_d.fits after a debayer; c_<stem>.fits stays beside it with --keep-mosaic)
  measure/               measure.json, measure.csv, fits/<stem>.p<plane>.athf, stars/ (with --stars)
  register/              transforms.json, weights.json, and the resampled frames with --write-registered
  ln/                    normalize.json, <stem>.athln
  stack/                 master.fits (or .xisf), master_rejlow.fits, master_rejhigh.fits, master_drizzle<s>x.fits, weights.json
  logs/                  <UTC stamp>-<command>.jsonl
  .tmp/                  scratch; the next writing command clears it
```

A stage builds each product beside its destination and renames it into place, and writes its manifest entry last, so an interrupted stage leaves no entry and nothing named like a finished product. One command runs in a working folder at a time: nothing locks it.

**Frames from the previous stage.** A command given no frames takes them from the manifest, in their recorded order: `measure` takes `calibrate`'s products, and `register`, `normalize` and `stack` take `measure`'s frames. Frames given on the command line win. `--measure FILE`, `--transforms FILE` and `--ln DIR` name the products to read instead of the ones in the working folder; a product named so is read together with its own folder (`--measure X` reads the star fits from the `fits/` folder beside X, and takes X's frames when given none; `--ln DIR` reads the `.athln` files from DIR), so a stage can run in a fresh working folder.

**Manual mode.** The stages are not linked. The command never compares one stage's products with another's and never refuses a product as stale: if you change the config after `measure`, you decide whether to run `measure` again, and what to delete. A missing input (a frame with no measurement, transform or `.athln`) is an ordinary error that names the frame and the command that makes it. The manifest is a journal for you to read, and nothing checks it.

**Logs and history.** Every command except `facts` and `config show` writes `logs/<UTC stamp>-<command>.jsonl`: debug-level JSON from `starloom`, `astroimage` and `solvemyastro`, whatever `-v` says, including every `error:`, `note:` and `warning:` line with its reason and the effective config. It also appends a `history` row to the manifest (when it fails or is refused too) and ends with one line on stderr: `starloom <command>: <N> frames (<k> excluded) in <elapsed>; wrote <main product>; log <file>`. stdout carries products only. Logs are never pruned; they are your files. Exit codes: 0 done, 1 an internal or I/O failure, 2 an input you must change (a usage error, which leaves no log and no history row; a bad `--set` or config; a frame with no transform; a calibration that does not fit; fewer than three frames). There is no Ctrl-C handler: a killed command leaves only `.tmp/` and part files.

### Formats and versions

Each file the working folder holds is tagged with its version and refused when another version wrote it: the manifest `starloom.json` (`MANIFEST_VERSION` 1), `measure.json` (`MEASURE_REPORT_VERSION` 1), `transforms.json` (`TRANSFORMS_VERSION` 1), `weights.json` (`WEIGHTS_VERSION` 1) and `ln/normalize.json` (`NORMALIZE_REPORT_VERSION` 1), besides the `.athf` star fits (`FITS_ARTIFACT_VERSION` 3) and the `.athln` grids (`LN_GRID_VERSION` 1) that Athenaeum writes too. A non-finite float is stored as `null`. Every manifest entry and report records all the version constants of the build by their own names (`PIPELINE_VERSION` and the stage versions below), for you to read.

### The cards

Every FITS or XISF file the command writes carries `SLM_VER` (the crate version), `SLM_CFG` (the hash of the whole algorithm config) and `SLM_PLV` (`PIPELINE_VERSION`), and never an `ATH_*` card. Masters carry `SLM_REC` (the recipe, as Athenaeum's `ATH_REJ` states it) and a flat `SLM_FNRM`, `SLM_FNR`, `SLM_FNG`, `SLM_FNB`; a master flat's constants are read back from those cards, and a flat without them is recomputed from its pixels. Calibrated lights carry an `SLM_` twin of every card Athenaeum writes there (`SLM_CDRK`, `SLM_CFLT` and `SLM_CBIA` hold the master's file name; `SLM_CSCL` the scale divisor; `SLM_LCV`; `SLM_CHPX` when the hot-pixel pass ran; `SLM_CDBM` after a debayer). Master lights carry `SLM_REF` (the registration reference) and `SLM_MSV`, `SLM_MV`, `SLM_ALV`, `SLM_LNV`, `SLM_INV` (the measure-stage, rustafits measure, align, LN-model and integrate versions); drizzle outputs add `SLM_DRZ`, `SLM_DRZP`, `SLM_DRZK` and `SLM_DZV`. The gain a measurement uses is `--gain`, else `EGAIN` times `SLM_CSCL`, else none (the SNR then has no shot term, and one warning names the frames and `--gain`).

### Compressed raw input

`facts`, `master`, `calibrate` and `run` take `.fz`, `.fits.gz` (`.fit.gz`, `.fts.gz`) and FITS with extension HDUs, besides plain FITS and XISF, by their logical header. The products are plain FITS, or XISF with `--format xisf`. Reading a gzip file's header inflates the whole file, which takes seconds per frame, and a gzip light is inflated more than once. `measure`, `register`, `normalize` and `stack` read plain FITS only, as Athenaeum's run does, and refuse an XISF or compressed frame with the reason.

### The star profile

`--psf auto|moffat4|moffat=<beta>` sets `measurement.psfModel` (and, with `--set normalization.local.psfModel=...`, local normalization's). `auto` measures in two passes with one group beta; `moffat4` in one pass at beta 4; `moffat=<beta>` in one pass at your beta (1.5 to 100). `moffat=4` measures bit for bit like `moffat4` but has its own hash.

### Where it differs from Athenaeum

On the pixel path the command mirrors Athenaeum's run step for step. It differs on purpose here:

- one reference for every frame: a calibrated FITS you name (`--reference FILE`, inside or outside the frames) or `auto`, the best-weighted frame at register time. There is no two-pass reference pick, no plate-solve seed, no project grid and no `registration.geometry`; a frame Athenaeum rescues with the plate-solve seed can fail here, and a one-plane reference outside the frames gets fresh star detection instead of `measure`'s star fits;
- with fewer than three included frames `measure` and `register` warn and go on, where Athenaeum skips the group; `stack` still refuses to integrate fewer than three;
- an exact tie when the normalization anchor is picked breaks on the frame index, not the catalog id; so does an exact weight tie in local normalization's seed-calibration frame selection (another sample can mean another scale);
- a Bayer drizzle needs the calibrated mosaics (`calibrate --keep-mosaic`) and is refused without them; no drizzle coverage map is written;
- frames are ordered by DATE-OBS, then by input position; frame paths are compared as given, case-sensitively, also on Windows;
- `run --reference FILE` needs the file to exist before calibration: give a calibrated frame, or use `auto`;
- no cache, no staleness checks, no artifact reuse, no crop, no normalized-frame files, and no user-configured header rules.

### Examples

One `run` over a plan file. `plan.toml` (relative paths and globs resolve against the plan's own folder; a group's `dark`, `flat` and `bias` are one master file, or a folder or glob of raw frames, or a list of those; `flat_precal` is a master file or a number of DN):

```toml
[[group]]
lights = "lights/*.fits"
dark = "darks"
flat = "flats/*.fits"
flat_precal = 500.0

[[group]]
lights = ["night2/lights"]
dark = "masters/dark_300s.fits"
flat = "masters/flat_L.fits"
```

```
starloom run --plan plan.toml --preset maximum-quality --psf moffat=3.5 -w work
```

The same thing as a chain, where every command after `calibrate` is given no frames and finds them in the working folder:

```
starloom master dark 'darks/*.fits' -w work
starloom master flat 'flats/*.fits' --synthetic-bias 500 -w work
starloom calibrate lights --dark work/masters/dark.fits --flat work/masters/flat.fits -w work
starloom measure -w work
starloom register --reference auto -w work
starloom normalize -w work
starloom stack --drizzle -w work
```

## Repository

- **CI** (`.github/workflows/ci.yml`): Linux runs `cargo test --locked` for every layer (none, `engine`, `stack`, `ts`, `cli`, which holds `pipeline`, and the default `stack` + `ts`), prints the stage golden rows of its platform and installs the binary and starts it; macOS and Windows run `cargo test --locked --features cli` and print their rows. `Cargo.lock` is committed, and the stage goldens are valid for the rustafits it locks.
- **Releases** come from `main` only: `CHANGELOG.md` gets a section covering every commit since the last tag, `Cargo.toml` the version, and the tag `vX.Y.Z` must equal it. The release workflow checks that and takes the notes from the changelog.
- **crates.io:** `cargo publish` fails until solvemyastro is published there, because crates.io refuses a git dependency; the CI job that tries it is allowed to fail.
- **Athenaeum** builds starloom in its own workspace and patches rustafits and solvemyastro back to its own submodules, so it builds one copy of each.
- **Citations.** Comments cite "spec §N", "math reference §N", ruling IDs (`R-…`), plans and tasks, and modules such as `stacking::run`: those are Athenaeum's design documents, its core and its history, where this crate was written. The two most cited: [the extraction spec](https://github.com/eg013ra1n/athenaeum/blob/ff25561c/docs/superpowers/specs/2026-11-22-starloom-extraction-design.md) and [the stacking math reference](https://github.com/eg013ra1n/athenaeum/blob/ff25561c/docs/superpowers/research/2026-09-08-stacking-math-reference.md).

## Versions

A stage version is a public constant. A change to a stage's output bits bumps that stage's version in the same change. The stage versions below are the ones the dependent application's tripwire test asserts (the working-folder format versions are not pinned there; the `pipeline` golden row is keyed on `PIPELINE_VERSION`); the ones marked "hash" enter a stage hash or a file header, so a bump invalidates the matching cached results.

| Constant | Value | Layer | Enters |
| ---- | ---- | ---- | ---- |
| `ALIGN_VERSION` (`stacking/register/align.rs`) | 1 | `stack` | the registration stage hash and the two-pass key (hash) |
| `LN_MODEL_VERSION` (`stacking/ln/mod.rs`) | 4 | `stack` | the normalization stage hash and the seeds-calibration hash (hash) |
| `SNR_GAIN_VERSION` (`stacking/integrate.rs`) | 2 | `stack` | the group statistics and the run summary |
| `DRIZZLE_KERNEL_VERSION` (`stacking/config.rs`) | 2 | `stack` | the run-level config hash (hash) |
| `PASS1_ARTIFACT_VERSION` (`stacking/pass1_artifact.rs`) | 1 | `stack` | the pass-1 artifact file header and its stage hash (hash) |
| `LN_GRID_VERSION` (`stacking/ln/grid.rs`, the `.athln` `VERSION`) | 1 | `stack` | the `.athln` sidecar header |
| `FITS_ARTIFACT_VERSION` (`stacking/fits_artifact.rs`, the `.athf` version) | 3 | `stack` | the `.athf` file header and the measurement stage hash (`fitsArtifactVersion`) (hash) |
| `LIGHT_CAL_VERSION` (`calibration_library/mod.rs`) | 1 | base | nothing inside starloom; the caller marks its calibrated lights with it |
| `MEASURE_STAGE_VERSION` (`stacking/measure.rs`) | 1 | `stack` | the measurement, registration and normalization hashes when not 1 (hash); a bump also raises `PASS1_ARTIFACT_VERSION` with it, because the pass-1 header checks only rustafits' version. In Athenaeum the pass-1 artifact already misses by key without that raise (its stage hash, `pass1_hash_for`, hashes the measurement subtree, which carries the fold); the raise is for a reader of the header alone (a caller with no hash). A bump also re-points two asserts in the application's `stacking/config.rs` tests ("the shipped helper is the versioned one…" and "the shipped measurement helper is the fits-versioned one…") |
| `INTEGRATE_VERSION` (`stacking/integrate.rs`) | 1 | `stack` | signal: enters no hash |
| `ENGINE_VERSION` (`integration/engine.rs`) | 1 | `engine` | signal: enters no hash |
| `PIPELINE_VERSION` (`pipeline/mod.rs`) | 1 | `pipeline` | `SLM_PLV` on every file the command writes, and the manifest |
| `MANIFEST_VERSION`, `MEASURE_REPORT_VERSION`, `TRANSFORMS_VERSION`, `WEIGHTS_VERSION`, `NORMALIZE_REPORT_VERSION` (`pipeline/formats/`) | 1 each | `pipeline` | the tag of the working-folder file each names; a file by another version is refused |
| `MEASURE_VERSION` | 8 | lives in `rustafits` | star measurement; unchanged by starloom |

A change that moves a stage's bytes bumps the stage's version in the same change; the stage goldens below are what notices when it does not.

## Stage goldens

`tests/stage_goldens.rs` runs every stage (measure, pass-1 artifact, register, warp, normalization model, `.athln` and `.athf` formats, integrate, SNR gain, drizzle, master bias, master flat, light calibration, cosmetic correction, and with the `pipeline` feature the whole `run` chain with its drizzle) on a fixed synthetic input on a 2-thread pool and checksums its output. A change that moves a stage's bytes while its version stays is red, and so is a version bump with no new row.

- **A row is keyed by** the stage name, the platform and the versions it was recorded at: the stage's own version first, then the upstream versions its hash folds (for example the registration row also carries `MEASURE_STAGE_VERSION` and rustafits' `MEASURE_VERSION`).
- **The platform key** is `<arch>-<os>`, because `exp` and `powf` round one ulp differently from one system math library to the next. Float rows exist for `aarch64-macos`; the macOS version and the rustafits revision they were recorded with are in the comment above them. A platform with no rows (`x86_64-linux` and `x86_64-windows` until CI has printed theirs) checks determinism only (two runs agree) and prints the rows it would record. The one float-free stage, the `.athln` format, has the platform `"*"` and holds everywhere. A red float row when neither starloom nor rustafits changed points at the system math library first.
- **The build profile.** The float rows hold for the dev profile at opt-level 2, Athenaeum's (its root `Cargo.toml`) and this crate's own standalone (`[profile.dev]` here). At opt-level 0 one measured FWHM rounds one ulp apart and the measure and pass-1 rows go red; never re-record rows from an opt-level 0 build.
- **How a row is recorded.** Nothing writes the table. Run `cargo test -p starloom --features cli --test stage_goldens -- --nocapture` (the `cli` feature also computes the `pipeline` row), copy the literal the failure message prints into the table by hand, and do it in the same change as the version bump. A change that moves no number records its row with the reason in its commit message. The rows of another platform come from its CI log: the `Stage golden rows of this platform` step of the Linux `cli`, macOS and Windows jobs runs the test with `--nocapture`, so the literals of a platform with no rows are in that step's output (copy them from there, never from another machine).
- **The rustafits note.** A row is valid for the rustafits revision `Cargo.lock` locks (Athenaeum's workspace lock inside Athenaeum): the measure, pass-1, register, normalization and integrate stages run rustafits' star fit. A red row right after a rustafits bump means rustafits moved numbers; if its `MEASURE_VERSION` rose, record new rows, and if not, stop the bump.
- **The pass-1 hook.** To make a stored first pass byte-identical between runs, the test-only `FramePass1::fix_stamp_and_zero_timings` zeroes every wall-clock field it stores; the two inside each rustafits `PlanePrep` go through rustafits' own test hook `PlanePrep::zero_timings` (rustafits 1.9.1, feature `test-support`, which starloom's `test-support` forwards). Fields the artifact does not store stay as measured.

## Writers

Every writer family that reaches the XISF `CreatorApplication` has plain forms with defaults, which write the default creator `starloom <version>`, and one `_full` form that takes every parameter, the creator string last (`write_image_f32_full`, `write_xisf_f32_full`, `write_xisf_f32_to_full`).
