//! The `starloom` command: the working-folder pipeline behind a command line.
//!
//! This module is the one place in starloom that writes to stdout and stderr (P2-R13), through [`out`]. stdout carries
//! command products only (`facts`, `config show`); progress, warnings, notes and the summary line go to stderr.
//!
//! # What a command does
//!
//! Every command except `facts` and `config show` (which only read, and make no work folder, log or history row):
//! 1. makes the working folder and opens `<work>/logs/<stamp>-<command>.jsonl` ([`log`]);
//! 2. builds the effective config: preset, `--config` file, the short flags' `--set`s, then the user's `--set`s;
//! 3. builds the one rayon pool (`--threads`, the only pool starloom builds outside tests) and clears `<work>/.tmp/`;
//! 4. runs the stage on the frames it was given or, with none, on the previous stage's (P2-R20);
//! 5. prints the summary line on stderr and appends a `history` row to the manifest, also when the stage failed or was
//!    refused (P2-R19).
//!
//! # Exit codes
//!
//! - 0: done;
//! - 1: a stage failed (an I/O error, an unreadable frame, a frame with no transform);
//! - 2: the command refused its inputs before running a stage: a usage error, a bad `--set` or config, a bound
//!   (`--psf moffat=1`), a file named on the command line that is missing or not a frame, a product a stage needs that
//!   is not there, and no frames to work on.
//!
//! There is no Ctrl-C handler this cycle: a killed command leaves only `<work>/.tmp/` and part files, which the next
//! writing command clears, and no log close or history row.
//!
//! The stages are not linked (P2-R24): a command reads the products it is given (`--measure`, `--transforms`, `--ln`)
//! or else the ones in the working folder, and never judges them stale.

pub mod args;
pub mod log;
pub mod out;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use clap::Parser;

use crate::fits_writer::OutputFormat;
use crate::integration::combine::IntegrationRecipe;
use crate::pipeline::formats::{
    FrameStatus, MeasureReport, NormalizeReport, NormalizeStatus, TransformsReport, WeightSettings, WeightsReport,
    MEASURE_REPORT_FILE, NORMALIZE_FILE, TRANSFORMS_FILE,
};
use crate::pipeline::{
    append_history, begin_writing, build_master, calibrate, calibrated_dir, config_hash, effective_config,
    expand_calibration_frames, expand_inputs, frames_or_upstream, ln_dir, measure, measure_dir, normalize, read_plan, register,
    register_dir, run, stack, stack_dir, CalGroup, CalSource, Ctx, FlatPrecalSource, HistoryRow, InputKind, Manifest, MasterKind,
    ReferenceChoice, RegisterReport, Stage,
};
use crate::stacking::config::AlgorithmConfig;

use args::{parse_memory, parse_set, CalFlags, CalibrateArgs, Cli, Command, ConfigCommand, FactsArgs, Global, MasterArgs,
    MeasureArgs, NormalizeArgs, RegisterArgs, RunArgs, Set, StackArgs};
use out::{product_line, stderr_line, StderrProgress};

/// How a command ended when it did not finish: refused (exit code 2) or failed (exit code 1).
#[derive(Debug)]
enum Fail {
    Refused(anyhow::Error),
    Failed(anyhow::Error),
}

impl Fail {
    fn code(&self) -> i32 {
        match self {
            Fail::Refused(_) => 2,
            Fail::Failed(_) => 1,
        }
    }

    fn error(&self) -> &anyhow::Error {
        match self {
            Fail::Refused(e) | Fail::Failed(e) => e,
        }
    }

    fn verb(&self) -> &'static str {
        match self {
            Fail::Refused(_) => "refused",
            Fail::Failed(_) => "failed",
        }
    }
}

/// Classifies an error as a refusal (exit 2) or a failure (exit 1).
trait Classify<T> {
    fn refuse(self) -> Result<T, Fail>;
    fn fail(self) -> Result<T, Fail>;
}

impl<T, E: Into<anyhow::Error>> Classify<T> for Result<T, E> {
    fn refuse(self) -> Result<T, Fail> {
        self.map_err(|e| Fail::Refused(e.into()))
    }

    fn fail(self) -> Result<T, Fail> {
        self.map_err(|e| Fail::Failed(e.into()))
    }
}

/// What a finished command made, for the summary line.
struct Outcome {
    frames: usize,
    excluded: usize,
    /// The main product, if the command wrote one.
    product: Option<PathBuf>,
}

/// The command's world: its working folder, effective config and the context every stage runs with.
struct Env {
    work: PathBuf,
    cfg: AlgorithmConfig,
    pool: Arc<rayon::ThreadPool>,
    cancel: AtomicBool,
    progress: StderrProgress,
    memory: u64,
}

impl Env {
    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            pool: self.pool.clone(),
            cancel: &self.cancel,
            progress: &self.progress,
            memory_budget_bytes: self.memory,
            work_dir: &self.work,
        }
    }
}

/// Runs the command line of the process and returns its exit code.
pub fn main() -> i32 {
    run_args(std::env::args_os().collect())
}

/// Runs the command line `args` (the program name first) and returns its exit code. A process runs it once: it installs
/// the process's `tracing` subscriber.
pub fn run_args(args: Vec<OsString>) -> i32 {
    let cli = match Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(e) => {
            let code = e.exit_code();
            let _ = e.print();
            return code;
        }
    };
    match &cli.command {
        Command::Facts(_) | Command::Config(_) => read_only(&cli),
        _ => logged(&cli, &args),
    }
}

/// A command that only reads: no work folder, log file or history row.
fn read_only(cli: &Cli) -> i32 {
    log::init(cli.global.verbose, None);
    let result = match &cli.command {
        Command::Facts(a) => facts(a),
        Command::Config(ConfigCommand::Show) => config_show(&cli.global),
        _ => unreachable!("only the read-only commands come here"),
    };
    match result {
        Ok(()) => 0,
        Err(f) => {
            stderr_line(&format!("starloom {}: error: {:#}", cli.command.name(), f.error()));
            f.code()
        }
    }
}

/// A command that writes: the log, the stage, the summary line and the history row.
fn logged(cli: &Cli, args: &[OsString]) -> i32 {
    let name = cli.command.name();
    let started = Utc::now();
    let clock = Instant::now();
    let work = match std::path::absolute(&cli.global.work) {
        Ok(w) => w,
        Err(e) => {
            stderr_line(&format!("starloom {name}: error: cannot make {} absolute: {e}", cli.global.work.display()));
            return 1;
        }
    };
    let (file, log_file) = match std::fs::create_dir_all(&work)
        .with_context(|| format!("cannot create the working folder {}", work.display()))
        .and_then(|()| log::open_log(&work, name, started))
    {
        Ok(opened) => opened,
        Err(e) => {
            stderr_line(&format!("starloom {name}: error: {e:#}"));
            return 1;
        }
    };
    log::init(cli.global.verbose, Some(file));
    tracing::info!(command = name, version = env!("CARGO_PKG_VERSION"), "command started");

    let mut hash = String::new();
    let result = execute(cli, &work, &mut hash);
    let elapsed = elapsed_text(clock.elapsed());
    let (mut code, summary) = match &result {
        Ok(o) => (
            0,
            format!(
                "starloom {name}: {} frames ({} excluded) in {elapsed}; wrote {}; log {}",
                o.frames,
                o.excluded,
                o.product.as_ref().map_or_else(|| "nothing".to_string(), |p| p.display().to_string()),
                log_file.display()
            ),
        ),
        Err(f) => {
            stderr_line(&format!("error: {:#}", f.error()));
            (f.code(), format!("starloom {name}: {} in {elapsed}; log {}", f.verb(), log_file.display()))
        }
    };
    tracing::info!(command = name, exit_code = code, "command finished");
    stderr_line(&summary);
    let row = HistoryRow {
        command: name.to_string(),
        args: args.iter().skip(1).map(|a| a.to_string_lossy().into_owned()).collect(),
        started,
        finished: Utc::now(),
        exit_code: code,
        starloom_version: env!("CARGO_PKG_VERSION").to_string(),
        config_hash: hash,
        log_file,
        summary,
    };
    if let Err(e) = append_history(&work, row) {
        stderr_line(&format!("error: cannot record the command in the history of {}: {e:#}", work.display()));
        if code == 0 {
            code = 1;
        }
    }
    code
}

/// `12.3s`, or `2m05s` from a minute on.
fn elapsed_text(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let whole = d.as_secs();
        format!("{}m{:02}s", whole / 60, whole % 60)
    }
}

/// The user's `--set`s, parsed.
fn user_sets(global: &Global) -> Result<Vec<Set>> {
    global.set.iter().map(|s| parse_set(s)).collect()
}

/// The effective config of a command: its short flags' `--set`s, then the user's. Names the keys a file did not read.
fn build_config(global: &Global, mut sets: Vec<Set>) -> Result<AlgorithmConfig> {
    sets.extend(user_sets(global)?);
    let (cfg, ignored) = effective_config(global.preset.map(Into::into), global.config.as_deref(), &sets)?;
    if !ignored.is_empty() {
        let file = global.config.as_deref().map(|p| p.display().to_string()).unwrap_or_default();
        stderr_line(&format!("note: config {file}: keys not read: {}", ignored.join(", ")));
    }
    Ok(cfg)
}

/// What `--threads` and `--memory` ask for: the pool (the one starloom builds) and the memory budget.
fn pool_and_memory(global: &Global) -> Result<(Arc<rayon::ThreadPool>, u64)> {
    let memory = match &global.memory {
        Some(raw) => parse_memory(raw)?,
        None => crate::sys::total_ram_bytes().unwrap_or(4 << 30),
    };
    let threads = match global.threads {
        Some(n) if n > 0 => n,
        _ => std::thread::available_parallelism().map_or(1, usize::from),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("cannot build the thread pool")?;
    tracing::debug!(threads, memory_bytes = memory, "thread pool built");
    Ok((Arc::new(pool), memory))
}

/// Runs the stage a writing command names.
fn execute(cli: &Cli, work: &Path, hash: &mut String) -> Result<Outcome, Fail> {
    let sugar: Vec<Set> = match &cli.command {
        Command::Calibrate(a) => a.sugar(),
        Command::Measure(a) => a.sugar().refuse()?,
        Command::Stack(a) => a.sugar(),
        Command::Run(a) => a.sugar().refuse()?,
        _ => Vec::new(),
    };
    let cfg = build_config(&cli.global, sugar).refuse()?;
    *hash = config_hash(&cfg);
    let (pool, memory) = pool_and_memory(&cli.global).refuse()?;
    begin_writing(work).fail()?;
    let env = Env { work: work.to_path_buf(), cfg, pool, cancel: AtomicBool::new(false), progress: StderrProgress::new(), memory };
    match &cli.command {
        Command::Master(a) => cmd_master(&env, a),
        Command::Calibrate(a) => cmd_calibrate(&env, a),
        Command::Measure(a) => cmd_measure(&env, a),
        Command::Register(a) => cmd_register(&env, a),
        Command::Normalize(a) => cmd_normalize(&env, a),
        Command::Stack(a) => cmd_stack(&env, a),
        Command::Run(a) => cmd_run(&env, a),
        Command::Facts(_) | Command::Config(_) => unreachable!("read-only commands do not come here"),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------------------------

/// The frames of a stage: the ones on the command line, else the previous stage's from the manifest (P2-R20).
fn frames_for(env: &Env, stage: Stage, given: &[OsString]) -> Result<Vec<PathBuf>, Fail> {
    let named = if given.is_empty() { Vec::new() } else { expand_inputs(given, InputKind::Calibrated).refuse()? };
    let manifest = Manifest::load(&env.work).refuse()?;
    frames_or_upstream(&manifest, stage, named).refuse()
}

/// A calibration source from `--dark`, `--flat` or `--bias`: one existing file is a master, anything else is raw frames
/// to integrate into one.
fn cal_source(values: &[OsString], what: &str) -> Result<Option<CalSource>> {
    if values.is_empty() {
        return Ok(None);
    }
    if let [one] = values {
        let p = Path::new(one);
        if p.is_file() {
            return Ok(Some(CalSource::Master(std::path::absolute(p).with_context(|| format!("cannot make {} absolute", p.display()))?)));
        }
    }
    Ok(Some(CalSource::Raw(expand_calibration_frames(values).with_context(|| format!("--{what}"))?)))
}

/// The calibration groups of `calibrate` and `run`: a plan's, or one group from the lights and the flags.
fn cal_groups(lights: &[OsString], cal: &CalFlags, plan: Option<&Path>) -> Result<Vec<CalGroup>> {
    if let Some(plan) = plan {
        if !lights.is_empty() || !cal.dark.is_empty() || !cal.flat.is_empty() || !cal.bias.is_empty() {
            bail!("--plan replaces LIGHTS and --dark, --flat and --bias; give one or the other");
        }
        return read_plan(plan);
    }
    if lights.is_empty() {
        bail!("no lights given; give LIGHTS or --plan");
    }
    Ok(vec![CalGroup {
        lights: expand_inputs(lights, InputKind::Raw)?,
        dark: cal_source(&cal.dark, "dark")?,
        flat: cal_source(&cal.flat, "flat")?,
        bias: cal_source(&cal.bias, "bias")?,
        flat_precal: None,
    }])
}

/// `--reference`: `auto`, or a file that exists.
fn reference_choice(raw: &str) -> Result<ReferenceChoice> {
    if raw == "auto" {
        return Ok(ReferenceChoice::Auto);
    }
    let path = Path::new(raw);
    if !path.is_file() {
        bail!("--reference {raw} is not a file; give a calibrated FITS file or `auto`");
    }
    Ok(ReferenceChoice::File(std::path::absolute(path).with_context(|| format!("cannot make {raw} absolute"))?))
}

/// The measurement report: the one named, else `<work>/measure/measure.json`.
fn load_measure(env: &Env, explicit: Option<&Path>) -> Result<MeasureReport, Fail> {
    let path = explicit.map_or_else(|| measure_dir(&env.work).join(MEASURE_REPORT_FILE), Path::to_path_buf);
    if !path.is_file() {
        return Err(Fail::Refused(anyhow!("no measurement report at {}; give --measure or run `starloom measure`", path.display())));
    }
    MeasureReport::read(&path).with_context(|| format!("cannot read {}", path.display())).refuse()
}

/// The transforms: the report named, else `<work>/register/transforms.json`. The weights a later stage uses are never
/// read from a file (P2-R21): they are made again from the measurements under the command's config.
fn load_register(env: &Env, explicit: Option<&Path>) -> Result<RegisterReport, Fail> {
    let path = explicit.map_or_else(|| register_dir(&env.work).join(TRANSFORMS_FILE), Path::to_path_buf);
    if !path.is_file() {
        return Err(Fail::Refused(anyhow!("no transforms at {}; give --transforms or run `starloom register`", path.display())));
    }
    let transforms = TransformsReport::read(&path).with_context(|| format!("cannot read {}", path.display())).refuse()?;
    let weights = WeightsReport::new(WeightSettings {
        weight_mode: env.cfg.measurement.weight_mode,
        formula: env.cfg.measurement.formula,
        keyword: env.cfg.measurement.keyword.clone(),
        selection: env.cfg.selection,
    });
    Ok(RegisterReport { transforms, weights, registered: Vec::new() })
}

// ---------------------------------------------------------------------------------------------------------------
// The read-only commands
// ---------------------------------------------------------------------------------------------------------------

/// `facts`: the header facts of every file, as text or one JSON object per line.
fn facts(args: &FactsArgs) -> Result<(), Fail> {
    let files = expand_calibration_frames(&args.files).refuse()?;
    let mut unreadable = 0usize;
    for file in &files {
        let read = astroimage::header::read_frame_facts(file).with_context(|| format!("cannot read the header of {}", file.display()));
        let facts = match read {
            Ok(f) => f,
            Err(e) => {
                unreadable += 1;
                stderr_line(&format!("error: {e:#}"));
                continue;
            }
        };
        let mut value = serde_json::to_value(&facts).map_err(anyhow::Error::from).fail()?;
        let object = value.as_object_mut().ok_or_else(|| anyhow!("frame facts are not an object")).fail()?;
        if args.json {
            object.insert("file".to_string(), serde_json::Value::String(file.display().to_string()));
            product_line(&serde_json::to_string(&value).map_err(anyhow::Error::from).fail()?);
        } else {
            product_line(&file.display().to_string());
            for (key, v) in object.iter().filter(|(_, v)| !v.is_null()) {
                product_line(&format!("  {key}: {v}"));
            }
        }
    }
    if unreadable > 0 {
        return Err(Fail::Refused(anyhow!("{unreadable} of {} files could not be read", files.len())));
    }
    Ok(())
}

/// `config show`: the effective config as JSON.
fn config_show(global: &Global) -> Result<(), Fail> {
    let cfg = build_config(global, Vec::new()).refuse()?;
    product_line(&serde_json::to_string_pretty(&cfg).map_err(anyhow::Error::from).fail()?);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// The stages
// ---------------------------------------------------------------------------------------------------------------

fn cmd_master(env: &Env, a: &MasterArgs) -> Result<Outcome, Fail> {
    let kind: MasterKind = a.kind.into();
    let frames = expand_calibration_frames(&a.files).refuse()?;
    if kind != MasterKind::Flat && (a.precal.is_some() || a.synthetic_bias.is_some()) {
        return Err(Fail::Refused(anyhow!("--precal and --synthetic-bias apply to flats only, not to {} frames", kind.name())));
    }
    let precal = match (&a.precal, a.synthetic_bias) {
        (Some(p), _) => {
            if !p.is_file() {
                return Err(Fail::Refused(anyhow!("--precal {} is not a file", p.display())));
            }
            Some(FlatPrecalSource::Master(std::path::absolute(p).map_err(anyhow::Error::from).refuse()?))
        }
        (None, Some(dn)) => Some(FlatPrecalSource::SyntheticBias(dn)),
        (None, None) => None,
    };
    let recipe: Option<IntegrationRecipe> = match &a.recipe {
        Some(json) => Some(serde_json::from_str(json).context("--recipe is not a valid integration recipe").refuse()?),
        None => None,
    };
    let out = match &a.out {
        Some(p) => std::path::absolute(p).map_err(anyhow::Error::from).refuse()?,
        None => env.work.join("masters").join(format!("{}.fits", kind.name())),
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display())).fail()?;
    }
    let format = OutputFormat::from_path(&out);
    let report = build_master(&env.ctx(), &env.cfg, kind, &frames, precal.as_ref(), recipe, &out, format).fail()?;
    Ok(Outcome { frames: report.frames, excluded: 0, product: Some(report.path) })
}

fn cmd_calibrate(env: &Env, a: &CalibrateArgs) -> Result<Outcome, Fail> {
    let groups = cal_groups(&a.lights, &a.cal, a.plan.as_deref()).refuse()?;
    let lights: usize = groups.iter().map(|g| g.lights.len()).sum();
    let mut opts = env.cfg.calibration.clone();
    opts.keep_mosaic = a.keep_mosaic;
    let dir = calibrated_dir(&env.work);
    let report = calibrate(&env.ctx(), &env.cfg, &groups, &opts, &dir, a.format.into()).fail()?;
    Ok(Outcome { frames: report.frames.len(), excluded: lights.saturating_sub(report.frames.len()), product: Some(dir) })
}

fn cmd_measure(env: &Env, a: &MeasureArgs) -> Result<Outcome, Fail> {
    let frames = frames_for(env, Stage::Measure, &a.frames)?;
    let report = measure(&env.ctx(), &env.cfg, &frames, a.gain, a.stars).fail()?;
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.frames.iter().filter(|f| !f.included).count(),
        product: Some(measure_dir(&env.work).join(MEASURE_REPORT_FILE)),
    })
}

fn cmd_register(env: &Env, a: &RegisterArgs) -> Result<Outcome, Fail> {
    let frames = frames_for(env, Stage::Register, &a.frames)?;
    let measured = load_measure(env, a.measure.as_deref())?;
    let reference = reference_choice(&a.reference).refuse()?;
    let report = register(&env.ctx(), &frames, &measured, &reference, &env.cfg, a.write_registered).fail()?;
    Ok(Outcome {
        frames: report.transforms.frames.len(),
        excluded: report.transforms.frames.iter().filter(|f| !f.ok).count(),
        product: Some(register_dir(&env.work).join(TRANSFORMS_FILE)),
    })
}

fn cmd_normalize(env: &Env, a: &NormalizeArgs) -> Result<Outcome, Fail> {
    let frames = frames_for(env, Stage::Normalize, &a.frames)?;
    let measured = load_measure(env, a.measure.as_deref())?;
    let registered = load_register(env, a.transforms.as_deref())?;
    let report = normalize(&env.ctx(), &frames, &measured, &registered, &env.cfg).fail()?;
    if let (NormalizeStatus::NotRequested | NormalizeStatus::Off, Some(note)) = (report.status, &report.note) {
        stderr_line(&format!("note: {note}"));
    }
    let product = (report.status != NormalizeStatus::NotRequested).then(|| ln_dir(&env.work).join(NORMALIZE_FILE));
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.frames.iter().filter(|f| f.status == FrameStatus::Excluded).count(),
        product,
    })
}

fn cmd_stack(env: &Env, a: &StackArgs) -> Result<Outcome, Fail> {
    let frames = frames_for(env, Stage::Stack, &a.frames)?;
    let measured = load_measure(env, a.measure.as_deref())?;
    let registered = load_register(env, a.transforms.as_deref())?;
    let normalized: Option<NormalizeReport> = match &a.ln {
        Some(dir) => {
            let path = dir.join(NORMALIZE_FILE);
            if !path.is_file() {
                return Err(Fail::Refused(anyhow!("no {NORMALIZE_FILE} in {}; give --ln or run `starloom normalize`", dir.display())));
            }
            Some(NormalizeReport::read(&path).with_context(|| format!("cannot read {}", path.display())).refuse()?)
        }
        None => None,
    };
    let out = match &a.out {
        Some(p) => std::path::absolute(p).map_err(anyhow::Error::from).refuse()?,
        None => stack_dir(&env.work),
    };
    std::fs::create_dir_all(&out).with_context(|| format!("cannot create {}", out.display())).fail()?;
    let report = stack(&env.ctx(), &env.cfg, &frames, &measured, &registered, normalized.as_ref(), &out, a.format.into()).fail()?;
    if let Some(note) = &report.reweighed {
        stderr_line(&format!("note: {note}"));
    }
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.weights.frames.iter().filter(|f| !f.included).count(),
        product: Some(report.master),
    })
}

fn cmd_run(env: &Env, a: &RunArgs) -> Result<Outcome, Fail> {
    let groups = cal_groups(&a.lights, &a.cal, a.plan.as_deref()).refuse()?;
    let reference = reference_choice(&a.reference).refuse()?;
    let out = match &a.out {
        Some(p) => std::path::absolute(p).map_err(anyhow::Error::from).refuse()?,
        None => stack_dir(&env.work),
    };
    std::fs::create_dir_all(&out).with_context(|| format!("cannot create {}", out.display())).fail()?;
    let report = run(&env.ctx(), &groups, &env.cfg, &reference, &out, a.format.into()).fail()?;
    if let Some(note) = &report.stack.reweighed {
        stderr_line(&format!("note: {note}"));
    }
    Ok(Outcome {
        frames: report.stack.frames.len(),
        excluded: report.stack.weights.frames.iter().filter(|f| !f.included).count(),
        product: Some(report.stack.master),
    })
}
