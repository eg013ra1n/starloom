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
//! The code is decided by the kind of the error, not by where it arises:
//!
//! - 0: done;
//! - 1: the machine or the code failed: an I/O error, a thread pool that would not build, a stage that panicked (it is
//!   caught, logged, summarised and recorded like any failure), a frame that would not decode;
//! - 2: the input is at fault and the user must change it: a usage error (clap's own, which leaves no log and no
//!   history row, the working folder being unknown before the command line is read), a bad `--set` or config, a bound
//!   (`--psf moffat=1`), a file named on the command line that is missing or not a frame, a product a stage needs that
//!   is not there, no frames to work on, a frame with no transform, a calibration that does not fit, a light that is
//!   really a dark (a shell expanded a glob), fewer than three frames to stack. The pipeline marks these with a typed
//!   [`Refusal`]; every other error it returns is a failure.
//!
//! A refusal and a failure both keep their log file (which holds the `error:` line and why) and append a history row.
//!
//! There is no Ctrl-C handler this cycle: a killed command leaves only `<work>/.tmp/` and part files, which the next
//! writing command clears, and no log close or history row.
//!
//! The stages are not linked (P2-R24): a command reads the products it is given (`--measure`, `--transforms`, `--ln`)
//! or else the ones in the working folder, and never judges them stale. A product named on the command line is read
//! with its own folder: Measure's `.athf` star fits from the `fits/` folder beside `--measure`'s report, the `.athln`
//! grids from `--ln`'s folder, so a stage run in a fresh working folder on products made in another gets the same
//! result.

pub mod args;
pub mod log;
pub mod out;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use clap::Parser;

use crate::fits_writer::OutputFormat;
use crate::integration::combine::IntegrationRecipe;
use crate::pipeline::formats::{
    FrameStatus, MeasureReport, NormalizeReport, NormalizeStatus, TransformsReport, WeightSettings, WeightsReport,
    MEASURE_REPORT_FILE, NORMALIZE_FILE, TRANSFORMS_FILE,
};
use crate::pipeline::{
    is_refusal, Refusal, append_history, begin_writing, build_master, calibrate, calibrated_dir, config_hash, effective_config,
    expand_calibration_frames, expand_inputs, frames_or_upstream, ln_dir, measure, measure_dir, normalize, read_plan, register,
    register_dir, run, stack, stack_dir, CalGroup, CalSource, Ctx, FlatPrecalSource, HistoryRow, InputKind, Manifest, MasterKind,
    ReferenceChoice, RegisterReport, Stage,
};
use crate::stacking::config::AlgorithmConfig;

use args::{parse_memory, parse_set, CalFlags, CalibrateArgs, Cli, Command, ConfigCommand, FactsArgs, Global, MasterArgs,
    MeasureArgs, NormalizeArgs, RegisterArgs, RunArgs, Set, StackArgs};
use out::{error_line, note_line, product_line, stderr_line, warning_line, StderrProgress};

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

impl Fail {
    /// An error of the pipeline or the machine: a refusal when the pipeline typed it as one, else a failure.
    fn of(error: anyhow::Error) -> Fail {
        if is_refusal(&error) {
            Fail::Refused(error)
        } else {
            Fail::Failed(error)
        }
    }
}

/// Classifies an error as a refusal (exit 2) or a failure (exit 1).
trait Classify<T> {
    /// Always a refusal: for a call whose every error is the input's (a pattern that matched nothing, a bad `--set`).
    fn refuse(self) -> Result<T, Fail>;
    /// By kind: a refusal when the error is a typed [`Refusal`], else a failure (an I/O or internal error).
    fn fail(self) -> Result<T, Fail>;
}

impl<T, E: Into<anyhow::Error>> Classify<T> for Result<T, E> {
    fn refuse(self) -> Result<T, Fail> {
        self.map_err(|e| Fail::Refused(e.into()))
    }

    fn fail(self) -> Result<T, Fail> {
        self.map_err(|e| Fail::of(e.into()))
    }
}

/// A refusal that says `message`.
fn refusal(message: impl Into<String>) -> Fail {
    Fail::Refused(Refusal::new(message).into())
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

/// The message of a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string())
}

/// Runs `run`, turning a panic (also one of a pool worker, which rayon hands to the caller) into a failure: the panic
/// is logged with its place by a hook (so the default hook does not print it a second time), and the command goes on to
/// its summary line and its history row.
fn guarded<T>(run: impl FnOnce() -> Result<T, Fail>) -> Result<T, Fail> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        let place = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        tracing::error!(place = %place, text = %panic_message(info.payload()), "stage panicked");
    }));
    let result = std::panic::catch_unwind(AssertUnwindSafe(run));
    std::panic::set_hook(previous);
    match result {
        Ok(r) => r,
        Err(payload) => Err(Fail::Failed(anyhow!("internal error: a stage panicked: {}", panic_message(&*payload)))),
    }
}

/// A command that only reads: no work folder, log file or history row.
fn read_only(cli: &Cli) -> i32 {
    log::init(cli.verbose(), None);
    let result = match &cli.command {
        Command::Facts(a) => facts(a),
        Command::Config(ConfigCommand::Show(_)) => config_show(cli),
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
    logged_with(cli, args, |work, hash| execute(cli, work, hash))
}

/// [`logged`] with the stage to run given: the log, the stage (its panic caught), the summary line and the history row.
fn logged_with(cli: &Cli, args: &[OsString], run: impl FnOnce(&Path, &mut String) -> Result<Outcome, Fail>) -> i32 {
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
    log::init(cli.verbose(), Some(file));
    tracing::info!(command = name, version = env!("CARGO_PKG_VERSION"), "command started");

    let mut hash = String::new();
    let result = guarded(|| run(&work, &mut hash));
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
            error_line(f.error());
            (f.code(), format!("starloom {name}: {} in {elapsed}; log {}", f.verb(), log_file.display()))
        }
    };
    tracing::info!(command = name, exit_code = code, summary = %summary, "command finished");
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

/// The user's `--set`s, parsed: the ones before the subcommand, then the ones after it.
fn user_sets(cli: &Cli) -> Result<Vec<Set>> {
    cli.sets().iter().map(|s| parse_set(s)).collect()
}

/// The effective config of a command: its short flags' `--set`s, then the user's. Names the keys a file did not read.
fn build_config(cli: &Cli, mut sets: Vec<Set>) -> Result<AlgorithmConfig> {
    sets.extend(user_sets(cli)?);
    let global = &cli.global;
    let (cfg, ignored) = effective_config(global.preset.map(Into::into), global.config.as_deref(), &sets)?;
    if !ignored.is_empty() {
        let file = global.config.as_deref().map(|p| p.display().to_string()).unwrap_or_default();
        note_line(&format!("config {file}: keys not read: {}", ignored.join(", ")));
    }
    Ok(cfg)
}

/// What `--threads` and `--memory` ask for: the pool (the one starloom builds) and the memory budget.
///
/// Without `--threads` the pool has rayon's own count (every core, or `RAYON_NUM_THREADS`). Without `--memory` the
/// budget is the machine's RAM; when the machine does not say, 4 GiB, with a warning. A bad `--memory` is a refusal; a
/// pool that will not build is a failure.
fn pool_and_memory(global: &Global) -> Result<(Arc<rayon::ThreadPool>, u64), Fail> {
    let memory = match &global.memory {
        Some(raw) => parse_memory(raw).refuse()?,
        None => crate::sys::total_ram_bytes().unwrap_or_else(|| {
            warning_line("the machine's RAM is unknown; planning for 4 GiB of working memory (give --memory to say)");
            4 << 30
        }),
    };
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = global.threads.filter(|n| *n > 0) {
        builder = builder.num_threads(n);
    }
    let pool = builder.build().context("cannot build the thread pool").fail()?;
    tracing::debug!(threads = pool.current_num_threads(), memory_bytes = memory, "thread pool built");
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
    let cfg = build_config(cli, sugar).refuse()?;
    *hash = config_hash(&cfg);
    tracing::debug!(config = %serde_json::to_string(&cfg).map_err(anyhow::Error::from).fail()?, "effective config");
    let (pool, memory) = pool_and_memory(&cli.global)?;
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

/// The frames of a stage: the ones on the command line, else the ones of the measurement report the command was given
/// (`--measure`), else the previous stage's from the manifest (P2-R20).
fn frames_for(env: &Env, stage: Stage, given: &[OsString], explicit: Option<&MeasureReport>) -> Result<Vec<PathBuf>, Fail> {
    let named = if given.is_empty() { Vec::new() } else { expand_inputs(given, InputKind::Calibrated).refuse()? };
    if let (true, Some(report)) = (named.is_empty(), explicit) {
        return Ok(report.frames.iter().map(|f| f.file.clone()).collect());
    }
    let manifest = Manifest::load(&env.work).fail()?;
    frames_or_upstream(&manifest, stage, named).refuse()
}

/// The frames of a stage and its measurement report: the report named by `--measure`, else the working folder's.
fn frames_and_measure(
    env: &Env,
    stage: Stage,
    given: &[OsString],
    measure: Option<&Path>,
) -> Result<(Vec<PathBuf>, MeasureReport), Fail> {
    let explicit = measure.map(|p| load_measure(env, Some(p))).transpose()?;
    let frames = frames_for(env, stage, given, explicit.as_ref())?;
    let measured = match explicit {
        Some(m) => m,
        None => load_measure(env, None)?,
    };
    Ok((frames, measured))
}

/// `path` made absolute; a failure is the machine's, not the input's.
fn absolute(path: &Path) -> Result<PathBuf, Fail> {
    std::path::absolute(path).with_context(|| format!("cannot make {} absolute", path.display())).fail()
}

/// A calibration source from `--dark`, `--flat` or `--bias`: one existing file is a master, anything else is raw frames
/// to integrate into one.
fn cal_source(values: &[OsString], what: &str) -> Result<Option<CalSource>, Fail> {
    if values.is_empty() {
        return Ok(None);
    }
    if let [one] = values {
        let p = Path::new(one);
        if p.is_file() {
            return Ok(Some(CalSource::Master(absolute(p)?)));
        }
    }
    Ok(Some(CalSource::Raw(expand_calibration_frames(values).with_context(|| format!("--{what}")).refuse()?)))
}

/// The calibration groups of `calibrate` and `run`: a plan's, or one group from the lights and the flags.
fn cal_groups(lights: &[OsString], cal: &CalFlags, plan: Option<&Path>) -> Result<Vec<CalGroup>, Fail> {
    if let Some(plan) = plan {
        if !lights.is_empty() || !cal.dark.is_empty() || !cal.flat.is_empty() || !cal.bias.is_empty() {
            return Err(refusal("--plan replaces LIGHTS and --dark, --flat and --bias; give one or the other"));
        }
        return Ok(read_plan(plan).refuse()?);
    }
    if lights.is_empty() {
        return Err(refusal("no lights given; give LIGHTS or --plan"));
    }
    Ok(vec![CalGroup {
        lights: expand_inputs(lights, InputKind::Raw).refuse()?,
        dark: cal_source(&cal.dark, "dark")?,
        flat: cal_source(&cal.flat, "flat")?,
        bias: cal_source(&cal.bias, "bias")?,
        flat_precal: None,
    }])
}

/// `--reference`: `auto`, or a file that exists.
fn reference_choice(raw: &str) -> Result<ReferenceChoice, Fail> {
    if raw == "auto" {
        return Ok(ReferenceChoice::Auto);
    }
    let path = Path::new(raw);
    if !path.is_file() {
        return Err(refusal(format!("--reference {raw} is not a file; give a calibrated FITS file or `auto`")));
    }
    Ok(ReferenceChoice::File(absolute(path)?))
}

/// The measurement report: the one named, else `<work>/measure/measure.json`. A report that is not one, or that another
/// starloom made, is a refusal; one that cannot be read is a failure.
fn load_measure(env: &Env, explicit: Option<&Path>) -> Result<MeasureReport, Fail> {
    let path = explicit.map_or_else(|| measure_dir(&env.work).join(MEASURE_REPORT_FILE), Path::to_path_buf);
    if !path.is_file() {
        return Err(refusal(format!("no measurement report at {}; give --measure or run `starloom measure`", path.display())));
    }
    MeasureReport::read(&path).with_context(|| format!("cannot read {}", path.display())).fail()
}

/// The transforms: the report named, else `<work>/register/transforms.json`. The weights a later stage uses are never
/// read from a file (P2-R21): they are made again from the measurements under the command's config.
fn load_register(env: &Env, explicit: Option<&Path>) -> Result<RegisterReport, Fail> {
    let path = explicit.map_or_else(|| register_dir(&env.work).join(TRANSFORMS_FILE), Path::to_path_buf);
    if !path.is_file() {
        return Err(refusal(format!("no transforms at {}; give --transforms or run `starloom register`", path.display())));
    }
    let transforms = TransformsReport::read(&path).with_context(|| format!("cannot read {}", path.display())).fail()?;
    let weights = WeightsReport::new(WeightSettings {
        weight_mode: env.cfg.measurement.weight_mode,
        formula: env.cfg.measurement.formula.clone(),
        keyword: env.cfg.measurement.keyword.clone(),
        selection: env.cfg.selection,
    });
    Ok(RegisterReport { transforms, weights, registered: Vec::new() })
}

/// The normalization report of `dir` (`--ln`), its `.athln` rows pointing into `dir`: `normalize` recorded the paths
/// where it wrote them, and a folder that was copied or moved keeps the files, not those paths.
fn load_normalize(dir: &Path) -> Result<NormalizeReport, Fail> {
    let dir = absolute(dir)?;
    let path = dir.join(NORMALIZE_FILE);
    if !path.is_file() {
        return Err(refusal(format!("no {NORMALIZE_FILE} in {}; give --ln or run `starloom normalize`", dir.display())));
    }
    let mut report = NormalizeReport::read(&path).with_context(|| format!("cannot read {}", path.display())).fail()?;
    for row in &mut report.frames {
        if let Some(name) = row.athln.as_ref().and_then(|p| p.file_name()) {
            row.athln = Some(dir.join(name));
        }
    }
    Ok(report)
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
fn config_show(cli: &Cli) -> Result<(), Fail> {
    let cfg = build_config(cli, Vec::new()).refuse()?;
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
            Some(FlatPrecalSource::Master(absolute(p)?))
        }
        (None, Some(dn)) => Some(FlatPrecalSource::SyntheticBias(dn)),
        (None, None) => None,
    };
    let recipe: Option<IntegrationRecipe> = match &a.recipe {
        Some(json) => Some(serde_json::from_str(json).context("--recipe is not a valid integration recipe").refuse()?),
        None => None,
    };
    let out = match &a.out {
        Some(p) => absolute(p)?,
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
    let groups = cal_groups(&a.lights, &a.cal, a.plan.as_deref())?;
    let lights: usize = groups.iter().map(|g| g.lights.len()).sum();
    let mut opts = env.cfg.calibration.clone();
    opts.keep_mosaic = a.keep_mosaic;
    let dir = calibrated_dir(&env.work);
    let report = calibrate(&env.ctx(), &env.cfg, &groups, &opts, &dir, a.format.into()).fail()?;
    Ok(Outcome { frames: report.frames.len(), excluded: lights.saturating_sub(report.frames.len()), product: Some(dir) })
}

fn cmd_measure(env: &Env, a: &MeasureArgs) -> Result<Outcome, Fail> {
    let frames = frames_for(env, Stage::Measure, &a.frames, None)?;
    let report = measure(&env.ctx(), &env.cfg, &frames, a.gain, a.stars).fail()?;
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.frames.iter().filter(|f| !f.included).count(),
        product: Some(measure_dir(&env.work).join(MEASURE_REPORT_FILE)),
    })
}

fn cmd_register(env: &Env, a: &RegisterArgs) -> Result<Outcome, Fail> {
    let (frames, measured) = frames_and_measure(env, Stage::Register, &a.frames, a.measure.as_deref())?;
    let reference = reference_choice(&a.reference)?;
    let report = register(&env.ctx(), &frames, &measured, &reference, &env.cfg, a.write_registered).fail()?;
    Ok(Outcome {
        frames: report.transforms.frames.len(),
        excluded: report.transforms.frames.iter().filter(|f| !f.ok).count(),
        product: Some(register_dir(&env.work).join(TRANSFORMS_FILE)),
    })
}

fn cmd_normalize(env: &Env, a: &NormalizeArgs) -> Result<Outcome, Fail> {
    let (frames, measured) = frames_and_measure(env, Stage::Normalize, &a.frames, a.measure.as_deref())?;
    let registered = load_register(env, a.transforms.as_deref())?;
    let report = normalize(&env.ctx(), &frames, &measured, &registered, &env.cfg).fail()?;
    if let (NormalizeStatus::NotRequested | NormalizeStatus::Off, Some(note)) = (report.status, &report.note) {
        note_line(note);
    }
    let product = (report.status != NormalizeStatus::NotRequested).then(|| ln_dir(&env.work).join(NORMALIZE_FILE));
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.frames.iter().filter(|f| f.status == FrameStatus::Excluded).count(),
        product,
    })
}

fn cmd_stack(env: &Env, a: &StackArgs) -> Result<Outcome, Fail> {
    let (frames, measured) = frames_and_measure(env, Stage::Stack, &a.frames, a.measure.as_deref())?;
    let registered = load_register(env, a.transforms.as_deref())?;
    let normalized: Option<NormalizeReport> = a.ln.as_deref().map(load_normalize).transpose()?;
    let out = match &a.out {
        Some(p) => absolute(p)?,
        None => stack_dir(&env.work),
    };
    std::fs::create_dir_all(&out).with_context(|| format!("cannot create {}", out.display())).fail()?;
    let report = stack(&env.ctx(), &env.cfg, &frames, &measured, &registered, normalized.as_ref(), &out, a.format.into()).fail()?;
    if let Some(note) = &report.reweighed {
        note_line(note);
    }
    Ok(Outcome {
        frames: report.frames.len(),
        excluded: report.weights.frames.iter().filter(|f| !f.included).count(),
        product: Some(report.master),
    })
}

fn cmd_run(env: &Env, a: &RunArgs) -> Result<Outcome, Fail> {
    let groups = cal_groups(&a.lights, &a.cal, a.plan.as_deref())?;
    let reference = reference_choice(&a.reference)?;
    let out = match &a.out {
        Some(p) => absolute(p)?,
        None => stack_dir(&env.work),
    };
    std::fs::create_dir_all(&out).with_context(|| format!("cannot create {}", out.display())).fail()?;
    let report = run(&env.ctx(), &groups, &env.cfg, &reference, &out, a.format.into()).fail()?;
    if let Some(note) = &report.stack.reweighed {
        note_line(note);
    }
    Ok(Outcome {
        frames: report.stack.frames.len(),
        excluded: report.stack.weights.frames.iter().filter(|f| !f.included).count(),
        product: Some(report.stack.master),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_becomes_a_failure_with_its_message() {
        let result: Result<(), Fail> = guarded(|| panic!("the stage broke"));
        let fail = result.unwrap_err();
        assert_eq!(fail.code(), 1);
        assert!(format!("{:#}", fail.error()).contains("the stage broke"), "{:#}", fail.error());
        assert_eq!(guarded(|| Ok::<_, Fail>(7)).unwrap(), 7);
    }

    #[test]
    fn a_panicking_stage_still_leaves_its_summary_its_history_row_and_exit_1() {
        let tmp = tempfile::tempdir().unwrap();
        let work = tmp.path().join("work");
        let args: Vec<OsString> = ["starloom", "-w", work.to_str().unwrap(), "measure", "a.fits"].map(OsString::from).to_vec();
        let cli = Cli::try_parse_from(&args).unwrap();
        let code = logged_with(&cli, &args, |_, _| -> Result<Outcome, Fail> { panic!("boom in a stage") });
        assert_eq!(code, 1);
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(work.join("starloom.json")).unwrap()).unwrap();
        let rows = manifest["history"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["exitCode"], 1);
        assert!(rows[0]["summary"].as_str().unwrap().starts_with("starloom measure: failed in"), "{}", rows[0]["summary"]);
        // The log says why (the subscriber is the first one this test process installs).
        let log_dir = work.join("logs");
        let log = std::fs::read_to_string(std::fs::read_dir(log_dir).unwrap().next().unwrap().unwrap().path()).unwrap();
        assert!(log.contains("boom in a stage"), "{log}");
    }
}
