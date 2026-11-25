//! The command line's grammar: one struct per command, the global flags, and the short flags that are sugar for a
//! `--set`.
//!
//! Every short flag that changes the algorithm is one `--set path=value` ([`sugar`](MeasureArgs::sugar) on each
//! command), and the command's log file holds the effective config at debug level, so what a flag did is on record. A
//! flag that is not a config value (a file to read, a place to write, `--gain`) is not sugar.
//!
//! `--set` and `-v` may be given before the subcommand, after it, or both ([`Levels`]): clap does not merge a global
//! flag's values across the two levels, so they are two flags and [`Cli::sets`] / [`Cli::verbose`] join them.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::fits_writer::OutputFormat;
use crate::pipeline::MasterKind;
use crate::stacking::config::StackingPreset;

/// A `(dotted camelCase config path, value)` pair, as `--set` spells it.
pub type Set = (String, String);

/// The `starloom` command line.
#[derive(Debug, Parser)]
#[command(
    name = "starloom",
    version,
    about = "Calibrate, measure, register, normalize and stack astrophotography frames",
    long_about = "Calibrate, measure, register, normalize and stack astrophotography frames.\n\n\
        Each stage is a command that leaves its products in a working folder (-w); a command given no frames takes \
        them from the previous stage's entry there. `starloom run` does every stage in one go. Products go to stdout \
        only for `facts` and `config show`; progress, warnings and the one summary line go to stderr.",
    arg_required_else_help = true
)]
pub struct Cli {
    /// The flags every command takes.
    #[command(flatten)]
    pub global: Global,
    /// `--set` and `-v` given before the subcommand.
    #[command(flatten)]
    pub early: Levels,
    /// The command.
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Every `--set`, in the order given: the ones before the subcommand, then the ones after it.
    pub fn sets(&self) -> Vec<String> {
        self.early.set.iter().chain(self.command.late().map(|l| &l.set).into_iter().flatten()).cloned().collect()
    }

    /// How many `-v` were given, before and after the subcommand together.
    pub fn verbose(&self) -> u8 {
        self.early.verbose.saturating_add(self.command.late().map_or(0, |l| l.verbose))
    }
}

/// The flags that are accepted both before and after the subcommand, and joined: `starloom --set a=1 stack --set b=2`
/// sets both.
#[derive(Debug, Clone, Default, Args)]
pub struct Levels {
    /// Sets one config value over the file, by dotted camelCase path: `--set integration.minWeight=0.01`. A value is
    /// JSON when it parses as JSON (`0.2`, `true`, `{"moffat":8}`), else a bare word (`moffat4`). Repeatable.
    #[arg(long = "set", value_name = "PATH=VALUE")]
    pub set: Vec<String>,
    /// More on stderr: `-v` adds the info level, `-vv` the debug level. The log file always holds debug.
    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// A built-in config preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PresetArg {
    /// The shipped default.
    Default,
    /// A fast preview: bilinear interpolation, sigma clipping, no local normalization.
    FastPreview,
    /// Polynomial distortion, rejection maps, local normalization and 2x drizzle.
    MaximumQuality,
}

impl From<PresetArg> for StackingPreset {
    fn from(p: PresetArg) -> StackingPreset {
        match p {
            PresetArg::Default => StackingPreset::Default,
            PresetArg::FastPreview => StackingPreset::FastPreview,
            PresetArg::MaximumQuality => StackingPreset::MaximumQuality,
        }
    }
}

/// The output container of a calibrated frame or a master.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// 32-bit float FITS.
    Fits,
    /// Monolithic XISF 1.0.
    Xisf,
}

impl From<FormatArg> for OutputFormat {
    fn from(f: FormatArg) -> OutputFormat {
        match f {
            FormatArg::Fits => OutputFormat::Fits,
            FormatArg::Xisf => OutputFormat::Xisf,
        }
    }
}

/// The flags every command takes.
#[derive(Debug, Args)]
pub struct Global {
    /// The working folder: the manifest, the stages' products, the logs and `.tmp/` live under it.
    #[arg(short = 'w', long, global = true, value_name = "DIR", default_value = "starloom-work")]
    pub work: PathBuf,
    /// A built-in config preset, the base of the effective config.
    #[arg(long, global = true, value_enum, value_name = "NAME")]
    pub preset: Option<PresetArg>,
    /// A config file (`.toml` or `.json`) over the preset; an Athenaeum stacking config works, and the keys that are
    /// not read are named on stderr.
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// The number of worker threads (the one thread pool the command builds); default: rayon's own (every core, or
    /// `RAYON_NUM_THREADS`).
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
    /// The working memory a stage may plan for: bytes, or a size such as `16GB` (units are 1024-based); default: the
    /// machine's total RAM, or 4 GiB (with a warning) when the machine does not say.
    #[arg(long, global = true, value_name = "SIZE")]
    pub memory: Option<String>,
}

/// The commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Prints the header facts of frames (one JSON object per line with `--json`).
    Facts(FactsArgs),
    /// Integrates raw bias, dark or flat frames into a master.
    Master(MasterArgs),
    /// Calibrates raw lights with master darks, flats and biases.
    Calibrate(CalibrateArgs),
    /// Measures the stars of calibrated frames and weighs them.
    Measure(MeasureArgs),
    /// Registers every frame onto one reference.
    Register(RegisterArgs),
    /// Builds the local normalization of every frame.
    Normalize(NormalizeArgs),
    /// Integrates the frames into a master light (and the drizzle, when asked).
    Stack(StackArgs),
    /// Every stage in one go: calibrate, measure, register, normalize, stack.
    Run(RunArgs),
    /// Config commands.
    #[command(subcommand)]
    Config(ConfigCommand),
}

impl Command {
    /// The `--set` and `-v` given after the subcommand.
    pub fn late(&self) -> Option<&Levels> {
        Some(match self {
            Command::Facts(a) => &a.late,
            Command::Master(a) => &a.late,
            Command::Calibrate(a) => &a.late,
            Command::Measure(a) => &a.late,
            Command::Register(a) => &a.late,
            Command::Normalize(a) => &a.late,
            Command::Stack(a) => &a.late,
            Command::Run(a) => &a.late,
            Command::Config(ConfigCommand::Show(a)) => &a.late,
        })
    }

    /// The command's name, as the log file, the history and the summary line spell it.
    pub fn name(&self) -> &'static str {
        match self {
            Command::Facts(_) => "facts",
            Command::Master(_) => "master",
            Command::Calibrate(_) => "calibrate",
            Command::Measure(_) => "measure",
            Command::Register(_) => "register",
            Command::Normalize(_) => "normalize",
            Command::Stack(_) => "stack",
            Command::Run(_) => "run",
            Command::Config(_) => "config",
        }
    }
}

/// `config` subcommands.
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Prints the effective config (preset, file, `--set`) as JSON.
    Show(ShowArgs),
}

/// `starloom config show`.
#[derive(Debug, Args)]
pub struct ShowArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
}

/// `starloom facts`.
#[derive(Debug, Args)]
pub struct FactsArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Frames, folders or globs (compressed raw frames included).
    #[arg(required = true, value_name = "FILES")]
    pub files: Vec<OsString>,
    /// One JSON object per file, one per line.
    #[arg(long)]
    pub json: bool,
}

/// The kind of master to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MasterKindArg {
    /// Bias frames.
    Bias,
    /// Dark frames.
    Dark,
    /// Flat frames.
    Flat,
}

impl From<MasterKindArg> for MasterKind {
    fn from(k: MasterKindArg) -> MasterKind {
        match k {
            MasterKindArg::Bias => MasterKind::Bias,
            MasterKindArg::Dark => MasterKind::Dark,
            MasterKindArg::Flat => MasterKind::Flat,
        }
    }
}

/// `starloom master`.
#[derive(Debug, Args)]
pub struct MasterArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// What the frames are.
    #[arg(value_enum)]
    pub kind: MasterKindArg,
    /// Raw frames, folders or globs.
    #[arg(required = true, value_name = "FILES")]
    pub files: Vec<OsString>,
    /// Flats only: a master (a dark flat or a bias) subtracted from every flat first.
    #[arg(long, value_name = "FILE")]
    pub precal: Option<PathBuf>,
    /// Flats only: a constant, in DN, subtracted from every flat first.
    #[arg(long, value_name = "DN", conflicts_with = "precal")]
    pub synthetic_bias: Option<f32>,
    /// The integration recipe as JSON (`{"combination":"average","rejection":{"method":"winsorized_sigma",...}}`);
    /// default: the automatic one for the kind and the frame count.
    #[arg(long, value_name = "JSON")]
    pub recipe: Option<String>,
    /// The master file (`.fits` or `.xisf`); default `<work>/masters/<kind>.fits`.
    #[arg(short = 'o', long, value_name = "FILE")]
    pub out: Option<PathBuf>,
}

/// The calibration frames of `calibrate` and `run`: one master file, or raw frames to integrate into one.
#[derive(Debug, Args, Default)]
pub struct CalFlags {
    /// The dark: a master file, or raw frames, a folder or a glob (quote a glob, `--dark 'darks/*.fits'`: the shell
    /// would expand it and `calibrate` would take the other files for lights).
    #[arg(long, value_name = "MASTER|FRAMES")]
    pub dark: Vec<OsString>,
    /// The flat: a master file, or raw frames, a folder or a glob (quote a glob, `--flat 'flats/*.fits'`).
    #[arg(long, value_name = "MASTER|FRAMES")]
    pub flat: Vec<OsString>,
    /// The bias, used only when there is no dark: a master file, or raw frames, a folder or a glob (quote a glob).
    #[arg(long, value_name = "MASTER|FRAMES")]
    pub bias: Vec<OsString>,
}

/// `starloom calibrate`.
#[derive(Debug, Args)]
pub struct CalibrateArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Raw lights, folders or globs (compressed frames included).
    #[arg(value_name = "LIGHTS")]
    pub lights: Vec<OsString>,
    /// The calibration frames.
    #[command(flatten)]
    pub cal: CalFlags,
    /// A calibration plan (`.toml` or `.json`) with `[[group]]`s, in place of LIGHTS and the flags above.
    #[arg(long, value_name = "FILE")]
    pub plan: Option<PathBuf>,
    /// Keep the calibrated CFA mosaic beside each debayered frame (what a Bayer drizzle reads).
    #[arg(long)]
    pub keep_mosaic: bool,
    /// Leave an OSC light as a mosaic (`--set calibration.debayerOsc=false`).
    #[arg(long)]
    pub no_debayer: bool,
    /// Skip the hot-pixel replacement (`--set calibration.hotPixelCorrection=false`).
    #[arg(long)]
    pub no_hot_pixels: bool,
    /// The container of the calibrated frames. Later stages read FITS only.
    #[arg(long, value_enum, default_value = "fits")]
    pub format: FormatArg,
}

impl CalibrateArgs {
    /// The `--set`s the short flags stand for.
    pub fn sugar(&self) -> Vec<Set> {
        let mut sets = Vec::new();
        if self.no_debayer {
            sets.push(("calibration.debayerOsc".to_string(), "false".to_string()));
        }
        if self.no_hot_pixels {
            sets.push(("calibration.hotPixelCorrection".to_string(), "false".to_string()));
        }
        sets
    }
}

/// `starloom measure`.
#[derive(Debug, Args)]
pub struct MeasureArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Calibrated frames, folders or globs; default: the products of `calibrate`.
    #[arg(value_name = "FRAMES")]
    pub frames: Vec<OsString>,
    /// The camera gain in e-/ADU, when the headers do not say.
    #[arg(long, value_name = "E/ADU")]
    pub gain: Option<f64>,
    /// The star profile: `auto`, `moffat4` or `moffat=<beta>` (`--set measurement.psfModel=...`).
    #[arg(long, value_name = "MODEL")]
    pub psf: Option<String>,
    /// Also write the stars: `measure/stars/<frame>.p<plane>.csv`, one row per measured star.
    #[arg(long)]
    pub stars: bool,
}

impl MeasureArgs {
    /// The `--set`s the short flags stand for.
    pub fn sugar(&self) -> Result<Vec<Set>> {
        psf_sugar(self.psf.as_deref())
    }
}

/// `--psf` as a `--set`, spelled as the config wire spells the model (P2-R18): `auto`, `moffat4`, `{"moffat":<beta>}`.
/// The bounds on beta are the config's: a value outside them is refused by the config's own decoder.
pub fn psf_sugar(psf: Option<&str>) -> Result<Vec<Set>> {
    let Some(raw) = psf else {
        return Ok(Vec::new());
    };
    let value = match raw {
        "auto" | "moffat4" => raw.to_string(),
        _ => match raw.strip_prefix("moffat=") {
            Some(beta) => {
                let beta: f64 = beta
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--psf moffat=<beta>: `{beta}` is not a number"))?;
                if !beta.is_finite() {
                    bail!("--psf moffat=<beta>: beta must be a finite number, got {beta}");
                }
                serde_json::json!({ "moffat": beta }).to_string()
            }
            None => bail!("--psf must be `auto`, `moffat4` or `moffat=<beta>`, got `{raw}`"),
        },
    };
    Ok(vec![("measurement.psfModel".to_string(), value)])
}

/// `starloom register`.
#[derive(Debug, Args)]
pub struct RegisterArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Calibrated frames, folders or globs; default: the frames `measure` measured.
    #[arg(value_name = "FRAMES")]
    pub frames: Vec<OsString>,
    /// What every frame is registered onto: a calibrated FITS file, or `auto` (the best-weighted frame).
    #[arg(long, value_name = "FILE|auto", default_value = "auto")]
    pub reference: String,
    /// The measurement report; default `<work>/measure/measure.json`. The star fits are read from the `fits/` folder
    /// beside it, and with no FRAMES the command takes the frames it lists.
    #[arg(long, value_name = "FILE")]
    pub measure: Option<PathBuf>,
    /// Also write every frame resampled into the reference's grid.
    #[arg(long)]
    pub write_registered: bool,
}

/// `starloom normalize`.
#[derive(Debug, Args)]
pub struct NormalizeArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Calibrated frames, folders or globs; default: the frames `measure` measured.
    #[arg(value_name = "FRAMES")]
    pub frames: Vec<OsString>,
    /// The measurement report; default `<work>/measure/measure.json`. The star fits are read from the `fits/` folder
    /// beside it, and with no FRAMES the command takes the frames it lists.
    #[arg(long, value_name = "FILE")]
    pub measure: Option<PathBuf>,
    /// The transforms report; default `<work>/register/transforms.json`.
    #[arg(long, value_name = "FILE")]
    pub transforms: Option<PathBuf>,
}

/// `starloom stack`.
#[derive(Debug, Args)]
pub struct StackArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Calibrated frames, folders or globs; default: the frames `measure` measured.
    #[arg(value_name = "FRAMES")]
    pub frames: Vec<OsString>,
    /// The measurement report; default `<work>/measure/measure.json`. The star fits are read from the `fits/` folder
    /// beside it, and with no FRAMES the command takes the frames it lists.
    #[arg(long, value_name = "FILE")]
    pub measure: Option<PathBuf>,
    /// The transforms report; default `<work>/register/transforms.json`.
    #[arg(long, value_name = "FILE")]
    pub transforms: Option<PathBuf>,
    /// The folder holding `normalize.json` and the `.athln` files; default `<work>/ln`. The grids are read from this
    /// folder, wherever `normalize` first wrote them.
    #[arg(long, value_name = "DIR")]
    pub ln: Option<PathBuf>,
    /// Also drizzle (`--set drizzle.enabled=true`).
    #[arg(long)]
    pub drizzle: bool,
    /// The rejection: a method name (`auto`, `none`) or a JSON object (`{"method":"sigmaClip","sigmaLow":3,"sigmaHigh":3}`),
    /// as `--set integration.rejection=...`.
    #[arg(long, value_name = "JSON|METHOD")]
    pub rejection: Option<String>,
    /// The weight mode, re-weighing from the stored measurements: `psfSignalWeight`, `psfSnr`, `noise`, `formula`,
    /// `exposure`, `keyword` or `none` (`--set measurement.weightMode=...`).
    #[arg(long, value_name = "MODE")]
    pub weights: Option<String>,
    /// The folder of the master light; default `<work>/stack`.
    #[arg(short = 'o', long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// The container of the master.
    #[arg(long, value_enum, default_value = "fits")]
    pub format: FormatArg,
}

impl StackArgs {
    /// The `--set`s the short flags stand for.
    pub fn sugar(&self) -> Vec<Set> {
        let mut sets = Vec::new();
        if self.drizzle {
            sets.push(("drizzle.enabled".to_string(), "true".to_string()));
        }
        if let Some(r) = &self.rejection {
            let value = if r.trim_start().starts_with('{') { r.clone() } else { serde_json::json!({ "method": r }).to_string() };
            sets.push(("integration.rejection".to_string(), value));
        }
        if let Some(w) = &self.weights {
            sets.push(("measurement.weightMode".to_string(), w.clone()));
        }
        sets
    }
}

/// `starloom run`.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// `--set` and `-v` after the subcommand.
    #[command(flatten)]
    pub late: Levels,
    /// Raw lights, folders or globs.
    #[arg(value_name = "LIGHTS")]
    pub lights: Vec<OsString>,
    /// The calibration frames.
    #[command(flatten)]
    pub cal: CalFlags,
    /// A calibration plan (`.toml` or `.json`) with `[[group]]`s, in place of LIGHTS and the flags above.
    #[arg(long, value_name = "FILE")]
    pub plan: Option<PathBuf>,
    /// What every frame is registered onto: a calibrated FITS file, or `auto` (the best-weighted frame).
    #[arg(long, value_name = "FILE|auto", default_value = "auto")]
    pub reference: String,
    /// The star profile: `auto`, `moffat4` or `moffat=<beta>`.
    #[arg(long, value_name = "MODEL")]
    pub psf: Option<String>,
    /// The folder of the master light; default `<work>/stack`.
    #[arg(short = 'o', long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// The container of the master.
    #[arg(long, value_enum, default_value = "fits")]
    pub format: FormatArg,
}

impl RunArgs {
    /// The `--set`s the short flags stand for.
    pub fn sugar(&self) -> Result<Vec<Set>> {
        psf_sugar(self.psf.as_deref())
    }
}

/// Splits `--set path=value` at the first `=`.
pub fn parse_set(raw: &str) -> Result<Set> {
    match raw.split_once('=') {
        Some((path, value)) if !path.trim().is_empty() => Ok((path.trim().to_string(), value.to_string())),
        _ => bail!("--set expects PATH=VALUE (a dotted camelCase path, then a value), got `{raw}`"),
    }
}

/// Parses a size: a byte count, or a number with a unit (`B`, `K`/`KB`/`KiB`, `M`, `G`, `T`; case-insensitive,
/// 1024-based), such as `16GB` or `1.5G`.
pub fn parse_memory(raw: &str) -> Result<u64> {
    let text = raw.trim();
    let split = text.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number.parse().map_err(|_| anyhow::anyhow!("--memory expects bytes or a size such as 16GB, got `{raw}`"))?;
    let unit = unit.trim().to_ascii_lowercase();
    let shift = match unit.as_str() {
        "" | "b" => 0,
        "k" | "kb" | "kib" => 10,
        "m" | "mb" | "mib" => 20,
        "g" | "gb" | "gib" => 30,
        "t" | "tb" | "tib" => 40,
        _ => bail!("--memory: unknown unit `{unit}` in `{raw}` (use B, KB, MB, GB or TB)"),
    };
    let bytes = value * (1u64 << shift) as f64;
    if !bytes.is_finite() || bytes < 1.0 {
        bail!("--memory must be at least one byte, got `{raw}`");
    }
    Ok(bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psf_is_spelled_as_the_config_wire_spells_it() {
        let one = |p: &str| psf_sugar(Some(p)).unwrap().remove(0);
        assert_eq!(one("auto").1, "auto");
        assert_eq!(one("moffat4").1, "moffat4");
        assert_eq!(one("moffat=8").1, r#"{"moffat":8.0}"#);
        assert_eq!(one("moffat=4").1, r#"{"moffat":4.0}"#);
        assert_eq!(one("moffat=2.5").1, r#"{"moffat":2.5}"#);
        assert!(psf_sugar(Some("moffat=abc")).is_err());
        assert!(psf_sugar(Some("gauss")).is_err());
        assert!(psf_sugar(None).unwrap().is_empty());
    }

    #[test]
    fn a_set_splits_at_the_first_equals_and_keeps_json_whole() {
        assert_eq!(parse_set("a.b=1").unwrap(), ("a.b".to_string(), "1".to_string()));
        assert_eq!(parse_set(r#"x.y={"k":"a=b"}"#).unwrap().1, r#"{"k":"a=b"}"#);
        assert!(parse_set("novalue").is_err());
        assert!(parse_set("=1").is_err());
    }

    #[test]
    fn memory_takes_bytes_and_sizes() {
        assert_eq!(parse_memory("1048576").unwrap(), 1 << 20);
        assert_eq!(parse_memory("16GB").unwrap(), 16 << 30);
        assert_eq!(parse_memory("1.5g").unwrap(), 3 << 29);
        assert_eq!(parse_memory("512 MiB").unwrap(), 512 << 20);
        assert!(parse_memory("lots").is_err());
        assert!(parse_memory("5XB").is_err());
        assert!(parse_memory("0").is_err());
    }

    #[test]
    fn set_and_v_join_across_the_subcommand() {
        let cli = Cli::try_parse_from(["starloom", "-v", "--set", "a.b=1", "stack", "--set", "c.d=2", "-v", "--set", "e.f=3"]).unwrap();
        assert_eq!(cli.sets(), ["a.b=1", "c.d=2", "e.f=3"], "the ones before the subcommand first");
        assert_eq!(cli.verbose(), 2);
        let cli = Cli::try_parse_from(["starloom", "config", "show", "-vv", "--set", "x.y=1"]).unwrap();
        assert_eq!((cli.sets(), cli.verbose()), (vec!["x.y=1".to_string()], 2));
        let cli = Cli::try_parse_from(["starloom", "facts", "a.fits"]).unwrap();
        assert!(cli.sets().is_empty());
        assert_eq!(cli.verbose(), 0);
    }

    #[test]
    fn the_rejection_help_spells_the_wire_in_camel_case() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let stack = cmd.find_subcommand("stack").unwrap();
        let help = stack.get_arguments().find(|a| a.get_id() == "rejection").unwrap().get_help().unwrap().to_string();
        assert!(help.contains("sigmaClip") && help.contains("sigmaLow") && !help.contains("sigma_low"), "{help}");
    }

    #[test]
    fn rejection_is_a_method_name_or_json() {
        let args = |r: &str| StackArgs {
            late: Levels::default(),
            frames: vec![],
            measure: None,
            transforms: None,
            ln: None,
            drizzle: false,
            rejection: Some(r.to_string()),
            weights: None,
            out: None,
            format: FormatArg::Fits,
        };
        assert_eq!(args("auto").sugar()[0].1, r#"{"method":"auto"}"#);
        assert_eq!(args(r#"{"method":"none"}"#).sugar()[0].1, r#"{"method":"none"}"#);
    }
}
