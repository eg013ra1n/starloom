//! What a command is given: frame arguments (files, folders, globs), the order a stack takes lights in, and the
//! calibration plan file.
//!
//! The command expands globs itself, because Windows shells do not. Every path that leaves this module is absolute
//! (`std::path::absolute`: no symlink is resolved and no case is changed) and is compared as a [`Path`], never as
//! text.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use astroimage::header::FrameFacts;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;

/// The files a command may be given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Frames the pipeline reads as pixels only after its own calibration: plain FITS or XISF, case-insensitively
    /// `.fit`, `.fits`, `.fts` and `.xisf`. A calibrated frame is always one of these.
    Calibrated,
    /// Frames as a camera writes them: everything [`InputKind::Calibrated`] accepts, plus the compressed forms
    /// `.fz`, `.fit.gz`, `.fits.gz` and `.fts.gz` (P2-R22).
    Raw,
}

impl InputKind {
    fn describe(self) -> &'static str {
        match self {
            InputKind::Calibrated => ".fit, .fits, .fts or .xisf",
            InputKind::Raw => ".fit, .fits, .fts, .xisf, .fz, .fit.gz, .fits.gz or .fts.gz",
        }
    }
}

/// Every frame suffix, the longest of a family first, so `a.fits.gz` drops `.fits.gz` and not `.gz`.
const PLAIN_SUFFIXES: [&str; 4] = [".fits", ".fit", ".fts", ".xisf"];
const COMPRESSED_SUFFIXES: [&str; 7] = [".fits.fz", ".fit.fz", ".fts.fz", ".fits.gz", ".fit.gz", ".fts.gz", ".fz"];

/// The length of the frame suffix `name` ends with (case-insensitively) among those `kind` accepts.
fn suffix_len(name: &str, kind: InputKind) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    let compressed = (kind == InputKind::Raw).then_some(COMPRESSED_SUFFIXES.as_slice()).unwrap_or(&[]);
    compressed
        .iter()
        .chain(PLAIN_SUFFIXES.iter())
        .find(|s| lower.ends_with(**s))
        .map(|s| s.len())
}

/// A frame file's stem: its name without the whole frame suffix, so `Light_0001.fits.gz` and `Light_0001.fit` are
/// both `Light_0001`. A name without a frame suffix drops its last extension only, and a name that is not UTF-8 has
/// the empty stem ([`expand_inputs`] refuses such names).
pub fn frame_stem(path: &Path) -> &str {
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return "";
    };
    match suffix_len(name, InputKind::Raw) {
        Some(n) => &name[..name.len() - n],
        None => path.file_stem().and_then(OsStr::to_str).unwrap_or(""),
    }
}

fn is_glob(arg: &str) -> bool {
    arg.contains(['*', '?', '['])
}

fn absolute(p: &Path) -> Result<PathBuf> {
    std::path::absolute(p).with_context(|| format!("cannot make {} absolute", p.display()))
}

/// Sorts by file name, then by the whole path: the order a folder or glob expands in.
fn sort_by_file_name(files: &mut [PathBuf]) {
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()).then_with(|| a.cmp(b)));
}

fn accepted(path: &Path, kind: InputKind) -> bool {
    path.file_name().and_then(OsStr::to_str).is_some_and(|n| suffix_len(n, kind).is_some())
}

fn expand_one(arg: &OsStr, kind: InputKind) -> Result<Vec<PathBuf>> {
    let p = Path::new(arg);
    if p.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(p).with_context(|| format!("cannot read folder {}", p.display()))? {
            let path = entry.with_context(|| format!("cannot read folder {}", p.display()))?.path();
            if path.is_file() && accepted(&path, kind) {
                files.push(absolute(&path)?);
            } else {
                tracing::debug!(path = %path.display(), "not a frame file; skipped");
            }
        }
        if files.is_empty() {
            bail!("folder {} holds no frame files ({})", p.display(), kind.describe());
        }
        sort_by_file_name(&mut files);
        return Ok(files);
    }
    if p.is_file() {
        if !accepted(p, kind) {
            bail!("{} is not a frame file ({})", p.display(), kind.describe());
        }
        return Ok(vec![absolute(p)?]);
    }
    let text = arg.to_str().ok_or_else(|| anyhow!("{} does not exist", p.display()))?;
    if !is_glob(text) {
        bail!("{} does not exist", p.display());
    }
    // Case-insensitive on every platform: the extension rules are, and a pattern should behave the same everywhere.
    let options = glob::MatchOptions {
        case_sensitive: false,
        ..glob::MatchOptions::new()
    };
    let mut files = Vec::new();
    for hit in glob::glob_with(text, options).with_context(|| format!("`{text}` is not a valid pattern"))? {
        let path = hit.with_context(|| format!("cannot read a match of `{text}`"))?;
        if path.is_file() && accepted(&path, kind) {
            files.push(absolute(&path)?);
        } else {
            tracing::debug!(path = %path.display(), "not a frame file; skipped");
        }
    }
    if files.is_empty() {
        bail!("`{text}` matched no frame files ({})", kind.describe());
    }
    sort_by_file_name(&mut files);
    Ok(files)
}

/// Refuses a file given twice.
fn check_given_once(files: &[PathBuf]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for f in files {
        if !seen.insert(f) {
            bail!("{} is given twice", f.display());
        }
    }
    Ok(())
}

/// Refuses two frames with the same stem (compared case-insensitively, as a Windows or macOS volume compares): the
/// products of both would be written to one name, and the second would silently replace the first.
fn check_unique_stems(files: &[PathBuf]) -> Result<()> {
    let mut seen: BTreeMap<String, &PathBuf> = BTreeMap::new();
    for f in files {
        if f.file_name().and_then(OsStr::to_str).is_none() {
            bail!("{} has a file name that is not valid UTF-8; rename it", f.display());
        }
        let stem = frame_stem(f).to_lowercase();
        if let Some(first) = seen.insert(stem, f) {
            if first == f {
                bail!("{} is given twice", f.display());
            }
            bail!(
                "{} and {} have the same name `{}`, so their products would overwrite each other; \
                 give frames with distinct names",
                first.display(),
                f.display(),
                frame_stem(f)
            );
        }
    }
    Ok(())
}

fn expand_all(args: &[OsString], kind: InputKind) -> Result<Vec<PathBuf>> {
    let mut all = Vec::new();
    for arg in args {
        all.extend(expand_one(arg, kind)?);
    }
    tracing::debug!(arguments = args.len(), files = all.len(), "inputs expanded");
    Ok(all)
}

/// Expands the frame arguments of a command whose frames each get products of their own (the lights of `calibrate`,
/// the frames of `measure`, `register`, `normalize` and `stack`): each argument is a file, a folder (its frame files,
/// not recursive) or a glob pattern the shell did not expand. The result is absolute, in argument order, and sorted by
/// file name within a folder or a glob (P2-R10).
///
/// It refuses: an argument that names nothing, a file without a frame suffix for `kind`, a folder or a glob without
/// a frame file, and two frames with the same stem ([`frame_stem`]), such as two nights' `Light_0001.fit` or
/// `Light_0001.fit` next to `Light_0001.fits.gz`; the message names both paths.
pub fn expand_inputs(args: &[OsString], kind: InputKind) -> Result<Vec<PathBuf>> {
    let all = expand_all(args, kind)?;
    check_unique_stems(&all)?;
    Ok(all)
}

/// Expands raw calibration frames (darks, flats, biases) as [`expand_inputs`] expands raw frames, without the stem
/// rule: they are integrated into a master and leave no product of their own, so two sessions' `Dark_0001.fit` are
/// two frames. Only the same file given twice is refused.
pub fn expand_calibration_frames(args: &[OsString]) -> Result<Vec<PathBuf>> {
    let all = expand_all(args, InputKind::Raw)?;
    check_given_once(&all)?;
    Ok(all)
}

/// Parses a `DATE-OBS` (and the `TIME-OBS` that completes a date-only one) as Athenaeum's scanner does: RFC 3339, an
/// ISO 8601 stamp without a zone (UTC is assumed, with or without fractional seconds), the two joined by `T`, then a
/// bare date. `None` when nothing parses.
pub fn parse_date_obs(date_obs: &str, time_obs: Option<&str>) -> Option<DateTime<Utc>> {
    // ISO 8601 with a zone (YYYY-MM-DDTHH:MM:SS.SSSZ).
    if let Ok(dt) = DateTime::parse_from_rfc3339(date_obs) {
        return Some(dt.with_timezone(&Utc));
    }
    // Without a zone: UTC.
    if let Ok(naive) = NaiveDateTime::parse_from_str(date_obs, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(date_obs, "%Y-%m-%dT%H:%M:%S") {
        return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
    }
    // DATE-OBS and TIME-OBS combined.
    if let Some(time) = time_obs {
        let joined = format!("{date_obs}T{time}");
        if let Ok(dt) = DateTime::parse_from_rfc3339(&joined) {
            return Some(dt.with_timezone(&Utc));
        }
        if let Ok(naive) = NaiveDateTime::parse_from_str(&joined, "%Y-%m-%dT%H:%M:%S%.f") {
            return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
        }
    }
    // The date alone (YYYY-MM-DD).
    if let Ok(naive) = NaiveDateTime::parse_from_str(&format!("{date_obs} 00:00:00"), "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
    }
    None
}

/// The order a stack takes its lights in (P2-R10): by `DATE-OBS` as the facts reader gives it (parsed with
/// [`parse_date_obs`]), then by input position, which stands in for the catalog id Athenaeum breaks ties with. A light
/// without a parsable date goes after the dated ones, in input order. Returns indices into `lights`; `facts[i]` is
/// `lights[i]`'s, and a count that differs is an error.
pub fn order_lights(lights: &[PathBuf], facts: &[FrameFacts]) -> Result<Vec<usize>> {
    if lights.len() != facts.len() {
        bail!("{} lights but {} facts records; each light needs its own", lights.len(), facts.len());
    }
    let dates: Vec<Option<DateTime<Utc>>> = facts
        .iter()
        .map(|f| f.date_obs.as_deref().and_then(|d| parse_date_obs(d, f.time_obs.as_deref())))
        .collect();
    let mut order: Vec<usize> = (0..lights.len()).collect();
    // `sort_by_key` is stable, so equal keys keep their input position.
    order.sort_by_key(|&i| (dates[i].is_none(), dates[i]));
    let undated = dates.iter().filter(|d| d.is_none()).count();
    if undated > 0 {
        let first = order.iter().copied().find(|&i| dates[i].is_none()).map(|i| lights[i].display().to_string());
        tracing::warn!(undated, first = first.as_deref().unwrap_or(""), "lights without a DATE-OBS go last, in input order");
    }
    Ok(order)
}

/// Where a calibration frame comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum CalSource {
    /// A master file, used as it is.
    Master(PathBuf),
    /// Raw frames the pipeline integrates into a master first.
    Raw(Vec<PathBuf>),
}

/// How the flats of a group are pre-calibrated before they are integrated.
#[derive(Debug, Clone, PartialEq)]
pub enum FlatPrecalSource {
    /// A master (dark flat or bias) subtracted from every flat.
    Master(PathBuf),
    /// A constant, in DN, subtracted from every flat.
    SyntheticBias(f32),
}

/// One calibration group: lights that share their calibration frames.
#[derive(Debug, Clone, PartialEq)]
pub struct CalGroup {
    /// The raw lights, in plan order (the order a stack sorts them in is [`order_lights`]'s).
    pub lights: Vec<PathBuf>,
    /// The dark, if any.
    pub dark: Option<CalSource>,
    /// The flat, if any.
    pub flat: Option<CalSource>,
    /// The bias, if any (used only when there is no dark).
    pub bias: Option<CalSource>,
    /// How the flats are pre-calibrated, if at all.
    pub flat_precal: Option<FlatPrecalSource>,
}

/// A plan value that is one string or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn into_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }
    }
}

/// A plan's flat pre-calibration: a path, or a number of DN.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawPrecal {
    Path(String),
    Dn(f64),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGroup {
    lights: OneOrMany,
    dark: Option<OneOrMany>,
    flat: Option<OneOrMany>,
    bias: Option<OneOrMany>,
    flat_precal: Option<RawPrecal>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlan {
    #[serde(default)]
    group: Vec<RawGroup>,
}

/// A plan's path or glob, resolved against the plan's own folder when it is relative. The folder part of a glob is
/// escaped, so a folder name holding `[`, `*` or `?` matches as itself.
fn plan_relative(base: &Path, value: &str) -> String {
    let p = Path::new(value);
    if p.is_absolute() {
        return value.to_string();
    }
    if is_glob(value) {
        let base = glob::Pattern::escape(&base.to_string_lossy());
        return Path::new(&base).join(value).to_string_lossy().into_owned();
    }
    base.join(p).to_string_lossy().into_owned()
}

/// A dark, flat or bias value: one existing file is a master; a folder, a glob or several entries are raw frames.
fn cal_source(base: &Path, values: OneOrMany, what: &str) -> Result<CalSource> {
    let values: Vec<String> = values.into_vec().iter().map(|v| plan_relative(base, v)).collect();
    if let [one] = values.as_slice() {
        let p = Path::new(one);
        if p.is_file() {
            return Ok(CalSource::Master(absolute(p)?));
        }
    }
    let args: Vec<OsString> = values.into_iter().map(OsString::from).collect();
    Ok(CalSource::Raw(expand_calibration_frames(&args).with_context(|| format!("the {what} of a plan group"))?))
}

/// Reads a calibration plan: a TOML or JSON file (by its extension) with a `[[group]]` array. Each group has
/// `lights` (files, folders or globs), and optionally `dark`, `flat` and `bias` (one master file, or a folder or glob
/// of raw frames, or a list of those) and `flat_precal` (a master file, or a number of DN). Relative paths and globs
/// resolve against the plan's own folder, so a plan kept beside its frames works from any current directory.
///
/// It refuses a plan without a group, a group without a light, an unknown key, and a light stem that appears twice
/// anywhere in the plan (calibration frames may share stems: they leave no product of their own).
pub fn read_plan(path: &Path) -> Result<Vec<CalGroup>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read plan {}", path.display()))?;
    let ext = path.extension().and_then(OsStr::to_str).map(str::to_ascii_lowercase);
    let plan: RawPlan = match ext.as_deref() {
        Some("toml") => toml::from_str(&text).with_context(|| format!("plan {} is not valid TOML for a plan", path.display()))?,
        Some("json") => {
            serde_json::from_str(&text).with_context(|| format!("plan {} is not valid JSON for a plan", path.display()))?
        }
        _ => bail!("plan {} must be a .toml or .json file", path.display()),
    };
    if plan.group.is_empty() {
        bail!("plan {} has no [[group]]", path.display());
    }
    let plan_dir = absolute(path)?.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut groups = Vec::new();
    for (i, g) in plan.group.into_iter().enumerate() {
        let n = i + 1;
        let args: Vec<OsString> =
            g.lights.into_vec().iter().map(|v| OsString::from(plan_relative(&plan_dir, v))).collect();
        if args.is_empty() {
            bail!("plan {}: group {n} has no lights", path.display());
        }
        let lights = expand_inputs(&args, InputKind::Raw).with_context(|| format!("lights of plan group {n}"))?;
        let flat_precal = match g.flat_precal {
            None => None,
            Some(RawPrecal::Path(p)) => {
                let p = plan_relative(&plan_dir, &p);
                let p = Path::new(&p);
                if !p.is_file() {
                    bail!("plan group {n}: flat_precal {} is not a file", p.display());
                }
                Some(FlatPrecalSource::Master(absolute(p)?))
            }
            Some(RawPrecal::Dn(v)) => {
                if !v.is_finite() {
                    bail!("plan group {n}: flat_precal must be a finite number of DN");
                }
                Some(FlatPrecalSource::SyntheticBias(v as f32))
            }
        };
        groups.push(CalGroup {
            lights,
            dark: g.dark.map(|v| cal_source(&plan_dir, v, "dark")).transpose().with_context(|| format!("plan group {n}"))?,
            flat: g.flat.map(|v| cal_source(&plan_dir, v, "flat")).transpose().with_context(|| format!("plan group {n}"))?,
            bias: g.bias.map(|v| cal_source(&plan_dir, v, "bias")).transpose().with_context(|| format!("plan group {n}"))?,
            flat_precal,
        });
    }
    let all_lights: Vec<PathBuf> = groups.iter().flat_map(|g| g.lights.iter().cloned()).collect();
    check_unique_stems(&all_lights).with_context(|| format!("lights of plan {}", path.display()))?;
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"x").unwrap();
        p
    }

    fn names(files: &[PathBuf]) -> Vec<String> {
        files.iter().map(|f| f.file_name().unwrap().to_str().unwrap().to_string()).collect()
    }

    #[test]
    fn inputs_expand_globs_and_folders_in_file_name_order() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("my frames");
        std::fs::create_dir(&dir).unwrap();
        for n in ["b.FITS", "a.fit", "c.xisf", "d.fits.gz", "e.fz", "notes.txt"] {
            touch(&dir, n);
        }
        let folder = vec![dir.clone().into_os_string()];
        let cal = expand_inputs(&folder, InputKind::Calibrated).unwrap();
        assert_eq!(names(&cal), ["a.fit", "b.FITS", "c.xisf"]);
        assert!(cal.iter().all(|p| p.is_absolute()));
        let raw = expand_inputs(&folder, InputKind::Raw).unwrap();
        assert_eq!(names(&raw), ["a.fit", "b.FITS", "c.xisf", "d.fits.gz", "e.fz"]);

        // A glob the shell did not expand: sorted by file name, the extension filter applied, case-insensitive.
        let pattern = dir.join("*.fit*").into_os_string();
        let globbed = expand_inputs(&[pattern], InputKind::Raw).unwrap();
        assert_eq!(names(&globbed), ["a.fit", "b.FITS", "d.fits.gz"]);
        let upper = expand_inputs(&[dir.join("A.FI?").into_os_string()], InputKind::Calibrated).unwrap();
        assert_eq!(names(&upper), ["a.fit"], "a pattern matches case-insensitively on every platform");

        // Arguments keep their order; each expands on its own.
        let mixed = expand_inputs(&[dir.join("c.xisf").into_os_string(), dir.join("a.fit").into_os_string()], InputKind::Raw).unwrap();
        assert_eq!(names(&mixed), ["c.xisf", "a.fit"]);

        let none = expand_inputs(&[dir.join("*.none").into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(none.contains("*.none"), "{none}");
        let txt = expand_inputs(&[dir.join("notes.txt").into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(txt.contains("notes.txt"), "{txt}");
        let fz_as_calibrated = expand_inputs(&[dir.join("e.fz").into_os_string()], InputKind::Calibrated);
        assert!(fz_as_calibrated.is_err(), "a compressed frame is not a calibrated frame");
        let missing = expand_inputs(&[dir.join("absent.fit").into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(missing.contains("absent.fit"), "{missing}");
        let empty = tmp.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        let none = expand_inputs(&[empty.clone().into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(none.contains("holds no frame files"), "{none}");
    }

    #[test]
    fn frame_stems_drop_the_whole_suffix() {
        for (name, stem) in [
            ("Light_0001.fit", "Light_0001"),
            ("Light_0001.FITS", "Light_0001"),
            ("Light_0001.fits.gz", "Light_0001"),
            ("Light_0001.Fit.GZ", "Light_0001"),
            ("Light_0001.fits.fz", "Light_0001"),
            ("Light_0001.fz", "Light_0001"),
            ("m.31.xisf", "m.31"),
            ("odd.name.txt", "odd.name"),
        ] {
            assert_eq!(frame_stem(Path::new(name)), stem, "{name}");
        }
    }

    #[test]
    fn inputs_refuse_a_file_stem_given_twice() {
        let tmp = tempfile::tempdir().unwrap();
        let (n1, n2) = (tmp.path().join("night 1"), tmp.path().join("night 2"));
        std::fs::create_dir(&n1).unwrap();
        std::fs::create_dir(&n2).unwrap();
        let a = touch(&n1, "Light_0001.fit");
        let b = touch(&n2, "Light_0001.fit");
        let err = expand_inputs(&[n1.clone().into_os_string(), n2.into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(err.contains(&a.display().to_string()) && err.contains(&b.display().to_string()), "{err}");

        let c = touch(&n1, "Light_0002.fit");
        let d = touch(&n1, "Light_0002.fits.gz");
        let err = expand_inputs(&[c.clone().into_os_string(), d.clone().into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(err.contains(&c.display().to_string()) && err.contains(&d.display().to_string()), "{err}");

        // A folder holding both is refused the same way; the same file twice says so.
        let err = expand_inputs(&[n1.into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(err.contains("Light_0002"), "{err}");
        let err = expand_inputs(&[a.clone().into_os_string(), a.into_os_string()], InputKind::Raw).unwrap_err().to_string();
        assert!(err.contains("given twice"), "{err}");
    }

    /// M1: calibration frames leave no product of their own, so two sessions' `Dark_0001.fit` are two darks; the
    /// same file twice is still refused.
    #[test]
    fn calibration_frames_may_share_a_stem() {
        let tmp = tempfile::tempdir().unwrap();
        let (s1, s2) = (tmp.path().join("session 1"), tmp.path().join("session 2"));
        std::fs::create_dir(&s1).unwrap();
        std::fs::create_dir(&s2).unwrap();
        let a = touch(&s1, "Dark_0001.fit");
        touch(&s2, "Dark_0001.fit");
        touch(&s2, "Dark_0001.fits.gz");
        let darks = expand_calibration_frames(&[s1.clone().into_os_string(), s2.clone().into_os_string()]).unwrap();
        assert_eq!(names(&darks), ["Dark_0001.fit", "Dark_0001.fit", "Dark_0001.fits.gz"]);
        assert!(expand_inputs(&[s1.into_os_string(), s2.into_os_string()], InputKind::Raw).is_err(), "lights may not");
        let err = expand_calibration_frames(&[a.clone().into_os_string(), a.into_os_string()]).unwrap_err().to_string();
        assert!(err.contains("given twice"), "{err}");
    }

    /// M6: a plan's relative paths and globs resolve against the plan's folder, whatever the current directory.
    #[test]
    fn a_plan_s_relative_paths_resolve_against_its_own_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("my [target]");
        for d in ["lights", "darks 1", "darks 2"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let l1 = touch(&root.join("lights"), "L1.fit");
        touch(&root.join("lights"), "L2.fit");
        touch(&root.join("darks 1"), "Dark_0001.fit");
        touch(&root.join("darks 2"), "Dark_0001.fit");
        let flat = touch(&root, "master flat.fits");
        let precal = touch(&root, "dark flat.fits");
        let plan = root.join("plan.toml");
        std::fs::write(
            &plan,
            "[[group]]\nlights = \"lights/*.fit\"\ndark = [\"darks 1\", \"darks 2\"]\nflat = \"master flat.fits\"\n\
             flat_precal = \"dark flat.fits\"\n",
        )
        .unwrap();
        let groups = read_plan(&plan).unwrap();
        assert_eq!(groups[0].lights, [l1.clone(), root.join("lights").join("L2.fit")]);
        assert!(groups[0].lights.iter().all(|p| p.is_absolute()));
        match groups[0].dark.as_ref().unwrap() {
            CalSource::Raw(frames) => assert_eq!(frames.len(), 2, "two sessions' darks of one stem"),
            other => panic!("two folders are raw frames: {other:?}"),
        }
        assert_eq!(groups[0].flat, Some(CalSource::Master(flat)));
        assert_eq!(groups[0].flat_precal, Some(FlatPrecalSource::Master(precal)));
        // An absolute path stays as it is.
        let other = tempfile::tempdir().unwrap();
        let plan2 = other.path().join("plan.json");
        std::fs::write(&plan2, format!("{{\"group\":[{{\"lights\":[{:?}]}}]}}", l1.to_str().unwrap())).unwrap();
        assert_eq!(read_plan(&plan2).unwrap()[0].lights, [l1]);
    }

    fn facts(date: Option<&str>, time: Option<&str>) -> FrameFacts {
        let mut f = FrameFacts::default();
        f.date_obs = date.map(str::to_string);
        f.time_obs = time.map(str::to_string);
        f
    }

    #[test]
    fn lights_sort_by_date_then_input_position() {
        let lights: Vec<PathBuf> = (0..6).map(|i| PathBuf::from(format!("l{i}.fit"))).collect();
        let f = vec![
            facts(Some("2024-01-02T00:00:00"), None),
            facts(None, None),
            facts(Some("2024-01-01T00:00:00"), None),
            facts(Some("2024-01-01"), Some("00:00:00")), // the same instant as 2: input position breaks the tie
            facts(Some("not a date"), None),
            facts(Some("2023-12-31T23:59:59.5Z"), None),
        ];
        assert_eq!(order_lights(&lights, &f).unwrap(), [5, 2, 3, 0, 1, 4]);
        // M4: a facts list of another length is an error in every build, never a silent drop.
        let err = order_lights(&lights, &f[..5]).unwrap_err().to_string();
        assert!(err.contains("6 lights but 5 facts"), "{err}");
    }

    #[test]
    fn parse_date_obs_joins_time_obs_and_sorts_dateless_frames_last() {
        let want = |s: &str| Some(DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc));
        assert_eq!(parse_date_obs("2024-01-15T20:30:45.123Z", None), want("2024-01-15T20:30:45.123Z"));
        assert_eq!(parse_date_obs("2024-01-15T20:30:45+02:00", None), want("2024-01-15T18:30:45Z"));
        assert_eq!(parse_date_obs("2024-01-15T20:30:45.5", None), want("2024-01-15T20:30:45.5Z"));
        assert_eq!(parse_date_obs("2024-01-15T20:30:45", None), want("2024-01-15T20:30:45Z"));
        assert_eq!(parse_date_obs("2024-01-15", Some("20:30:45")), want("2024-01-15T20:30:45Z"));
        assert_eq!(parse_date_obs("2024-01-15", Some("20:30:45.25")), want("2024-01-15T20:30:45.25Z"));
        assert_eq!(parse_date_obs("2024-01-15", None), want("2024-01-15T00:00:00Z"));
        assert_eq!(parse_date_obs("2024-01-15", Some("later")), want("2024-01-15T00:00:00Z"));
        assert_eq!(parse_date_obs("yesterday", Some("20:30:45")), None);
        // Dateless frames go last, in input order.
        let lights: Vec<PathBuf> = (0..3).map(|i| PathBuf::from(format!("l{i}.fit"))).collect();
        let f = vec![facts(None, None), facts(Some("2024-01-15"), None), facts(None, Some("20:30:45"))];
        assert_eq!(order_lights(&lights, &f).unwrap(), [1, 0, 2]);
    }

    #[test]
    fn a_toml_plan_reads_groups_masters_raw_frames_and_flat_precal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let lights = root.join("lights");
        let darks = root.join("darks");
        std::fs::create_dir(&lights).unwrap();
        std::fs::create_dir(&darks).unwrap();
        touch(&lights, "L1.fit");
        touch(&lights, "L2.fit");
        touch(&darks, "D1.fit");
        touch(&darks, "D2.fit");
        let flat = touch(root, "master flat.fits");
        let precal = touch(root, "dark flat.fits");
        let lit = |p: &Path| p.display().to_string().replace('\\', "/");
        let other_lights = root.join("more lights");
        std::fs::create_dir(&other_lights).unwrap();
        touch(&other_lights, "L1.fit");
        let plan = format!(
            "[[group]]\nlights = [\"{l}/*.fit\"]\ndark = \"{d}\"\nflat = \"{f}\"\nflat_precal = \"{p}\"\n\n\
             [[group]]\nlights = \"{o}\"\nflat_precal = 500\n",
            l = lit(&lights),
            d = lit(&darks),
            f = lit(&flat),
            p = lit(&precal),
            o = lit(&other_lights),
        );
        // The second group's folder holds another L1: one stem in two places is refused across groups.
        let path = root.join("plan.toml");
        std::fs::write(&path, &plan).unwrap();
        let err = format!("{:#}", read_plan(&path).unwrap_err());
        assert!(err.contains("L1"), "{err}");

        std::fs::remove_file(other_lights.join("L1.fit")).unwrap();
        touch(&other_lights, "L3.fit");
        let groups = read_plan(&path).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(names(&groups[0].lights), ["L1.fit", "L2.fit"]);
        match groups[0].dark.as_ref().unwrap() {
            CalSource::Raw(frames) => assert_eq!(names(frames), ["D1.fit", "D2.fit"]),
            other => panic!("a folder is raw frames: {other:?}"),
        }
        assert!(matches!(groups[0].flat.as_ref().unwrap(), CalSource::Master(p) if p.file_name().unwrap() == "master flat.fits"));
        assert!(matches!(groups[0].flat_precal.as_ref().unwrap(), FlatPrecalSource::Master(_)));
        assert_eq!(groups[1].flat_precal, Some(FlatPrecalSource::SyntheticBias(500.0)));
        assert!(groups[1].dark.is_none() && groups[1].bias.is_none());
    }

    #[test]
    fn a_json_plan_is_read_by_extension_and_bad_plans_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let l = touch(root, "L1.fit");
        let json = root.join("plan.json");
        std::fs::write(&json, format!("{{\"group\":[{{\"lights\":[{:?}]}}]}}", l.to_str().unwrap())).unwrap();
        let groups = read_plan(&json).unwrap();
        assert_eq!(groups[0].lights.len(), 1);

        let empty = root.join("empty.toml");
        std::fs::write(&empty, "").unwrap();
        assert!(read_plan(&empty).unwrap_err().to_string().contains("no [[group]]"));
        let typo = root.join("typo.toml");
        std::fs::write(&typo, "[[group]]\nlight = \"x\"\n").unwrap();
        assert!(read_plan(&typo).is_err());
        let other = root.join("plan.yaml");
        std::fs::write(&other, "").unwrap();
        assert!(read_plan(&other).unwrap_err().to_string().contains(".toml or .json"));
        let nolights = root.join("nolights.toml");
        std::fs::write(&nolights, "[[group]]\nlights = []\n").unwrap();
        assert!(read_plan(&nolights).unwrap_err().to_string().contains("no lights"));
    }
}
