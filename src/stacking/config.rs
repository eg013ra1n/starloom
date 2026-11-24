//! Stacking configuration: the measurement and drizzle sections, the pure items a caller's whole stacking
//! configuration tree is built from (the caller owns the tree itself and its hashing).

use serde::{Deserialize, Serialize};

use crate::calibration_library::options::CalibratedLightOptions;
use crate::integration::stats::{RejectionNormalization, ScaleEstimator};
use crate::resample::Interpolation;
use crate::stacking::fits_artifact::FITS_ARTIFACT_VERSION;
use crate::stacking::integrate::{IntegrationConfig, NormalizationConfig, RejectionChoice};
use crate::stacking::measure::{MeasureOptions, MEASURE_STAGE_VERSION};
use crate::stacking::prefilter::SeedPrefilter;
use crate::stacking::psf_signal::PsfModel;
use crate::stacking::register::align::ALIGN_VERSION;
use crate::stacking::register::{DistortionChoice, RegistrationConfig};
use crate::stacking::structure::SeedDetector;
use crate::stacking::weights::{FormulaWeights, SelectionConfig, WeightMode};
use crate::star_measure::MEASURE_VERSION;

/// The measurement settings of a stacking run, chosen once per run: how frames are weighted, which star profile is
/// fitted and how stars are found (spec §9.2 `measurement:`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct MeasurementConfig {
    /// How a frame's measurement becomes its stacking weight.
    pub weight_mode: WeightMode,
    /// The star profile fitted to each detected star.
    #[cfg_attr(feature = "ts", ts(as = "crate::stacking::psf_signal::PsfModelTs"))]
    pub psf_model: PsfModel,
    /// Detection cap: the most stars kept per frame (spec §9.2 `maxStars`; stars, not pixels).
    pub max_stars: usize,
    /// The coefficients of the user formula weight; read only by `WeightMode::Formula`.
    pub formula: FormulaWeights,
    /// FITS keyword `WeightMode::Keyword` reads its value from.
    pub keyword: String,
    /// Star-detection threshold for the quality measurement, in sigma above
    /// the local background (spec §9.2). The measurement detector's two
    /// levels sit at `background + k*noise` and `background + (k/2)*noise`,
    /// so the seed population follows THIS frame's sky instead of a fixed
    /// bright-pixel budget (M4a Task 2, ruling R-M4a-1). Peak only: the
    /// matched filter and the structure map do not read it.
    pub detection_sigma: f64,
    /// What the seed detection runs on (math reference §5.1): the plane as
    /// it is, or its 3×3 median. Everything downstream of detection always
    /// measures the untouched plane. Read by the peak detector only.
    pub seed_prefilter: SeedPrefilter,
    /// Legacy mirror of [`Self::seed_source`] (peak vs structure, R-M4c-11),
    /// kept so older readers of the stored document still find the field. It
    /// steers nothing: the measurement stage reads `seed_source` only.
    /// A stored config written before M4c Task 0 has no such field and
    /// decodes to the shipped default.
    pub seed_detector: SeedDetector,
    /// The seed source (spec §2.2): the star-measurement module's matched filter (default), the peak threshold
    /// (reads `detectionSigma` and `seedPrefilter`), or the structure map. A document without it is decoded by
    /// `legacy_seed_source`; one that carries it uses it as is.
    pub seed_source: SeedSource,
    /// The matched filter's threshold in units of the filtered plane's noise (ruling R-OSP-2). Clamped to
    /// `MIN_SEED_THRESHOLD`..=`MAX_SEED_THRESHOLD` by `resolve_config`.
    pub seed_threshold: f64,
    /// Fast measurement (FM-2, FM-3): the first K seeds in detection order (filter flux descending) get the PSF
    /// fits, the rest keep their detection record and are not fitted; `None` = all (the shipped behaviour). No
    /// imputation: the frame aggregates, `c` and the F-6 medians are over the fitted set. Clamped up to
    /// `max(MIN_FIT_STARS, registration.maxStars)` and canonicalized to `None` at or above
    /// `measurement.maxStars` by `resolve_config`. Absent from the document at `None`, so every All-mode hash is
    /// the one before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts", ts(optional))]
    pub fit_stars: Option<u32>,
}

/// Where the Measure stage takes its star seeds from, the starting points the profile fits refine (spec §2.2, ruling
/// R-OSP-2; R-M4c-11 for the other two).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub enum SeedSource {
    /// The module's two-pass matched filter ([`crate::star_measure::seeds::detect`]) at `seedThreshold`.
    #[default]
    MatchedFilter,
    /// The peak threshold: `detectionSigma` and `seedPrefilter`.
    Peak,
    /// The structure map.
    Structure,
}

impl SeedSource {
    /// The serde name, for logs: `"matchedFilter"` / `"peak"` / `"structure"`.
    pub fn as_str(self) -> &'static str {
        match self {
            SeedSource::MatchedFilter => "matchedFilter",
            SeedSource::Peak => "peak",
            SeedSource::Structure => "structure",
        }
    }
}

/// The one decode for every stored document that has no `seedSource` (set, global, user preset — spec §2.2):
/// Structure → Structure; otherwise `seedPrefilter = median3` or a `detectionSigma` other than the old default
/// (20) → Peak; otherwise → MatchedFilter. A stored value is never reinterpreted in other units.
fn legacy_seed_source(
    detector: Option<SeedDetector>,
    prefilter: Option<SeedPrefilter>,
    sigma: Option<f64>,
) -> SeedSource {
    if detector == Some(SeedDetector::Structure) {
        return SeedSource::Structure;
    }
    let old_sigma = crate::stacking::measure::DEFAULT_DETECTION_SIGMA as f64;
    if prefilter == Some(SeedPrefilter::Median3) || sigma.is_some_and(|s| s != old_sigma) {
        return SeedSource::Peak;
    }
    SeedSource::MatchedFilter
}

/// The wire shape of [`MeasurementConfig`]: every field optional so the decode can tell "absent" from "default"
/// (the legacy decode needs it).
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct MeasurementConfigWire {
    weight_mode: Option<WeightMode>,
    psf_model: Option<PsfModel>,
    max_stars: Option<usize>,
    formula: Option<FormulaWeights>,
    keyword: Option<String>,
    detection_sigma: Option<f64>,
    seed_prefilter: Option<SeedPrefilter>,
    seed_detector: Option<SeedDetector>,
    seed_source: Option<SeedSource>,
    seed_threshold: Option<f64>,
    fit_stars: Option<u32>,
}

impl<'de> Deserialize<'de> for MeasurementConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = MeasurementConfigWire::deserialize(d)?;
        let base = MeasurementConfig::default();
        let seed_source = w.seed_source.unwrap_or_else(|| {
            legacy_seed_source(w.seed_detector, w.seed_prefilter, w.detection_sigma)
        });
        Ok(MeasurementConfig {
            weight_mode: w.weight_mode.unwrap_or(base.weight_mode),
            psf_model: w.psf_model.unwrap_or(base.psf_model),
            max_stars: w.max_stars.unwrap_or(base.max_stars),
            formula: w.formula.unwrap_or(base.formula),
            keyword: w.keyword.unwrap_or(base.keyword),
            detection_sigma: w.detection_sigma.unwrap_or(base.detection_sigma),
            seed_prefilter: w.seed_prefilter.unwrap_or(base.seed_prefilter),
            // Kept on the wire for round-trips; it mirrors the source (Structure <=> Structure, else Peak).
            seed_detector: if seed_source == SeedSource::Structure {
                SeedDetector::Structure
            } else {
                SeedDetector::Peak
            },
            seed_source,
            seed_threshold: w.seed_threshold.unwrap_or(base.seed_threshold),
            fit_stars: w.fit_stars,
        })
    }
}

/// The calibrated [`MeasurementConfig::detection_sigma`] default (used by
/// `Default` and by the legacy decode when a stored document has no such
/// field — not to `0.0`).
fn default_detection_sigma() -> f64 {
    crate::stacking::measure::DEFAULT_DETECTION_SIGMA as f64
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        MeasurementConfig {
            weight_mode: WeightMode::PsfSignalWeight,
            psf_model: PsfModel::Auto,
            max_stars: 9_000,
            formula: FormulaWeights::default(),
            keyword: "SSWEIGHT".to_string(),
            detection_sigma: default_detection_sigma(),
            seed_prefilter: crate::stacking::measure::DEFAULT_SEED_PREFILTER,
            seed_detector: SeedDetector::Peak,
            seed_source: SeedSource::MatchedFilter,
            seed_threshold: crate::star_measure::seeds::DEFAULT_THRESHOLD,
            fit_stars: None,
        }
    }
}

impl MeasurementConfig {
    /// [`MeasureOptions`] for Plan 2's `measure_frame`/`measure_plane`,
    /// built from this config's fields. `scale_estimator` comes from the
    /// caller (the orchestrator keeps it equal to
    /// `NormalizationConfig::scale_estimator`, spec §9.2 — the two are
    /// stored separately because normalization's estimator also drives
    /// stage 5, which never measures a frame). `min_snr` is not yet a
    /// `StackingConfig` field, so this takes the detector's own default
    /// sensitivity (`MeasureOptions::default().min_snr`).
    pub fn measure_options(&self, scale_estimator: ScaleEstimator) -> MeasureOptions {
        MeasureOptions {
            psf_model: self.psf_model,
            max_stars: self.max_stars,
            scale_estimator,
            min_snr: MeasureOptions::default().min_snr,
            detection_sigma: self.detection_sigma as f32,
            seed_prefilter: self.seed_prefilter,
            seed_source: self.seed_source,
            seed_threshold: self.seed_threshold,
            // Not a config field: a run picks the detector, not its dials
            // (M4c Task 0, ruling R-M4c-11).
            structure: MeasureOptions::default().structure,
            // The run fits every seed (route A); only the probes opt into the two populations.
            population: None,
            fit_stars: self.fit_stars.map(|k| k as usize),
        }
    }
}

/// The drizzle settings of a stacking run (spec §9.2 `drizzle:`): whether it runs, the output scale, the drop size,
/// the kernel and which maps feed it. Off by default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase", default)]
pub struct DrizzleConfig {
    /// Whether the run also produces a drizzled master (default off).
    pub enabled: bool,
    /// Output pixels per input pixel along each axis: 2 gives an image twice as wide and as tall (default 2).
    pub scale: u32,
    /// The drop's side as a fraction of an input pixel, the drizzle `pixfrac`: 1 deposits the whole pixel, smaller
    /// values a smaller drop (default 0.9; dimensionless).
    pub drop_shrink: f64,
    /// The shape each drop deposits its flux with.
    pub kernel: DrizzleKernel,
    /// Leave out the pixels the rejection stage flagged in each frame's rejection bitmap (default on).
    pub use_rejection: bool,
    /// Weight each frame's deposit by its stacking weight (default on).
    pub use_weights: bool,
    /// Whether the drizzle follows local output normalization (default on).
    ///
    /// B8 (M3 final fix wave, M5 ruling): the drizzle follows the plain
    /// master's OUTPUT normalization, so this toggle only has an effect when
    /// local normalization drives output normalization
    /// (`normalization.local.enabled == true`). With it off, the drizzle
    /// falls back to each included frame's own global output-normalization
    /// pair, exactly as [`super::integrate`]'s own engine does for the plain
    /// master - including when LN was computed for rejection alone
    /// (`normalization.rejection == "local"`, the default): those grids never
    /// drive the drizzle's output (pcores review r1, I1).
    pub use_local_normalization: bool,
    /// Also keep the drizzle's weight map, the accumulated deposit weight per output pixel divided by its maximum, so
    /// it lies in [0, 1] (default off).
    pub write_weight_map: bool,
    /// Bayer drizzle: whether each colour is deposited from its own samples of the calibrated mosaic instead of the
    /// debayered planes (default off).
    ///
    /// M4d Task 1 (ruling R-M4d-2): deposit each colour's OWN samples from
    /// the calibrated CFA mosaic instead of the debayered planes'
    /// interpolated ones. Ignored for a mono group (silently — there is no
    /// mosaic to deposit), and it makes stage 1 keep one extra artifact per
    /// OSC frame.
    ///
    /// `#[serde(default)]` rides the struct-level `default`, so every stored
    /// config written before M4d decodes as `false` and no
    /// `STACKING_CONFIG_VERSION` bump is needed. It deliberately does NOT
    /// enter `calibration_subtree`: the debayered artifact is
    /// byte-identical either way, so flipping this must not invalidate a
    /// set's whole calibrated cache — the missing `calibrated_mosaic`
    /// artifact is what makes the run regenerate the pair (`stacking::run`).
    pub bayer: bool,
}

impl Default for DrizzleConfig {
    fn default() -> Self {
        DrizzleConfig {
            enabled: false,
            scale: 2,
            drop_shrink: 0.9,
            kernel: DrizzleKernel::Square,
            use_rejection: true,
            use_weights: true,
            use_local_normalization: true,
            write_weight_map: false,
            bayer: false,
        }
    }
}

/// The shape of the drop a drizzle deposits: how an input pixel's flux is spread over the output pixels it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub enum DrizzleKernel {
    /// A uniform square drop, clipped exactly against each output pixel (the default).
    #[default]
    Square,
    /// A uniform disc inscribed in the drop's square.
    Circle,
    /// A Gaussian falling off from the drop's centre, truncated at the drop's edge.
    Gaussian,
}

/// [`crate::stacking::integrate::LargeScaleRejection::protected_layers`]
/// bounds (M4c Task 3): 1 (the cascade's first median, window 3) to 6
/// (window 65, effective support ≈ 127 px) — the range the Integrate
/// panel's field offers.
pub const MIN_PROTECTED_LAYERS: u8 = 1;
/// The largest `protected_layers` the Integrate panel offers (window 65 pixels, support about 127 pixels).
pub const MAX_PROTECTED_LAYERS: u8 = 6;
/// [`crate::stacking::integrate::LargeScaleRejection::growth`]'s maximum
/// (0 — no dilation — is legitimate).
pub const MAX_GROWTH: u8 = 4;

/// The drizzle deposit kernel's own generation (Tier C item C2, spec
/// `2026-09-20-stacking-compute-tierC-design.md` §3, ruling C-3).
///
/// * `1` — M3..M4d: the exact Sutherland-Hodgman clip of every source
///   pixel's mapped drop against every output pixel it touches.
/// * `2` — Tier C: the same clip, tabulated over a `PHASES x PHASES` grid
///   of sub-pixel phases (per frame for a linear map, per 256-px source
///   tile under distortion). Level-preserving by construction; the drop's
///   mass is split among its neighbours as if the drop sat up to
///   `1 / (2 · PHASES)` = 1/128 of an output pixel from where it really
///   is. `2` names the phase table AS SHIPPED, i.e. at
///   `drizzle::phase_table::PHASES = 64` (ruling C-21). The 32-phase
///   variant of the same kernel never left this branch, so it gets no
///   generation of its own — a generation marks a change in what a
///   RELEASED run produces.
///
/// **Why it rides `config_hash` and no `stage_hash`.** The drizzle
/// stage has no cache: it re-runs from the calibrated frames on every run,
/// so there is no stored artifact for a version bump to invalidate and no
/// per-stage subtree that folds in `drizzle` at all (the four are
/// calibration/measurement/registration/normalization). What the constant
/// DOES have to move is the run-level fingerprint — two runs of the same
/// `StackingConfig` across this change produce different drizzled pixels,
/// and the fingerprint is what a provenance row, a results card and a
/// re-run comparison read to tell one run's settings from another's.
///
/// **One thing does go stale, and it is not an artifact.** The plan gate's
/// memory of which frames a run EXCLUDED for local normalization
/// (`plan::build_plan`'s `ln_runtime_excluded_by_frame`) is gated on
/// `last_run.config_hash == hash`, deliberately — an exclusion is only
/// trusted as "will fail again for the same reason" when it came from a run
/// of this exact config. Moving the fingerprint therefore empties that set
/// once, so on the FIRST plan after a bump an LN-excluded frame reverts to
/// ordinary "no fresh `ln` artifact" staleness and is re-owed one Normalize
/// pass. That costs those frames one pass on one plan and then settles;
/// every stored `stacking_artifacts` row is untouched.
pub const DRIZZLE_KERNEL_VERSION: u32 = 2;

/// spec §9.2: built-in config transforms. Editing any field afterward makes
/// the effective config "Custom" — that bookkeeping belongs to the caller
/// (the Stacking tab / run orchestration), not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub enum StackingPreset {
    /// The shipped default.
    Default,
    /// A fast preview: bilinear interpolation, sigma clipping and no local normalization.
    FastPreview,
    /// The most the algorithms offer: polynomial distortion, rejection maps, local normalization and 2x drizzle.
    MaximumQuality,
}

/// The seven algorithm sections of a stacking configuration: what the stages compute with, and nothing of how a caller
/// groups frames, where it keeps files or what it does with the intermediates. A whole Athenaeum `StackingConfig`
/// document parses into it (it reads these seven keys and ignores the others), and a stage hash is built from its
/// subtrees ([`calibration_subtree`], [`measurement_subtree`], [`registration_subtree`], [`normalization_subtree`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AlgorithmConfig {
    /// How lights are calibrated.
    pub calibration: CalibratedLightOptions,
    /// How stars are measured and frames weighted.
    pub measurement: MeasurementConfig,
    /// Which frames are kept.
    pub selection: SelectionConfig,
    /// How frames are registered.
    pub registration: RegistrationConfig,
    /// How frames and rejection are normalized.
    pub normalization: NormalizationConfig,
    /// How frames are rejected and combined.
    pub integration: IntegrationConfig,
    /// The drizzle stage.
    pub drizzle: DrizzleConfig,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        AlgorithmConfig {
            calibration: CalibratedLightOptions::default(),
            measurement: MeasurementConfig::default(),
            selection: SelectionConfig::default(),
            registration: RegistrationConfig::default(),
            normalization: NormalizationConfig::default(),
            integration: IntegrationConfig::default(),
            drizzle: DrizzleConfig::default(),
        }
    }
}

/// Resolves a [`StackingPreset`] to its algorithm sections.
///
/// `MaximumQuality` (M3 Task 5, spec §9.2): turns on everything the spec text always described: `distortion`,
/// `writeRejectionMaps`, LOCAL NORMALIZATION and 2x DRIZZLE.
///
/// `FastPreview` (M4a Task 4, ruling R-M4a-18) pins the rejection normalization to `ScaleZeroOffset` (owner ruling
/// 2026-11-01, OD-3): the shipped default is `Local`, whose Normalize stage costs about as much as the integration
/// itself, and a preview has no use for it. `Default` and `MaximumQuality` inherit `Local`. A caller's own sections
/// (Athenaeum's `FastPreview` also turns off the two-pass reference pick and deletes intermediates) are the caller's.
pub fn algorithm_preset(p: StackingPreset) -> AlgorithmConfig {
    match p {
        StackingPreset::Default => AlgorithmConfig::default(),
        StackingPreset::FastPreview => {
            let mut c = AlgorithmConfig::default();
            c.registration.interpolation = Interpolation::Bilinear;
            c.integration.rejection = RejectionChoice::SigmaClip {
                sigma_low: 4.0,
                sigma_high: 3.0,
            };
            // OD-3: a preview's promise is speed - no Normalize stage.
            c.normalization.rejection = RejectionNormalization::ScaleZeroOffset;
            c
        }
        StackingPreset::MaximumQuality => {
            let mut c = AlgorithmConfig::default();
            c.registration.distortion = DistortionChoice::Polynomial3;
            c.integration.write_rejection_maps = true;
            c.normalization.local.enabled = true;
            c.drizzle.enabled = true;
            c.drizzle.scale = 2;
            c
        }
    }
}

/// The `grouping.exposureToleranceSec` value [`calibration_subtree`] writes, whatever the config says (spec
/// 2026-11-08 §5.2). It holds the hash shape — a set at the shipped default tolerance keeps a byte-identical hash —
/// and says nothing about the run: the tolerance decides which frames share a group, never what a calibrated frame
/// contains.
pub const CALIBRATION_HASH_GROUPING_TOLERANCE_SEC: f64 = 2.0;

/// Stage 1 (calibration) config subtree (spec §9.3): the frame's
/// calibration options plus the grouping key's shape, written as the
/// constant [`CALIBRATION_HASH_GROUPING_TOLERANCE_SEC`] (not the config's
/// tolerance).
pub fn calibration_subtree(cfg: &AlgorithmConfig) -> serde_json::Value {
    // `calibration.format` (the XISF plan, 2026-09-18, ruling R6) is the
    // container an EXPORT or a SEND writes; the run forces FITS for its own
    // calibrated frames before the spec is resolved (`stacking::run`), so
    // the field can never change a cached calibrated artifact and must not
    // move this hash — otherwise every set would recalibrate once for a
    // field it ignores. Pinned by `format_moves_the_run_fingerprint_but_no_stage_hash`.
    let mut calibration =
        serde_json::to_value(&cfg.calibration).expect("CalibratedLightOptions serializes");
    if let Some(obj) = calibration.as_object_mut() {
        obj.remove("format");
    }
    // Spec 2026-10-02 §8 (fable C1): the grouping half is written out field
    // by field, in exactly today's shape. A new `GroupingConfig` field
    // regroups frames (its group keys change, so group-keyed artifacts are
    // recomputed under the new keys) but must never move this hash, which
    // would make every frame set recompute once. Pinned by
    // `stage_hashes_are_pinned_to_golden_literals`.
    serde_json::json!({
        "calibration": calibration,
        "grouping": { "exposureToleranceSec": CALIBRATION_HASH_GROUPING_TOLERANCE_SEC },
    })
}

/// Stage 3 (measurement) config subtree: the measurement settings, the
/// scale estimator normalization keeps in lockstep with it (see
/// [`MeasurementConfig::measure_options`]), and
/// [`crate::star_measure::MEASURE_VERSION`] — the config alone
/// cannot express "the fitter itself now accepts different stars", which is
/// what M4a Task 2 did to every stored measurement (ruling R-M4a-15).
pub fn measurement_subtree(cfg: &AlgorithmConfig) -> serde_json::Value {
    measurement_subtree_with_fits_artifact_version(cfg, FITS_ARTIFACT_VERSION)
}

/// [`measurement_subtree`] with the `.athf` format version supplied (Plan D D2c r1): the version the Measure stage's
/// `fits.<plane>` files are written at is folded beside [`MEASURE_VERSION`] (`fitsArtifactVersion`), so a row measured at
/// another format version is keyed on another hash and misses by key. The plan gate then needs no read of the file: a
/// fresh row is its hash and its stat (`plan::is_fresh`). Nothing else should call this with anything but
/// [`FITS_ARTIFACT_VERSION`]; the pins of earlier versions go through
/// [`measurement_subtree_with_measure_version`], which keeps the shape from before the fold.
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn measurement_subtree_with_fits_artifact_version(
    cfg: &AlgorithmConfig,
    fits_artifact_version: u16,
) -> serde_json::Value {
    let mut subtree = measurement_subtree_with_measure_version(cfg, MEASURE_VERSION);
    subtree["fitsArtifactVersion"] = serde_json::json!(fits_artifact_version);
    fold_measure_stage_version(&mut subtree, MEASURE_STAGE_VERSION);
    subtree
}

/// Folds starloom's [`MEASURE_STAGE_VERSION`] into a subtree that reads the measure stage's numbers, as
/// `measureStageVersion`, only when it is not 1: at 1 the key is absent, so every hash keeps the bytes it had before
/// the version existed (starloom phase 1, P1-R1). A bump recomputes the measurement, registration and normalization
/// artifacts, and is raised together with `PASS1_ARTIFACT_VERSION`.
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn fold_measure_stage_version(subtree: &mut serde_json::Value, version: u32) {
    if version != 1 {
        subtree["measureStageVersion"] = serde_json::json!(version);
    }
}

/// [`measurement_subtree`] with the measure version supplied, so a test can
/// prove the hash actually follows it (nothing else should call this with
/// anything but [`MEASURE_VERSION`]).
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn measurement_subtree_with_measure_version(
    cfg: &AlgorithmConfig,
    measure_version: u32,
) -> serde_json::Value {
    // Spec 2026-11-08 §5.4: the weights are computed in every run from the stored measurements (`run/`
    // `stage_measure_with`, `compute_weights`) and no fit reads `weightMode`, `formula` or `keyword`, so they are
    // written as the shipped defaults: changing the weighting must not re-measure. A set with default weights keeps
    // its hash byte for byte.
    let defaults = MeasurementConfig::default();
    let mut measurement = cfg.measurement.clone();
    measurement.weight_mode = defaults.weight_mode;
    measurement.formula = defaults.formula;
    measurement.keyword = defaults.keyword;
    serde_json::json!({
        "measurement": measurement,
        "normalization": { "scaleEstimator": cfg.normalization.scale_estimator },
        "measureVersion": measure_version,
    })
}

/// Stage 5 (registration) config subtree, plus
/// [`crate::star_measure::MEASURE_VERSION`] (Tier C Task 3, ruling
/// C-6): a mono frame's registration star list can now come straight from
/// Measure's own accepted PSF fits (`stacking::register::detect::
/// stars_from_fits`) instead of a fresh detection, so a fitter change —
/// which changes what those fits SAY — has to re-register every mono set
/// exactly the way it already re-measures and re-normalizes them (the same
/// reasoning [`measurement_subtree`]/[`normalization_subtree`] already
/// follow). The first run after this fold-in re-registers every set once
/// on purpose, rewriting every `registered` artifact
/// (perf tier A Task 6a) and re-running every downstream stage that reads
/// it (Normalize, Integrate).
///
/// It also carries [`crate::stacking::register::align::ALIGN_VERSION`]
/// (`alignVersion`, robust quad seed RQ-5): what `align` returns for an
/// unchanged star list (the seed, the coverage gate) is no config field
/// either, so a change in it moves this hash the way a fitter change does —
/// every set re-registers once, and Calibrate and Measure stay cached.
pub fn registration_subtree(cfg: &AlgorithmConfig) -> serde_json::Value {
    let mut subtree = registration_subtree_with_versions(cfg, MEASURE_VERSION, ALIGN_VERSION);
    fold_measure_stage_version(&mut subtree, MEASURE_STAGE_VERSION);
    subtree
}

/// [`registration_subtree`] with both versions supplied, so a test can prove
/// the hash follows each (nothing else should call this with anything but
/// [`MEASURE_VERSION`] and [`ALIGN_VERSION`]). The measure-version pin
/// goes through [`registration_subtree_with_measure_version`], which
/// keeps the shape from before the fold.
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn registration_subtree_with_versions(
    cfg: &AlgorithmConfig,
    measure_version: u32,
    align_version: u32,
) -> serde_json::Value {
    let mut subtree = registration_subtree_with_measure_version(cfg, measure_version);
    subtree["alignVersion"] = serde_json::json!(align_version);
    subtree
}

/// [`registration_subtree`] with the measure version supplied and no
/// `alignVersion` — see [`measurement_subtree_with_measure_version`].
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn registration_subtree_with_measure_version(
    cfg: &AlgorithmConfig,
    measure_version: u32,
) -> serde_json::Value {
    serde_json::json!({
        "registration": cfg.registration,
        "measureVersion": measure_version,
    })
}

/// Stage 6 (local normalization, M2) config subtree: the whole
/// `normalization` block (global output/rejection choice AND the `local`
/// block — a change to either invalidates a stored `.athln`/reference; this
/// already covers the PSF model [`crate::stacking::ln::normalize_frame`]
/// actually hands [`crate::stacking::ln::scale::relative_scale`], which is
/// `normalization.local.psfModel`, never `measurement.psfModel`) plus
/// `measurement.maxStars` — the ONE genuinely extra fold-in, since it is the
/// detection/PSF-fit budget `relative_scale` uses for both planes but lives
/// under `measurement`, not `normalization`. `measurement.psfModel` rides
/// along too (harmless — it just widens what invalidates a sidecar — but
/// plays no role in `relative_scale`'s own model choice; fix round 1, item
/// 11: an earlier version of this comment claimed otherwise), plus
/// [`crate::star_measure::MEASURE_VERSION`]: `relative_scale` fits
/// its matched stars through the same `fit_stars` /
/// `FitParams::default()` the measurement uses, so a change to what the
/// fitter accepts silently changes every `.athln` — and no config field
/// moves when it does (ruling R-M4a-15). It also carries
/// [`crate::stacking::ln::LN_MODEL_VERSION`] (`lnModelVersion`): a change in
/// what an existing config's LN model produces moves the hashes the same way.
pub fn normalization_subtree(cfg: &AlgorithmConfig) -> serde_json::Value {
    let mut subtree = normalization_subtree_with_measure_version(cfg, MEASURE_VERSION);
    fold_measure_stage_version(&mut subtree, MEASURE_STAGE_VERSION);
    subtree
}

/// [`normalization_subtree`] with the measure version supplied — see
/// [`measurement_subtree_with_measure_version`].
// doc(hidden): public only for Athenaeum core's pinned hash tests; a stage hash built at another version
// is not API, so the CLI never calls it.
#[doc(hidden)]
pub fn normalization_subtree_with_measure_version(
    cfg: &AlgorithmConfig,
    measure_version: u32,
) -> serde_json::Value {
    serde_json::json!({
        "normalization": canonical_normalization(cfg),
        "measurement": {
            "psfModel": cfg.measurement.psf_model,
            "maxStars": cfg.measurement.max_stars,
        },
        "measureVersion": measure_version,
        "lnModelVersion": crate::stacking::ln::LN_MODEL_VERSION,
    })
}

/// The normalization block as the LN hashes see it (L-19): `scale`,
/// `scaleMode` and `gridSize` leave it - their effect reaches the hashes as
/// the resolved-scale token - so auto <-> fixed at the same resolved value
/// invalidates nothing.
fn canonical_normalization(cfg: &AlgorithmConfig) -> serde_json::Value {
    let mut v = serde_json::to_value(&cfg.normalization).expect("NormalizationConfig serializes");
    if let Some(local) = v.get_mut("local").and_then(|l| l.as_object_mut()) {
        for key in ["scale", "scaleMode", "gridSize"] {
            local.remove(key);
        }
        // The aperture only matters under the common aperture (spec §7).
        if cfg.normalization.local.flux_estimator
            != crate::stacking::integrate::LnFluxEstimator::CommonAperture
        {
            local.remove("aperture");
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whole Athenaeum stacking configuration document parses into the algorithm sections: the other sections
    /// (grouping, reference, output, paths and the version) are ignored, and the seven it reads come out as the defaults.
    #[test]
    fn a_whole_default_stacking_document_parses_into_the_default_algorithm_config() {
        let text = include_str!("../../tests/fixtures/stacking-config-default.json");
        let parsed: AlgorithmConfig = serde_json::from_str(text).expect("the fixture parses");
        assert_eq!(parsed, AlgorithmConfig::default());
        assert_eq!(algorithm_preset(StackingPreset::Default), AlgorithmConfig::default());
    }

    #[test]
    fn the_presets_differ_from_the_default_where_they_say_they_do() {
        let fast = algorithm_preset(StackingPreset::FastPreview);
        assert_eq!(fast.registration.interpolation, Interpolation::Bilinear);
        assert_eq!(fast.normalization.rejection, RejectionNormalization::ScaleZeroOffset);
        let max = algorithm_preset(StackingPreset::MaximumQuality);
        assert!(max.drizzle.enabled && max.drizzle.scale == 2);
        assert!(max.normalization.local.enabled && max.integration.write_rejection_maps);
        assert_eq!(max.registration.distortion, DistortionChoice::Polynomial3);
    }

    #[test]
    fn the_calibration_subtree_drops_the_output_format_and_pins_the_grouping_tolerance() {
        let mut cfg = AlgorithmConfig::default();
        let subtree = calibration_subtree(&cfg);
        assert!(subtree["calibration"].get("format").is_none());
        assert_eq!(
            subtree["grouping"]["exposureToleranceSec"],
            serde_json::json!(CALIBRATION_HASH_GROUPING_TOLERANCE_SEC)
        );
        cfg.calibration.format = crate::fits_writer::OutputFormat::Xisf;
        assert_eq!(calibration_subtree(&cfg), subtree, "the container moves no stage hash");
    }

    #[test]
    fn a_measure_stage_version_other_than_one_is_folded_into_a_subtree() {
        let mut subtree = serde_json::json!({});
        fold_measure_stage_version(&mut subtree, 1);
        assert!(subtree.get("measureStageVersion").is_none());
        fold_measure_stage_version(&mut subtree, 2);
        assert_eq!(subtree["measureStageVersion"], serde_json::json!(2));
    }
}
