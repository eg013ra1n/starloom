//! One vocabulary for the versions a working folder records.
//!
//! Every manifest entry and every report records [`all_versions`]: each version constant of this build under the
//! constant's own name, as `tests/stage_goldens.rs` spells it (`ALIGN_VERSION`, never `align`). They are a record for
//! the user to read (P2-R24): nothing compares them, and a product made by another build is used as it is.

use std::collections::BTreeMap;

use crate::calibration_library::LIGHT_CAL_VERSION;
use crate::integration::engine::ENGINE_VERSION;
use crate::pipeline::formats::{MANIFEST_VERSION, MEASURE_REPORT_VERSION, TRANSFORMS_VERSION, WEIGHTS_VERSION};
use crate::pipeline::PIPELINE_VERSION;
use crate::stacking::config::DRIZZLE_KERNEL_VERSION;
use crate::stacking::fits_artifact::FITS_ARTIFACT_VERSION;
use crate::stacking::integrate::{INTEGRATE_VERSION, SNR_GAIN_VERSION};
use crate::stacking::ln::grid::LN_GRID_VERSION;
use crate::stacking::ln::LN_MODEL_VERSION;
use crate::stacking::measure::MEASURE_STAGE_VERSION;
use crate::stacking::pass1_artifact::PASS1_ARTIFACT_VERSION;
use crate::stacking::register::align::ALIGN_VERSION;
use crate::star_measure::MEASURE_VERSION;

/// Every version of this build, by its constant's name: the twelve stage and format versions the stage goldens are
/// keyed on, the four working-folder format versions and [`PIPELINE_VERSION`].
pub fn all_versions() -> BTreeMap<&'static str, u32> {
    BTreeMap::from([
        ("ALIGN_VERSION", ALIGN_VERSION),
        ("DRIZZLE_KERNEL_VERSION", DRIZZLE_KERNEL_VERSION),
        ("ENGINE_VERSION", ENGINE_VERSION),
        ("FITS_ARTIFACT_VERSION", u32::from(FITS_ARTIFACT_VERSION)),
        ("INTEGRATE_VERSION", INTEGRATE_VERSION),
        ("LIGHT_CAL_VERSION", LIGHT_CAL_VERSION),
        ("LN_GRID_VERSION", LN_GRID_VERSION),
        ("LN_MODEL_VERSION", LN_MODEL_VERSION),
        ("MANIFEST_VERSION", MANIFEST_VERSION),
        ("MEASURE_REPORT_VERSION", MEASURE_REPORT_VERSION),
        ("MEASURE_STAGE_VERSION", MEASURE_STAGE_VERSION),
        ("MEASURE_VERSION", MEASURE_VERSION),
        ("PASS1_ARTIFACT_VERSION", u32::from(PASS1_ARTIFACT_VERSION)),
        ("PIPELINE_VERSION", PIPELINE_VERSION),
        ("SNR_GAIN_VERSION", SNR_GAIN_VERSION),
        ("TRANSFORMS_VERSION", TRANSFORMS_VERSION),
        ("WEIGHTS_VERSION", WEIGHTS_VERSION),
    ])
}

/// [`all_versions`] in the form a manifest entry or a report stores it.
pub fn recorded_versions() -> BTreeMap<String, u32> {
    all_versions().into_iter().map(|(name, v)| (name.to_string(), v)).collect()
}

/// The version cards a product carries (P2-R6), each with the [`all_versions`] key whose value it holds.
pub const VERSION_CARDS: [(&str, &str); 8] = [
    ("SLM_MSV", "MEASURE_STAGE_VERSION"),
    ("SLM_MV", "MEASURE_VERSION"),
    ("SLM_ALV", "ALIGN_VERSION"),
    ("SLM_LNV", "LN_MODEL_VERSION"),
    ("SLM_INV", "INTEGRATE_VERSION"),
    ("SLM_DZV", "DRIZZLE_KERNEL_VERSION"),
    ("SLM_LCV", "LIGHT_CAL_VERSION"),
    ("SLM_PLV", "PIPELINE_VERSION"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_version_name_holds_its_constant_and_every_version_card_names_one() {
        let expected: [(&str, u32); 17] = [
            ("ALIGN_VERSION", ALIGN_VERSION),
            ("DRIZZLE_KERNEL_VERSION", DRIZZLE_KERNEL_VERSION),
            ("ENGINE_VERSION", ENGINE_VERSION),
            ("FITS_ARTIFACT_VERSION", u32::from(FITS_ARTIFACT_VERSION)),
            ("INTEGRATE_VERSION", INTEGRATE_VERSION),
            ("LIGHT_CAL_VERSION", LIGHT_CAL_VERSION),
            ("LN_GRID_VERSION", LN_GRID_VERSION),
            ("LN_MODEL_VERSION", LN_MODEL_VERSION),
            ("MANIFEST_VERSION", MANIFEST_VERSION),
            ("MEASURE_REPORT_VERSION", MEASURE_REPORT_VERSION),
            ("MEASURE_STAGE_VERSION", MEASURE_STAGE_VERSION),
            ("MEASURE_VERSION", MEASURE_VERSION),
            ("PASS1_ARTIFACT_VERSION", u32::from(PASS1_ARTIFACT_VERSION)),
            ("PIPELINE_VERSION", PIPELINE_VERSION),
            ("SNR_GAIN_VERSION", SNR_GAIN_VERSION),
            ("TRANSFORMS_VERSION", TRANSFORMS_VERSION),
            ("WEIGHTS_VERSION", WEIGHTS_VERSION),
        ];
        assert_eq!(all_versions(), BTreeMap::from(expected));
        assert_eq!(recorded_versions().len(), expected.len());
        let all = all_versions();
        for (card, name) in VERSION_CARDS {
            assert!(all.contains_key(name), "{card} names {name}, which all_versions lacks");
            assert!(card.starts_with("SLM_") && card.len() <= 8, "{card} is not a FITS keyword of ours");
        }
        let cards: std::collections::BTreeSet<&str> = VERSION_CARDS.iter().map(|(c, _)| *c).collect();
        assert_eq!(cards.len(), VERSION_CARDS.len(), "a card is listed twice");
        assert!(VERSION_CARDS.contains(&(crate::pipeline::SLM_PLV, "PIPELINE_VERSION")));
    }
}
