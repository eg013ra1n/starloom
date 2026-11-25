//! The cards every file the pipeline writes carries (P2-R6). A stage adds its own on top of these.

use anyhow::{anyhow, Result};

use crate::fits_writer::{Card, CardValue};
use crate::pipeline::versions::{all_versions, VERSION_CARDS};
use crate::pipeline::PIPELINE_VERSION;

/// The starloom version that wrote the file.
pub const SLM_VER: &str = "SLM_VER";
/// The hash of the whole effective config ([`config_hash`](crate::pipeline::config_hash)) the file was made with.
pub const SLM_CFG: &str = "SLM_CFG";
/// The pipeline version ([`PIPELINE_VERSION`]) that wrote the file.
pub const SLM_PLV: &str = "SLM_PLV";

/// The recipe a master calibration frame was integrated with, as Athenaeum writes it in `ATH_REJ`: the recipe's own
/// description and the frame count (`Average | Winsorized sigma (3.0/3.0) n=24`).
pub const SLM_REC: &str = "SLM_REC";
/// The number of frames a master calibration frame integrates.
pub const SLM_N: &str = "SLM_N";
/// The lowest member `CCD-TEMP` of a master calibration frame, in degrees Celsius.
pub const SLM_TMIN: &str = "SLM_TMIN";
/// The highest member `CCD-TEMP` of a master calibration frame, in degrees Celsius.
pub const SLM_TMAX: &str = "SLM_TMAX";
/// The file name of the light a calibrated light was made from, percent-encoded as Athenaeum encodes it (the twin of
/// `ATH_CSRN`).
pub const SLM_CSRN: &str = "SLM_CSRN";
/// The file name of the master dark a calibrated light was made with (the twin of `ATH_CDRK`; no catalog id).
pub const SLM_CDRK: &str = "SLM_CDRK";
/// The file name of the master flat a calibrated light was made with (the twin of `ATH_CFLT`).
pub const SLM_CFLT: &str = "SLM_CFLT";
/// The file name of the master bias a calibrated light was made with, present only when no dark applied (the twin of
/// `ATH_CBIA`).
pub const SLM_CBIA: &str = "SLM_CBIA";
/// The flat-normalization divisor a calibrated light was divided by, `1.0` when off; in per-channel mode the mosaic
/// weighted blend of the three (the twin of `ATH_CFNM`).
pub const SLM_CFNM: &str = "SLM_CFNM";
/// The output pedestal, in DN, a calibrated light carries (always written; the twin of `ATH_CPED`).
pub const SLM_CPED: &str = "SLM_CPED";
/// Whether the flat was normalized per CFA channel (the twin of `ATH_CCFA`; written only when it was).
pub const SLM_CCFA: &str = "SLM_CCFA";
/// The red per-channel flat-normalization divisor (the twin of `ATH_CFNR`).
pub const SLM_CFNR: &str = "SLM_CFNR";
/// The green per-channel flat-normalization divisor (the twin of `ATH_CFNG`).
pub const SLM_CFNG: &str = "SLM_CFNG";
/// The blue per-channel flat-normalization divisor (the twin of `ATH_CFNB`).
pub const SLM_CFNB: &str = "SLM_CFNB";
/// The per-tail trim fraction of the whole-frame trimmed-mean flat normalization (the twin of `ATH_CTRM`; written only
/// in that mode with a flat applied).
pub const SLM_CTRM: &str = "SLM_CTRM";
/// The number of hot pixels the cosmetic pass replaced in a calibrated light (present when the pass ran).
pub const SLM_CHPX: &str = "SLM_CHPX";
/// The debayer a calibrated light went through (`VNG`; present only after a debayer).
pub const SLM_CDBM: &str = "SLM_CDBM";

/// The version card named `card` (one of [`VERSION_CARDS`], e.g. `SLM_LCV`) holding the version constant it stands for.
pub fn version_card(card: &str) -> Result<Card> {
    let (_, key) = VERSION_CARDS
        .iter()
        .find(|(name, _)| *name == card)
        .ok_or_else(|| anyhow!("{card} is not a version card"))?;
    let value = all_versions().get(key).copied().ok_or_else(|| anyhow!("{card} names an unknown version {key}"))?;
    Ok(Card::new(card, CardValue::Integer(i64::from(value)))?.with_comment(key))
}

/// The three cards of every pipeline file: [`SLM_VER`] (this crate's version), [`SLM_CFG`] (`cfg_hash`, from
/// [`config_hash`](crate::pipeline::config_hash)) and [`SLM_PLV`] ([`PIPELINE_VERSION`]), in that order.
pub fn common_cards(cfg_hash: &str) -> Result<Vec<Card>> {
    Ok(vec![
        Card::new(SLM_VER, CardValue::Str(env!("CARGO_PKG_VERSION").to_string()))?.with_comment("starloom version"),
        Card::new(SLM_CFG, CardValue::Str(cfg_hash.to_string()))?.with_comment("effective config hash"),
        Card::new(SLM_PLV, CardValue::Integer(i64::from(PIPELINE_VERSION)))?.with_comment("pipeline version"),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_card_holds_the_constant_it_names() {
        let card = version_card("SLM_LCV").unwrap();
        assert!(matches!(card.value, Some(CardValue::Integer(v)) if v == i64::from(crate::calibration_library::LIGHT_CAL_VERSION)));
        assert!(version_card("SLM_XXX").is_err());
    }

    #[test]
    fn every_file_carries_the_version_the_config_hash_and_the_pipeline_version() {
        let cards = common_cards("0123456789abcdef").unwrap();
        let keys: Vec<&str> = cards.iter().map(|c| c.keyword.as_str()).collect();
        assert_eq!(keys, ["SLM_VER", "SLM_CFG", "SLM_PLV"]);
        assert!(matches!(&cards[0].value, Some(CardValue::Str(v)) if v == env!("CARGO_PKG_VERSION")));
        assert!(matches!(&cards[1].value, Some(CardValue::Str(v)) if v == "0123456789abcdef"));
        assert!(matches!(&cards[2].value, Some(CardValue::Integer(1))));
    }
}
