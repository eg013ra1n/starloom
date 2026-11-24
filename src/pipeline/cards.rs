//! The cards every file the pipeline writes carries (P2-R6). A stage adds its own on top of these.

use anyhow::Result;

use crate::fits_writer::{Card, CardValue};
use crate::pipeline::PIPELINE_VERSION;

/// The starloom version that wrote the file.
pub const SLM_VER: &str = "SLM_VER";
/// The hash of the whole effective config ([`config_hash`](crate::pipeline::config_hash)) the file was made with.
pub const SLM_CFG: &str = "SLM_CFG";
/// The pipeline version ([`PIPELINE_VERSION`]) that wrote the file.
pub const SLM_PLV: &str = "SLM_PLV";

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
    fn every_file_carries_the_version_the_config_hash_and_the_pipeline_version() {
        let cards = common_cards("0123456789abcdef").unwrap();
        let keys: Vec<&str> = cards.iter().map(|c| c.keyword.as_str()).collect();
        assert_eq!(keys, ["SLM_VER", "SLM_CFG", "SLM_PLV"]);
        assert!(matches!(&cards[0].value, Some(CardValue::Str(v)) if v == env!("CARGO_PKG_VERSION")));
        assert!(matches!(&cards[1].value, Some(CardValue::Str(v)) if v == "0123456789abcdef"));
        assert!(matches!(&cards[2].value, Some(CardValue::Integer(1))));
    }
}
