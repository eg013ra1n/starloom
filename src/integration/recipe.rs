//! The recipe a master calibration frame is integrated with when the caller names none.

use super::combine::{IntegrationRecipe, Rejection};

/// spec §2/§9: bias-like N>=15 Average+Winsorized else Median; flat N>=15
/// Average+Winsorized else Average+Percentile. Explicit override always wins.
///
/// `imagetyp` is the calibration frame type (`"Flat"` takes the flat rule; any other value takes the bias-like one) and
/// `n` the number of frames.
pub fn resolve_recipe(
    explicit: Option<IntegrationRecipe>,
    imagetyp: &str,
    n: i64,
) -> IntegrationRecipe {
    if let Some(recipe) = explicit {
        return recipe;
    }
    let is_flat = imagetyp == "Flat";
    if n >= 15 {
        IntegrationRecipe::average(Rejection::WinsorizedSigma {
            sigma_low: 3.0,
            sigma_high: 3.0,
        })
    } else if is_flat {
        IntegrationRecipe::average(Rejection::PercentileClip {
            low: 0.2,
            high: 0.02,
        })
    } else {
        IntegrationRecipe::median(Rejection::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_recipe_rules() {
        // spec §2/§9: bias-like N>=15 Average+Winsorized else Median; flat
        // N>=15 Average+Winsorized else Average+Percentile.
        assert_eq!(
            resolve_recipe(None, "Dark", 20),
            IntegrationRecipe::average(Rejection::WinsorizedSigma {
                sigma_low: 3.0,
                sigma_high: 3.0
            })
        );
        assert_eq!(
            resolve_recipe(None, "Dark", 5),
            IntegrationRecipe::median(Rejection::None)
        );
        assert_eq!(
            resolve_recipe(None, "Bias", 14),
            IntegrationRecipe::median(Rejection::None)
        );
        assert_eq!(
            resolve_recipe(None, "Flat", 20),
            IntegrationRecipe::average(Rejection::WinsorizedSigma {
                sigma_low: 3.0,
                sigma_high: 3.0
            })
        );
        assert_eq!(
            resolve_recipe(None, "Flat", 6),
            IntegrationRecipe::average(Rejection::PercentileClip {
                low: 0.2,
                high: 0.02
            })
        );
        // explicit override wins
        let override_recipe = IntegrationRecipe::average(Rejection::None);
        assert_eq!(
            resolve_recipe(Some(override_recipe), "Flat", 6),
            override_recipe
        );
    }
}
