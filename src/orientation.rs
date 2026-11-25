//! The canonical `ROWORDER` spellings, which a writer stamps when it has to state the row order explicitly, and the rule
//! that reads a `ROWORDER` value.

/// The canonical `ROWORDER` spellings, so a writer that has to state the
/// order explicitly (the XISF writer) spells it the same way the row-order
/// rule reads it.
pub const ROW_ORDER_TOP_DOWN: &str = "TOP-DOWN";
/// The spelling of a bottom-up row order (row 0 is the bottom row of the sky image), the `ROWORDER` card value.
pub const ROW_ORDER_BOTTOM_UP: &str = "BOTTOM-UP";

/// The row-order rule over an already-read `ROWORDER` value — the ONE
/// place the table in the module docs is encoded (M4d Task 2 fix round 1,
/// ruling R-T2-1: the XISF writer and the stacking run both need this
/// answer over a card they already hold, and a second copy of the rule is
/// exactly how a 180° flip goes unnoticed).
///
/// `TOP-DOWN` → `false`; `BOTTOM-UP`, an unrecognized explicit value, and a
/// missing card → `true`, the astronomical default. Case-insensitive, and
/// tolerant of the surrounding quotes some writers leave in the value
/// despite the FITS card already delimiting it.
pub fn row_order_is_bottom_up(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let normalized = value
                .trim()
                .trim_matches('\'')
                .trim()
                .to_ascii_uppercase();
            match normalized.as_str() {
                s if s == ROW_ORDER_TOP_DOWN => false,
                s if s == ROW_ORDER_BOTTOM_UP => true,
                // Unknown explicit value — assume astronomical default.
                _ => true,
            }
        }
        // Missing ROWORDER → astronomical default = bottom-up = flip.
        None => true,
    }
}

