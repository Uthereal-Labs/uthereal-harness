//! Deterministic fixes for the coordinator's and specialists' own tool calls.
//!
//! A fix runs only when the correction is unambiguous, and the tool result
//! tells the model what was fixed ("Automatic correction: ..."). Anything else
//! is an error the model corrects. Editor fixes live with the editors; every
//! fix is listed in uthereal-src docs/engineering/cortex-deterministic-fixes.md.
//!
//! Coordinator (`delegate`):
//! - previous_task_id with no earlier task for the artifact in this turn: the
//!   value is dropped and the delegation proceeds. It names a follow-up on an
//!   earlier task, so with no earlier task there is nothing it could mean.
//!   A value naming the wrong task is an error that names the right one.
//!
//! Specialists: none. Their editor tools (uthereal-src
//! uth/agents/apps/cortex/harness/mcp.py) only validate and return errors.

use rmcp::model::ContentBlock;

/// Decides what to do with a previous_task_id given the earlier tasks that own
/// the artifact in this session. `Ok(None)` keeps it, `Ok(Some(note))` drops it
/// with the note to report, and `Err` rejects it with the correct task ID.
pub(crate) fn previous_task_id_for_artifact(
    previous: &str,
    owners: &[String],
) -> Result<Option<String>, String> {
    if owners.is_empty() {
        return Ok(Some(format!(
            "previous_task_id \"{previous}\" was ignored automatically: it is only for a follow-up on a task you delegated earlier in this turn, and no earlier task exists for this artifact."
        )));
    }
    if owners.iter().any(|owner| owner == previous) {
        return Ok(None);
    }
    Err(format!(
        "previous_task_id \"{previous}\" is not the earlier task for this artifact. Use previous_task_id: \"{}\".",
        owners[0]
    ))
}

/// Appends an automatic-correction note to a delegate result.
pub(crate) fn with_correction(
    mut content: Vec<ContentBlock>,
    correction: Option<String>,
) -> Vec<ContentBlock> {
    if let Some(correction) = correction {
        content.push(ContentBlock::text(format!(
            "Automatic correction: {correction}"
        )));
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previous_task_id_is_dropped_only_when_no_earlier_task_exists() {
        let note = previous_task_id_for_artifact("20261002_1", &[]).unwrap();
        assert!(note.unwrap().contains("ignored automatically"));
        let owners = vec!["20261002_2".to_string()];
        assert_eq!(
            previous_task_id_for_artifact("20261002_2", &owners),
            Ok(None)
        );
        let error = previous_task_id_for_artifact("20261002_1", &owners).unwrap_err();
        assert!(error.contains("Use previous_task_id: \"20261002_2\""));
    }

    #[test]
    fn a_correction_is_appended_to_the_result() {
        let content = with_correction(vec![], Some("dropped".to_string()));
        assert_eq!(content.len(), 1);
        assert!(with_correction(vec![], None).is_empty());
    }
}
