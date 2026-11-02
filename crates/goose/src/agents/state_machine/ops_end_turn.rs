//! Ends the turn after a tool result that asks to end it without a message.
//!
//! summon's `wait` lets an event-driven parent sleep until the next delegated
//! task report or question without writing to the user. The default agent loop
//! honors the same `goose.endTurn` tool-result metadata.

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::platform_extensions::summon::{
    delegation_batch_ends_turn, tool_result_ends_turn,
};
use crate::agents::state_machine::effects::GooseEffect;
use crate::agents::state_machine::{not_applicable, yielded, Emitter, Operation, OperationResult};
use crate::conversation::message::{Message, MessageContent};
use crate::conversation::Conversation;
use crate::session::Session;

pub struct EndTurnOperation;

/// True when the tool results that close the latest assistant turn include one
/// that ends the turn. Registered after tool execution, so every tool call in
/// that turn already has its result here.
fn trailing_results_end_turn(messages: &[Message]) -> bool {
    delegation_batch_ends_turn(messages)
        || messages
            .iter()
            .rev()
            .take_while(|message| message.is_tool_response())
            .flat_map(|message| message.content.iter())
            .any(|content| {
                matches!(content, MessageContent::ToolResponse(response)
            if response.tool_result.as_ref().is_ok_and(tool_result_ends_turn))
            })
}

#[async_trait]
impl Operation<Session, GooseEffect> for EndTurnOperation {
    fn name(&self) -> &'static str {
        "end_turn"
    }

    async fn run(
        &self,
        _session: &Session,
        conversation: &Conversation,
        _emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if !trailing_results_end_turn(conversation.messages()) {
            return not_applicable();
        }
        yielded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::platform_extensions::summon::END_TURN_META_KEY;
    use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, MetaObject};

    fn result(ends_turn: bool) -> CallToolResult {
        let mut result = CallToolResult::success(vec![ContentBlock::text("ok")]);
        if ends_turn {
            result.meta = Some(MetaObject(
                serde_json::json!({ END_TURN_META_KEY: true })
                    .as_object()
                    .unwrap()
                    .clone(),
            ));
        }
        result
    }

    fn turn(ends_turn: bool) -> Vec<Message> {
        vec![
            Message::user().with_text("Make the deck"),
            Message::assistant().with_tool_request("call", Ok(CallToolRequestParams::new("wait"))),
            Message::user().with_tool_response("call", Ok(result(ends_turn))),
        ]
    }

    #[test]
    fn a_wait_result_ends_the_turn_until_a_new_message_arrives() {
        assert!(trailing_results_end_turn(&turn(true)));
        assert!(!trailing_results_end_turn(&turn(false)));

        // A report or user message after the wait starts a new turn.
        let mut resumed = turn(true);
        resumed.push(Message::user().with_text("Internal delegated-task reports follow."));
        assert!(!trailing_results_end_turn(&resumed));
    }

    #[test]
    fn only_complete_event_driven_delegation_batches_park() {
        let mut admitted = result(false);
        admitted.structured_content = Some(serde_json::json!({
            "task_status": "running", "taskAdmission": {"task_id": "task"},
            "automatic_report_delivery": true,
        }));
        // Admission is authoritative even when the caller did not ask for async.
        let mut messages = vec![
            Message::assistant()
                .with_tool_request(
                    "a",
                    Ok(CallToolRequestParams::new("delegate").with_arguments(
                        serde_json::json!({"yield_after_delegation": true})
                            .as_object()
                            .unwrap()
                            .clone(),
                    )),
                )
                .with_tool_request("b", Ok(CallToolRequestParams::new("delegate"))),
            Message::user().with_tool_response("a", Ok(admitted.clone())),
            Message::user().with_tool_response("b", Ok(admitted.clone())),
        ];
        assert!(delegation_batch_ends_turn(&messages));
        assert!(trailing_results_end_turn(&messages));
        let ready = messages[0].clone();
        messages[0] = Message::assistant()
            .with_tool_request("a", Ok(CallToolRequestParams::new("delegate")))
            .with_tool_request("b", Ok(CallToolRequestParams::new("delegate")));
        assert!(!trailing_results_end_turn(&messages)); // More independent work remains.
        messages[0] = ready.clone();
        let last = messages.pop().unwrap();
        assert!(!trailing_results_end_turn(&messages)); // Still awaiting a result.
        messages.push(Message::user().with_tool_response("b", Ok(result(false))));
        assert!(!trailing_results_end_turn(&messages)); // Synchronous/failed admission.
        messages.pop();
        messages.push(last);
        messages.push(Message::user().with_text("The task completed before parking."));
        assert!(!trailing_results_end_turn(&messages));
        messages.pop();
        messages[0] = Message::assistant()
            .with_tool_request("a", Ok(CallToolRequestParams::new("delegate")))
            .with_tool_request("b", Ok(CallToolRequestParams::new("read_file")));
        assert!(!trailing_results_end_turn(&messages));
        messages[0] = ready;
        admitted.is_error = Some(true);
        messages[2] = Message::user().with_tool_response("b", Ok(admitted));
        assert!(!trailing_results_end_turn(&messages));
    }
}
