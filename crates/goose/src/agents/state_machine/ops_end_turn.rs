//! Ends the turn after a tool result that asks to end it without a message.
//!
//! summon's `wait` lets an event-driven parent sleep until the next delegated
//! task report or question without writing to the user. The default agent loop
//! honors the same `goose.endTurn` tool-result metadata.

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::platform_extensions::summon::tool_result_ends_turn;
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
    messages
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
}
