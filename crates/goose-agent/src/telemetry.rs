use goose_provider_types::conversation::message::{Message, MessageContent, ToolResult};
use goose_provider_types::formats::openai::{tool_error_text, tool_result_text};
use rmcp::model::{CallToolRequestParams, Role, Tool};
use serde_json::{json, Value};

/// The request's messages as the model receives them: a tool result is the text the
/// provider sends (never its structured content or `_meta`), and parts the provider
/// drops (notifications, errors, confirmations) are left out.
pub fn input_messages_with_system_json(
    system_prompt: &str,
    messages: &[Message],
    supports_vision: bool,
) -> String {
    let mut values = Vec::with_capacity(messages.len() + 1);
    values.push(json!({
        "role": "system",
        "parts": [{"type": "text", "content": system_prompt}],
    }));
    values.extend(
        messages
            .iter()
            .map(|message| message_json(message, supports_vision))
            .filter(|message| {
                message["parts"]
                    .as_array()
                    .is_some_and(|parts| !parts.is_empty())
            }),
    );
    Value::Array(values).to_string()
}

/// The tools offered to the model, as the GenAI semantic convention's
/// `gen_ai.tool.definitions`: name, description and input schema, in the order
/// sent. The model sees these; the Langfuse exporter puts them in the generation's
/// input next to its messages.
pub fn tool_definitions_json(tools: &[Tool]) -> String {
    Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name.as_ref(),
                    "description": tool.description.as_deref(),
                    "parameters": Value::Object(tool.input_schema.as_ref().clone()),
                })
            })
            .collect(),
    )
    .to_string()
}

pub fn system_instructions_json(system_prompt: &str) -> String {
    json!([{"type": "text", "content": system_prompt}]).to_string()
}

pub fn output_message_json(message: &Message) -> String {
    // Message does not retain provider finish reasons; tool requests are the only
    // distinct completion signal available after streaming.
    let finish_reason = if message
        .content
        .iter()
        .any(|content| matches!(content, MessageContent::ToolRequest(_)))
    {
        "tool_call"
    } else {
        "stop"
    };
    let mut value = message_json(message, false);
    value["finish_reason"] = Value::String(finish_reason.to_string());
    Value::Array(vec![value]).to_string()
}

pub fn append_message(accumulated: &mut Option<Message>, message: &Message) {
    match accumulated {
        Some(accumulated) => accumulated.content.extend(message.content.iter().cloned()),
        None => *accumulated = Some(message.clone()),
    }
}

fn message_json(message: &Message, supports_vision: bool) -> Value {
    let role = if !message.content.is_empty()
        && message
            .content
            .iter()
            .all(|content| matches!(content, MessageContent::ToolResponse(_)))
    {
        "tool"
    } else {
        match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    };

    let parts = consolidated_parts(&message.content, supports_vision);
    json!({
        "role": role,
        "parts": parts,
    })
}

/// Merge consecutive text and reasoning parts into single entries so that
/// streaming tokens don't each get their own JSON object in the OTEL output.
fn consolidated_parts(content: &[MessageContent], supports_vision: bool) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    for item in content {
        let Some(value) = message_part_json(item, supports_vision) else {
            continue;
        };
        let item_type = value.get("type").and_then(|v| v.as_str());
        if matches!(item_type, Some("text" | "reasoning")) {
            if let Some(last) = result.last_mut() {
                if last.get("type") == value.get("type") {
                    if let (Some(existing), Some(new_content)) = (
                        last.get("content").and_then(|v| v.as_str()),
                        value.get("content").and_then(|v| v.as_str()),
                    ) {
                        last["content"] = Value::String(format!("{}{}", existing, new_content));
                        continue;
                    }
                }
            }
        }
        result.push(value);
    }
    result
}

fn tool_call_part(id: &str, tool_call: &ToolResult<CallToolRequestParams>) -> Value {
    match tool_call {
        Ok(tool_call) => json!({
            "type": "tool_call",
            "id": id,
            "name": tool_call.name,
            "arguments": tool_call
                .arguments
                .as_ref()
                .map(|arguments| Value::Object(arguments.clone()))
                .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
        }),
        Err(error) => json!({
            "type": "tool_call_error",
            "id": id,
            "error": error.to_string(),
        }),
    }
}

/// One message part as the provider sends it, or None for a part it never sends.
fn message_part_json(content: &MessageContent, supports_vision: bool) -> Option<Value> {
    Some(match content {
        MessageContent::Text(text) => json!({
            "type": "text",
            "content": text.text,
        }),
        MessageContent::Image(image) => json!({
            "type": "blob",
            "modality": "image",
            "mime_type": image.mime_type,
            "content": image.data,
        }),
        MessageContent::Document(document) => json!({
            "type": "blob",
            "modality": "document",
            "mime_type": document.mime_type,
            "name": document.name,
            "content": document.data,
        }),
        MessageContent::ToolRequest(request) => tool_call_part(&request.id, &request.tool_call),
        MessageContent::ToolResponse(response) => json!({
            "type": "tool_call_response",
            "id": response.id,
            "response": match &response.tool_result {
                Ok(result) => tool_result_text(result, supports_vision),
                Err(error) => tool_error_text(&error.to_string()),
            },
        }),
        MessageContent::Thinking(thinking) => json!({
            "type": "reasoning",
            "content": thinking.thinking,
        }),
        MessageContent::RedactedThinking(_)
        | MessageContent::ToolConfirmationRequest(_)
        | MessageContent::ActionRequired(_)
        | MessageContent::SystemNotification(_)
        | MessageContent::Error(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_tool_result_is_logged_as_the_text_the_model_receives() {
        let mut result = rmcp::model::CallToolResult::success(vec![
            rmcp::model::ContentBlock::text("{\"answer\": 1}"),
            rmcp::model::ContentBlock::image("aGk=", "image/png"),
        ]);
        result.structured_content = Some(json!({"answer": 1}));
        let messages = vec![
            Message::user().with_tool_response("call-1", Ok(result.clone())),
            Message::user().with_tool_response(
                "call-2",
                Err(rmcp::model::ErrorData::invalid_params("bad input", None)),
            ),
            Message::user().with_system_notification(
                goose_provider_types::conversation::message::SystemNotificationType::InlineMessage,
                "only the user sees this",
            ),
        ];

        let value: Value =
            serde_json::from_str(&input_messages_with_system_json("System", &messages, false))
                .unwrap();

        // The provider sends text only: structured content and _meta never reach the model.
        assert_eq!(
            value[1]["parts"][0]["response"],
            json!(tool_result_text(&result, false))
        );
        assert!(value[1]["parts"][0]["response"]
            .as_str()
            .unwrap()
            .starts_with("{\"answer\": 1} This tool result included an image that was omitted"));
        assert!(!value.to_string().contains("structuredContent"));
        assert_eq!(
            value[2]["parts"][0]["response"],
            json!(tool_error_text(
                &rmcp::model::ErrorData::invalid_params("bad input", None).to_string()
            ))
        );
        // A notification the provider never sends is left out, with its now-empty message.
        assert_eq!(value.as_array().unwrap().len(), 3);
    }

    #[test]
    fn tool_definitions_keep_name_description_and_schema_in_order() {
        let schema = json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let tools = vec![
            Tool::new(
                "search".to_string(),
                "Search the knowledge base.".to_string(),
                Arc::new(schema.as_object().unwrap().clone()),
            ),
            Tool::new(
                "read".to_string(),
                "Read one result.".to_string(),
                Arc::new(json!({"type": "object"}).as_object().unwrap().clone()),
            ),
        ];
        let definitions: Value = serde_json::from_str(&tool_definitions_json(&tools)).unwrap();
        assert_eq!(
            definitions,
            json!([
                {"type": "function", "name": "search", "description": "Search the knowledge base.", "parameters": schema},
                {"type": "function", "name": "read", "description": "Read one result.", "parameters": {"type": "object"}},
            ])
        );
    }
}
