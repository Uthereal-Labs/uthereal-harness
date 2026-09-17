use goose_provider_types::conversation::message::{Message, MessageContent, ToolResult};
use rmcp::model::{CallToolRequestParams, Role};
use serde_json::{json, Value};

pub fn input_messages_with_system_json(system_prompt: &str, messages: &[Message]) -> String {
    let mut values = Vec::with_capacity(messages.len() + 1);
    values.push(json!({
        "role": "system",
        "parts": [{"type": "text", "content": system_prompt}],
    }));
    values.extend(messages.iter().map(message_json));
    Value::Array(values).to_string()
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
    let mut value = message_json(message);
    value["finish_reason"] = Value::String(finish_reason.to_string());
    Value::Array(vec![value]).to_string()
}

pub fn append_message(accumulated: &mut Option<Message>, message: &Message) {
    match accumulated {
        Some(accumulated) => accumulated.content.extend(message.content.iter().cloned()),
        None => *accumulated = Some(message.clone()),
    }
}

fn message_json(message: &Message) -> Value {
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

    let parts = consolidated_parts(&message.content);
    json!({
        "role": role,
        "parts": parts,
    })
}

/// Merge consecutive text and reasoning parts into single entries so that
/// streaming tokens don't each get their own JSON object in the OTEL output.
fn consolidated_parts(content: &[MessageContent]) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    for item in content {
        let value = message_part_json(item);
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

fn message_part_json(content: &MessageContent) -> Value {
    match content {
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
                Ok(result) => serde_json::to_value(result)
                    .expect("CallToolResult must serialize"),
                Err(error) => json!({ "error": error.to_string() }),
            },
        }),
        MessageContent::Thinking(thinking) => json!({
            "type": "reasoning",
            "content": thinking.thinking,
        }),
        MessageContent::RedactedThinking(_) => json!({
            "type": "redacted_reasoning",
        }),
        MessageContent::ToolConfirmationRequest(request) => json!({
            "type": "tool_confirmation",
            "id": request.id,
            "name": request.tool_name,
            "arguments": request.arguments,
        }),
        MessageContent::ActionRequired(action) => json!({
            "type": "action_required",
            "data": action.data,
        }),
        MessageContent::SystemNotification(notification) => json!({
            "type": "system_notification",
            "content": notification.msg,
        }),
        MessageContent::Error(error) => json!({
            "type": "error",
            "kind": error.kind,
            "content": error.message,
        }),
    }
}
