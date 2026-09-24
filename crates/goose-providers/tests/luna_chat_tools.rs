use futures::StreamExt;
use goose_providers::api_client::{ApiClient, AuthMethod};
use goose_providers::base::Provider;
use goose_providers::conversation::message::Message;
use goose_providers::model::ModelConfig;
use goose_providers::openai::OpenAiProviderBuilder;
use goose_providers::thinking::ThinkingEffort;
use rmcp::model::{CallToolResult, ContentBlock, Tool};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn luna_none_survives_normalization_and_streamed_tool_continuation() {
    let server = MockServer::start().await;
    let tool: Tool = serde_json::from_value(json!({
        "name": "echo", "description": "Echo a value",
        "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}
    }))
    .unwrap();
    let chunks = [
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_echo", "type": "function",
            "function": {"name": "echo", "arguments": "{\"value\":"}
        }]}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": "\"ok\"}"}
        }]}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let body = chunks
        .iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .expect(2)
        .mount(&server)
        .await;

    let provider = OpenAiProviderBuilder::new(
        ApiClient::new_with_tls(format!("{}/v1", server.uri()), AuthMethod::NoAuth, None).unwrap(),
    )
    .base_path("chat/completions")
    .build();
    let config = ModelConfig::new("azure-openai/azure_harness_balanced-gpt-6-luna-none");
    assert_eq!(config.thinking_effort(), Some(ThinkingEffort::Off));
    let mut messages = vec![Message::user().with_text("Echo ok")];
    let mut stream = provider
        .stream(&config, "", &messages, std::slice::from_ref(&tool))
        .await
        .unwrap();
    let mut tool_message = None;
    while let Some(event) = stream.next().await {
        if let (Some(message), _) = event.unwrap() {
            if message
                .content
                .iter()
                .any(|part| part.as_tool_request().is_some())
            {
                tool_message = Some(message);
            }
        }
    }
    let tool_message = tool_message.expect("stream should yield the assembled tool call");
    let request = tool_message
        .content
        .iter()
        .find_map(|part| part.as_tool_request())
        .unwrap();
    assert_eq!(request.id, "call_echo");
    let call = request.tool_call.as_ref().unwrap();
    assert_eq!(call.name, "echo");
    assert_eq!(
        call.arguments.as_ref().unwrap().get("value"),
        Some(&json!("ok"))
    );
    messages.push(tool_message);
    messages.push(Message::user().with_tool_response(
        "call_echo",
        Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
    ));
    let mut continuation = provider
        .stream(&config, "", &messages, &[tool])
        .await
        .unwrap();
    while let Some(event) = continuation.next().await {
        event.unwrap();
    }
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        let payload: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            payload["model"],
            "azure-openai/azure_harness_balanced-gpt-6-luna"
        );
        assert_eq!(payload["reasoning_effort"], "none");
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["tools"][0]["function"]["name"], "echo");
        assert!(payload.get("thinking_effort").is_none());
    }
    let followup: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(followup["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["role"] == "tool" && message["tool_call_id"] == "call_echo"));
}
