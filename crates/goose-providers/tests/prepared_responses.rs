use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use goose_providers::api_client::{ApiClient, AuthMethod};
use goose_providers::base::{collect_stream, Provider};
use goose_providers::conversation::message::Message;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use goose_providers::openai::{parse_openai_base_url, OpenAiProvider, OpenAiProviderBuilder};
use goose_providers::thinking::ThinkingEffort;
use rmcp::model::Tool;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn frame(event: Value) -> String {
    format!("data: {event}\n\n")
}

fn completed(status: &str) -> String {
    let mut event = json!({"type":"response.completed", "sequence_number":2, "response": {
        "id":"resp_test", "object":"response", "created_at":1,
        "status":status, "model":"gpt-6-luna-prod", "output":[],
        "usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}
    }});
    if status == "incomplete" {
        event["type"] = json!("response.incomplete");
        event["response"]["incomplete_details"] = json!({"reason":"max_output_tokens"});
    }
    frame(event)
}

fn text_delta() -> String {
    frame(
        json!({"type":"response.output_text.delta", "sequence_number":1,
        "item_id":"msg_test", "output_index":0,"content_index":0,"delta":"hello"}),
    )
}

fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

async fn scripted(responses: Vec<ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    let next = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(move |_: &Request| {
            let attempt = next.fetch_add(1, Ordering::SeqCst);
            responses
                .get(attempt)
                .cloned()
                .unwrap_or_else(|| ResponseTemplate::new(418))
        })
        .mount(&server)
        .await;
    server
}

fn provider(server: &MockServer, timeout: Duration, attempts: usize) -> OpenAiProvider {
    let (host, query, has_v1) =
        parse_openai_base_url(&format!("{}/openai/v1", server.uri())).unwrap();
    assert!(query.is_empty());
    assert!(has_v1);
    OpenAiProviderBuilder::new(
        ApiClient::with_timeout_and_tls(
            host,
            AuthMethod::BearerToken("local-test-key".into()),
            timeout,
            None,
        )
        .unwrap(),
    )
    .base_path("v1/responses")
    .prepared_request_max_attempts(Some(attempts))
    .build()
}

fn config() -> ModelConfig {
    ModelConfig::new("gpt-6-luna-prod")
        .with_thinking_effort(ThinkingEffort::Off)
        .with_max_tokens(Some(16384))
        .with_vision_support(true)
}

#[tokio::test]
async fn identical_primary_payload_survives_http_retry_with_tools_images_none_and_usage() {
    let server = scripted(vec![
        ResponseTemplate::new(503),
        sse(text_delta() + &completed("completed")),
    ])
    .await;
    let provider = provider(&server, Duration::from_secs(60), 2);
    assert!(provider.owns_stream_recovery());
    let tool: Tool =
        serde_json::from_value(json!({"name":"lookup", "description":"Look up evidence",
        "inputSchema":{"type":"object","properties":{"query":{"type":"string"}}}}))
        .unwrap();
    let messages = [Message::user()
        .with_text("Look at this image")
        .with_image("dGVzdA==", "image/png")];
    let stream = provider
        .stream(&config(), "Preserve the evidence.", &messages, &[tool])
        .await
        .unwrap();
    let (message, usage) = collect_stream(stream).await.unwrap();
    assert_eq!(message.as_concat_text(), "hello");
    assert_eq!(usage.usage.input_tokens, Some(12));
    assert_eq!(usage.usage.output_tokens, Some(3));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].body, requests[1].body);
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer local-test-key"
    );
    let payload: Value = requests[0].body_json().unwrap();
    assert_eq!(payload["model"], "gpt-6-luna-prod");
    assert_eq!(payload["reasoning"]["effort"], "none", "{payload}");
    assert_eq!(payload["max_output_tokens"], 16384);
    assert_eq!(payload["tools"][0]["name"], "lookup");
    assert!(payload
        .to_string()
        .contains("data:image/png;base64,dGVzdA=="));
}

#[tokio::test]
async fn http_and_pre_output_stream_failures_share_two_attempts_in_both_orders() {
    for responses in [
        vec![ResponseTemplate::new(503), sse(String::new())],
        vec![sse(String::new()), ResponseTemplate::new(503)],
    ] {
        let server = scripted(responses).await;
        let result = provider(&server, Duration::from_secs(60), 2)
            .stream(&config(), "", &[], &[])
            .await;
        assert!(result.is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
}

#[tokio::test]
async fn pre_output_stream_disconnect_retries_and_succeeds() {
    let server = scripted(vec![sse(String::new()), sse(completed("completed"))]).await;
    let stream = provider(&server, Duration::from_secs(60), 2)
        .stream(&config(), "", &[], &[])
        .await
        .unwrap();
    let (_, usage) = collect_stream(stream).await.unwrap();
    assert_eq!(usage.usage.output_tokens, Some(3));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn output_commits_stream_and_disconnect_never_replays() {
    let server = scripted(vec![sse(text_delta())]).await;
    let mut stream = provider(&server, Duration::from_secs(60), 2)
        .stream(&config(), "", &[], &[])
        .await
        .unwrap();
    assert_eq!(
        stream
            .next()
            .await
            .unwrap()
            .unwrap()
            .0
            .unwrap()
            .as_concat_text(),
        "hello"
    );
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(ProviderError::NetworkError(_))
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_protocol_is_terminal_and_does_not_echo_payload() {
    for body in [
        "data: private-secret-invalid-json\n\n",
        "data: {\"type\":\"response.completed\"}\n\n",
        "data: {\"missing_type\":true}\n\n",
    ] {
        let server = scripted(vec![sse(body.into())]).await;
        let error = provider(&server, Duration::from_secs(60), 2)
            .stream(&config(), "", &[], &[])
            .await
            .err()
            .unwrap();
        assert!(matches!(error, ProviderError::ProtocolError(_)));
        assert_eq!(error.telemetry_type(), "protocol");
        assert!(!error.to_string().contains("private-secret"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn permanent_http_errors_and_explicit_provider_failures_never_retry() {
    let mut responses = vec![400, 401, 403, 404]
        .into_iter()
        .map(ResponseTemplate::new)
        .collect::<Vec<_>>();
    responses.push(sse(frame(
        json!({"type":"response.failed", "sequence_number":1,
        "error":{"code":"content_filter","message":"filtered"}}),
    )));
    for response in responses {
        let server = scripted(vec![response]).await;
        let result = provider(&server, Duration::from_secs(60), 2)
            .stream(&config(), "", &[], &[])
            .await;
        assert!(result.is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn streaming_output_limit_is_returned_without_replay() {
    let server = scripted(vec![sse(completed("incomplete"))]).await;
    let stream = provider(&server, Duration::from_secs(60), 2)
        .stream(&config(), "", &[], &[])
        .await
        .unwrap();
    let (message, usage) = collect_stream(stream).await.unwrap();
    assert!(message.metadata.output_token_limit_reached);
    assert_eq!(usage.finish_reasons, Some(vec!["max_output_tokens".into()]));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn existing_header_timeout_recovers_within_same_allowance() {
    let server = scripted(vec![
        sse(completed("completed")).set_delay(Duration::from_millis(200)),
        sse(completed("completed")),
    ])
    .await;
    let stream = provider(&server, Duration::from_millis(50), 2)
        .stream(&config(), "", &[], &[])
        .await
        .unwrap();
    collect_stream(stream).await.unwrap();
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn retry_after_is_honored_without_extra_attempts() {
    let server = scripted(vec![
        ResponseTemplate::new(429).insert_header("Retry-After", "1"),
        sse(completed("completed")),
    ])
    .await;
    let started = std::time::Instant::now();
    let stream = provider(&server, Duration::from_secs(60), 2)
        .stream(&config(), "", &[], &[])
        .await
        .unwrap();
    collect_stream(stream).await.unwrap();
    assert!(started.elapsed() >= Duration::from_secs(1));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn cancellation_during_header_wait_and_backoff_never_dispatches_again() {
    for response in [
        sse(completed("completed")).set_delay(Duration::from_secs(5)),
        ResponseTemplate::new(503),
    ] {
        let server = scripted(vec![response]).await;
        let provider = provider(&server, Duration::from_secs(60), 2);
        let config = config();
        assert!(tokio::time::timeout(
            Duration::from_millis(100),
            provider.stream(&config, "", &[], &[])
        )
        .await
        .is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn stricter_one_attempt_allowance_remains_terminal() {
    let server = scripted(vec![ResponseTemplate::new(503)]).await;
    assert!(provider(&server, Duration::from_secs(60), 1)
        .stream(&config(), "", &[], &[])
        .await
        .is_err());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn primary_production_deployment_retains_low_effort_instead_of_pro_capabilities() {
    let server = scripted(vec![sse(completed("completed"))]).await;
    let stream = provider(&server, Duration::from_secs(60), 2)
        .stream(
            &config().with_thinking_effort(ThinkingEffort::Low),
            "",
            &[],
            &[],
        )
        .await
        .unwrap();
    collect_stream(stream).await.unwrap();
    let payload: Value = server.received_requests().await.unwrap()[0]
        .body_json()
        .unwrap();
    assert_eq!(payload["model"], "gpt-6-luna-prod");
    assert_eq!(payload["reasoning"]["effort"], "low");
}

#[tokio::test]
async fn eligible_http_timeouts_recover_without_switching_model() {
    for status in [408, 499] {
        let server = scripted(vec![
            ResponseTemplate::new(status),
            sse(completed("completed")),
        ])
        .await;
        let stream = provider(&server, Duration::from_secs(60), 2)
            .stream(&config(), "", &[], &[])
            .await
            .unwrap();
        collect_stream(stream).await.unwrap();
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
}

#[tokio::test]
async fn incomplete_content_filter_is_terminal_instead_of_empty_completion() {
    let event = completed("incomplete").replace("max_output_tokens", "content_filter");
    let server = scripted(vec![sse(event)]).await;
    let error = provider(&server, Duration::from_secs(60), 2)
        .stream(&config(), "", &[], &[])
        .await
        .err()
        .unwrap();
    assert!(matches!(error, ProviderError::Refusal { .. }));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn interrupted_http_body_is_transport_failure_and_recovers_before_output() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut received = Vec::new();
            let mut buffer = [0; 2048];
            while !received.windows(4).any(|window| window == b"\r\n\r\n") {
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                received.extend_from_slice(&buffer[..count]);
            }
            observed.fetch_add(1, Ordering::SeqCst);
            let body = if attempt == 0 {
                ": heartbeat\n\n".to_string()
            } else {
                completed("completed")
            };
            let declared_length = body.len() + if attempt == 0 { 100 } else { 0 };
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n{body}").as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    let provider = OpenAiProviderBuilder::new(
        ApiClient::new_with_tls(format!("http://{address}/openai"), AuthMethod::NoAuth, None)
            .unwrap(),
    )
    .base_path("v1/responses")
    .prepared_request_max_attempts(Some(2))
    .build();
    let stream = provider.stream(&config(), "", &[], &[]).await.unwrap();
    collect_stream(stream).await.unwrap();
    server.await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
