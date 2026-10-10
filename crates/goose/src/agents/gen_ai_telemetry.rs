use crate::conversation::message::ToolResult;
use crate::session::Session;
use goose_providers::conversation::token_usage::{ProviderUsage, Usage};
use goose_providers::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use serde_json::{json, Value};
use tracing::Span;

pub(super) const CAPTURE_MESSAGE_CONTENT_ENV: &str =
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

pub(super) fn capture_message_content() -> bool {
    std::env::var(CAPTURE_MESSAGE_CONTENT_ENV).is_ok_and(|value| value.eq_ignore_ascii_case("true"))
}

pub(super) use goose_agent::telemetry::{
    append_message, input_messages_with_system_json, output_message_json, system_instructions_json,
    tool_definitions_json,
};

pub(super) fn simple_input_json(text: &str) -> String {
    json!([{"role": "user", "content": text}]).to_string()
}

pub(super) fn simple_output_json(text: &str) -> String {
    json!([{"role": "assistant", "content": text, "finish_reason": "stop"}]).to_string()
}

pub(super) fn record_usage(span: &Span, usage: &Usage) {
    if let Some(tokens) = usage.input_tokens {
        span.record("gen_ai.usage.input_tokens", tokens);
    }
    if let Some(tokens) = usage.output_tokens {
        span.record("gen_ai.usage.output_tokens", tokens);
    }
    if let Some(tokens) = usage.cache_read_input_tokens {
        span.record("gen_ai.usage.cache_read.input_tokens", tokens);
    }
    if let Some(tokens) = usage.cache_write_input_tokens {
        span.record("gen_ai.usage.cache_creation.input_tokens", tokens);
    }
}

pub(super) fn record_provider_usage(span: &Span, usage: &ProviderUsage) {
    let canonical_model = usage
        .model
        .strip_prefix("gpt-6-luna-")
        .filter(|suffix| chrono::NaiveDate::parse_from_str(suffix, "%Y-%m-%d").is_ok())
        .map_or(usage.model.as_str(), |_| "gpt-6-luna");
    span.record("gen_ai.response.model", canonical_model);
    if canonical_model != usage.model.as_str() {
        span.record("gen_ai.response.model.upstream", usage.model.as_str());
    }
    record_usage(span, &usage.usage);
    if let Some(reasons) = &usage.finish_reasons {
        let reasons_json = serde_json::to_string(reasons).unwrap_or_default();
        span.record("gen_ai.response.finish_reasons", reasons_json.as_str());
    }
    if let Some(id) = &usage.response_id {
        span.record("gen_ai.response.id", id.as_str());
    }
}

/// Ends a span when dropped.
///
/// tracing-opentelemetry ends a span at its last exit. A provider stream is
/// read after the instrumented function that created it returned, without
/// entering its span, so the generation would otherwise end when the stream
/// opened rather than when it finished. Entering and leaving the span here
/// moves its end to the moment the stream is done.
pub(super) struct EndSpanOnDrop(Span);

impl EndSpanOnDrop {
    pub(super) fn new(span: Span) -> Self {
        Self(span)
    }
}

impl Drop for EndSpanOnDrop {
    fn drop(&mut self) {
        let _entered = self.0.enter();
    }
}

/// Provider stream timing for the generation span: Langfuse shows
/// `completion_start_time` as time to first token, and the elapsed time marks
/// when the provider stream ended, separating generation from later agent work.
pub(super) fn record_stream_timing(
    span: &Span,
    request_started_at: chrono::DateTime<chrono::Utc>,
    usage: &ProviderUsage,
) {
    let Some(stats) = usage.stats.as_ref() else {
        return;
    };
    if let Some(first_token_ms) = stats.time_to_first_token_ms {
        let first_token_at =
            request_started_at + chrono::Duration::milliseconds(first_token_ms as i64);
        span.record(
            "langfuse.observation.completion_start_time",
            first_token_at
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
                .as_str(),
        );
    }
    if let Some(elapsed_ms) = stats.elapsed_ms {
        span.record("uthereal.provider_stream.elapsed_ms", elapsed_ms as i64);
    }
}

pub(super) fn record_request_params(span: &Span, model_config: &ModelConfig) {
    if let Some(temperature) = model_config.temperature {
        span.record("gen_ai.request.temperature", temperature as f64);
    }
    if let Some(max_tokens) = model_config.max_tokens {
        span.record("gen_ai.request.max_tokens", max_tokens as i64);
    }
}

pub(super) fn record_tool_arguments(span: &Span, tool_call: &CallToolRequestParams) {
    if capture_message_content() {
        let arguments = tool_call
            .arguments
            .as_ref()
            .map(|arguments| Value::Object(arguments.clone()))
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        span.record(
            "gen_ai.tool.call.arguments",
            tracing::field::display(arguments),
        );
    }
}

/// `_meta` key of a tool result whose call a rule refused. The agent gets the
/// refusal as an error result telling it what to do instead, but nothing
/// failed, so traces show it as a warning rather than an error.
pub(crate) const TOOL_REJECTION_META_KEY: &str = "uthereal/rejection";

/// An error result for a call a rule refused (see [`TOOL_REJECTION_META_KEY`]).
pub(crate) fn rule_rejection(text: impl Into<String>) -> CallToolResult {
    let mut result = CallToolResult::error(vec![rmcp::model::ContentBlock::text(text.into())]);
    let mut meta = result
        .meta
        .take()
        .unwrap_or_else(rmcp::model::MetaObject::new);
    meta.0
        .insert(TOOL_REJECTION_META_KEY.to_string(), json!("rule"));
    result.meta = Some(meta);
    result
}

/// Whether a tool result is a rule's refusal rather than a failure.
pub(crate) fn is_rule_rejection(result: &ToolResult<CallToolResult>) -> bool {
    matches!(result, Ok(result) if result.is_error == Some(true)
        && result.meta.as_ref().is_some_and(|meta| meta.0.contains_key(TOOL_REJECTION_META_KEY)))
}

pub(super) fn record_tool_result(span: &Span, result: &ToolResult<CallToolResult>) {
    let failed = !matches!(result, Ok(result) if result.is_error != Some(true));
    if is_rule_rejection(result) {
        span.record("langfuse.observation.level", "WARNING");
        span.record(
            "langfuse.observation.status_message",
            "Refused by a rule; the agent was told what to do instead",
        );
    } else if failed {
        span.record("error.type", "tool_execution_error");
        #[cfg(feature = "otel")]
        {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            span.set_status(opentelemetry::trace::Status::error("Tool execution failed"));
        }
    }
    if capture_message_content() {
        let value = match result {
            Ok(result) => serde_json::to_value(result).expect("CallToolResult must serialize"),
            Err(error) => json!({"error": error.to_string()}),
        };
        span.record("gen_ai.tool.call.result", value.to_string().as_str());
    }
}

/// Mark an agent run that finished normally as successful.
///
/// The OpenTelemetry layer turns any ERROR-level event emitted while the span
/// is entered into an error status, including events from background work
/// that only shares the span (a transport retry, a recovered tool failure).
/// A run that completed should not be reported as failed because of them; the
/// events themselves stay on the span.
pub(super) fn record_completed(span: &Span) {
    #[cfg(feature = "otel")]
    {
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        span.set_status(opentelemetry::trace::Status::Ok);
    }
    #[cfg(not(feature = "otel"))]
    let _ = span;
}

pub(super) fn agent_name(session: &Session) -> &str {
    session
        .recipe
        .as_ref()
        .map_or("goose", |recipe| recipe.title.as_str())
}

pub(super) fn tool_result_json(result: &ToolResult<CallToolResult>) -> String {
    match result {
        Ok(result) if result.is_error != Some(true) => json!({
            "status": "success",
            "value": result,
        }),
        Ok(result) => json!({
            "status": "error",
            "value": result,
        }),
        Err(error) => json!({
            "status": "error",
            "error": error.to_string(),
        }),
    }
    .to_string()
}

#[cfg(test)]
pub(super) mod test_support {
    use serde_json::{Map, Number, Value};
    use std::fmt;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{subscriber::DefaultGuard, Subscriber};
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::Layer;

    #[derive(Clone)]
    pub struct SpanFieldCapture {
        span_name: &'static str,
        fields: Arc<Mutex<Map<String, Value>>>,
    }

    impl SpanFieldCapture {
        pub fn new(span_name: &'static str) -> Self {
            Self {
                span_name,
                fields: Arc::new(Mutex::new(Map::new())),
            }
        }

        pub fn fields(&self) -> Map<String, Value> {
            self.fields.lock().unwrap().clone()
        }

        pub fn set_default(self) -> DefaultGuard {
            use tracing_subscriber::prelude::*;

            tracing::subscriber::set_default(tracing_subscriber::registry().with(self))
        }
    }

    impl<S> Layer<S> for SpanFieldCapture
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
            if attrs.metadata().name() == self.span_name {
                attrs.record(&mut FieldVisitor {
                    fields: &self.fields,
                });
            }
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
            if ctx
                .span(id)
                .is_some_and(|span| span.metadata().name() == self.span_name)
            {
                values.record(&mut FieldVisitor {
                    fields: &self.fields,
                });
            }
        }
    }

    struct FieldVisitor<'a> {
        fields: &'a Mutex<Map<String, Value>>,
    }

    impl FieldVisitor<'_> {
        fn insert(&self, field: &Field, value: Value) {
            self.fields
                .lock()
                .unwrap()
                .insert(field.name().to_string(), value);
        }
    }

    impl Visit for FieldVisitor<'_> {
        fn record_i64(&mut self, field: &Field, value: i64) {
            self.insert(field, Value::Number(value.into()));
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.insert(field, Value::Number(value.into()));
        }

        fn record_f64(&mut self, field: &Field, value: f64) {
            if let Some(value) = Number::from_f64(value) {
                self.insert(field, Value::Number(value));
            }
        }

        fn record_bool(&mut self, field: &Field, value: bool) {
            self.insert(field, Value::Bool(value));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.insert(field, Value::String(value.to_string()));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.insert(field, Value::String(format!("{value:?}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::Message;
    use goose_test_support::otel::clear_otel_env;
    use rmcp::{model::CallToolRequestParams, object};

    fn test_recipe(title: &str) -> crate::recipe::Recipe {
        serde_json::from_value(serde_json::json!({
            "title": title,
            "description": "test recipe",
            "instructions": "do stuff",
        }))
        .unwrap()
    }

    #[cfg(feature = "otel")]
    #[test]
    fn a_completed_run_is_not_reported_as_failed_after_an_error_event() {
        use opentelemetry::trace::{Status, TracerProvider as _};
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
        use tracing_subscriber::prelude::*;

        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("completed-test")));
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("reply_stream");
            span.in_scope(|| tracing::error!("recovered background failure"));
            record_completed(&span);
        });
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].status, Status::Ok);
    }

    #[test]
    fn content_capture_requires_explicit_opt_in() {
        let _env = clear_otel_env(&[]);
        assert!(!capture_message_content());

        drop(_env);
        let _env = clear_otel_env(&[(CAPTURE_MESSAGE_CONTENT_ENV, "TrUe")]);
        assert!(capture_message_content());
    }

    #[test]
    fn messages_follow_gen_ai_semantic_convention_shape() {
        let request = CallToolRequestParams::new("get_weather")
            .with_arguments(object!({ "location": "Paris" }));
        let messages = vec![
            Message::user().with_text("Weather?"),
            Message::assistant().with_tool_request("call-1", Ok(request)),
        ];

        let value: Value =
            serde_json::from_str(&input_messages_with_system_json("System", &messages, false))
                .unwrap();
        assert_eq!(value[0]["role"], "system");
        assert_eq!(value[1]["role"], "user");
        assert_eq!(value[1]["parts"][0]["type"], "text");
        assert_eq!(value[1]["parts"][0]["content"], "Weather?");
        assert_eq!(value[2]["parts"][0]["type"], "tool_call");
        assert_eq!(value[2]["parts"][0]["name"], "get_weather");
        assert_eq!(value[2]["parts"][0]["arguments"]["location"], "Paris");
    }

    #[test]
    fn system_instructions_preserve_the_actual_prompt() {
        let value: Value =
            serde_json::from_str(&system_instructions_json("Be concise.\nUse tools.")).unwrap();

        assert_eq!(value[0]["type"], "text");
        assert_eq!(value[0]["content"], "Be concise.\nUse tools.");
    }

    #[test]
    fn input_messages_include_the_actual_system_prompt() {
        let messages = vec![Message::user().with_text("Hello")];
        let value: Value = serde_json::from_str(&input_messages_with_system_json(
            "System prompt",
            &messages,
            false,
        ))
        .unwrap();

        assert_eq!(value[0]["role"], "system");
        assert_eq!(value[0]["parts"][0]["content"], "System prompt");
        assert_eq!(value[1]["role"], "user");
    }

    #[test]
    fn output_messages_include_finish_reason() {
        let message = Message::assistant().with_text("Sunny");
        let value: Value = serde_json::from_str(&output_message_json(&message)).unwrap();

        assert_eq!(value[0]["role"], "assistant");
        assert_eq!(value[0]["finish_reason"], "stop");
        assert_eq!(value[0]["parts"][0]["content"], "Sunny");

        let request = CallToolRequestParams::new("get_weather");
        let message = Message::assistant().with_tool_request("call-1", Ok(request));
        let value: Value = serde_json::from_str(&output_message_json(&message)).unwrap();
        assert_eq!(value[0]["finish_reason"], "tool_call");
    }

    #[test]
    fn record_request_params_records_temperature_and_max_tokens() {
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();

        let config = ModelConfig::new("test-model")
            .with_temperature(Some(0.5))
            .with_max_tokens(Some(4096));
        let span = tracing::info_span!(
            "test_span",
            "gen_ai.request.temperature" = tracing::field::Empty,
            "gen_ai.request.max_tokens" = tracing::field::Empty,
        );
        record_request_params(&span, &config);

        let fields = capture.fields();
        assert_eq!(fields["gen_ai.request.temperature"], 0.5);
        assert_eq!(fields["gen_ai.request.max_tokens"], 4096);
    }

    #[test]
    fn record_request_params_skips_none_values() {
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();

        let config = ModelConfig::new("test-model");
        let span = tracing::info_span!(
            "test_span",
            "gen_ai.request.temperature" = tracing::field::Empty,
            "gen_ai.request.max_tokens" = tracing::field::Empty,
        );
        record_request_params(&span, &config);

        let fields = capture.fields();
        assert!(!fields.contains_key("gen_ai.request.temperature"));
        assert!(!fields.contains_key("gen_ai.request.max_tokens"));
    }

    #[test]
    fn record_tool_arguments_gated_by_content_env() {
        let _env = clear_otel_env(&[]);
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();

        let tool_call =
            CallToolRequestParams::new("my_tool").with_arguments(object!({ "key": "value" }));
        let span = tracing::info_span!(
            "test_span",
            "gen_ai.tool.call.arguments" = tracing::field::Empty,
        );
        record_tool_arguments(&span, &tool_call);

        let fields = capture.fields();
        assert!(!fields.contains_key("gen_ai.tool.call.arguments"));

        drop(_env);
        let _env = clear_otel_env(&[(CAPTURE_MESSAGE_CONTENT_ENV, "true")]);
        let capture2 = test_support::SpanFieldCapture::new("test_span2");
        let _guard2 = capture2.clone().set_default();

        let span2 = tracing::info_span!(
            "test_span2",
            "gen_ai.tool.call.arguments" = tracing::field::Empty,
        );
        record_tool_arguments(&span2, &tool_call);

        let fields2 = capture2.fields();
        assert!(fields2.contains_key("gen_ai.tool.call.arguments"));
        let args: Value =
            serde_json::from_str(fields2["gen_ai.tool.call.arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["key"], "value");
    }

    #[test]
    fn record_tool_result_includes_failure_details() {
        let _env = clear_otel_env(&[(CAPTURE_MESSAGE_CONTENT_ENV, "true")]);
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();

        let success_result: ToolResult<CallToolResult> = Ok(CallToolResult::success(vec![
            rmcp::model::ContentBlock::text("ok"),
        ]));
        let span = tracing::info_span!(
            "test_span",
            "gen_ai.tool.call.result" = tracing::field::Empty,
        );
        record_tool_result(&span, &success_result);

        let fields = capture.fields();
        assert!(fields.contains_key("gen_ai.tool.call.result"));

        let capture2 = test_support::SpanFieldCapture::new("test_span2");
        let _guard2 = capture2.clone().set_default();

        let error_result: ToolResult<CallToolResult> = Err(rmcp::model::ErrorData::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            "failed".to_string(),
            None,
        ));
        let span2 = tracing::info_span!(
            "test_span2",
            "gen_ai.tool.call.result" = tracing::field::Empty,
        );
        record_tool_result(&span2, &error_result);

        let fields2 = capture2.fields();
        let result: Value =
            serde_json::from_str(fields2["gen_ai.tool.call.result"].as_str().unwrap()).unwrap();
        assert!(result["error"].as_str().unwrap().contains("failed"));
    }

    #[test]
    fn a_rule_rejection_is_recorded_as_a_warning_not_an_error() {
        let capture = test_support::SpanFieldCapture::new("rejected_span");
        let _guard = capture.clone().set_default();
        let rejected: ToolResult<CallToolResult> = Ok(rule_rejection("Nothing to wait for."));
        assert!(is_rule_rejection(&rejected));
        let span = tracing::info_span!(
            "rejected_span",
            "error.type" = tracing::field::Empty,
            "langfuse.observation.level" = tracing::field::Empty,
            "langfuse.observation.status_message" = tracing::field::Empty,
        );
        record_tool_result(&span, &rejected);
        let fields = capture.fields();
        assert_eq!(fields["langfuse.observation.level"], "WARNING");
        assert!(!fields.contains_key("error.type"));

        // A plain error result is still an error.
        let failed: ToolResult<CallToolResult> = Ok(CallToolResult::error(vec![
            rmcp::model::ContentBlock::text("failed"),
        ]));
        assert!(!is_rule_rejection(&failed));
    }

    #[test]
    fn record_provider_usage_includes_finish_reasons_and_response_id() {
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();

        let usage = ProviderUsage::new(
            "test-model".to_string(),
            Usage::new(Some(10), Some(20), None),
        )
        .with_finish_reasons(vec!["stop".to_string()])
        .with_response_id("resp-123".to_string());

        let span = tracing::info_span!(
            "test_span",
            "gen_ai.response.model" = tracing::field::Empty,
            "gen_ai.response.finish_reasons" = tracing::field::Empty,
            "gen_ai.response.id" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
        );
        record_provider_usage(&span, &usage);

        let fields = capture.fields();
        assert_eq!(fields["gen_ai.response.model"], "test-model");
        assert_eq!(fields["gen_ai.response.finish_reasons"], "[\"stop\"]");
        assert_eq!(fields["gen_ai.response.id"], "resp-123");
        assert_eq!(fields["gen_ai.usage.input_tokens"], 10);
        assert_eq!(fields["gen_ai.usage.output_tokens"], 20);
    }

    #[test]
    fn record_stream_timing_sets_completion_start_and_elapsed() {
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();
        let mut usage = ProviderUsage::new("test-model".to_string(), Usage::default());
        usage.stats = Some(goose_providers::conversation::token_usage::ProviderStats {
            time_to_first_token_ms: Some(250),
            elapsed_ms: Some(4000),
            ..Default::default()
        });
        let span = tracing::info_span!(
            "test_span",
            "langfuse.observation.completion_start_time" = tracing::field::Empty,
            "uthereal.provider_stream.elapsed_ms" = tracing::field::Empty,
        );
        let started = chrono::DateTime::parse_from_rfc3339("2026-10-04T17:55:56.000Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        record_stream_timing(&span, started, &usage);

        let fields = capture.fields();
        assert_eq!(
            fields["langfuse.observation.completion_start_time"],
            "2026-10-04T17:55:56.250Z"
        );
        assert_eq!(fields["uthereal.provider_stream.elapsed_ms"], 4000);
    }

    #[test]
    fn record_provider_usage_normalizes_dated_gpt_6_luna_name() {
        let capture = test_support::SpanFieldCapture::new("test_span");
        let _guard = capture.clone().set_default();
        let usage = ProviderUsage::new("gpt-6-luna-2026-09-22".to_string(), Usage::default());
        let span = tracing::info_span!(
            "test_span",
            "gen_ai.response.model" = tracing::field::Empty,
            "gen_ai.response.model.upstream" = tracing::field::Empty,
        );

        record_provider_usage(&span, &usage);

        let fields = capture.fields();
        assert_eq!(fields["gen_ai.response.model"], "gpt-6-luna");
        assert_eq!(
            fields["gen_ai.response.model.upstream"],
            "gpt-6-luna-2026-09-22"
        );
    }

    #[test]
    fn agent_name_returns_recipe_title_when_present() {
        let session = Session {
            recipe: Some(test_recipe("My Recipe")),
            ..Default::default()
        };
        assert_eq!(agent_name(&session), "My Recipe");
    }

    #[test]
    fn agent_name_returns_goose_default() {
        let session = Session::default();
        assert_eq!(agent_name(&session), "goose");
    }
}
