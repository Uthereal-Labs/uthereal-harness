use reqwest::header::{HeaderName, HeaderValue};

pub const SESSION_ID_HEADER: &str = "agent-session-id";

pub const TOOL_CALL_REQUEST_ID_HEADER: &str = "agent-tool-call-request-id";
pub const WORKING_DIR_HEADER: &str = "agent-working-dir";

tokio::task_local! {
    pub static SESSION_ID: Option<String>;
    pub static TELEMETRY_SESSION_ID: Option<String>;
}

pub async fn with_session_id<F>(session_id: Option<String>, f: F) -> F::Output
where
    F: std::future::Future,
{
    SESSION_ID.scope(session_id, f).await
}

pub fn current_session_id() -> Option<String> {
    SESSION_ID.try_with(|id| id.clone()).ok().flatten()
}

pub async fn with_telemetry_session_id<F>(session_id: Option<String>, future: F) -> F::Output
where
    F: std::future::Future,
{
    TELEMETRY_SESSION_ID.scope(session_id, future).await
}

/// The telemetry session ID set for this task, if any.
pub fn current_telemetry_session_id() -> Option<String> {
    TELEMETRY_SESSION_ID.try_with(Clone::clone).ok().flatten()
}

pub fn telemetry_session_id(execution_session_id: &str) -> String {
    TELEMETRY_SESSION_ID
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .unwrap_or_else(|| execution_session_id.to_string())
}

pub fn session_id_request_builder() -> goose_providers::api_client::RequestBuilderDecorator {
    session_id_request_builder_with_header_name(HeaderName::from_static(SESSION_ID_HEADER))
}

pub(crate) fn session_id_request_builder_with_header_override(
    header_name_override: Option<&str>,
) -> Result<goose_providers::api_client::RequestBuilderDecorator, reqwest::header::InvalidHeaderName>
{
    let header_name = match header_name_override {
        Some(header_name) => HeaderName::from_bytes(header_name.as_bytes())?,
        None => HeaderName::from_static(SESSION_ID_HEADER),
    };

    Ok(session_id_request_builder_with_header_name(header_name))
}

fn session_id_request_builder_with_header_name(
    header_name: HeaderName,
) -> goose_providers::api_client::RequestBuilderDecorator {
    std::sync::Arc::new(move |request| {
        let (client, request) = request.build_split();
        let mut request = request?;
        let session_header = header_name.clone();
        request.headers_mut().remove(&session_header);

        if let Some(session_id) = current_session_id() {
            let value = HeaderValue::from_str(&session_id)?;
            request.headers_mut().insert(session_header, value);
        }

        request.headers_mut().remove("traceparent");
        request.headers_mut().remove("tracestate");
        #[cfg(feature = "otel")]
        {
            use opentelemetry::trace::TraceContextExt;
            use tracing_opentelemetry::OpenTelemetrySpanExt;

            let context = tracing::Span::current().context();
            let span = context.span();
            let parent = span.span_context();
            if parent.is_valid() {
                request.headers_mut().insert(
                    "traceparent",
                    HeaderValue::from_str(&format!(
                        "00-{}-{}-{:02x}",
                        parent.trace_id(),
                        parent.span_id(),
                        parent.trace_flags().to_u8(),
                    ))?,
                );
                // Native chat spans are exported to both backends; replace any
                // Python ancestor so gateway attempts attach to this generation.
                if let Ok(state) = parent.trace_state().insert(
                    "uthlf",
                    format!("{}-{}", parent.trace_id(), parent.span_id()),
                ) {
                    request
                        .headers_mut()
                        .insert("tracestate", HeaderValue::from_str(&state.header())?);
                }
            }
        }

        Ok(reqwest::RequestBuilder::from_parts(client, request))
    })
}

/// Local OS user running goose, shared by the OTLP `user.name` resource
/// attribute and the `session.user` span attribute so the two never drift.
pub fn session_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Hostname of the machine running goose, shared by the OTLP `host.name`
/// resource attribute and the `session.host` span attribute.
pub fn session_host() -> String {
    gethostname::gethostname().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_id_available_when_set() {
        with_session_id(Some("test-session-123".to_string()), async {
            assert_eq!(current_session_id(), Some("test-session-123".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_when_not_set() {
        let id = current_session_id();
        assert_eq!(id, None);
    }

    #[tokio::test]
    async fn test_session_id_none_when_explicitly_none() {
        with_session_id(None, async {
            assert_eq!(current_session_id(), None);
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_clears_outer_scope() {
        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(None, async {
                assert_eq!(current_session_id(), None);
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_scoped_correctly() {
        assert_eq!(current_session_id(), None);

        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(Some("inner-session".to_string()), async {
                assert_eq!(current_session_id(), Some("inner-session".to_string()));
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;

        assert_eq!(current_session_id(), None);
    }

    #[tokio::test]
    async fn test_session_id_across_await_points() {
        with_session_id(Some("persistent-session".to_string()), async {
            assert_eq!(current_session_id(), Some("persistent-session".to_string()));

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            assert_eq!(current_session_id(), Some("persistent-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_custom_header() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate =
                session_id_request_builder_with_header_override(Some("x-opencode-session"))
                    .unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get("x-opencode-session").unwrap(),
                "test-session-123"
            );
            assert!(request.headers().get(SESSION_ID_HEADER).is_none());
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_default_without_override() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate = session_id_request_builder_with_header_override(None).unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get(SESSION_ID_HEADER).unwrap(),
                "test-session-123"
            );
        })
        .await;
    }

    #[test]
    fn test_session_id_request_builder_rejects_invalid_header_override() {
        assert!(session_id_request_builder_with_header_override(Some("invalid header")).is_err());
    }

    #[test]
    fn test_request_without_active_trace_drops_stale_carrier() {
        let subscriber = tracing::subscriber::NoSubscriber::default();
        let _guard = tracing::subscriber::set_default(subscriber);
        let request = session_id_request_builder()(
            reqwest::Client::new()
                .get("http://localhost")
                .header("traceparent", "stale")
                .header("tracestate", "uthlf=stale"),
        )
        .unwrap()
        .build()
        .unwrap();
        assert!(!request.headers().contains_key("traceparent"));
        assert!(!request.headers().contains_key("tracestate"));
    }

    #[cfg(feature = "otel")]
    #[tokio::test]
    async fn test_concurrent_requests_propagate_current_generation() {
        use opentelemetry::trace::{
            SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, TracerProvider,
        };
        use opentelemetry_sdk::trace::SdkTracerProvider;
        use tracing::instrument::WithSubscriber;
        use tracing::Instrument;
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        use tracing_subscriber::prelude::*;

        let provider = SdkTracerProvider::builder().build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        let dispatch = tracing::Dispatch::new(subscriber);
        let decorate = session_id_request_builder();
        let run = |id: u128| {
            let dispatch = dispatch.clone();
            let decorate = decorate.clone();
            async move {
                let span = tracing::info_span!("chat");
                let parent = SpanContext::new(
                    TraceId::from(id),
                    SpanId::from(1),
                    TraceFlags::SAMPLED,
                    true,
                    TraceState::from_key_value([("vendor", "kept"), ("uthlf", "stale")]).unwrap(),
                );
                span.set_parent(opentelemetry::Context::new().with_remote_span_context(parent))
                    .unwrap();
                async move {
                    tokio::task::yield_now().await;
                    let context = tracing::Span::current().context();
                    let current = context.span().span_context().clone();
                    let request = decorate(reqwest::Client::new().get("http://localhost"))
                        .unwrap()
                        .build()
                        .unwrap();
                    assert_eq!(
                        request.headers()["traceparent"],
                        format!("00-{}-{}-01", current.trace_id(), current.span_id())
                    );
                    assert_eq!(
                        request.headers()["tracestate"],
                        format!(
                            "uthlf={}-{},vendor=kept",
                            current.trace_id(),
                            current.span_id()
                        )
                    );
                    current.trace_id()
                }
                .instrument(span)
                .await
            }
            .with_subscriber(dispatch)
        };
        let (first, second) = tokio::join!(run(41), run(42));
        assert_ne!(first, second);
        provider.shutdown().unwrap();
    }
}
