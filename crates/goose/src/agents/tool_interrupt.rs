//! Tool calls that a message from the parent task or a channel may interrupt.
//!
//! A child task blocked in a long wait (for example an editor job) cannot read
//! its mailbox until the call returns. A tool that declares
//! [`INTERRUPT_ON_MESSAGE_META_KEY`] in its definition `_meta` is raced against
//! the child's mailbox: when a parent message or a waking channel notice is
//! pending,
//! the call is detached rather than cancelled, so the work it started keeps
//! running, and the declared notice is returned as the tool result. The agent
//! then reads the message at its next checkpoint. Cancelling the task still
//! cancels the call.

use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{CallToolResult, ContentBlock, ErrorCode, ErrorData, MetaObject};
use serde_json::Value;
use tokio::task::AbortHandle;
use tracing::Instrument;

use crate::mcp_utils::ToolResult;
use crate::session::{MailboxMessage, SessionManager};

/// Tool-definition metadata key. Its string value is returned as the tool
/// result when a parent message or a waking channel notice interrupts the call.
pub const INTERRUPT_ON_MESSAGE_META_KEY: &str = "goose.interruptOnMessage";

/// Tool-result metadata key set on an interrupted call's result.
pub const INTERRUPTED_META_KEY: &str = "goose.interrupted";

const MAILBOX_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// True when a tool result came from an interrupted wait rather than the tool.
pub fn was_interrupted(result: &CallToolResult) -> bool {
    result
        .meta
        .as_ref()
        .and_then(|meta| meta.0.get(INTERRUPTED_META_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The interruption notice a tool definition declares, if any.
pub fn interrupt_notice(tool_meta: Option<&Value>) -> Option<String> {
    tool_meta
        .and_then(|meta| meta.get(INTERRUPT_ON_MESSAGE_META_KEY))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|notice| !notice.is_empty())
        .map(str::to_string)
}

fn newest_interrupting_message(messages: &[MailboxMessage]) -> Option<i64> {
    messages
        .iter()
        .filter(|message| message.interrupts_wait())
        .map(|message| message.id)
        .max()
}

/// Aborts the spawned call when dropped, unless it was detached.
struct AbortUnlessDetached(Option<AbortHandle>);

impl AbortUnlessDetached {
    fn detach(&mut self) {
        self.0.take();
    }
}

impl Drop for AbortUnlessDetached {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

fn interrupted_result(notice: String) -> CallToolResult {
    let mut result = CallToolResult::success(vec![ContentBlock::text(notice)]);
    result.structured_content = Some(serde_json::json!({ "interrupted": true }));
    result.meta = Some(MetaObject(
        serde_json::json!({ INTERRUPTED_META_KEY: true })
            .as_object()
            .unwrap()
            .clone(),
    ));
    result
}

/// Runs `call` until it finishes or a pending parent message or waking channel
/// notice wakes `session_id`.
pub async fn interruptible_by_parent_message<F>(
    call: F,
    session_manager: Arc<SessionManager>,
    session_id: String,
    notice: String,
) -> ToolResult<CallToolResult>
where
    F: std::future::Future<Output = ToolResult<CallToolResult>> + Send + 'static,
{
    let mut call = tokio::spawn(with_caller_context(call));
    // Cancelling this wait (dropping it) still cancels the call.
    let mut guard = AbortUnlessDetached(Some(call.abort_handle()));
    loop {
        tokio::select! {
            joined = &mut call => {
                guard.detach();
                return joined.unwrap_or_else(|error| {
                    Err(ErrorData::new(
                        ErrorCode::INTERNAL_ERROR,
                        format!("The tool call stopped unexpectedly: {error}"),
                        None,
                    ))
                });
            }
            _ = tokio::time::sleep(MAILBOX_POLL_INTERVAL) => {
                let newest = session_manager
                    .pending_session_messages(&session_id)
                    .await
                    .ok()
                    .and_then(|pending| newest_interrupting_message(&pending));
                if newest.is_some() {
                    // Dropping the JoinHandle detaches the call; it keeps running.
                    guard.detach();
                    return Ok(interrupted_result(notice));
                }
            }
        }
    }
}

/// Carry the caller's tracing span and session task-locals into the spawned
/// call. A spawned task starts with neither, so without this the MCP request
/// would carry no `traceparent` (its server-side spans would lose their place
/// under the tool span) and no session ID.
fn with_caller_context<F>(call: F) -> impl std::future::Future<Output = F::Output> + Send + 'static
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    let span = tracing::Span::current();
    let session_id = crate::session_context::current_session_id();
    let telemetry_session_id = crate::session_context::current_telemetry_session_id();
    crate::session_context::with_telemetry_session_id(
        telemetry_session_id,
        crate::session_context::with_session_id(session_id, call),
    )
    .instrument(span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::session::SessionType;
    use tempfile::TempDir;

    #[tokio::test]
    async fn the_call_keeps_the_callers_span_and_session() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(SessionManager::new(directory.path().join("sessions")));
        let subscriber = tracing_subscriber::registry();
        let _guard = tracing::subscriber::set_default(subscriber);
        let span = tracing::info_span!("tool_call");
        let expected = span.id();
        let observed = crate::session_context::with_telemetry_session_id(
            Some("telemetry".to_string()),
            crate::session_context::with_session_id(
                Some("session".to_string()),
                interruptible_by_parent_message(
                    async {
                        let seen = (
                            tracing::Span::current().id(),
                            crate::session_context::current_session_id(),
                            crate::session_context::telemetry_session_id("fallback"),
                        );
                        Ok(CallToolResult::success(vec![ContentBlock::text(
                            serde_json::to_string(&(
                                seen.0.map(|id| id.into_u64()),
                                seen.1,
                                seen.2,
                            ))
                            .unwrap(),
                        )]))
                    },
                    Arc::clone(&manager),
                    "missing-session".to_string(),
                    "Interrupted.".to_string(),
                ),
            ),
        )
        .instrument(span)
        .await
        .unwrap();
        let text = observed.content[0].as_text().unwrap().text.clone();
        assert_eq!(
            text,
            serde_json::to_string(&(
                expected.map(|id| id.into_u64()),
                Some("session"),
                "telemetry"
            ))
            .unwrap()
        );
    }

    #[test]
    fn notice_comes_from_the_tool_definition() {
        let meta = serde_json::json!({ INTERRUPT_ON_MESSAGE_META_KEY: " Keep waiting later. " });
        assert_eq!(
            interrupt_notice(Some(&meta)).as_deref(),
            Some("Keep waiting later.")
        );
        assert_eq!(interrupt_notice(Some(&serde_json::json!({}))), None);
        assert_eq!(interrupt_notice(None), None);
    }

    #[tokio::test]
    async fn guidance_pending_at_wait_start_wakes_without_consuming_it() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(SessionManager::new(directory.path().to_path_buf()));
        let parent = manager
            .create_session(
                directory.path().to_path_buf(),
                "parent".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let child = manager
            .create_session(
                directory.path().to_path_buf(),
                "child".into(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        manager
            .update(&child.id)
            .parent_session_id(Some(parent.id.clone()))
            .apply()
            .await
            .unwrap();
        manager
            .send_to_child(&parent.id, &child.id, "already pending guidance")
            .await
            .unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        let stopped = Arc::new(tokio::sync::Notify::new());
        let task_token = token.clone();
        let task_stopped = stopped.clone();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            interruptible_by_parent_message(
                async move {
                    task_token.cancelled().await;
                    task_stopped.notify_one();
                    Ok(CallToolResult::success(vec![]))
                },
                manager.clone(),
                child.id.clone(),
                "Read pending guidance".into(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(was_interrupted(&result));
        assert_eq!(
            manager
                .pending_session_messages(&child.id)
                .await
                .unwrap()
                .len(),
            1
        );
        token.cancel();
        tokio::time::timeout(Duration::from_secs(1), stopped.notified())
            .await
            .expect("real cancellation reaches the detached call");
    }

    #[tokio::test]
    async fn cancelling_the_attached_wait_aborts_the_call() {
        struct Dropped(Arc<tokio::sync::Notify>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.notify_one();
            }
        }
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(SessionManager::new(directory.path().to_path_buf()));
        let started = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(tokio::sync::Notify::new());
        let call_started = started.clone();
        let call_dropped = dropped.clone();
        let waiting = tokio::spawn(interruptible_by_parent_message(
            async move {
                let _guard = Dropped(call_dropped);
                call_started.notify_one();
                std::future::pending::<ToolResult<CallToolResult>>().await
            },
            manager,
            "missing".into(),
            "notice".into(),
        ));
        started.notified().await;
        waiting.abort();
        let _ = waiting.await;
        tokio::time::timeout(Duration::from_secs(1), dropped.notified())
            .await
            .expect("attached call is aborted");
    }

    #[tokio::test]
    async fn a_parent_message_interrupts_without_cancelling_the_call() {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(SessionManager::new(directory.path().join("sessions")));
        let parent = manager
            .create_session(
                directory.path().to_path_buf(),
                "Coordinator".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let child = manager
            .create_session(
                directory.path().to_path_buf(),
                "Specialist".to_string(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        manager
            .update(&child.id)
            .parent_session_id(Some(parent.id.clone()))
            .apply()
            .await
            .unwrap();
        let finished = Arc::new(tokio::sync::Notify::new());
        let observed = Arc::clone(&finished);
        let call = async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            observed.notify_one();
            Ok(CallToolResult::success(vec![ContentBlock::text("done")]))
        };
        let waiting = tokio::spawn(interruptible_by_parent_message(
            call,
            Arc::clone(&manager),
            child.id.clone(),
            "Interrupted; call again to keep waiting.".to_string(),
        ));
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(!waiting.is_finished());
        manager
            .send_to_child(&parent.id, &child.id, "Use four slides")
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("the newer message interrupts the wait")
            .unwrap()
            .unwrap();
        assert!(was_interrupted(&result));
        assert_eq!(
            result.structured_content,
            Some(serde_json::json!({ "interrupted": true }))
        );
        // The detached call keeps running to completion.
        tokio::time::timeout(Duration::from_secs(3), finished.notified())
            .await
            .expect("the detached call still finishes");
    }
}
