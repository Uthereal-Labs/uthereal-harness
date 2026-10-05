use crate::{
    agents::{subagent_task_config::TaskConfig, Agent, AgentConfig, AgentEvent, SessionConfig},
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    prompt_template::render_template,
    recipe::Recipe,
};
use anyhow::{anyhow, Result};
use futures::StreamExt;
use rmcp::model::{ErrorCode, ErrorData, Notification, ServerNotification};
#[expect(deprecated)]
use rmcp::model::{LoggingLevel, LoggingMessageNotificationParam};
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, Instrument};

pub type OnMessageCallback = Arc<dyn Fn(&Message) + Send + Sync>;

#[derive(Serialize)]
pub struct SubagentPromptContext {
    pub max_turns: usize,
    pub task_instructions: String,
    pub tool_count: usize,
    pub available_tools: String,
}

type AgentMessagesFuture =
    Pin<Box<dyn Future<Output = Result<(Conversation, Option<String>)>> + Send>>;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub task_config: TaskConfig,
    pub return_last_only: bool,
    pub session_id: String,
    pub cancellation_token: Option<CancellationToken>,
    pub on_message: Option<OnMessageCallback>,
    pub notification_tx: Option<tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
    pub parent_span: tracing::Span,
    pub telemetry_session_id: String,
}

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String, anyhow::Error> {
    let return_last_only = params.return_last_only;
    let telemetry_session_id = params.telemetry_session_id.clone();
    let parent_span = params.parent_span.clone();
    let execution = get_agent_messages(params).instrument(parent_span);
    let (messages, final_output) =
        crate::session_context::with_telemetry_session_id(Some(telemetry_session_id), execution)
            .await
            .map_err(|e| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to execute task: {}", e),
                    None,
                )
            })?;

    if let Some(output) = final_output {
        return Ok(output);
    }

    if let Some(error) = terminal_subagent_failure(&messages) {
        return Err(anyhow!(error));
    }

    Ok(extract_response_text(&messages, return_last_only))
}

fn terminal_subagent_failure(messages: &Conversation) -> Option<String> {
    let message = messages.messages().last()?;
    if message.as_concat_text() == crate::agents::state_machine::MAX_TURNS_MESSAGE {
        return Some("Subagent reached its maximum action limit".to_string());
    }
    message
        .content
        .iter()
        .find_map(|content| content.as_error())
        .map(|error| format!("Subagent failed: {}", error.message))
}

fn extract_response_text(messages: &Conversation, return_last_only: bool) -> String {
    if return_last_only {
        messages
            .messages()
            .last()
            .and_then(|message| {
                message.content.iter().find_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| String::from("No text content in last message"))
    } else {
        let all_text_content: Vec<String> = messages
            .iter()
            .flat_map(|message| {
                message.content.iter().filter_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    crate::conversation::message::MessageContent::ToolResponse(tool_response) => {
                        if let Ok(result) = &tool_response.tool_result {
                            let texts: Vec<String> = result
                                .content
                                .iter()
                                .filter_map(|content| {
                                    if let rmcp::model::ContentBlock::Text(raw_text_content) =
                                        content
                                    {
                                        Some(raw_text_content.text.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !texts.is_empty() {
                                Some(format!("Tool result: {}", texts.join("\n")))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
            })
            .collect();

        all_text_content.join("\n")
    }
}

pub const SUBAGENT_TOOL_REQUEST_TYPE: &str = "subagent_tool_request";
pub const SUBAGENT_TOOL_RESPONSE_TYPE: &str = "subagent_tool_response";

struct TaskCleanup {
    signal: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}
impl TaskCleanup {
    fn new(agent: Arc<Agent>, session_id: String) -> Self {
        let (signal, receiver) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            // Sender drop (including cancellation and panic) starts cleanup too.
            let _ = receiver.await;
            if tokio::time::timeout(
                std::time::Duration::from_secs(6),
                agent.extension_manager.notify_task_ended(&session_id),
            )
            .await
            .is_err()
            {
                tracing::warn!("Task {session_id} tool-server cleanup deadline exceeded");
            }
            let settle = async {
                loop {
                    let manager = &agent.config.session_manager;
                    if manager.close_task_notice_admission(&session_id).await? {
                        return anyhow::Ok(());
                    }
                    for notice in manager.pending_session_messages(&session_id).await? {
                        if notice.kind == crate::session::MailboxMessageKind::Channel {
                            manager
                                .deliver_session_message(&session_id, notice.id, &notice.prompt()?)
                                .await?;
                        }
                    }
                }
            };
            match tokio::time::timeout(std::time::Duration::from_secs(5), settle).await {
                Ok(Ok(())) => {}
                outcome => {
                    tracing::warn!("Task {session_id} mailbox cleanup did not settle: {outcome:?}")
                }
            }
        });
        Self {
            signal: Some(signal),
            handle: Some(handle),
        }
    }
    async fn finish(mut self) {
        if let Some(signal) = self.signal.take() {
            let _ = signal.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

fn get_agent_messages(params: SubagentRunParams) -> AgentMessagesFuture {
    Box::pin(async move {
        let SubagentRunParams {
            config,
            recipe,
            mut task_config,
            session_id,
            cancellation_token,
            on_message,
            notification_tx,
            ..
        } = params;

        let system_instructions = recipe.instructions.clone().unwrap_or_default();
        let user_task = recipe
            .prompt
            .clone()
            .unwrap_or_else(|| "Begin.".to_string());

        let agent = Arc::new(Agent::with_config(config));

        if let Some(crate::agents::ExtensionConfig::Platform {
            available_tools, ..
        }) = task_config
            .extensions
            .iter_mut()
            .find(|extension| extension.name() == "summon")
        {
            // Summon lists wait only to a specialist that starts editor tasks.
            for tool in ["message_parent", "wait"] {
                if !available_tools.is_empty() && !available_tools.iter().any(|name| name == tool) {
                    available_tools.push(tool.to_string());
                }
            }
        } else {
            task_config
                .extensions
                .push(crate::agents::ExtensionConfig::Platform {
                    name: "summon".to_string(),
                    description: String::new(),
                    display_name: None,
                    bundled: None,
                    available_tools: vec!["message_parent".to_string(), "wait".to_string()],
                });
        }

        agent
            .update_provider(
                task_config.provider.clone(),
                task_config.model_config.clone(),
                &session_id,
            )
            .await
            .map_err(|e| anyhow!("Failed to set provider on sub agent: {}", e))?;

        for extension in &task_config.extensions {
            let extension_name = extension.name();
            if let Err(e) = agent.add_extension(extension.clone(), &session_id).await {
                if task_config
                    .required_extension_names
                    .iter()
                    .any(|required| required == &extension_name)
                {
                    return Err(anyhow!(
                        "Failed to load required extension '{}': {}",
                        extension_name,
                        e
                    ));
                }
                debug!(
                    "Failed to add extension '{}' to subagent: {}",
                    extension_name, e
                );
            }
        }

        let has_response_schema = recipe.response.is_some();
        agent
            .apply_recipe_components(recipe.response.clone(), true)
            .await?;

        let max_turns = task_config
            .max_turns
            .expect("TaskConfig always sets max_turns");
        let subagent_prompt =
            build_subagent_prompt(&agent, max_turns, &session_id, system_instructions).await?;
        agent.override_system_prompt(subagent_prompt).await;

        let user_message =
            Message::user().with_text(format!("Subagent ID: {session_id}\n\n{user_task}"));
        let mut conversation = Conversation::new_unvalidated(vec![user_message.clone()]);

        agent
            .config
            .session_manager
            .update(&session_id)
            .recipe(Some(recipe.clone()))
            .apply()
            .await?;

        if let Some(activities) = recipe.activities {
            for activity in activities {
                info!("Recipe activity: {}", activity);
            }
        }
        let session_config = SessionConfig {
            id: session_id.clone(),
            schedule_id: None,
            max_turns: task_config.max_turns.map(|v| v as u32),
            retry_config: recipe.retry,
        };

        let cleanup = TaskCleanup::new(Arc::clone(&agent), session_id.clone());
        let mut stream =
            crate::session_context::with_session_id(Some(session_id.to_string()), async {
                agent
                    .reply(
                        user_message,
                        session_config,
                        crate::agents::state_machine::enabled(),
                        cancellation_token,
                    )
                    .await
            })
            .await
            .map_err(|e| anyhow!("Failed to get reply from agent: {}", e))?;

        let streamed = stream_subagent_messages(
            &mut stream,
            &session_id,
            on_message.as_ref(),
            notification_tx.as_ref(),
            &mut conversation,
        )
        .await;
        drop(stream);
        // The task's tool servers stop the work it started (such as editor
        // tasks) and settle what it owed others.
        cleanup.finish().await;
        streamed?;

        let final_output = get_final_output(&agent, has_response_schema).await;

        Ok((conversation, final_output))
    })
}

async fn stream_subagent_messages(
    stream: &mut (impl futures::Stream<Item = Result<AgentEvent>> + Unpin),
    session_id: &str,
    on_message: Option<&OnMessageCallback>,
    notification_tx: Option<&tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
    conversation: &mut Conversation,
) -> Result<()> {
    while let Some(message_result) = stream.next().await {
        match message_result {
            Ok(AgentEvent::Message(msg)) => {
                if let Some(callback) = on_message {
                    callback(&msg);
                }
                if let Some(tx) = notification_tx {
                    for content in &msg.content {
                        if let Some(notif) = create_tool_notification(content, session_id) {
                            if tx.send(notif).is_err() {
                                debug!("Notification receiver dropped for subagent {}", session_id);
                            }
                        }
                    }
                }
                conversation.push(msg);
            }
            Ok(AgentEvent::Usage(_)) => {}
            Ok(AgentEvent::MessageUsage { .. }) => {}
            Ok(AgentEvent::McpNotification(_)) => {}
            Ok(AgentEvent::HistoryReplaced(updated_conversation)) => {
                *conversation = updated_conversation;
            }
            Err(e) => {
                return Err(anyhow!("Subagent stream failed: {e}"));
            }
        }
    }
    Ok(())
}

async fn build_subagent_prompt(
    agent: &Agent,
    max_turns: usize,
    session_id: &str,
    system_instructions: String,
) -> Result<String> {
    let mut tool_names: Vec<_> = agent
        .list_tools(session_id, None)
        .await
        .into_iter()
        .filter(super::reply_parts::is_tool_visible_to_model)
        .map(|t| t.name.to_string())
        .collect();
    tool_names.sort_unstable();
    render_template(
        "subagent_system.md",
        &SubagentPromptContext {
            max_turns,
            task_instructions: system_instructions,
            tool_count: tool_names.len(),
            available_tools: tool_names.join(", "),
        },
    )
    .map_err(|e| anyhow!("Failed to render subagent system prompt: {}", e))
}

async fn get_final_output(agent: &Agent, has_response_schema: bool) -> Option<String> {
    if has_response_schema {
        agent
            .final_output_tool
            .lock()
            .await
            .as_ref()
            .and_then(|tool| tool.final_output.clone())
    } else {
        None
    }
}

#[expect(deprecated)]
pub fn create_tool_notification(
    content: &MessageContent,
    subagent_id: &str,
) -> Option<ServerNotification> {
    let data = if let MessageContent::ToolRequest(req) = content {
        let tool_call = req.tool_call.as_ref().ok()?;
        serde_json::json!({
            "type": SUBAGENT_TOOL_REQUEST_TYPE,
            "subagent_id": subagent_id,
            "tool_call": {"id": req.id, "name": tool_call.name, "arguments": tool_call.arguments}
        })
    } else if let MessageContent::ToolResponse(response) = content {
        serde_json::json!({
            "type": SUBAGENT_TOOL_RESPONSE_TYPE,
            "subagent_id": subagent_id,
            "tool_call": {"id": response.id},
            "status": if response.tool_result.as_ref().is_ok_and(|result| result.is_error != Some(true)) { "completed" } else { "failed" }
        })
    } else {
        return None;
    };
    Some(ServerNotification::LoggingMessageNotification(
        Notification::new(
            LoggingMessageNotificationParam::new(LoggingLevel::Info, data)
                .with_logger(format!("subagent:{}", subagent_id)),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        create_tool_notification, terminal_subagent_failure, SUBAGENT_TOOL_REQUEST_TYPE,
        SUBAGENT_TOOL_RESPONSE_TYPE,
    };
    use crate::conversation::message::{Message, MessageContent, MessageErrorKind};
    use crate::conversation::Conversation;
    use rmcp::model::{CallToolRequestParams, ServerNotification};
    use serde_json::json;

    #[tokio::test]
    async fn dropped_execution_settles_durable_notices_and_closes_admission() -> anyhow::Result<()>
    {
        use crate::agents::{Agent, AgentConfig, GoosePlatform};
        use crate::config::{permission::PermissionManager, GooseMode};
        use crate::session::{SessionManager, SessionType};
        use std::sync::Arc;
        let directory = tempfile::tempdir()?;
        let manager = Arc::new(SessionManager::new(directory.path().to_path_buf()));
        let parent = manager
            .create_session(
                directory.path().into(),
                "parent".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?;
        let lease = manager
            .claim_prompt_attempt(
                &uuid::Uuid::new_v4().to_string(),
                &"a".repeat(64),
                &parent.id,
            )
            .await?
            .unwrap();
        let child = manager
            .create_session(
                directory.path().into(),
                "child".into(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await?;
        let admission = manager
            .capture_task_admission(&parent.id, &child.id, "specialist")
            .await?;
        let mut data = child.extension_data;
        data.set_extension_state(
            "summon",
            "task_admission_v1",
            serde_json::to_value(admission)?,
        );
        manager
            .update(&child.id)
            .parent_session_id(Some(parent.id))
            .extension_data(data)
            .apply()
            .await?;
        let notice = goose_sdk_types::custom_requests::TaskNoticeRequest {
            attempt_key: lease.key,
            task_id: Some(child.id.clone()),
            artifact_keys: vec![],
            text: "Durable notice".into(),
            wake: true,
            refresh_tools: false,
            dedupe_key: Some("cleanup:notice".into()),
            editor_result: None,
            channel_wait: None,
        };
        manager.queue_task_notice(&notice).await?;
        let agent = Arc::new(Agent::with_config(AgentConfig::new(
            Arc::clone(&manager),
            Arc::new(PermissionManager::new(directory.path().join("permissions"))),
            None,
            GooseMode::Auto,
            true,
            GoosePlatform::GooseCli,
        )));
        let cleanup = super::TaskCleanup::new(agent, child.id.clone());
        drop(cleanup);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let session = manager.get_session(&child.id, true).await?;
                if session
                    .extension_data
                    .get_extension_state("summon", "notice_closed_v1")
                    == Some(&json!(true))
                {
                    assert!(manager
                        .pending_session_messages(&child.id)
                        .await?
                        .is_empty());
                    assert!(session
                        .conversation
                        .unwrap()
                        .messages()
                        .iter()
                        .any(|message| message.as_concat_text() == "Durable notice"));
                    return anyhow::Ok(());
                }
                tokio::task::yield_now().await;
            }
        })
        .await??;
        // Retried acceptance remains acknowledged, while a new notice is refused.
        assert_eq!(
            manager.queue_task_notice(&notice).await?.status,
            goose_sdk_types::custom_requests::TaskNoticeStatus::Queued
        );
        let fresh = goose_sdk_types::custom_requests::TaskNoticeRequest {
            dedupe_key: Some("cleanup:fresh".into()),
            ..notice
        };
        assert_eq!(
            manager.queue_task_notice(&fresh).await?.status,
            goose_sdk_types::custom_requests::TaskNoticeStatus::NotRunning
        );
        Ok(())
    }

    #[test]
    #[expect(deprecated)]
    fn create_tool_notification_for_tool_request() {
        let tool_call = CallToolRequestParams::new("developer__shell".to_string())
            .with_arguments(json!({"command": "ls"}).as_object().unwrap().clone());
        let content = MessageContent::tool_request("req1", Ok(tool_call));
        let notification =
            create_tool_notification(&content, "session_1").expect("expected notification");

        let ServerNotification::LoggingMessageNotification(log_notif) = notification else {
            panic!("expected logging notification");
        };
        let data = log_notif
            .params
            .data
            .as_object()
            .expect("expected object data");
        assert_eq!(
            data.get("type").and_then(|v| v.as_str()),
            Some(SUBAGENT_TOOL_REQUEST_TYPE)
        );
        assert_eq!(
            data.get("subagent_id").and_then(|v| v.as_str()),
            Some("session_1")
        );
        let tool_call = data
            .get("tool_call")
            .and_then(|v| v.as_object())
            .expect("expected tool_call object");
        assert_eq!(
            tool_call.get("name").and_then(|v| v.as_str()),
            Some("developer__shell")
        );
    }

    #[test]
    fn create_tool_notification_ignores_non_tool_request() {
        let content = MessageContent::text("hello");
        assert!(create_tool_notification(&content, "session_1").is_none());
    }

    #[test]
    #[expect(deprecated)]
    fn create_tool_notification_for_tool_response_exposes_only_status() {
        let content = MessageContent::tool_response(
            "req1",
            Ok(rmcp::model::CallToolResult::success(vec![
                rmcp::model::ContentBlock::text("private result"),
            ])),
        );
        let notification =
            create_tool_notification(&content, "session_1").expect("expected notification");
        let ServerNotification::LoggingMessageNotification(log_notif) = notification else {
            panic!("expected logging notification");
        };
        let data = log_notif
            .params
            .data
            .as_object()
            .expect("expected object data");
        assert_eq!(data["type"], SUBAGENT_TOOL_RESPONSE_TYPE);
        assert_eq!(data["tool_call"]["id"], "req1");
        assert_eq!(data["status"], "completed");
        assert!(!log_notif.params.data.to_string().contains("private result"));
    }

    #[test]
    fn terminal_error_message_fails_subagent() {
        let conversation = Conversation::new_unvalidated(vec![Message::assistant().with_content(
            MessageContent::error(MessageErrorKind::Other, "provider unavailable"),
        )]);

        assert_eq!(
            terminal_subagent_failure(&conversation).as_deref(),
            Some("Subagent failed: provider unavailable")
        );
    }

    #[test]
    fn max_turn_message_fails_subagent() {
        let conversation = Conversation::new_unvalidated(vec![
            Message::assistant().with_text(crate::agents::state_machine::MAX_TURNS_MESSAGE)
        ]);

        assert_eq!(
            terminal_subagent_failure(&conversation).as_deref(),
            Some("Subagent reached its maximum action limit")
        );
    }
}
