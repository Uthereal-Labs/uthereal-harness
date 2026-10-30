//! Covers `Agent::reply_with_state_machine`, the entry point the CLI and desktop
//! reach when the state machine is enabled.

use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    Annotations as AcpAnnotations, ContentBlock as AcpContentBlock, EmbeddedResource,
    EmbeddedResourceResource, ResourceLink, Role as AcpRole, TextContent as AcpTextContent,
    TextResourceContents,
};
use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use super::calculator_extension::{value, CalculatorExtension, ADD};
use super::dummy_api::{DummyApi, ProviderFeatures};
use crate::acp::server::GooseAcpAgent;
use crate::agents::extension::ExtensionConfig;
use crate::agents::mcp_client::McpClientTrait;
use crate::agents::{Agent, AgentConfig, AgentEvent, GoosePlatform, SessionConfig};
use crate::config::permission::PermissionManager;
use crate::config::GooseMode;
use crate::conversation::message::{ActionRequiredData, Message, MessageContent};
use crate::permission::Permission;
use crate::providers::base::Provider;
use crate::session::{SessionManager, SessionType};
use goose_providers::model::ModelConfig;

async fn agent_with_dummy_api() -> Result<(Agent, Arc<DummyApi>, String, tempfile::TempDir)> {
    let api = Arc::new(DummyApi::start(ProviderFeatures::default()).await);
    let api_client = goose_providers::api_client::ApiClient::new_with_tls(
        api.uri(),
        goose_providers::api_client::AuthMethod::NoAuth,
        None,
    )?;
    let provider: Arc<dyn Provider> = Arc::new(
        goose_providers::openai::OpenAiProviderBuilder::new(api_client)
            .name("openai")
            .build(),
    );

    let temp_dir = tempfile::tempdir()?;
    let session_manager = Arc::new(SessionManager::new(temp_dir.path().to_path_buf()));
    let session = session_manager
        .create_session(
            temp_dir.path().to_path_buf(),
            "state-machine-reply".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let agent = Agent::with_config(AgentConfig::new(
        session_manager,
        Arc::new(PermissionManager::new(temp_dir.path().join("permissions"))),
        None,
        GooseMode::Auto,
        true,
        GoosePlatform::GooseCli,
    ));
    agent
        .update_provider(
            provider,
            ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                .with_canonical_limits("openai"),
            &session.id,
        )
        .await?;

    Ok((agent, api, session.id, temp_dir))
}

async fn agent_with_calculator() -> Result<(
    Agent,
    Arc<DummyApi>,
    String,
    Arc<CalculatorExtension>,
    tempfile::TempDir,
)> {
    let (agent, api, session_id, temp_dir) = agent_with_dummy_api().await?;
    agent
        .update_goose_mode(GooseMode::Approve, &session_id)
        .await?;
    let calculator = Arc::new(CalculatorExtension::new(
        agent.config.session_manager.action_required(),
    ));
    agent
        .extension_manager
        .add_client(
            "calculator".to_string(),
            ExtensionConfig::Platform {
                name: "calculator".to_string(),
                description: "Stateful test calculator".to_string(),
                display_name: None,
                bundled: None,
                available_tools: vec![],
            },
            calculator.clone(),
            calculator.get_info().cloned(),
        )
        .await;
    Ok((agent, api, session_id, calculator, temp_dir))
}

fn confirmation_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::ActionRequired(action) => match &action.data {
                ActionRequiredData::ToolConfirmation { id, .. } => Some(id.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

async fn stream_messages(
    mut stream: futures::stream::BoxStream<'_, Result<AgentEvent>>,
) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            messages.push(message);
        }
    }
    Ok(messages)
}

#[tokio::test]
async fn durable_both_loops_restore_wait_into_empty_native_storage() -> Result<()> {
    use crate::session::durable::{self, AcceptedReply, DurableAdmission, DurableSnapshot};
    use serde_json::{json, Value};
    use wiremock::{matchers::method, Mock, MockServer, Request, ResponseTemplate};

    for state_machine in [false, true] {
        let callback = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(|request: &Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                ResponseTemplate::new(200)
                    .set_body_json(json!({"accepted":true,"generation":body["generation"]}))
            })
            .mount(&callback)
            .await;
        let (first, api, _initial_root, calculator, first_dir) = agent_with_calculator().await?;
        let root = format!("durable_{}", uuid::Uuid::new_v4());
        first
            .config
            .session_manager
            .prepare_durable_session(
                &root,
                first_dir.path(),
                "durable",
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await?;
        first
            .update_provider(
                first.provider().await?,
                ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                    .with_canonical_limits("openai"),
                &root,
            )
            .await?;
        first.update_goose_mode(GooseMode::Auto, &root).await?;
        let manager = first.config.session_manager.clone();
        let admission = DurableAdmission {
            logical_run_id: "loop-test".into(),
            attempt_id: uuid::Uuid::new_v4().to_string(),
            epoch: 1,
            request_digest: "a".repeat(64),
            expected_generation: 0,
            callback_url: callback.uri(),
            callback_token: "old-authority".into(),
            restore: None,
            accepted_reply: None,
            continuation_context: None,
        };
        durable::admit(manager.clone(), &root, admission.clone()).await?;
        durable::bind_execution(
            &manager,
            &root,
            "first-native-run",
            Some(&admission.attempt_id),
        )
        .await?;
        api.on("ask the human")
            .call(super::calculator_extension::PAUSE, json!({}));
        let config = SessionConfig {
            id: root.clone(),
            schedule_id: None,
            max_turns: Some(4),
            retry_config: None,
        };
        tokio::time::timeout(
            Duration::from_secs(10),
            stream_messages(
                first
                    .reply(
                        Message::user().with_text("ask the human"),
                        config.clone(),
                        state_machine,
                        Some(CancellationToken::new()),
                    )
                    .await?,
            ),
        )
        .await??;
        assert!(durable::suspended(&root).await, "loop={state_machine}");
        assert_eq!(api.call_count(), 1, "loop={state_machine}");
        let contexts = calculator.contexts();
        assert_eq!(contexts.len(), 1);
        assert!(manager
            .get_session(&root, true)
            .await?
            .conversation
            .unwrap()
            .messages()
            .iter()
            .all(|message| message.get_tool_response_ids().is_empty()));
        durable::parked_receipt(&root).await?;
        let requests = callback.received_requests().await.unwrap();
        let last: Value = serde_json::from_slice(&requests.last().unwrap().body)?;
        let snapshot: DurableSnapshot = serde_json::from_value(last["snapshot"].clone())?;
        let call = snapshot.pending_wait.as_ref().unwrap().tool_call_id.clone();
        durable::release(&root);
        drop(first);
        drop(manager);
        drop(first_dir);

        let (second, second_api, _unused_session, fresh_calculator, new_dir) =
            agent_with_calculator().await?;
        let manager = second.config.session_manager.clone();
        manager
            .prepare_durable_session(
                &root,
                new_dir.path(),
                "restored",
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await?;
        second
            .update_provider(
                second.provider().await?,
                ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                    .with_canonical_limits("openai"),
                &root,
            )
            .await?;
        second.update_goose_mode(GooseMode::Auto, &root).await?;
        let mut next = admission;
        next.attempt_id = uuid::Uuid::new_v4().to_string();
        next.epoch = 2;
        next.callback_token = "fresh-authority".into();
        next.continuation_context = Some(durable::ContinuationContext { id:"events-through-7".into(), text:"Authenticated intervening conversation: the manuscript now uses the latest terminology.".into() });
        next.expected_generation = snapshot.generation;
        next.restore = Some(snapshot);
        next.accepted_reply = Some(AcceptedReply {
            wait_id: "question-1".into(),
            tool_call_id: call.clone(),
            input_revision: 1,
            result: rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(
                "human says add two",
            )]),
        });
        let message = durable::admit(manager.clone(), &root, next).await?.unwrap();
        durable::bind_execution(&manager, &root, "restored-native-run", None).await?;
        second_api.on("human says add two").call(ADD, value(2));
        second_api
            .on("result: 2")
            .reply("completed with accepted answer");
        let output = tokio::time::timeout(
            Duration::from_secs(10),
            stream_messages(
                second
                    .reply(
                        message,
                        config,
                        state_machine,
                        Some(CancellationToken::new()),
                    )
                    .await?,
            ),
        )
        .await??;
        assert!(
            output
                .iter()
                .map(|message| message.as_concat_text())
                .collect::<String>()
                .contains("completed with accepted answer"),
            "loop={state_machine}: {output:?}"
        );
        assert_eq!(fresh_calculator.total(), 2, "loop={state_machine}");
        assert!(second_api.calls()[0].input_contains("Authenticated intervening conversation"));
        assert_eq!(
            fresh_calculator.contexts().len(),
            1,
            "suspended ask must not be repeated"
        );
        assert_eq!(
            manager
                .get_session(&root, true)
                .await?
                .conversation
                .unwrap()
                .messages()
                .iter()
                .flat_map(|message| message.get_tool_response_ids())
                .filter(|id| *id == call)
                .count(),
            1
        );
        durable::release(&root);
    }
    Ok(())
}

#[tokio::test]
async fn state_machine_confirmation_through_agent_resumes_tool_call() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    let (agent, api, session_id, calculator, _temp_dir) = agent_with_calculator().await?;
    let agent = Arc::new(agent);

    api.on("add one").call(ADD, value(1));
    api.on("result: 1").reply("the result is one");

    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let mut stream = agent
        .reply(
            Message::user().with_text("add one"),
            session_config.clone(),
            true,
            Some(CancellationToken::new()),
        )
        .await?;
    let mut messages = Vec::new();
    let confirmation_id = loop {
        let event = stream
            .next()
            .await
            .expect("state machine should request confirmation")?;
        if let AgentEvent::Message(message) = event {
            let confirmation_id = confirmation_ids(std::slice::from_ref(&message)).pop();
            messages.push(message);
            if let Some(confirmation_id) = confirmation_id {
                break confirmation_id;
            }
        }
    };
    assert_eq!(calculator.total(), 0);
    {
        let session = agent
            .config
            .session_manager
            .get_session(&session_config.id, true)
            .await?;
        assert!(confirmation_ids(
            session
                .conversation
                .as_ref()
                .expect("session conversation")
                .messages()
        )
        .contains(&confirmation_id));
    }

    agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await?;
    {
        let session = agent
            .config
            .session_manager
            .get_session(&session_config.id, true)
            .await?;
        assert!(session
            .conversation
            .as_ref()
            .expect("session conversation")
            .messages()
            .iter()
            .any(|message| {
                message.content.iter().any(|content| {
                    matches!(
                        content,
                        MessageContent::ActionRequired(action)
                            if matches!(
                                &action.data,
                                ActionRequiredData::ToolConfirmationResponse { id, permission }
                                    if id == &confirmation_id && permission == &Permission::AllowOnce
                            )
                    )
                })
            }));
    }
    agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await?;
    assert!(agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::DenyOnce)
        .await
        .is_err());
    drop(stream);
    let stream = agent
        .resume_state_machine_turn(session_config.clone(), CancellationToken::new())
        .await?
        .expect("persisted confirmation response should resume the state-machine turn");
    messages.extend(stream_messages(stream).await?);
    assert!(messages.iter().any(|message| message
        .get_tool_response_ids()
        .contains(&confirmation_id.as_str())));
    assert_eq!(calculator.total(), 1);
    assert_eq!(api.call_count(), 2);

    assert!(agent
        .submit_tool_confirmation(&session_config.id, &confirmation_id, Permission::AllowOnce)
        .await
        .is_err());
    assert_eq!(calculator.total(), 1);

    assert!(agent
        .submit_tool_confirmation(&session_config.id, "stale-request", Permission::AllowOnce)
        .await
        .is_err());

    let session = agent
        .config
        .session_manager
        .get_session(&session_config.id, true)
        .await?;
    let messages = session
        .conversation
        .as_ref()
        .expect("session conversation")
        .messages();
    let confirmation_responses = messages
        .iter()
        .filter(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    MessageContent::ActionRequired(action)
                        if matches!(
                            &action.data,
                            ActionRequiredData::ToolConfirmationResponse { id, .. }
                                if id == &confirmation_id
                        )
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(confirmation_responses.len(), 1);
    assert!(!confirmation_responses[0].is_user_visible());
    assert!(!confirmation_responses[0].is_agent_visible());
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.role == rmcp::model::Role::User
                    && message.is_user_visible()
                    && !message.is_tool_response()
            })
            .count(),
        1
    );

    Ok(())
}

#[tokio::test]
async fn reply_streams_the_turn_and_ends() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("are you there?").reply("still here");

    let session_config = SessionConfig {
        id: session_id.clone(),
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply_with_state_machine(
            Message::user().with_text("are you there?"),
            session_config,
            Some(CancellationToken::new()),
        )
        .await?;

    let replies = tokio::time::timeout(Duration::from_secs(30), async move {
        tokio::pin!(stream);
        let mut replies = Vec::new();
        while let Some(event) = stream.next().await {
            if let AgentEvent::Message(message) = event? {
                replies.push(message.as_concat_text());
            }
        }
        anyhow::Ok(replies)
    })
    .await??;

    assert!(
        replies.iter().any(|reply| reply == "still here"),
        "expected the scripted reply, got {replies:?}"
    );
    assert_eq!(api.call_count(), 1);

    Ok(())
}

#[tokio::test]
async fn bang_shell_uses_state_machine_when_explicitly_enabled() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let session_config = SessionConfig {
        id: session_id,
        schedule_id: None,
        max_turns: Some(2),
        retry_config: None,
    };
    let stream = agent
        .reply(
            Message::user().with_text("!echo hello"),
            session_config,
            true,
            Some(CancellationToken::new()),
        )
        .await?;
    tokio::pin!(stream);
    let mut requested_shell = false;
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            requested_shell |= message.content.iter().any(|content| {
                matches!(
                    content,
                    crate::conversation::message::MessageContent::ToolRequest(request)
                        if request.tool_call.as_ref().is_ok_and(|call| call.name == "shell")
                )
            });
        }
    }

    assert!(requested_shell);
    assert_eq!(api.call_count(), 0);

    Ok(())
}

async fn reply_messages(
    agent: &Agent,
    session_id: String,
    message: Message,
) -> Result<Vec<Message>> {
    let stream = agent
        .reply(
            message,
            SessionConfig {
                id: session_id,
                schedule_id: None,
                max_turns: Some(2),
                retry_config: None,
            },
            crate::agents::state_machine::enabled(),
            Some(CancellationToken::new()),
        )
        .await?;
    tokio::pin!(stream);
    let mut messages = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Message(message) = event? {
            messages.push(message);
        }
    }
    Ok(messages)
}

fn assistant_only_acp_annotations() -> AcpAnnotations {
    AcpAnnotations::new().audience(vec![AcpRole::Assistant])
}

fn assistant_only_acp_text(text: &str) -> AcpContentBlock {
    AcpContentBlock::Text(AcpTextContent::new(text).annotations(assistant_only_acp_annotations()))
}

fn empty_audience_acp_annotations() -> AcpAnnotations {
    AcpAnnotations::new().audience(Vec::new())
}

fn empty_audience_acp_text(text: &str) -> AcpContentBlock {
    AcpContentBlock::Text(AcpTextContent::new(text).annotations(empty_audience_acp_annotations()))
}

fn assistant_only_embedded_resource(text: &str) -> AcpContentBlock {
    AcpContentBlock::Resource(
        EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new(text, "file:///hidden-resource.txt"),
        ))
        .annotations(assistant_only_acp_annotations()),
    )
}

fn empty_audience_embedded_resource(text: &str) -> AcpContentBlock {
    AcpContentBlock::Resource(
        EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new(text, "file:///empty-audience-resource.txt"),
        ))
        .annotations(empty_audience_acp_annotations()),
    )
}

fn assistant_only_resource_link(text: &str) -> Result<(AcpContentBlock, tempfile::NamedTempFile)> {
    let file = tempfile::NamedTempFile::new()?;
    std::fs::write(file.path(), text)?;
    let uri = url::Url::from_file_path(file.path())
        .map_err(|()| anyhow::anyhow!("temporary resource path is not a valid file URL"))?;
    let link = ResourceLink::new("hidden-resource.txt", uri.to_string())
        .annotations(assistant_only_acp_annotations());
    Ok((AcpContentBlock::ResourceLink(link), file))
}

fn shell_commands(messages: &[Message]) -> Vec<&str> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            MessageContent::ToolRequest(request) => request
                .tool_call
                .as_ref()
                .ok()
                .filter(|call| call.name == "shell")
                .and_then(|call| call.arguments.as_ref())
                .and_then(|arguments| arguments.get("command"))
                .and_then(serde_json::Value::as_str),
            _ => None,
        })
        .collect()
}

async fn assert_bang_shell_uses_only_user_visible_content() -> Result<()> {
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let hidden_text_prefix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        assistant_only_acp_text("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_text_prefix).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let empty_audience_text = GooseAcpAgent::convert_acp_prompt_to_message(&[
        empty_audience_acp_text("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, empty_audience_text).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let hidden_text_suffix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        AcpContentBlock::Text(AcpTextContent::new("!echo visible")),
        assistant_only_acp_text("&& echo hidden"),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_text_suffix).await?;
    assert_eq!(shell_commands(&messages), ["echo visible"]);
    assert_eq!(api.call_count(), 0);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let hidden_resource_prefix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        assistant_only_embedded_resource("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, hidden_resource_prefix).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("benign visible input")
        .reply("handled as ordinary input");
    let empty_audience_resource = GooseAcpAgent::convert_acp_prompt_to_message(&[
        empty_audience_embedded_resource("!echo hidden"),
        AcpContentBlock::Text(AcpTextContent::new("benign visible input")),
    ]);
    let messages = reply_messages(&agent, session_id, empty_audience_resource).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);

    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    let (hidden_link, _resource_file) = assistant_only_resource_link("&& echo hidden")?;
    let hidden_link_suffix = GooseAcpAgent::convert_acp_prompt_to_message(&[
        AcpContentBlock::Text(AcpTextContent::new("!echo visible")),
        hidden_link,
    ]);
    let messages = reply_messages(&agent, session_id, hidden_link_suffix).await?;
    assert_eq!(shell_commands(&messages), ["echo visible"]);
    assert_eq!(api.call_count(), 0);

    Ok(())
}

#[tokio::test]
async fn bang_shell_not_executed_in_legacy_loop() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", None::<&str>)]);
    let (agent, api, session_id, _temp_dir) = agent_with_dummy_api().await?;
    api.on("!echo hello").reply("treated as text");
    let messages =
        reply_messages(&agent, session_id, Message::user().with_text("!echo hello")).await?;
    assert!(shell_commands(&messages).is_empty());
    assert_eq!(api.call_count(), 1);
    Ok(())
}

#[tokio::test]
async fn bang_shell_visibility_is_enforced_when_state_machine_is_enabled() -> Result<()> {
    let _guard = env_lock::lock_env([("GOOSE_STATE_MACHINE", Some("1"))]);
    assert_bang_shell_uses_only_user_visible_content().await
}

#[tokio::test]
async fn both_loops_wait_for_durable_editor_result_before_finishing() -> Result<()> {
    for state_machine in [false, true] {
        let _guard = env_lock::lock_env([(
            "GOOSE_STATE_MACHINE",
            Some(if state_machine { "1" } else { "0" }),
        )]);
        let (agent, api, parent, directory) = agent_with_dummy_api().await?;
        let manager = &agent.config.session_manager;
        let lease = manager
            .claim_prompt_attempt(&uuid::Uuid::new_v4().to_string(), &"a".repeat(64), &parent)
            .await?
            .unwrap();
        let child = manager
            .create_session(
                directory.path().to_path_buf(),
                "specialist".into(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await?;
        agent
            .update_provider(
                agent.provider().await?,
                ModelConfig::new(goose_providers::openai::OPEN_AI_DEFAULT_MODEL)
                    .with_canonical_limits("openai"),
                &child.id,
            )
            .await?;
        let admission = manager
            .capture_task_admission(&parent, &child.id, "specialist")
            .await?;
        let mut data = child.extension_data;
        let tool = "cortex_document__delegate_editor_task";
        data.set_extension_state(
            "summon",
            "task_admission_v1",
            serde_json::to_value(admission)?,
        );
        data.set_extension_state(
            "summon",
            "v1",
            serde_json::json!({"artifact_key":"document:notes", "artifact_result_tools":[tool]}),
        );
        manager
            .update(&child.id)
            .parent_session_id(Some(parent.clone()))
            .extension_data(data)
            .apply()
            .await?;
        manager
            .add_message(&child.id, &Message::user().with_text("Original assignment"))
            .await?;
        manager
            .add_message(
                &child.id,
                &Message::assistant().with_tool_request(
                    "editor",
                    Ok(
                        rmcp::model::CallToolRequestParams::new(tool).with_arguments(
                            serde_json::json!({"idempotency_key":"notes"})
                                .as_object()
                                .unwrap()
                                .clone(),
                        ),
                    ),
                ),
            )
            .await?;
        manager
            .add_message(
                &child.id,
                &Message::user().with_tool_response(
                    "editor",
                    Ok(rmcp::model::CallToolResult::structured(
                        serde_json::json!({"interrupted":true,"job_id":"job-1"}),
                    )),
                ),
            )
            .await?;
        api.on("Try to finish").reply("Premature final report");
        api.on("Editor finished")
            .reply("Reviewed durable editor result");
        let reply = async {
            let stream = agent
                .reply(
                    Message::user().with_text("Try to finish"),
                    SessionConfig {
                        id: child.id.clone(),
                        schedule_id: None,
                        max_turns: Some(5),
                        retry_config: None,
                    },
                    state_machine,
                    Some(CancellationToken::new()),
                )
                .await?;
            stream_messages(stream).await
        };
        let deliver = async {
            tokio::time::timeout(Duration::from_secs(5), async {
                while api.call_count() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await?;
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(
                api.call_count(),
                1,
                "waiting must not make more model calls"
            );
            assert!(manager
                .terminal_report_for_child(&parent, &child.id)
                .await?
                .is_none());
            let response = manager.queue_task_notice(&goose_sdk_types::custom_requests::TaskNoticeRequest {
                attempt_key:lease.key, task_id:Some(child.id.clone()), artifact_keys:vec![], text:"Editor finished".into(), wake:true, refresh_tools:false, dedupe_key:Some("editor:job-1".into()),
                editor_result:Some(serde_json::json!({"idempotency_key":"notes", "receipt":{"job_id":"job-1","document_id":"notes","document_revision":1,"status":"completed"}})), channel_wait:None,
            }).await?;
            assert_eq!(
                response.status,
                goose_sdk_types::custom_requests::TaskNoticeStatus::Queued
            );
            anyhow::Ok(())
        };
        let (messages, delivered) = tokio::time::timeout(Duration::from_secs(10), async {
            tokio::join!(reply, deliver)
        })
        .await?;
        delivered?;
        let messages = messages?;
        assert!(
            messages
                .iter()
                .map(Message::as_concat_text)
                .collect::<String>()
                .contains("Reviewed durable editor result"),
            "loop={state_machine}, calls={}, messages={messages:?}",
            api.call_count()
        );
        assert_eq!(api.call_count(), 2);
        assert!(manager
            .pending_session_messages(&child.id)
            .await?
            .is_empty());
    }
    Ok(())
}
