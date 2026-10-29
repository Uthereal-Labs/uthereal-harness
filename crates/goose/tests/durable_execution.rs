use std::sync::Arc;

use anyhow::Result;
use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::session::durable::{self, AcceptedReply, DurableAdmission, DurableSnapshot};
use goose::session::{SessionManager, SessionType};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, MetaObject};
use serde_json::{json, Value};
use wiremock::{matchers::method, Mock, MockServer, Request, ResponseTemplate};

async fn callback() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(200)
                .set_body_json(json!({"accepted":true,"generation":body["generation"]}))
        })
        .mount(&server)
        .await;
    server
}

fn admission(server: &MockServer, restore: Option<DurableSnapshot>) -> DurableAdmission {
    DurableAdmission {
        logical_run_id: "logical-test".into(),
        attempt_id: uuid::Uuid::new_v4().to_string(),
        epoch: 1 + u64::from(restore.is_some()),
        request_digest: "a".repeat(64),
        expected_generation: restore.as_ref().map_or(0, |snapshot| snapshot.generation),
        callback_url: server.uri(),
        callback_token: "fresh-callback-secret".into(),
        restore,
        accepted_reply: None,
        continuation_context: None,
    }
}

async fn last_snapshot(server: &MockServer) -> Result<DurableSnapshot> {
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests.last().unwrap().body)?;
    Ok(serde_json::from_value(body["snapshot"].clone())?)
}

async fn manager() -> Result<(Arc<SessionManager>, String, tempfile::TempDir)> {
    let temp = tempfile::tempdir()?;
    let manager = Arc::new(SessionManager::new(temp.path().to_path_buf()));
    let session = manager
        .prepare_durable_session(
            &format!("durable_{}", uuid::Uuid::new_v4()),
            temp.path(),
            "durable",
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    Ok((manager, session.id, temp))
}

fn waiting() -> CallToolResult {
    let mut result = CallToolResult::success(vec![]);
    result.meta = Some(MetaObject(
        json!({durable::WAIT_META:{"version":1,"waitId":"question-1","dependentTaskIds":[]}})
            .as_object()
            .unwrap()
            .clone(),
    ));
    result
}

#[tokio::test]
async fn durable_wait_and_accepted_reply_survive_loss_of_native_database() -> Result<()> {
    let server = callback().await;
    let (first, root, first_dir) = manager().await?;
    durable::admit(first.clone(), &root, admission(&server, None)).await?;
    first
        .add_message(
            &root,
            &Message::user().with_text("Full manuscript, not a summary"),
        )
        .await?;
    let mut request = CallToolRequestParams::new("ask_user");
    durable::before_tool(&root, "call-1", &mut request).await?;
    durable::after_tool(&root, "call-1", &Ok(waiting())).await?;
    let snapshot = last_snapshot(&server).await?;
    assert_eq!(snapshot.consumed_input_revision, 0);
    assert!(!serde_json::to_string(&snapshot)?.contains("fresh-callback-secret"));
    durable::release(&root);
    drop(first);
    drop(first_dir);

    let new_dir = tempfile::tempdir()?;
    let second = Arc::new(SessionManager::new(new_dir.path().to_path_buf()));
    second
        .prepare_durable_session(
            &root,
            new_dir.path(),
            "restored",
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    durable::admit(
        second.clone(),
        &root,
        admission(&server, Some(snapshot.clone())),
    )
    .await?;
    assert!(durable::suspended(&root).await);
    assert!(durable::parked_receipt(&root).await?.is_some());
    durable::release(&root);

    let mut next = admission(&server, Some(snapshot));
    next.accepted_reply = Some(AcceptedReply {
        wait_id: "question-1".into(),
        tool_call_id: "call-1".into(),
        input_revision: 1,
        result: CallToolResult::success(vec![ContentBlock::text("the accepted answer")]),
    });
    let message = durable::admit(second.clone(), &root, next).await?.unwrap();
    second.add_message(&root, &message).await?;
    let restored = second.get_session(&root, true).await?.conversation.unwrap();
    assert_eq!(
        restored
            .messages()
            .iter()
            .filter(|message| message.as_concat_text().contains("Full manuscript"))
            .count(),
        1
    );
    assert_eq!(
        restored
            .messages()
            .iter()
            .flat_map(|message| message.get_tool_request_ids())
            .filter(|id| *id == "call-1")
            .count(),
        1
    );
    assert_eq!(
        restored
            .messages()
            .iter()
            .flat_map(|message| message.get_tool_response_ids())
            .filter(|id| *id == "call-1")
            .count(),
        1
    );
    assert_eq!(last_snapshot(&server).await?.consumed_input_revision, 1);
    assert!(!durable::suspended(&root).await);
    let replay = durable::before_tool(&root, "call-1", &mut request)
        .await?
        .unwrap()
        .unwrap();
    assert_eq!(
        replay.content,
        message
            .content
            .iter()
            .find_map(|content| {
                if let goose::conversation::message::MessageContent::ToolResponse(response) =
                    content
                {
                    response
                        .tool_result
                        .as_ref()
                        .ok()
                        .map(|result| result.content.clone())
                } else {
                    None
                }
            })
            .unwrap()
    );
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_uncertain_effect_never_redispatches_after_storage_loss() -> Result<()> {
    let server = callback().await;
    let (first, root, _dir) = manager().await?;
    durable::admit(first.clone(), &root, admission(&server, None)).await?;
    let mut request = CallToolRequestParams::new("write_document");
    durable::before_tool(&root, "mutation-1", &mut request).await?;
    let snapshot = last_snapshot(&server).await?;
    durable::release(&root);
    let error = durable::admit(first, &root, admission(&server, Some(snapshot)))
        .await
        .err()
        .unwrap();
    assert!(matches!(
        error.downcast_ref::<durable::DurableError>(),
        Some(durable::DurableError::EffectUncertain(_))
    ));
    assert!(!durable::enabled(&root));
    Ok(())
}

#[tokio::test]
async fn durable_callback_failure_fences_following_dispatch() -> Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(409))
        .mount(&server)
        .await;
    let (manager, root, _dir) = manager().await?;
    durable::admit(manager, &root, admission(&server, None)).await?;
    let mut request = CallToolRequestParams::new("write_document");
    assert!(durable::before_tool(&root, "mutation-1", &mut request)
        .await
        .is_err());
    assert!(durable::before_tool(&root, "mutation-2", &mut request)
        .await
        .is_err());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_snapshot_rejects_disconnected_task_before_mutating_history() -> Result<()> {
    let server = callback().await;
    let (manager, root, _dir) = manager().await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    manager
        .add_message(&root, &Message::user().with_text("keep me"))
        .await?;
    let mut snapshot = last_snapshot(&server).await?;
    let mut outside = snapshot.tasks[0].clone();
    outside.session_id = "unrelated-session".into();
    snapshot.tasks.push(outside);
    durable::release(&root);
    assert!(
        durable::admit(manager.clone(), &root, admission(&server, Some(snapshot)))
            .await
            .is_err()
    );
    assert_eq!(
        manager
            .get_session(&root, true)
            .await?
            .conversation
            .unwrap()
            .messages()[0]
            .as_concat_text(),
        "keep me"
    );
    Ok(())
}

#[tokio::test]
async fn durable_restore_preserves_specialist_mailbox_and_independent_work() -> Result<()> {
    let server = callback().await;
    let (manager, root, _dir) = manager().await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    let mut children = Vec::new();
    for name in ["dependent", "independent"] {
        let child = manager
            .prepare_durable_session(
                &format!("durable_{}", uuid::Uuid::new_v4()),
                _dir.path(),
                name,
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await?;
        manager
            .update(&child.id)
            .parent_session_id(Some(root.clone()))
            .apply()
            .await?;
        durable::inherit(&root, &child.id);
        manager
            .add_message(
                &child.id,
                &Message::user().with_text(format!("original {name} manuscript")),
            )
            .await?;
        children.push(child.id);
    }
    manager
        .send_to_child(&root, &children[0], "already acknowledged")
        .await?;
    let acknowledged = manager.pending_session_messages(&children[0]).await?[0].id;
    manager
        .acknowledge_session_messages(&children[0], acknowledged)
        .await?;
    manager
        .send_to_child(&root, &children[0], "keep this pending channel message")
        .await?;
    manager
        .send_to_child(&root, &children[0], "keep this later channel message")
        .await?;
    let mut request = CallToolRequestParams::new("ask_user");
    durable::before_tool(&root, "call-1", &mut request).await?;
    let mut result = waiting();
    result.meta.as_mut().unwrap().0[durable::WAIT_META]["dependentTaskIds"] = json!([children[0]]);
    durable::after_tool(&root, "call-1", &Ok(result)).await?;
    let snapshot = last_snapshot(&server).await?;
    durable::release(&root);

    let fresh = tempfile::tempdir()?;
    let restored = Arc::new(SessionManager::new(fresh.path().to_path_buf()));
    let unrelated_parent = restored
        .prepare_durable_session(
            &format!("durable_{}", uuid::Uuid::new_v4()),
            fresh.path(),
            "unrelated parent",
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    let unrelated_child = restored
        .prepare_durable_session(
            &format!("durable_{}", uuid::Uuid::new_v4()),
            fresh.path(),
            "unrelated child",
            SessionType::SubAgent,
            GooseMode::Auto,
        )
        .await?;
    restored
        .update(&unrelated_child.id)
        .parent_session_id(Some(unrelated_parent.id.clone()))
        .apply()
        .await?;
    let unrelated_message = restored
        .send_to_child(&unrelated_parent.id, &unrelated_child.id, "unrelated work")
        .await?;
    assert!(snapshot
        .mailbox
        .iter()
        .any(|message| message.id == unrelated_message));
    restored
        .prepare_durable_session(
            &root,
            fresh.path(),
            "restored",
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    durable::admit(restored.clone(), &root, admission(&server, Some(snapshot))).await?;
    assert!(durable::suspended(&children[0]).await);
    assert!(!durable::suspended(&children[1]).await);
    let runnable = durable::recovered_tasks(&root).await;
    assert_eq!(runnable.len(), 1);
    assert_eq!(runnable[0].session_id, children[1]);
    let pending = restored.pending_session_messages(&children[0]).await?;
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].body, "keep this pending channel message");
    assert_eq!(pending[1].body, "keep this later channel message");
    assert!(pending[0].id > unrelated_message && pending[1].id > pending[0].id);
    restored
        .acknowledge_session_messages(&children[0], pending[0].id)
        .await?;
    let remaining = restored.pending_session_messages(&children[0]).await?;
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].body, "keep this later channel message");
    let unrelated = restored
        .pending_session_messages(&unrelated_child.id)
        .await?;
    assert_eq!(unrelated.len(), 1);
    assert_eq!(unrelated[0].id, unrelated_message);
    assert_eq!(unrelated[0].body, "unrelated work");
    assert_eq!(
        restored
            .get_session(&children[0], true)
            .await?
            .conversation
            .unwrap()
            .messages()[0]
            .as_concat_text(),
        "original dependent manuscript"
    );
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_continuation_context_is_checkpointed_once_across_recovery() -> Result<()> {
    let server = callback().await;
    let (manager, root, _dir) = manager().await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    manager
        .add_message(&root, &Message::user().with_text("original request"))
        .await?;
    let snapshot = last_snapshot(&server).await?;
    durable::release(&root);
    let context = durable::ContinuationContext {
        id: "events-through-7".into(),
        text: "authenticated intervening conversation".into(),
    };
    let mut next = admission(&server, Some(snapshot));
    next.continuation_context = Some(context.clone());
    durable::admit(manager.clone(), &root, next).await?;
    assert!(durable::apply_continuation_context(&root).await?.is_some());
    assert!(durable::apply_continuation_context(&root).await?.is_none());
    let snapshot = last_snapshot(&server).await?;
    assert_eq!(snapshot.applied_context_ids, ["events-through-7"]);
    durable::release(&root);
    let mut next = admission(&server, Some(snapshot));
    next.continuation_context = Some(context);
    durable::admit(manager.clone(), &root, next).await?;
    assert!(durable::apply_continuation_context(&root).await?.is_none());
    assert_eq!(
        manager
            .get_session(&root, true)
            .await?
            .conversation
            .unwrap()
            .messages()
            .iter()
            .filter(|message| message.as_concat_text() == "authenticated intervening conversation")
            .count(),
        1
    );
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_completion_is_restored_for_publication_without_execution() -> Result<()> {
    let server = callback().await;
    let (manager, root, _dir) = manager().await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    manager
        .add_message(
            &root,
            &Message::assistant().with_text("immutable completed answer"),
        )
        .await?;
    durable::checkpoint(&root, "completed").await?;
    let snapshot = last_snapshot(&server).await?;
    assert!(
        snapshot
            .tasks
            .iter()
            .find(|task| task.session_id == root)
            .unwrap()
            .completed
    );
    durable::release(&root);
    let fresh = tempfile::tempdir()?;
    let restored = Arc::new(SessionManager::new(fresh.path().to_path_buf()));
    restored
        .prepare_durable_session(
            &root,
            fresh.path(),
            "restored",
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;
    durable::admit(restored.clone(), &root, admission(&server, Some(snapshot))).await?;
    assert!(durable::completed(&root).await);
    assert_eq!(
        restored
            .get_session(&root, true)
            .await?
            .conversation
            .unwrap()
            .messages()[0]
            .as_concat_text(),
        "immutable completed answer"
    );
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_parked_task_notice_cannot_acknowledge_local_only_delivery() -> Result<()> {
    use goose_sdk_types::custom_requests::{TaskNoticeRequest, TaskNoticeStatus};
    let server = callback().await;
    let (manager, root, directory) = manager().await?;
    let admission = admission(&server, None);
    let lease = manager
        .claim_prompt_attempt(&admission.attempt_id, &admission.request_digest, &root)
        .await?
        .unwrap();
    durable::admit(manager.clone(), &root, admission).await?;
    let child = manager
        .prepare_durable_session(
            &format!("durable_{}", uuid::Uuid::new_v4()),
            directory.path(),
            "specialist",
            SessionType::SubAgent,
            GooseMode::Auto,
        )
        .await?;
    let task = manager
        .capture_task_admission(&root, &child.id, "inline")
        .await?;
    let mut data = child.extension_data;
    data.set_extension_state("summon", "task_admission_v1", serde_json::to_value(task)?);
    manager
        .update(&child.id)
        .parent_session_id(Some(root.clone()))
        .extension_data(data)
        .apply()
        .await?;
    durable::inherit(&root, &child.id);
    let mut notice = TaskNoticeRequest {
        attempt_key: lease.key.clone(),
        task_id: Some(child.id),
        text: "durable channel notice".into(),
        dedupe_key: Some("notice-1".into()),
        ..Default::default()
    };
    assert_eq!(
        manager.queue_task_notice(&notice).await?.status,
        TaskNoticeStatus::Queued
    );
    let mut request = CallToolRequestParams::new("ask_user");
    durable::before_tool(&root, "wait", &mut request).await?;
    durable::after_tool(&root, "wait", &Ok(waiting())).await?;
    durable::parked_receipt(&root).await?;
    notice.dedupe_key = Some("late-notice".into());
    assert!(manager.queue_task_notice(&notice).await.is_err());
    durable::release(&root);
    assert!(manager.queue_task_notice(&notice).await.is_err());
    Ok(())
}

#[tokio::test]
async fn durable_snapshot_excludes_extension_authority_and_recipe_parameter_secrets() -> Result<()>
{
    use goose::session::{EnabledExtensionsState, ExtensionState};
    let server = callback().await;
    let (manager, root, _dir) = manager().await?;
    let extension = json!({"type":"streamable_http","name":"scoped-tool","uri":"http://tools.internal/mcp","headers":{"Authorization":"Bearer old-tool-token"}});
    let mut data = manager.get_session(&root, false).await?.extension_data;
    EnabledExtensionsState::new(vec![serde_json::from_value(extension.clone())?])
        .to_extension_data(&mut data)?;
    let recipe = serde_json::from_value(
        json!({"title":"specialist","description":"full instructions","instructions":"preserve exact manuscript","extensions":[extension],"parameters":[{"key":"token","input_type":"string","requirement":"optional","default":"recipe-parameter-secret","description":"authority"}]}),
    )?;
    manager
        .update(&root)
        .extension_data(data)
        .recipe(Some(recipe))
        .apply()
        .await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    manager
        .add_message(
            &root,
            &Message::user().with_text("full resource-backed manuscript"),
        )
        .await?;
    let snapshot = last_snapshot(&server).await?;
    let serialized = serde_json::to_string(&snapshot)?;
    assert!(!serialized.contains("old-tool-token"));
    assert!(!serialized.contains("recipe-parameter-secret"));
    assert!(!serialized.contains("fresh-callback-secret"));
    assert!(serialized.contains("full resource-backed manuscript"));
    assert_eq!(snapshot.tasks[0].extension_names, ["scoped-tool"]);
    durable::release(&root);
    Ok(())
}

#[tokio::test]
async fn durable_epochs_reuse_session_while_legacy_claims_stay_single_admission() -> Result<()> {
    let server = callback().await;
    let (manager, root, directory) = manager().await?;
    durable::admit(manager.clone(), &root, admission(&server, None)).await?;
    for _ in 0..2 {
        assert!(manager
            .claim_prompt_attempt(&uuid::Uuid::new_v4().to_string(), &"a".repeat(64), &root)
            .await?
            .is_some());
    }
    let legacy = manager
        .create_session(
            directory.path().to_path_buf(),
            "legacy".into(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await?;
    assert!(manager
        .claim_prompt_attempt(
            &uuid::Uuid::new_v4().to_string(),
            &"a".repeat(64),
            &legacy.id
        )
        .await?
        .is_some());
    assert!(manager
        .claim_prompt_attempt(
            &uuid::Uuid::new_v4().to_string(),
            &"a".repeat(64),
            &legacy.id
        )
        .await?
        .is_none());
    durable::release(&root);
    Ok(())
}
