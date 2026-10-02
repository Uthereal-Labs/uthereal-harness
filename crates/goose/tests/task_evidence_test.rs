use goose::config::GooseMode;
use goose::conversation::message::Message;
use goose::session::{PromptAttemptLease, SessionManager, SessionType};
use goose_sdk_types::custom_requests::{
    PromptAttemptState, TaskAdmission, TaskEvidenceRequest, TaskTerminalStatus,
    ToolReceiptTransportStatus,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};
use serde_json::{json, Value};
use tempfile::TempDir;

struct Fixture {
    directory: TempDir,
    manager: SessionManager,
    parent: String,
    lease: PromptAttemptLease,
}
impl Fixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let manager = SessionManager::new(directory.path().to_path_buf());
        let parent = manager
            .create_session(
                directory.path().to_path_buf(),
                "parent".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap()
            .id;
        let lease = manager
            .claim_prompt_attempt(&uuid::Uuid::new_v4().to_string(), &"a".repeat(64), &parent)
            .await
            .unwrap()
            .unwrap();
        Self {
            directory,
            manager,
            parent,
            lease,
        }
    }
    async fn child(&self) -> TaskAdmission {
        let session = self
            .manager
            .create_session(
                self.directory.path().to_path_buf(),
                "private child description".into(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let admission = self
            .manager
            .capture_task_admission(&self.parent, &session.id, "registered-specialist")
            .await
            .unwrap();
        let mut data = session.extension_data;
        data.set_extension_state(
            "summon",
            "task_admission_v1",
            serde_json::to_value(&admission).unwrap(),
        );
        self.manager
            .update(&session.id)
            .parent_session_id(Some(self.parent.clone()))
            .extension_data(data)
            .apply()
            .await
            .unwrap();
        admission
    }
    fn request(&self, tools: &[&str]) -> TaskEvidenceRequest {
        TaskEvidenceRequest {
            task_id: None,
            attempt_key: self.lease.key.clone(),
            tool_names: tools.iter().map(|name| name.to_string()).collect(),
            after_task_id: None,
        }
    }
    async fn call(&self, child: &str, id: &str, tool: &str, result: Value) {
        let request = Message::assistant().with_tool_request(
            id,
            Ok(CallToolRequestParams::new(tool.to_string()).with_arguments(
                json!({"private_argument":"do not return"})
                    .as_object()
                    .unwrap()
                    .clone(),
            )),
        );
        let mut response = CallToolResult::success(vec![ContentBlock::text("private tool output")]);
        response.structured_content = Some(result);
        response.is_error = Some(false);
        self.manager.add_message(child, &request).await.unwrap();
        self.manager
            .add_message(child, &Message::user().with_tool_response(id, Ok(response)))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn typed_outcome_first_wins_after_same_database_reconstruction_and_stop() {
    let f = Fixture::new().await;
    let admission = f.child().await;
    assert_eq!(
        admission.parent_run_id.as_deref(),
        Some(f.lease.run_id.as_str())
    );
    f.manager
        .enqueue_task_outcome(
            &admission.task_id,
            "private completed narrative",
            TaskTerminalStatus::Completed,
        )
        .await
        .unwrap();
    assert!(!f
        .manager
        .enqueue_task_outcome(
            &admission.task_id,
            "later cancellation",
            TaskTerminalStatus::Cancelled
        )
        .await
        .unwrap());
    f.manager
        .finish_prompt_attempt(&f.lease, PromptAttemptState::Cancelled, None)
        .await
        .unwrap();
    let reconstructed = SessionManager::new(f.directory.path().to_path_buf());
    let evidence = reconstructed.task_evidence(&f.request(&[])).await.unwrap();
    assert_eq!(
        evidence.tasks[0].outcome.as_ref().unwrap().status,
        TaskTerminalStatus::Completed
    );
    assert_eq!(evidence.tasks[0].admission, admission);
    assert!(evidence.evidence_complete);
    assert!(!serde_json::to_string(&evidence)
        .unwrap()
        .contains("private"));
    let report = reconstructed
        .terminal_report_for_child(&f.parent, &admission.task_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(report.body, "private completed narrative");
}

#[tokio::test]
async fn untyped_report_is_unknown_and_passive_read_does_not_interrupt_running_attempt() {
    let f = Fixture::new().await;
    let child = f.child().await;
    f.manager
        .enqueue_completion_to_parent(
            &child.task_id,
            "Task completed successfully. Model says every operation succeeded.",
        )
        .await
        .unwrap();
    let evidence = f.manager.task_evidence(&f.request(&[])).await.unwrap();
    assert!(evidence.tasks[0].outcome.is_none());
    assert!(!f
        .manager
        .prompt_attempt_cancel_requested(&f.lease.key)
        .await
        .unwrap());
}

#[tokio::test]
async fn scope_rejects_other_attempt_cursor_and_unbound_tasks() {
    let f = Fixture::new().await;
    let foreign_parent = f
        .manager
        .create_session(
            f.directory.path().to_path_buf(),
            "foreign".into(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await
        .unwrap()
        .id;
    let _foreign_lease = f
        .manager
        .claim_prompt_attempt(
            &uuid::Uuid::new_v4().to_string(),
            &"b".repeat(64),
            &foreign_parent,
        )
        .await
        .unwrap()
        .unwrap();
    let foreign = f
        .manager
        .create_session(
            f.directory.path().to_path_buf(),
            "foreign child".into(),
            SessionType::SubAgent,
            GooseMode::Auto,
        )
        .await
        .unwrap();
    let admission = f
        .manager
        .capture_task_admission(&foreign_parent, &foreign.id, "other")
        .await
        .unwrap();
    let mut data = foreign.extension_data;
    data.set_extension_state(
        "summon",
        "task_admission_v1",
        serde_json::to_value(admission).unwrap(),
    );
    f.manager
        .update(&foreign.id)
        .parent_session_id(Some(foreign_parent))
        .extension_data(data)
        .apply()
        .await
        .unwrap();
    let own = f.child().await;
    let mut request = f.request(&[]);
    request.after_task_id = Some(foreign.id);
    assert!(f.manager.task_evidence(&request).await.is_err());
    request.after_task_id = None;
    request.attempt_key = uuid::Uuid::new_v4().to_string();
    assert!(f.manager.task_evidence(&request).await.is_err());
    assert_eq!(
        f.manager
            .task_evidence(&f.request(&[]))
            .await
            .unwrap()
            .tasks[0]
            .admission,
        own
    );
}

#[tokio::test]
async fn exact_tool_receipts_return_only_canonical_structured_fields() {
    let f = Fixture::new().await;
    let child = f.child().await;
    f.call(
        &child.task_id,
        "selected",
        "exact__save",
        json!({"operation_id":"op1","revision":7}),
    )
    .await;
    f.call(
        &child.task_id,
        "unselected",
        "exact__save_extra",
        json!({"secret":"unselected"}),
    )
    .await;
    let evidence = f
        .manager
        .task_evidence(&f.request(&["exact__save"]))
        .await
        .unwrap();
    let task = &evidence.tasks[0];
    assert!(task.evidence_complete);
    assert_eq!(task.receipts.len(), 1);
    assert_eq!(task.receipts[0].call_id, "selected");
    assert_eq!(
        task.receipts[0].transport_status,
        ToolReceiptTransportStatus::Success
    );
    assert_eq!(task.receipts[0].is_error, Some(false));
    assert_eq!(
        task.receipts[0].structured_result,
        Some(json!({"operation_id":"op1","revision":7}))
    );
    let wire = serde_json::to_string(&evidence).unwrap();
    for private in ["private", "unselected", "arguments", "content"] {
        assert!(!wire.contains(private), "{private}");
    }
}

#[tokio::test]
async fn ambiguous_and_unmatched_selected_calls_are_incomplete() {
    let f = Fixture::new().await;
    let child = f.child().await;
    f.call(
        &child.task_id,
        "duplicate",
        "selected",
        json!({"revision":1}),
    )
    .await;
    f.call(
        &child.task_id,
        "duplicate",
        "selected",
        json!({"revision":2}),
    )
    .await;
    f.manager
        .add_message(
            &child.task_id,
            &Message::assistant()
                .with_tool_request("pending", Ok(CallToolRequestParams::new("selected"))),
        )
        .await
        .unwrap();
    let evidence = f
        .manager
        .task_evidence(&f.request(&["selected"]))
        .await
        .unwrap();
    assert!(!evidence.evidence_complete);
    assert!(!evidence.tasks[0].evidence_complete);
    assert!(evidence.tasks[0].receipts.is_empty());
}

#[tokio::test]
async fn structured_byte_and_page_caps_never_claim_complete_or_truncate_json() {
    let f = Fixture::new().await;
    let child = f.child().await;
    f.call(
        &child.task_id,
        "oversize",
        "selected",
        json!({"value":"é".repeat(17000)}),
    )
    .await;
    for n in 0..12 {
        f.call(
            &child.task_id,
            &format!("call-{n:02}"),
            "selected",
            json!({"value":"x".repeat(30000)}),
        )
        .await;
    }
    let evidence = f
        .manager
        .task_evidence(&f.request(&["selected"]))
        .await
        .unwrap();
    assert!(!evidence.tasks[0].evidence_complete);
    assert!(serde_json::to_vec(&evidence).unwrap().len() <= 262144);
    assert!(evidence.tasks[0]
        .receipts
        .iter()
        .find(|r| r.call_id == "oversize")
        .is_none_or(|r| r.structured_result.is_none()));
    assert!(evidence.tasks[0]
        .receipts
        .iter()
        .filter_map(|r| r.structured_result.as_ref())
        .all(|result| result["value"].as_str().unwrap().len() == 30000));
}

#[tokio::test]
async fn child_pagination_preserves_per_task_completeness_and_scope() {
    let f = Fixture::new().await;
    for _ in 0..34 {
        f.child().await;
    }
    let first = f.manager.task_evidence(&f.request(&[])).await.unwrap();
    assert_eq!(first.tasks.len(), 32);
    assert!(!first.evidence_complete);
    assert!(first.tasks.iter().all(|t| t.evidence_complete));
    let mut request = f.request(&[]);
    request.after_task_id = first.next_task_id;
    let last = f.manager.task_evidence(&request).await.unwrap();
    assert_eq!(last.tasks.len(), 2);
    assert!(last.evidence_complete);
    assert!(last.next_task_id.is_none());
}

#[tokio::test]
async fn receipt_and_transcript_block_caps_mark_only_affected_task_incomplete() {
    let f = Fixture::new().await;
    let affected = f.child().await;
    let unaffected = f.child().await;
    for n in 0..129 {
        f.call(
            &affected.task_id,
            &format!("call-{n}"),
            "selected",
            json!({"n":n}),
        )
        .await;
    }
    let receipt_capped = f
        .manager
        .task_evidence(&f.request(&["selected"]))
        .await
        .unwrap();
    assert!(
        !receipt_capped
            .tasks
            .iter()
            .find(|t| t.admission == affected)
            .unwrap()
            .evidence_complete
    );
    assert!(
        receipt_capped
            .tasks
            .iter()
            .find(|t| t.admission == unaffected)
            .unwrap()
            .evidence_complete
    );
    assert!(receipt_capped.tasks.iter().all(|t| t.receipts.len() <= 128));
    let mut blocks = Message::assistant();
    for n in 0..1025 {
        blocks = blocks.with_tool_request(
            format!("unselected-{n}"),
            Ok(CallToolRequestParams::new("other")),
        );
    }
    f.manager
        .add_message(&affected.task_id, &blocks)
        .await
        .unwrap();
    assert!(
        !f.manager
            .task_evidence(&f.request(&["none"]))
            .await
            .unwrap()
            .tasks
            .iter()
            .find(|t| t.admission == affected)
            .unwrap()
            .evidence_complete
    );
}

#[tokio::test]
async fn receipts_preserve_execution_order_and_reject_response_before_request() {
    let f = Fixture::new().await;
    let child = f.child().await;
    f.call(
        &child.task_id,
        "z-failed",
        "selected",
        json!({"status":"failed"}),
    )
    .await;
    f.call(
        &child.task_id,
        "a-recovered",
        "selected",
        json!({"status":"completed"}),
    )
    .await;
    let evidence = f
        .manager
        .task_evidence(&f.request(&["selected"]))
        .await
        .unwrap();
    assert!(evidence.tasks[0].evidence_complete);
    assert_eq!(
        evidence.tasks[0]
            .receipts
            .iter()
            .map(|r| r.call_id.as_str())
            .collect::<Vec<_>>(),
        vec!["z-failed", "a-recovered"]
    );
    f.manager
        .add_message(
            &child.task_id,
            &Message::user().with_tool_response("early", Ok(CallToolResult::success(vec![]))),
        )
        .await
        .unwrap();
    f.manager
        .add_message(
            &child.task_id,
            &Message::assistant()
                .with_tool_request("early", Ok(CallToolRequestParams::new("selected"))),
        )
        .await
        .unwrap();
    let incomplete = f
        .manager
        .task_evidence(&f.request(&["selected"]))
        .await
        .unwrap();
    assert!(!incomplete.tasks[0].evidence_complete);
    assert!(!incomplete.tasks[0]
        .receipts
        .iter()
        .any(|r| r.call_id == "early"));
}

#[tokio::test]
async fn ordinary_unbound_async_task_has_typed_outcome_but_no_attempt_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let parent = manager
        .create_session(
            directory.path().to_path_buf(),
            "CLI".into(),
            SessionType::User,
            GooseMode::Auto,
        )
        .await
        .unwrap();
    let child = manager
        .create_session(
            directory.path().to_path_buf(),
            "inline task".into(),
            SessionType::SubAgent,
            GooseMode::Auto,
        )
        .await
        .unwrap();
    let admission = manager
        .capture_task_admission(&parent.id, &child.id, "inline")
        .await
        .unwrap();
    assert!(admission.attempt_key.is_none() && admission.parent_run_id.is_none());
    let mut data = child.extension_data;
    data.set_extension_state(
        "summon",
        "task_admission_v1",
        serde_json::to_value(&admission).unwrap(),
    );
    manager
        .update(&child.id)
        .parent_session_id(Some(parent.id.clone()))
        .extension_data(data)
        .apply()
        .await
        .unwrap();
    manager
        .enqueue_task_outcome(
            &child.id,
            "ordinary CLI terminal output",
            TaskTerminalStatus::Completed,
        )
        .await
        .unwrap();
    assert_eq!(
        manager
            .terminal_report_for_child(&parent.id, &child.id)
            .await
            .unwrap()
            .unwrap()
            .task_outcome()
            .unwrap()
            .unwrap()
            .admission,
        admission
    );
    assert!(manager
        .task_evidence(&TaskEvidenceRequest {
            task_id: None,
            attempt_key: uuid::Uuid::new_v4().to_string(),
            tool_names: vec![],
            after_task_id: None
        })
        .await
        .is_err());
}

#[tokio::test]
async fn exact_task_filter_exports_only_persisted_artifact_policy() {
    let fixture = Fixture::new().await;
    let selected = fixture.child().await;
    let _other = fixture.child().await;
    let session = fixture
        .manager
        .get_session(&selected.task_id, false)
        .await
        .unwrap();
    let mut data = session.extension_data;
    data.set_extension_state("summon", "v1", json!({"artifact_key": "new:document:numina", "previous_task_id": null, "private_instructions": "must not escape"}));
    fixture
        .manager
        .update(&selected.task_id)
        .extension_data(data)
        .apply()
        .await
        .unwrap();
    let mut request = fixture.request(&[]);
    request.task_id = Some(selected.task_id.clone());
    let response = fixture.manager.task_evidence(&request).await.unwrap();
    assert_eq!(response.tasks.len(), 1);
    assert_eq!(response.tasks[0].admission.task_id, selected.task_id);
    assert_eq!(
        response.tasks[0].artifact_key.as_deref(),
        Some("new:document:numina")
    );
    assert!(!serde_json::to_string(&response)
        .unwrap()
        .contains("must not escape"));
    request.task_id = Some("unrelated".into());
    assert!(fixture
        .manager
        .task_evidence(&request)
        .await
        .unwrap()
        .tasks
        .is_empty());
}

#[tokio::test]
async fn large_native_review_cannot_crowd_out_compact_terminal_proof() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/large-presentation-terminal.json")).unwrap();
    let review = serde_json::to_string(&fixture["review_response"]).unwrap();
    assert_eq!(review.len(), 55_832);
    let proof = fixture["structuredContent"].clone();
    assert!(serde_json::to_vec(&proof).unwrap().len() <= 8192);
    let f = Fixture::new().await;
    let child = f.child().await;
    let tool = "cortex_presentation__delegate_editor_task";
    let request = Message::assistant().with_tool_request(
        "native-call",
        Ok(CallToolRequestParams::new(tool.to_string())),
    );
    let mut response = CallToolResult::success(vec![ContentBlock::text(review.clone())]);
    response.structured_content = Some(proof.clone());
    response.is_error = Some(false);
    f.manager
        .add_message(&child.task_id, &request)
        .await
        .unwrap();
    f.manager
        .add_message(
            &child.task_id,
            &Message::user().with_tool_response("native-call", Ok(response)),
        )
        .await
        .unwrap();
    f.manager
        .enqueue_task_outcome(&child.task_id, "completed", TaskTerminalStatus::Completed)
        .await
        .unwrap();
    let evidence = f.manager.task_evidence(&f.request(&[tool])).await.unwrap();
    assert!(evidence.evidence_complete);
    assert!(evidence.tasks[0].evidence_complete);
    assert_eq!(evidence.tasks[0].receipts[0].structured_result, Some(proof));
    assert_eq!(
        evidence.tasks[0].outcome.as_ref().unwrap().status,
        TaskTerminalStatus::Completed
    );
    assert!(!serde_json::to_string(&evidence)
        .unwrap()
        .contains("saved_document"));

    // The unchanged generic 32 KiB boundary still rejects the old expanded envelope.
    let legacy = f.child().await;
    f.call(
        &legacy.task_id,
        "old-call",
        tool,
        fixture["review_response"].clone(),
    )
    .await;
    let rejected = f.manager.task_evidence(&f.request(&[tool])).await.unwrap();
    let legacy_task = rejected
        .tasks
        .iter()
        .find(|task| task.admission.task_id == legacy.task_id)
        .unwrap();
    assert!(!legacy_task.evidence_complete);
    assert!(legacy_task.receipts[0].structured_result.is_none());
}
