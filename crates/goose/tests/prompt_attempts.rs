use goose::acp::custom_requests::PromptAttemptState;
use goose::session::SessionManager;
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn simultaneous_claims_execute_once_and_retain_the_first_binding() {
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let key = Uuid::new_v4().to_string();
    let digest = "a".repeat(64);
    let (first, second) = tokio::join!(
        manager.claim_prompt_attempt(&key, &digest, "session-a"),
        manager.claim_prompt_attempt(&key, &digest, "session-b"),
    );
    let first = first.unwrap();
    let second = second.unwrap();
    assert_ne!(first.is_some(), second.is_some());
    let lease = first.or(second).unwrap();
    let live = manager.prompt_attempt_status(&key).await.unwrap();
    assert_eq!(live.state, Some(PromptAttemptState::Running));
    assert!(!live.stopped);
    assert_eq!(live.run_id.as_deref(), Some(lease.run_id.as_str()));
    let result = serde_json::json!({"stopReason": "end_turn"});
    manager
        .finish_prompt_attempt(&lease, PromptAttemptState::Completed, Some(result.clone()))
        .await
        .unwrap();
    // A failed transport notification must not replace durable completion.
    manager
        .finish_prompt_attempt(&lease, PromptAttemptState::Interrupted, None)
        .await
        .unwrap();
    assert!(!manager.prompt_attempt_status(&key).await.unwrap().stopped);
    drop(lease);
    let terminal = manager.prompt_attempt_status(&key).await.unwrap();
    assert!(terminal.stopped);
    assert_eq!(terminal.result, Some(result));
    assert_eq!(terminal.state, Some(PromptAttemptState::Completed));
    assert!(manager
        .claim_prompt_attempt(&key, &digest, "session-c")
        .await
        .unwrap()
        .is_none());
    assert!(manager
        .claim_prompt_attempt(&key, &"b".repeat(64), "session-d")
        .await
        .is_err());
}

#[tokio::test]
async fn cancellation_before_start_is_a_permanent_tombstone() {
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let key = Uuid::new_v4().to_string();
    let cancelled = manager.cancel_prompt_attempt(&key).await.unwrap();
    assert_eq!(cancelled.state, Some(PromptAttemptState::Cancelled));
    assert!(cancelled.stopped);
    assert!(manager
        .claim_prompt_attempt(&key, &"a".repeat(64), "late-session")
        .await
        .unwrap()
        .is_none());
    // Tombstones are independent of both process memory and session retention.
    drop(manager);
    let reopened = SessionManager::new(directory.path().to_path_buf());
    assert_eq!(
        reopened.prompt_attempt_status(&key).await.unwrap().state,
        Some(PromptAttemptState::Cancelled)
    );
    assert!(reopened
        .claim_prompt_attempt(&key, &"a".repeat(64), "new-session")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn cancellation_during_admission_is_not_reported_as_stopped() {
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let key = Uuid::new_v4().to_string();
    let lease = manager
        .claim_prompt_attempt(&key, &"a".repeat(64), "session")
        .await
        .unwrap()
        .unwrap();
    // No in-memory active-run registration exists yet.
    let cancelled = manager.cancel_prompt_attempt(&key).await.unwrap();
    assert_eq!(cancelled.state, Some(PromptAttemptState::CancelRequested));
    assert!(!cancelled.stopped);
    assert!(manager.prompt_attempt_cancel_requested(&key).await.unwrap());
    manager
        .finish_prompt_attempt(
            &lease,
            PromptAttemptState::Completed,
            Some(serde_json::json!({"late": true})),
        )
        .await
        .unwrap();
    drop(lease);
    let terminal = manager.prompt_attempt_status(&key).await.unwrap();
    assert!(terminal.stopped);
    assert_eq!(terminal.state, Some(PromptAttemptState::Cancelled));
    assert!(terminal.result.is_none());
}

#[tokio::test]
async fn abandoned_attempt_is_interrupted_and_never_reexecuted() {
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let other_manager = SessionManager::new(directory.path().to_path_buf());
    let key = Uuid::new_v4().to_string();
    let lease = manager
        .claim_prompt_attempt(&key, &"a".repeat(64), "session")
        .await
        .unwrap()
        .unwrap();
    assert!(
        !other_manager
            .prompt_attempt_status(&key)
            .await
            .unwrap()
            .stopped
    );
    drop(lease);
    let interrupted = other_manager.prompt_attempt_status(&key).await.unwrap();
    assert_eq!(interrupted.state, Some(PromptAttemptState::Interrupted));
    assert!(interrupted.stopped);
    assert!(other_manager
        .claim_prompt_attempt(&key, &"a".repeat(64), "retry")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn attempt_keys_cannot_choose_filesystem_paths() {
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    assert!(manager.cancel_prompt_attempt("../outside").await.is_err());
    assert!(manager.prompt_attempt_status("../outside").await.is_err());
    assert!(manager
        .claim_prompt_attempt("../outside", &"a".repeat(64), "session")
        .await
        .is_err());
}

#[tokio::test]
async fn stopped_transcripts_are_bounded_and_reassemble_unicode() {
    use goose::acp::custom_requests::PromptAttemptTranscriptRequest;
    use goose::config::GooseMode;
    use goose::conversation::message::Message;
    use goose::session::SessionType;
    let directory = TempDir::new().unwrap();
    let manager = SessionManager::new(directory.path().to_path_buf());
    let session = manager
        .create_session(
            directory.path().to_path_buf(),
            "attempt".into(),
            SessionType::Acp,
            GooseMode::Auto,
        )
        .await
        .unwrap();
    let key = Uuid::new_v4().to_string();
    let lease = manager
        .claim_prompt_attempt(&key, &"a".repeat(64), &session.id)
        .await
        .unwrap()
        .unwrap();
    let mut request = PromptAttemptTranscriptRequest {
        attempt_key: key,
        ..Default::default()
    };
    assert!(manager.prompt_attempt_transcript(&request).await.is_err());
    let text = "é🧪".repeat(20000);
    manager
        .add_message(&session.id, &Message::assistant().with_text(&text))
        .await
        .unwrap();
    manager
        .finish_prompt_attempt(&lease, PromptAttemptState::Completed, None)
        .await
        .unwrap();
    drop(lease);
    let mut document = String::new();
    loop {
        let fragment = manager.prompt_attempt_transcript(&request).await.unwrap();
        assert!(fragment.data.chars().count() <= 16384);
        if fragment.done {
            break;
        }
        document.push_str(&fragment.data);
        request.after = fragment.next_after;
        request.offset = fragment.next_offset;
    }
    let message: serde_json::Value = serde_json::from_str(&document).unwrap();
    assert_eq!(message["content"][0]["text"], text);
}
