use super::SessionManager;
use crate::acp::custom_requests::{
    PromptAttemptResponse, PromptAttemptState, PromptAttemptTranscriptRequest,
    PromptAttemptTranscriptResponse,
};
use anyhow::{bail, Result};
use fs2::FileExt;
use sqlx::Row;
use std::fs::{File, OpenOptions};
use uuid::Uuid;

// No session foreign key: deleting a session must never permit replay.
pub(crate) const CREATE_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS prompt_attempts (
    attempt_key TEXT PRIMARY KEY,
    request_digest TEXT,
    session_id TEXT,
    run_id TEXT,
    state TEXT NOT NULL CHECK (state IN
        ('running', 'cancel_requested', 'completed', 'failed', 'cancelled', 'interrupted')),
    result_json TEXT,
    steering_sealed INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"#;

/// Held through agent shutdown. Lock files are retained to avoid inode replacement races.
pub struct PromptAttemptLease {
    pub key: String,
    pub run_id: String,
    _lock: File,
}

#[derive(Debug, thiserror::Error)]
pub enum SteeringAdmissionError {
    #[error("steering target has finished")]
    TargetFinished,
    #[error("steering delivery key is bound to different content")]
    Conflict,
    #[error(transparent)]
    Storage(#[from] anyhow::Error),
}

impl SessionManager {
    pub async fn steering_delivery_status(
        &self,
        attempt_key: &str,
        delivery_id: &str,
    ) -> Result<Option<crate::acp::custom_requests::SteerSessionResponse>> {
        let row = sqlx::query("SELECT m.body, m.delivered_at IS NOT NULL AS consumed, p.run_id FROM session_mailbox m JOIN prompt_attempts p ON p.attempt_key = m.steering_attempt_key WHERE m.steering_attempt_key = ? AND m.dedupe_key = ?")
            .bind(attempt_key).bind(format!("steering:{delivery_id}")).fetch_optional(self.storage().pool().await?).await?;
        row.map(|row| {
            let message: crate::conversation::message::Message =
                serde_json::from_str(&row.try_get::<String, _>("body")?)?;
            Ok(crate::acp::custom_requests::SteerSessionResponse {
                run_id: row.try_get("run_id")?,
                message_id: message
                    .id
                    .ok_or_else(|| anyhow::anyhow!("steering message has no identity"))?,
                delivery_state: if row.try_get::<bool, _>("consumed")? {
                    crate::acp::custom_requests::SteeringDeliveryState::Consumed
                } else {
                    crate::acp::custom_requests::SteeringDeliveryState::Queued
                },
            })
        })
        .transpose()
    }

    pub async fn admit_steering_delivery(
        &self,
        attempt_key: &str,
        session_id: &str,
        run_id: &str,
        delivery_id: &str,
        digest: &str,
        message: &crate::conversation::message::Message,
    ) -> std::result::Result<
        crate::acp::custom_requests::SteerSessionResponse,
        SteeringAdmissionError,
    > {
        let admission: Result<_> = async {
            let pool = self.storage().pool().await?;
            let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
            let dedupe = format!("steering:{delivery_id}");
            if let Some(row) = sqlx::query("SELECT body, steering_digest, recipient_session_id FROM session_mailbox WHERE steering_attempt_key = ? AND dedupe_key = ?")
                .bind(attempt_key).bind(&dedupe).fetch_optional(&mut *tx).await? {
                let stored: crate::conversation::message::Message = serde_json::from_str(&row.try_get::<String, _>("body")?)?;
                let original_run: Option<String> = sqlx::query_scalar("SELECT run_id FROM prompt_attempts WHERE attempt_key = ?").bind(attempt_key).fetch_one(&mut *tx).await?;
                if row.try_get::<String, _>("steering_digest")? != digest || row.try_get::<String, _>("recipient_session_id")? != session_id || original_run.as_deref() != Some(run_id) || serde_json::to_value(&stored.content)? != serde_json::to_value(&message.content)? {
                    return Err(SteeringAdmissionError::Conflict.into());
                }
                tx.commit().await?;
                return self.steering_delivery_status(attempt_key, delivery_id).await?.ok_or_else(|| anyhow::anyhow!("accepted steering receipt disappeared"));
            }
            let live: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM prompt_attempts WHERE attempt_key = ? AND session_id = ? AND run_id = ? AND state = 'running' AND steering_sealed = 0)")
                .bind(attempt_key).bind(session_id).bind(run_id).fetch_one(&mut *tx).await?;
            if !live { return Err(SteeringAdmissionError::TargetFinished.into()); }
            sqlx::query("INSERT INTO session_mailbox (sender_session_id, recipient_session_id, kind, body, dedupe_key, steering_attempt_key, steering_digest) VALUES (?, ?, 'steering', ?, ?, ?, ?)")
                .bind(session_id).bind(session_id).bind(serde_json::to_string(message)?).bind(dedupe).bind(attempt_key).bind(digest).execute(&mut *tx).await?;
            tx.commit().await?;
            self.steering_delivery_status(attempt_key, delivery_id).await?.ok_or_else(|| anyhow::anyhow!("accepted steering receipt disappeared"))
        }.await;
        admission.map_err(|error| match error.downcast::<SteeringAdmissionError>() {
            Ok(error) => error,
            Err(error) => SteeringAdmissionError::Storage(error),
        })
    }

    /// SQLite serializes admission with the last pending check, so a successful seal cannot lose an accepted message.
    pub async fn seal_prompt_attempt_steering(&self, lease: &PromptAttemptLease) -> Result<bool> {
        let mut tx = self
            .storage()
            .pool()
            .await?
            .begin_with("BEGIN IMMEDIATE")
            .await?;
        let pending: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM session_mailbox WHERE steering_attempt_key = ? AND delivered_at IS NULL)").bind(&lease.key).fetch_one(&mut *tx).await?;
        if pending {
            tx.commit().await?;
            return Ok(false);
        }
        sqlx::query(
            "UPDATE prompt_attempts SET steering_sealed = 1 WHERE attempt_key = ? AND run_id = ?",
        )
        .bind(&lease.key)
        .bind(&lease.run_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(true)
    }

    pub async fn prompt_attempt_transcript(
        &self,
        request: &PromptAttemptTranscriptRequest,
    ) -> Result<PromptAttemptTranscriptResponse> {
        if request.after < 0 {
            bail!("invalid transcript cursor");
        }
        let status = self.prompt_attempt_status(&request.attempt_key).await?;
        if !status.stopped {
            bail!("attempt has not stopped");
        }
        let row = sqlx::query(
            r#"SELECT id, substr(document, ?, 16384) AS fragment, length(document) AS size FROM (
                SELECT id, json_object('id', id, 'messageId', message_id, 'role', role,
                    'content', json(content_json), 'metadata', json(coalesce(metadata_json, '{}')),
                    'created', created_timestamp) AS document
                FROM messages WHERE session_id = ? AND id > ? ORDER BY id LIMIT 1
            )"#,
        )
        .bind(i64::from(request.offset) + 1)
        .bind(status.session_id)
        .bind(request.after)
        .fetch_optional(self.storage().pool().await?)
        .await?;
        let Some(row) = row else {
            return Ok(PromptAttemptTranscriptResponse {
                done: true,
                ..Default::default()
            });
        };
        let id: i64 = row.try_get("id")?;
        let size: i64 = row.try_get("size")?;
        if i64::from(request.offset) >= size {
            bail!("transcript offset is outside the message");
        }
        let next_offset = request
            .offset
            .checked_add(16384)
            .ok_or_else(|| anyhow::anyhow!("transcript offset overflow"))?;
        let complete = i64::from(next_offset) >= size;
        Ok(PromptAttemptTranscriptResponse {
            message_id: Some(id),
            data: row.try_get("fragment")?,
            next_after: if complete { id } else { request.after },
            next_offset: if complete { 0 } else { next_offset },
            done: false,
        })
    }

    pub async fn session_has_prompt_attempt(&self, session_id: &str) -> Result<bool> {
        Ok(
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM prompt_attempts WHERE session_id = ?)")
                .bind(session_id)
                .fetch_one(self.storage().pool().await?)
                .await?,
        )
    }

    fn try_attempt_lock(&self, key: &str) -> Result<Option<File>> {
        if Uuid::parse_str(key)?.to_string() != key {
            bail!("attempt key must be a canonical UUID");
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(self.storage().attempt_lock_path(key))?;
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(Some(file)),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub async fn claim_prompt_attempt(
        &self,
        key: &str,
        digest: &str,
        session_id: &str,
    ) -> Result<Option<PromptAttemptLease>> {
        if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
            bail!("invalid prompt attempt digest");
        }
        let pool = self.storage().pool().await?;
        let Some(lock) = self.try_attempt_lock(key)? else {
            return Ok(None);
        };
        let run_id = format!("run_{}", Uuid::new_v4());
        let inserted = sqlx::query(
            "INSERT OR IGNORE INTO prompt_attempts
             (attempt_key, request_digest, session_id, run_id, state) VALUES (?, ?, ?, ?, 'running')",
        )
        .bind(key)
        .bind(digest)
        .bind(session_id)
        .bind(&run_id)
        .execute(pool)
        .await?
        .rows_affected();
        if inserted == 0 {
            let original: Option<String> = sqlx::query_scalar(
                "SELECT request_digest FROM prompt_attempts WHERE attempt_key = ?",
            )
            .bind(key)
            .fetch_one(pool)
            .await?;
            if original.is_some_and(|original| original != digest) {
                bail!("attempt key is already bound to a different request");
            }
            return Ok(None);
        }
        Ok(Some(PromptAttemptLease {
            key: key.to_owned(),
            run_id,
            _lock: lock,
        }))
    }

    pub async fn prompt_attempt_cancel_requested(&self, key: &str) -> Result<bool> {
        Ok(sqlx::query_scalar::<_, String>(
            "SELECT state FROM prompt_attempts WHERE attempt_key = ?",
        )
        .bind(key)
        .fetch_one(self.storage().pool().await?)
        .await?
            != "running")
    }

    /// Terminal CAS preserves a completed result when a later transport notification fails.
    pub async fn finish_prompt_attempt(
        &self,
        lease: &PromptAttemptLease,
        state: PromptAttemptState,
        result: Option<serde_json::Value>,
    ) -> Result<()> {
        let state = match state {
            PromptAttemptState::Completed => "completed",
            PromptAttemptState::Failed => "failed",
            PromptAttemptState::Cancelled => "cancelled",
            PromptAttemptState::Interrupted => "interrupted",
            _ => bail!("attempt completion requires a terminal state"),
        };
        let mut tx = self
            .storage()
            .pool()
            .await?
            .begin_with("BEGIN IMMEDIATE")
            .await?;
        if state == "completed" {
            let pending: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM session_mailbox m JOIN prompt_attempts p ON p.attempt_key = m.steering_attempt_key WHERE p.attempt_key = ? AND p.run_id = ? AND p.state = 'running' AND m.delivered_at IS NULL)")
                .bind(&lease.key).bind(&lease.run_id).fetch_one(&mut *tx).await?;
            if pending {
                bail!("cannot complete an attempt with unconsumed steering");
            }
        }
        sqlx::query(
            "UPDATE prompt_attempts SET
             state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' ELSE ? END,
             result_json = CASE WHEN state = 'cancel_requested' THEN NULL ELSE ? END,
             steering_sealed = 1,
             updated_at = CURRENT_TIMESTAMP
             WHERE attempt_key = ? AND run_id = ? AND state IN ('running', 'cancel_requested')",
        )
        .bind(state)
        .bind(result.map(|result| result.to_string()))
        .bind(&lease.key)
        .bind(&lease.run_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn cancel_prompt_attempt(&self, key: &str) -> Result<PromptAttemptResponse> {
        // Validate before storing the tombstone; no prompt or session loading occurs here.
        if Uuid::parse_str(key)?.to_string() != key {
            bail!("attempt key must be a canonical UUID");
        }
        sqlx::query(
            "INSERT INTO prompt_attempts (attempt_key, state) VALUES (?, 'cancel_requested')
             ON CONFLICT (attempt_key) DO UPDATE SET
             state = CASE WHEN state = 'running' THEN 'cancel_requested' ELSE state END,
             updated_at = CURRENT_TIMESTAMP",
        )
        .bind(key)
        .execute(self.storage().pool().await?)
        .await?;
        self.prompt_attempt_status(key).await
    }

    pub async fn prompt_attempt_status(&self, key: &str) -> Result<PromptAttemptResponse> {
        let pool = self.storage().pool().await?;
        let lock = self.try_attempt_lock(key)?;
        if lock.is_some() {
            // An acquired OS lock proves no executor still owns this attempt, across processes.
            sqlx::query(
                "UPDATE prompt_attempts SET
                 state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' ELSE 'interrupted' END,
                 updated_at = CURRENT_TIMESTAMP
                 WHERE attempt_key = ? AND state IN ('running', 'cancel_requested')",
            )
            .bind(key)
            .execute(pool)
            .await?;
        }
        let row = sqlx::query(
            "SELECT state, session_id, run_id, result_json FROM prompt_attempts WHERE attempt_key = ?",
        )
        .bind(key)
        .fetch_optional(pool)
        .await?;
        let Some(row) = row else {
            return Ok(PromptAttemptResponse::default());
        };
        Ok(PromptAttemptResponse {
            state: Some(serde_json::from_value(serde_json::Value::String(
                row.try_get("state")?,
            ))?),
            session_id: row.try_get("session_id")?,
            run_id: row.try_get("run_id")?,
            stopped: lock.is_some(),
            result: row
                .try_get::<Option<String>, _>("result_json")?
                .map(|result| serde_json::from_str(&result))
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod steering_tests {
    use super::*;
    use crate::acp::custom_requests::{SteerSessionResponse, SteeringDeliveryState};
    use crate::config::GooseMode;
    use crate::conversation::message::Message;
    use crate::session::SessionType;

    async fn setup() -> (
        tempfile::TempDir,
        SessionManager,
        String,
        PromptAttemptLease,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let manager = SessionManager::new(directory.path().to_path_buf());
        let session = manager
            .create_session(
                directory.path().to_path_buf(),
                "owner".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let lease = manager
            .claim_prompt_attempt(&Uuid::new_v4().to_string(), &"a".repeat(64), &session.id)
            .await
            .unwrap()
            .unwrap();
        (directory, manager, session.id, lease)
    }

    #[tokio::test]
    async fn steering_receipts_survive_a_second_worker_and_consumption() {
        let (directory, manager, session, lease) = setup().await;
        let message = Message::user()
            .with_text("four slides")
            .with_id("steer_stable")
            .with_steer();
        let queued: SteerSessionResponse = manager
            .admit_steering_delivery(
                &lease.key,
                &session,
                &lease.run_id,
                "turn",
                &"b".repeat(64),
                &message,
            )
            .await
            .unwrap();
        assert_eq!(queued.delivery_state, SteeringDeliveryState::Queued);
        let worker = SessionManager::new(directory.path().to_path_buf());
        let mut retry = Message::user()
            .with_text("four slides")
            .with_id("steer_other")
            .with_steer();
        retry.created = message.created + 60;
        let duplicate = worker
            .admit_steering_delivery(
                &lease.key,
                &session,
                &lease.run_id,
                "turn",
                &"b".repeat(64),
                &retry,
            )
            .await
            .unwrap();
        assert_eq!(queued.message_id, duplicate.message_id);
        assert_eq!(
            worker
                .pending_session_messages(&session)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(!manager.seal_prompt_attempt_steering(&lease).await.unwrap());
        let pending = worker
            .pending_session_messages(&session)
            .await
            .unwrap()
            .remove(0);
        worker
            .deliver_session_message(&session, pending.id, &pending.prompt().unwrap())
            .await
            .unwrap();
        worker
            .add_message(&session, &pending.prompt().unwrap())
            .await
            .unwrap();
        assert_eq!(
            worker
                .get_session(&session, true)
                .await
                .unwrap()
                .conversation
                .unwrap()
                .messages()
                .len(),
            1
        );
        assert!(manager.seal_prompt_attempt_steering(&lease).await.unwrap());
        manager
            .finish_prompt_attempt(&lease, PromptAttemptState::Completed, None)
            .await
            .unwrap();
        let receipt = worker
            .admit_steering_delivery(
                &lease.key,
                &session,
                &lease.run_id,
                "turn",
                &"b".repeat(64),
                &retry,
            )
            .await
            .unwrap();
        assert_eq!(receipt.delivery_state, SteeringDeliveryState::Consumed);
        assert!(matches!(
            worker
                .admit_steering_delivery(
                    &lease.key,
                    &session,
                    &lease.run_id,
                    "turn",
                    &"c".repeat(64),
                    &retry
                )
                .await,
            Err(SteeringAdmissionError::Conflict)
        ));
        assert!(matches!(
            worker
                .admit_steering_delivery(
                    &lease.key,
                    &session,
                    &lease.run_id,
                    "turn",
                    &"b".repeat(64),
                    &Message::user().with_text("five slides")
                )
                .await,
            Err(SteeringAdmissionError::Conflict)
        ));
        assert!(matches!(
            worker
                .admit_steering_delivery(
                    &lease.key,
                    &session,
                    &lease.run_id,
                    "other",
                    &"b".repeat(64),
                    &retry
                )
                .await,
            Err(SteeringAdmissionError::TargetFinished)
        ));
        assert!(worker
            .steering_delivery_status(&lease.key, "unknown")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn admission_and_sealing_cannot_both_win() {
        for _ in 0..12 {
            let (_directory, manager, session, lease) = setup().await;
            let message = Message::user()
                .with_text("guidance")
                .with_id("steer_race")
                .with_steer();
            let digest = "b".repeat(64);
            let (sealed, admitted) = tokio::join!(
                manager.seal_prompt_attempt_steering(&lease),
                manager.admit_steering_delivery(
                    &lease.key,
                    &session,
                    &lease.run_id,
                    "turn",
                    &digest,
                    &message
                )
            );
            let sealed = sealed.unwrap();
            assert_eq!(sealed, admitted.is_err());
            if sealed {
                assert!(matches!(
                    admitted,
                    Err(SteeringAdmissionError::TargetFinished)
                ));
            } else {
                assert_eq!(
                    manager
                        .pending_session_messages(&session)
                        .await
                        .unwrap()
                        .len(),
                    1
                );
            }
        }
    }

    #[tokio::test]
    async fn stopped_or_lost_executor_rejects_new_admission_without_replay() {
        let (directory, manager, session, lease) = setup().await;
        let key = lease.key.clone();
        let run = lease.run_id.clone();
        let message = Message::user()
            .with_text("guidance")
            .with_id("steer_stopped")
            .with_steer();
        manager.cancel_prompt_attempt(&key).await.unwrap();
        assert!(matches!(
            manager
                .admit_steering_delivery(&key, &session, &run, "turn", &"b".repeat(64), &message)
                .await,
            Err(SteeringAdmissionError::TargetFinished)
        ));
        drop(lease);
        let worker = SessionManager::new(directory.path().to_path_buf());
        assert!(worker.prompt_attempt_status(&key).await.unwrap().stopped);
        assert!(worker
            .steering_delivery_status(&key, "turn")
            .await
            .unwrap()
            .is_none());
    }
}
