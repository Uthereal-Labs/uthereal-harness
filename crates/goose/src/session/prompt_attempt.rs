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
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"#;

/// Held through agent shutdown. Lock files are retained to avoid inode replacement races.
pub struct PromptAttemptLease {
    pub key: String,
    pub run_id: String,
    _lock: File,
}

impl SessionManager {
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
        sqlx::query(
            "UPDATE prompt_attempts SET
             state = CASE WHEN state = 'cancel_requested' THEN 'cancelled' ELSE ? END,
             result_json = CASE WHEN state = 'cancel_requested' THEN NULL ELSE ? END,
             updated_at = CURRENT_TIMESTAMP
             WHERE attempt_key = ? AND run_id = ? AND state IN ('running', 'cancel_requested')",
        )
        .bind(state)
        .bind(result.map(|result| result.to_string()))
        .bind(&lease.key)
        .bind(&lease.run_id)
        .execute(self.storage().pool().await?)
        .await?;
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
