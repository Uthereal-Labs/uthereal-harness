use super::SessionManager;
use anyhow::{bail, Result};
use goose_sdk_types::custom_requests::{
    TaskAdmission, TaskEvidence, TaskEvidenceRequest, TaskEvidenceResponse, TaskNoticeRequest,
    TaskNoticeResponse, TaskNoticeStatus, TaskOutcome, TaskTerminalStatus, TaskToolReceipt,
    ToolReceiptTransportStatus,
};
use sqlx::Row;
use std::collections::{HashMap, HashSet};

pub(crate) const TASK_ADMISSION_VERSION: &str = "task_admission_v1";
const TASK_PAGE_LIMIT: usize = 32;
const TOOL_BLOCK_LIMIT: i64 = 1024;
const RECEIPT_LIMIT: usize = 128;
const STRUCTURED_RESULT_LIMIT: usize = 32768;
const RESPONSE_LIMIT: usize = 262144;
const NOTICE_TEXT_LIMIT: usize = 65536;
const EDITOR_RESULT_LIMIT: usize = 16384;

/// The form in which a delegated task's artifact key is stored and compared:
/// whitespace collapsed and lowercased. Tool servers that name artifacts (such
/// as Cortex channels) must apply the same normalization.
pub fn normalize_artifact_key(key: &str) -> String {
    key.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
const NOTICE_ARTIFACT_KEYS_LIMIT: usize = 4;

impl SessionManager {
    pub async fn capture_task_admission(
        &self,
        parent_session_id: &str,
        task_id: &str,
        source_name: &str,
    ) -> Result<TaskAdmission> {
        let binding: Option<(String, String)> =
            sqlx::query_as("SELECT attempt_key, run_id FROM prompt_attempts WHERE session_id = ?")
                .bind(parent_session_id)
                .fetch_optional(self.storage().pool().await?)
                .await?;
        let (attempt_key, parent_run_id) = binding
            .map(|(key, run)| (Some(key), Some(run)))
            .unwrap_or_default();
        Ok(TaskAdmission {
            task_id: task_id.to_string(),
            parent_session_id: parent_session_id.to_string(),
            parent_run_id,
            attempt_key,
            source_name: source_name.to_string(),
            artifact_key: None,
            previous_task_id: None,
        })
    }

    pub async fn task_admission(&self, task_id: &str) -> Result<Option<TaskAdmission>> {
        let child = self.get_session(task_id, false).await?;
        let admission = child
            .extension_data
            .get_extension_state("summon", TASK_ADMISSION_VERSION)
            .map(|value| serde_json::from_value::<TaskAdmission>(value.clone()))
            .transpose()?;
        if let Some(admission) = &admission {
            if admission.task_id != task_id
                || child.parent_session_id.as_deref() != Some(&admission.parent_session_id)
            {
                bail!("Task admission ownership does not match the child session");
            }
        }
        Ok(admission)
    }

    pub async fn enqueue_task_outcome(
        &self,
        task_id: &str,
        body: &str,
        status: TaskTerminalStatus,
    ) -> Result<bool> {
        if body.trim().is_empty() {
            bail!("Session message cannot be empty");
        }
        let admission = self
            .task_admission(task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Task has no immutable async admission"))?;
        let outcome = TaskOutcome { admission, status };
        let result = sqlx::query(
            "INSERT OR IGNORE INTO session_mailbox
             (sender_session_id, recipient_session_id, kind, body, outcome_json, dedupe_key)
             SELECT id, parent_session_id, 'completion', ?, ?, ? FROM sessions
             WHERE id = ? AND parent_session_id = ?",
        )
        .bind(body)
        .bind(serde_json::to_string(&outcome)?)
        .bind(format!("completion:{task_id}"))
        .bind(task_id)
        .bind(&outcome.admission.parent_session_id)
        .execute(self.storage().pool().await?)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn task_evidence(
        &self,
        request: &TaskEvidenceRequest,
    ) -> Result<TaskEvidenceResponse> {
        if uuid::Uuid::parse_str(&request.attempt_key)?.to_string() != request.attempt_key
            || request.tool_names.len() > 8
            || request.tool_names.iter().any(|name| {
                name.is_empty() || name.len() > 200 || name.chars().any(char::is_control)
            })
        {
            bail!("Invalid task evidence request");
        }
        let pool = self.storage().pool().await?;
        let binding: Option<(String, String)> = sqlx::query_as(
            "SELECT session_id, run_id FROM prompt_attempts
             WHERE attempt_key = ? AND session_id IS NOT NULL AND run_id IS NOT NULL",
        )
        .bind(&request.attempt_key)
        .fetch_optional(pool)
        .await?;
        let Some((parent_session_id, parent_run_id)) = binding else {
            bail!("Attempt does not own a parent session and run");
        };
        if let Some(cursor) = &request.after_task_id {
            let admission = self.task_admission(cursor).await?;
            if !admission.is_some_and(|a| {
                a.parent_session_id == parent_session_id
                    && a.parent_run_id.as_deref() == Some(&parent_run_id)
                    && a.attempt_key.as_deref() == Some(&request.attempt_key)
            }) {
                bail!("Task evidence cursor does not belong to this attempt");
            }
        }
        let rows = sqlx::query(
            r#"SELECT id, extension_data, json_extract(extension_data, '$."summon.task_admission_v1"') AS admission
             FROM sessions WHERE parent_session_id = ? AND id > ? AND (? IS NULL OR id = ?)
              AND json_extract(extension_data, '$."summon.task_admission_v1".parentSessionId') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".parentRunId') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".attemptKey') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".taskId') = id
             ORDER BY id LIMIT ?"#,
        )
        .bind(&parent_session_id)
        .bind(request.after_task_id.as_deref().unwrap_or(""))
        .bind(&request.task_id)
        .bind(&request.task_id)
        .bind(&parent_session_id)
        .bind(&parent_run_id)
        .bind(&request.attempt_key)
        .bind((TASK_PAGE_LIMIT + 1) as i64)
        .fetch_all(pool)
        .await?;
        let more = rows.len() > TASK_PAGE_LIMIT;
        let mut response = TaskEvidenceResponse {
            attempt_key: request.attempt_key.clone(),
            parent_session_id,
            parent_run_id,
            tasks: Vec::new(),
            next_task_id: None,
            evidence_complete: !more,
        };
        // Reserve cursor and array punctuation before retaining bounded projections.
        let mut bytes = serde_json::to_vec(&response)?.len() + 256;
        for row in rows.into_iter().take(TASK_PAGE_LIMIT) {
            let task_id: String = row.try_get("id")?;
            let admission: TaskAdmission =
                serde_json::from_str(&row.try_get::<String, _>("admission")?)?;
            let outcome_json: Option<String> = sqlx::query_scalar(
                "SELECT outcome_json FROM session_mailbox
                 WHERE sender_session_id = ? AND recipient_session_id = ? AND kind = 'completion'
                 ORDER BY id LIMIT 1",
            )
            .bind(&task_id)
            .bind(&response.parent_session_id)
            .fetch_optional(pool)
            .await?
            .flatten();
            let outcome: Option<TaskOutcome> = outcome_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?;
            if outcome.as_ref().is_some_and(|o| o.admission != admission) {
                bail!("Task outcome does not match its immutable admission");
            }
            let extension_data: serde_json::Value =
                serde_json::from_str(&row.try_get::<String, _>("extension_data")?)?;
            let policy = &extension_data["summon.v1"];
            let mut task = TaskEvidence {
                artifact_key: policy["artifact_key"].as_str().map(str::to_owned),
                previous_task_id: policy["previous_task_id"].as_str().map(str::to_owned),
                admission,
                outcome,
                receipts: Vec::new(),
                evidence_complete: true,
            };
            let task_bytes = serde_json::to_vec(&task)?.len();
            if bytes + task_bytes > RESPONSE_LIMIT {
                response.evidence_complete = false;
                response.next_task_id = response.tasks.last().map(|t| t.admission.task_id.clone());
                break;
            }
            bytes += task_bytes + 1;
            self.task_receipts(&request.tool_names, &mut task, &mut bytes)
                .await?;
            response.evidence_complete &= task.evidence_complete;
            response.tasks.push(task);
        }
        if more && response.next_task_id.is_none() {
            response.next_task_id = response.tasks.last().map(|t| t.admission.task_id.clone());
        }
        if serde_json::to_vec(&response)?.len() > RESPONSE_LIMIT {
            bail!("Task admission exceeds the task evidence response limit");
        }
        Ok(response)
    }

    /// Queue a notice (channel news or an editor result) for the attempt's
    /// running task that holds one of the requested artifacts (or has the
    /// requested ID). A task is running
    /// until its terminal outcome is queued for its parent; a finished task is
    /// reported as `not_running` and receives nothing.
    pub async fn queue_task_notice(
        &self,
        request: &TaskNoticeRequest,
    ) -> Result<TaskNoticeResponse> {
        if uuid::Uuid::parse_str(&request.attempt_key)?.to_string() != request.attempt_key
            || request.text.trim().is_empty()
            || request.text.len() > NOTICE_TEXT_LIMIT
            || (request.task_id.is_none() && request.artifact_keys.is_empty())
            || request.artifact_keys.len() > NOTICE_ARTIFACT_KEYS_LIMIT
            || request
                .artifact_keys
                .iter()
                .any(|key| key.is_empty() || key.len() > 512)
            || request
                .dedupe_key
                .as_deref()
                .is_some_and(|key| key.is_empty() || key.len() > 200)
            || request.editor_result.as_ref().is_some_and(|result| {
                result
                    .get("idempotency_key")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(|key| key.is_empty() || key.len() > 240)
                    || !result.get("receipt").is_some_and(serde_json::Value::is_object)
                    || !serde_json::to_vec(result).is_ok_and(|bytes| bytes.len() <= EDITOR_RESULT_LIMIT)
            })
        {
            bail!("Invalid task notice request");
        }
        let pool = self.storage().pool().await?;
        let binding: Option<(String, String)> = sqlx::query_as(
            "SELECT session_id, run_id FROM prompt_attempts
             WHERE attempt_key = ? AND session_id IS NOT NULL AND run_id IS NOT NULL",
        )
        .bind(&request.attempt_key)
        .fetch_optional(pool)
        .await?;
        let Some((parent_session_id, parent_run_id)) = binding else {
            bail!("Attempt does not own a parent session and run");
        };
        let task_id: Option<String> = sqlx::query_scalar(
            r#"SELECT id FROM sessions
             WHERE parent_session_id = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".parentSessionId') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".parentRunId') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".attemptKey') = ?
              AND json_extract(extension_data, '$."summon.task_admission_v1".taskId') = id
              AND (id = ? OR json_extract(extension_data, '$."summon.v1".artifact_key') IN (SELECT value FROM json_each(?)))
              AND NOT EXISTS (
                SELECT 1 FROM session_mailbox m
                WHERE m.sender_session_id = sessions.id
                  AND m.recipient_session_id = sessions.parent_session_id
                  AND m.kind = 'completion'
              )
             ORDER BY created_at DESC, id DESC LIMIT 1"#,
        )
        .bind(&parent_session_id)
        .bind(&parent_session_id)
        .bind(&parent_run_id)
        .bind(&request.attempt_key)
        .bind(request.task_id.as_deref())
        .bind(serde_json::to_string(
            &request
                .artifact_keys
                .iter()
                .map(String::as_str)
                .map(normalize_artifact_key)
                .collect::<Vec<_>>(),
        )?)
        .fetch_optional(pool)
        .await?;
        let Some(task_id) = task_id else {
            return Ok(TaskNoticeResponse {
                status: TaskNoticeStatus::NotRunning,
                task_id: None,
            });
        };
        let notice = super::ChannelNotice {
            text: request.text.clone(),
            wake: request.wake,
            refresh_tools: request.refresh_tools,
            editor_result: request.editor_result.clone(),
        };
        sqlx::query(
            "INSERT OR IGNORE INTO session_mailbox
             (sender_session_id, recipient_session_id, kind, body, dedupe_key)
             VALUES (?, ?, 'channel', ?, ?)",
        )
        .bind(&parent_session_id)
        .bind(&task_id)
        .bind(serde_json::to_string(&notice)?)
        .bind(request.dedupe_key.as_deref())
        .execute(pool)
        .await?;
        Ok(TaskNoticeResponse {
            status: TaskNoticeStatus::Queued,
            task_id: Some(task_id),
        })
    }

    async fn task_receipts(
        &self,
        tool_names: &[String],
        task: &mut TaskEvidence,
        bytes: &mut usize,
    ) -> Result<()> {
        if tool_names.is_empty() {
            return Ok(());
        }
        let pool = self.storage().pool().await?;
        let task_id = &task.admission.task_id;
        let block_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM (SELECT 1 FROM messages m, json_each(m.content_json) c
             WHERE m.session_id = ? AND json_extract(c.value, '$.type') IN ('toolRequest', 'toolResponse') LIMIT ?)",
        )
        .bind(task_id)
        .bind(TOOL_BLOCK_LIMIT + 1)
        .fetch_one(pool)
        .await?;
        task.evidence_complete = block_count <= TOOL_BLOCK_LIMIT;
        let requests = sqlx::query(
            "SELECT json_extract(c.value, '$.id') AS call_id,
                    json_extract(c.value, '$.toolCall.value.name') AS tool_name,
                    m.id AS message_id, CAST(c.key AS INTEGER) AS block_id
             FROM messages m, json_each(m.content_json) c WHERE m.session_id = ?
              AND json_extract(c.value, '$.type') = 'toolRequest'
              AND json_extract(c.value, '$.toolCall.status') = 'success'
              AND json_extract(c.value, '$.toolCall.value.name') IN (SELECT value FROM json_each(?))
             ORDER BY m.id, CAST(c.key AS INTEGER) LIMIT ?",
        )
        .bind(task_id)
        .bind(serde_json::to_string(tool_names)?)
        .bind((RECEIPT_LIMIT + 1) as i64)
        .fetch_all(pool)
        .await?;
        task.evidence_complete &= requests.len() <= RECEIPT_LIMIT;
        let mut calls = HashMap::new();
        let mut ambiguous = HashSet::new();
        for row in requests.into_iter().take(RECEIPT_LIMIT) {
            let id: String = row.try_get("call_id")?;
            let name: String = row.try_get("tool_name")?;
            let position = (
                row.try_get::<i64, _>("message_id")?,
                row.try_get::<i64, _>("block_id")?,
            );
            if calls.insert(id.clone(), (name, position)).is_some() {
                ambiguous.insert(id);
                task.evidence_complete = false;
            }
        }
        let call_ids = serde_json::to_string(&calls.keys().collect::<Vec<_>>())?;
        let request_counts = sqlx::query(
            "SELECT json_extract(c.value, '$.id') AS call_id, COUNT(*) AS count
             FROM messages m, json_each(m.content_json) c WHERE m.session_id = ?
              AND json_extract(c.value, '$.type') = 'toolRequest'
              AND json_extract(c.value, '$.id') IN (SELECT value FROM json_each(?))
             GROUP BY json_extract(c.value, '$.id') LIMIT ?",
        )
        .bind(task_id)
        .bind(&call_ids)
        .bind(RECEIPT_LIMIT as i64)
        .fetch_all(pool)
        .await?;
        for row in request_counts {
            if row.try_get::<i64, _>("count")? != 1 {
                ambiguous.insert(row.try_get::<String, _>("call_id")?);
                task.evidence_complete = false;
            }
        }
        let rows = sqlx::query(
            "SELECT json_extract(c.value, '$.id') AS call_id,
                    m.id AS message_id, CAST(c.key AS INTEGER) AS block_id,
                    json_extract(c.value, '$.toolResult.status') AS transport_status,
                    json_extract(c.value, '$.toolResult.value.isError') AS is_error,
                    CASE WHEN json_type(c.value, '$.toolResult.value.structuredContent') IS NOT NULL
                         AND length(CAST(json_quote(json_extract(c.value, '$.toolResult.value.structuredContent')) AS BLOB)) <= ?
                         THEN json_quote(json_extract(c.value, '$.toolResult.value.structuredContent')) END AS structured_result,
                    CASE WHEN length(CAST(json_quote(json_extract(c.value, '$.toolResult.value.structuredContent')) AS BLOB)) > ? THEN 1 ELSE 0 END AS oversized
             FROM messages m, json_each(m.content_json) c WHERE m.session_id = ?
              AND json_extract(c.value, '$.type') = 'toolResponse'
              AND json_extract(c.value, '$.id') IN (SELECT value FROM json_each(?))
             ORDER BY m.id, CAST(c.key AS INTEGER) LIMIT ?",
        )
        .bind(STRUCTURED_RESULT_LIMIT as i64)
        .bind(STRUCTURED_RESULT_LIMIT as i64)
        .bind(task_id)
        .bind(&call_ids)
        .bind((RECEIPT_LIMIT + 1) as i64)
        .fetch_all(pool)
        .await?;
        task.evidence_complete &= rows.len() <= RECEIPT_LIMIT;
        let mut receipts = HashMap::new();
        for row in rows.into_iter().take(RECEIPT_LIMIT) {
            let id: String = row.try_get("call_id")?;
            let position = (
                row.try_get::<i64, _>("message_id")?,
                row.try_get::<i64, _>("block_id")?,
            );
            if position <= calls[&id].1 {
                ambiguous.insert(id.clone());
                task.evidence_complete = false;
            }
            if ambiguous.contains(&id) {
                continue;
            }
            let transport_status = match row.try_get::<String, _>("transport_status")?.as_str() {
                "success" => ToolReceiptTransportStatus::Success,
                "error" => ToolReceiptTransportStatus::Error,
                _ => {
                    task.evidence_complete = false;
                    continue;
                }
            };
            let result: Option<String> = row.try_get("structured_result")?;
            let receipt = TaskToolReceipt {
                task_id: task_id.clone(),
                call_id: id.clone(),
                tool_name: calls[&id].0.clone(),
                transport_status,
                is_error: row.try_get::<Option<bool>, _>("is_error")?,
                structured_result: result.as_deref().map(serde_json::from_str).transpose()?,
            };
            task.evidence_complete &= !row.try_get::<bool, _>("oversized")?;
            if let Some((_, previous)) = receipts.get(&id) {
                if previous != &receipt {
                    ambiguous.insert(id);
                    task.evidence_complete = false;
                }
            } else {
                receipts.insert(id, (position, receipt));
            }
        }
        task.evidence_complete &= calls.keys().all(|id| receipts.contains_key(id));
        let mut receipts: Vec<_> = receipts.into_values().collect();
        receipts.sort_by_key(|(position, _)| *position);
        for (_, receipt) in receipts {
            if ambiguous.contains(&receipt.call_id) {
                continue;
            }
            let size = serde_json::to_vec(&receipt)?.len();
            if *bytes + size > RESPONSE_LIMIT {
                task.evidence_complete = false;
                break;
            }
            *bytes += size + 1;
            task.receipts.push(receipt);
        }
        Ok(())
    }
}
