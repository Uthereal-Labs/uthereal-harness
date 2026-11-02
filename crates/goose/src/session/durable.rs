//! Credential-free execution state, committed before effect dispatch and after observations.
use super::{SessionManager, SessionType};
use crate::conversation::{message::Message, Conversation};
use anyhow::{ensure, Context, Result};
use rmcp::model::{CallToolRequestParams, CallToolResult, ErrorData};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

pub const WAIT_META: &str = "uthereal/durableWait";
pub const OPERATION_META: &str = "uthereal/durableOperation";

#[derive(Debug, thiserror::Error)]
pub enum DurableError {
    #[error("durable_effect_uncertain: {0}")]
    EffectUncertain(String),
    #[error("durable_snapshot_invalid: {0}")]
    SnapshotInvalid(String),
    #[error("durable_completion_invalid: {0}")]
    CompletionInvalid(String),
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurableAdmission {
    pub logical_run_id: String,
    pub attempt_id: String,
    pub epoch: u64,
    pub request_digest: String,
    pub expected_generation: u64,
    pub callback_url: String,
    pub callback_token: String,
    pub restore: Option<DurableSnapshot>,
    pub accepted_reply: Option<AcceptedReply>,
    pub continuation_context: Option<ContinuationContext>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuationContext {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceptedReply {
    pub wait_id: String,
    pub tool_call_id: String,
    pub input_revision: u64,
    pub result: CallToolResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurableWait {
    pub version: u32,
    pub wait_id: String,
    #[serde(default)]
    pub dependent_task_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuspendedCall {
    pub task_id: String,
    pub tool_call_id: String,
    pub wait: DurableWait,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectReceipt {
    pub task_id: String,
    pub tool_call_id: String,
    pub operation_id: String,
    pub request: CallToolRequestParams,
    /// None means the external effect may have happened. Never redispatch it.
    pub result: Option<std::result::Result<CallToolResult, ErrorData>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableTask {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub name: String,
    #[serde(default = "root_working_dir")]
    pub working_dir: String,
    pub session_type: SessionType,
    pub conversation: Conversation,
    pub task_admission: Option<serde_json::Value>,
    pub task_execution: Option<serde_json::Value>,
    pub task_policy: Option<serde_json::Value>,
    pub recipe: Option<crate::recipe::Recipe>,
    pub extension_names: Vec<String>,
    pub completed: bool,
}

fn root_working_dir() -> String {
    ".".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableMailbox {
    pub id: i64,
    pub sender_session_id: String,
    pub recipient_session_id: String,
    pub kind: String,
    pub body: String,
    pub outcome_json: Option<String>,
    pub dedupe_key: Option<String>,
    pub steering_attempt_key: Option<String>,
    pub steering_digest: Option<String>,
    pub delivered: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompletionMarker {
    pub version: u32,
    pub message_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurableCompletion {
    pub version: u32,
    pub logical_run_id: String,
    pub request_digest: String,
    pub root_session_id: String,
    pub native_run_id: String,
    pub producing_attempt_id: String,
    pub generation: u64,
    pub consumed_input_revision: u64,
    pub message_id: String,
    pub text: String,
}

fn required_nullable<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

pub use super::durable_usage::DurableUsageReceipt;

pub const RESPONSE_OPERATION: &str = "durable";
pub const RESPONSE_NOTE: &str = "inference";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InferenceReceipt {
    pub version: u32,
    pub task_id: String,
    pub native_run_id: String,
    pub producing_attempt_id: String,
    #[serde(rename = "responseMessageIds")]
    pub message_ids: Vec<String>,
    pub processed: bool,
    pub usage: DurableUsageReceipt,
    #[serde(deserialize_with = "required_nullable")]
    pub output_message_id: Option<String>,
}

fn response_receipts(conversation: &Conversation, task_id: &str) -> Result<Vec<InferenceReceipt>> {
    let mut identities: HashMap<&str, (&Message, usize)> = HashMap::new();
    for message in conversation.messages() {
        if let Some(id) = &message.id {
            let entry = identities.entry(id).or_insert((message, 0));
            entry.1 += 1;
        }
    }
    let mut receipts = Vec::new();
    for (end, message) in conversation.messages().iter().enumerate() {
        let Some(value) = message
            .metadata
            .operation_note(RESPONSE_OPERATION, RESPONSE_NOTE)
        else {
            continue;
        };
        let receipt: InferenceReceipt = serde_json::from_value(value.clone())?;
        receipt.usage.validate()?;
        ensure!(
            receipt.version == 1
                && receipt.task_id == task_id
                && !receipt.native_run_id.is_empty()
                && !receipt.producing_attempt_id.is_empty()
                && !receipt.message_ids.is_empty()
                && receipt.message_ids.iter().all(|id| !id.is_empty())
                && receipt.message_ids.len() <= end + 1,
            "invalid inference receipt authority"
        );
        let start = end + 1 - receipt.message_ids.len();
        ensure!(
            conversation.messages()[start..=end]
                .iter()
                .zip(&receipt.message_ids)
                .all(|(message, id)| message.id.as_ref() == Some(id))
                && receipt.message_ids.iter().collect::<HashSet<_>>().len()
                    == receipt.message_ids.len(),
            "inference response batch is incomplete"
        );
        ensure!(
            receipt.message_ids.iter().all(|id| identities
                .get(id.as_str())
                .is_some_and(|(_, count)| *count == 1)),
            "ambiguous inference response identity"
        );
        ensure!(
            receipt.processed == receipt.output_message_id.is_some(),
            "invalid inference processing state"
        );
        if let Some(id) = &receipt.output_message_id {
            ensure!(!id.is_empty() && identities.get(id.as_str()).is_some_and(|(message, count)| *count == 1 && eligible_completion(message)), "processed inference has no unique valid output");
        }
        receipts.push(receipt);
    }
    Ok(receipts)
}

pub(crate) fn eligible_completion(message: &Message) -> bool {
    message.role == rmcp::model::Role::Assistant
        && message.is_user_visible()
        && !message.metadata.output_token_limit_reached
        && !message.is_tool_call()
        && !message
            .content
            .iter()
            .any(|content| content.as_error().is_some())
        && !message.as_concat_text().trim().is_empty()
        && message.as_concat_text() != crate::agents::state_machine::MAX_TURNS_MESSAGE
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurableSnapshot {
    pub schema_version: u32,
    pub logical_run_id: String,
    pub root_session_id: String,
    pub request_digest: String,
    pub generation: u64,
    pub consumed_input_revision: u64,
    pub native_run_id: Option<String>,
    #[serde(deserialize_with = "required_nullable")]
    pub producing_attempt_id: Option<String>,
    pub tasks: Vec<DurableTask>,
    pub mailbox: Vec<DurableMailbox>,
    pub effects: Vec<EffectReceipt>,
    pub pending_wait: Option<SuspendedCall>,
    #[serde(deserialize_with = "required_nullable")]
    pub completion: Option<CompletionMarker>,
    #[serde(default)]
    pub applied_context_ids: Vec<String>,
}

fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

impl DurableSnapshot {
    pub fn validate(&self) -> Result<()> {
        self.validate_snapshot()
            .map_err(|error| DurableError::SnapshotInvalid(error.to_string()).into())
    }

    fn validate_snapshot(&self) -> Result<()> {
        ensure!(
            self.schema_version == 2,
            "unsupported durable snapshot schema"
        );
        ensure!(
            self.native_run_id.is_some() == self.producing_attempt_id.is_some(),
            "durable producing authority mismatch"
        );
        ensure!(
            self.producing_attempt_id
                .as_ref()
                .is_none_or(|id| !id.is_empty()),
            "empty durable producing attempt"
        );
        let mut tasks = HashMap::new();
        for task in &self.tasks {
            ensure!(
                std::path::Path::new(&task.working_dir)
                    .components()
                    .all(|component| matches!(
                        component,
                        std::path::Component::CurDir | std::path::Component::Normal(_)
                    )),
                "durable task working directory escapes its workspace"
            );
            ensure!(
                valid_session_id(&task.session_id)
                    && tasks.insert(task.session_id.as_str(), task).is_none(),
                "invalid or duplicate durable task identity"
            );
            ensure!(
                task.recipe
                    .as_ref()
                    .is_none_or(|recipe| recipe.extensions.is_none()
                        && recipe.sub_recipes.is_none()
                        && recipe.parameters.is_none()),
                "durable recipes cannot contain extension authority"
            );
        }
        let root = tasks
            .get(self.root_session_id.as_str())
            .context("durable root task missing")?;
        ensure!(
            root.parent_session_id.is_none(),
            "durable root has a parent"
        );
        ensure!(
            !(root.completed && self.pending_wait.is_some()),
            "completed durable execution cannot be suspended"
        );
        ensure!(
            root.completed == self.completion.is_some(),
            "durable completion marker mismatch"
        );
        if let Some(marker) = &self.completion {
            ensure!(marker.version == 1, "unsupported durable completion marker");
            ensure!(
                self.native_run_id.as_ref().is_some_and(|id| !id.is_empty()),
                "completed durable execution has no producing run"
            );
            ensure!(
                self.effects.iter().all(|effect| effect.result.is_some()),
                "completed durable execution has uncertain effects"
            );
            let message = root
                .conversation
                .messages()
                .iter()
                .rev()
                .find(|message| {
                    message.role == rmcp::model::Role::Assistant && message.is_user_visible()
                })
                .filter(|message| eligible_completion(message))
                .context("completed durable execution has no visible answer")?;
            ensure!(
                message.id.as_deref() == Some(marker.message_id.as_str())
                    && !marker.message_id.is_empty(),
                "durable completion message mismatch"
            );
            ensure!(
                root.conversation
                    .messages()
                    .iter()
                    .filter(|message| message.id.as_deref() == Some(marker.message_id.as_str()))
                    .count()
                    == 1,
                "duplicate durable completion message"
            );
        }
        for task in &self.tasks {
            let mut visited = HashSet::new();
            let mut current = task;
            while current.session_id != self.root_session_id {
                ensure!(
                    visited.insert(current.session_id.as_str()),
                    "durable task tree contains a cycle"
                );
                current = tasks
                    .get(
                        current
                            .parent_session_id
                            .as_deref()
                            .context("disconnected durable task")?,
                    )
                    .context("durable task parent missing")?;
            }
        }
        let mut calls = HashSet::new();
        for effect in &self.effects {
            ensure!(
                tasks.contains_key(effect.task_id.as_str())
                    && calls.insert((effect.task_id.as_str(), effect.tool_call_id.as_str())),
                "invalid or duplicate durable effect identity"
            );
            ensure!(
                effect.operation_id
                    == format!(
                        "{}:{}:{}",
                        self.logical_run_id, effect.task_id, effect.tool_call_id
                    ),
                "durable operation identity mismatch"
            );
        }
        if let Some(pending) = &self.pending_wait {
            ensure!(
                pending.wait.version == 1
                    && !pending.wait.wait_id.is_empty()
                    && calls.contains(&(pending.task_id.as_str(), pending.tool_call_id.as_str())),
                "invalid suspended effect"
            );
            ensure!(
                pending
                    .wait
                    .dependent_task_ids
                    .iter()
                    .all(|id| tasks.contains_key(id.as_str())),
                "durable wait dependency outside the logical run"
            );
        }
        let mut mailbox_ids = HashSet::new();
        for message in &self.mailbox {
            ensure!(
                message.id > 0
                    && mailbox_ids.insert(message.id)
                    && tasks.contains_key(message.sender_session_id.as_str())
                    && tasks.contains_key(message.recipient_session_id.as_str()),
                "invalid durable mailbox identity"
            );
        }
        Ok(())
    }

    pub fn authoritative_completion(&self) -> Result<Option<DurableCompletion>> {
        self.validate()?;
        let Some(marker) = &self.completion else {
            return Ok(None);
        };
        let message = self
            .tasks
            .iter()
            .find(|task| task.session_id == self.root_session_id)
            .and_then(|task| {
                task.conversation
                    .messages()
                    .iter()
                    .find(|message| message.id.as_deref() == Some(marker.message_id.as_str()))
            })
            .context("durable completion message missing")?;
        Ok(Some(DurableCompletion {
            version: 1,
            logical_run_id: self.logical_run_id.clone(),
            request_digest: self.request_digest.clone(),
            root_session_id: self.root_session_id.clone(),
            native_run_id: self
                .native_run_id
                .clone()
                .context("durable producing run missing")?,
            producing_attempt_id: self
                .producing_attempt_id
                .clone()
                .context("durable producing attempt missing")?,
            generation: self.generation,
            consumed_input_revision: self.consumed_input_revision,
            message_id: marker.message_id.clone(),
            text: message.as_concat_text(),
        }))
    }
}

struct Execution {
    admission: DurableAdmission,
    root: String,
    generation: u64,
    consumed_input_revision: u64,
    pending_wait: Option<SuspendedCall>,
    effects: Vec<EffectReceipt>,
    manager: Arc<SessionManager>,
    resume_sessions: HashSet<String>,
    fault: Option<String>,
    native_run_id: Option<String>,
    producing_attempt_id: Option<String>,
    applied_context_ids: HashSet<String>,
    completed: bool,
    completion: Option<DurableCompletion>,
    parked: bool,
    recorded_response_ids: HashSet<(String, String)>,
}

type SharedExecution = Arc<AsyncMutex<Execution>>;
fn executions() -> &'static Mutex<HashMap<String, SharedExecution>> {
    static EXECUTIONS: OnceLock<Mutex<HashMap<String, SharedExecution>>> = OnceLock::new();
    EXECUTIONS.get_or_init(Default::default)
}
fn lookup(session_id: &str) -> Option<SharedExecution> {
    executions().lock().unwrap().get(session_id).cloned()
}

pub fn release(session_id: &str) {
    if let Some(runtime) = lookup(session_id) {
        executions()
            .lock()
            .unwrap()
            .retain(|_, candidate| !Arc::ptr_eq(candidate, &runtime));
    }
}

pub async fn bind_execution(
    manager: &SessionManager,
    root: &str,
    run_id: &str,
    attempt_key: Option<&str>,
) -> Result<()> {
    if !enabled(root) {
        return Ok(());
    }
    if let Some(runtime) = lookup(root) {
        let mut execution = runtime.lock().await;
        ensure!(
            !execution.completed,
            "completed durable execution cannot be rebound"
        );
        execution.native_run_id = Some(run_id.to_string());
        execution.producing_attempt_id = Some(execution.admission.attempt_id.clone());
    }
    let binding = serde_json::json!({"parentRunId":run_id,"attemptKey":attempt_key});
    sqlx::query(r#"UPDATE sessions SET extension_data=json_set(extension_data, '$."summon.task_execution_v1"', json(?)) WHERE parent_session_id=? AND json_extract(extension_data, '$."summon.task_admission_v1"') IS NOT NULL"#)
        .bind(binding.to_string()).bind(root).execute(manager.storage().pool().await?).await?;
    Ok(())
}

pub fn inherit(parent: &str, child: &str) {
    if let Some(runtime) = lookup(parent) {
        executions()
            .lock()
            .unwrap()
            .insert(child.to_string(), runtime);
    }
}

pub fn enabled(session_id: &str) -> bool {
    lookup(session_id).is_some()
}

pub async fn suspended(session_id: &str) -> bool {
    let Some(runtime) = lookup(session_id) else {
        return false;
    };
    let runtime = runtime.lock().await;
    runtime.pending_wait.as_ref().is_some_and(|call| {
        runtime.root == session_id
            || call.task_id == session_id
            || call
                .wait
                .dependent_task_ids
                .iter()
                .any(|id| id == session_id)
    })
}

impl Execution {
    async fn snapshot(&self) -> Result<DurableSnapshot> {
        let pool = self.manager.storage().pool().await?;
        let ids: Vec<String> = sqlx::query_scalar("WITH RECURSIVE tree(id) AS (SELECT id FROM sessions WHERE id = ? UNION ALL SELECT s.id FROM sessions s JOIN tree t ON s.parent_session_id = t.id) SELECT id FROM tree")
            .bind(&self.root).fetch_all(pool).await?;
        let mut tasks = Vec::new();
        let mut mailbox = Vec::new();
        let root_dir = self
            .manager
            .get_session(&self.root, false)
            .await?
            .working_dir;
        for id in ids {
            let session = self.manager.get_session(&id, true).await?;
            let mut recipe = session.recipe.clone();
            if let Some(recipe) = &mut recipe {
                recipe.extensions = None;
                recipe.sub_recipes = None;
                recipe.parameters = None;
            }
            let completed: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM session_mailbox WHERE sender_session_id=? AND kind='completion')").bind(&id).fetch_one(pool).await?;
            let extension_names =
                super::EnabledExtensionsState::from_extension_data(&session.extension_data)
                    .map(|state| {
                        state
                            .extensions
                            .into_iter()
                            .map(|extension| extension.name())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        session
                            .extension_data
                            .extension_states
                            .get("summon.durable_extensions_v1")
                            .and_then(|names| serde_json::from_value(names.clone()).ok())
                            .unwrap_or_default()
                    });
            let mut conversation = session.conversation.clone().unwrap_or_default();
            {
                let mut receipts = response_receipts(&conversation, &id)
                    .map_err(|error| DurableError::SnapshotInvalid(error.to_string()))?;
                if let Some(mut receipt) = receipts.pop() {
                    receipt.usage = DurableUsageReceipt::from_session(&session)?;
                    if let Some(holder) = conversation
                        .messages_mut()
                        .iter_mut()
                        .find(|message| message.id.as_ref() == receipt.message_ids.last())
                    {
                        holder.metadata.set_operation_note(
                            RESPONSE_OPERATION,
                            RESPONSE_NOTE,
                            serde_json::to_value(receipt)?,
                        );
                    }
                }
            }
            tasks.push(DurableTask {
                session_id: id.clone(),
                parent_session_id: session.parent_session_id,
                name: session.name,
                working_dir: session
                    .working_dir
                    .strip_prefix(&root_dir)
                    .context("durable task working directory is outside root")?
                    .to_string_lossy()
                    .to_string(),
                session_type: session.session_type,
                conversation,
                task_admission: session
                    .extension_data
                    .extension_states
                    .get("summon.task_admission_v1")
                    .cloned(),
                task_execution: session
                    .extension_data
                    .extension_states
                    .get("summon.task_execution_v1")
                    .cloned(),
                task_policy: session
                    .extension_data
                    .extension_states
                    .get("summon.v1")
                    .cloned(),
                recipe,
                extension_names,
                completed,
            });
            for row in sqlx::query("SELECT id, sender_session_id, recipient_session_id, kind, body, outcome_json, dedupe_key, steering_attempt_key, steering_digest, delivered_at IS NOT NULL AS delivered FROM session_mailbox WHERE recipient_session_id = ? ORDER BY id")
                .bind(&id).fetch_all(pool).await? {
                mailbox.push(DurableMailbox { id: row.try_get("id")?, sender_session_id: row.try_get("sender_session_id")?, recipient_session_id: row.try_get("recipient_session_id")?, kind: row.try_get("kind")?, body: row.try_get("body")?, outcome_json: row.try_get("outcome_json")?, dedupe_key: row.try_get("dedupe_key")?, steering_attempt_key: row.try_get("steering_attempt_key")?, steering_digest: row.try_get("steering_digest")?, delivered: row.try_get("delivered")? });
            }
        }
        Ok(DurableSnapshot {
            schema_version: 2,
            logical_run_id: self.admission.logical_run_id.clone(),
            root_session_id: self.root.clone(),
            request_digest: self.admission.request_digest.clone(),
            generation: self.generation + 1,
            consumed_input_revision: self.consumed_input_revision,
            native_run_id: self.native_run_id.clone(),
            producing_attempt_id: self.producing_attempt_id.clone(),
            tasks,
            mailbox,
            effects: self.effects.clone(),
            pending_wait: self.pending_wait.clone(),
            completion: self.completion.as_ref().map(|completion| CompletionMarker {
                version: 1,
                message_id: completion.message_id.clone(),
            }),
            applied_context_ids: {
                let mut ids = self.applied_context_ids.iter().cloned().collect::<Vec<_>>();
                ids.sort();
                ids
            },
        })
    }

    async fn commit(&mut self, boundary: &str) -> Result<()> {
        ensure!(
            !self.parked && !self.completed,
            "durable execution is sealed"
        );
        if let Some(fault) = &self.fault {
            anyhow::bail!("durable execution blocked: {fault}");
        }
        let result = self.commit_snapshot(boundary).await;
        if let Err(error) = &result {
            self.fault = Some(error.to_string());
        }
        result
    }

    async fn commit_snapshot(&mut self, boundary: &str) -> Result<()> {
        let mut snapshot = self.snapshot().await?;
        if boundary == "completed" {
            let marker = (|| -> Result<()> {
                let root = snapshot
                    .tasks
                    .iter_mut()
                    .find(|task| task.session_id == self.root)
                    .context("durable root missing")?;
                let message = root
                    .conversation
                    .messages()
                    .iter()
                    .rev()
                    .find(|message| {
                        message.role == rmcp::model::Role::Assistant && message.is_user_visible()
                    })
                    .filter(|message| eligible_completion(message))
                    .context("durable completion requires a persisted visible answer")?;
                if root
                    .conversation
                    .messages()
                    .iter()
                    .any(|message| message.metadata.inference.is_some())
                {
                    let receipt = response_receipts(&root.conversation, &self.root)?
                        .pop()
                        .context("completion has no processed inference boundary")?;
                    ensure!(
                        receipt.processed && receipt.output_message_id == message.id,
                        "completion response processing is unfinished"
                    );
                }
                snapshot.completion = Some(CompletionMarker {
                    version: 1,
                    message_id: message
                        .id
                        .clone()
                        .context("durable completion message has no identity")?,
                });
                root.completed = true;
                Ok(())
            })();
            marker.map_err(|error| DurableError::CompletionInvalid(error.to_string()))?;
        }
        let completion = snapshot.authoritative_completion()?;
        let generation = snapshot.generation;
        let body = serde_json::json!({"schemaVersion":1,"logicalRunId":self.admission.logical_run_id,"attemptId":self.admission.attempt_id,"epoch":self.admission.epoch,"expectedGeneration":self.generation,"generation":generation,"consumedInputRevision":self.consumed_input_revision,"boundary":boundary,"snapshot":snapshot});
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let response = client
            .post(&self.admission.callback_url)
            .bearer_auth(&self.admission.callback_token)
            .json(&body)
            .send()
            .await
            .context("durable checkpoint commit failed")?;
        ensure!(
            response.status().is_success(),
            "durable checkpoint rejected: {}",
            response.status()
        );
        let receipt: serde_json::Value = response.json().await?;
        ensure!(
            receipt.get("accepted").and_then(|v| v.as_bool()) == Some(true)
                && receipt.get("generation").and_then(|v| v.as_u64()) == Some(generation),
            "durable checkpoint acknowledgement mismatch"
        );
        self.generation = generation;
        self.completed |= boundary == "completed";
        self.completion = completion;
        self.parked |= boundary == "parked";
        Ok(())
    }
}

pub async fn checkpoint(session_id: &str, boundary: &str) -> Result<()> {
    if let Some(runtime) = lookup(session_id) {
        runtime.lock().await.commit(boundary).await?;
    }
    Ok(())
}

pub(super) struct TaskNoticeCheckpoint(OwnedMutexGuard<Execution>);

impl TaskNoticeCheckpoint {
    pub(super) async fn commit(&mut self) -> Result<()> {
        self.0.commit("task_notice").await
    }
}

pub(super) async fn task_notice_checkpoint(
    session_id: &str,
) -> Result<Option<TaskNoticeCheckpoint>> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(None);
    };
    let execution = runtime.lock_owned().await;
    if execution.parked || execution.completed {
        return Ok(None);
    }
    if let Some(fault) = &execution.fault {
        anyhow::bail!("durable execution blocked: {fault}");
    }
    Ok(Some(TaskNoticeCheckpoint(execution)))
}

pub async fn checkpoint_notice(session_id: &str) -> Result<()> {
    let mut checkpoint = task_notice_checkpoint(session_id)
        .await?
        .context("durable task notice target is parked or interrupted")?;
    checkpoint.commit().await
}

pub async fn admit(
    manager: Arc<SessionManager>,
    session_id: &str,
    admission: DurableAdmission,
) -> Result<Option<Message>> {
    ensure!(
        !admission.logical_run_id.is_empty()
            && !admission.attempt_id.is_empty()
            && admission.epoch > 0
            && !admission.callback_token.is_empty(),
        "missing durable authority"
    );
    let url = reqwest::Url::parse(&admission.callback_url)?;
    ensure!(
        matches!(url.scheme(), "https" | "http")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "invalid durable callback URL"
    );
    ensure!(
        admission.request_digest.len() == 64
            && admission
                .request_digest
                .bytes()
                .all(|c| c.is_ascii_hexdigit()),
        "invalid logical request digest"
    );
    let mut consumed = 0;
    let mut effects = Vec::new();
    let mut pending_wait = None;
    if let Some(context) = &admission.continuation_context {
        ensure!(
            admission.restore.is_some() && !context.id.is_empty() && !context.text.is_empty(),
            "continuation context requires a restored execution and identity"
        );
    }
    if let Some(snapshot) = &admission.restore {
        ensure!(
            snapshot.logical_run_id == admission.logical_run_id
                && snapshot.request_digest == admission.request_digest
                && snapshot.generation == admission.expected_generation
                && snapshot.root_session_id == session_id,
            DurableError::SnapshotInvalid("durable restore identity or generation mismatch".into())
        );
        snapshot.validate()?;
        consumed = snapshot.consumed_input_revision;
        effects = snapshot.effects.clone();
        pending_wait = snapshot.pending_wait.clone();
    } else {
        ensure!(
            admission.expected_generation == 0 && admission.accepted_reply.is_none(),
            "continuation requires a snapshot"
        );
    }
    let accepted_message = if let Some(reply) = &admission.accepted_reply {
        let pending = pending_wait
            .as_ref()
            .context("accepted reply has no suspended call")?;
        ensure!(
            pending.wait.wait_id == reply.wait_id
                && pending.tool_call_id == reply.tool_call_id
                && reply.input_revision > consumed,
            "accepted reply does not match suspension"
        );

        ensure!(
            reply
                .result
                .meta
                .as_ref()
                .is_none_or(|meta| !meta.0.contains_key(WAIT_META)),
            "accepted reply cannot suspend"
        );
        let effect = effects
            .iter_mut()
            .find(|effect| {
                effect.task_id == pending.task_id && effect.tool_call_id == reply.tool_call_id
            })
            .context("suspended effect missing")?;
        effect.result = Some(Ok(reply.result.clone()));
        consumed = reply.input_revision;
        let reply_task_id = pending.task_id.clone();
        pending_wait = None;
        let message = Message::user()
            .with_tool_response(reply.tool_call_id.clone(), Ok(reply.result.clone()))
            .with_generated_id();
        Some((reply_task_id, message))
    } else {
        None
    };
    for effect in &effects {
        if effect.result.is_none() {
            return Err(DurableError::EffectUncertain(effect.operation_id.clone()).into());
        }
    }
    let mut recorded_response_ids = HashSet::new();
    if let Some(snapshot) = &admission.restore {
        if !snapshot
            .tasks
            .iter()
            .any(|task| task.session_id == session_id && task.completed)
        {
            for task in &snapshot.tasks {
                if task.completed {
                    continue;
                }
                let receipts = response_receipts(&task.conversation, &task.session_id)
                    .map_err(|error| DurableError::SnapshotInvalid(error.to_string()))?;
                if let Some(receipt) = receipts.last().filter(|receipt| !receipt.processed) {
                    if let Some(tail) = task
                        .conversation
                        .messages()
                        .iter()
                        .rev()
                        .find(|message| !message.content.is_empty())
                    {
                        ensure!(
                            tail.role != rmcp::model::Role::Assistant
                                || tail
                                    .id
                                    .as_ref()
                                    .is_some_and(|id| receipt.message_ids.contains(id)),
                            DurableError::SnapshotInvalid(
                                "unrecorded derived response processing phase".into()
                            )
                        );
                    }
                }
                let covered: HashSet<&str> = receipts
                    .iter()
                    .flat_map(|receipt| {
                        receipt
                            .message_ids
                            .iter()
                            .chain(receipt.output_message_id.iter())
                    })
                    .map(String::as_str)
                    .collect();
                ensure!(
                    task.conversation
                        .messages()
                        .iter()
                        .filter(|message| message.metadata.inference.is_some())
                        .all(|message| message
                            .id
                            .as_deref()
                            .is_some_and(|id| covered.contains(id))),
                    DurableError::SnapshotInvalid(
                        "unfinished inference has no durable response boundary".into()
                    )
                );
                recorded_response_ids.extend(
                    receipts
                        .into_iter()
                        .flat_map(|receipt| receipt.message_ids)
                        .map(|id| (task.session_id.clone(), id)),
                );
            }
        }
    }
    ensure!(!enabled(session_id), "durable session already admitted");
    if let Some(snapshot) = &admission.restore {
        manager.import_durable_snapshot(snapshot).await?;
    }
    sqlx::query(r#"UPDATE sessions SET extension_data=json_set(extension_data, '$."durable.execution_v1"', json(?)) WHERE id=?"#)
        .bind(serde_json::json!({"logicalRunId":admission.logical_run_id}).to_string()).bind(session_id).execute(manager.storage().pool().await?).await?;
    let accepted_message = if let Some((task, message)) = accepted_message {
        if task == session_id {
            Some(message)
        } else {
            manager.storage().add_message(&task, &message).await?;
            None
        }
    } else {
        None
    };
    let mut resume_sessions: HashSet<String> = admission
        .restore
        .as_ref()
        .map(|snapshot| {
            snapshot
                .tasks
                .iter()
                .filter(|task| !task.completed)
                .map(|task| task.session_id.clone())
                .collect()
        })
        .unwrap_or_default();
    if accepted_message.is_some() {
        resume_sessions.remove(session_id);
    }
    let applied_context_ids = admission
        .restore
        .as_ref()
        .map(|snapshot| snapshot.applied_context_ids.iter().cloned().collect())
        .unwrap_or_default();
    let completed = admission.restore.as_ref().is_some_and(|snapshot| {
        snapshot
            .tasks
            .iter()
            .any(|task| task.session_id == session_id && task.completed)
    });
    let completion = admission
        .restore
        .as_ref()
        .map(DurableSnapshot::authoritative_completion)
        .transpose()?
        .flatten();
    let native_run_id = admission
        .restore
        .as_ref()
        .and_then(|snapshot| snapshot.native_run_id.clone());
    let producing_attempt_id = admission
        .restore
        .as_ref()
        .and_then(|snapshot| snapshot.producing_attempt_id.clone());
    let runtime = Arc::new(AsyncMutex::new(Execution {
        generation: admission.expected_generation,
        admission,
        root: session_id.to_string(),
        consumed_input_revision: consumed,
        pending_wait,
        effects,
        manager,
        resume_sessions,
        fault: None,
        native_run_id,
        producing_attempt_id,
        applied_context_ids,
        completed,
        completion,
        parked: false,
        recorded_response_ids,
    }));
    let mut registry = executions().lock().unwrap();
    if let Some(snapshot) = &runtime.try_lock().unwrap().admission.restore {
        for task in &snapshot.tasks {
            registry.insert(task.session_id.clone(), runtime.clone());
        }
    }
    registry.insert(session_id.to_string(), runtime);
    Ok(accepted_message)
}

pub async fn before_tool(
    session_id: &str,
    call_id: &str,
    request: &mut CallToolRequestParams,
) -> Result<Option<std::result::Result<CallToolResult, ErrorData>>> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(None);
    };
    let mut execution = runtime.lock().await;
    ensure!(
        !execution.parked && !execution.completed,
        "durable execution is sealed"
    );
    ensure!(
        execution.fault.is_none(),
        "durable execution blocked: {}",
        execution.fault.as_deref().unwrap_or_default()
    );
    if let Some(effect) = execution
        .effects
        .iter()
        .find(|effect| effect.task_id == session_id && effect.tool_call_id == call_id)
    {
        ensure!(
            serde_json::to_value(&effect.request.arguments)?
                == serde_json::to_value(&request.arguments)?
                && effect.request.name == request.name,
            "durable call identity reused with different arguments"
        );
        return effect
            .result
            .clone()
            .map(Some)
            .ok_or_else(|| DurableError::EffectUncertain(effect.operation_id.clone()).into());
    }
    ensure!(
        !execution
            .pending_wait
            .as_ref()
            .is_some_and(|pending| execution.root == session_id
                || pending.task_id == session_id
                || pending
                    .wait
                    .dependent_task_ids
                    .iter()
                    .any(|id| id == session_id)),
        "task is suspended"
    );
    let operation_id = format!(
        "{}:{}:{}",
        execution.admission.logical_run_id, session_id, call_id
    );
    let meta = request.meta.get_or_insert_with(Default::default);
    meta.0.insert(OPERATION_META.to_string(), serde_json::json!({"logicalRunId":execution.admission.logical_run_id,"taskId":session_id,"toolCallId":call_id,"operationId":operation_id}));
    execution.effects.push(EffectReceipt {
        task_id: session_id.to_string(),
        tool_call_id: call_id.to_string(),
        operation_id,
        request: request.clone(),
        result: None,
    });
    execution.commit("tool_intent").await?;
    Ok(None)
}

pub async fn after_tool(
    session_id: &str,
    call_id: &str,
    result: &std::result::Result<CallToolResult, ErrorData>,
) -> Result<()> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(());
    };
    let mut execution = runtime.lock().await;
    let recorded: Result<()> = async {
        if let Ok(result) = result {
            if let Some(revision) = result
                .meta
                .as_ref()
                .and_then(|meta| meta.0.get("uthereal/inputRevision"))
                .and_then(|revision| revision.as_u64())
            {
                execution.consumed_input_revision = execution.consumed_input_revision.max(revision);
            }
            if let Some(wait) = result.meta.as_ref().and_then(|meta| meta.0.get(WAIT_META)) {
                let wait: DurableWait = serde_json::from_value(wait.clone())?;
                ensure!(
                    wait.version == 1 && !wait.wait_id.is_empty(),
                    "invalid durable suspension"
                );
                for task in &wait.dependent_task_ids {
                    ensure!(
                        lookup(task)
                            .as_ref()
                            .is_some_and(|candidate| Arc::ptr_eq(candidate, &runtime)),
                        "durable wait dependency is outside this logical run"
                    );
                }
                ensure!(
                    execution.pending_wait.is_none(),
                    "only one durable wait may be admitted at a time"
                );
                execution.pending_wait = Some(SuspendedCall {
                    task_id: session_id.to_string(),
                    tool_call_id: call_id.to_string(),
                    wait,
                });
            }
        }
        let effect = execution
            .effects
            .iter_mut()
            .find(|effect| effect.task_id == session_id && effect.tool_call_id == call_id)
            .context("durable tool result has no intent")?;
        if result.is_err() {
            execution.commit("effect_uncertain").await?;
            execution.fault = Some(format!(
                "uncertain external effect {call_id}; reconciliation required"
            ));
            return Err(DurableError::EffectUncertain(call_id.to_string()).into());
        }
        effect.result = Some(result.clone());
        execution.commit("tool_result").await
    }
    .await;
    if let Err(error) = &recorded {
        execution.fault = Some(error.to_string());
    }
    recorded
}

pub fn is_wait(result: &std::result::Result<CallToolResult, ErrorData>) -> bool {
    result.as_ref().is_ok_and(|result| {
        result
            .meta
            .as_ref()
            .is_some_and(|meta| meta.0.contains_key(WAIT_META))
    })
}

pub async fn recorded_response_message(session_id: &str, message_id: &str) -> bool {
    let Some(runtime) = lookup(session_id) else {
        return false;
    };
    let execution = runtime.lock().await;
    execution
        .recorded_response_ids
        .contains(&(session_id.to_string(), message_id.to_string()))
}

pub async fn record_response(session_id: &str, messages: &mut [Message]) -> Result<()> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(());
    };
    if messages.is_empty() {
        return Ok(());
    }
    let mut execution = runtime.lock().await;
    let ids = messages
        .iter()
        .map(|message| {
            message
                .id
                .clone()
                .context("inference response message has no identity")
        })
        .collect::<Result<Vec<_>>>()?;
    let receipt = InferenceReceipt {
        version: 1,
        task_id: session_id.into(),
        native_run_id: execution
            .native_run_id
            .clone()
            .context("inference has no native run")?,
        producing_attempt_id: execution
            .producing_attempt_id
            .clone()
            .context("inference has no attempt")?,
        message_ids: ids.clone(),
        processed: false,
        usage: super::durable_usage::capture(execution.manager.as_ref(), session_id).await?,
        output_message_id: None,
    };
    messages.last_mut().unwrap().metadata.set_operation_note(
        RESPONSE_OPERATION,
        RESPONSE_NOTE,
        serde_json::to_value(receipt)?,
    );
    if let Err(error) = execution
        .manager
        .storage()
        .add_messages(session_id, messages)
        .await
    {
        execution.fault = Some("inference response persistence failed".into());
        return Err(error);
    }
    execution
        .recorded_response_ids
        .extend(ids.into_iter().map(|id| (session_id.to_string(), id)));
    execution.commit("inference").await
}

pub(crate) fn pending_response_receipt(
    conversation: &Conversation,
    session_id: &str,
) -> Result<Option<InferenceReceipt>> {
    let Some(receipt) = response_receipts(conversation, session_id)?.pop() else {
        return Ok(None);
    };
    let tail = conversation
        .messages()
        .iter()
        .rev()
        .find(|message| !message.content.is_empty());
    let Some(tail_id) = tail.and_then(|message| message.id.as_ref()) else {
        return Ok(None);
    };
    if receipt.processed {
        if receipt.output_message_id.as_ref() != Some(tail_id) {
            return Ok(None);
        }
    } else if !receipt.message_ids.contains(tail_id) {
        return Ok(None);
    }
    Ok(Some(receipt))
}

pub async fn pending_response(
    session_id: &str,
) -> Result<Option<(InferenceReceipt, Vec<Message>)>> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(None);
    };
    let execution = runtime.lock().await;
    let conversation = execution
        .manager
        .get_session(session_id, true)
        .await?
        .conversation
        .unwrap_or_default();
    let Some(receipt) = pending_response_receipt(&conversation, session_id)? else {
        return Ok(None);
    };
    let end = conversation
        .messages()
        .iter()
        .position(|message| message.id.as_ref() == receipt.message_ids.last())
        .unwrap();
    let start = end + 1 - receipt.message_ids.len();
    let messages = conversation.messages()[start..=end].to_vec();
    Ok(Some((receipt, messages)))
}

pub async fn finish_response(session_id: &str, output_message_id: Option<&str>) -> Result<()> {
    let Some(output_message_id) = output_message_id else {
        return Ok(());
    };
    let Some(runtime) = lookup(session_id) else {
        return Ok(());
    };
    let mut execution = runtime.lock().await;
    if execution.pending_wait.is_some()
        || execution
            .effects
            .iter()
            .any(|effect| effect.result.is_none())
    {
        return Ok(());
    }
    let session = execution.manager.get_session(session_id, true).await?;
    let usage = DurableUsageReceipt::from_session(&session)?;
    let conversation = session.conversation.unwrap_or_default();
    let Some(last) = conversation.messages().last().filter(|message| {
        eligible_completion(message) && message.id.as_deref() == Some(output_message_id)
    }) else {
        return Ok(());
    };
    let Some(mut receipt) = response_receipts(&conversation, session_id)?.pop() else {
        return Ok(());
    };
    if receipt.processed {
        return Ok(());
    }
    receipt.usage = usage;
    receipt.processed = true;
    receipt.output_message_id = last.id.clone();
    let holder = receipt.message_ids.last().unwrap();
    let mut metadata = conversation
        .messages()
        .iter()
        .find(|message| message.id.as_ref() == Some(holder))
        .unwrap()
        .metadata
        .clone();
    metadata.set_operation_note(
        RESPONSE_OPERATION,
        RESPONSE_NOTE,
        serde_json::to_value(&receipt)?,
    );
    sqlx::query("UPDATE messages SET metadata_json=? WHERE session_id=? AND message_id=?")
        .bind(serde_json::to_string(&metadata)?)
        .bind(session_id)
        .bind(holder)
        .execute(execution.manager.storage().pool().await?)
        .await?;
    execution.commit("response_processed").await
}

pub async fn resuming(session_id: &str) -> bool {
    let Some(runtime) = lookup(session_id) else {
        return false;
    };
    let execution = runtime.lock().await;
    execution.resume_sessions.contains(session_id)
}

pub async fn take_resume(session_id: &str) -> bool {
    let Some(runtime) = lookup(session_id) else {
        return false;
    };
    let resumed = runtime.lock().await.resume_sessions.remove(session_id);
    resumed
}

pub async fn completed(session_id: &str) -> bool {
    let Some(runtime) = lookup(session_id) else {
        return false;
    };
    let execution = runtime.lock().await;
    execution.root == session_id && execution.completed
}

pub async fn completion(session_id: &str) -> Option<DurableCompletion> {
    let runtime = lookup(session_id)?;
    let execution = runtime.lock().await;
    if execution.root != session_id {
        return None;
    }
    execution.completion.clone()
}

pub async fn uncertain_operation(session_id: &str) -> Option<String> {
    let runtime = lookup(session_id)?;
    let execution = runtime.lock().await;
    execution
        .effects
        .iter()
        .find(|effect| effect.result.is_none())
        .map(|effect| effect.operation_id.clone())
}

pub async fn recovered_tasks(session_id: &str) -> Vec<DurableTask> {
    let Some(runtime) = lookup(session_id) else {
        return Vec::new();
    };
    let runtime = runtime.lock().await;
    runtime
        .admission
        .restore
        .as_ref()
        .map(|snapshot| {
            snapshot
                .tasks
                .iter()
                .filter(|task| {
                    task.parent_session_id.as_deref() == Some(session_id)
                        && !task.completed
                        && runtime.resume_sessions.contains(&task.session_id)
                        && !runtime.pending_wait.as_ref().is_some_and(|pending| {
                            pending.task_id == task.session_id
                                || pending.wait.dependent_task_ids.contains(&task.session_id)
                        })
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

pub async fn continuation_message(manager: &SessionManager, session_id: &str) -> Result<Message> {
    manager
        .get_session(session_id, true)
        .await?
        .conversation
        .and_then(|conversation| conversation.messages().last().cloned())
        .context("durable continuation has no conversation")
}

pub async fn apply_continuation_context(session_id: &str) -> Result<Option<Message>> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(None);
    };
    let mut execution = runtime.lock().await;
    if execution.root != session_id {
        return Ok(None);
    }
    let Some(context) = execution.admission.continuation_context.clone() else {
        return Ok(None);
    };
    if execution.applied_context_ids.contains(&context.id) {
        return Ok(None);
    }
    let session = execution.manager.get_session(session_id, true).await?;
    if let Some(conversation) = session.conversation.as_ref() {
        if pending_response_receipt(conversation, session_id)?.is_some()
            || pending_tool_message(conversation).is_some()
        {
            return Ok(None);
        }
    }
    let message = Message::user()
        .with_text(context.text)
        .with_visibility(false, true)
        .with_generated_id();
    execution
        .manager
        .storage()
        .add_message(session_id, &message)
        .await?;
    execution.applied_context_ids.insert(context.id);
    execution.commit("continuation_context").await?;
    Ok(Some(message))
}

pub fn pending_tool_message(conversation: &Conversation) -> Option<Message> {
    let answered: HashSet<&str> = conversation
        .messages()
        .iter()
        .flat_map(|message| message.get_tool_response_ids())
        .collect();
    let requests = conversation
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            crate::conversation::message::MessageContent::ToolRequest(request)
                if !answered.contains(request.id.as_str()) =>
            {
                Some(content.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if requests.is_empty() {
        None
    } else {
        let mut message = Message::assistant().with_generated_id();
        message.content = requests;
        Some(message)
    }
}

pub async fn parked_receipt(session_id: &str) -> Result<Option<serde_json::Value>> {
    let Some(runtime) = lookup(session_id) else {
        return Ok(None);
    };
    let mut execution = runtime.lock().await;
    if execution.pending_wait.is_none() {
        return Ok(None);
    }
    ensure!(
        execution
            .effects
            .iter()
            .all(|effect| effect.result.is_some()),
        "durable parking requires settled effects"
    );
    execution.commit("parked").await?;
    Ok(Some(
        serde_json::json!({"state":"parked","generation":execution.generation,"waitId":execution.pending_wait.as_ref().unwrap().wait.wait_id,"consumedInputRevision":execution.consumed_input_revision}),
    ))
}

impl SessionManager {
    pub async fn prepare_durable_session(
        &self,
        id: &str,
        cwd: &std::path::Path,
        name: &str,
        session_type: SessionType,
        mode: crate::config::GooseMode,
    ) -> Result<super::Session> {
        ensure!(valid_session_id(id), "invalid durable session id");
        let pool = self.storage().pool().await?;
        let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query("DROP INDEX IF EXISTS idx_prompt_attempt_session")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_prompt_attempt_session ON prompt_attempts(session_id) WHERE session_id IS NOT NULL").execute(&mut *tx).await?;
        sqlx::query("INSERT INTO sessions(id,name,session_type,working_dir,goose_mode) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET working_dir=excluded.working_dir, goose_mode=excluded.goose_mode")
            .bind(id).bind(name).bind(session_type.to_string()).bind(cwd.to_string_lossy().to_string()).bind(serde_json::to_value(mode)?.as_str().context("invalid goose mode")?).execute(&mut *tx).await?;
        tx.commit().await?;
        self.get_session(id, false).await
    }

    pub async fn import_durable_snapshot(&self, snapshot: &DurableSnapshot) -> Result<()> {
        snapshot.validate()?;
        let pool = self.storage().pool().await?;
        let root = self.get_session(&snapshot.root_session_id, false).await?;
        let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
        for task in &snapshot.tasks {
            if let Some(existing_parent) = sqlx::query_scalar::<_, Option<String>>(
                "SELECT parent_session_id FROM sessions WHERE id=?",
            )
            .bind(&task.session_id)
            .fetch_optional(&mut *tx)
            .await?
            {
                ensure!(
                    existing_parent == task.parent_session_id,
                    DurableError::SnapshotInvalid(
                        "durable task conflicts with an existing session".into()
                    )
                );
            }
        }
        for task in &snapshot.tasks {
            let mut extension_data = serde_json::Map::new();
            extension_data.insert(
                "summon.durable_extensions_v1".to_string(),
                serde_json::json!(task.extension_names),
            );
            if let Some(admission) = &task.task_admission {
                extension_data.insert("summon.task_admission_v1".to_string(), admission.clone());
            }
            if let Some(execution) = &task.task_execution {
                extension_data.insert("summon.task_execution_v1".to_string(), execution.clone());
            }
            if let Some(policy) = &task.task_policy {
                extension_data.insert("summon.v1".to_string(), policy.clone());
            }
            sqlx::query("INSERT INTO sessions(id,name,session_type,working_dir,parent_session_id,extension_data,goose_mode,recipe_json) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET working_dir=excluded.working_dir, goose_mode=excluded.goose_mode, parent_session_id=excluded.parent_session_id, recipe_json=excluded.recipe_json, extension_data=CASE WHEN excluded.parent_session_id IS NULL THEN sessions.extension_data ELSE excluded.extension_data END")
                .bind(&task.session_id).bind(&task.name).bind(task.session_type.to_string()).bind(root.working_dir.join(&task.working_dir).to_string_lossy().to_string()).bind(&task.parent_session_id).bind(serde_json::to_string(&extension_data)?).bind(serde_json::to_value(root.goose_mode)?.as_str().context("invalid goose mode")?).bind(task.recipe.as_ref().map(serde_json::to_string).transpose()?).execute(&mut *tx).await?;
            sqlx::query("DELETE FROM messages WHERE session_id = ?")
                .bind(&task.session_id)
                .execute(&mut *tx)
                .await?;
            for message in task.conversation.messages() {
                super::session_manager::SessionStorage::message_insert(
                    &task.session_id,
                    message,
                    message.id.clone(),
                    message.created,
                )?
                .execute(&mut *tx)
                .await?;
            }
            let receipts = response_receipts(&task.conversation, &task.session_id)
                .map_err(|error| DurableError::SnapshotInvalid(error.to_string()))?;
            if let Some(receipt) = receipts.last() {
                super::durable_usage::restore(&mut tx, &task.session_id, &receipt.usage).await?;
            }
            sqlx::query("DELETE FROM session_mailbox WHERE recipient_session_id=?")
                .bind(&task.session_id)
                .execute(&mut *tx)
                .await?;
        }
        let mut mailbox: Vec<_> = snapshot.mailbox.iter().collect();
        mailbox.sort_unstable_by_key(|message| message.id);
        // Mailbox IDs and acknowledgement cursors belong to the local database.
        // Allocate fresh IDs while preserving source delivery order and receipts.
        for message in mailbox {
            sqlx::query("INSERT INTO session_mailbox(sender_session_id,recipient_session_id,kind,body,outcome_json,dedupe_key,steering_attempt_key,steering_digest,delivered_at) VALUES(?,?,?,?,?,?,?,?,CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)")
                .bind(&message.sender_session_id).bind(&message.recipient_session_id).bind(&message.kind).bind(&message.body).bind(&message.outcome_json).bind(&message.dedupe_key).bind(&message.steering_attempt_key).bind(&message.steering_digest).bind(message.delivered).execute(&mut *tx).await?;
        }
        if let (Some(run_id), Some(attempt_id)) =
            (&snapshot.native_run_id, &snapshot.producing_attempt_id)
        {
            sqlx::query("INSERT INTO acp_runs(session_id,run_id,attempt_key) VALUES(?,?,?) ON CONFLICT(session_id,run_id) DO NOTHING")
                .bind(&snapshot.root_session_id).bind(run_id).bind(attempt_id).execute(&mut *tx).await?;
            let existing: Option<String> = sqlx::query_scalar(
                "SELECT attempt_key FROM acp_runs WHERE session_id=? AND run_id=?",
            )
            .bind(&snapshot.root_session_id)
            .bind(run_id)
            .fetch_one(&mut *tx)
            .await?;
            ensure!(
                existing.as_ref() == Some(attempt_id),
                DurableError::SnapshotInvalid(
                    "durable producing run conflicts with existing authority".into()
                )
            );
        }
        tx.commit().await?;
        // A legacy loop dispatches before appending its call to conversation. Recover
        // the exact provider call from the committed intent, never a rebuilt prompt.
        for effect in &snapshot.effects {
            let session = self.get_session(&effect.task_id, true).await?;
            let conversation = session.conversation.unwrap_or_default();
            if !conversation.messages().iter().any(|message| {
                message
                    .get_tool_request_ids()
                    .contains(&effect.tool_call_id.as_str())
            }) {
                self.storage()
                    .add_message(
                        &effect.task_id,
                        &Message::assistant()
                            .with_tool_request(
                                effect.tool_call_id.clone(),
                                Ok(effect.request.clone()),
                            )
                            .with_generated_id(),
                    )
                    .await?;
            }
            let suspended = snapshot.pending_wait.as_ref().is_some_and(|pending| {
                pending.task_id == effect.task_id && pending.tool_call_id == effect.tool_call_id
            });
            if !suspended
                && !conversation.messages().iter().any(|message| {
                    message
                        .get_tool_response_ids()
                        .contains(&effect.tool_call_id.as_str())
                })
            {
                if let Some(result) = &effect.result {
                    self.storage()
                        .add_message(
                            &effect.task_id,
                            &Message::user()
                                .with_tool_response(effect.tool_call_id.clone(), result.clone())
                                .with_generated_id(),
                        )
                        .await?;
                }
            }
        }
        Ok(())
    }
}
