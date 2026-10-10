use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::subagent_handler::{run_subagent_task, OnMessageCallback, SubagentRunParams};
use crate::agents::subagent_task_config::{TaskConfig, DEFAULT_SUBAGENT_MAX_TURNS};
use crate::agents::tool_execution::{ToolCallContext, ToolCallNotificationEmitter};
use crate::agents::AgentConfig;
use crate::config::paths::Paths;
use crate::config::{Config, GooseMode};
use crate::conversation::message::{Message, MessageContent};
use crate::providers;
use crate::recipe::build_recipe::build_recipe_from_template;
use crate::recipe::local_recipes::load_local_recipe_file;
use crate::recipe::{Recipe, RecipeParameter, Settings, RECIPE_FILE_EXTENSIONS};
use crate::session::extension_data::EnabledExtensionsState;
use crate::session::SessionType;
use crate::sources::parse_frontmatter;
use crate::utils::safe_truncate;
use anyhow::Result;
use async_trait::async_trait;
use futures::FutureExt;
use goose_agent::operation::messages_since_kickoff;
use goose_sdk_types::custom_requests::{
    ChannelWaitContext, SourceEntry, SourceType, TaskTerminalStatus,
};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, Implementation, InitializeResult,
    JsonObject, ListToolsResult, MetaObject, Role, ServerCapabilities, ServerNotification, Tool,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Tool-result metadata key: the agent ends its turn after this tool result,
/// without a user-visible message. Only `wait` sets it.
pub const END_TURN_META_KEY: &str = "goose.endTurn";

/// True when a successful tool result asks the agent to end its turn.
pub fn tool_result_ends_turn(result: &CallToolResult) -> bool {
    result.is_error != Some(true)
        && result
            .meta
            .as_ref()
            .and_then(|meta| meta.0.get(END_TURN_META_KEY))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
}
use tokio::sync::Mutex;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

pub static EXTENSION_NAME: &str = "summon";

const SUBAGENT_DESCRIPTION_BUDGET: usize = 160;

const TASK_LABEL_BUDGET: usize = 60;

/// A specialist's wait: default and longest sleep, and how often it checks its
/// mailbox.
const SPECIALIST_WAIT_DEFAULT_SECS: u64 = 120;
const SPECIALIST_WAIT_MAX_SECS: u64 = 180;
const SPECIALIST_WAIT_POLL: Duration = Duration::from_millis(250);

/// Whether a message to the parent asks something (message_parent is for questions only).
fn asks_question(message: &str) -> bool {
    message.contains(['?', '\u{FF1F}', '\u{061F}'])
}

/// How the parent continues a specialist that can no longer receive messages.
fn redelegate_hint(task_id: &str) -> String {
    format!("A message cannot reach task {task_id}. To have its work continue, delegate its artifact again with the same artifact_key and an instruction that includes what you meant to send.")
}

fn durable_assistant_turn_count(conversation: &crate::conversation::Conversation) -> u32 {
    let Ok(messages) = messages_since_kickoff(conversation) else {
        return 0;
    };
    let mut turns = 0;
    let mut in_assistant_block = false;
    for message in messages.iter().rev() {
        // Compaction summaries and continuation prompts are assistant-role
        // scaffolding, but are not user-visible task turns. Ignore them
        // without merging across a hidden user replay below.
        if message.role == Role::Assistant && !message.is_user_visible() {
            continue;
        }
        if message.role == Role::Assistant {
            if !in_assistant_block {
                turns += 1;
                in_assistant_block = true;
            }
        } else {
            in_assistant_block = false;
        }
    }
    turns
}

fn kind_plural(kind: SourceType) -> &'static str {
    match kind {
        SourceType::Subrecipe => "Subrecipes",
        SourceType::Recipe => "Recipes",
        SourceType::Agent => "Agents",
        _ => "Other",
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct DelegateParams {
    pub instructions: Option<String>,
    pub source: Option<String>,
    pub artifact_key: Option<String>,
    pub artifact_title: Option<String>,
    /// Set by delegate, never by the caller: the artifact's last task, which a follow-up continues.
    #[serde(skip)]
    pub previous_task_id: Option<String>,
    /// Set by delegate from the artifact tool: "new" while the artifact has no document yet.
    #[serde(skip)]
    pub artifact_status: Option<String>,
    /// Existing artifacts the task may read but not edit (artifact IDs).
    #[serde(default)]
    pub reference_artifacts: Vec<String>,
    /// Set by delegate from the artifact tool: the references, frozen at their saved revisions.
    #[serde(skip)]
    pub references: Vec<ArtifactReference>,
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    pub extensions: Option<Vec<String>>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_turns: Option<usize>,
    pub context: Option<String>,
    pub working_dir: Option<String>,
    #[serde(default)]
    pub r#async: bool,
}

#[derive(Debug, Deserialize)]
struct SendParams {
    task_id: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct MessageParentParams {
    message: String,
}

/// A read-only reference: an existing artifact a task may read, frozen at the saved
/// revision it had when the task was delegated.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ArtifactReference {
    pub artifact: String,
    pub revision_id: Option<String>,
    #[serde(default)]
    pub revision: Option<i64>,
    #[serde(default)]
    pub title: Option<String>,
}

/// The most read-only references one delegation may carry.
const MAX_REFERENCE_ARTIFACTS: usize = 8;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct SummonTaskPolicy {
    #[serde(default)]
    event_driven_parent: bool,
    artifact_key: Option<String>,
    previous_task_id: Option<String>,
    #[serde(default)]
    artifact_result_tools: Vec<String>,
    /// Read-only references; a task delegated later to edit one revokes it (see
    /// `SessionManager::reference_revoked`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    references: Vec<ArtifactReference>,
}

impl SummonTaskPolicy {
    fn from_session(session: &crate::session::Session) -> Option<Self> {
        session
            .extension_data
            .get_extension_state("summon", "v1")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }
}

pub struct BackgroundTask {
    pub id: String,
    pub parent_session_id: String,
    pub non_blocking: bool,
    pub completion_delivery_error: Arc<Mutex<Option<String>>>,
    terminal_status: Arc<Mutex<Option<TaskTerminalStatus>>>,
    pub description: String,
    pub started_at: Instant,
    pub turns: Arc<AtomicU32>,
    pub last_activity: Arc<AtomicU64>,
    pub handle: JoinHandle<Result<String>>,
    pub cancellation_token: CancellationToken,
    completion_token: CancellationToken,
    notification_sink: SharedNotificationSink,
}

fn spawn_background_task<F>(future: F) -> (JoinHandle<Result<String>>, CancellationToken)
where
    F: Future<Output = Result<String>> + Send + 'static,
{
    let completion_token = CancellationToken::new();
    let completion_guard = completion_token.clone().drop_guard();
    let handle = tokio::spawn(async move {
        let _completion_guard = completion_guard;
        future.await
    });
    (handle, completion_token)
}

async fn task_execution_outcome<F>(
    future: F,
    cancellation: &CancellationToken,
) -> (Result<String>, TaskTerminalStatus)
where
    F: Future<Output = Result<String>>,
{
    let (result, status) = match std::panic::AssertUnwindSafe(future).catch_unwind().await {
        Ok(result) => {
            let status = if result.is_ok() {
                TaskTerminalStatus::Completed
            } else {
                TaskTerminalStatus::Failed
            };
            (result, status)
        }
        Err(panic) => {
            let detail = panic
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            (
                Err(anyhow::anyhow!("Task panicked: {detail}")),
                TaskTerminalStatus::Panicked,
            )
        }
    };
    (
        result,
        if cancellation.is_cancelled() {
            TaskTerminalStatus::Cancelled
        } else {
            status
        },
    )
}

pub struct CompletedTask {
    pub id: String,
    pub parent_session_id: String,
    pub completion_delivery_error: Option<String>,
    terminal_status: TaskTerminalStatus,
    pub description: String,
    pub result: Result<String, String>,
    pub turns_taken: u32,
    pub duration: Duration,
    pub completed_at: Instant,
    notification_sink: SharedNotificationSink,
}

enum NotificationSink {
    Buffer(Vec<ServerNotification>),
    Emitter(ToolCallNotificationEmitter),
}

type SharedNotificationSink = Arc<Mutex<NotificationSink>>;

async fn yield_to_outer_tool_stream() {
    // The outer select may have polled its receiver before this future queues a
    // notification. Keep the result pending for the following select pass so
    // the now-ready receiver is observed before the terminal result.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
}

impl NotificationSink {
    fn route(&mut self, notification: ServerNotification) {
        match self {
            Self::Buffer(buffer) => buffer.push(notification),
            Self::Emitter(emitter) => emitter.emit_best_effort(notification),
        }
    }

    async fn attach(&mut self, emitter: Option<ToolCallNotificationEmitter>) {
        let Some(emitter) = emitter else {
            return;
        };
        while let Self::Buffer(buffered) = self {
            let Some(notification) = buffered.first().cloned() else {
                break;
            };
            emitter.emit_best_effort(notification);
            yield_to_outer_tool_stream().await;
            buffered.remove(0);
        }
        *self = Self::Emitter(emitter);
    }

    fn detach(&mut self) {
        if matches!(self, Self::Emitter(_)) {
            *self = Self::Buffer(Vec::new());
        }
    }

    fn buffered_len(&self) -> usize {
        match self {
            Self::Buffer(buffer) => buffer.len(),
            Self::Emitter(_) => 0,
        }
    }
}

/// Fields of delegate itself that a model sometimes nests under `parameters`.
const DELEGATE_TOP_LEVEL_FIELDS: &[&str] = &[
    "instructions",
    "source",
    "artifact_key",
    "artifact_title",
    "reference_artifacts",
    "context",
    "async",
    "max_turns",
];

/// Delegate fields passed inside `parameters`, with the shape delegate expects.
fn nested_delegate_fields_error(params: &DelegateParams) -> Option<String> {
    let parameters = params.parameters.as_ref()?;
    let nested: Vec<&str> = DELEGATE_TOP_LEVEL_FIELDS
        .iter()
        .copied()
        .filter(|field| parameters.contains_key(*field))
        .collect();
    // A recipe source may declare a parameter that shares a name; delegate's
    // own fields are only assumed when the call cannot work without them.
    let misplaced = params.source.is_none()
        || nested.contains(&"instructions")
        || nested.contains(&"artifact_key");
    if nested.is_empty() || !misplaced {
        return None;
    }
    Some(format!(
        "{} {} of delegate itself, not of parameters: pass {} at the top level, for example delegate({{\"source\": \"<specialist>\", \"instructions\": \"<complete task>\", \"artifact_key\": \"<key>\"}}). parameters is only for a recipe source's own declared parameters.",
        nested.join(", "),
        if nested.len() == 1 {
            "is a field"
        } else {
            "are fields"
        },
        if nested.len() == 1 { "it" } else { "them" },
    ))
}

/// An unknown delegate source, with the sources that can be delegated to.
fn unknown_delegate_source_error(source_name: &str, available: &[String]) -> String {
    if available.is_empty() {
        return format!(
            "Source '{}' not found, and no source can be delegated to here. Delegate with instructions only.",
            source_name
        );
    }
    format!(
        "Source '{}' not found. Pass one of the sources you can delegate to: {}.",
        source_name,
        available.join(", ")
    )
}

fn merge_subrecipe_parameters(
    fixed_values: Option<&HashMap<String, String>>,
    provided_parameters: Option<&HashMap<String, serde_json::Value>>,
) -> HashMap<String, String> {
    let mut merged = fixed_values.cloned().unwrap_or_default();
    if let Some(provided_parameters) = provided_parameters {
        for (key, value) in provided_parameters {
            let value = match value {
                serde_json::Value::String(value) => value.clone(),
                other => other.to_string(),
            };
            merged.entry(key.clone()).or_insert(value);
        }
    }
    merged
}

/// Result from handle_load_task_result with structured metadata for the caller
#[derive(Debug)]
struct TaskLoadResult {
    content: Vec<ContentBlock>,
    status: &'static str,
    turns: Option<u32>,
    duration_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct AgentMetadata {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    required_extensions: Vec<String>,
    #[serde(default)]
    required_skills: Vec<String>,
    #[serde(default)]
    always_async: bool,
    #[serde(default)]
    non_blocking: bool,
    #[serde(default)]
    artifact_guard: bool,
    #[serde(default)]
    event_driven_parent: bool,
    #[serde(default)]
    delegate_only: bool,
    #[serde(default)]
    artifact_result_tools: Vec<String>,
    /// The parent tool delegate calls to give a delegated artifact its ID.
    #[serde(default)]
    artifact_tool: Option<String>,
    /// The parent tool that connects artifact tasks, and the one-time reminder to use it.
    #[serde(default)]
    connect_tool: Option<String>,
    #[serde(default)]
    connect_reminder: Option<String>,
}

fn parse_agent_content(content: &str, path: &Path) -> Option<SourceEntry> {
    let (metadata, body): (AgentMetadata, String) = match parse_frontmatter(content) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => return None,
        Err(e) => {
            // Missing fields means this file has valid YAML but isn't an agent — skip silently.
            // Only warn on actual YAML syntax errors.
            if e.to_string().contains("missing field") {
                return None;
            }
            warn!("Failed to parse agent file {}: {}", path.display(), e);
            return None;
        }
    };

    let description = metadata.description.unwrap_or_else(|| {
        let model_info = metadata
            .model
            .as_ref()
            .map(|m| format!(" ({})", m))
            .unwrap_or_default();
        format!("Agent{}", model_info)
    });

    let mut properties = std::collections::HashMap::new();
    if let Some(model) = metadata.model {
        properties.insert("model".to_string(), serde_json::Value::String(model));
    }
    properties.insert(
        "required_extensions".to_string(),
        serde_json::json!(metadata.required_extensions),
    );
    properties.insert(
        "required_skills".to_string(),
        serde_json::json!(metadata.required_skills),
    );
    properties.insert(
        "always_async".to_string(),
        serde_json::json!(metadata.always_async),
    );
    properties.insert(
        "non_blocking".to_string(),
        serde_json::json!(metadata.non_blocking),
    );
    properties.insert(
        "delegate_only".to_string(),
        serde_json::json!(metadata.delegate_only),
    );
    properties.insert(
        "artifact_guard".to_string(),
        serde_json::json!(metadata.artifact_guard),
    );
    properties.insert(
        "event_driven_parent".to_string(),
        serde_json::json!(metadata.event_driven_parent),
    );
    properties.insert(
        "artifact_result_tools".to_string(),
        serde_json::json!(metadata.artifact_result_tools),
    );
    for (key, value) in [
        ("artifact_tool", metadata.artifact_tool),
        ("connect_tool", metadata.connect_tool),
        ("connect_reminder", metadata.connect_reminder),
    ] {
        if let Some(value) = value {
            properties.insert(key.to_string(), serde_json::Value::String(value));
        }
    }

    Some(SourceEntry {
        source_type: SourceType::Agent,
        name: metadata.name,
        description,
        content: body,
        path: path.to_string_lossy().into_owned(),
        global: false,
        writable: true,
        supporting_files: Vec::new(),
        properties,
    })
}

fn scan_recipes_from_dir(
    dir: &Path,
    kind: SourceType,
    suppress_config_warnings: bool,
    sources: &mut Vec<SourceEntry>,
    seen: &mut std::collections::HashSet<String>,
) {
    let Ok(source_dir) = dir.canonicalize() else {
        return;
    };
    let entries = match std::fs::read_dir(&source_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let path = source_dir.join(&file_name);

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !RECIPE_FILE_EXTENSIONS.contains(&ext) {
            continue;
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        if name.is_empty() || seen.contains(&name) {
            continue;
        }

        let content = match crate::skills::read_source_file(&source_dir, Path::new(&file_name)) {
            Ok(content) => content,
            Err(error) => {
                warn!("Failed to read recipe {}: {}", path.display(), error);
                continue;
            }
        };

        match Recipe::from_content(&content) {
            Ok(recipe) => {
                seen.insert(name.clone());
                sources.push(SourceEntry {
                    source_type: kind,
                    name,
                    description: recipe.description.clone(),
                    content: recipe.instructions.clone().unwrap_or_default(),
                    path: path.to_string_lossy().into_owned(),
                    global: false,
                    writable: true,
                    supporting_files: Vec::new(),
                    properties: std::collections::HashMap::new(),
                });
            }
            Err(e) => {
                // The working directory commonly contains project config like package.json
                // and tsconfig.json, which parse as valid JSON but lack Recipe fields. In that
                // case treat them as "not a recipe" rather than warning. Dedicated recipe
                // directories still warn so a real recipe with a typo is not silently dropped.
                if suppress_config_warnings && e.to_string().contains("missing field") {
                    continue;
                }
                warn!("Failed to parse recipe {}: {}", path.display(), e);
            }
        }
    }
}

fn scan_agents_from_dir(
    dir: &Path,
    sources: &mut Vec<SourceEntry>,
    seen: &mut std::collections::HashSet<String>,
) {
    let Ok(source_dir) = dir.canonicalize() else {
        return;
    };
    let entries = match std::fs::read_dir(&source_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let path = source_dir.join(&file_name);

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "md" {
            continue;
        }

        let content = match crate::skills::read_source_file(&source_dir, Path::new(&file_name)) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read agent file {}: {}", path.display(), e);
                continue;
            }
        };

        if let Some(source) = parse_agent_content(&content, &path) {
            if !seen.contains(&source.name) {
                seen.insert(source.name.clone());
                sources.push(source);
            }
        }
    }
}

pub fn discover_filesystem_sources(working_dir: &Path) -> Vec<SourceEntry> {
    let mut sources: Vec<SourceEntry> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let home = dirs::home_dir();
    let config = Paths::config_dir();

    let local_recipe_dirs: Vec<PathBuf> = vec![
        working_dir.join(".goose/recipes"),
        working_dir.join(".agents/recipes"),
    ];

    let global_recipe_dirs: Vec<PathBuf> = std::env::var("GOOSE_RECIPE_PATH")
        .ok()
        .into_iter()
        .flat_map(|p| {
            let sep = if cfg!(windows) { ';' } else { ':' };
            p.split(sep).map(PathBuf::from).collect::<Vec<_>>()
        })
        .chain(
            [
                home.as_ref().map(|h| h.join(".goose/recipes")),
                Some(config.join("recipes")),
                home.as_ref().map(|h| h.join(".agents/recipes")),
            ]
            .into_iter()
            .flatten(),
        )
        .collect();

    let local_agent_dirs: Vec<PathBuf> = vec![
        working_dir.join(".goose/agents"),
        working_dir.join(".claude/agents"),
        working_dir.join(".agents/agents"),
    ];

    let global_agent_dirs: Vec<PathBuf> = [
        home.as_ref().map(|h| h.join(".goose/agents")),
        home.as_ref().map(|h| h.join(".agents/agents")),
        Some(config.join("agents")),
        home.as_ref().map(|h| h.join(".claude/agents")),
    ]
    .into_iter()
    .flatten()
    .collect();

    scan_recipes_from_dir(
        working_dir,
        SourceType::Recipe,
        true,
        &mut sources,
        &mut seen,
    );

    for dir in local_recipe_dirs {
        scan_recipes_from_dir(&dir, SourceType::Recipe, false, &mut sources, &mut seen);
    }

    for dir in local_agent_dirs {
        scan_agents_from_dir(&dir, &mut sources, &mut seen);
    }

    for dir in global_recipe_dirs {
        scan_recipes_from_dir(&dir, SourceType::Recipe, false, &mut sources, &mut seen);
    }

    for dir in global_agent_dirs {
        scan_agents_from_dir(&dir, &mut sources, &mut seen);
    }

    sources
}

fn build_instructions_with_context(context: &str, instructions: &str) -> String {
    let mut result = format!("# Reference Context\n\n{}", context);
    if !instructions.is_empty() {
        result.push_str(&format!("\n\n# Task Instructions\n\n{}", instructions));
    }
    result
}

fn build_subagent_instructions(session: Option<&crate::session::Session>) -> String {
    let Some(session) = session else {
        return String::new();
    };

    if session.session_type == SessionType::SubAgent {
        return "Own the delegated task within your specialist scope. Apply parent guidance at the next checkpoint. Only the main agent communicates with the user. Use message_parent(message: ...) only in exceptional cases, to ask one concrete question you cannot resolve yourself and cannot continue without; the parent answers or asks the user and replies through your task. Your final result, including any limitation or warning, is delivered automatically, so never use message_parent for progress, results, or limitations. Do not delegate further.".to_string();
    }

    // filter the sources down to what we want even though currently that is what we get
    let mut sources: Vec<SourceEntry> = discover_filesystem_sources(&session.working_dir)
        .into_iter()
        .filter(|s| {
            matches!(
                s.source_type,
                SourceType::Agent | SourceType::Recipe | SourceType::Subrecipe
            )
        })
        .collect();

    // If the session is started from a recipe, also use the subrecipes for
    // that recipe as delegate targets
    if let Some(recipe) = session.recipe.as_ref() {
        if let Some(subs) = recipe.sub_recipes.as_ref() {
            let mut seen: std::collections::HashSet<String> =
                sources.iter().map(|s| s.name.clone()).collect();
            for sr in subs {
                if !seen.insert(sr.name.clone()) {
                    continue;
                }
                sources.push(SourceEntry {
                    source_type: SourceType::Subrecipe,
                    name: sr.name.clone(),
                    description: sr.description.clone().unwrap_or_default(),
                    content: String::new(),
                    path: sr.path.clone(),
                    global: false,
                    writable: false,
                    supporting_files: Vec::new(),
                    properties: std::collections::HashMap::new(),
                });
            }
        }
    }

    sources.sort_by(|a, b| (&a.source_type, &a.name).cmp(&(&b.source_type, &b.name)));
    let subagents: Vec<&SourceEntry> = sources.iter().collect();

    let names = subagents
        .iter()
        .map(|s| s.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let mut out = String::from(
        "Route each part of the request by the available named specialists' scope, including each requested deliverable in a compound task. Delegate specialist-owned work; complete work without a matching specialist yourself. When a requested deliverable matches a specialist, hand it off even if you could produce an inline answer yourself. For a specialist-owned deliverable with ready inputs, delegate directly; after completing prerequisites, delegate the remaining deliverable before declaring the request complete. A skill alone is not a reason to invent an ad-hoc delegate. Complete prerequisites before handing off dependent work: for research followed by document creation, prepare verified content and citations before delegating document operations. Pass the completed content or a local artifact, not a bare URL list or unfinished research. Keep the specialist's domain tools and private instructions in its child session.\n\nAvailable specialists are isolated execution targets; invoke them with delegate, never load their instructions into your own context:\n",
    );

    let mut current_kind: Option<SourceType> = None;
    for s in &subagents {
        if current_kind != Some(s.source_type) {
            out.push_str(&format!("\n{}:", kind_plural(s.source_type)));
            current_kind = Some(s.source_type);
        }
        out.push_str(&format!(
            "\n• {} — {}",
            s.name,
            safe_truncate(&s.description, SUBAGENT_DESCRIPTION_BUDGET)
        ));
    }

    let event_driven = subagents.iter().any(|source| {
        source
            .properties
            .get("event_driven_parent")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    });
    let waiting = if event_driven {
        "After delegating, tell the user once, in their terms, what you are working on, then call wait. What you write is shown to \
         the user as an update, so write only when the user learns something new, such as an artifact \
         finishing or failing; otherwise call wait without writing, for example after answering a specialist's \
         question. You are resumed automatically for each terminal report, each specialist question, and each \
         new user message; a user message sent while tasks run is yours to act on: send a change to the running \
         task it concerns, or handle a new request alongside the running work. \
         When the last task has reported, wait is unavailable: write the complete answer."
    } else {
        "Use load(source: task_id, peek: true) for requested status."
    };
    out.push_str(&format!(
        "\n\nWhen to call a subagent (one of [{names}]):\n\
         • `@<name>` in the user's message — always call that subagent.\n\
         • The user mentions a subagent by name without `@` — infer from \
         context whether they want it invoked, and if so, call it.\n\
         • The user's request strongly matches a subagent's description — \
         call it.\n\n\
         Call delegate(source: \"<name>\", instructions: ...) and put the complete task in instructions: \
         the requested outcome and every user requirement for that specialist's deliverable. \
         Specialists marked always_async or non_blocking run in the background: never poll or block on them. \
         Tell the user what you are doing in their terms (for example, \"Creating the presentation.\"), \
         never which specialist, agent, or task handles it. \
         Use the returned task ID for send(task_id: ..., message: ...) only to steer a running task with new guidance \
         or to answer its question; never use send to hand over the initial task, request status, or ask it to hurry. \
         {waiting} Questions and terminal reports arrive automatically as internal evidence. \
         Review reports against the latest user instructions, verify requested artifacts, and author \
         the user-facing response yourself. Never claim success when a specialist reports failure.",
    ));

    if subagents.iter().any(|source| {
        source
            .properties
            .get("artifact_guard")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    }) {
        out.push_str("\nArtifact reports: already_satisfied is successful verification of the existing saved revision without a new save. Accept its current revision-bound verification when it meets the user requirements; do not demand a mutation solely to obtain a new save. Partial and empty outcomes still require review. Respect edit_cycle_stalled and exhausted correction budgets: do not evade them by starting a fresh specialist or changing the task key. Preserve verified predecessor receipts and address only remaining user requirements.\n");
    }
    out
}

fn round_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", (secs / 10) * 10)
    } else {
        format!("{}m", secs / 60)
    }
}

fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const ARTIFACT_RESULTS_HEADING: &str = "## Artifact results";
const ARTIFACT_TEXT_BUDGET: usize = 400;
const ARTIFACT_RECEIPT_PREFIX: &str = "  Saved artifact receipt: ";

/// An artifact ID, such as `presentation-3`: how Cortex names an artifact for its whole
/// conversation. `kind` limits it to the artifacts one specialist role works on.
fn is_artifact_id(key: &str, kind: Option<&str>) -> bool {
    let Some((prefix, number)) = key.rsplit_once('-') else {
        return false;
    };
    !prefix.is_empty()
        && prefix.bytes().all(|byte| byte.is_ascii_lowercase())
        && kind.is_none_or(|kind| kind == prefix)
        && !number.is_empty()
        && number.len() <= 9
        && !number.starts_with('0')
        && number.bytes().all(|byte| byte.is_ascii_digit())
}

/// What the coordinator is told when a delegation revokes another task's read-only reference.
fn revocation_message(artifact: &str, holder: &str, holder_artifact: Option<&str>) -> String {
    match holder_artifact {
        Some(follower) => format!(
            "Read access to {artifact} was revoked for running task {holder} ({follower}), because this task now edits {artifact}. If {follower} should still follow {artifact}, open a channel with {artifact} leading and {follower} following; its running task joins it."
        ),
        None => format!(
            "Read access to {artifact} was revoked for running task {holder}, because this task now edits {artifact}."
        ),
    }
}

fn artifact_assignment(params: &DelegateParams, previous_results: Option<&str>) -> Option<String> {
    if params.artifact_key.is_none() && params.artifact_title.is_none() {
        return None;
    }
    let mut lines = vec!["Assigned artifact (set by the coordinator for this task):".to_string()];
    if let Some(key) = params.artifact_key.as_deref() {
        let new = params
            .artifact_status
            .as_deref()
            .map_or(key.starts_with("new:"), |status| status == "new");
        lines.push(if new {
            format!("- Artifact: {key}, a new artifact requested in this turn. Create it.")
        } else {
            format!("- Artifact: {key}, which already exists. Revise it; do not create another.")
        });
    }
    if let Some(title) = params.artifact_title.as_deref() {
        lines.push(format!("- Title: {title}"));
    }
    if !params.references.is_empty() {
        let listed: Vec<String> = params
            .references
            .iter()
            .map(|reference| {
                let mut line = reference.artifact.clone();
                if let Some(title) = reference.title.as_deref() {
                    line.push_str(&format!(" (\"{title}\")"));
                }
                if let Some(revision) = reference.revision {
                    line.push_str(&format!(", saved revision {revision}"));
                }
                line
            })
            .collect();
        lines.push(format!(
            "- Read-only references: {}. Read them with read_document; you cannot edit them, and they stay at the revision named. You can read only your artifact and these references.",
            listed.join("; ")
        ));
    }
    if let Some(previous) = params.previous_task_id.as_deref() {
        lines.push(format!(
            "- Follow-up to task {previous} for the same artifact."
        ));
        match previous_results {
            Some(results) => {
                lines.push(format!("- That task's artifact results:\n{results}"));
                lines.push(
                    "- If it saved a document, continue that document instead of creating another copy."
                        .to_string(),
                );
            }
            None => lines.push("- That task reported no artifact results.".to_string()),
        }
    }
    Some(lines.join("\n"))
}

fn artifact_results_from_report(body: &str) -> Option<&str> {
    body.split_once(ARTIFACT_RESULTS_HEADING)
        .map(|(_, results)| results.trim())
}

fn artifact_target(arguments: Option<&JsonObject>) -> String {
    let argument = |key: &str| {
        arguments
            .and_then(|arguments| arguments.get(key))
            .and_then(serde_json::Value::as_str)
    };
    match (argument("document_id"), argument("title")) {
        (Some(id), _) => format!("document {id}"),
        (None, Some(title)) => format!("a new document titled \"{title}\""),
        // A revision names no document: the task's artifact is its target.
        (None, None) => "the task's artifact".to_string(),
    }
}

fn tool_result_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .map(|text| text.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The artifact tool's answer for one artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactResolution {
    artifact: String,
    /// "new" while the artifact has no document yet, otherwise "existing".
    status: String,
    /// The saved revision of an existing artifact.
    revision_id: Option<String>,
    revision: Option<i64>,
    title: Option<String>,
}

/// The artifact tool's answer: the artifact ID, "new" or "existing", and an existing
/// artifact's saved revision, given as structured content and as JSON text.
fn artifact_resolution(result: &CallToolResult) -> Result<ArtifactResolution, String> {
    let text = tool_result_text(result);
    if result.is_error == Some(true) {
        return Err(text);
    }
    let value = result
        .structured_content
        .clone()
        .or_else(|| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    match (value["artifact"].as_str(), value["status"].as_str()) {
        (Some(artifact), Some(status)) => Ok(ArtifactResolution {
            artifact: crate::session::normalize_artifact_key(artifact),
            status: status.to_string(),
            revision_id: value["revision_id"].as_str().map(str::to_owned),
            revision: value["revision"].as_i64(),
            title: value["title"].as_str().map(str::to_owned),
        }),
        _ => Err(format!("The artifact_key could not be resolved: {text}")),
    }
}

fn editor_result_value(result: &CallToolResult) -> Option<serde_json::Value> {
    result
        .structured_content
        .clone()
        .or_else(|| {
            result
                .content
                .iter()
                .filter_map(|content| content.as_text())
                .find_map(|text| serde_json::from_str(&text.text).ok())
        })
        .filter(|value: &serde_json::Value| value.get("document_id").is_some())
}

fn describe_editor_result(value: &serde_json::Value) -> String {
    let text = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    };
    let status = match text("status") {
        "" => "unknown",
        status => status,
    };
    let revision = value
        .get("document_revision")
        .and_then(serde_json::Value::as_i64)
        .map_or_else(
            || "no saved revision".to_string(),
            |revision| {
                if status == "already_satisfied" {
                    format!("existing revision {revision} unchanged; no new save")
                } else {
                    format!("saved revision {revision}")
                }
            },
        );
    // Reports name the document by its conversation handle when the receipt carries one.
    let document = match text("document") {
        "" => text("document_id"),
        handle => handle,
    };
    let mut line = format!("- Editor job {status}: document {document}, {revision}");
    for (label, key) in [("Summary", "summary"), ("Remaining work", "remaining_work")] {
        let field = text(key).trim();
        if !field.is_empty() {
            line.push_str(&format!(
                ". {label}: {}",
                safe_truncate(&field.replace(['\n', '\r'], " "), ARTIFACT_TEXT_BUDGET)
            ));
        }
    }
    if let Some(receipt) = saved_artifact_receipt(value) {
        line.push_str(&format!(
            "\n{ARTIFACT_RECEIPT_PREFIX}{}",
            serde_json::to_string(&receipt).unwrap()
        ));
    }
    line
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EditorArtifactStatus {
    Completed,
    AlreadySatisfied,
    Partial,
    Empty,
    Failed,
    Cancelled,
}

#[derive(Deserialize)]
struct EditorArtifactReceipt {
    #[serde(default)]
    job_id: Option<String>,
    document_id: String,
    /// The document's conversation handle (document-3), which the model uses in its calls.
    #[serde(default)]
    document: Option<String>,
    status: EditorArtifactStatus,
    document_revision: Option<u64>,
}

/// One artifact's identity across calls and receipts: its conversation handle when known, so a
/// call naming `document-3` and the receipt for that document's ID are the same artifact.
fn artifact_identity(document_id: &str, document: Option<&str>) -> String {
    document
        .filter(|handle| !handle.is_empty())
        .unwrap_or(document_id)
        .to_owned()
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactVerificationCheck {
    name: String,
    valid: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactVerification {
    engine_revision: u64,
    source_digest: String,
    valid: bool,
    scope: String,
    #[serde(default)]
    checks: Vec<ArtifactVerificationCheck>,
    layout_issue_count: Option<u64>,
    #[serde(default)]
    issues: Vec<String>,
    #[serde(default)]
    omitted_checks: u64,
    #[serde(default)]
    omitted_issues: u64,
    #[serde(default)]
    diagnostics_compacted: bool,
}

impl ArtifactVerification {
    fn bounded(&self) -> bool {
        self.scope == "artifact"
            && self.source_digest.len() == 64
            && self
                .source_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            && self.checks.len() <= 8
            && self
                .checks
                .iter()
                .all(|check| check.name.chars().count() <= 120)
            && self.issues.len() <= 6
            && serde_json::to_vec(self).is_ok_and(|bytes| bytes.len() <= 4096)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactIntegrity {
    validation_status: String,
    saved_associations_count: u32,
    dropped_count: u32,
    unresolved_count: u32,
}

impl ArtifactIntegrity {
    fn bounded(&self) -> bool {
        matches!(self.validation_status.as_str(), "valid" | "issues")
            && (self.validation_status == "issues")
                == (self.dropped_count > 0 || self.unresolved_count > 0)
            && [
                self.saved_associations_count,
                self.dropped_count,
                self.unresolved_count,
            ]
            .iter()
            .all(|count| *count <= i32::MAX as u32)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SavedArtifactReceipt {
    job_id: Option<String>,
    document_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    document: Option<String>,
    status: EditorArtifactStatus,
    document_revision: u64,
    saved_revision_id: Option<String>,
    verification: Option<ArtifactVerification>,
    grounding_integrity: Option<ArtifactIntegrity>,
    policy_stop_reason: Option<String>,
}

fn saved_artifact_receipt(value: &serde_json::Value) -> Option<SavedArtifactReceipt> {
    let receipt: EditorArtifactReceipt = serde_json::from_value(value.clone()).ok()?;
    let document_revision = receipt.document_revision.filter(|revision| *revision > 0)?;
    if receipt.document_id.is_empty() || receipt.document_id.len() > 200 {
        return None;
    }
    let bounded_text = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.is_empty() && text.len() <= 200)
            .map(str::to_owned)
    };
    let saved_revision_id = bounded_text("saved_revision_id");
    let verification = saved_revision_id.as_ref().and_then(|_| {
        serde_json::from_value::<ArtifactVerification>(value.get("verification")?.clone())
            .ok()
            .filter(ArtifactVerification::bounded)
    });
    if receipt.status == EditorArtifactStatus::AlreadySatisfied
        && !verification.as_ref().is_some_and(|proof| proof.valid)
    {
        return None;
    }
    let grounding_integrity = saved_revision_id.as_ref().and_then(|_| {
        serde_json::from_value::<ArtifactIntegrity>(value.get("grounding_integrity")?.clone())
            .ok()
            .filter(ArtifactIntegrity::bounded)
    });
    Some(SavedArtifactReceipt {
        job_id: receipt.job_id.filter(|id| id.len() <= 200),
        document_id: receipt.document_id,
        document: receipt
            .document
            .filter(|handle| !handle.is_empty() && handle.len() <= 40),
        status: receipt.status,
        document_revision,
        saved_revision_id,
        verification,
        grounding_integrity,
        policy_stop_reason: bounded_text("policy_stop_reason"),
    })
}

fn confirmed_editor_status(value: &serde_json::Value) -> Option<EditorArtifactStatus> {
    let receipt: EditorArtifactReceipt = serde_json::from_value(value.clone()).ok()?;
    match receipt.status {
        EditorArtifactStatus::AlreadySatisfied => {
            saved_artifact_receipt(value).map(|receipt| receipt.status)
        }
        EditorArtifactStatus::Completed | EditorArtifactStatus::Partial
            if receipt
                .document_revision
                .is_none_or(|revision| revision == 0) =>
        {
            None
        }
        status => Some(status),
    }
}

fn predecessor_receipts(messages: &[Message]) -> Vec<SavedArtifactReceipt> {
    messages
        .iter()
        .take(1)
        .filter(|message| message.role == Role::User)
        .flat_map(|message| &message.content)
        .filter_map(|content| {
            let text = content.as_text()?;
            text.starts_with("Assigned artifact (set by the coordinator for this task):")
                .then_some(text)
        })
        .flat_map(|text| text.lines().take_while(|line| !line.is_empty()))
        .filter_map(|line| {
            let value: serde_json::Value =
                serde_json::from_str(line.strip_prefix(ARTIFACT_RECEIPT_PREFIX)?).ok()?;
            saved_artifact_receipt(&value)
        })
        .collect()
}

struct ArtifactResultSummary {
    lines: Vec<String>,
    current: HashMap<String, Option<EditorArtifactStatus>>,
}

impl ArtifactResultSummary {
    fn completion_description(&self) -> &'static str {
        if self.current.is_empty() || self.current.values().any(|status| status.is_none()) {
            "ended without a confirmed artifact outcome"
        } else if self.current.values().any(|status| {
            matches!(
                status,
                Some(
                    EditorArtifactStatus::Failed
                        | EditorArtifactStatus::Cancelled
                        | EditorArtifactStatus::Empty
                )
            )
        }) {
            "ended with incomplete artifact output"
        } else if self
            .current
            .values()
            .any(|status| *status == Some(EditorArtifactStatus::Partial))
        {
            "ended with partial artifact output"
        } else {
            "completed successfully"
        }
    }

    fn section(&self) -> String {
        let body = if self.lines.is_empty() {
            "- No editor job was started, so nothing was saved.".to_string()
        } else {
            self.lines.join("\n")
        };
        format!("{ARTIFACT_RESULTS_HEADING}\n{body}")
    }
}

/// The conversation with each editor result the tool server delivered to the
/// task's mailbox written as the tool call and response it stands for: a call of
/// the first artifact result tool with the task's idempotency_key, answered by
/// its receipt. It follows the delivered message, so it supersedes the
/// still-running result of the call that started the task.
fn with_delivered_editor_results(messages: &[Message], tools: &[String]) -> Vec<Message> {
    let Some(tool) = tools.first() else {
        return messages.to_vec();
    };
    let mut normalized = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        normalized.push(message.clone());
        let Some(result) = message
            .metadata
            .operation_note(crate::session::EDITOR_RESULT_NOTE, "v1")
        else {
            continue;
        };
        let (Some(key), Some(receipt)) = (
            result
                .get("idempotency_key")
                .and_then(serde_json::Value::as_str),
            result.get("receipt").filter(|receipt| receipt.is_object()),
        ) else {
            continue;
        };
        let id = format!("delivered_editor_result_{index}");
        let mut arguments = JsonObject::new();
        arguments.insert("idempotency_key".to_string(), serde_json::json!(key));
        normalized.push(Message::assistant().with_tool_request(
            id.clone(),
            Ok(rmcp::model::CallToolRequestParams::new(tool.clone()).with_arguments(arguments)),
        ));
        normalized.push(
            Message::user().with_tool_response(id, Ok(CallToolResult::structured(receipt.clone()))),
        );
    }
    normalized
}

fn artifact_result_summary(messages: &[Message], tools: &[String]) -> ArtifactResultSummary {
    let normalized = with_delivered_editor_results(messages, tools);
    let messages = normalized.as_slice();
    let mut calls = Vec::new();
    let mut responses = HashMap::new();
    let mut response_order = HashMap::new();
    for (position, message) in messages.iter().enumerate() {
        for content in &message.content {
            match content {
                MessageContent::ToolRequest(request) => {
                    if let Ok(call) = &request.tool_call {
                        if tools.iter().any(|tool| call.name == tool.as_str()) {
                            calls.push((request.id.as_str(), call.arguments.as_ref()));
                        }
                    }
                }
                MessageContent::ToolResponse(response) => {
                    responses.insert(response.id.as_str(), &response.tool_result);
                    response_order.insert(response.id.as_str(), position);
                }
                _ => {}
            }
        }
    }
    let task_key = |arguments: Option<&JsonObject>| {
        arguments
            .and_then(|args| args.get("idempotency_key"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    let interrupted = |id: &str| matches!(responses.get(id), Some(Ok(result)) if crate::agents::tool_interrupt::was_interrupted(result));
    let mut targets = HashMap::new();
    for (_, arguments) in &calls {
        if let Some(key) = task_key(*arguments) {
            if arguments.is_some_and(|args| args.contains_key("instruction")) {
                targets.insert(key, artifact_target(*arguments));
            }
        }
    }
    let inherited = predecessor_receipts(messages);
    let mut current: HashMap<_, _> = inherited
        .iter()
        .map(|receipt| {
            (
                artifact_identity(&receipt.document_id, receipt.document.as_deref()),
                Some(receipt.status),
            )
        })
        .collect();
    let mut lines: Vec<String> = inherited
        .iter()
        .map(|receipt| {
            format!(
                "- Predecessor saved outcome (checks apply only to its recorded revision):\n{}",
                describe_editor_result(&serde_json::to_value(receipt).unwrap())
            )
        })
        .collect();
    lines.extend(calls.iter().enumerate()
        .filter(|(index, (id, arguments))| {
            if !interrupted(id) { return true; }
            let Some(key) = task_key(*arguments) else { return true; };
            !calls[index + 1..].iter().any(|(later_id, later_args)| {
                task_key(*later_args).as_ref() == Some(&key)
                    && response_order.get(later_id).zip(response_order.get(id)).is_some_and(|(later, earlier)| later > earlier)
                    && responses.get(later_id).and_then(|response| response.as_ref().ok())
                        .filter(|result| !crate::agents::tool_interrupt::was_interrupted(result) && result.is_error != Some(true))
                        .and_then(editor_result_value)
                        .filter(|value| confirmed_editor_status(value).is_some())
                        .and_then(|value| serde_json::from_value::<EditorArtifactReceipt>(value).ok())
                        .is_some_and(|receipt| receipt.job_id.is_some_and(|job| !job.is_empty()) && !receipt.document_id.is_empty() && (!matches!(receipt.status, EditorArtifactStatus::Completed | EditorArtifactStatus::Partial) || receipt.document_revision.is_some_and(|revision| revision > 0)) && arguments.and_then(|args| args.get("document_id")).and_then(serde_json::Value::as_str).is_none_or(|document| document == receipt.document_id || receipt.document.as_deref() == Some(document)))
            })
        })
        .map(|(_, (id, arguments))| (*id, *arguments))
        .map(|(id, arguments)| {

            let target = task_key(arguments)
                .and_then(|key| targets.get(&key).cloned())
                .unwrap_or_else(|| artifact_target(arguments));
            if interrupted(id) {
                current.insert(format!("call:{id}"), None);
                return format!(
                    "- Editor job for {target} was still running when this task ended; its saved state is unknown."
                );
            }
            let receipt = responses.get(id).and_then(|response| response.as_ref().ok()).and_then(editor_result_value).and_then(|value| serde_json::from_value::<EditorArtifactReceipt>(value).ok());
            let identity = receipt.as_ref().map(|receipt| artifact_identity(&receipt.document_id, receipt.document.as_deref())).or_else(|| arguments.and_then(|arguments| arguments.get("document_id")).and_then(serde_json::Value::as_str).map(str::to_owned)).unwrap_or_else(|| format!("call:{id}"));
            let rejected = responses.get(id).and_then(|response| response.as_ref().ok()).is_some_and(|result| {
                serde_json::from_str::<serde_json::Value>(&tool_result_text(result)).ok().is_some_and(|value| value.get("code").and_then(serde_json::Value::as_str) == Some("editor_policy_exhausted"))
            });
            if !rejected {
            current.insert(identity, responses.get(id).and_then(|response| response.as_ref().ok()).and_then(editor_result_value).and_then(|value| confirmed_editor_status(&value)));
            }

            match responses.get(id) {
                None => format!(
                    "- Editor job for {target} was still running when this task ended; its saved state is unknown."
                ),
                Some(Err(error)) => format!(
                    "- Editor call for {target} failed: {}",
                    safe_truncate(&error.message, ARTIFACT_TEXT_BUDGET)
                ),
                Some(Ok(result)) => match editor_result_value(result) {
                    Some(value) => describe_editor_result(&value),
                    None if rejected => format!(
                        "- Continuation for {target} was rejected; no NEW revision was saved. Existing saved artifact receipts and their recorded checks remain unchanged: {}",
                        safe_truncate(&tool_result_text(result), ARTIFACT_TEXT_BUDGET)
                    ),
                    None => format!(
                        "- Editor call for {target} returned no saved document: {}",
                        safe_truncate(&tool_result_text(result), ARTIFACT_TEXT_BUDGET)
                    ),
                },
            }
        })
        );
    ArtifactResultSummary { lines, current }
}

/// The idempotency keys of editor tasks a specialist started whose latest
/// result says the editor is still working (a delivered editor result counts
/// as the latest). A specialist must not finish while one runs: its report
/// would not have the editor's result.
pub(crate) fn running_editor_tasks(
    session: &crate::session::Session,
    messages: &[Message],
) -> Vec<String> {
    SummonTaskPolicy::from_session(session)
        .map(|policy| running_editor_task_keys(messages, &policy.artifact_result_tools))
        .unwrap_or_default()
}

/// Wait without another model call until durable news can be reviewed.
pub(crate) async fn wait_for_editor_notice(
    manager: &crate::session::SessionManager,
    session_id: &str,
    cancel: &CancellationToken,
    timeout: Duration,
) -> Result<()> {
    let waiting = async {
        loop {
            if manager
                .pending_session_messages(session_id)
                .await?
                .iter()
                .any(|message| {
                    message.interrupts_wait()
                        || message.kind == crate::session::MailboxMessageKind::Channel
                })
            {
                return Ok(());
            }
            tokio::time::sleep(SPECIALIST_WAIT_POLL).await;
        }
    };
    tokio::select! {
        biased;
        _ = cancel.cancelled() => anyhow::bail!("Incomplete artifact output: editor wait cancelled"),
        result = tokio::time::timeout(timeout, waiting) => {
            result.map_err(|_| anyhow::anyhow!("Incomplete artifact output: editor result deadline exceeded"))?
        }
    }
}

fn running_editor_task_keys(messages: &[Message], tools: &[String]) -> Vec<String> {
    if tools.is_empty() {
        return Vec::new();
    }
    let normalized = with_delivered_editor_results(messages, tools);
    let messages = normalized.as_slice();
    let mut keys: HashMap<&str, String> = HashMap::new();
    let mut running: HashMap<String, bool> = HashMap::new();
    for message in messages {
        for content in &message.content {
            match content {
                MessageContent::ToolRequest(request) => {
                    if let Ok(call) = &request.tool_call {
                        if tools.iter().any(|tool| call.name == tool.as_str()) {
                            if let Some(key) = call
                                .arguments
                                .as_ref()
                                .and_then(|args| args.get("idempotency_key"))
                                .and_then(serde_json::Value::as_str)
                            {
                                keys.insert(request.id.as_str(), key.to_owned());
                            }
                        }
                    }
                }
                MessageContent::ToolResponse(response) => {
                    if let Some(key) = keys.get(response.id.as_str()) {
                        let still_running = matches!(
                            &response.tool_result,
                            Ok(result) if crate::agents::tool_interrupt::was_interrupted(result)
                        );
                        running.insert(key.clone(), still_running);
                    }
                }
                _ => {}
            }
        }
    }
    let mut keys: Vec<String> = running
        .into_iter()
        .filter_map(|(key, still_running)| still_running.then_some(key))
        .collect();
    keys.sort();
    keys
}

#[cfg(test)]
fn artifact_result_lines(messages: &[Message], tools: &[String]) -> Vec<String> {
    artifact_result_summary(messages, tools).lines
}

async fn artifact_results(
    manager: &crate::session::SessionManager,
    task_id: &str,
) -> Option<ArtifactResultSummary> {
    let session = manager.get_session(task_id, true).await.ok()?;
    let policy = SummonTaskPolicy::from_session(&session)?;
    if policy.artifact_result_tools.is_empty() {
        return None;
    }
    Some(
        session
            .conversation
            .as_ref()
            .map(|conversation| {
                artifact_result_summary(conversation.messages(), &policy.artifact_result_tools)
            })
            .unwrap_or_else(|| ArtifactResultSummary {
                lines: Vec::new(),
                current: HashMap::new(),
            }),
    )
}

async fn artifact_results_section(
    manager: &crate::session::SessionManager,
    task_id: &str,
) -> Option<String> {
    artifact_results(manager, task_id)
        .await
        .map(|summary| summary.section())
}

/// Get maximum number of concurrent background tasks
fn max_background_tasks() -> usize {
    Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(5)
}

fn completed_task_ttl() -> Duration {
    let secs = Config::global()
        .get_param::<u64>("GOOSE_COMPLETED_TASK_TTL_SECS")
        .unwrap_or(600);
    Duration::from_secs(secs)
}

fn is_session_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('_').collect();
    parts.len() == 2 && parts[0].len() == 8 && parts[0].chars().all(|c| c.is_ascii_digit())
}

pub struct SummonClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
    source_cache: Mutex<Option<(Instant, PathBuf, Vec<SourceEntry>)>>,
    background_tasks: Mutex<HashMap<String, BackgroundTask>>,
    completed_tasks: Mutex<HashMap<String, CompletedTask>>,
    artifact_tasks: Mutex<HashMap<(String, String), String>>,
    event_driven_parents: Mutex<HashSet<String>>,
    /// Highest mailbox message ID delivered to each parent's current report
    /// turn. Delivered messages stay pending until the turn is acknowledged.
    delivered_reports: Mutex<HashMap<String, i64>>,
    /// Parents already shown a source's connect_reminder.
    connect_reminded: Mutex<HashSet<String>>,
}

impl Drop for SummonClient {
    fn drop(&mut self) {
        // Best-effort cancellation of running tasks on shutdown
        if let Ok(tasks) = self.background_tasks.try_lock() {
            for task in tasks.values() {
                task.cancellation_token.cancel();
            }
        }
    }
}

impl SummonClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(EXTENSION_NAME, "1.0.0").with_title("Summon"));

        Ok(Self {
            info,
            context,
            source_cache: Mutex::new(None),
            background_tasks: Mutex::new(HashMap::new()),
            completed_tasks: Mutex::new(HashMap::new()),
            artifact_tasks: Mutex::new(HashMap::new()),
            event_driven_parents: Mutex::new(HashSet::new()),
            delivered_reports: Mutex::new(HashMap::new()),
            connect_reminded: Mutex::new(HashSet::new()),
        })
    }

    async fn create_subagent_session(
        &self,
        task_config: &TaskConfig,
        name: String,
        policy: Option<&SummonTaskPolicy>,
        source_name: &str,
    ) -> Result<crate::session::Session, String> {
        let session = self
            .context
            .session_manager
            .create_session(
                task_config.parent_working_dir.clone(),
                name,
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .map_err(|e| format!("Failed to create subagent session: {}", e))?;

        if !task_config.parent_session_id.is_empty() {
            let mut extension_data = session.extension_data.clone();
            if let Some(policy) = policy {
                extension_data.set_extension_state(
                    "summon",
                    "v1",
                    serde_json::to_value(policy).map_err(|error| error.to_string())?,
                );
                let mut admission = self
                    .context
                    .session_manager
                    .capture_task_admission(
                        &task_config.parent_session_id,
                        &session.id,
                        source_name,
                    )
                    .await
                    .map_err(|error| error.to_string())?;
                admission.artifact_key = policy.artifact_key.clone();
                admission.previous_task_id = policy.previous_task_id.clone();
                extension_data.set_extension_state(
                    "summon",
                    "task_admission_v1",
                    serde_json::to_value(admission).map_err(|error| error.to_string())?,
                );
            }
            self.context
                .session_manager
                .update(&session.id)
                .parent_session_id(Some(task_config.parent_session_id.clone()))
                .extension_data(extension_data)
                .apply()
                .await
                .map_err(|e| format!("Failed to link subagent to parent session: {}", e))?;
        }

        Ok(session)
    }

    fn notification_sink(emitter: Option<ToolCallNotificationEmitter>) -> SharedNotificationSink {
        Arc::new(Mutex::new(match emitter {
            Some(emitter) => NotificationSink::Emitter(emitter),
            None => NotificationSink::Buffer(Vec::new()),
        }))
    }

    /// The artifact a delegation names: an artifact ID of the source's kind
    /// (`presentation-3` for cortex-presentation), or `new:<kind>` for a new artifact
    /// (anything after the kind is ignored once the artifact tool gives it its ID).
    fn artifact_key(params: &DelegateParams) -> Result<String, String> {
        let role = params
            .source
            .as_deref()
            .and_then(|name| name.strip_prefix("cortex-"))
            .ok_or_else(|| "Artifact-guarded sources must have a cortex role".to_string())?;
        let key = params
            .artifact_key
            .as_deref()
            .map(crate::session::normalize_artifact_key)
            .filter(|key| {
                is_artifact_id(key, Some(role))
                    || key
                        .strip_prefix("new:")
                        .and_then(|rest| rest.strip_prefix(role))
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with(':'))
            })
            .filter(|key| key.len() <= 300)
            .ok_or_else(|| format!("Office delegation requires artifact_key: the artifact ID ({role}-N) of an artifact that has one, or new:{role} for a new one"))?;
        if params
            .artifact_title
            .as_ref()
            .is_some_and(|title| title.trim().is_empty() || title.len() > 240)
        {
            return Err("Artifact titles must be nonempty labels of at most 240 bytes".to_string());
        }
        Ok(key)
    }

    /// A follow-up continues its artifact's last task: delegate sets previous_task_id
    /// to the task that owns the artifact in this session, whatever the caller passed.
    async fn derive_previous_task_id(
        &self,
        session_id: &str,
        artifact_keys: &[String],
        params: &mut DelegateParams,
    ) -> Result<(), String> {
        params.previous_task_id = None;
        if artifact_keys.is_empty() {
            return Ok(());
        }
        let mut artifact_tasks = self.artifact_tasks.lock().await;
        self.reconstruct_artifact_tasks(session_id, &mut artifact_tasks, artifact_keys)
            .await?;
        params.previous_task_id = artifact_keys
            .iter()
            .find_map(|key| artifact_tasks.get(&(session_id.to_string(), key.clone())))
            .cloned();
        Ok(())
    }

    /// Asks the parent's artifact tool (a source's `artifact_tool`) for the artifact ID a
    /// delegation names: `new:<kind>` reserves the next one, an artifact ID is checked.
    /// Returns the ID and whether its document exists yet ("new" or "existing").
    async fn resolve_artifact(
        &self,
        session_id: &str,
        working_dir: &Path,
        tool: &str,
        key: &str,
        kind: &str,
    ) -> Result<ArtifactResolution, String> {
        let manager = self
            .context
            .extension_manager
            .as_ref()
            .and_then(|manager| manager.upgrade())
            .ok_or_else(|| "Artifact IDs are unavailable in this session.".to_string())?;
        let ctx = ToolCallContext::new(
            session_id.to_string(),
            Some(working_dir.to_path_buf()),
            None,
        );
        let mut arguments = JsonObject::new();
        arguments.insert("artifact_key".to_string(), serde_json::json!(key));
        arguments.insert("kind".to_string(), serde_json::json!(kind));
        let call = CallToolRequestParams::new(tool.to_string()).with_arguments(arguments);
        let dispatched = manager
            .dispatch_tool_call(&ctx, call, CancellationToken::new())
            .await
            .map_err(|error| {
                format!("The artifact_key could not be resolved: {}", error.message)
            })?;
        let result = dispatched.result.await.map_err(|error| {
            format!("The artifact_key could not be resolved: {}", error.message)
        })?;
        artifact_resolution(&result)
    }

    /// Resolves a delegation's read-only references through the artifact tool: each
    /// must be an existing artifact with a saved revision, other than the task's own.
    async fn resolve_references(
        &self,
        session_id: &str,
        working_dir: &Path,
        tool: &str,
        requested: &[String],
        own: Option<&str>,
    ) -> Result<Vec<ArtifactReference>, String> {
        if requested.len() > MAX_REFERENCE_ARTIFACTS {
            return Err(format!(
                "A delegation can carry at most {MAX_REFERENCE_ARTIFACTS} reference_artifacts."
            ));
        }
        let mut references: Vec<ArtifactReference> = Vec::new();
        for key in requested {
            let key = crate::session::normalize_artifact_key(key);
            if key.starts_with("new:") {
                return Err(format!(
                    "reference_artifacts names existing artifacts by their IDs; {key} is not one. A reference must already exist."
                ));
            }
            let resolved = self
                .resolve_artifact(session_id, working_dir, tool, &key, "")
                .await?;
            if Some(resolved.artifact.as_str()) == own {
                return Err(format!(
                    "{} is the task's own artifact; reference_artifacts lists only other artifacts it reads.",
                    resolved.artifact
                ));
            }
            let Some(revision_id) = resolved
                .revision_id
                .filter(|_| resolved.status == "existing")
            else {
                return Err(format!(
                    "{} has no saved document yet, so it cannot be a reference. If this artifact must follow it while it is being written, use a channel.",
                    resolved.artifact
                ));
            };
            if references
                .iter()
                .all(|reference| reference.artifact != resolved.artifact)
            {
                references.push(ArtifactReference {
                    artifact: resolved.artifact,
                    revision_id: Some(revision_id),
                    revision: resolved.revision,
                    title: resolved.title,
                });
            }
        }
        Ok(references)
    }

    /// A reference must be frozen: no running task of this parent may be editing it.
    async fn check_references_frozen(
        &self,
        session_id: &str,
        reference_keys: &[String],
        artifact_tasks: &HashMap<(String, String), String>,
    ) -> Result<(), String> {
        let running = self.background_tasks.lock().await;
        for key in reference_keys {
            if let Some(owner) = artifact_tasks
                .get(&(session_id.to_string(), key.clone()))
                .filter(|owner| running.contains_key(owner.as_str()))
            {
                return Err(format!(
                    "This delegation was rejected and no task started: {key} is already assigned to running task {owner}, so it cannot be a read-only reference, which must be an artifact no running task is editing. If you want this artifact aligned with {key}, do it through a channel: open one with {key} leading and this artifact following, then delegate this artifact again without {key} in reference_artifacts."
                ));
            }
        }
        Ok(())
    }

    /// Running tasks of this parent that still hold `artifact` as a read-only
    /// reference, with the artifact each works on; delegating `artifact` revokes it.
    async fn referencing_tasks(
        &self,
        session_id: &str,
        artifact: &str,
    ) -> Result<Vec<(String, Option<String>)>, String> {
        let running: Vec<String> = self
            .background_tasks
            .lock()
            .await
            .values()
            .filter(|task| task.parent_session_id == session_id)
            .map(|task| task.id.clone())
            .collect();
        let mut holders = Vec::new();
        for task_id in running {
            let Some(policy) = self.task_policy(&task_id).await? else {
                continue;
            };
            if !policy
                .references
                .iter()
                .any(|reference| reference.artifact == artifact)
            {
                continue;
            }
            let revoked = self
                .context
                .session_manager
                .reference_revoked(session_id, &task_id, artifact)
                .await
                .map_err(|error| error.to_string())?;
            if !revoked {
                holders.push((task_id, policy.artifact_key));
            }
        }
        Ok(holders)
    }

    /// Tells a running task that its read-only reference was revoked.
    async fn notify_revoked_reference(&self, task_id: &str, artifact: &str) {
        let manager = &self.context.session_manager;
        let attempt_key = match manager.task_admission(task_id).await {
            Ok(Some(admission)) => admission.attempt_key,
            _ => None,
        };
        let Some(attempt_key) = attempt_key else {
            warn!("Revoked reference {artifact} of task {task_id} has no attempt to notify");
            return;
        };
        let notice = goose_sdk_types::custom_requests::TaskNoticeRequest {
            attempt_key,
            task_id: Some(task_id.to_string()),
            artifact_keys: vec![],
            text: format!(
                "Your read access to {artifact} is revoked: another task is now editing it, so it is no longer frozen. Do not read it again. If the coordinator adds you to a channel with it, follow that channel's publications instead."
            ),
            wake: true,
            refresh_tools: false,
            dedupe_key: Some(format!("reference-revoked:{artifact}")),
            editor_result: None,
            channel_wait: None,
        };
        if let Err(error) = manager.queue_task_notice(&notice).await {
            warn!("Failed to notify task {task_id} that its reference {artifact} was revoked: {error}");
        }
    }

    async fn check_artifact_owners(
        &self,
        session_id: &str,
        artifact_keys: &[String],
        previous_task_id: Option<&str>,
        artifact_tasks: &HashMap<(String, String), String>,
    ) -> Result<(), String> {
        for key in artifact_keys {
            if let Some(existing) = artifact_tasks.get(&(session_id.to_string(), key.clone())) {
                if self.background_tasks.lock().await.contains_key(existing) {
                    return Err(format!("Artifact {key} already has active specialist task {existing}. Steer that task with send."));
                }
                self.recover_completion_delivery(existing).await?;
                if self
                    .context
                    .session_manager
                    .terminal_report_for_child(session_id, existing)
                    .await
                    .map_err(|error| error.to_string())?
                    .is_none()
                {
                    return Err(format!("The previous task {existing} has an interrupted or unknown outcome, without canonical terminal evidence. Inspect the saved artifact before recovery; no replacement task was started."));
                }
                if previous_task_id != Some(existing.as_str()) {
                    return Err(format!("Review the terminal result of task {existing} and the saved artifact before delegating a follow-up with previous_task_id."));
                }
            }
        }
        if let Some(previous) = previous_task_id {
            if !artifact_keys.iter().any(|key| {
                artifact_tasks
                    .get(&(session_id.to_string(), key.clone()))
                    .map(String::as_str)
                    == Some(previous)
            }) {
                return Err(
                    "previous_task_id does not refer to an earlier task for this artifact"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    async fn task_policy(&self, task_id: &str) -> Result<Option<SummonTaskPolicy>, String> {
        let session = self
            .context
            .session_manager
            .get_session(task_id, false)
            .await
            .map_err(|error| error.to_string())?;
        Ok(SummonTaskPolicy::from_session(&session))
    }

    async fn reconstruct_artifact_tasks(
        &self,
        session_id: &str,
        owners: &mut HashMap<(String, String), String>,
        requested: &[String],
    ) -> Result<(), String> {
        let children = self
            .context
            .session_manager
            .list_subagent_sessions(session_id)
            .await
            .map_err(|error| error.to_string())?;
        let mut artifacts: HashMap<String, Vec<(String, Option<String>)>> = HashMap::new();
        for child in &children {
            let Some(policy) = SummonTaskPolicy::from_session(child) else {
                continue;
            };
            if policy.event_driven_parent {
                self.event_driven_parents
                    .lock()
                    .await
                    .insert(session_id.to_string());
            }
            if let Some(key) = policy.artifact_key {
                artifacts
                    .entry(key)
                    .or_default()
                    .push((child.id.clone(), policy.previous_task_id));
            }
        }
        for (key, tasks) in artifacts {
            if !requested.contains(&key) {
                continue;
            }
            let predecessors: HashSet<_> = tasks
                .iter()
                .filter_map(|(_, previous)| previous.as_deref())
                .collect();
            let heads: Vec<_> = tasks
                .iter()
                .filter(|(id, _)| !predecessors.contains(id.as_str()))
                .collect();
            let [head] = heads.as_slice() else {
                return Err(format!("Artifact {key} has ambiguous task ownership. Inspect its saved state before continuing."));
            };
            owners.insert((session_id.to_string(), key), head.0.clone());
        }
        Ok(())
    }

    async fn enqueue_task_completion(
        manager: &crate::session::SessionManager,
        task_id: &str,
        result: &anyhow::Result<String>,
        terminal_status: TaskTerminalStatus,
    ) -> anyhow::Result<()> {
        let artifacts = artifact_results(manager, task_id).await;
        let status = match terminal_status {
            TaskTerminalStatus::Completed => artifacts.as_ref().map_or(
                "completed successfully",
                ArtifactResultSummary::completion_description,
            ),
            TaskTerminalStatus::Cancelled => "was cancelled",
            TaskTerminalStatus::Failed => "failed",
            TaskTerminalStatus::Panicked => "panicked",
        };
        let output = match result {
            Ok(output) => output.clone(),
            Err(error) => error.to_string(),
        };
        let mut body = format!("Task {task_id} {status}.\n\n{output}");
        if let Some(artifacts) = artifacts {
            body.push_str("\n\n");
            body.push_str(&artifacts.section());
        }
        manager
            .enqueue_task_outcome(task_id, &body, terminal_status)
            .await?;
        Ok(())
    }

    async fn recover_completion_delivery(&self, task_id: &str) -> Result<(), String> {
        let pending = self
            .completed_tasks
            .lock()
            .await
            .get(task_id)
            .and_then(|task| {
                task.completion_delivery_error
                    .as_ref()
                    .map(|error| (task.result.clone(), task.terminal_status, error.clone()))
            });
        let Some((result, status, original_error)) = pending else {
            return Ok(());
        };
        let result = result.map_err(anyhow::Error::msg);
        Self::enqueue_task_completion(&self.context.session_manager, task_id, &result, status).await
            .map_err(|error| format!("Failed to deliver background task completion: {original_error}; recovery failed: {error}"))?;
        if let Some(task) = self.completed_tasks.lock().await.get_mut(task_id) {
            task.completion_delivery_error = None;
        }
        Ok(())
    }

    async fn recovered_task_result(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<TaskLoadResult, String> {
        let report = self
            .context
            .session_manager
            .terminal_report_for_child(session_id, task_id)
            .await
            .map_err(|error| error.to_string())?;
        let Some(report) = report else {
            return Ok(TaskLoadResult {
                content: vec![ContentBlock::text(format!("Task {task_id} has no canonical terminal report. Its outcome is interrupted or unknown; inspect the saved artifact before recovery. No replacement task was started."))],
                status: "unknown", turns: None, duration_secs: None,
            });
        };
        let outcome = report.task_outcome().map_err(|error| error.to_string())?;
        let status = if let Some(outcome) = outcome {
            match outcome.status {
                TaskTerminalStatus::Completed => "completed",
                TaskTerminalStatus::Cancelled => "cancelled",
                TaskTerminalStatus::Failed => "failed",
                TaskTerminalStatus::Panicked => "panicked",
            }
        } else if report
            .body
            .starts_with(&format!("Task {task_id} completed successfully."))
        {
            "completed"
        } else if report
            .body
            .starts_with(&format!("Task {task_id} was cancelled."))
        {
            "cancelled"
        } else if report.body.starts_with(&format!("Task {task_id} failed.")) {
            "failed"
        } else {
            "terminal"
        };
        Ok(TaskLoadResult {
            content: vec![ContentBlock::text(report.body)],
            status,
            turns: None,
            duration_secs: None,
        })
    }

    async fn attach_notification_emitter(
        sink: &SharedNotificationSink,
        emitter: Option<ToolCallNotificationEmitter>,
    ) {
        sink.lock().await.attach(emitter).await;
    }

    async fn run_subagent_with_notifications<Run, RunFuture>(
        sink: SharedNotificationSink,
        run_subagent: Run,
    ) -> Result<String>
    where
        Run: FnOnce(tokio::sync::mpsc::UnboundedSender<ServerNotification>) -> RunFuture,
        RunFuture: Future<Output = Result<String>>,
    {
        let (notification_tx, mut notification_rx) = tokio::sync::mpsc::unbounded_channel();
        let run = run_subagent(notification_tx);
        tokio::pin!(run);

        loop {
            tokio::select! {
                biased;
                result = &mut run => {
                    while let Ok(notification) = notification_rx.try_recv() {
                        sink.lock().await.route(notification);
                        yield_to_outer_tool_stream().await;
                    }
                    yield_to_outer_tool_stream().await;
                    return result;
                }
                Some(notification) = notification_rx.recv() => {
                    sink.lock().await.route(notification);
                    yield_to_outer_tool_stream().await;
                }
            }
        }
    }

    fn create_load_tool(&self) -> Tool {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Name of the source to load. If omitted, lists all available sources."
                },
                "cancel": {
                    "type": "boolean",
                    "default": false,
                    "description": "For running background tasks: cancel and return output."
                },
                "peek": {
                    "type": "boolean",
                    "default": false,
                    "description": "For running background tasks: check progress without blocking. Returns durable assistant-turn count, idle time, and recent tool activity."
                }
            }
        });

        Tool::new(
            "load",
            "Load knowledge into your current context or discover available sources.\n\n\
             Call with no arguments to list all available sources (subrecipes, recipes, agents).\n\
             Call with a source name to load its content into your context.\n\
             For background tasks: load(source: \"task_id\") waits for the task and returns the result.\n\
             To cancel a running task: load(source: \"task_id\", cancel: true) stops and returns output.\n\
             To check progress: load(source: \"task_id\", peek: true) returns status without blocking.\n\n\
             Examples:\n\
             - load() → Lists available sources\n\
             - load(source: \"deploy\") → Loads the deploy recipe\n\
             - load(source: \"20260219_1\") → Waits for background task, then returns result\n\
             - load(source: \"20260219_1\", peek: true) → Check task progress without waiting"
                .to_string(),
            schema.as_object().unwrap().clone(),
        )
    }

    fn create_delegate_tool(&self, instructions_required: bool) -> Tool {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": "The complete task for the delegate: the requested outcome and every requirement it must meet. Required for ad-hoc tasks and for specialist agents. The delegate starts from these instructions; send is only for steering a task that is already running."
                },
                "source": {
                    "type": "string",
                    "description": "Name of a recipe or agent to run."
                },
                "artifact_key": {
                    "type": "string",
                    "description": "The artifact ID (such as presentation-4) of an artifact that has one, including one from an earlier turn, or new:<kind> (document, spreadsheet or presentation) for a new one. The result names a new artifact's ID: use that ID for it from then on, including follow-ups, which continue its last task's saved work."
                },
                "artifact_title": {
                    "type": "string",
                    "description": "Current saved or requested title, used only as a display label. Titles do not establish artifact identity."
                },
                "reference_artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_REFERENCE_ARTIFACTS,
                    "description": "IDs of existing artifacts this task may read but not edit, frozen at their saved revisions, such as one it builds on, describes or prepares for. A specialist can read only its own artifact and these. A reference must not be an artifact a running task is editing (use a channel for that); delegating an artifact later revokes it for running tasks that reference it."
                },
                "parameters": {
                    "type": "object",
                    "additionalProperties": true,
                    "description": "Parameters for the source (only valid with source)."
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extensions to enable. Omit to inherit all, empty array for none."
                },
                "provider": {
                    "type": "string",
                    "description": "Override LLM provider."
                },
                "model": {
                    "type": "string",
                    "description": "Override model."
                },
                "temperature": {
                    "type": "number",
                    "description": "Override temperature."
                },
                "max_turns": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum turns for this delegate. Overrides recipe settings.max_turns and GOOSE_SUBAGENT_MAX_TURNS."
                },
                "context": {
                    "type": "string",
                    "description": "Reference context to inject into the delegate's system prompt. Use for background information, file contents, or constraints the delegate needs but that aren't part of the task instructions."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the delegate. Must be within the parent session's working directory. Defaults to the parent's working directory."
                },
                "async": {
                    "type": "boolean",
                    "default": false,
                    "description": "Run in background (default: false)."
                }
            }
        });
        if instructions_required {
            // Every available source is a specialist, so no delegation can
            // start without the complete task. Specialists take no recipe
            // parameters: without that field, every argument sits at the top
            // level, where the delegation reads it.
            schema["required"] = serde_json::json!(["instructions"]);
            schema["additionalProperties"] = serde_json::json!(false);
            if let Some(properties) = schema["properties"].as_object_mut() {
                properties.remove("parameters");
            }
        }

        Tool::new(
            "delegate",
            "Delegate a task to a subagent that runs independently with its own context.\n\n\
             Modes:\n\
             1. Ad-hoc: Provide `instructions` for a custom task\n\
             2. Recipe: Provide `source` name to run a subrecipe or recipe with its own prompt\n\
             3. Agent: Pair an agent source with the complete task (e.g., source: \"reviewer\", instructions: \"review the auth changes\"). Specialist agents always need instructions.\n\n\
             Effective Delegation:\n\
             - Delegates know only instructions + source content, so put the whole task in instructions\n\
             - Delegates exchange progress with the parent through native task messages; sibling delegates do not communicate directly. Same-file work can still conflict.\n\
             - Parallel: async: true. Results report back automatically; use send(task_id: task_id, message: \"...\") only to steer a running task or answer its question. load(source: task_id) can wait or inspect status. Agents marked non_blocking always return status immediately from load while running. Single: sync.\n\n\
             Research (read-only): parallelize freely - delegates explore and report back.\n\
             Work (writes): partition files strictly - no two delegates touch the same file.\n\n\
             Decompose → start async delegates → continue useful work → incorporate automatic reports."
                .to_string(),
            schema.as_object().unwrap().clone(),
        )
    }

    fn create_send_tool(&self) -> Tool {
        Tool::new(
            "send",
            "Send updated instructions or context to a running delegated task. The message is delivered at the task's next safe turn checkpoint.".to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["task_id", "message"],
                "properties": {
                    "task_id": {"type": "string", "description": "Delegated task/session ID."},
                    "message": {"type": "string", "description": "Guidance to send to the task."}
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        )
    }

    fn create_wait_tool(&self) -> Tool {
        Tool::new(
            "wait",
            "End your turn without writing to the user and sleep until the next delegated-task report, question, or user message arrives. Use it whenever you have nothing new for the user, for example right after answering a specialist's question. Available only while a delegated task is still running or a report is waiting.".to_string(),
            serde_json::json!({"type": "object", "properties": {}})
                .as_object()
                .unwrap()
                .clone(),
        )
    }

    /// End the parent's turn silently while its delegated work is outstanding.
    /// With nothing left to wait for, the agent must answer the user instead.
    async fn handle_wait(&self, session_id: &str) -> CallToolResult {
        let running = self
            .background_tasks
            .lock()
            .await
            .values()
            .any(|task| task.parent_session_id == session_id && !task.handle.is_finished());
        let reports_waiting = self
            .context
            .session_manager
            .pending_session_messages(session_id)
            .await
            .map(|messages| !messages.is_empty())
            .unwrap_or(false);
        if !running && !reports_waiting {
            return CallToolResult::error(vec![ContentBlock::text(
                "Error: No delegated task is running and no report is waiting. Every task has reported: write your answer to the user now.",
            )]);
        }
        let mut result = CallToolResult::success(vec![ContentBlock::text(
            "Waiting. You will be resumed with the next report or question.",
        )]);
        result.meta = Some(MetaObject(
            serde_json::json!({ END_TURN_META_KEY: true })
                .as_object()
                .unwrap()
                .clone(),
        ));
        result
    }

    /// The specialist wait tool; channel rules only for a task in a channel.
    fn create_specialist_wait_tool(&self, in_channel: bool) -> Tool {
        let description = if in_channel {
            "Wait for an outstanding dependency: an editor task you started, the first publication from a live member you follow, a reply to your open channel question/request, or an answer to a successfully sent message_parent question. Already-pending coordinator or waking channel news is delivered immediately. What woke you arrives after this result. A channel membership, followers, an unrelated tool call, or a plan you already received does not permit waiting. If you need a further publication, ask its owner with channel_post kind ask first. Never wait for followers, rendering by another member, or finished members. Once your editor results have been reviewed and channel obligations answered, write your final report. On timeout, wait again only for a dependency still outstanding."
        } else {
            "Wait for an outstanding dependency: an editor task you started, or an answer to a successfully sent message_parent question. Already-pending coordinator news is delivered immediately. What woke you arrives after this result. An unrelated tool call does not permit waiting. Once your editor results have been reviewed, write your final report. On timeout, wait again only for a dependency still outstanding."
        };
        Tool::new(
            "wait",
            description.to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": SPECIALIST_WAIT_MAX_SECS,
                        "description": format!("Longest wait in seconds (default {SPECIALIST_WAIT_DEFAULT_SECS}).")
                    }
                },
                "additionalProperties": false
            })
            .as_object()
            .unwrap()
            .clone(),
        )
    }

    /// The reminder a source asks for, once per parent session, when two or more of
    /// the parent's artifact tasks are running and it has not called `tool`.
    async fn connect_reminder(
        &self,
        session_id: &str,
        tool: &str,
        reminder: String,
    ) -> Option<String> {
        let running = {
            // The same lock order as delegation: artifact tasks, then background tasks.
            let artifact_tasks = self.artifact_tasks.lock().await;
            let tasks = self.background_tasks.lock().await;
            artifact_tasks
                .iter()
                .filter(|((parent, _), task_id)| {
                    parent == session_id
                        && tasks
                            .get(*task_id)
                            .is_some_and(|task| !task.handle.is_finished())
                })
                .map(|(_, task_id)| task_id.clone())
                .collect::<HashSet<_>>()
                .len()
        };
        if running < 2 || self.connect_reminded.lock().await.contains(session_id) {
            return None;
        }
        let session = self
            .context
            .session_manager
            .get_session(session_id, true)
            .await
            .ok()?;
        let called = session.conversation.as_ref().is_some_and(|conversation| {
            conversation.messages().iter().any(|message| {
                message.content.iter().any(|content| match content {
                    MessageContent::ToolRequest(request) => request
                        .tool_call
                        .as_ref()
                        .is_ok_and(|call| call.name == tool),
                    _ => false,
                })
            })
        });
        if called {
            return None;
        }
        self.connect_reminded
            .lock()
            .await
            .insert(session_id.to_string());
        Some(reminder)
    }

    fn parent_reply_pending(messages: &[Message]) -> bool {
        let mut questions = HashSet::new();
        let mut pending = false;
        for message in messages {
            if message
                .metadata
                .operation_note(crate::session::MAILBOX_NOTE, "kind")
                .and_then(serde_json::Value::as_str)
                == Some("message")
            {
                pending = false;
                questions.clear();
            }
            for content in &message.content {
                match content {
                    MessageContent::ToolRequest(request)
                        if request
                            .tool_call
                            .as_ref()
                            .is_ok_and(|call| call.name == "message_parent") =>
                    {
                        questions.insert(request.id.as_str());
                    }
                    MessageContent::ToolResponse(response)
                        if questions.contains(response.id.as_str())
                            && response
                                .tool_result
                                .as_ref()
                                .is_ok_and(|result| result.is_error != Some(true)) =>
                    {
                        pending = true;
                    }
                    _ => {}
                }
            }
        }
        pending
    }

    fn channel_wait_contexts(messages: &[Message]) -> HashMap<String, ChannelWaitContext> {
        let mut contexts: HashMap<String, ChannelWaitContext> = HashMap::new();
        let mut record = |context: ChannelWaitContext| {
            // A queued notice may be delivered after a newer MCP read. The log
            // position orders transport snapshots, not facet/artifact freshness.
            if contexts
                .get(&context.channel_id)
                .is_none_or(|current| context.sequence >= current.sequence)
            {
                contexts.insert(context.channel_id.clone(), context);
            }
        };
        for message in messages {
            if let Some(value) = message.metadata.operation_note(
                crate::session::MAILBOX_NOTE,
                crate::session::CHANNEL_WAIT_META_KEY,
            ) {
                if let Ok(context) = serde_json::from_value(value.clone()) {
                    record(context);
                }
            }
            for content in &message.content {
                if let MessageContent::ToolResponse(response) = content {
                    if let Ok(result) = &response.tool_result {
                        if result.is_error == Some(true) {
                            continue;
                        }
                        if let Some(value) = result
                            .meta
                            .as_ref()
                            .and_then(|meta| meta.0.get(crate::session::CHANNEL_WAIT_META_KEY))
                        {
                            if let Ok(updates) =
                                serde_json::from_value::<Vec<ChannelWaitContext>>(value.clone())
                            {
                                for context in updates {
                                    record(context);
                                }
                            }
                        }
                    }
                }
            }
        }
        contexts
    }

    /// Sleep until the specialist's mailbox holds something that should wake
    /// it: an editor result, a coordinator message, or waking channel news.
    /// Goose delivers it at the checkpoint right after this result.
    async fn handle_specialist_wait(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
        cancellation_token: CancellationToken,
    ) -> CallToolResult {
        let timeout = arguments
            .as_ref()
            .and_then(|args| args.get("timeout_s"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(SPECIALIST_WAIT_DEFAULT_SECS)
            .clamp(1, SPECIALIST_WAIT_MAX_SECS);
        let manager = &self.context.session_manager;
        let session = match manager.get_session(session_id, true).await {
            Ok(session) => session,
            Err(error) => {
                return CallToolResult::error(vec![ContentBlock::text(format!("Error: {error}"))])
            }
        };
        let messages = session
            .conversation
            .as_ref()
            .map(|conversation| conversation.messages().to_vec())
            .unwrap_or_default();
        let running = running_editor_tasks(&session, &messages);
        let channel_contexts = Self::channel_wait_contexts(&messages);
        let in_channel = !channel_contexts.is_empty();
        let channel_waiting = channel_contexts
            .values()
            .any(ChannelWaitContext::is_waiting);
        let parent_waiting = Self::parent_reply_pending(&messages);
        let started = Instant::now();
        loop {
            match manager.pending_session_messages(session_id).await {
                Ok(pending) if pending.iter().any(|message| message.interrupts_wait()) => {
                    return CallToolResult::success(vec![ContentBlock::text(format!(
                        "Woken after {}s: what arrived follows this result. Act on it, then call wait again while you still wait for something.",
                        started.elapsed().as_secs()
                    ))]);
                }
                Ok(_) => {}
                Err(error) => {
                    return CallToolResult::error(vec![ContentBlock::text(format!(
                        "Error: {error}"
                    ))])
                }
            }
            if running.is_empty() && !channel_waiting && !parent_waiting {
                // Channel wording only for a task that is in a channel.
                let text = if in_channel {
                    "Error: Nothing to wait for: no editor task of yours is running, no first upstream publication is outstanding, and no channel question/request or successfully sent parent question awaits a reply. Channel membership and followers do not permit waiting. If you need a further publication, ask its owner with channel_post kind ask; otherwise finish your review, answer any channel obligations, and write your final report."
                } else {
                    "Error: Nothing to wait for: no editor task of yours is running and no successfully sent parent question awaits a reply. Finish your review and write your final report."
                };
                return crate::agents::gen_ai_telemetry::rule_rejection(text);
            }
            if started.elapsed() >= Duration::from_secs(timeout) {
                let still = if running.is_empty() {
                    " Your publication or reply dependency is still outstanding; do independent work if any remains.".to_string()
                } else {
                    format!(" Editor tasks still running: {}.", running.join(", "))
                };
                return CallToolResult::success(vec![ContentBlock::text(format!(
                    "Nothing arrived in {timeout}s.{still} Call wait again only while something you wait for can still arrive (an editor result, a reply to your question, or a publish you follow); otherwise write your final report."
                ))]);
            }
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    return CallToolResult::error(vec![ContentBlock::text("Error: The wait was cancelled.")]);
                }
                _ = tokio::time::sleep(SPECIALIST_WAIT_POLL) => {}
            }
        }
    }

    fn create_message_parent_tool(&self) -> Tool {
        Tool::new(
            "message_parent",
            "Ask the parent one concrete question without ending this task. Use it only in exceptional cases: when you cannot continue without an answer that only the parent or the user can give. The parent replies through your task. Never use it for progress, results, warnings, or limitations; your final report, including any limitation, is delivered automatically. A message that asks no question is refused.".to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {"type": "string", "description": "The question the parent must answer before you can continue, with the context needed to answer it."}
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        )
    }

    async fn get_working_dir(&self, session_id: &str) -> PathBuf {
        self.context
            .session_manager
            .get_session(session_id, false)
            .await
            .ok()
            .map(|s| s.working_dir)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    async fn get_sources(&self, session_id: &str, working_dir: &Path) -> Vec<SourceEntry> {
        let fs_sources = self.get_filesystem_sources(working_dir).await;

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut sources: Vec<SourceEntry> = Vec::new();

        self.add_subrecipes(session_id, &mut sources, &mut seen)
            .await;

        for source in fs_sources {
            if !seen.contains(&source.name) {
                seen.insert(source.name.clone());
                sources.push(source);
            }
        }

        sources.sort_by(|a, b| (&a.source_type, &a.name).cmp(&(&b.source_type, &b.name)));
        sources
    }

    async fn get_filesystem_sources(&self, working_dir: &Path) -> Vec<SourceEntry> {
        let mut cache = self.source_cache.lock().await;
        if let Some((cached_at, cached_dir, sources)) = cache.as_ref() {
            if cached_dir == working_dir && cached_at.elapsed() < Duration::from_secs(60) {
                return sources.clone();
            }
        }
        let sources = self.discover_filesystem_sources(working_dir);
        *cache = Some((Instant::now(), working_dir.to_path_buf(), sources.clone()));
        sources
    }

    async fn resolve_source(
        &self,
        session_id: &str,
        name: &str,
        working_dir: &Path,
    ) -> Result<Option<SourceEntry>, String> {
        let sources = self.get_sources(session_id, working_dir).await;

        Ok(sources.iter().find(|s| s.name == name).cloned())
    }

    async fn load_subrecipe_content(&self, session_id: &str, name: &str) -> Result<String, String> {
        let session = match self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
        {
            Ok(s) => s,
            Err(_) => return Ok(String::new()),
        };

        let sub_recipes = match session.recipe.as_ref().and_then(|r| r.sub_recipes.as_ref()) {
            Some(sr) => sr,
            None => return Ok(String::new()),
        };

        let sr = match sub_recipes.iter().find(|sr| sr.name == name) {
            Some(sr) => sr,
            None => return Ok(String::new()),
        };

        match load_local_recipe_file(&sr.path) {
            Ok(recipe_file) => Self::format_subrecipe_content(name, &recipe_file.content),
            Err(_) => Ok(String::new()),
        }
    }

    fn format_subrecipe_content(name: &str, raw_content: &str) -> Result<String, String> {
        let recipe = Recipe::from_content(raw_content)
            .map_err(|_| format!("Subrecipe '{}' is not a valid recipe", name))?;
        let mut content = recipe.instructions.unwrap_or_default();
        if let Some(params) = &recipe.parameters {
            if !params.is_empty() {
                content.push_str("\n\n");
                content.push_str(&Self::format_parameters(params));
            }
        }
        Ok(content)
    }

    fn discover_filesystem_sources(&self, working_dir: &Path) -> Vec<SourceEntry> {
        discover_filesystem_sources(working_dir)
    }

    async fn add_subrecipes(
        &self,
        session_id: &str,
        sources: &mut Vec<SourceEntry>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        let session = match self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
        {
            Ok(s) => s,
            Err(_) => return,
        };

        let sub_recipes = match session.recipe.as_ref().and_then(|r| r.sub_recipes.as_ref()) {
            Some(sr) => sr,
            None => return,
        };

        for sr in sub_recipes {
            if seen.contains(&sr.name) {
                continue;
            }
            seen.insert(sr.name.clone());

            let description = self.build_subrecipe_description(sr).await;

            sources.push(SourceEntry {
                source_type: SourceType::Subrecipe,
                name: sr.name.clone(),
                description,
                content: String::new(),
                path: sr.path.clone(),
                global: false,
                writable: true,
                supporting_files: Vec::new(),
                properties: std::collections::HashMap::new(),
            });
        }
    }

    async fn build_subrecipe_description(&self, sr: &crate::recipe::SubRecipe) -> String {
        if let Some(desc) = &sr.description {
            return desc.clone();
        }

        if let Ok(recipe_file) = load_local_recipe_file(&sr.path) {
            if let Ok(recipe) = Recipe::from_content(&recipe_file.content) {
                let mut desc = recipe.description.clone();

                if let Some(params) = &recipe.parameters {
                    if !params.is_empty() {
                        desc = format!("{}\n{}", desc, Self::format_parameters(params));
                    }
                }

                return desc;
            }
        }

        format!("Subrecipe from {}", sr.path)
    }

    fn format_parameters(params: &[RecipeParameter]) -> String {
        let mut out = String::from("Parameters:");
        for p in params {
            let mut detail = format!("\n  - {} ({}, {})", p.key, p.input_type, p.requirement);
            if let Some(default) = &p.default {
                detail.push_str(&format!(", default: \"{}\"", default));
            }
            if let Some(options) = &p.options {
                if !options.is_empty() {
                    detail.push_str(&format!(", options: [{}]", options.join(", ")));
                }
            }
            detail.push_str(&format!(": {}", p.description));
            out.push_str(&detail);
        }
        out
    }

    async fn handle_load(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
        notification_emitter: Option<ToolCallNotificationEmitter>,
    ) -> Result<CallToolResult, String> {
        self.cleanup_completed_tasks().await;

        let source_name = arguments
            .as_ref()
            .and_then(|args| args.get("source"))
            .and_then(|v| v.as_str());

        let cancel = arguments
            .as_ref()
            .and_then(|args| args.get("cancel"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let peek = arguments
            .as_ref()
            .and_then(|args| args.get("peek"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let working_dir = self.get_working_dir(session_id).await;

        if source_name.is_none() {
            return self
                .handle_load_discovery(session_id, &working_dir)
                .await
                .map(CallToolResult::success);
        }

        let name = source_name.unwrap();

        if is_session_id(name) {
            let running_owner = self
                .background_tasks
                .lock()
                .await
                .get(name)
                .map(|task| task.parent_session_id.clone());
            let completed_owner = self
                .completed_tasks
                .lock()
                .await
                .get(name)
                .map(|task| task.parent_session_id.clone());
            if running_owner
                .or(completed_owner)
                .is_some_and(|owner| !owner.is_empty() && owner != session_id)
            {
                return Err(format!("Task '{name}' does not belong to this session"));
            }
            let running = self.background_tasks.lock().await.contains_key(name);
            let cached = self.completed_tasks.lock().await.contains_key(name);
            if running
                && !cancel
                && (self.event_driven_parents.lock().await.contains(session_id)
                    || self
                        .task_policy(name)
                        .await?
                        .is_some_and(|policy| policy.event_driven_parent))
            {
                return Err("This task reports automatically. Call wait to sleep until it reports; use send to steer it or load(cancel: true) to stop it.".to_string());
            }
            let task_result = if running || cached {
                self.recover_completion_delivery(name).await?;
                self.handle_load_task_result(name, cancel, peek, notification_emitter)
                    .await?
            } else {
                self.recovered_task_result(session_id, name).await?
            };
            let mut meta = MetaObject::new();
            meta.0.insert(
                "subagent_session_id".to_string(),
                serde_json::Value::String(name.to_string()),
            );
            meta.0.insert(
                "task_status".to_string(),
                serde_json::Value::String(task_result.status.to_string()),
            );
            if let Some(turns) = task_result.turns {
                meta.0.insert(
                    "turns_taken".to_string(),
                    serde_json::Value::Number(turns.into()),
                );
            }
            if let Some(secs) = task_result.duration_secs {
                meta.0.insert(
                    "duration_secs".to_string(),
                    serde_json::Value::Number(secs.into()),
                );
            }
            let mut result = CallToolResult::success(task_result.content).with_meta(Some(meta));
            result.structured_content = Some(serde_json::json!({
                "subagent_session_id": name,
                "task_status": task_result.status
            }));
            return Ok(result);
        }

        self.handle_load_source(session_id, name, &working_dir)
            .await
            .map(CallToolResult::success)
    }

    async fn handle_load_task_result(
        &self,
        task_id: &str,
        cancel: bool,
        peek: bool,
        notification_emitter: Option<ToolCallNotificationEmitter>,
    ) -> Result<TaskLoadResult, String> {
        let mut completed = self.completed_tasks.lock().await;

        let completed_entry = completed.get(task_id).map(|task| {
            (
                task.result.clone(),
                task.completion_delivery_error.clone(),
                task.description.clone(),
                task.duration,
                task.turns_taken,
                Arc::clone(&task.notification_sink),
            )
        });

        if let Some((
            result,
            completion_delivery_error,
            description,
            duration,
            turns_taken,
            notification_sink,
        )) = completed_entry
        {
            if let Some(error) = completion_delivery_error {
                Self::attach_notification_emitter(&notification_sink, notification_emitter).await;
                return Err(format!(
                    "Task '{task_id}' completed, but its parent report could not be delivered: {error}"
                ));
            }
            if !peek {
                Self::attach_notification_emitter(&notification_sink, notification_emitter).await;
                completed.remove(task_id);
            }
            let status_key = match &result {
                Ok(_) => "completed",
                Err(e) if e.starts_with("Task panicked:") => "panicked",
                Err(e) if e.starts_with("Task was cancelled:") => "cancelled",
                Err(_) => "failed",
            };
            let status = match status_key {
                "completed" => "✓ Completed",
                "panicked" => "✗ Panicked",
                "cancelled" => "⊘ Cancelled",
                _ => "✗ Failed",
            };
            let output = match result {
                Ok(output) => output,
                Err(error) => format!("Error: {}", error),
            };
            return Ok(TaskLoadResult {
                content: vec![ContentBlock::text(format!(
                    "# Background Task Result: {}\n\n\
                     **Task:** {}\n\
                     **Status:** {}\n\
                     **Duration:** {} ({} turns)\n\n\
                     ## Output\n\n{}",
                    task_id,
                    description,
                    status,
                    round_duration(duration),
                    turns_taken,
                    output
                ))],
                status: status_key,
                turns: Some(turns_taken),
                duration_secs: Some(duration.as_secs()),
            });
        }

        let running = self.background_tasks.lock().await;
        drop(completed);
        if running.contains_key(task_id) {
            if !cancel && (peek || running.get(task_id).unwrap().non_blocking) {
                let task = running.get(task_id).unwrap();
                let elapsed = task.started_at.elapsed();
                let turns = Arc::clone(&task.turns);
                let last_activity = Arc::clone(&task.last_activity);
                let description = task.description.clone();
                let non_blocking = task.non_blocking;
                let notification_sink = Arc::clone(&task.notification_sink);

                drop(running);

                let turns_taken = self.refresh_task_turns(task_id, &turns).await;
                let now = current_epoch_millis();
                let last_activity_at = last_activity.load(Ordering::Relaxed);
                let idle_ms = if last_activity_at == 0 {
                    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
                } else {
                    now.saturating_sub(last_activity_at)
                };
                let buffered_count = notification_sink.lock().await.buffered_len();

                let mut output = format!(
                    "# Background Task Status: {}\n\n**Task:** {}\n**Status:** ⏳ Running\n**Elapsed:** {}\n**Turns taken:** {}\n**Idle:** {}\n**Buffered tool calls:** {}",
                    task_id,
                    description,
                    round_duration(elapsed),
                    turns_taken,
                    round_duration(Duration::from_millis(idle_ms)),
                    buffered_count,
                );

                if buffered_count == 0 && last_activity_at == 0 {
                    output.push_str("\n\n_Task is initialising (no tool activity yet)._");
                }

                if non_blocking {
                    output.push_str("\n\nCompletion arrives automatically. Do not poll or sleep; finish your reply when no independent work remains. Use send to steer this task.");
                }

                return Ok(TaskLoadResult {
                    content: vec![ContentBlock::text(output)],
                    status: "running",
                    turns: Some(turns_taken),
                    duration_secs: Some(elapsed.as_secs()),
                });
            }

            if cancel {
                if running.get(task_id).unwrap().handle.is_finished() {
                    drop(running);
                    self.cleanup_completed_tasks().await;
                    return Box::pin(self.handle_load_task_result(
                        task_id,
                        false,
                        peek,
                        notification_emitter,
                    ))
                    .await;
                }
                let task = running.get(task_id).unwrap();
                let notification_sink = Arc::clone(&task.notification_sink);
                let completion_token = task.completion_token.clone();
                let cancellation_token = task.cancellation_token.clone();
                drop(running);
                Self::attach_notification_emitter(&notification_sink, notification_emitter).await;
                cancellation_token.cancel();
                let aborted = tokio::time::timeout(
                    Duration::from_secs(5),
                    self.wait_for_background_task_completion(task_id, &completion_token),
                )
                .await
                .is_err();
                if aborted {
                    if let Some(task) = self.background_tasks.lock().await.get(task_id) {
                        task.handle.abort();
                    }
                    self.wait_for_background_task_completion(task_id, &completion_token)
                        .await;
                }
                self.cleanup_completed_tasks().await;
                let completed = self.completed_tasks.lock().await;
                let Some(task) = completed.get(task_id) else {
                    return Err(format!("Cancelled task '{task_id}' has no terminal result"));
                };
                let output = if aborted {
                    "Task did not stop in time (aborted)".to_string()
                } else {
                    task.result
                        .clone()
                        .unwrap_or_else(|error| format!("Error: {error}"))
                };
                let parent = task.parent_session_id.clone();
                let description = task.description.clone();
                let duration = task.duration;
                let turns_taken = task.turns_taken;
                drop(completed);
                let output =
                    match artifact_results_section(&self.context.session_manager, task_id).await {
                        Some(section) => format!("{output}\n\n{section}"),
                        None => output,
                    };
                if !parent.is_empty() {
                    if let Err(error) = self
                        .context
                        .session_manager
                        .enqueue_task_outcome(
                            task_id,
                            &format!("Task {task_id} was cancelled.\n\n{output}"),
                            TaskTerminalStatus::Cancelled,
                        )
                        .await
                    {
                        let error = error.to_string();
                        if let Some(task) = self.completed_tasks.lock().await.get_mut(task_id) {
                            task.completion_delivery_error = Some(error.clone());
                        }
                        return Err(format!("Task '{task_id}' was cancelled, but its parent report could not be delivered: {error}"));
                    }
                    if let Some(task) = self.completed_tasks.lock().await.get_mut(task_id) {
                        task.completion_delivery_error = None;
                    }
                }
                return Ok(TaskLoadResult {
                    content: vec![ContentBlock::text(format!(
                        "# Background Task Result: {task_id}\n\n**Task:** {description}\n**Status:** ⊘ Cancellation requested\n**Duration:** {} ({} turns)\n\nCancellation was requested and the task stopped. If the task finished concurrently, its automatic completion report is authoritative.\n\n## Output\n\n{output}",
                        round_duration(duration), turns_taken,
                    ))],
                    status: "cancellation_requested", turns: Some(turns_taken), duration_secs: Some(duration.as_secs()),
                });
            }

            // Wait for the running task to complete, keeping the tool call
            // alive so notifications (subagent tool calls) stream in real time.
            let task = running.get(task_id).unwrap();
            let notification_sink = Arc::clone(&task.notification_sink);
            let completion_token = task.completion_token.clone();
            drop(running);
            Self::attach_notification_emitter(&notification_sink, notification_emitter).await;

            tokio::select! {
                _ = self.wait_for_background_task_completion(task_id, &completion_token) => {
                    self.cleanup_completed_tasks().await;
                    return Box::pin(
                        self.handle_load_task_result(task_id, false, false, None)
                    )
                    .await;
                }
                _ = tokio::time::sleep(Duration::from_secs(300)) => {
                    notification_sink.lock().await.detach();

                    return Err(format!(
                        "Task '{task_id}' is still running after waiting 5 min. \
                         Use load(source: \"{task_id}\") to wait again, or \
                         load(source: \"{task_id}\", cancel: true) to stop."
                    ));
                }
            }
        }

        Err(format!("Task '{}' not found.", task_id))
    }

    async fn handle_load_discovery(
        &self,
        session_id: &str,
        working_dir: &Path,
    ) -> Result<Vec<ContentBlock>, String> {
        {
            let mut cache = self.source_cache.lock().await;
            *cache = None;
        }

        let sources = self.get_sources(session_id, working_dir).await;
        let completed = self.completed_tasks.lock().await;

        if sources.is_empty() && completed.is_empty() {
            return Ok(vec![ContentBlock::text(
                "No sources available for load/delegate.\n\n\
                 Sources are discovered from:\n\
                 • Current recipe's sub_recipes\n\
                 • .agents/recipes/, .agents/agents/ (project-level)\n\
                 • ~/.agents/agents/ (global)\n\
                 • GOOSE_RECIPE_PATH directories",
            )]);
        }

        let mut output = String::from("Available sources for load/delegate:\n");

        if !completed.is_empty() {
            output.push_str("\nCompleted Tasks (awaiting retrieval):\n");
            let mut sorted_completed: Vec<_> = completed.values().collect();
            sorted_completed.sort_by_key(|t| &t.id);
            for task in sorted_completed {
                let status = if task.result.is_ok() {
                    "completed"
                } else {
                    "failed"
                };
                output.push_str(&format!(
                    "• {} - \"{}\" ({})\n",
                    task.id, task.description, status
                ));
            }
        }

        for kind in [SourceType::Subrecipe, SourceType::Recipe, SourceType::Agent] {
            let kind_sources: Vec<_> = sources.iter().filter(|s| s.source_type == kind).collect();
            if !kind_sources.is_empty() {
                output.push_str(&format!("\n{}:\n", kind_plural(kind)));
                for source in kind_sources {
                    output.push_str(&format!(
                        "• {} - {}\n",
                        source.name,
                        safe_truncate(&source.description, SUBAGENT_DESCRIPTION_BUDGET)
                    ));
                }
            }
        }

        output.push_str("\nUse load(source: \"name\") to load into context.\n");
        output.push_str("Use delegate(source: \"name\") to run as subagent.");

        Ok(vec![ContentBlock::text(output)])
    }

    async fn handle_load_source(
        &self,
        session_id: &str,
        name: &str,
        working_dir: &Path,
    ) -> Result<Vec<ContentBlock>, String> {
        let source = self.resolve_source(session_id, name, working_dir).await?;

        match source {
            Some(mut source) => {
                if source.source_type == SourceType::Agent {
                    return Err(format!(
                        "Agent '{}' is delegate-only and cannot be loaded as context",
                        source.name
                    ));
                }
                if source.source_type == SourceType::Subrecipe && source.content.is_empty() {
                    source.content = self
                        .load_subrecipe_content(session_id, &source.name)
                        .await?;
                }
                let content = source.to_load_text();

                let output = format!(
                    "# Loaded: {} ({})\n\n{}\n\n---\nThis knowledge is now available in your context.",
                    source.name, source.source_type, content
                );

                Ok(vec![ContentBlock::text(output)])
            }
            None => {
                let sources = self.get_sources(session_id, working_dir).await;

                let suggestions: Vec<&str> = sources
                    .iter()
                    .filter(|s| {
                        s.name.to_lowercase().contains(&name.to_lowercase())
                            || name.to_lowercase().contains(&s.name.to_lowercase())
                    })
                    .take(3)
                    .map(|s| s.name.as_str())
                    .collect();

                let error_msg = if suggestions.is_empty() {
                    format!(
                        "Source '{}' not found. Use load() to see available sources.",
                        name
                    )
                } else {
                    format!(
                        "Source '{}' not found. Did you mean: {}?",
                        name,
                        suggestions.join(", ")
                    )
                };

                Err(error_msg)
            }
        }
    }

    async fn handle_delegate(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
        cancellation_token: CancellationToken,
        notification_emitter: Option<ToolCallNotificationEmitter>,
    ) -> Result<CallToolResult, String> {
        self.cleanup_completed_tasks().await;

        let mut params: DelegateParams = arguments
            .map(|args| serde_json::from_value(serde_json::Value::Object(args)))
            .transpose()
            .map_err(|e| format!("Invalid parameters: {}", e))?
            .unwrap_or_default();
        // An empty `parameters: {}` passes nothing; treat it as absent.
        if params.parameters.as_ref().is_some_and(HashMap::is_empty) {
            params.parameters = None;
        }

        self.validate_delegate_params(&params)?;

        let session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|e| format!("Failed to get session: {}", e))?;

        if session.session_type == SessionType::SubAgent {
            return Err("Delegated tasks cannot spawn further delegations".to_string());
        }

        let agent_source = if let Some(source_name) = params.source.as_deref() {
            self.resolve_source(session_id, source_name, &session.working_dir)
                .await?
                .filter(|source| source.source_type == SourceType::Agent)
        } else {
            None
        };
        Self::require_specialist_instructions(&params, agent_source.as_ref())?;
        let force_async = agent_source.as_ref().is_some_and(|source| {
            source
                .properties
                .get("always_async")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                || source
                    .properties
                    .get("artifact_guard")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                || source
                    .properties
                    .get("event_driven_parent")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                || source
                    .properties
                    .get("non_blocking")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
        });

        if params.r#async || force_async {
            let (content, task_id) = self.handle_async_delegate(session_id, params).await?;
            let mut meta = MetaObject::new();
            meta.0.insert(
                "subagent_session_id".to_string(),
                serde_json::Value::String(task_id.clone()),
            );
            let admission = self
                .context
                .session_manager
                .task_admission(&task_id)
                .await
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "Async task has no immutable admission".to_string())?;
            meta.0.insert(
                "taskAdmission".to_string(),
                serde_json::to_value(&admission).map_err(|error| error.to_string())?,
            );
            let mut result = CallToolResult::success(content).with_meta(Some(meta));
            result.structured_content = Some(serde_json::json!({
                "subagent_session_id": task_id,
                "task_status": "running",
                "taskAdmission": admission,
            }));
            return Ok(result);
        }

        let working_dir = session.working_dir.clone();
        let mut params = params;
        params.previous_task_id = None;
        let recipe = self
            .build_delegate_recipe(&params, session_id, &working_dir)
            .await?;

        let task_config = self
            .build_task_config(&params, &recipe, &session)
            .await
            .map_err(|e| format!("Failed to build task config: {}", e))?;

        // Subagents must use Auto until get_agent_messages forwards
        // ActionRequired messages to the parent. Until then, any mode
        // that requires approval will hang on the subagent's confirmation_rx.
        let mut agent_config = AgentConfig::new(
            self.context.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true, // disable session naming for subagents
            crate::agents::GoosePlatform::GooseCli,
        )
        .with_use_login_shell_path(self.context.use_login_shell_path);
        agent_config.is_subagent = true;

        let subagent_session = self
            .create_subagent_session(&task_config, "Delegated task".to_string(), None, "inline")
            .await?;

        let subagent_session_id = subagent_session.id.clone();
        let telemetry_session_id =
            crate::session_context::telemetry_session_id(&task_config.parent_session_id);

        let params = SubagentRunParams {
            config: agent_config,
            recipe,
            task_config,
            return_last_only: true,
            session_id: subagent_session.id,
            cancellation_token: Some(cancellation_token),
            on_message: None,
            notification_tx: None,
            parent_span: tracing::Span::current(),
            telemetry_session_id,
        };
        let result = Self::run_subagent_with_notifications(
            Self::notification_sink(notification_emitter),
            move |notification_tx| {
                let mut params = params;
                params.notification_tx = Some(notification_tx);
                run_subagent_task(params)
            },
        )
        .await;

        let mut meta = MetaObject::new();
        meta.0.insert(
            "subagent_session_id".to_string(),
            serde_json::Value::String(subagent_session_id),
        );

        match result {
            Ok(text) => {
                Ok(CallToolResult::success(vec![ContentBlock::text(text)]).with_meta(Some(meta)))
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Delegation failed: {}",
                e
            ))])
            .with_meta(Some(meta))),
        }
    }

    async fn send_acknowledgement(&self, session_id: &str, task_id: &str) -> String {
        let queued = format!("Message queued for task {task_id}.");
        if !self.event_driven_parents.lock().await.contains(session_id) {
            return queued;
        }
        let waiting = self
            .context
            .session_manager
            .pending_session_messages(task_id)
            .await
            .map(|pending| {
                pending
                    .iter()
                    .filter(|message| {
                        message.kind == crate::session::MailboxMessageKind::Message
                            && message.sender_session_id == session_id
                    })
                    .count()
            })
            .unwrap_or(1);
        let next = if self.event_driven_parents.lock().await.contains(session_id) {
            "Call wait without writing to the user unless you still have independent work or news for them: \
             you are resumed automatically when a task reports or asks for something."
        } else {
            "Finish your reply unless you still have independent work: you are resumed automatically when a task reports or asks for something."
        };
        format!(
            "{queued} The task receives it at its next checkpoint; undelivered messages from you to this task: {waiting}. {next}"
        )
    }

    fn only_specialist_sources(sources: &[SourceEntry]) -> bool {
        !sources.is_empty() && sources.iter().all(Self::is_specialist_source)
    }

    fn is_specialist_source(source: &SourceEntry) -> bool {
        ["artifact_guard", "event_driven_parent"]
            .iter()
            .any(|flag| {
                source
                    .properties
                    .get(*flag)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
    }

    fn require_specialist_instructions(
        params: &DelegateParams,
        agent_source: Option<&SourceEntry>,
    ) -> Result<(), String> {
        let specialist =
            params.artifact_key.is_some() || agent_source.is_some_and(Self::is_specialist_source);
        let has_instructions = params
            .instructions
            .as_deref()
            .is_some_and(|instructions| !instructions.trim().is_empty());
        if specialist && params.parameters.is_some() {
            return Err(
                "Specialists take no parameters: pass instructions, artifact_key and the other fields at the top level of delegate."
                    .to_string(),
            );
        }
        if specialist && !has_instructions {
            return Err(format!(
                "Delegation to {} requires instructions with the complete task: the requested outcome and the user's requirements for this deliverable, plus any Research-dependent and Evidence IDs lines. The specialist starts from these instructions; send is only for steering it after it is running.",
                params.source.as_deref().unwrap_or("a specialist"),
            ));
        }
        Ok(())
    }

    fn validate_delegate_params(&self, params: &DelegateParams) -> Result<(), String> {
        if let Some(error) = nested_delegate_fields_error(params) {
            return Err(error);
        }
        if params.instructions.is_none() && params.source.is_none() {
            return Err("Must provide 'instructions' or 'source' (or both)".to_string());
        }

        if params.parameters.is_some() && params.source.is_none() {
            return Err("'parameters' can only be used with 'source'".to_string());
        }

        if !params.reference_artifacts.is_empty() && !params.r#async {
            return Err("reference_artifacts is only for asynchronous specialist delegations (async: true).".to_string());
        }

        if let Some(max) = params.max_turns {
            if max < 1 {
                return Err("'max_turns' must be at least 1".to_string());
            }
        }

        Ok(())
    }

    async fn build_delegate_recipe(
        &self,
        params: &DelegateParams,
        session_id: &str,
        working_dir: &Path,
    ) -> Result<Recipe, String> {
        let mut recipe = if let Some(source_name) = &params.source {
            self.build_source_recipe(source_name, params, session_id, working_dir)
                .await?
        } else {
            self.build_adhoc_recipe(params)?
        };

        if let Some(ref context) = params.context {
            let existing = recipe.instructions.unwrap_or_default();
            recipe.instructions = Some(build_instructions_with_context(context, &existing));
        }

        Ok(recipe)
    }

    fn build_adhoc_recipe(&self, params: &DelegateParams) -> Result<Recipe, String> {
        let task = params
            .instructions
            .as_ref()
            .ok_or("Instructions required for ad-hoc task")?;

        Recipe::builder()
            .version("1.0.0")
            .title("Delegated Task")
            .description("Ad-hoc delegated task")
            .prompt(task)
            .build()
            .map_err(|e| format!("Failed to build recipe: {}", e))
    }

    async fn build_source_recipe(
        &self,
        source_name: &str,
        params: &DelegateParams,
        session_id: &str,
        working_dir: &Path,
    ) -> Result<Recipe, String> {
        let source = match self
            .resolve_source(session_id, source_name, working_dir)
            .await?
        {
            Some(source) => source,
            None => {
                let available: Vec<String> = self
                    .get_sources(session_id, working_dir)
                    .await
                    .into_iter()
                    .filter(|source| {
                        matches!(
                            source.source_type,
                            SourceType::Agent | SourceType::Recipe | SourceType::Subrecipe
                        )
                    })
                    .map(|source| source.name)
                    .collect();
                return Err(unknown_delegate_source_error(source_name, &available));
            }
        };

        let mut recipe = match source.source_type {
            SourceType::Recipe | SourceType::Subrecipe => {
                self.build_recipe_from_source(&source, params, session_id)
                    .await?
            }
            SourceType::Agent => self.build_recipe_from_agent(&source, params, working_dir)?,
            _ => {
                return Err(format!(
                    "Source '{}' has kind '{}' which cannot be delegated from summon",
                    source_name, source.source_type
                ));
            }
        };

        if let Some(extra_instructions) = &params.instructions {
            if recipe.prompt.is_some() {
                let current_prompt = recipe.prompt.take().unwrap();
                recipe.prompt = Some(format!("{}\n\n{}", current_prompt, extra_instructions));
            } else {
                recipe.prompt = Some(extra_instructions.clone());
            }
        }

        if source.source_type == SourceType::Agent {
            let previous_results = match params.previous_task_id.as_deref() {
                Some(previous) => self
                    .context
                    .session_manager
                    .terminal_report_for_child(session_id, previous)
                    .await
                    .map_err(|error| error.to_string())?
                    .and_then(|report| {
                        artifact_results_from_report(&report.body).map(str::to_string)
                    }),
                None => None,
            };
            if let Some(assignment) = artifact_assignment(params, previous_results.as_deref()) {
                recipe.prompt = Some(match recipe.prompt.take() {
                    Some(prompt) => format!("{assignment}\n\n{prompt}"),
                    None => assignment,
                });
            }
        }

        Ok(recipe)
    }

    async fn build_recipe_from_source(
        &self,
        source: &SourceEntry,
        params: &DelegateParams,
        session_id: &str,
    ) -> Result<Recipe, String> {
        let session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|e| format!("Failed to get session: {}", e))?;

        if source.source_type == SourceType::Subrecipe {
            let sub_recipes = session.recipe.as_ref().and_then(|r| r.sub_recipes.as_ref());

            if let Some(sub_recipes) = sub_recipes {
                if let Some(sr) = sub_recipes.iter().find(|sr| sr.name == source.name) {
                    let recipe_file = load_local_recipe_file(&sr.path).map_err(|e| {
                        format!("Failed to load subrecipe '{}': {}", source.name, e)
                    })?;

                    let merged =
                        merge_subrecipe_parameters(sr.values.as_ref(), params.parameters.as_ref());
                    let param_values: Vec<(String, String)> = merged.into_iter().collect();

                    return build_recipe_from_template(
                        recipe_file.content,
                        &recipe_file.parent_dir,
                        param_values,
                        None::<fn(&str, &str) -> Result<String, anyhow::Error>>,
                    )
                    .map_err(|e| format!("Failed to build subrecipe: {}", e));
                }
            }
        }

        let recipe_file = load_local_recipe_file(&source.path)
            .map_err(|e| format!("Failed to load recipe '{}': {}", source.name, e))?;

        let param_values: Vec<(String, String)> = params
            .parameters
            .as_ref()
            .map(|p| {
                p.iter()
                    .map(|(k, v)| {
                        let value_str = match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k.clone(), value_str)
                    })
                    .collect()
            })
            .unwrap_or_default();

        build_recipe_from_template(
            recipe_file.content,
            &recipe_file.parent_dir,
            param_values,
            None::<fn(&str, &str) -> Result<String, anyhow::Error>>,
        )
        .map_err(|e| format!("Failed to build recipe: {}", e))
    }

    fn build_recipe_from_agent(
        &self,
        source: &SourceEntry,
        params: &DelegateParams,
        working_dir: &Path,
    ) -> Result<Recipe, String> {
        if source.path.is_empty() {
            return Err("Agent source has no path".to_string());
        }

        let model = source
            .properties
            .get("model")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);

        // max_turns is set later in build_task_config so it can incorporate params.max_turns
        // with the correct priority ordering; setting it here would cause it to be overridden
        // by the parent session's recipe instead.
        let settings = model.map(|m| Settings {
            goose_model: Some(m),
            goose_provider: params.provider.clone(),
            temperature: params.temperature,
            max_turns: None,
        });

        let mut instructions = source.content.clone();
        let required_skills = source
            .properties
            .get("required_skills")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>();
        if !required_skills.is_empty() {
            let skills = crate::skills::discover_skills(Some(working_dir));
            for required in required_skills {
                let skill = skills
                    .iter()
                    .find(|skill| skill.name == required)
                    .ok_or_else(|| format!("Required skill '{}' was not found", required))?;
                let content = crate::skills::loaded_skill_context_with_args(skill, None).map_err(
                    |error| format!("Failed to load required skill '{}': {}", required, error),
                )?;
                instructions.push_str("\n\n");
                instructions.push_str(&content);
            }
        }

        let mut builder = Recipe::builder()
            .version("1.0.0")
            .title(format!("Agent: {}", source.name))
            .description(source.description.clone())
            .instructions(instructions);

        if let Some(settings) = settings {
            builder = builder.settings(settings);
        }

        if params.instructions.is_none() {
            builder = builder.prompt("Proceed with your expertise to produce a useful result.");
        }

        builder
            .build()
            .map_err(|e| format!("Failed to build recipe from agent: {}", e))
    }

    async fn build_task_config(
        &self,
        params: &DelegateParams,
        recipe: &Recipe,
        session: &crate::session::Session,
    ) -> Result<TaskConfig, anyhow::Error> {
        let required_extensions = if let Some(source_name) = params.source.as_deref() {
            self.resolve_source(&session.id, source_name, &session.working_dir)
                .await
                .map_err(anyhow::Error::msg)?
                .filter(|source| source.source_type == SourceType::Agent)
                .and_then(|source| source.properties.get("required_extensions").cloned())
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default()
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let session_extensions = EnabledExtensionsState::extensions_or_default(
            Some(&session.extension_data),
            Config::global(),
        );
        let mut extensions = if required_extensions.is_empty() {
            session_extensions
        } else {
            required_extensions
                .iter()
                .map(|name| {
                    session_extensions
                        .iter()
                        .find(|extension| extension.name() == *name)
                        .cloned()
                        .or_else(|| crate::config::get_extension_by_name(name))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Required extension '{}' is not configured", name)
                        })
                })
                .collect::<std::result::Result<Vec<_>, anyhow::Error>>()?
        };
        let required_extension_names = if required_extensions.is_empty() {
            Vec::new()
        } else {
            extensions
                .iter()
                .map(|extension| extension.name())
                .collect()
        };

        if required_extensions.is_empty() {
            if let Some(filter) = &params.extensions {
                extensions = filter
                    .iter()
                    .map(|name| {
                        extensions
                            .iter()
                            .find(|extension| extension.name() == *name)
                            .cloned()
                            .or_else(|| crate::config::get_extension_by_name(name))
                            .ok_or_else(|| {
                                anyhow::anyhow!("Requested extension '{}' is not configured", name)
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }

        let (provider, model_config) = self
            .resolve_provider(params, recipe, session, &extensions)
            .await?;

        let max_turns = params
            .max_turns
            .or_else(|| recipe.settings.as_ref().and_then(|s| s.max_turns))
            .unwrap_or_else(|| self.resolve_max_turns(session));

        if max_turns == 0 || max_turns > u32::MAX as usize {
            anyhow::bail!(
                "max_turns must be between 1 and {} (got {})",
                u32::MAX,
                max_turns
            );
        }

        let effective_working_dir = match &params.working_dir {
            Some(dir) => resolve_working_dir(&session.working_dir, dir)?,
            None => session.working_dir.clone(),
        };

        let task_config = TaskConfig::new(
            provider,
            model_config,
            &session.id,
            &effective_working_dir,
            extensions,
        )
        .with_max_turns(Some(max_turns))
        .with_required_extension_names(required_extension_names);

        Ok(task_config)
    }

    fn resolve_model_config(
        &self,
        params: &DelegateParams,
        recipe: &Recipe,
        session: &crate::session::Session,
        provider_name: &str,
        provider_default_model: Option<&str>,
    ) -> Result<goose_providers::model::ModelConfig, anyhow::Error> {
        let env_model = std::env::var("GOOSE_SUBAGENT_MODEL").ok();
        let env_provider = std::env::var("GOOSE_SUBAGENT_PROVIDER").ok();
        let recipe_settings = recipe.settings.as_ref();
        let configured = Config::global().all_values().ok();
        let configured_provider = configured
            .as_ref()
            .and_then(|values| values.get("GOOSE_SUBAGENT_PROVIDER"))
            .and_then(serde_json::Value::as_str);
        let configured_model = configured
            .as_ref()
            .and_then(|values| values.get("GOOSE_SUBAGENT_MODEL"))
            .and_then(serde_json::Value::as_str);
        let matches_provider =
            |candidate: Option<&str>| candidate.is_none() || candidate == Some(provider_name);
        let model = recipe_settings
            .and_then(|settings| settings.goose_model.clone())
            .filter(|_| {
                matches_provider(
                    recipe_settings.and_then(|settings| settings.goose_provider.as_deref()),
                )
            })
            .or_else(|| {
                env_model
                    .clone()
                    .filter(|_| matches_provider(env_provider.as_deref()))
            })
            .or_else(|| {
                params
                    .model
                    .clone()
                    .filter(|_| matches_provider(params.provider.as_deref()))
            })
            .or_else(|| {
                configured_model
                    .filter(|_| matches_provider(configured_provider))
                    .map(str::to_string)
            })
            .or_else(|| {
                session
                    .model_config
                    .as_ref()
                    .filter(|_| matches_provider(session.provider_name.as_deref()))
                    .map(|config| config.model_name.clone())
            })
            .or_else(|| {
                provider_default_model
                    .filter(|model| !model.is_empty())
                    .map(str::to_string)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No model configured for provider '{}'; set GOOSE_SUBAGENT_MODEL",
                    provider_name
                )
            })?;

        let parent = session.model_config.as_ref();
        let mut model_config = if parent.is_some_and(|config| {
            matches_provider(session.provider_name.as_deref()) && config.model_name == model
        }) {
            parent.unwrap().clone()
        } else {
            let mut cfg = crate::model_config::model_config_from_user_config_with_session_settings(
                provider_name,
                &model,
                parent,
                None,
                None,
            )?;
            if let Some(parent) = parent {
                cfg.toolshim = parent.toolshim;
                cfg.toolshim_model = parent.toolshim_model.clone();
                cfg.temperature = cfg.temperature.or(parent.temperature);
            }
            cfg
        };

        if let Some(temp) = params.temperature {
            model_config = model_config.with_temperature(Some(temp));
        } else if let Some(temp) = recipe.settings.as_ref().and_then(|s| s.temperature) {
            model_config = model_config.with_temperature(Some(temp));
        }
        if let Some(max_tokens) = Config::global().get_goose_subagent_max_tokens()? {
            model_config = model_config.with_max_tokens(Some(max_tokens));
        }

        Ok(model_config)
    }

    async fn resolve_provider(
        &self,
        params: &DelegateParams,
        recipe: &Recipe,
        session: &crate::session::Session,
        extensions: &[crate::config::ExtensionConfig],
    ) -> Result<
        (
            Arc<dyn crate::providers::base::Provider>,
            goose_providers::model::ModelConfig,
        ),
        anyhow::Error,
    > {
        let env_provider = std::env::var("GOOSE_SUBAGENT_PROVIDER").ok();
        let provider_name = recipe
            .settings
            .as_ref()
            .and_then(|s| s.goose_provider.clone())
            .or_else(|| env_provider.clone())
            .or_else(|| params.provider.clone())
            .or_else(|| {
                Config::global()
                    .get_param::<String>("GOOSE_SUBAGENT_PROVIDER")
                    .ok()
            })
            .or_else(|| session.provider_name.clone())
            .ok_or_else(|| anyhow::anyhow!("No provider configured"))?;

        let provider_entry = providers::get_from_registry(&provider_name).await;
        let provider_default_model = provider_entry
            .as_ref()
            .ok()
            .map(|entry| entry.metadata().default_model.as_str());
        let model_config = self.resolve_model_config(
            params,
            recipe,
            session,
            &provider_name,
            provider_default_model,
        )?;
        let provider = match provider_entry {
            Ok(entry) => entry.create(extensions.to_vec()).await?,
            Err(error) => {
                let parent_provider = if let Some(extension_manager) = self
                    .context
                    .extension_manager
                    .as_ref()
                    .and_then(|weak| weak.upgrade())
                {
                    extension_manager.get_provider().lock().await.clone()
                } else {
                    None
                };

                match parent_provider {
                    Some(provider)
                        if provider.get_name() == provider_name
                            && !provider.manages_own_context() =>
                    {
                        provider
                    }
                    _ => return Err(error),
                }
            }
        };
        Ok((provider, model_config))
    }

    fn resolve_max_turns(&self, session: &crate::session::Session) -> usize {
        session
            .recipe
            .as_ref()
            .and_then(|r| r.settings.as_ref())
            .and_then(|s| s.max_turns)
            .or_else(|| {
                std::env::var("GOOSE_SUBAGENT_MAX_TURNS")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .or_else(|| {
                Config::global()
                    .get_param::<usize>("GOOSE_SUBAGENT_MAX_TURNS")
                    .ok()
            })
            .unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS)
    }

    /// Count durable, user-visible assistant blocks in the active task turn,
    /// excluding assistant-only compaction scaffolding.
    async fn refresh_task_turns(&self, task_id: &str, cached_turns: &AtomicU32) -> u32 {
        match self
            .context
            .session_manager
            .get_session(task_id, true)
            .await
        {
            Ok(session) => {
                let turns = session
                    .conversation
                    .as_ref()
                    .map(durable_assistant_turn_count)
                    .unwrap_or_default();
                cached_turns.store(turns, Ordering::Relaxed);
                turns
            }
            Err(error) => {
                warn!(
                    "Failed to refresh turn count for background task {}: {}",
                    task_id, error
                );
                cached_turns.load(Ordering::Relaxed)
            }
        }
    }

    async fn refresh_running_task_turns(&self) -> HashMap<String, u32> {
        let tasks: Vec<_> = self
            .background_tasks
            .lock()
            .await
            .values()
            .map(|task| (task.id.clone(), Arc::clone(&task.turns)))
            .collect();
        let mut refreshed = HashMap::with_capacity(tasks.len());
        for (id, turns) in tasks {
            let count = self.refresh_task_turns(&id, &turns).await;
            refreshed.insert(id, count);
        }
        refreshed
    }

    async fn wait_for_background_task_completion(
        &self,
        task_id: &str,
        completion_token: &CancellationToken,
    ) {
        completion_token.cancelled().await;
        loop {
            let finished_or_moved = self
                .background_tasks
                .lock()
                .await
                .get(task_id)
                .map(|task| task.handle.is_finished())
                .unwrap_or(true);
            if finished_or_moved {
                return;
            }
            tokio::task::yield_now().await;
        }
    }

    async fn cleanup_completed_tasks(&self) {
        let finished: Vec<(String, Arc<AtomicU32>)> = self
            .background_tasks
            .lock()
            .await
            .iter()
            .filter(|(_, task)| task.handle.is_finished())
            .map(|(id, task)| (id.clone(), Arc::clone(&task.turns)))
            .collect();

        let mut refreshed = HashMap::with_capacity(finished.len());
        for (id, turns) in &finished {
            let count = self.refresh_task_turns(id, turns).await;
            refreshed.insert(id.clone(), count);
        }

        // Keep the same lock order as task lookup so the running -> completed
        // transition is atomic from callers' perspective.
        let mut completed = self.completed_tasks.lock().await;
        let mut tasks = self.background_tasks.lock().await;
        for (id, _) in finished {
            let Some(task) = tasks.remove(&id) else {
                continue;
            };
            let turns_taken = refreshed
                .remove(&id)
                .unwrap_or_else(|| task.turns.load(Ordering::Relaxed));
            let duration = task.started_at.elapsed();

            let joined = task.handle.await;
            let join_status = match &joined {
                Err(error) if error.is_panic() => Some(TaskTerminalStatus::Panicked),
                Err(error) if error.is_cancelled() => Some(TaskTerminalStatus::Cancelled),
                _ => None,
            };
            let result = match joined {
                Ok(Ok(output)) => {
                    info!("Background task {} completed successfully", id);
                    Ok(output)
                }
                Ok(Err(e)) => {
                    warn!("Background task {} failed: {}", id, e);
                    Err(e.to_string())
                }
                Err(e) => {
                    warn!("Background task {} panicked: {}", id, e);
                    Err(format!("Task panicked: {}", e))
                }
            };

            let canonical = if task.parent_session_id.is_empty() {
                None
            } else {
                self.context
                    .session_manager
                    .terminal_report_for_child(&task.parent_session_id, &id)
                    .await
                    .ok()
                    .flatten()
            };
            let native_status = task
                .terminal_status
                .lock()
                .await
                .or(join_status)
                .unwrap_or_else(|| {
                    if task.cancellation_token.is_cancelled() {
                        TaskTerminalStatus::Cancelled
                    } else if result.is_ok() {
                        TaskTerminalStatus::Completed
                    } else {
                        TaskTerminalStatus::Failed
                    }
                });
            let status = canonical
                .as_ref()
                .and_then(|report| report.task_outcome().ok().flatten())
                .map(|outcome| outcome.status)
                .unwrap_or(native_status);
            let result = if status == TaskTerminalStatus::Cancelled {
                Err(format!(
                    "Task was cancelled: {}",
                    result.unwrap_or_else(|error| error)
                ))
            } else {
                result
            };
            let mut delivery_error = task.completion_delivery_error.lock().await.clone();
            if canonical.is_none() && !task.parent_session_id.is_empty() {
                let native_result = result.clone().map_err(anyhow::Error::msg);
                if let Err(error) = Self::enqueue_task_completion(
                    &self.context.session_manager,
                    &id,
                    &native_result,
                    status,
                )
                .await
                {
                    delivery_error = Some(error.to_string());
                }
            }
            completed.insert(
                id.clone(),
                CompletedTask {
                    id,
                    parent_session_id: task.parent_session_id,
                    completion_delivery_error: delivery_error,
                    terminal_status: status,
                    description: task.description,
                    result,
                    turns_taken,
                    duration,
                    completed_at: Instant::now(),
                    notification_sink: task.notification_sink,
                },
            );
        }

        let ttl = completed_task_ttl();
        completed.retain(|_id, task| {
            task.completion_delivery_error.is_some() || task.completed_at.elapsed() <= ttl
        });
    }

    fn get_task_description(params: &DelegateParams) -> String {
        match (&params.source, &params.instructions) {
            (Some(source), Some(instructions)) => format!("{}: {}", source, instructions),
            (Some(source), None) => source.clone(),
            (None, Some(instructions)) => instructions.clone(),
            (None, None) => "Unknown task".to_string(),
        }
    }

    async fn handle_async_delegate(
        &self,
        session_id: &str,
        mut params: DelegateParams,
    ) -> Result<(Vec<ContentBlock>, String), String> {
        let task_count = self.background_tasks.lock().await.len();
        let max_tasks = max_background_tasks();
        if task_count >= max_tasks {
            return Err(format!(
                "Maximum {} background tasks already running. Wait for completion or use sync mode.",
                max_tasks
            ));
        }

        let session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|e| format!("Failed to get session: {}", e))?;

        let working_dir = session.working_dir.clone();
        let source = if let Some(source_name) = params.source.as_deref() {
            self.resolve_source(session_id, source_name, &working_dir)
                .await?
        } else {
            None
        };
        let source_flag = |name: &str| {
            source
                .as_ref()
                .and_then(|source| source.properties.get(name))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        };
        let non_blocking = source_flag("non_blocking");
        let event_driven_parent = source_flag("event_driven_parent");
        let source_text = |name: &str| {
            source
                .as_ref()
                .and_then(|source| source.properties.get(name))
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        };
        // A source can remind its parent, once, to connect artifacts it delegates in
        // parallel: shown when two or more artifact tasks run and the parent has not
        // called the connecting tool.
        let connect_reminder = source_text("connect_reminder").zip(source_text("connect_tool"));
        let mut artifact_keys = if source_flag("artifact_guard") {
            vec![Self::artifact_key(&params)?]
        } else {
            Vec::new()
        };
        // The parent's artifact tool gives a new artifact its ID and checks an existing one.
        if let (Some(tool), Some(key), Some(kind)) = (
            source_text("artifact_tool"),
            artifact_keys.first().cloned(),
            params
                .source
                .as_deref()
                .and_then(|name| name.strip_prefix("cortex-"))
                .map(str::to_string),
        ) {
            let resolved = self
                .resolve_artifact(session_id, &working_dir, &tool, &key, &kind)
                .await?;
            artifact_keys = vec![resolved.artifact.clone()];
            params.artifact_key = Some(resolved.artifact);
            params.artifact_status = Some(resolved.status);
        }
        params.references =
            match source_text("artifact_tool") {
                Some(tool) => {
                    self.resolve_references(
                        session_id,
                        &working_dir,
                        &tool,
                        &params.reference_artifacts,
                        artifact_keys.first().map(String::as_str),
                    )
                    .await?
                }
                None if params.reference_artifacts.is_empty() => Vec::new(),
                None => return Err(
                    "reference_artifacts is only for specialist sources that work on artifacts."
                        .to_string(),
                ),
            };
        let reference_keys: Vec<String> = params
            .references
            .iter()
            .map(|reference| reference.artifact.clone())
            .collect();
        self.derive_previous_task_id(session_id, &artifact_keys, &mut params)
            .await?;
        let artifact = artifact_keys.first().cloned();
        let recipe = self
            .build_delegate_recipe(&params, session_id, &working_dir)
            .await?;

        let task_config = self
            .build_task_config(&params, &recipe, &session)
            .await
            .map_err(|e| format!("Failed to build task config: {}", e))?;

        let description = safe_truncate(&Self::get_task_description(&params), TASK_LABEL_BUDGET);

        // Subagents must use Auto until get_agent_messages forwards
        // ActionRequired messages to the parent. Until then, any mode
        // that requires approval will hang on the subagent's confirmation_rx.
        let mut agent_config = AgentConfig::new(
            self.context.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true, // disable session naming for subagents
            crate::agents::GoosePlatform::GooseCli,
        )
        .with_use_login_shell_path(self.context.use_login_shell_path);
        agent_config.is_subagent = true;

        let artifact_result_tools = source
            .as_ref()
            .and_then(|source| source.properties.get("artifact_result_tools"))
            .and_then(|tools| serde_json::from_value::<Vec<String>>(tools.clone()).ok())
            .unwrap_or_default();
        let policy = SummonTaskPolicy {
            event_driven_parent,
            artifact_key: artifact_keys.first().cloned(),
            previous_task_id: params.previous_task_id.clone(),
            artifact_result_tools,
            references: params.references.clone(),
        };
        let mut artifact_tasks = self.artifact_tasks.lock().await;
        let owned_or_referenced: Vec<String> = artifact_keys
            .iter()
            .chain(&reference_keys)
            .cloned()
            .collect();
        self.reconstruct_artifact_tasks(session_id, &mut artifact_tasks, &owned_or_referenced)
            .await?;
        self.check_artifact_owners(
            session_id,
            &artifact_keys,
            params.previous_task_id.as_deref(),
            &artifact_tasks,
        )
        .await?;
        self.check_references_frozen(session_id, &reference_keys, &artifact_tasks)
            .await?;
        // Running tasks that read this task's artifact lose that access once it starts.
        let revoked = match artifact_keys.first() {
            Some(key) => self.referencing_tasks(session_id, key).await?,
            None => Vec::new(),
        };
        if self.background_tasks.lock().await.len() >= max_background_tasks() {
            return Err("Maximum background tasks already running".to_string());
        }
        let subagent_session = self
            .create_subagent_session(
                &task_config,
                description.clone(),
                Some(&policy),
                source
                    .as_ref()
                    .map(|source| source.name.as_str())
                    .unwrap_or("inline"),
            )
            .await?;

        let task_id = subagent_session.id.clone();
        let completion_session_manager = Arc::clone(&self.context.session_manager);
        let completion_delivery_error = Arc::new(Mutex::new(None));
        let task_completion_delivery_error = Arc::clone(&completion_delivery_error);
        let terminal_status = Arc::new(Mutex::new(None));
        let task_terminal_status = Arc::clone(&terminal_status);

        let turns = Arc::new(AtomicU32::new(0));
        let last_activity = Arc::new(AtomicU64::new(0));

        let last_activity_clone = Arc::clone(&last_activity);

        let on_message: OnMessageCallback = Arc::new(move |_msg| {
            last_activity_clone.store(current_epoch_millis(), Ordering::Relaxed);
        });

        let task_token = CancellationToken::new();
        let task_token_clone = task_token.clone();
        let completion_cancellation_token = task_token.clone();
        let parent_span = tracing::Span::current();
        let telemetry_session_id =
            crate::session_context::telemetry_session_id(&task_config.parent_session_id);

        let notification_sink = Self::notification_sink(None);
        let task_notification_sink = Arc::clone(&notification_sink);

        let completion_task_id = task_id.clone();
        let (handle, completion_token) = spawn_background_task(async move {
            let params = SubagentRunParams {
                config: agent_config,
                recipe,
                task_config,
                return_last_only: true,
                session_id: subagent_session.id,
                cancellation_token: Some(task_token_clone),
                on_message: Some(on_message),
                notification_tx: None,
                parent_span,
                telemetry_session_id,
            };
            let (result, status) = task_execution_outcome(
                Self::run_subagent_with_notifications(
                    task_notification_sink,
                    move |notification_tx| {
                        let mut params = params;
                        params.notification_tx = Some(notification_tx);
                        run_subagent_task(params)
                    },
                ),
                &completion_cancellation_token,
            )
            .await;
            *task_terminal_status.lock().await = Some(status);
            if let Err(error) = Self::enqueue_task_completion(
                &completion_session_manager,
                &completion_task_id,
                &result,
                status,
            )
            .await
            {
                *task_completion_delivery_error.lock().await = Some(error.to_string());
                warn!(
                    "Failed to enqueue completion for background task {}: {}",
                    completion_task_id, error
                );
            }
            result
        });

        let task = BackgroundTask {
            id: task_id.clone(),
            parent_session_id: session_id.to_string(),
            non_blocking,
            completion_delivery_error,
            terminal_status,
            description: description.clone(),
            started_at: Instant::now(),
            turns,
            last_activity,
            handle,
            cancellation_token: task_token,
            completion_token,
            notification_sink,
        };

        self.background_tasks
            .lock()
            .await
            .insert(task_id.clone(), task);
        for key in artifact_keys {
            artifact_tasks.insert((session_id.to_string(), key), task_id.clone());
        }
        drop(artifact_tasks);
        if let Some(key) = artifact.as_deref() {
            for (holder, _) in &revoked {
                self.notify_revoked_reference(holder, key).await;
            }
        }
        if event_driven_parent {
            self.event_driven_parents
                .lock()
                .await
                .insert(session_id.to_string());
        }
        let reminder = match connect_reminder {
            Some((reminder, tool)) => self.connect_reminder(session_id, &tool, reminder).await,
            None => None,
        };

        if event_driven_parent {
            let mut content = vec![ContentBlock::text(format!(
                "Task {task_id} started in background: \"{description}\"\n\
                 It already has its complete task and reports back automatically: you are resumed for its terminal report or for a question it asks. \
                 Once you have delegated everything and finished any independent work, tell the user once, in their terms, what you are working on, then call wait. \
                 Use send(task_id: \"{task_id}\", message: \"...\") only to steer it with new guidance or answer its question."
            ))];
            if let Some(key) = artifact.as_deref() {
                content.push(ContentBlock::text(format!(
                    "The task works on artifact {key}: name it {key} from now on, in follow-ups, channels and your answer."
                )));
            }
            if !reference_keys.is_empty() {
                content.push(ContentBlock::text(format!(
                    "It can read these artifacts as read-only references, frozen at their saved revisions: {}.",
                    reference_keys.join(", ")
                )));
            }
            if let Some(key) = artifact.as_deref() {
                for (holder, holder_artifact) in &revoked {
                    content.push(ContentBlock::text(revocation_message(
                        key,
                        holder,
                        holder_artifact.as_deref(),
                    )));
                }
            }
            if let Some(reminder) = reminder {
                content.push(ContentBlock::text(reminder));
            }
            return Ok((content, task_id));
        }
        let retrieval = if non_blocking {
            format!(
                "It will report back automatically. Do not poll or sleep; finish your reply when no independent work remains. Use load(source: \"{task_id}\", peek: true) only for requested status."
            )
        } else {
            format!(
                "It will report back automatically. load(source: \"{task_id}\") waits for the result when needed."
            )
        };
        let content = vec![ContentBlock::text(format!(
            "Task {task_id} started in background: \"{description}\"\n\
             Continue with other work. {retrieval} Use send(task_id: \"{task_id}\", message: \"...\") to provide new guidance."
        ))];
        Ok((content, task_id))
    }
}

#[async_trait]
impl McpClientTrait for SummonClient {
    async fn list_tools(
        &self,
        session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        self.cleanup_completed_tasks().await;

        let session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .ok();
        let is_subagent = session
            .as_ref()
            .is_some_and(|s| s.session_type == SessionType::SubAgent);

        let mut tools = vec![self.create_load_tool()];

        if is_subagent {
            tools.push(self.create_message_parent_tool());
            // A specialist that starts editor tasks sleeps until their results,
            // its parent's messages or its channel news arrive.
            if session
                .as_ref()
                .and_then(SummonTaskPolicy::from_session)
                .is_some_and(|policy| !policy.artifact_result_tools.is_empty())
            {
                // A channel notice or channel tool result records the task's
                // membership; Goose lists tools again when membership changes.
                let in_channel = self
                    .context
                    .session_manager
                    .get_session(session_id, true)
                    .await
                    .ok()
                    .and_then(|session| session.conversation)
                    .is_some_and(|conversation| {
                        !Self::channel_wait_contexts(conversation.messages()).is_empty()
                    });
                tools.push(self.create_specialist_wait_tool(in_channel));
            }
        } else {
            let working_dir = self.get_working_dir(session_id).await;
            let sources = self.get_sources(session_id, &working_dir).await;
            let event_driven = sources.iter().any(|source| {
                source
                    .properties
                    .get("event_driven_parent")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            });
            tools.push(self.create_delegate_tool(Self::only_specialist_sources(&sources)));
            tools.push(self.create_send_tool());
            if event_driven {
                tools.push(self.create_wait_tool());
            }
        }

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let session_id = &ctx.session_id;
        match name {
            "load" => match self
                .handle_load(session_id, arguments, ctx.notification_emitter().cloned())
                .await
            {
                Ok(result) => Ok(result),
                Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                    "Error: {}",
                    error
                ))])),
            },
            "delegate" => {
                match self
                    .handle_delegate(
                        session_id,
                        arguments,
                        cancellation_token,
                        ctx.notification_emitter().cloned(),
                    )
                    .await
                {
                    Ok(result) => Ok(result),
                    Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "Error: {}",
                        error
                    ))])),
                }
            }
            "send" => {
                let params: std::result::Result<SendParams, _> = arguments
                    .map(|args| serde_json::from_value(serde_json::Value::Object(args)))
                    .transpose()
                    .map(|params| {
                        params.unwrap_or(SendParams {
                            task_id: String::new(),
                            message: String::new(),
                        })
                    });
                match params {
                    Ok(params) => match async {
                        let tasks = self.background_tasks.lock().await;
                        let Some(task) = tasks.get(&params.task_id) else {
                            return Err(anyhow::anyhow!(
                                "Task '{}' is not running. {}",
                                params.task_id,
                                redelegate_hint(&params.task_id)
                            ));
                        };
                        if task.handle.is_finished() {
                            return Err(anyhow::anyhow!(
                                "Task '{}' has already finished. {}",
                                params.task_id,
                                redelegate_hint(&params.task_id)
                            ));
                        }
                        self.context
                            .session_manager
                            .send_to_child(session_id, &params.task_id, &params.message)
                            .await
                    }
                    .await
                    {
                        Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(
                            self.send_acknowledgement(session_id, &params.task_id).await,
                        )])),
                        Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                            "Error: {error}"
                        ))])),
                    },
                    Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "Error: Invalid parameters: {error}"
                    ))])),
                }
            }
            "wait" => {
                let is_subagent = self
                    .context
                    .session_manager
                    .get_session(session_id, false)
                    .await
                    .is_ok_and(|session| session.session_type == SessionType::SubAgent);
                if is_subagent {
                    Ok(self
                        .handle_specialist_wait(session_id, arguments, cancellation_token)
                        .await)
                } else {
                    Ok(self.handle_wait(session_id).await)
                }
            }
            "message_parent" => {
                let params: std::result::Result<MessageParentParams, _> = arguments
                    .map(|args| serde_json::from_value(serde_json::Value::Object(args)))
                    .transpose()
                    .map(|params| {
                        params.unwrap_or(MessageParentParams {
                            message: String::new(),
                        })
                    });
                match params {
                    Ok(params) if params.message.trim().is_empty() => {
                        Ok(CallToolResult::error(vec![ContentBlock::text(
                            "Error: message_parent requires the question to ask.",
                        )]))
                    }
                    Ok(params) if !asks_question(&params.message) => {
                        Ok(CallToolResult::error(vec![ContentBlock::text(
                            "Error: message_parent carries only a question the parent must answer, and this message asks none. Progress, results and limitations reach the parent in your final report; continue your work, or ask one concrete question.",
                        )]))
                    }
                    Ok(params) => {
                        match self
                            .context
                            .session_manager
                            .send_to_parent(session_id, &params.message)
                            .await
                        {
                            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(
                                "Question queued for the parent task. Wait for its reply before depending on the answer.",
                            )])),
                            Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(
                                format!("Error: {error}"),
                            )])),
                        }
                    }
                    Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "Error: Invalid parameters: {error}"
                    ))])),
                }
            }
            _ => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error: Unknown tool: {}",
                name
            ))])),
        }
    }

    async fn shutdown_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.delivered_reports.lock().await.remove(session_id);
        let task_ids = {
            let tasks = self.background_tasks.lock().await;
            tasks
                .values()
                .filter(|task| task.parent_session_id == session_id)
                .map(|task| {
                    task.cancellation_token.cancel();
                    task.id.clone()
                })
                .collect::<Vec<_>>()
        };
        let mut first_error = None;
        for task_id in task_ids {
            if let Err(error) = self
                .handle_load_task_result(&task_id, true, false, None)
                .await
            {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => anyhow::bail!(error),
            None => self.has_active_tasks(session_id).await.map(|_| ()),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }

    fn get_instructions(&self) -> Option<String> {
        let instructions = build_subagent_instructions(self.context.session.as_deref());
        if instructions.is_empty() {
            None
        } else {
            Some(instructions)
        }
    }

    async fn get_moim(&self, session_id: &str) -> Option<String> {
        self.cleanup_completed_tasks().await;
        let refreshed_turns = self.refresh_running_task_turns().await;

        let event_driven = self.event_driven_parents.lock().await.contains(session_id);
        let pending_messages = if event_driven {
            self.context
                .session_manager
                .pending_session_messages(session_id)
                .await
                .ok()
        } else {
            None
        };
        let delivered_through = self.delivered_reports.lock().await.get(session_id).copied();
        let delivered = |message: &crate::session::MailboxMessage| {
            delivered_through.is_some_and(|through| message.id <= through)
        };
        // Pending terminal reports by task, and whether the current turn
        // already carries each one.
        let pending_reports: Option<HashMap<String, bool>> =
            pending_messages.as_ref().map(|pending| {
                let mut reports = HashMap::new();
                for message in pending.iter().filter(|message| {
                    message.kind == crate::session::MailboxMessageKind::Completion
                }) {
                    let in_turn = delivered(message);
                    reports
                        .entry(message.sender_session_id.clone())
                        .and_modify(|seen: &mut bool| *seen |= in_turn)
                        .or_insert(in_turn);
                }
                reports
            });
        let turn_ends_task = pending_reports
            .as_ref()
            .is_some_and(|reports| reports.values().any(|in_turn| *in_turn));
        let turn_asks_question = pending_messages.as_ref().is_some_and(|pending| {
            pending.iter().any(|message| {
                message.kind == crate::session::MailboxMessageKind::Message && delivered(message)
            })
        });

        let completed = self.completed_tasks.lock().await;
        let running = self.background_tasks.lock().await;
        let belongs = |parent: &str| parent.is_empty() || parent == session_id;

        let mut sorted_running: Vec<_> = running
            .values()
            .filter(|task| belongs(task.parent_session_id.as_str()))
            .collect();
        sorted_running.sort_by_key(|task| &task.id);
        let mut sorted_completed: Vec<_> = completed
            .values()
            .filter(|task| belongs(task.parent_session_id.as_str()))
            .filter(|task| match &pending_reports {
                Some(pending) => {
                    pending.contains_key(&task.id) || task.completion_delivery_error.is_some()
                }
                None => true,
            })
            .collect();
        sorted_completed.sort_by_key(|task| &task.id);

        if sorted_running.is_empty() && sorted_completed.is_empty() {
            return None;
        }

        let mut lines = vec!["Background tasks:".to_string()];
        let now = current_epoch_millis();
        let has_running = !sorted_running.is_empty();

        for task in sorted_running {
            let elapsed = task.started_at.elapsed();
            let last_activity_at = task.last_activity.load(Ordering::Relaxed);
            let idle_ms = if last_activity_at == 0 {
                u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
            } else {
                now.saturating_sub(last_activity_at)
            };

            lines.push(format!(
                "• {}: \"{}\" - running {}, {} turns, idle {}",
                task.id,
                task.description,
                round_duration(elapsed),
                refreshed_turns
                    .get(&task.id)
                    .copied()
                    .unwrap_or_else(|| task.turns.load(Ordering::Relaxed)),
                round_duration(Duration::from_millis(idle_ms)),
            ));
        }

        let delivery = |task_id: &str| {
            if !event_driven {
                "completion is reported automatically"
            } else if pending_reports
                .as_ref()
                .and_then(|reports| reports.get(task_id))
                .copied()
                .unwrap_or(false)
            {
                "its report is in the message above"
            } else {
                "its report is pending"
            }
        };
        for task in sorted_completed {
            let status = match task.terminal_status {
                TaskTerminalStatus::Completed if task.result.is_ok() => "completed",
                TaskTerminalStatus::Completed | TaskTerminalStatus::Failed => "failed",
                TaskTerminalStatus::Cancelled => "cancelled",
                TaskTerminalStatus::Panicked => "panicked",
            };
            lines.push(format!(
                "• {}: \"{}\" - {} in {} ({} turns) - {}",
                task.id,
                task.description,
                status,
                round_duration(task.duration),
                task.turns_taken,
                delivery(&task.id),
            ));
        }

        if has_running {
            lines.push(if event_driven && turn_ends_task {
                "\n→ A task in the report above has ended while others still run. Unless that report is unrelated to the current request: if it shows that a requested artifact failed, or names requested content still missing or wrong in it, and another attempt can still succeed in this turn, delegate its follow-up now with the same artifact_key, before you write. A partial result that names no such concrete gap is reported to the user as it is, not delegated again. Then write a brief update giving the current status of every requested artifact, each one finished so far and each one still in progress, then call wait. You are resumed for each terminal report and for each specialist question. Use send only to give a running specialist new guidance or answer its question."
                    .to_string()
            } else if event_driven && turn_asks_question {
                "\n→ The report above is a specialist question and no task has ended. Answer it with send when the request, research, or conversation settles it, then call wait without writing; ask the user only when only they can decide. You are resumed for each terminal report and for each specialist question."
                    .to_string()
            } else if event_driven {
                "\n→ Specialists report automatically. Call wait now unless you still have independent work, a new user message to act on, or news for the user: you are resumed for each terminal report, each specialist question, and each new user message. Use send only to give a running specialist new guidance or answer its question."
                    .to_string()
            } else {
                "\n→ Reports arrive automatically. Do not poll or sleep; finish your reply when no independent work remains. Use send to steer an existing task, load(source: \"<id>\", peek: true) for requested status, or load(source: \"<id>\", cancel: true) to stop it"
                    .to_string()
            });
        }

        Some(lines.join("\n"))
    }

    async fn note_reports_delivered(&self, session_id: &str, through_id: i64) {
        let mut delivered = self.delivered_reports.lock().await;
        let entry = delivered
            .entry(session_id.to_string())
            .or_insert(through_id);
        *entry = (*entry).max(through_id);
    }

    async fn has_active_tasks(&self, session_id: &str) -> anyhow::Result<bool> {
        self.cleanup_completed_tasks().await;
        let deliveries: Vec<_> = self
            .completed_tasks
            .lock()
            .await
            .values()
            .filter(|task| {
                task.parent_session_id == session_id && task.completion_delivery_error.is_some()
            })
            .map(|task| task.id.clone())
            .collect();
        for task_id in deliveries {
            self.recover_completion_delivery(&task_id)
                .await
                .map_err(anyhow::Error::msg)?;
        }
        let running = self
            .background_tasks
            .lock()
            .await
            .values()
            .filter(|task| task.parent_session_id == session_id)
            .map(|task| {
                (
                    task.handle.is_finished(),
                    Arc::clone(&task.completion_delivery_error),
                )
            })
            .collect::<Vec<_>>();
        let completed = self
            .completed_tasks
            .lock()
            .await
            .values()
            .filter(|task| task.parent_session_id == session_id)
            .filter_map(|task| task.completion_delivery_error.clone())
            .collect::<Vec<_>>();
        for (_, error) in &running {
            if let Some(error) = error.lock().await.clone() {
                anyhow::bail!("Failed to deliver background task completion: {error}");
            }
        }
        if let Some(error) = completed.first() {
            anyhow::bail!("Failed to deliver background task completion: {error}");
        }
        Ok(running.iter().any(|(finished, _)| !finished))
    }
}

/// Resolve a requested `working_dir` override against the parent session
/// directory. Relative paths are joined to the parent dir; the result must
/// canonicalize to an existing directory contained within the parent dir.
fn resolve_working_dir(parent_dir: &Path, requested: &str) -> Result<PathBuf, anyhow::Error> {
    let requested_path = PathBuf::from(requested);
    let resolved = if requested_path.is_absolute() {
        requested_path
    } else {
        parent_dir.join(&requested_path)
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("working_dir '{}' could not be resolved: {}", requested, e))?;
    let parent_canonical = parent_dir
        .canonicalize()
        .unwrap_or_else(|_| parent_dir.to_path_buf());
    if !canonical.starts_with(&parent_canonical) {
        anyhow::bail!(
            "working_dir '{}' is outside the parent session directory",
            requested
        );
    }
    if !canonical.is_dir() {
        anyhow::bail!("working_dir '{}' is not a directory", requested);
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExtensionConfig;
    use crate::conversation::message::Message;
    use futures::StreamExt;
    use serial_test::serial;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn verified_partial_receipt() -> serde_json::Value {
        serde_json::json!({
            "job_id":"d70ecd94-6446-5049-9ed0-de3d78a858ba", "document_id":"budget",
            "status":"partial", "document_revision":2, "saved_revision_id":"saved-2",
            "policy_stop_reason":"decision_budget",
            "verification": {"engine_revision":27,"source_digest":"a".repeat(64),"valid":true,
                "scope":"artifact","checks":[{"name":"All supplied estimates","valid":true},
                {"name":"Missing workstream rate keeps cost and totals blank","valid":true}],
                "layout_issue_count":0,"issues":[],"omitted_checks":0,"omitted_issues":0,"diagnostics_compacted":false},
            "grounding_integrity":{"validation_status":"valid","saved_associations_count":8,"dropped_count":0,"unresolved_count":0}
        })
    }

    #[test]
    fn delivered_receipts_preserve_verified_existing_and_partial_outcomes() {
        let tools = ["cortex_spreadsheet__delegate_editor_task".to_owned()];
        for status in ["already_satisfied", "partial"] {
            let mut receipt = verified_partial_receipt();
            receipt["status"] = serde_json::json!(status);
            let notice = |receipt: serde_json::Value| {
                let mut message = Message::user().with_text("Editor result");
                message.metadata.set_operation_note(
                    crate::session::EDITOR_RESULT_NOTE,
                    "v1",
                    serde_json::json!({"idempotency_key":"budget", "receipt":receipt}),
                );
                message
            };
            let summary = artifact_result_summary(&[notice(receipt.clone())], &tools);
            assert_eq!(
                summary.completion_description(),
                if status == "partial" {
                    "ended with partial artifact output"
                } else {
                    "completed successfully"
                }
            );
            if status == "already_satisfied" {
                assert!(summary
                    .section()
                    .contains("existing revision 2 unchanged; no new save"));
                receipt["verification"]["valid"] = serde_json::json!(false);
                assert_eq!(
                    artifact_result_summary(&[notice(receipt)], &tools).completion_description(),
                    "ended without a confirmed artifact outcome"
                );
            }
        }
    }

    #[tokio::test]
    async fn completion_wait_wakes_and_fails_truthfully_without_inference() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let child = reliability_child(&client, &parent, "document:doc", None).await;
        let cancel = CancellationToken::new();
        assert!(
            wait_for_editor_notice(&manager, &child, &cancel, Duration::ZERO)
                .await
                .unwrap_err()
                .to_string()
                .contains("deadline")
        );
        cancel.cancel();
        assert!(
            wait_for_editor_notice(&manager, &child, &cancel, Duration::from_secs(1))
                .await
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
        manager
            .send_to_child(&parent, &child, "Review your editor result")
            .await
            .unwrap();
        wait_for_editor_notice(
            &manager,
            &child,
            &CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
    }

    #[test]
    fn already_satisfied_requires_verified_existing_revision() {
        let mut value = verified_partial_receipt();
        value["status"] = serde_json::json!("already_satisfied");
        assert_eq!(
            confirmed_editor_status(&value),
            Some(EditorArtifactStatus::AlreadySatisfied)
        );
        let report = describe_editor_result(&value);
        assert!(report.contains("existing revision 2 unchanged; no new save"));
        let tool = "cortex_spreadsheet__delegate_editor_task";
        let mut result = CallToolResult::success(vec![]);
        result.structured_content = Some(value.clone());
        let messages = vec![
            Message::assistant()
                .with_tool_request("verify", Ok(rmcp::model::CallToolRequestParams::new(tool))),
            Message::user().with_tool_response("verify", Ok(result)),
        ];
        assert_eq!(
            artifact_result_summary(&messages, &[tool.to_owned()]).completion_description(),
            "completed successfully"
        );
        let assignment = artifact_assignment(
            &DelegateParams {
                artifact_key: Some("document:budget".into()),
                previous_task_id: Some("previous".into()),
                ..Default::default()
            },
            Some(&report),
        )
        .unwrap();
        let inherited =
            artifact_result_summary(&[Message::user().with_text(assignment)], &[tool.to_owned()]);
        assert_eq!(inherited.completion_description(), "completed successfully");
        assert!(inherited
            .section()
            .contains("existing revision 2 unchanged; no new save"));
        let receipt = saved_artifact_receipt(&value).unwrap();
        assert_eq!(receipt.document_revision, 2);
        assert_eq!(receipt.saved_revision_id.as_deref(), Some("saved-2"));
        let summary = ArtifactResultSummary {
            lines: vec![report],
            current: HashMap::from([(
                "budget".into(),
                Some(EditorArtifactStatus::AlreadySatisfied),
            )]),
        };
        assert_eq!(summary.completion_description(), "completed successfully");
        for key in ["document_revision", "saved_revision_id", "verification"] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert_eq!(confirmed_editor_status(&missing), None);
        }
        value["verification"]["valid"] = serde_json::json!(false);
        assert_eq!(confirmed_editor_status(&value), None);
        assert_eq!(
            confirmed_editor_status(&serde_json::json!({"document_id":"budget", "status":"empty"})),
            Some(EditorArtifactStatus::Empty)
        );
    }

    #[test]
    fn verified_partial_survives_successor_policy_rejection() {
        let original = describe_editor_result(&verified_partial_receipt());
        let assignment = artifact_assignment(
            &DelegateParams {
                artifact_key: Some("document:budget".into()),
                previous_task_id: Some("previous".into()),
                ..Default::default()
            },
            Some(&original),
        )
        .unwrap();
        let tool = "cortex_spreadsheet__delegate_editor_task";
        let rejection=CallToolResult::error(vec![ContentBlock::text(serde_json::json!({
            "code":"editor_policy_exhausted","job_id":"d70ecd94-6446-5049-9ed0-de3d78a858ba","policy_stop_reason":"decision_budget"
        }).to_string())]);
        let messages = vec![
            Message::user().with_text(assignment),
            Message::assistant().with_tool_request(
                "retry",
                Ok(
                    rmcp::model::CallToolRequestParams::new(tool).with_arguments(
                        serde_json::json!({"document_id":"budget"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                ),
            ),
            Message::user().with_tool_response("retry", Ok(rejection)),
        ];
        let summary = artifact_result_summary(&messages, &[tool.to_owned()]);
        assert_eq!(
            summary.completion_description(),
            "ended with partial artifact output"
        );
        assert_eq!(
            summary.current.get("budget"),
            Some(&Some(EditorArtifactStatus::Partial))
        );
        assert!(summary.section().contains("no NEW revision"));
        assert!(!summary.section().contains("returned no saved document"));
        let receipts = predecessor_receipts(&messages);
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].saved_revision_id.as_deref(), Some("saved-2"));
        assert_eq!(
            receipts[0].policy_stop_reason.as_deref(),
            Some("decision_budget")
        );
        let verification = receipts[0].verification.as_ref().unwrap();
        assert!(verification.valid);
        assert_eq!(verification.engine_revision, 27);
        assert_eq!(verification.source_digest, "a".repeat(64));
        assert_eq!(
            receipts[0]
                .grounding_integrity
                .as_ref()
                .unwrap()
                .saved_associations_count,
            8
        );
    }

    #[test]
    fn saved_receipts_do_not_invent_or_promote_verification() {
        let mut value = verified_partial_receipt();
        value["verification"] = serde_json::Value::Null;
        assert!(saved_artifact_receipt(&value)
            .unwrap()
            .verification
            .is_none());
        value = verified_partial_receipt();
        value["verification"]["valid"] = false.into();
        value["verification"]["checks"][0]["valid"] = false.into();
        assert!(
            !saved_artifact_receipt(&value)
                .unwrap()
                .verification
                .unwrap()
                .valid
        );
        value["saved_revision_id"] = serde_json::Value::Null;
        let receipt = saved_artifact_receipt(&value).unwrap();
        assert!(receipt.verification.is_none() && receipt.grounding_integrity.is_none());
        value["document_revision"] = 0.into();
        assert!(saved_artifact_receipt(&value).is_none());
    }

    #[test]
    fn predecessor_receipts_are_limited_to_the_runtime_assignment() {
        let receipt = verified_partial_receipt();
        let canonical = format!("{ARTIFACT_RECEIPT_PREFIX}{}", receipt);
        assert!(
            predecessor_receipts(&[Message::assistant().with_text(format!(
                "Assigned artifact (set by the coordinator for this task):\n{canonical}"
            ))])
            .is_empty()
        );
        assert!(predecessor_receipts(&[Message::user().with_text(format!("Assigned artifact (set by the coordinator for this task):\n- Artifact key: document:budget\n\nUser instruction: \n{canonical}"))]).is_empty());
        let mut injected = receipt;
        injected["summary"] = format!("summary\n{canonical}").into();
        let text = describe_editor_result(&injected);
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with(ARTIFACT_RECEIPT_PREFIX))
                .count(),
            1
        );
    }

    #[test]
    fn a_new_save_does_not_inherit_predecessor_verification() {
        let original = describe_editor_result(&verified_partial_receipt());
        let assignment = artifact_assignment(
            &DelegateParams {
                artifact_key: Some("document:budget".into()),
                previous_task_id: Some("previous".into()),
                ..Default::default()
            },
            Some(&original),
        )
        .unwrap();
        let tool = "cortex_spreadsheet__delegate_editor_task";
        let mut changed = verified_partial_receipt();
        changed["document_revision"] = 3.into();
        changed["saved_revision_id"] = "saved-3".into();
        changed["verification"] = serde_json::Value::Null;
        let mut result = CallToolResult::success(vec![]);
        result.structured_content = Some(changed);
        let messages = vec![
            Message::user().with_text(assignment),
            Message::assistant().with_tool_request(
                "change",
                Ok(
                    rmcp::model::CallToolRequestParams::new(tool).with_arguments(
                        serde_json::json!({"document_id":"budget"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                ),
            ),
            Message::user().with_tool_response("change", Ok(result)),
        ];
        let summary = artifact_result_summary(&messages, &[tool.to_owned()]);
        assert!(summary.lines[0].contains("checks apply only to its recorded revision"));
        let last = summary.lines.last().unwrap();
        assert!(last.contains("saved revision 3"));
        let canonical = last
            .lines()
            .find_map(|line| line.strip_prefix(ARTIFACT_RECEIPT_PREFIX))
            .unwrap();
        let receipt: SavedArtifactReceipt = serde_json::from_str(canonical).unwrap();
        assert!(receipt.verification.is_none());
    }

    #[test]
    fn canonical_verification_rejects_malformed_and_unbounded_summaries() {
        for (key, bad) in [
            (
                "checks",
                serde_json::json!([{"name":"x".repeat(121),"valid":true}]),
            ),
            ("issues", serde_json::json!(["x".repeat(5000)])),
            ("source_digest", serde_json::json!("not-a-digest")),
            ("scope", serde_json::json!("other")),
            ("engine_revision", serde_json::json!(-1)),
        ] {
            let mut value = verified_partial_receipt();
            value["verification"][key] = bad;
            assert!(
                saved_artifact_receipt(&value)
                    .unwrap()
                    .verification
                    .is_none(),
                "{key}"
            );
        }
        let mut value = verified_partial_receipt();
        value["grounding_integrity"]["dropped_count"] = 1.into();
        assert!(saved_artifact_receipt(&value)
            .unwrap()
            .grounding_integrity
            .is_none());
        value["summary"] = "x".repeat(10000).into();
        assert!(describe_editor_result(&value).len() < 5500);
    }

    #[test]
    fn artifact_result_lines_describe_saved_failed_and_unfinished_editor_jobs() {
        let tool = "cortex_document__delegate_editor_task";
        let call = |id: &str, arguments: serde_json::Value| {
            Message::assistant().with_tool_request(
                id,
                Ok(rmcp::model::CallToolRequestParams::new(tool)
                    .with_arguments(arguments.as_object().unwrap().clone())),
            )
        };
        let mut saved = CallToolResult::success(vec![ContentBlock::text("saved")]);
        saved.structured_content = Some(serde_json::json!({
            "job_id": "job-1", "status": "partial", "summary": "Created six sections",
            "remaining_work": "Add the rubric", "document_id": "doc-1", "document_revision": 2,
        }));
        let cancelled = CallToolResult::error(vec![ContentBlock::text(
            serde_json::json!({"status": "cancelled", "summary": "Stopped", "document_id": "doc-2", "document_revision": null})
                .to_string(),
        )]);
        let messages = vec![
            call(
                "c1",
                serde_json::json!({"editor": "quire", "create_new": true, "title": "Speaker Notes"}),
            ),
            Message::user().with_tool_response("c1", Ok(saved)),
            call(
                "c2",
                serde_json::json!({"editor": "quire", "document_id": "doc-2"}),
            ),
            Message::user().with_tool_response("c2", Ok(cancelled)),
            Message::assistant().with_tool_request(
                "c3",
                Ok(rmcp::model::CallToolRequestParams::new(
                    "cortex_document__read_document",
                )),
            ),
            call(
                "c4",
                serde_json::json!({"editor": "quire", "document_id": "doc-1"}),
            ),
        ];

        let lines = artifact_result_lines(&messages, &[tool.to_string()]);

        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].contains("partial") && lines[0].contains("document doc-1"));
        assert!(lines[0].contains("saved revision 2"));
        assert!(lines[0].contains("Remaining work: Add the rubric"));
        assert!(lines[1].contains("cancelled") && lines[1].contains("no saved revision"));
        assert!(lines[2].contains("document doc-1") && lines[2].contains("still running"));
    }

    #[test]
    fn artifact_completion_uses_current_receipts_instead_of_specialist_prose() {
        let tool = "cortex_document__delegate_editor_task";
        let messages_for = |outcomes: &[(&str, &str, Option<u64>)]| {
            let mut messages = Vec::new();
            for (index, (document, status, revision)) in outcomes.iter().enumerate() {
                let id = format!("call-{index}");
                messages.push(
                    Message::assistant().with_tool_request(
                        &id,
                        Ok(
                            rmcp::model::CallToolRequestParams::new(tool).with_arguments(
                                serde_json::json!({"document_id":document})
                                    .as_object()
                                    .unwrap()
                                    .clone(),
                            ),
                        ),
                    ),
                );
                let mut result = CallToolResult::success(vec![ContentBlock::text("All done")]);
                result.structured_content = Some(
                    serde_json::json!({"document_id":document,"status":status,"document_revision":revision}),
                );
                messages.push(Message::user().with_tool_response(&id, Ok(result)));
            }
            messages.push(Message::assistant().with_text("All requested artifacts are complete."));
            messages
        };
        let tools = vec![tool.to_owned()];
        let partial =
            artifact_result_summary(&messages_for(&[("doc", "partial", Some(2))]), &tools);
        assert_eq!(
            partial.completion_description(),
            "ended with partial artifact output"
        );
        let failed = artifact_result_summary(&messages_for(&[("doc", "failed", Some(2))]), &tools);
        assert_eq!(
            failed.completion_description(),
            "ended with incomplete artifact output"
        );
        let retry = artifact_result_summary(
            &messages_for(&[("doc", "failed", Some(2)), ("doc", "completed", Some(3))]),
            &tools,
        );
        assert_eq!(retry.completion_description(), "completed successfully");
        assert_eq!(retry.lines.len(), 2);
        let distinct = artifact_result_summary(
            &messages_for(&[
                ("doc-1", "partial", Some(2)),
                ("doc-2", "completed", Some(2)),
            ]),
            &tools,
        );
        assert_eq!(
            distinct.completion_description(),
            "ended with partial artifact output"
        );
        assert_eq!(distinct.current.len(), 2);
        let unsaved = artifact_result_summary(&messages_for(&[("doc", "completed", None)]), &tools);
        assert_eq!(
            unsaved.completion_description(),
            "ended without a confirmed artifact outcome"
        );
    }

    #[test]
    fn an_interrupted_editor_wait_is_superseded_by_the_later_wait_for_the_same_task() {
        let delegate = "cortex_presentation__delegate_editor_task";
        let wait = "cortex_presentation__editor_result";
        let call = |tool: &str, id: &str, arguments: serde_json::Value| {
            Message::assistant().with_tool_request(
                id,
                Ok(rmcp::model::CallToolRequestParams::new(tool.to_string())
                    .with_arguments(arguments.as_object().unwrap().clone())),
            )
        };
        let interrupted = || {
            let mut result = CallToolResult::success(vec![ContentBlock::text("Stopped waiting")]);
            result.meta = Some(MetaObject(
                serde_json::json!({ crate::agents::tool_interrupt::INTERRUPTED_META_KEY: true })
                    .as_object()
                    .unwrap()
                    .clone(),
            ));
            result
        };
        let mut saved = CallToolResult::success(vec![ContentBlock::text("saved")]);
        saved.structured_content = Some(serde_json::json!({
            "job_id": "job-1", "status": "completed", "summary": "Four slides",
            "document_id": "deck-1", "document_revision": 3,
        }));
        let tools = [delegate.to_string(), wait.to_string()];
        let messages = vec![
            call(
                delegate,
                "c1",
                serde_json::json!({"editor": "aurelia_slides", "title": "Deck", "instruction": "Build it", "idempotency_key": "deck"}),
            ),
            Message::user().with_tool_response("c1", Ok(interrupted())),
            call(wait, "c2", serde_json::json!({"idempotency_key": "deck"})),
            Message::user().with_tool_response("c2", Ok(saved)),
            call(
                delegate,
                "c3",
                serde_json::json!({"editor": "aurelia_slides", "document_id": "deck-1", "instruction": "Fix", "idempotency_key": "fix"}),
            ),
            Message::user().with_tool_response("c3", Ok(interrupted())),
        ];

        let lines = artifact_result_lines(&messages, &tools);

        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("completed") && lines[0].contains("deck-1"));
        assert!(lines[1].contains("document deck-1") && lines[1].contains("still running"));
        // The first task has its result; the follow-up still runs.
        assert_eq!(
            running_editor_task_keys(&messages, &tools),
            vec!["fix".to_string()]
        );
    }

    #[test]
    fn a_started_editor_task_runs_until_its_delivered_result() {
        let delegate = "cortex_document__delegate_editor_task";
        let started = || {
            let mut result = CallToolResult::success(vec![ContentBlock::text("queued")]);
            result.structured_content =
                Some(serde_json::json!({ "interrupted": true, "job_id": "job-1" }));
            result
        };
        let delivered = || {
            let mut message = Message::user()
                .with_text("Your editor task notes finished.")
                .with_visibility(false, true);
            message.metadata.set_operation_note(
                crate::session::EDITOR_RESULT_NOTE,
                "v1",
                serde_json::json!({
                    "idempotency_key": "notes",
                    "receipt": {
                        "job_id": "job-1", "status": "completed", "summary": "Notes",
                        "document_id": "doc-1", "document_revision": 2,
                    },
                }),
            );
            message
        };
        let tools = [delegate.to_string()];
        let mut messages = vec![
            Message::assistant().with_tool_request(
                "c1",
                Ok(rmcp::model::CallToolRequestParams::new(delegate.to_string()).with_arguments(
                    serde_json::json!({"editor": "quire", "instruction": "Write", "idempotency_key": "notes"})
                        .as_object()
                        .unwrap()
                        .clone(),
                )),
            ),
            Message::user().with_tool_response("c1", Ok(started())),
        ];
        assert_eq!(
            running_editor_task_keys(&messages, &tools),
            vec!["notes".to_string()]
        );
        let lines = artifact_result_lines(&messages, &tools);
        assert!(lines[0].contains("still running"), "{lines:?}");

        messages.push(delivered());
        assert!(running_editor_task_keys(&messages, &tools).is_empty());
        let lines = artifact_result_lines(&messages, &tools);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("completed") && lines[0].contains("doc-1"));
    }

    #[test]
    fn a_call_naming_a_document_handle_is_settled_by_the_receipt_for_that_document() {
        let delegate = "cortex_document__delegate_editor_task";
        let mut started = CallToolResult::success(vec![ContentBlock::text("queued")]);
        started.structured_content =
            Some(serde_json::json!({ "interrupted": true, "job_id": "job-2" }));
        let mut delivered = Message::user()
            .with_text("Your editor task notes-fix finished.")
            .with_visibility(false, true);
        delivered.metadata.set_operation_note(
            crate::session::EDITOR_RESULT_NOTE,
            "v1",
            serde_json::json!({
                "idempotency_key": "notes-fix",
                "receipt": {
                    "job_id": "job-2", "status": "completed", "summary": "Notes",
                    "document_id": "723070dc", "document": "document-2", "document_revision": 3,
                },
            }),
        );
        let tools = [delegate.to_string()];
        let messages = vec![
            Message::assistant().with_tool_request(
                "c1",
                Ok(rmcp::model::CallToolRequestParams::new(delegate.to_string()).with_arguments(
                    serde_json::json!({"editor": "quire", "document_id": "document-2", "instruction": "Fix", "idempotency_key": "notes-fix"})
                        .as_object()
                        .unwrap()
                        .clone(),
                )),
            ),
            Message::user().with_tool_response("c1", Ok(started)),
            delivered,
        ];
        let lines = artifact_result_lines(&messages, &tools);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("completed") && lines[0].contains("document document-2"),
            "{lines:?}"
        );
    }

    #[tokio::test]
    async fn a_specialist_wait_is_offered_with_editor_tools_and_refuses_with_nothing_to_wait_for() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let child = reliability_child(&client, &parent, "document:doc", None).await;
        let listed = |result: ListToolsResult| result.tools.iter().any(|tool| tool.name == "wait");
        assert!(!listed(
            client
                .list_tools(&child, None, CancellationToken::new())
                .await
                .unwrap()
        ));

        let mut session = manager.get_session(&child, false).await.unwrap();
        let mut policy = SummonTaskPolicy::from_session(&session).unwrap();
        policy.artifact_result_tools = vec!["cortex_document__delegate_editor_task".to_string()];
        session.extension_data.set_extension_state(
            "summon",
            "v1",
            serde_json::to_value(policy).unwrap(),
        );
        manager
            .update(&child)
            .extension_data(session.extension_data)
            .apply()
            .await
            .unwrap();
        assert!(listed(
            client
                .list_tools(&child, None, CancellationToken::new())
                .await
                .unwrap()
        ));

        let ctx = ToolCallContext::new(child.clone(), None, None);
        let refused = client
            .call_tool(&ctx, "wait", None, CancellationToken::new())
            .await
            .unwrap();
        assert!(refused.is_error.unwrap_or(false));
        assert!(!tool_result_ends_turn(&refused));

        let mut historical = Message::user().with_text("Channel news");
        historical.metadata.set_operation_note(
            crate::session::MAILBOX_NOTE,
            "kind",
            serde_json::json!("channel"),
        );
        manager.add_message(&child, &historical).await.unwrap();
        let args = || serde_json::json!({"timeout_s": 1}).as_object().cloned();
        let empty = tokio::time::timeout(
            Duration::from_millis(500),
            client.call_tool(&ctx, "wait", args(), CancellationToken::new()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(empty.is_error, Some(true));

        let unrelated = Message::assistant().with_tool_request(
            "read-1",
            Ok(rmcp::model::CallToolRequestParams::new(
                "cortex_document__channel_read",
            )),
        );
        manager.add_message(&child, &unrelated).await.unwrap();
        let repeated = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(repeated.is_error, Some(true));

        let context = |sequence: u64, publications: &[&str], replies: &[&str]| {
            let mut result = CallToolResult::success(vec![]);
            result.meta = Some(MetaObject(
                serde_json::json!({
                    crate::session::CHANNEL_WAIT_META_KEY: [{
                        "channelId": "c1", "sequence": sequence,
                        "publications": publications, "replies": replies,
                    }]
                })
                .as_object()
                .unwrap()
                .clone(),
            ));
            Message::user().with_tool_response("context", Ok(result))
        };
        manager
            .add_message(&child, &context(1, &["deck"], &[]))
            .await
            .unwrap();
        let publication = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_ne!(publication.is_error, Some(true));
        manager
            .add_message(&child, &context(2, &[], &["question-1"]))
            .await
            .unwrap();
        let reply = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_ne!(reply.is_error, Some(true));
        manager
            .add_message(&child, &context(3, &[], &[]))
            .await
            .unwrap();
        // A delayed join cannot restore a dependency cleared by a newer tool result.
        manager
            .add_message(&child, &context(1, &["deck"], &[]))
            .await
            .unwrap();
        let cleared = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(cleared.is_error, Some(true));

        let editor_call = Message::assistant().with_tool_request(
            "editor",
            Ok(
                rmcp::model::CallToolRequestParams::new("cortex_document__delegate_editor_task")
                    .with_arguments(
                        serde_json::json!({"idempotency_key": "notes"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
            ),
        );
        let mut queued = CallToolResult::success(vec![]);
        queued.structured_content = Some(serde_json::json!({"job_id": "j1", "status": "queued"}));
        queued.meta = Some(MetaObject(
            serde_json::json!({crate::agents::tool_interrupt::INTERRUPTED_META_KEY: true})
                .as_object()
                .unwrap()
                .clone(),
        ));
        manager.add_message(&child, &editor_call).await.unwrap();
        manager
            .add_message(
                &child,
                &Message::user().with_tool_response("editor", Ok(queued)),
            )
            .await
            .unwrap();
        let editor_wait = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_ne!(editor_wait.is_error, Some(true));

        manager
            .send_to_child(&parent, &child, "Use the revised plan.")
            .await
            .unwrap();
        let pending = client
            .call_tool(&ctx, "wait", args(), CancellationToken::new())
            .await
            .unwrap();
        assert_ne!(pending.is_error, Some(true));
        assert!(format!("{:?}", pending.content).contains("Woken"));
    }

    #[test]
    fn wait_context_preserves_other_channels_and_ignores_failed_or_malformed_updates() {
        let notice = |channel: &str, sequence: u64, replies: &[&str]| {
            let mut message = Message::user().with_text("Channel notice");
            message.metadata.set_operation_note(
                crate::session::MAILBOX_NOTE, crate::session::CHANNEL_WAIT_META_KEY,
                serde_json::json!({"channelId": channel, "sequence": sequence, "publications": [], "replies": replies}),
            );
            message
        };
        let mut failed = CallToolResult::error(vec![]);
        failed.meta = Some(MetaObject(
            serde_json::json!({
                crate::session::CHANNEL_WAIT_META_KEY: [
                    {"channelId": "c1", "sequence": 99, "publications": [], "replies": []}
                ]
            })
            .as_object()
            .unwrap()
            .clone(),
        ));
        let mut malformed = Message::user().with_text("Bad context");
        malformed.metadata.set_operation_note(
            crate::session::MAILBOX_NOTE,
            crate::session::CHANNEL_WAIT_META_KEY,
            serde_json::json!({"channelId": "c1", "replies": []}),
        );
        let contexts = SummonClient::channel_wait_contexts(&[
            notice("c1", 1, &["q1"]),
            notice("c2", 2, &[]),
            Message::user().with_tool_response("failed", Ok(failed)),
            malformed,
        ]);
        assert!(contexts["c1"].is_waiting());
        assert!(!contexts["c2"].is_waiting());
    }

    #[test]
    fn only_a_successfully_sent_parent_question_permits_waiting_until_a_reply() {
        let question = Message::assistant().with_tool_request(
            "q1",
            Ok(rmcp::model::CallToolRequestParams::new("message_parent")),
        );
        let mut messages = vec![
            question.clone(),
            Message::user().with_tool_response("q1", Ok(CallToolResult::error(vec![]))),
        ];
        assert!(!SummonClient::parent_reply_pending(&messages));
        messages
            .push(Message::user().with_tool_response("q1", Ok(CallToolResult::success(vec![]))));
        assert!(SummonClient::parent_reply_pending(&messages));
        let mut reply = Message::user().with_text("Parent answer");
        reply.metadata.set_operation_note(
            crate::session::MAILBOX_NOTE,
            "kind",
            serde_json::json!("message"),
        );
        messages.push(reply);
        assert!(!SummonClient::parent_reply_pending(&messages));
        messages.push(question);
        assert!(!SummonClient::parent_reply_pending(&messages));
    }

    #[test]
    fn message_parent_accepts_only_questions() {
        assert!(asks_question(
            "Should the notes follow the four-slide deck?"
        ));
        assert!(!asks_question("I am aligning the notes task now."));
    }

    #[test]
    fn a_send_to_a_task_that_cannot_receive_it_says_how_to_continue() {
        let hint = redelegate_hint("20261003_4");
        assert!(hint.contains("task 20261003_4"));
        assert!(hint.contains("the same artifact_key"));
    }

    #[test]
    fn artifact_assignment_carries_identity_and_previous_results() {
        let report = format!(
            "Task 20260929_14 was cancelled.\n\nNo text content in last message\n\n{ARTIFACT_RESULTS_HEADING}\n- Editor job partial: document doc-1, saved revision 2"
        );
        let previous = artifact_results_from_report(&report);
        assert_eq!(
            previous,
            Some("- Editor job partial: document doc-1, saved revision 2")
        );

        let follow_up = DelegateParams {
            artifact_key: Some("document-4".to_string()),
            artifact_title: Some("Speaker Notes".to_string()),
            previous_task_id: Some("20260929_14".to_string()),
            artifact_status: Some("existing".to_string()),
            ..Default::default()
        };
        let assignment = artifact_assignment(&follow_up, previous).unwrap();
        assert!(assignment.contains("Artifact: document-4, which already exists. Revise it"));
        assert!(assignment.contains("Title: Speaker Notes"));
        assert!(assignment.contains("Follow-up to task 20260929_14"));
        assert!(assignment.contains("document doc-1, saved revision 2"));
        assert!(assignment.contains("continue that document"));

        let new = DelegateParams {
            artifact_key: Some("presentation-3".to_string()),
            artifact_status: Some("new".to_string()),
            ..Default::default()
        };
        assert!(artifact_assignment(&new, None).unwrap().contains(
            "Artifact: presentation-3, a new artifact requested in this turn. Create it."
        ));
        // Without an artifact tool, a new: key stays the artifact key.
        let unresolved = DelegateParams {
            artifact_key: Some("new:slides:quarterly".to_string()),
            ..Default::default()
        };
        assert!(artifact_assignment(&unresolved, None)
            .unwrap()
            .contains("a new artifact requested in this turn"));
        assert!(is_artifact_id("document-12", None));
        assert!(is_artifact_id("presentation-3", Some("presentation")));
        assert!(!is_artifact_id("presentation-3", Some("document")));
        for key in [
            "document-0",
            "document-",
            "document-3a",
            "document:3",
            "new:document:3",
            "-3",
        ] {
            assert!(!is_artifact_id(key, None), "{key}");
        }
        assert!(describe_editor_result(&serde_json::json!({"document_id": "3beac1e9", "document": "document-3", "status": "completed", "document_revision": 2}))
            .starts_with("- Editor job completed: document document-3, saved revision 2"));
        assert!(artifact_assignment(&DelegateParams::default(), None).is_none());

        let source = parse_agent_content(
            "---\nname: cortex-document\nartifact_result_tools: [cortex_document__delegate_editor_task]\nartifact_tool: uthereal_cortex__resolve_artifact\nconnect_tool: uthereal_cortex__open_channel\nconnect_reminder: Connect them.\n---\nWrite.",
            Path::new("document.md"),
        )
        .unwrap();
        assert_eq!(
            source.properties["artifact_result_tools"],
            serde_json::json!(["cortex_document__delegate_editor_task"])
        );
        // Source properties delegate reads come through the agent file.
        assert_eq!(
            source.properties["artifact_tool"],
            serde_json::json!("uthereal_cortex__resolve_artifact")
        );
        assert_eq!(
            source.properties["connect_tool"],
            serde_json::json!("uthereal_cortex__open_channel")
        );
        assert_eq!(
            source.properties["connect_reminder"],
            serde_json::json!("Connect them.")
        );
        let plain = parse_agent_content(
            "---\nname: cortex-internet\n---\nSearch.",
            Path::new("internet.md"),
        )
        .unwrap();
        assert!(!plain.properties.contains_key("artifact_tool"));
    }

    fn create_test_context() -> PlatformExtensionContext {
        create_test_context_with_session_manager(Arc::new(
            crate::session::SessionManager::instance(),
        ))
    }

    fn create_test_context_with_session_manager(
        session_manager: Arc<crate::session::SessionManager>,
    ) -> PlatformExtensionContext {
        PlatformExtensionContext {
            extension_manager: None,
            session_manager,
            scheduler: None,
            session: None,
            use_login_shell_path: false,
        }
    }

    async fn create_test_subagent_session(
        session_manager: &crate::session::SessionManager,
        working_dir: &Path,
        messages: &[Message],
    ) -> String {
        let session = session_manager
            .create_session(
                working_dir.to_path_buf(),
                "Background task".to_string(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        for message in messages {
            session_manager
                .add_message(&session.id, message)
                .await
                .unwrap();
        }
        session.id
    }

    async fn reliability_fixture() -> (
        TempDir,
        Arc<crate::session::SessionManager>,
        String,
        SummonClient,
    ) {
        let directory = TempDir::new().unwrap();
        let manager = Arc::new(crate::session::SessionManager::new(
            directory.path().join("sessions"),
        ));
        let parent = manager
            .create_session(
                directory.path().to_path_buf(),
                "Coordinator".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let client = SummonClient::new(create_test_context_with_session_manager(Arc::clone(
            &manager,
        )))
        .unwrap();
        (directory, manager, parent.id, client)
    }

    async fn reliability_child(
        client: &SummonClient,
        parent: &str,
        key: &str,
        previous: Option<&str>,
    ) -> String {
        let session = client
            .context
            .session_manager
            .get_session(parent, false)
            .await
            .unwrap();
        let provider = Arc::new(
            crate::providers::testprovider::TestProvider::new_replaying(
                session
                    .working_dir
                    .join("unused-records.json")
                    .display()
                    .to_string(),
            )
            .unwrap(),
        );
        let config = TaskConfig::new(
            provider,
            goose_providers::model::ModelConfig::new("test-model"),
            parent,
            &session.working_dir,
            Vec::new(),
        );
        client
            .create_subagent_session(
                &config,
                "Specialist".to_string(),
                Some(&SummonTaskPolicy {
                    event_driven_parent: true,
                    artifact_key: Some(key.to_string()),
                    previous_task_id: previous.map(str::to_string),
                    ..Default::default()
                }),
                "inline",
            )
            .await
            .unwrap()
            .id
    }

    /// A running child that works on `key` and reads `reference` as a read-only reference.
    async fn reference_child(
        client: &SummonClient,
        parent: &str,
        key: &str,
        reference: &str,
    ) -> String {
        let session = client
            .context
            .session_manager
            .get_session(parent, false)
            .await
            .unwrap();
        let provider = Arc::new(
            crate::providers::testprovider::TestProvider::new_replaying(
                session
                    .working_dir
                    .join("unused-records.json")
                    .display()
                    .to_string(),
            )
            .unwrap(),
        );
        let config = TaskConfig::new(
            provider,
            goose_providers::model::ModelConfig::new("test-model"),
            parent,
            &session.working_dir,
            Vec::new(),
        );
        let child = client
            .create_subagent_session(
                &config,
                "Specialist".to_string(),
                Some(&SummonTaskPolicy {
                    event_driven_parent: true,
                    artifact_key: Some(key.to_string()),
                    references: vec![ArtifactReference {
                        artifact: reference.to_string(),
                        revision_id: Some("revision-2".to_string()),
                        revision: Some(2),
                        title: Some("Deck".to_string()),
                    }],
                    ..Default::default()
                }),
                "inline",
            )
            .await
            .unwrap()
            .id;
        let token = CancellationToken::new();
        let future_token = token.clone();
        let (handle, completion) = spawn_background_task(async move {
            future_token.cancelled().await;
            Ok("Stopped".to_string())
        });
        client.background_tasks.lock().await.insert(
            child.clone(),
            reliability_running(&child, parent, token, handle, completion),
        );
        child
    }

    #[tokio::test]
    async fn a_reference_stays_frozen_until_a_later_task_edits_it_which_revokes_it_for_good() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let reader = reference_child(&client, &parent, "document-2", "presentation-1").await;
        assert_eq!(
            client
                .referencing_tasks(&parent, "presentation-1")
                .await
                .unwrap(),
            vec![(reader.clone(), Some("document-2".to_string()))]
        );
        assert!(!manager
            .reference_revoked(&parent, &reader, "presentation-1")
            .await
            .unwrap());

        // A task delegated later to edit the artifact revokes the earlier reader's access.
        let editor = reliability_child(&client, &parent, "presentation-1", None).await;
        assert!(manager
            .reference_revoked(&parent, &reader, "presentation-1")
            .await
            .unwrap());
        assert!(client
            .referencing_tasks(&parent, "presentation-1")
            .await
            .unwrap()
            .is_empty());
        let references = manager
            .task_references(
                &parent,
                &reader,
                &serde_json::json!({"references": [{"artifact": "presentation-1", "revision_id": "revision-2"}]}),
            )
            .await
            .unwrap();
        assert_eq!(references.len(), 1);
        assert!(references[0].revoked);
        assert_eq!(references[0].revision_id.as_deref(), Some("revision-2"));

        // While the editor runs, the artifact cannot become a reference.
        let token = CancellationToken::new();
        let future_token = token.clone();
        let (handle, completion) = spawn_background_task(async move {
            future_token.cancelled().await;
            Ok("Stopped".to_string())
        });
        client.background_tasks.lock().await.insert(
            editor.clone(),
            reliability_running(&editor, &parent, token, handle, completion),
        );
        let owners = HashMap::from([(
            (parent.clone(), "presentation-1".to_string()),
            editor.clone(),
        )]);
        let refused = client
            .check_references_frozen(&parent, &["presentation-1".to_string()], &owners)
            .await
            .unwrap_err();
        assert!(refused.starts_with("This delegation was rejected and no task started"));
        assert!(refused.contains(&format!("already assigned to running task {editor}")));
        assert!(refused.contains("do it through a channel"));

        // A reader delegated after the edit reads the new saved state: not revoked.
        let later = reference_child(&client, &parent, "document-3", "presentation-1").await;
        assert!(!manager
            .reference_revoked(&parent, &later, "presentation-1")
            .await
            .unwrap());
    }

    #[test]
    fn references_are_named_in_the_assignment_and_revocation_tells_the_coordinator_how_to_keep_following(
    ) {
        let params = DelegateParams {
            artifact_key: Some("document-2".to_string()),
            artifact_status: Some("new".to_string()),
            references: vec![ArtifactReference {
                artifact: "presentation-1".to_string(),
                revision_id: Some("revision-id".to_string()),
                revision: Some(3),
                title: Some("Quarterly review".to_string()),
            }],
            ..Default::default()
        };
        let assignment = artifact_assignment(&params, None).unwrap();
        assert!(assignment.contains(
            "- Read-only references: presentation-1 (\"Quarterly review\"), saved revision 3. Read them with read_document; you cannot edit them"
        ));
        let message = revocation_message("presentation-1", "20261007_4", Some("document-2"));
        assert!(message.starts_with(
            "Read access to presentation-1 was revoked for running task 20261007_4 (document-2)"
        ));
        assert!(
            message.contains("open a channel with presentation-1 leading and document-2 following")
        );
    }

    fn reliability_running(
        task_id: &str,
        parent: &str,
        token: CancellationToken,
        handle: JoinHandle<Result<String>>,
        completion_token: CancellationToken,
    ) -> BackgroundTask {
        BackgroundTask {
            id: task_id.to_string(),
            parent_session_id: parent.to_string(),
            non_blocking: true,
            completion_delivery_error: Arc::new(Mutex::new(None)),
            terminal_status: Arc::new(Mutex::new(None)),
            description: "Specialist".to_string(),
            started_at: Instant::now(),
            turns: Arc::new(AtomicU32::new(0)),
            last_activity: Arc::new(AtomicU64::new(0)),
            handle,
            cancellation_token: token,
            completion_token,
            notification_sink: buffered_notification_sink(Vec::new()),
        }
    }

    #[test]
    fn reliability_frontmatter_preserves_guard_and_event_policy() {
        let source = parse_agent_content("---\nname: cortex-slides\nartifact_guard: true\nevent_driven_parent: true\n---\nCreate slides.", Path::new("slides.md")).unwrap();
        assert_eq!(source.properties["artifact_guard"], serde_json::json!(true));
        assert_eq!(
            source.properties["event_driven_parent"],
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn artifact_partial_is_durable_even_when_specialist_claims_completion() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let task = reliability_child(&client, &parent, "document:doc", None).await;
        let mut session = manager.get_session(&task, false).await.unwrap();
        let mut policy = SummonTaskPolicy::from_session(&session).unwrap();
        let tool = "cortex_document__delegate_editor_task";
        policy.artifact_result_tools = vec![tool.to_owned()];
        session.extension_data.set_extension_state(
            "summon",
            "v1",
            serde_json::to_value(policy).unwrap(),
        );
        manager
            .update(&task)
            .extension_data(session.extension_data)
            .apply()
            .await
            .unwrap();
        let mut result = CallToolResult::success(vec![ContentBlock::text("Saved")]);
        result.structured_content = Some(
            serde_json::json!({"document_id":"doc", "status":"partial", "document_revision":3, "remaining_work":"One identified citation concern remains."}),
        );
        let messages = vec![
            Message::assistant()
                .with_tool_request("edit", Ok(rmcp::model::CallToolRequestParams::new(tool))),
            Message::user().with_tool_response("edit", Ok(result)),
        ];
        manager
            .replace_conversation(
                &task,
                &crate::conversation::Conversation::new_unvalidated(messages),
            )
            .await
            .unwrap();
        SummonClient::enqueue_task_completion(
            &manager,
            &task,
            &Ok("All requested work is complete.".to_owned()),
            TaskTerminalStatus::Completed,
        )
        .await
        .unwrap();
        let report = manager
            .terminal_report_for_child(&parent, &task)
            .await
            .unwrap()
            .unwrap();
        assert!(report
            .body
            .starts_with(&format!("Task {task} ended with partial artifact output.")));
        assert!(!report.body.contains("completed successfully"));
        assert!(report
            .body
            .contains("Remaining work: One identified citation concern remains."));
        assert_eq!(
            report.task_outcome().unwrap().unwrap().status,
            TaskTerminalStatus::Completed
        );
        let recovered = client.recovered_task_result(&parent, &task).await.unwrap();
        assert_eq!(recovered.status, "completed");
        assert!(recovered.content[0]
            .as_text()
            .unwrap()
            .text
            .contains("ended with partial artifact output"));
    }

    #[tokio::test]
    async fn typed_task_status_comes_from_execution_not_output_prose() {
        let token = CancellationToken::new();
        let (output, status) =
            task_execution_outcome(async { Ok("failed and cancelled".to_string()) }, &token).await;
        assert!(output.is_ok());
        assert_eq!(status, TaskTerminalStatus::Completed);
        let (_, status) = task_execution_outcome(
            async { Err(anyhow::anyhow!("completed successfully")) },
            &token,
        )
        .await;
        assert_eq!(status, TaskTerminalStatus::Failed);
        let (_, status) = task_execution_outcome(
            async {
                panic!("native panic");
                #[allow(unreachable_code)]
                Ok(String::new())
            },
            &token,
        )
        .await;
        assert_eq!(status, TaskTerminalStatus::Panicked);
        token.cancel();
        let (_, status) =
            task_execution_outcome(async { Ok("saved partial result".to_string()) }, &token).await;
        assert_eq!(status, TaskTerminalStatus::Cancelled);
    }

    #[tokio::test]
    #[serial]
    async fn reliability_real_delegation_activates_frontmatter_guard_and_forces_async() {
        let (directory, manager, parent, _client) = reliability_fixture().await;
        let agents = directory.path().join(".goose/agents");
        fs::create_dir_all(&agents).unwrap();
        fs::write(agents.join("cortex-slides.md"), "---\nname: cortex-slides\nartifact_guard: true\nevent_driven_parent: true\n---\nCreate slides.").unwrap();
        let provider: Arc<dyn crate::providers::base::Provider> = Arc::new(
            crate::providers::testprovider::TestProvider::new_replaying(
                directory
                    .path()
                    .join("unused-records.json")
                    .display()
                    .to_string(),
            )
            .unwrap(),
        );
        let extensions = Arc::new(
            crate::agents::extension_manager::ExtensionManager::new_without_provider(
                directory.path().to_path_buf(),
            ),
        );
        *extensions.get_provider().lock().await = Some(provider);
        let mut context = extensions.get_context().clone();
        context.session_manager = Arc::clone(&manager);
        context.extension_manager = Some(Arc::downgrade(&extensions));
        manager
            .update(&parent)
            .provider_name("test".to_string())
            .model_config(goose_providers::model::ModelConfig::new("test-model"))
            .apply()
            .await
            .unwrap();
        let client = SummonClient::new(context).unwrap();
        for instructions in [None, Some("   ")] {
            let mut missing = serde_json::json!({"source":"cortex-slides","artifact_key":"new:slides:quarterly","artifact_title":"Quarterly","extensions":[],"async":true}).as_object().unwrap().clone();
            if let Some(instructions) = instructions {
                missing.insert("instructions".to_string(), serde_json::json!(instructions));
            }
            let error = client
                .handle_delegate(&parent, Some(missing), CancellationToken::new(), None)
                .await
                .unwrap_err();
            assert!(error.contains("requires instructions"), "{error}");
        }
        // An empty `parameters` map is ignored rather than refused for a specialist.
        let arguments = serde_json::json!({"source":"cortex-slides","artifact_key":"new:slides:quarterly","artifact_title":"Quarterly","instructions":"Create the report.","parameters":{},"extensions":[],"async":false}).as_object().unwrap().clone();
        let first = client
            .handle_delegate(
                &parent,
                Some(arguments.clone()),
                CancellationToken::new(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            first.structured_content.as_ref().unwrap()["task_status"],
            serde_json::json!("running")
        );
        let structured_admission = &first.structured_content.as_ref().unwrap()["taskAdmission"];
        assert_eq!(
            first.meta.as_ref().unwrap().0["taskAdmission"],
            *structured_admission
        );
        assert_eq!(structured_admission["sourceName"], "cortex-slides");
        assert_eq!(structured_admission["parentSessionId"], parent);
        let id = first.structured_content.unwrap()["subagent_session_id"]
            .as_str()
            .unwrap()
            .to_string();
        let policy = client.task_policy(&id).await.unwrap().unwrap();
        assert!(policy.event_driven_parent);
        assert_eq!(policy.artifact_key.as_deref(), Some("new:slides:quarterly"));
        let duplicate = client
            .handle_delegate(&parent, Some(arguments), CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(
            duplicate.contains("active specialist")
                || duplicate.contains("Review the terminal result"),
            "{duplicate}"
        );
        assert_eq!(
            manager
                .list_sessions_by_types(&[SessionType::SubAgent])
                .await
                .unwrap()
                .len(),
            1
        );
        let inline = client
            .handle_delegate(
                &parent,
                Some(
                    serde_json::json!({
                        "instructions":"Perform a generic task.","extensions":[],"async":true,
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                ),
                CancellationToken::new(),
                None,
            )
            .await
            .unwrap();
        let admission = &inline.structured_content.as_ref().unwrap()["taskAdmission"];
        assert_eq!(admission["sourceName"], "inline");
        assert_eq!(admission["parentSessionId"], parent);
        assert!(admission["attemptKey"].is_null() && admission["parentRunId"].is_null());
        assert_eq!(inline.meta.as_ref().unwrap().0["taskAdmission"], *admission);
        assert_eq!(
            manager
                .task_admission(admission["taskId"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()
                .source_name,
            "inline"
        );
        client.shutdown_session(&parent).await.unwrap();
    }

    #[tokio::test]
    async fn reliability_child_owns_policy_and_parent_messages_are_questions() {
        let (_directory, manager, parent, coordinator) = reliability_fixture().await;
        let child = reliability_child(&coordinator, &parent, "new:slides:quarterly", None).await;
        let policy = coordinator.task_policy(&child).await.unwrap().unwrap();
        assert_eq!(policy.artifact_key.as_deref(), Some("new:slides:quarterly"));
        assert!(policy.event_driven_parent);
        let specialist = SummonClient::new(create_test_context_with_session_manager(Arc::clone(
            &manager,
        )))
        .unwrap();
        let context = ToolCallContext::new(child.clone(), None, None);
        let call = |message: &str| {
            serde_json::json!({ "message": message })
                .as_object()
                .unwrap()
                .clone()
        };
        let blank = specialist
            .call_tool(
                &context,
                "message_parent",
                Some(call("   ")),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(blank.is_error, Some(true));
        assert!(manager
            .pending_session_messages(&parent)
            .await
            .unwrap()
            .is_empty());
        let question = specialist
            .call_tool(
                &context,
                "message_parent",
                Some(call("Should the deck use the 2025 or the 2026 figures?")),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_ne!(question.is_error, Some(true));
        assert_eq!(
            manager
                .pending_session_messages(&parent)
                .await
                .unwrap()
                .len(),
            1
        );
        let schema = specialist.create_message_parent_tool().input_schema;
        assert!(schema["properties"].get("kind").is_none());
    }

    #[tokio::test]
    async fn a_follow_up_continues_the_artifacts_last_task_whatever_the_caller_passed() {
        let (_directory, _manager, parent, coordinator) = reliability_fixture().await;
        let mut params = DelegateParams {
            previous_task_id: Some("(not available)".to_string()),
            ..Default::default()
        };
        coordinator
            .derive_previous_task_id(&parent, &["presentation-9".to_string()], &mut params)
            .await
            .unwrap();
        assert!(params.previous_task_id.is_none());

        let key = "presentation-4".to_string();
        let owner = reliability_child(&coordinator, &parent, &key, None).await;
        let mut follow_up = DelegateParams {
            previous_task_id: Some("20260101_1".to_string()),
            ..Default::default()
        };
        coordinator
            .derive_previous_task_id(&parent, &[key], &mut follow_up)
            .await
            .unwrap();
        assert_eq!(follow_up.previous_task_id.as_deref(), Some(owner.as_str()));
    }

    #[test]
    fn the_artifact_tool_answer_gives_the_id_and_whether_it_is_new() {
        let answer = r#"{"artifact": "Presentation-1", "status": "new"}"#;
        let mut structured = CallToolResult::success(vec![ContentBlock::text(answer)]);
        structured.structured_content = Some(serde_json::from_str(answer).unwrap());
        let text_only = CallToolResult::success(vec![ContentBlock::text(answer)]);
        for result in [structured, text_only] {
            let resolved = artifact_resolution(&result).unwrap();
            assert_eq!(
                (resolved.artifact.as_str(), resolved.status.as_str()),
                ("presentation-1", "new")
            );
        }
        let refused = CallToolResult::error(vec![ContentBlock::text(
            "presentation-1 is a presentation; the document specialist works on document artifacts.",
        )]);
        assert!(artifact_resolution(&refused)
            .unwrap_err()
            .starts_with("presentation-1 is a presentation"));
        let malformed = CallToolResult::success(vec![ContentBlock::text("{}")]);
        assert!(artifact_resolution(&malformed)
            .unwrap_err()
            .contains("could not be resolved"));
    }

    #[test]
    fn a_delegation_names_an_artifact_of_its_specialists_kind_or_asks_for_a_new_one() {
        let params = |key: &str| DelegateParams {
            source: Some("cortex-presentation".to_string()),
            artifact_key: Some(key.to_string()),
            ..Default::default()
        };
        assert_eq!(
            SummonClient::artifact_key(&params("Presentation-4")).unwrap(),
            "presentation-4"
        );
        assert_eq!(
            SummonClient::artifact_key(&params("new:presentation")).unwrap(),
            "new:presentation"
        );
        assert_eq!(
            SummonClient::artifact_key(&params("new:presentation:Deck")).unwrap(),
            "new:presentation:deck"
        );
        for key in [
            "document-4",
            "new:document",
            "new:presentations",
            "document:abc",
            "deck",
        ] {
            let error = SummonClient::artifact_key(&params(key)).unwrap_err();
            assert!(error.contains("presentation-N"), "{key}: {error}");
        }
    }

    #[tokio::test]
    async fn a_parent_running_two_artifact_tasks_is_reminded_once_to_connect_them() {
        async fn start(coordinator: &SummonClient, parent: &str, id: &str) {
            let (handle, completion_token) = spawn_background_task(async move {
                std::future::pending::<()>().await;
                Ok("done".to_string())
            });
            coordinator.background_tasks.lock().await.insert(
                id.to_string(),
                BackgroundTask {
                    id: id.to_string(),
                    parent_session_id: parent.to_string(),
                    non_blocking: true,
                    completion_delivery_error: Arc::new(Mutex::new(None)),
                    terminal_status: Arc::new(Mutex::new(None)),
                    description: id.to_string(),
                    started_at: Instant::now(),
                    turns: Arc::new(AtomicU32::new(1)),
                    last_activity: Arc::new(AtomicU64::new(0)),
                    handle,
                    cancellation_token: CancellationToken::new(),
                    completion_token,
                    notification_sink: buffered_notification_sink(Vec::new()),
                },
            );
            coordinator.artifact_tasks.lock().await.insert(
                (parent.to_string(), format!("new:document:{id}")),
                id.to_string(),
            );
        }
        let (_directory, manager, parent, coordinator) = reliability_fixture().await;
        let reminder = "Connect them.".to_string();
        start(&coordinator, &parent, "a").await;
        assert_eq!(
            coordinator
                .connect_reminder(&parent, "connect", reminder.clone())
                .await,
            None
        );
        start(&coordinator, &parent, "b").await;
        assert_eq!(
            coordinator
                .connect_reminder(&parent, "connect", reminder.clone())
                .await
                .as_deref(),
            Some("Connect them.")
        );
        assert_eq!(
            coordinator
                .connect_reminder(&parent, "connect", reminder.clone())
                .await,
            None
        );

        // A parent that already called the connecting tool is not reminded.
        coordinator.connect_reminded.lock().await.clear();
        let called = Message::assistant().with_tool_request(
            "connect-1",
            Ok(rmcp::model::CallToolRequestParams::new("connect")),
        );
        manager.add_message(&parent, &called).await.unwrap();
        assert_eq!(
            coordinator
                .connect_reminder(&parent, "connect", reminder)
                .await,
            None
        );
    }

    #[tokio::test]
    async fn turn_context_marks_delivered_reports_and_asks_for_a_status_update() {
        let (_directory, manager, parent, coordinator) = reliability_fixture().await;
        let finished = reliability_child(&coordinator, &parent, "new:document:notes", None).await;
        let running = reliability_child(&coordinator, &parent, "new:slides:deck", None).await;
        coordinator
            .event_driven_parents
            .lock()
            .await
            .insert(parent.clone());
        coordinator.completed_tasks.lock().await.insert(
            finished.clone(),
            CompletedTask {
                id: finished.clone(),
                parent_session_id: parent.clone(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Completed,
                description: "Notes".to_string(),
                result: Ok("Created notes.".to_string()),
                turns_taken: 3,
                duration: Duration::from_secs(40),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let (handle, completion_token) = spawn_background_task(async move {
            stopped.cancelled().await;
            Ok("done".to_string())
        });
        coordinator.background_tasks.lock().await.insert(
            running.clone(),
            BackgroundTask {
                id: running.clone(),
                parent_session_id: parent.clone(),
                non_blocking: true,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Deck".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(1)),
                last_activity: Arc::new(AtomicU64::new(0)),
                handle,
                cancellation_token: CancellationToken::new(),
                completion_token,
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        manager
            .enqueue_completion_to_parent(
                &finished,
                &format!("Task {finished} completed successfully.\n\nCreated notes."),
            )
            .await
            .unwrap();
        let completion_id = manager.pending_session_messages(&parent).await.unwrap()[0].id;

        // Not yet delivered: the report is still on its way.
        let moim = coordinator.get_moim(&parent).await.unwrap();
        assert!(moim.contains("its report is pending"));
        assert!(moim.contains("Call wait now"));

        // Delivered in this turn but not yet acknowledged.
        coordinator
            .note_reports_delivered(&parent, completion_id)
            .await;
        let moim = coordinator.get_moim(&parent).await.unwrap();
        assert!(moim.contains("its report is in the message above"));
        assert!(!moim.contains("its report is pending"));
        assert!(moim.contains("A task in the report above has ended"));
        assert!(moim.contains("current status of every requested artifact"));
        assert!(!moim.contains("Call wait now"));

        // A question delivered in the next turn does not ask for an update.
        manager
            .acknowledge_session_messages(&parent, completion_id)
            .await
            .unwrap();
        let question_id = manager
            .send_to_parent(
                &running,
                "Should the deck use the 2025 or the 2026 figures?",
            )
            .await
            .unwrap();
        let moim = coordinator.get_moim(&parent).await.unwrap();
        assert!(moim.contains("Call wait now"));
        coordinator
            .note_reports_delivered(&parent, question_id)
            .await;
        let moim = coordinator.get_moim(&parent).await.unwrap();
        assert!(moim.contains("is a specialist question and no task has ended"));
        assert!(moim.contains("call wait without writing"));
        assert!(!moim.contains("current status of every requested artifact"));

        stop.cancel();
        coordinator.shutdown_session(&parent).await.unwrap();
        assert!(coordinator.delivered_reports.lock().await.is_empty());
    }

    #[tokio::test]
    async fn reliability_ownership_survives_compaction_acknowledgement_and_cache_expiry() {
        let (_directory, manager, parent, original) = reliability_fixture().await;
        let key = "new:slides:quarterly".to_string();
        let first = reliability_child(&original, &parent, &key, None).await;
        manager
            .enqueue_completion_to_parent(
                &first,
                &format!("Task {first} completed successfully.\n\nCreated deck."),
            )
            .await
            .unwrap();
        let last = reliability_child(&original, &parent, &key, Some(&first)).await;
        let report = format!("Task {last} completed successfully.\n\nPartial: saved translation; one unsupported claim remains.");
        manager
            .enqueue_completion_to_parent(&last, &report)
            .await
            .unwrap();
        manager
            .acknowledge_session_messages(&parent, i64::MAX)
            .await
            .unwrap();
        manager
            .replace_conversation(&parent, &crate::conversation::Conversation::default())
            .await
            .unwrap();
        original.completed_tasks.lock().await.insert(
            last.clone(),
            CompletedTask {
                id: last.clone(),
                parent_session_id: parent.clone(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Completed,
                description: "Translation".to_string(),
                result: Ok("Partial".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(1),
                completed_at: Instant::now() - Duration::from_secs(3600),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        original.cleanup_completed_tasks().await;
        assert!(original.completed_tasks.lock().await.is_empty());
        drop(original);
        let recovered = SummonClient::new(create_test_context_with_session_manager(Arc::clone(
            &manager,
        )))
        .unwrap();
        let mut owners = HashMap::from([((parent.clone(), key.clone()), first.clone())]);
        recovered
            .reconstruct_artifact_tasks(&parent, &mut owners, std::slice::from_ref(&key))
            .await
            .unwrap();
        assert_eq!(owners[&(parent.clone(), key.clone())], last);
        recovered
            .check_artifact_owners(&parent, std::slice::from_ref(&key), Some(&last), &owners)
            .await
            .unwrap();
        assert!(recovered
            .check_artifact_owners(&parent, std::slice::from_ref(&key), Some(&first), &owners)
            .await
            .unwrap_err()
            .contains("Review the terminal result"));
        let result = recovered
            .recovered_task_result(&parent, &last)
            .await
            .unwrap();
        assert_eq!(extract_text(&result.content[0]), report);
        assert!(recovered
            .check_artifact_owners(
                &parent,
                &["document:saved-id".to_string()],
                Some(&last),
                &owners
            )
            .await
            .unwrap_err()
            .contains("does not refer"));
        assert!(manager
            .pending_session_messages(&parent)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn reliability_saved_ids_allow_distinct_documents_with_the_same_title() {
        let (_directory, _manager, parent, client) = reliability_fixture().await;
        let params = |id: &str| DelegateParams {
            source: Some("cortex-slides".to_string()),
            artifact_key: Some(format!("slides-{id}")),
            artifact_title: Some("Quarterly report".to_string()),
            ..Default::default()
        };
        let first_key = SummonClient::artifact_key(&params("1")).unwrap();
        let second_key = SummonClient::artifact_key(&params("2")).unwrap();
        assert_ne!(first_key, second_key);
        let first = reliability_child(&client, &parent, &first_key, None).await;
        let owners = HashMap::from([((parent.clone(), first_key), first)]);
        client
            .check_artifact_owners(&parent, &[second_key], None, &owners)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn reliability_missing_terminal_evidence_is_unknown_and_does_not_replace_owner() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let key = "document:existing".to_string();
        let child = reliability_child(&client, &parent, &key, None).await;
        let mut owners = HashMap::new();
        client
            .reconstruct_artifact_tasks(&parent, &mut owners, std::slice::from_ref(&key))
            .await
            .unwrap();
        assert!(client
            .check_artifact_owners(&parent, std::slice::from_ref(&key), Some(&child), &owners)
            .await
            .unwrap_err()
            .contains("unknown outcome"));
        let result = client.recovered_task_result(&parent, &child).await.unwrap();
        assert_eq!(result.status, "unknown");
        assert!(extract_text(&result.content[0]).contains("No replacement task"));
        assert_eq!(owners[&(parent.clone(), key)], child);
        assert!(manager
            .pending_session_messages(&parent)
            .await
            .unwrap()
            .is_empty());
        client
            .check_artifact_owners(
                &parent,
                &["document:independent".to_string()],
                None,
                &owners,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn reliability_unknown_owner_is_scoped_to_its_parent_turn() {
        let (directory, manager, first_parent, client) = reliability_fixture().await;
        let key = "document:existing".to_string();
        let first = reliability_child(&client, &first_parent, &key, None).await;
        let next_parent = manager
            .create_session(
                directory.path().to_path_buf(),
                "Next user turn".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        let mut owners = HashMap::new();
        client
            .reconstruct_artifact_tasks(&first_parent, &mut owners, std::slice::from_ref(&key))
            .await
            .unwrap();
        client
            .reconstruct_artifact_tasks(&next_parent.id, &mut owners, std::slice::from_ref(&key))
            .await
            .unwrap();
        client
            .check_artifact_owners(&next_parent.id, std::slice::from_ref(&key), None, &owners)
            .await
            .unwrap();
        let next = reliability_child(&client, &next_parent.id, &key, None).await;
        let children = manager
            .list_subagent_sessions(&next_parent.id)
            .await
            .unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, next);
        assert_eq!(
            SummonTaskPolicy::from_session(&children[0])
                .unwrap()
                .artifact_key
                .as_deref(),
            Some(key.as_str())
        );
        assert!(client
            .check_artifact_owners(&first_parent, std::slice::from_ref(&key), None, &owners)
            .await
            .unwrap_err()
            .contains("unknown outcome"));
        client
            .check_artifact_owners(&next_parent.id, std::slice::from_ref(&key), None, &owners)
            .await
            .unwrap();
        client
            .reconstruct_artifact_tasks(&next_parent.id, &mut owners, std::slice::from_ref(&key))
            .await
            .unwrap();
        assert_eq!(owners[&(first_parent, key.clone())], first);
        assert_eq!(owners[&(next_parent.id, key)], next);
    }

    #[tokio::test]
    async fn reliability_terminal_lookup_authorizes_parent_even_after_acknowledgement() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let child = reliability_child(&client, &parent, "document:existing", None).await;
        manager
            .enqueue_completion_to_parent(
                &child,
                &format!("Task {child} failed.\n\nThe saved deck is unchanged."),
            )
            .await
            .unwrap();
        manager
            .acknowledge_session_messages(&parent, i64::MAX)
            .await
            .unwrap();
        assert_eq!(
            client
                .recovered_task_result(&parent, &child)
                .await
                .unwrap()
                .status,
            "failed"
        );
        let unrelated = manager
            .create_session(
                manager
                    .get_session(&parent, false)
                    .await
                    .unwrap()
                    .working_dir,
                "Other parent".to_string(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        assert!(manager
            .terminal_report_for_child(&unrelated.id, &child)
            .await
            .unwrap_err()
            .to_string()
            .contains("does not belong"));
    }

    #[tokio::test]
    async fn reliability_delivery_recovery_preserves_failure_and_queues_one_canonical_report() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let child = reliability_child(&client, &parent, "document:existing", None).await;
        client.completed_tasks.lock().await.insert(
            child.clone(),
            CompletedTask {
                id: child.clone(),
                parent_session_id: parent.clone(),
                completion_delivery_error: Some("temporary database outage".to_string()),
                terminal_status: TaskTerminalStatus::Failed,
                description: "Translation".to_string(),
                result: Err("Saved checkpoint; final validation failed".to_string()),
                turns_taken: 2,
                duration: Duration::from_secs(1),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        assert!(!client.has_active_tasks(&parent).await.unwrap());
        client.recover_completion_delivery(&child).await.unwrap();
        let reports = manager.pending_session_messages(&parent).await.unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0].body,
            format!("Task {child} failed.\n\nSaved checkpoint; final validation failed")
        );
        client.completed_tasks.lock().await.clear();
        assert_eq!(
            client
                .recovered_task_result(&parent, &child)
                .await
                .unwrap()
                .status,
            "failed"
        );
    }

    #[tokio::test]
    async fn reliability_event_coordinator_can_discover_cancel_and_recover_without_polling() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let child = reliability_child(&client, &parent, "document:existing", None).await;
        let token = CancellationToken::new();
        let future_token = token.clone();
        let (handle, completion) = spawn_background_task(async move {
            future_token.cancelled().await;
            Ok("Stopped".to_string())
        });
        client.background_tasks.lock().await.insert(
            child.clone(),
            reliability_running(&child, &parent, token, handle, completion),
        );
        client
            .event_driven_parents
            .lock()
            .await
            .insert(parent.clone());
        let tools = client
            .list_tools(&parent, None, CancellationToken::new())
            .await
            .unwrap();
        assert!(tools.tools.iter().any(|tool| tool.name == "load"));
        client.handle_load(&parent, None, None).await.unwrap();
        let args = |cancel: bool| {
            serde_json::json!({"source":child,"peek":true,"cancel":cancel})
                .as_object()
                .unwrap()
                .clone()
        };
        assert!(client
            .handle_load(&parent, Some(args(false)), None)
            .await
            .unwrap_err()
            .contains("reports automatically"));
        let result = client
            .handle_load(&parent, Some(args(true)), None)
            .await
            .unwrap();
        assert_eq!(
            result.meta.unwrap().0["task_status"],
            serde_json::json!("cancellation_requested")
        );
        assert!(manager
            .terminal_report_for_child(&parent, &child)
            .await
            .unwrap()
            .unwrap()
            .body
            .contains("was cancelled"));
    }

    #[tokio::test]
    async fn reliability_cancelled_canonical_reports_retain_partial_results_after_reconstruction() {
        let (_directory, manager, parent, original) = reliability_fixture().await;
        for (index, result) in [
            Ok("Saved revision 7; remaining work: translate the final slide".to_string()),
            Err(anyhow::anyhow!(
                "Saved checkpoint 3; stopped before final save"
            )),
        ]
        .into_iter()
        .enumerate()
        {
            let child = reliability_child(
                &original,
                &parent,
                &format!("document:cancelled-{index}"),
                None,
            )
            .await;
            let expected = match &result {
                Ok(output) => output.clone(),
                Err(error) => error.to_string(),
            };
            SummonClient::enqueue_task_completion(
                &manager,
                &child,
                &result,
                TaskTerminalStatus::Cancelled,
            )
            .await
            .unwrap();
            let report = manager
                .terminal_report_for_child(&parent, &child)
                .await
                .unwrap()
                .unwrap();
            manager
                .acknowledge_session_messages(&parent, report.id)
                .await
                .unwrap();
            // A retry must retain the first durable terminal outcome.
            SummonClient::enqueue_task_completion(
                &manager,
                &child,
                &Ok("later retry".into()),
                TaskTerminalStatus::Completed,
            )
            .await
            .unwrap();
            let reconstructed = SummonClient::new(create_test_context_with_session_manager(
                Arc::clone(&manager),
            ))
            .unwrap();
            let loaded = reconstructed
                .recovered_task_result(&parent, &child)
                .await
                .unwrap();
            assert_eq!(loaded.status, "cancelled");
            assert_eq!(
                extract_text(&loaded.content[0]),
                format!("Task {child} was cancelled.\n\n{expected}")
            );
            assert!(manager
                .pending_session_messages(&parent)
                .await
                .unwrap()
                .is_empty());
        }
    }

    #[tokio::test]
    async fn reliability_late_cancel_preserves_published_completion_in_cache_and_recovery() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let client = Arc::new(client);
        let child =
            reliability_child(&client, &parent, "document:finished-before-cancel", None).await;
        let task_id = child.clone();
        let task_manager = Arc::clone(&manager);
        let (finish, finished) = tokio::sync::oneshot::channel();
        let (handle, completion) = spawn_background_task(async move {
            finished.await.unwrap();
            let result = Ok("Saved final revision 8".to_string());
            SummonClient::enqueue_task_completion(
                &task_manager,
                &task_id,
                &result,
                TaskTerminalStatus::Completed,
            )
            .await
            .unwrap();
            result
        });
        let token = CancellationToken::new();
        let task = reliability_running(&child, &parent, token.clone(), handle, completion.clone());
        let sink = Arc::clone(&task.notification_sink);
        let attachment = sink.lock().await;
        client
            .background_tasks
            .lock()
            .await
            .insert(child.clone(), task);
        let cancel_client = Arc::clone(&client);
        let cancel_child = child.clone();
        let (emitter, _notifications) = notification_channel();
        let cancel = tokio::spawn(async move {
            cancel_client
                .handle_load_task_result(&cancel_child, true, false, Some(emitter))
                .await
        });
        tokio::task::yield_now().await;
        assert!(!token.is_cancelled());
        finish.send(()).unwrap();
        client
            .wait_for_background_task_completion(&child, &completion)
            .await;
        drop(attachment);
        let requested = cancel.await.unwrap().unwrap();
        assert_eq!(requested.status, "cancellation_requested");
        assert!(token.is_cancelled());
        let cached = client
            .handle_load_task_result(&child, false, true, None)
            .await
            .unwrap();
        assert_eq!(cached.status, "completed");
        assert!(extract_text(&cached.content[0]).contains("Saved final revision 8"));
        client
            .completed_tasks
            .lock()
            .await
            .get_mut(&child)
            .unwrap()
            .completed_at = Instant::now() - Duration::from_secs(3600);
        client.cleanup_completed_tasks().await;
        assert!(!client.completed_tasks.lock().await.contains_key(&child));
        let recovered = client.recovered_task_result(&parent, &child).await.unwrap();
        assert_eq!(recovered.status, cached.status);
        assert_eq!(
            extract_text(&recovered.content[0]),
            format!("Task {child} completed successfully.\n\nSaved final revision 8")
        );
        assert_eq!(
            manager
                .pending_session_messages(&parent)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn reliability_abort_before_first_poll_signals_completion() {
        let (_directory, _manager, parent, client) = reliability_fixture().await;
        let polled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed = Arc::clone(&polled);
        let (handle, completion) = spawn_background_task(async move {
            observed.store(true, Ordering::Relaxed);
            std::future::pending::<Result<String>>().await
        });
        handle.abort();
        client.background_tasks.lock().await.insert(
            "unpolled".to_string(),
            reliability_running(
                "unpolled",
                &parent,
                CancellationToken::new(),
                handle,
                completion.clone(),
            ),
        );
        tokio::time::timeout(
            Duration::from_secs(1),
            client.wait_for_background_task_completion("unpolled", &completion),
        )
        .await
        .unwrap();
        assert!(completion.is_cancelled());
        assert!(!polled.load(Ordering::Relaxed));
        assert!(client.background_tasks.lock().await["unpolled"]
            .handle
            .is_finished());
    }

    #[tokio::test]
    async fn reliability_cancelling_execution_keeps_artifact_reserved_until_it_stops() {
        let (_directory, manager, parent, client) = reliability_fixture().await;
        let key = "document:existing".to_string();
        let child = reliability_child(&client, &parent, &key, None).await;
        let token = CancellationToken::new();
        let future_token = token.clone();
        let (release, released) = tokio::sync::oneshot::channel();
        let (handle, completion) = spawn_background_task(async move {
            future_token.cancelled().await;
            released.await.unwrap();
            Ok("Stopped".to_string())
        });
        client.background_tasks.lock().await.insert(
            child.clone(),
            reliability_running(&child, &parent, token.clone(), handle, completion),
        );
        let client = Arc::new(client);
        let cancelling_client = Arc::clone(&client);
        let cancelling_id = child.clone();
        let cancellation = tokio::spawn(async move {
            cancelling_client
                .handle_load_task_result(&cancelling_id, true, false, None)
                .await
        });
        token.cancelled().await;
        let owners = HashMap::from([((parent.clone(), key.clone()), child.clone())]);
        assert!(client.background_tasks.lock().await.contains_key(&child));
        assert!(client
            .check_artifact_owners(&parent, std::slice::from_ref(&key), Some(&child), &owners)
            .await
            .unwrap_err()
            .contains("active specialist"));
        release.send(()).unwrap();
        cancellation.await.unwrap().unwrap();
        assert!(!client.background_tasks.lock().await.contains_key(&child));
        assert!(manager
            .terminal_report_for_child(&parent, &child)
            .await
            .unwrap()
            .is_some());
        client
            .check_artifact_owners(&parent, &[key], Some(&child), &owners)
            .await
            .unwrap();
    }

    #[test]
    fn test_agent_frontmatter_parsing() {
        let agent = r#"---
name: reviewer
model: sonnet
---
You review code."#;
        let source = parse_agent_content(agent, Path::new("")).unwrap();
        assert_eq!(source.name, "reviewer");
        assert!(source.description.contains("sonnet"));
        assert_eq!(
            source
                .properties
                .get("model")
                .and_then(|value| value.as_str()),
            Some("sonnet")
        );
    }

    #[test]
    fn test_resolve_working_dir_relative_subdir() {
        let temp_dir = TempDir::new().unwrap();
        let parent = temp_dir.path().canonicalize().unwrap();
        let subdir = parent.join("sub");
        fs::create_dir(&subdir).unwrap();

        let resolved = resolve_working_dir(&parent, "sub").unwrap();
        assert_eq!(resolved, subdir.canonicalize().unwrap());
    }

    #[test]
    fn test_resolve_working_dir_rejects_traversal_outside_parent() {
        let temp_dir = TempDir::new().unwrap();
        let parent = temp_dir.path().join("parent");
        let sibling = temp_dir.path().join("sibling");
        fs::create_dir(&parent).unwrap();
        fs::create_dir(&sibling).unwrap();

        let err = resolve_working_dir(&parent, "../sibling").unwrap_err();
        assert!(
            err.to_string()
                .contains("outside the parent session directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_resolve_working_dir_rejects_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let parent = temp_dir.path().canonicalize().unwrap();
        let file = parent.join("a.txt");
        fs::write(&file, "hello").unwrap();

        let err = resolve_working_dir(&parent, "a.txt").unwrap_err();
        assert!(
            err.to_string().contains("is not a directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_resolve_working_dir_rejects_nonexistent_path() {
        let temp_dir = TempDir::new().unwrap();
        let parent = temp_dir.path().canonicalize().unwrap();

        let err = resolve_working_dir(&parent, "does-not-exist").unwrap_err();
        assert!(
            err.to_string().contains("could not be resolved"),
            "unexpected error: {err}"
        );
    }
    #[test]
    fn test_agent_scan_skips_non_agent_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let agents_dir = temp_dir.path().join("agents");
        fs::create_dir_all(&agents_dir).unwrap();
        fs::write(
            agents_dir.join("README.md"),
            "---\ntitle: Notes\n---\nThis is not an agent.",
        )
        .unwrap();
        fs::write(
            agents_dir.join("notes.md"),
            "---\nauthor: someone\ntags: [docs]\n---\nJust documentation.",
        )
        .unwrap();
        fs::write(
            agents_dir.join("reviewer.md"),
            "---\nname: reviewer\nmodel: sonnet\n---\nYou review code.",
        )
        .unwrap();
        fs::write(agents_dir.join("plain.md"), "No frontmatter at all.").unwrap();
        fs::write(
            agents_dir.join("broken.md"),
            "---\nname: [unterminated\n---\nBroken YAML.",
        )
        .unwrap();

        let mut sources = Vec::new();
        let mut seen = HashSet::new();
        scan_agents_from_dir(&agents_dir, &mut sources, &mut seen);

        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name, "reviewer");
    }

    #[test]
    fn agent_frontmatter_preserves_specialist_policy() {
        let source = parse_agent_content(
            "---\nname: reviewer\nrequired_extensions: [review-tools]\nrequired_skills: [code-review]\nalways_async: true\nnon_blocking: true\ndelegate_only: true\n---\nReview code.",
            Path::new("reviewer.md"),
        )
        .unwrap();

        assert_eq!(
            source.properties["required_extensions"],
            serde_json::json!(["review-tools"])
        );
        assert_eq!(
            source.properties["required_skills"],
            serde_json::json!(["code-review"])
        );
        assert_eq!(source.properties["always_async"], serde_json::json!(true));
        assert_eq!(source.properties["non_blocking"], serde_json::json!(true));
        assert_eq!(source.properties["delegate_only"], serde_json::json!(true));
    }

    #[cfg(unix)]
    #[test]
    fn agent_scan_rejects_symlinked_source_file() {
        let temp_dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::write(
            outside.path().join("outside.md"),
            "---\nname: outside\n---\nUntrusted agent.",
        )
        .unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("outside.md"),
            temp_dir.path().join("outside.md"),
        )
        .unwrap();

        let mut sources = Vec::new();
        let mut seen = HashSet::new();
        scan_agents_from_dir(temp_dir.path(), &mut sources, &mut seen);

        assert!(sources.is_empty());
    }

    #[test]
    fn test_recipe_scan_skips_non_recipe_project_config_files() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("package.json"),
            r#"{"scripts":{"test":"cargo test"}}"#,
        )
        .unwrap();
        fs::write(
            temp_dir.path().join("tsconfig.json"),
            r#"{"compilerOptions":{"strict":true}}"#,
        )
        .unwrap();
        fs::write(
            temp_dir.path().join("valid.yaml"),
            "title: Valid\ndescription: Real recipe\ninstructions: Run valid steps",
        )
        .unwrap();

        let mut sources = Vec::new();
        let mut seen = HashSet::new();
        scan_recipes_from_dir(
            temp_dir.path(),
            SourceType::Recipe,
            true,
            &mut sources,
            &mut seen,
        );

        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name, "valid");
        assert_eq!(sources[0].description, "Real recipe");
    }

    #[cfg(unix)]
    #[test]
    fn recipe_scan_rejects_symlinked_source_file() {
        let temp_dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::write(
            outside.path().join("outside.yaml"),
            "title: Outside\ndescription: Outside recipe\ninstructions: Untrusted",
        )
        .unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("outside.yaml"),
            temp_dir.path().join("outside.yaml"),
        )
        .unwrap();

        let mut sources = Vec::new();
        let mut seen = HashSet::new();
        scan_recipes_from_dir(
            temp_dir.path(),
            SourceType::Recipe,
            false,
            &mut sources,
            &mut seen,
        );

        assert!(sources.is_empty());
    }

    #[tokio::test]
    async fn test_discover_recipes_and_agents() {
        let temp_dir = TempDir::new().unwrap();

        let recipes = temp_dir.path().join(".goose/recipes");
        fs::create_dir_all(&recipes).unwrap();
        fs::write(
            recipes.join("deploy.yaml"),
            "title: Deploy\ndescription: Deploy to production\ninstructions: Run deploy steps",
        )
        .unwrap();

        let agents = temp_dir.path().join(".goose/agents");
        fs::create_dir_all(&agents).unwrap();
        fs::write(
            agents.join("reviewer.md"),
            "---\nname: reviewer\nmodel: sonnet\ndescription: Code reviewer\n---\nYou review code.",
        )
        .unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let sources = client.discover_filesystem_sources(temp_dir.path());

        let recipe = sources
            .iter()
            .find(|s| s.name == "deploy" && s.source_type == SourceType::Recipe)
            .unwrap();
        assert_eq!(recipe.description, "Deploy to production");
        assert_eq!(recipe.content, "Run deploy steps");

        let agent = sources
            .iter()
            .find(|s| s.name == "reviewer" && s.source_type == SourceType::Agent)
            .unwrap();
        assert_eq!(agent.description, "Code reviewer");
        assert!(agent.content.contains("You review code"));
    }

    #[tokio::test]
    async fn test_recipe_deduplication_local_wins() {
        let temp_dir = TempDir::new().unwrap();

        let local = temp_dir.path().join(".goose/recipes");
        fs::create_dir_all(&local).unwrap();
        fs::write(
            local.join("deploy.yaml"),
            "title: Deploy\ndescription: Local deploy\ninstructions: local steps",
        )
        .unwrap();

        let also_local = temp_dir.path().join(".agents/recipes");
        fs::create_dir_all(&also_local).unwrap();
        fs::write(
            also_local.join("deploy.yaml"),
            "title: Deploy\ndescription: Agents deploy\ninstructions: agents steps",
        )
        .unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let sources = client.discover_filesystem_sources(temp_dir.path());

        let deploys: Vec<_> = sources.iter().filter(|s| s.name == "deploy").collect();
        assert_eq!(deploys.len(), 1);
    }

    #[tokio::test]
    async fn test_load_recipe_source() {
        let temp_dir = TempDir::new().unwrap();

        let recipes = temp_dir.path().join(".goose/recipes");
        fs::create_dir_all(&recipes).unwrap();
        fs::write(
            recipes.join("deploy.yaml"),
            "title: Deploy\ndescription: Deploy to production\ninstructions: Run deploy steps",
        )
        .unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let result = client
            .handle_load_source("test", "deploy", temp_dir.path())
            .await
            .unwrap();

        let text = &result[0].as_text().expect("expected text content").text;
        assert!(text.contains("deploy"));
        assert!(text.contains("Run deploy steps"));
        assert!(text.contains("now available in your context"));
    }

    #[test]
    fn test_invalid_external_subrecipe_content_is_not_returned() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("invalid.yaml");
        fs::write(&path, "api_key: SUPERSECRET\n").unwrap();

        let recipe_file = load_local_recipe_file(path.to_str().unwrap()).unwrap();
        let error =
            SummonClient::format_subrecipe_content("invalid", &recipe_file.content).unwrap_err();

        assert_eq!(error, "Subrecipe 'invalid' is not a valid recipe");
        assert!(!error.contains("SUPERSECRET"));
    }

    #[test]
    fn test_valid_external_subrecipe_content_still_loads() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("child.yaml");
        fs::write(
            &path,
            "title: Child\ndescription: External child\ninstructions: Run child steps",
        )
        .unwrap();

        let recipe_file = load_local_recipe_file(path.to_str().unwrap()).unwrap();
        let content =
            SummonClient::format_subrecipe_content("child", &recipe_file.content).unwrap();

        assert_eq!(content, "Run child steps");
    }

    #[tokio::test]
    async fn test_load_agent_source() {
        let temp_dir = TempDir::new().unwrap();

        let agents = temp_dir.path().join(".goose/agents");
        fs::create_dir_all(&agents).unwrap();
        fs::write(
            agents.join("reviewer.md"),
            "---\nname: reviewer\nmodel: sonnet\ndescription: Code reviewer\n---\nYou review code carefully.",
        )
        .unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let error = client
            .handle_load_source("test", "reviewer", temp_dir.path())
            .await
            .unwrap_err();

        assert!(error.contains("delegate-only"));
    }

    #[tokio::test]
    async fn test_load_nonexistent_source_suggests_similar() {
        let temp_dir = TempDir::new().unwrap();

        let recipes = temp_dir.path().join(".goose/recipes");
        fs::create_dir_all(&recipes).unwrap();
        fs::write(
            recipes.join("deploy.yaml"),
            "title: Deploy\ndescription: Deploy to production\ninstructions: steps",
        )
        .unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let err = client
            .handle_load_source("test", "deploy-prod", temp_dir.path())
            .await
            .unwrap_err();

        assert!(err.contains("not found"));
        assert!(err.contains("deploy"), "should suggest 'deploy': {}", err);
    }

    #[tokio::test]
    async fn test_load_completely_unknown_source() {
        let temp_dir = TempDir::new().unwrap();

        let client = SummonClient::new(create_test_context()).unwrap();
        let err = client
            .handle_load_source("test", "zzz-nonexistent", temp_dir.path())
            .await
            .unwrap_err();

        assert!(err.contains("not found"));
        assert!(err.contains("Use load()"));
    }

    #[tokio::test]
    async fn test_client_tools_and_unknown_tool() {
        let client = SummonClient::new(create_test_context()).unwrap();

        let result = client
            .list_tools("test", None, CancellationToken::new())
            .await
            .unwrap();
        let names: Vec<_> = result.tools.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"load") && names.contains(&"delegate"));

        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let result = client
            .call_tool(&ctx, "unknown", None, CancellationToken::new())
            .await
            .unwrap();
        assert!(result.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_wait_is_offered_only_to_event_driven_parents_and_needs_outstanding_work() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let result = client
            .list_tools("test", None, CancellationToken::new())
            .await
            .unwrap();
        // No event-driven specialist source: nothing reports back, so no wait.
        assert!(!result.tools.iter().any(|tool| tool.name == "wait"));

        // With no running task and no waiting report, wait refuses so the
        // agent must answer the user instead of sleeping forever.
        let ctx = ToolCallContext::new("test".to_string(), None, None);
        let refused = client
            .call_tool(&ctx, "wait", None, CancellationToken::new())
            .await
            .unwrap();
        assert!(refused.is_error.unwrap_or(false));
        assert!(!tool_result_ends_turn(&refused));
    }

    #[test]
    fn test_only_a_successful_result_with_the_end_turn_flag_ends_the_turn() {
        let flagged = |error: bool| {
            let mut result = if error {
                CallToolResult::error(vec![ContentBlock::text("x")])
            } else {
                CallToolResult::success(vec![ContentBlock::text("x")])
            };
            result.meta = Some(MetaObject(
                serde_json::json!({ END_TURN_META_KEY: true })
                    .as_object()
                    .unwrap()
                    .clone(),
            ));
            result
        };
        assert!(tool_result_ends_turn(&flagged(false)));
        assert!(!tool_result_ends_turn(&flagged(true)));
        assert!(!tool_result_ends_turn(&CallToolResult::success(vec![
            ContentBlock::text("x")
        ])));
    }

    #[tokio::test]
    async fn test_delegate_requires_instructions_when_every_source_is_a_specialist() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let source = |specialist: bool| SourceEntry {
            source_type: SourceType::Agent,
            name: "cortex-slides".to_string(),
            description: String::new(),
            content: String::new(),
            path: String::new(),
            global: false,
            writable: true,
            supporting_files: Vec::new(),
            properties: std::collections::HashMap::from([(
                "event_driven_parent".to_string(),
                serde_json::json!(specialist),
            )]),
        };

        assert!(SummonClient::only_specialist_sources(&[
            source(true),
            source(true)
        ]));
        assert!(!SummonClient::only_specialist_sources(&[
            source(true),
            source(false)
        ]));
        assert!(!SummonClient::only_specialist_sources(&[]));
        assert_eq!(
            client
                .create_delegate_tool(true)
                .input_schema
                .get("required"),
            Some(&serde_json::json!(["instructions"]))
        );
        let specialist_schema = client.create_delegate_tool(true).input_schema;
        assert_eq!(
            specialist_schema.get("additionalProperties"),
            Some(&serde_json::json!(false))
        );
        assert!(specialist_schema["properties"].get("parameters").is_none());
        let general_schema = client.create_delegate_tool(false).input_schema;
        assert!(general_schema.get("required").is_none());
        assert!(general_schema["properties"].get("parameters").is_some());
    }

    #[test]
    fn test_duration_rounding_for_moim() {
        assert_eq!(round_duration(Duration::from_secs(5)), "0s");
        assert_eq!(round_duration(Duration::from_secs(15)), "10s");
        assert_eq!(round_duration(Duration::from_secs(59)), "50s");

        assert_eq!(round_duration(Duration::from_secs(60)), "1m");
        assert_eq!(round_duration(Duration::from_secs(90)), "1m");
        assert_eq!(round_duration(Duration::from_secs(120)), "2m");
    }

    #[test]
    fn test_task_description_formatting() {
        let make_params = |source: Option<&str>, instructions: Option<&str>| DelegateParams {
            source: source.map(String::from),
            instructions: instructions.map(String::from),
            ..Default::default()
        };

        assert_eq!(
            SummonClient::get_task_description(&make_params(Some("recipe"), None)),
            "recipe"
        );
        assert_eq!(
            SummonClient::get_task_description(&make_params(None, Some("do stuff"))),
            "do stuff"
        );
        assert_eq!(
            SummonClient::get_task_description(&make_params(Some("r"), Some("task"))),
            "r: task"
        );
        assert_eq!(
            SummonClient::get_task_description(&make_params(None, None)),
            "Unknown task"
        );
    }

    #[tokio::test]
    async fn test_context_injected_into_adhoc_recipe() {
        let temp_dir = TempDir::new().unwrap();
        let client = SummonClient::new(create_test_context()).unwrap();

        let params = DelegateParams {
            instructions: Some("do the task".to_string()),
            context: Some("background info".to_string()),
            ..Default::default()
        };

        let recipe = client
            .build_delegate_recipe(&params, "test", temp_dir.path())
            .await
            .unwrap();

        assert_eq!(
            recipe.instructions.as_deref(),
            Some("# Reference Context\n\nbackground info")
        );
        assert_eq!(recipe.prompt.as_deref(), Some("do the task"));
    }

    #[test]
    fn test_delegate_fields_nested_under_parameters_get_the_top_level_shape() {
        let nested = DelegateParams {
            parameters: Some(HashMap::from([
                ("source".to_string(), serde_json::json!("writer")),
                ("instructions".to_string(), serde_json::json!("Write it.")),
            ])),
            ..Default::default()
        };
        let error = nested_delegate_fields_error(&nested).unwrap();
        assert!(error.starts_with("instructions, source are fields of delegate itself"));
        assert!(error.contains("at the top level"));

        // A recipe's own parameter that shares a name is left alone.
        let recipe = DelegateParams {
            source: Some("report".to_string()),
            parameters: Some(HashMap::from([(
                "context".to_string(),
                serde_json::json!("quarterly"),
            )])),
            ..Default::default()
        };
        assert!(nested_delegate_fields_error(&recipe).is_none());
    }

    #[test]
    fn test_unknown_delegate_source_lists_the_sources_to_use() {
        let error = unknown_delegate_source_error(
            "search_index",
            &["writer".to_string(), "analyst".to_string()],
        );
        assert!(error.starts_with("Source 'search_index' not found."));
        assert!(error.ends_with("writer, analyst."));
        assert!(unknown_delegate_source_error("search_index", &[]).contains("instructions only"));
    }

    #[test]
    fn test_subrecipe_fixed_values_take_precedence_over_delegate_parameters() {
        let fixed = HashMap::from([("fixed".to_string(), "parent-value".to_string())]);
        let provided = HashMap::from([
            (
                "fixed".to_string(),
                serde_json::Value::String("delegate-value".to_string()),
            ),
            (
                "caller".to_string(),
                serde_json::Value::String("caller-value".to_string()),
            ),
        ]);

        let merged = merge_subrecipe_parameters(Some(&fixed), Some(&provided));

        assert_eq!(
            merged.get("fixed").map(String::as_str),
            Some("parent-value")
        );
        assert_eq!(
            merged.get("caller").map(String::as_str),
            Some("caller-value")
        );
    }

    #[test]
    fn test_build_instructions_with_context_wraps_existing_instructions() {
        assert_eq!(
            build_instructions_with_context("background info", "Run deploy steps"),
            "# Reference Context\n\nbackground info\n\n# Task Instructions\n\nRun deploy steps"
        );
        assert_eq!(
            build_instructions_with_context("background info", ""),
            "# Reference Context\n\nbackground info"
        );
    }

    #[test]
    fn test_validate_delegate_params_rejects_zero_max_turns() {
        let context = create_test_context();
        let client = SummonClient::new(context).unwrap();

        let params = DelegateParams {
            instructions: Some("do something".to_string()),
            max_turns: Some(0),
            ..Default::default()
        };
        let result = client.validate_delegate_params(&params);
        assert_eq!(result, Err("'max_turns' must be at least 1".to_string()));
    }

    #[test]
    fn test_validate_delegate_params_accepts_positive_max_turns() {
        let context = create_test_context();
        let client = SummonClient::new(context).unwrap();

        let params = DelegateParams {
            instructions: Some("do something".to_string()),
            max_turns: Some(5),
            ..Default::default()
        };
        assert!(client.validate_delegate_params(&params).is_ok());
    }

    #[test]
    #[serial]
    fn test_resolve_max_turns_recipe_overrides_env_var() {
        let context = create_test_context();
        let client = SummonClient::new(context).unwrap();

        let session = crate::session::Session {
            recipe: Some(crate::recipe::Recipe {
                version: "1.0.0".to_string(),
                title: String::new(),
                description: String::new(),
                instructions: None,
                prompt: None,
                extensions: None,
                settings: Some(crate::recipe::Settings {
                    goose_provider: None,
                    goose_model: None,
                    temperature: None,
                    max_turns: Some(10),
                }),
                activities: None,
                author: None,
                parameters: None,
                response: None,
                sub_recipes: None,
                retry: None,
            }),
            ..Default::default()
        };

        // Set env var to a different value — recipe should still win
        std::env::set_var("GOOSE_SUBAGENT_MAX_TURNS", "99");
        let result = client.resolve_max_turns(&session);
        std::env::remove_var("GOOSE_SUBAGENT_MAX_TURNS");

        assert_eq!(
            result, 10,
            "recipe settings.max_turns should take priority over env var"
        );
    }

    #[test]
    #[serial]
    fn test_resolve_max_turns_falls_back_to_env_var() {
        let context = create_test_context();
        let client = SummonClient::new(context).unwrap();

        let session = crate::session::Session::default(); // no recipe

        std::env::set_var("GOOSE_SUBAGENT_MAX_TURNS", "7");
        let result = client.resolve_max_turns(&session);
        std::env::remove_var("GOOSE_SUBAGENT_MAX_TURNS");

        assert_eq!(
            result, 7,
            "should fall back to GOOSE_SUBAGENT_MAX_TURNS env var"
        );
    }

    #[test]
    #[serial]
    fn test_resolve_max_turns_falls_back_to_default() {
        let context = create_test_context();
        let client = SummonClient::new(context).unwrap();

        let session = crate::session::Session::default(); // no recipe

        std::env::remove_var("GOOSE_SUBAGENT_MAX_TURNS");
        let result = client.resolve_max_turns(&session);

        assert_eq!(
            result,
            crate::agents::subagent_task_config::DEFAULT_SUBAGENT_MAX_TURNS,
            "should fall back to DEFAULT_SUBAGENT_MAX_TURNS"
        );
    }

    fn empty_recipe() -> crate::recipe::Recipe {
        crate::recipe::Recipe {
            version: "1.0.0".to_string(),
            title: String::new(),
            description: String::new(),
            instructions: None,
            prompt: None,
            extensions: None,
            settings: None,
            activities: None,
            author: None,
            parameters: None,
            response: None,
            sub_recipes: None,
            retry: None,
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_provider_reuses_unregistered_parent_provider() {
        let temp_dir = TempDir::new().unwrap();
        let parent_provider: Arc<dyn crate::providers::base::Provider> = Arc::new(
            crate::providers::testprovider::TestProvider::new_replaying(
                temp_dir.path().join("records.json").display().to_string(),
            )
            .unwrap(),
        );
        let extension_manager = Arc::new(
            crate::agents::extension_manager::ExtensionManager::new_without_provider(
                temp_dir.path().to_path_buf(),
            ),
        );
        *extension_manager.get_provider().lock().await = Some(Arc::clone(&parent_provider));
        let mut context = extension_manager.get_context().clone();
        context.extension_manager = Some(Arc::downgrade(&extension_manager));
        let client = SummonClient::new(context).unwrap();
        let session = crate::session::Session {
            provider_name: Some(parent_provider.get_name().to_string()),
            model_config: Some(goose_providers::model::ModelConfig::new("test-model")),
            ..Default::default()
        };

        let params = DelegateParams {
            provider: Some(parent_provider.get_name().to_string()),
            model: Some("test-model".to_string()),
            ..Default::default()
        };
        let (resolved_provider, _) = client
            .resolve_provider(&params, &empty_recipe(), &session, &[])
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&parent_provider, &resolved_provider));
    }

    #[tokio::test]
    #[serial]
    async fn named_agent_uses_required_session_extension_with_scoped_headers() {
        use crate::session::extension_data::ExtensionState;

        let temp_dir = TempDir::new().unwrap();
        let parent_provider: Arc<dyn crate::providers::base::Provider> = Arc::new(
            crate::providers::testprovider::TestProvider::new_replaying(
                temp_dir.path().join("records.json").display().to_string(),
            )
            .unwrap(),
        );
        let extension_manager = Arc::new(
            crate::agents::extension_manager::ExtensionManager::new_without_provider(
                temp_dir.path().to_path_buf(),
            ),
        );
        *extension_manager.get_provider().lock().await = Some(Arc::clone(&parent_provider));
        let mut context = extension_manager.get_context().clone();
        context.extension_manager = Some(Arc::downgrade(&extension_manager));
        let client = SummonClient::new(context).unwrap();
        let agent_dir = temp_dir.path().join(".agents/agents");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("document.md"),
            "---\nname: document\nrequired_extensions: [scoped-document]\n---\nAuthor documents.",
        )
        .unwrap();
        let mut scoped = ExtensionConfig::streamable_http(
            "scoped-document",
            "http://localhost/mcp",
            "Owned document tools",
            60u64,
        );
        if let ExtensionConfig::StreamableHttp { headers, .. } = &mut scoped {
            headers.insert(
                "Authorization".to_string(),
                "Bearer scoped-test-token".to_string(),
            );
        }
        let mut session = crate::session::Session {
            provider_name: Some(parent_provider.get_name().to_string()),
            model_config: Some(goose_providers::model::ModelConfig::new("test-model")),
            working_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        EnabledExtensionsState::new(vec![
            scoped.clone(),
            ExtensionConfig::streamable_http(
                "unrelated",
                "http://localhost/other",
                "Other tools",
                60u64,
            ),
        ])
        .to_extension_data(&mut session.extension_data)
        .unwrap();
        let params = DelegateParams {
            source: Some("document".to_string()),
            instructions: Some("Write a story".to_string()),
            ..Default::default()
        };
        let config = client
            .build_task_config(&params, &empty_recipe(), &session)
            .await
            .unwrap();
        assert_eq!(config.extensions.len(), 1);
        assert_eq!(
            serde_json::to_value(&config.extensions[0]).unwrap(),
            serde_json::to_value(scoped).unwrap()
        );
        assert_eq!(config.required_extension_names, vec!["scoped-document"]);
    }

    #[tokio::test]
    async fn test_build_task_config_recreates_registered_parent_provider() {
        let temp_dir = TempDir::new().unwrap();
        let parent_provider = providers::create("openai", Vec::new()).await.unwrap();
        let extension_manager = Arc::new(
            crate::agents::extension_manager::ExtensionManager::new_without_provider(
                temp_dir.path().to_path_buf(),
            ),
        );
        *extension_manager.get_provider().lock().await = Some(Arc::clone(&parent_provider));
        let mut context = extension_manager.get_context().clone();
        context.extension_manager = Some(Arc::downgrade(&extension_manager));
        let client = SummonClient::new(context).unwrap();
        let session = crate::session::Session {
            provider_name: Some(parent_provider.get_name().to_string()),
            model_config: Some(goose_providers::model::ModelConfig::new("test-model")),
            working_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let params = DelegateParams {
            extensions: Some(Vec::new()),
            provider: Some(parent_provider.get_name().to_string()),
            model: Some("test-model".to_string()),
            ..Default::default()
        };

        let task_config = client
            .build_task_config(&params, &empty_recipe(), &session)
            .await
            .unwrap();

        assert!(!Arc::ptr_eq(&parent_provider, &task_config.provider));
        assert!(task_config.extensions.is_empty());
    }

    const PARENT_MODEL: &str = "claude-3-5-sonnet-20241022";
    const OVERRIDE_MODEL: &str = "claude-opus-4-6";
    const PROVIDER: &str = "anthropic";

    fn session_with(parent: goose_providers::model::ModelConfig) -> crate::session::Session {
        crate::session::Session {
            provider_name: Some(PROVIDER.to_string()),
            model_config: Some(parent),
            ..Default::default()
        }
    }

    fn resolve_with_override(
        model: Option<&str>,
        parent: goose_providers::model::ModelConfig,
    ) -> goose_providers::model::ModelConfig {
        let client = SummonClient::new(create_test_context()).unwrap();
        let params = DelegateParams {
            model: model.map(String::from),
            ..Default::default()
        };
        client
            .resolve_model_config(
                &params,
                &empty_recipe(),
                &session_with(parent),
                PROVIDER,
                None,
            )
            .expect("resolve_model_config")
    }

    fn parent_config() -> goose_providers::model::ModelConfig {
        goose_providers::model::ModelConfig::new(PARENT_MODEL).with_canonical_limits(PROVIDER)
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_applies_canonical_limits_to_overridden_model() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
        ]);

        let parent = parent_config();
        let overridden = goose_providers::model::ModelConfig::new(OVERRIDE_MODEL)
            .with_canonical_limits(PROVIDER);
        assert_ne!(parent.reasoning, overridden.reasoning);

        let resolved = resolve_with_override(Some(OVERRIDE_MODEL), parent);

        assert_eq!(resolved.model_name, OVERRIDE_MODEL);
        assert_eq!(resolved.max_tokens, overridden.max_tokens);
        assert_eq!(resolved.reasoning, overridden.reasoning);
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_does_not_inherit_provider_specific_request_params() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
        ]);

        // Parent session is a Claude model with anthropic_beta in request_params.
        // When delegate() overrides to a different model (e.g. Gemini), provider-
        // specific params like anthropic_beta must not bleed through — they would
        // cause a 400 INVALID_ARGUMENT from the target API.
        let mut parent = parent_config();
        parent.request_params = Some(HashMap::from([(
            "anthropic_beta".to_string(),
            serde_json::json!("custom-beta-header"),
        )]));

        let resolved = resolve_with_override(Some(OVERRIDE_MODEL), parent);

        assert_eq!(
            resolved
                .request_params
                .as_ref()
                .and_then(|p| p.get("anthropic_beta")),
            None,
            "anthropic_beta must not be inherited by a child session with a different model"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_inherits_thinking_effort_on_override() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
        ]);

        // Reasoning controls are model-family-agnostic and should be inherited,
        // while provider-specific params like anthropic_beta must not.
        let mut parent = parent_config();
        parent.request_params = Some(HashMap::from([
            ("thinking_effort".to_string(), serde_json::json!("high")),
            ("budget_tokens".to_string(), serde_json::json!(8192)),
            (
                "anthropic_beta".to_string(),
                serde_json::json!("custom-beta-header"),
            ),
        ]));

        let resolved = resolve_with_override(Some(OVERRIDE_MODEL), parent);

        assert_eq!(
            resolved
                .request_params
                .as_ref()
                .and_then(|p| p.get("thinking_effort")),
            Some(&serde_json::json!("high")),
            "thinking_effort should be inherited across model families"
        );
        assert_eq!(
            resolved
                .request_params
                .as_ref()
                .and_then(|p| p.get("budget_tokens")),
            Some(&serde_json::json!(8192)),
            "budget_tokens should be inherited across model families"
        );
        assert_eq!(
            resolved
                .request_params
                .as_ref()
                .and_then(|p| p.get("anthropic_beta")),
            None,
            "anthropic_beta must not be inherited alongside reasoning controls"
        );
    }

    fn extract_text(content: &ContentBlock) -> &str {
        use rmcp::model::ContentBlock;
        match content {
            ContentBlock::Text(t) => t.text.as_str(),
            _ => panic!("Expected text content"),
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_env_var_overrides_params_model() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", Some(OVERRIDE_MODEL)),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let params = DelegateParams {
            model: Some("params-model".to_string()),
            ..Default::default()
        };
        let result = client
            .resolve_model_config(
                &params,
                &empty_recipe(),
                &session_with(parent_config()),
                PROVIDER,
                None,
            )
            .expect("resolve_model_config");
        assert_eq!(
            result.model_name, OVERRIDE_MODEL,
            "GOOSE_SUBAGENT_MODEL must take priority over params.model"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_recipe_overrides_env_var() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", Some(OVERRIDE_MODEL)),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let mut recipe = empty_recipe();
        recipe.settings = Some(crate::recipe::Settings {
            goose_provider: None,
            goose_model: Some("recipe-model".to_string()),
            temperature: None,
            max_turns: None,
        });
        let result = client
            .resolve_model_config(
                &DelegateParams::default(),
                &recipe,
                &session_with(parent_config()),
                PROVIDER,
                None,
            )
            .expect("resolve_model_config");
        assert_eq!(
            result.model_name, "recipe-model",
            "recipe settings.goose_model must take priority over GOOSE_SUBAGENT_MODEL"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_provider_recipe_overrides_env_var() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_PROVIDER", Some("openai")),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
            ("ANTHROPIC_API_KEY", Some("test-key")),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let mut recipe = empty_recipe();
        recipe.settings = Some(crate::recipe::Settings {
            goose_provider: Some(PROVIDER.to_string()),
            goose_model: None,
            temperature: None,
            max_turns: None,
        });
        let (resolved_provider, _) = client
            .resolve_provider(
                &DelegateParams::default(),
                &recipe,
                &session_with(parent_config()),
                &[],
            )
            .await
            .expect("resolve_provider");
        assert_eq!(
            resolved_provider.get_name(),
            PROVIDER,
            "recipe settings.goose_provider must take priority over GOOSE_SUBAGENT_PROVIDER"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_recipe_provider_rejects_env_model_of_other_provider() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_PROVIDER", Some("openai")),
            ("GOOSE_SUBAGENT_MODEL", Some("gpt-5.2")),
            ("ANTHROPIC_API_KEY", Some("test-key")),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let mut recipe = empty_recipe();
        recipe.settings = Some(crate::recipe::Settings {
            goose_provider: Some(PROVIDER.to_string()),
            goose_model: None,
            temperature: None,
            max_turns: None,
        });
        let session = crate::session::Session::default();
        let (_, result) = client
            .resolve_provider(&DelegateParams::default(), &recipe, &session, &[])
            .await
            .expect("resolve_provider");

        assert_ne!(
            result.model_name, "gpt-5.2",
            "env model for another provider must not be sent to the recipe provider"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_env_provider_uses_provider_default_model() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_PROVIDER", Some(PROVIDER)),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
            ("ANTHROPIC_API_KEY", Some("test-key")),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let params = DelegateParams::default();
        let default_model = providers::get_from_registry(PROVIDER)
            .await
            .unwrap()
            .metadata()
            .default_model
            .clone();
        let session = crate::session::Session::default();
        let (_, result) = client
            .resolve_provider(&params, &empty_recipe(), &session, &[])
            .await
            .expect("resolve_provider");

        assert_eq!(result.model_name, default_model);
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_env_provider_keeps_matching_params_model() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_PROVIDER", Some(PROVIDER)),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
            ("ANTHROPIC_API_KEY", Some("test-key")),
        ]);

        let client = SummonClient::new(create_test_context()).unwrap();
        let params = DelegateParams {
            provider: Some(PROVIDER.to_string()),
            model: Some(OVERRIDE_MODEL.to_string()),
            ..Default::default()
        };
        let (_, result) = client
            .resolve_provider(
                &params,
                &empty_recipe(),
                &session_with(parent_config()),
                &[],
            )
            .await
            .expect("resolve_provider");

        assert_eq!(result.model_name, OVERRIDE_MODEL);
    }

    #[tokio::test]
    #[serial]
    async fn test_resolve_model_config_dynamic_provider_requires_model() {
        let _env = env_lock::lock_env([
            ("GOOSE_CONTEXT_LIMIT", None::<&str>),
            ("GOOSE_MAX_TOKENS", None::<&str>),
            ("GOOSE_SUBAGENT_MODEL", None::<&str>),
        ]);

        let default_model = providers::get_from_registry("lmstudio")
            .await
            .unwrap()
            .metadata()
            .default_model
            .clone();
        assert!(default_model.is_empty());

        let client = SummonClient::new(create_test_context()).unwrap();
        let params = DelegateParams {
            provider: Some("openai".to_string()),
            model: Some("openai-model".to_string()),
            ..Default::default()
        };
        let session = crate::session::Session {
            provider_name: Some("openai".to_string()),
            model_config: Some(goose_providers::model::ModelConfig::new(
                "parent-openai-model",
            )),
            ..Default::default()
        };
        let error = client
            .resolve_model_config(
                &params,
                &empty_recipe(),
                &session,
                "lmstudio",
                Some(&default_model),
            )
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("No model configured for provider 'lmstudio'"));
    }

    fn test_tool_notification(request_id: &str, subagent_id: &str) -> ServerNotification {
        use crate::agents::subagent_handler::create_tool_notification;
        use crate::conversation::message::MessageContent;
        use rmcp::model::CallToolRequestParams;

        let tool_call = CallToolRequestParams::new("developer__shell").with_arguments(
            serde_json::json!({"command": request_id})
                .as_object()
                .unwrap()
                .clone(),
        );
        let content = MessageContent::tool_request(request_id, Ok(tool_call));
        create_tool_notification(&content, subagent_id).unwrap()
    }

    fn notification_subagent_id(notification: &ServerNotification) -> Option<String> {
        let ServerNotification::LoggingMessageNotification(log) = notification else {
            return None;
        };
        serde_json::to_value(&log.params)
            .ok()?
            .get("data")?
            .get("subagent_id")?
            .as_str()
            .map(str::to_string)
    }

    fn notification_command(notification: &ServerNotification) -> Option<String> {
        let ServerNotification::LoggingMessageNotification(log) = notification else {
            return None;
        };
        serde_json::to_value(&log.params)
            .ok()?
            .get("data")?
            .get("tool_call")?
            .get("arguments")?
            .get("command")?
            .as_str()
            .map(str::to_string)
    }

    fn notification_channel() -> (
        ToolCallNotificationEmitter,
        tokio::sync::mpsc::Receiver<ServerNotification>,
    ) {
        let (sender, receiver) = tokio::sync::mpsc::channel(32);
        (ToolCallNotificationEmitter::new(sender), receiver)
    }

    fn buffered_notification_sink(
        notifications: Vec<ServerNotification>,
    ) -> SharedNotificationSink {
        Arc::new(Mutex::new(NotificationSink::Buffer(notifications)))
    }

    #[test]
    fn test_is_session_id() {
        assert!(is_session_id("20260204_1"));
        assert!(is_session_id("20260204_42"));
        assert!(is_session_id("20260204_999"));
        assert!(!is_session_id("task_12345_0001"));
        assert!(!is_session_id("my-recipe"));
        assert!(!is_session_id("2026020_1"));
        assert!(!is_session_id("20260204"));
    }

    #[tokio::test]
    async fn test_notification_sinks_isolate_concurrent_delegate_calls() {
        let (emitter_a, mut notifications_a) = notification_channel();
        let (emitter_b, mut notifications_b) = notification_channel();
        let sink_a = SummonClient::notification_sink(Some(emitter_a));
        let sink_b = SummonClient::notification_sink(Some(emitter_b));

        let (result_a, result_b) = tokio::join!(
            SummonClient::run_subagent_with_notifications(sink_a, |notification_tx| async move {
                notification_tx
                    .send(test_tool_notification("inner-a", "subagent-a"))
                    .unwrap();
                tokio::task::yield_now().await;
                Ok("delegate-a".to_string())
            }),
            SummonClient::run_subagent_with_notifications(sink_b, |notification_tx| async move {
                notification_tx
                    .send(test_tool_notification("inner-b", "subagent-b"))
                    .unwrap();
                tokio::task::yield_now().await;
                Ok("delegate-b".to_string())
            })
        );
        assert_eq!(result_a.unwrap(), "delegate-a");
        assert_eq!(result_b.unwrap(), "delegate-b");

        let notification_a = notifications_a.recv().await.unwrap();
        let notification_b = notifications_b.recv().await.unwrap();
        assert_eq!(
            notification_subagent_id(&notification_a).as_deref(),
            Some("subagent-a")
        );
        assert_eq!(
            notification_subagent_id(&notification_b).as_deref(),
            Some("subagent-b")
        );
        assert!(notifications_a.try_recv().is_err());
        assert!(notifications_b.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_live_notifications_precede_delegate_result() {
        use crate::agents::tool_execution::{tool_stream, ToolStreamItem};
        use tokio_stream::wrappers::ReceiverStream;

        for _ in 0..32 {
            let (emitter, notifications) = notification_channel();
            let sink = SummonClient::notification_sink(Some(emitter));
            let mut output = tool_stream(
                ReceiverStream::new(notifications),
                futures::stream::empty(),
                async move {
                    let result = SummonClient::run_subagent_with_notifications(
                        sink,
                        |notification_tx| async move {
                            for command in ["inner-live-0", "inner-live-1", "inner-live-2"] {
                                notification_tx
                                    .send(test_tool_notification(command, "subagent-live"))
                                    .unwrap();
                            }
                            Ok("delegate-result".to_string())
                        },
                    )
                    .await
                    .unwrap();
                    Ok::<_, rmcp::model::ErrorData>(CallToolResult::success(vec![
                        ContentBlock::text(result),
                    ]))
                },
            );

            let mut commands = Vec::new();
            let result = loop {
                match output.next().await.unwrap() {
                    ToolStreamItem::Message(notification) => {
                        assert_eq!(
                            notification_subagent_id(&notification).as_deref(),
                            Some("subagent-live")
                        );
                        commands.push(notification_command(&notification).unwrap());
                    }
                    ToolStreamItem::Result(result) => break result,
                    ToolStreamItem::ActionRequired(_) => {
                        panic!("delegate must not request an action")
                    }
                }
            };

            assert_eq!(commands, ["inner-live-0", "inner-live-1", "inner-live-2"]);
            assert!(result.is_ok());
            assert!(output.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn test_async_completion_before_load_replays_notifications() {
        use crate::agents::tool_execution::{tool_stream, ToolStreamItem};
        use tokio_stream::wrappers::ReceiverStream;

        let client = Arc::new(SummonClient::new(create_test_context()).unwrap());
        let task_id = "20260204_1";
        let buffered = vec![test_tool_notification("inner-completed", task_id)];
        client.completed_tasks.lock().await.insert(
            task_id.to_string(),
            CompletedTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Completed,
                description: "Completed task".to_string(),
                result: Ok("done".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(1),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(buffered),
            },
        );
        let (emitter, notifications) = notification_channel();
        let load_client = Arc::clone(&client);
        let mut output = tool_stream(
            ReceiverStream::new(notifications),
            futures::stream::empty(),
            async move {
                let result = load_client
                    .handle_load_task_result(task_id, false, false, Some(emitter))
                    .await
                    .unwrap();
                Ok::<_, rmcp::model::ErrorData>(CallToolResult::success(result.content))
            },
        );

        let ToolStreamItem::Message(notification) = output.next().await.unwrap() else {
            panic!("buffered notification must be emitted before the load result");
        };
        assert_eq!(
            notification_subagent_id(&notification).as_deref(),
            Some(task_id)
        );
        assert_eq!(
            notification_command(&notification).as_deref(),
            Some("inner-completed")
        );
        let ToolStreamItem::Result(result) = output.next().await.unwrap() else {
            panic!("load result must follow buffered notifications");
        };
        assert!(result.is_ok());
        assert!(output.next().await.is_none());
        assert!(!client.completed_tasks.lock().await.contains_key(task_id));
    }

    #[tokio::test]
    async fn test_cancelled_completed_load_remains_retrievable() {
        let client = Arc::new(SummonClient::new(create_test_context()).unwrap());
        let task_id = "20260204_1";
        client.completed_tasks.lock().await.insert(
            task_id.to_string(),
            CompletedTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Completed,
                description: "Completed task".to_string(),
                result: Ok("done".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(1),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(vec![
                    test_tool_notification("inner-0", task_id),
                    test_tool_notification("inner-1", task_id),
                ]),
            },
        );
        let (emitter, mut notifications) = notification_channel();
        let load_client = Arc::clone(&client);
        let load = tokio::spawn(async move {
            load_client
                .handle_load_task_result(task_id, false, false, Some(emitter))
                .await
        });

        let first = notifications.recv().await.unwrap();
        assert_eq!(notification_command(&first).as_deref(), Some("inner-0"));
        load.abort();
        assert!(load.await.unwrap_err().is_cancelled());
        assert!(client.completed_tasks.lock().await.contains_key(task_id));

        let (retry_emitter, mut retry_notifications) = notification_channel();
        let result = client
            .handle_load_task_result(task_id, false, false, Some(retry_emitter))
            .await
            .unwrap();

        assert_eq!(result.status, "completed");
        for command in ["inner-0", "inner-1"] {
            let notification = retry_notifications.try_recv().unwrap();
            assert_eq!(
                notification_command(&notification).as_deref(),
                Some(command)
            );
        }
        assert!(retry_notifications.try_recv().is_err());
        assert!(!client.completed_tasks.lock().await.contains_key(task_id));
    }

    #[tokio::test]
    async fn test_buffered_replay_preserves_order_and_emitter_capacity() {
        let sink = buffered_notification_sink(
            (0..33)
                .map(|index| test_tool_notification(&format!("inner-{index}"), "subagent"))
                .collect(),
        );
        let (emitter, mut notifications) = notification_channel();

        SummonClient::attach_notification_emitter(&sink, Some(emitter)).await;

        for index in 0..32 {
            let notification = notifications.try_recv().unwrap();
            assert_eq!(
                notification_command(&notification),
                Some(format!("inner-{index}"))
            );
        }
        assert!(notifications.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_load_completes_when_caller_does_not_consume_notifications() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let task_id = "20260204_1";
        client.completed_tasks.lock().await.insert(
            task_id.to_string(),
            CompletedTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Completed,
                description: "Completed task".to_string(),
                result: Ok("done".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(1),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(
                    (0..64)
                        .map(|index| test_tool_notification(&format!("inner-{index}"), task_id))
                        .collect(),
                ),
            },
        );
        let (emitter, _notifications) = notification_channel();

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            client.handle_load_task_result(task_id, false, false, Some(emitter)),
        )
        .await
        .expect("load must not wait for a notification consumer")
        .unwrap();

        assert_eq!(result.status, "completed");
    }

    #[tokio::test]
    async fn test_async_task_result_lifecycle() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let temp_dir = TempDir::new().unwrap();

        let result = client
            .handle_load_task_result("20260204_999", false, false, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));

        {
            let notification_sink =
                buffered_notification_sink(vec![test_tool_notification("req1", "20260204_1")]);
            let (handle, completion_token) = spawn_background_task(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok("done".to_string())
            });

            let mut running = client.background_tasks.lock().await;
            running.insert(
                "20260204_1".to_string(),
                BackgroundTask {
                    id: "20260204_1".to_string(),
                    parent_session_id: String::new(),
                    non_blocking: false,
                    completion_delivery_error: Arc::new(Mutex::new(None)),
                    terminal_status: Arc::new(Mutex::new(None)),
                    description: "Running task".to_string(),
                    started_at: Instant::now(),
                    turns: Arc::new(AtomicU32::new(2)),
                    last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                    handle,
                    cancellation_token: CancellationToken::new(),
                    completion_token,
                    notification_sink,
                },
            );
        }

        let (emitter, mut notifications) = notification_channel();
        let (result, notification) = tokio::join!(
            client.handle_load_task_result("20260204_1", false, false, Some(emitter)),
            notifications.recv()
        );
        let result = result.expect("load should wait and return result");
        let text = extract_text(&result.content[0]);
        assert!(text.contains("Completed"));
        assert!(text.contains("done"));

        let notif = notification.expect("load emitter should receive buffered notification");
        if let ServerNotification::LoggingMessageNotification(log) = notif {
            let params = serde_json::to_value(&log.params).unwrap();
            let data = params.get("data").and_then(|v| v.as_object()).unwrap();
            assert_eq!(
                data.get("subagent_id").and_then(|v| v.as_str()),
                Some("20260204_1")
            );
        } else {
            panic!("expected logging notification");
        }

        {
            let mut completed = client.completed_tasks.lock().await;
            completed.insert(
                "20260204_2".to_string(),
                CompletedTask {
                    id: "20260204_2".to_string(),
                    parent_session_id: String::new(),
                    completion_delivery_error: None,
                    terminal_status: TaskTerminalStatus::Completed,
                    description: "Successful task".to_string(),
                    result: Ok("Task completed successfully with output".to_string()),
                    turns_taken: 5,
                    duration: Duration::from_secs(60),
                    completed_at: Instant::now(),
                    notification_sink: buffered_notification_sink(Vec::new()),
                },
            );
            completed.insert(
                "20260204_3".to_string(),
                CompletedTask {
                    id: "20260204_3".to_string(),
                    parent_session_id: String::new(),
                    completion_delivery_error: None,
                    terminal_status: TaskTerminalStatus::Completed,
                    description: "Failed task".to_string(),
                    result: Err("Something went wrong".to_string()),
                    turns_taken: 3,
                    duration: Duration::from_secs(30),
                    completed_at: Instant::now(),
                    notification_sink: buffered_notification_sink(Vec::new()),
                },
            );
        }

        client.completed_tasks.lock().await.insert(
            "20260204_4".to_string(),
            CompletedTask {
                id: "20260204_4".to_string(),
                parent_session_id: "another-parent".to_string(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Cancelled,
                description: "Other parent's task".to_string(),
                result: Err("Task was cancelled".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(10),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        client.completed_tasks.lock().await.insert(
            "20260204_5".to_string(),
            CompletedTask {
                id: "20260204_5".to_string(),
                parent_session_id: String::new(),
                completion_delivery_error: None,
                terminal_status: TaskTerminalStatus::Cancelled,
                description: "Cancelled task".to_string(),
                result: Err("Task was cancelled".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(10),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let moim = client.get_moim("test").await.unwrap();
        assert!(moim.contains("20260204_2"));
        assert!(moim.contains("20260204_3"));
        assert!(moim.contains("completion is reported automatically"));
        assert!(!moim.contains("to get result"));
        assert!(moim.contains("\"Cancelled task\" - cancelled in"));
        assert!(!moim.contains("20260204_4"));

        client
            .event_driven_parents
            .lock()
            .await
            .insert("test".to_string());
        assert!(
            client.get_moim("test").await.is_none(),
            "delivered reports are not repeated to an event-driven parent"
        );
        client.event_driven_parents.lock().await.remove("test");

        let discovery = client
            .handle_load_discovery("test", temp_dir.path())
            .await
            .unwrap();
        let discovery_text = extract_text(&discovery[0]);
        assert!(discovery_text.contains("Completed Tasks (awaiting retrieval)"));
        assert!(discovery_text.contains("20260204_2"));
        assert!(discovery_text.contains("20260204_3"));

        let result = client
            .handle_load_task_result("20260204_2", false, false, None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        assert!(text.contains("20260204_2"));
        assert!(text.contains("Successful task"));
        assert!(text.contains("✓ Completed"));
        assert!(text.contains("1m"));
        assert!(text.contains("5 turns"));
        assert!(text.contains("Task completed successfully with output"));
        assert_eq!(result.status, "completed");
        assert_eq!(result.turns, Some(5));

        assert!(!client
            .completed_tasks
            .lock()
            .await
            .contains_key("20260204_2"));

        let result = client
            .handle_load_task_result("20260204_3", false, false, None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        assert!(text.contains("✗ Failed"));
        assert!(text.contains("Error: Something went wrong"));
        assert_eq!(result.status, "failed");

        let result = client
            .handle_load_task_result("20260204_3", false, false, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));

        client
            .handle_load_task_result("20260204_5", false, false, None)
            .await
            .unwrap();
        assert!(client.get_moim("test").await.is_none());
    }

    #[tokio::test]
    async fn test_completed_task_uses_durable_tool_turns_in_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let session_manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().join("sessions"),
        ));
        let task_id = create_test_subagent_session(
            &session_manager,
            temp_dir.path(),
            &[
                Message::user().with_text("Inspect the project"),
                Message::assistant().with_tool_request(
                    "tool-1",
                    Ok(rmcp::model::CallToolRequestParams::new("test_tool")),
                ),
                Message::user().with_tool_response(
                    "tool-1",
                    Ok(CallToolResult::success(vec![ContentBlock::text("done")])),
                ),
                Message::assistant().with_text("Inspection complete"),
            ],
        )
        .await;
        let client =
            SummonClient::new(create_test_context_with_session_manager(session_manager)).unwrap();

        let handle = tokio::spawn(async { Ok("Inspection complete".to_string()) });
        while !handle.is_finished() {
            tokio::task::yield_now().await;
        }
        client.background_tasks.lock().await.insert(
            task_id.clone(),
            BackgroundTask {
                id: task_id.clone(),
                parent_session_id: String::new(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Inspect the project".to_string(),
                started_at: Instant::now(),
                // Simulate hundreds of streamed message events for two durable turns.
                turns: Arc::new(AtomicU32::new(554)),
                last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                handle,
                cancellation_token: CancellationToken::new(),
                completion_token: CancellationToken::new(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let arguments = serde_json::json!({"source": task_id})
            .as_object()
            .unwrap()
            .clone();
        let result = client
            .handle_load("parent", Some(arguments), None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        let meta = result.meta.unwrap();

        assert!(text.contains("(2 turns)"));
        assert_eq!(meta.0.get("turns_taken"), Some(&serde_json::json!(2)));
        assert_eq!(
            meta.0.get("task_status"),
            Some(&serde_json::json!("completed"))
        );
    }

    #[test]
    fn test_durable_turn_count_ignores_compaction_scaffolding() {
        let compacted_tool_loop = crate::conversation::Conversation::new_unvalidated(vec![
            Message::user()
                .with_text("Previous task")
                .with_visibility(true, false),
            Message::assistant()
                .with_text("Previous result")
                .with_visibility(true, false),
            Message::user()
                .with_text("Inspect the project")
                .with_visibility(true, false),
            Message::assistant()
                .with_tool_request(
                    "tool-1",
                    Ok(rmcp::model::CallToolRequestParams::new("test_tool")),
                )
                .with_visibility(true, false),
            Message::user()
                .with_tool_response(
                    "tool-1",
                    Ok(CallToolResult::success(vec![ContentBlock::text("done")])),
                )
                .with_visibility(true, false),
            // A later compaction archives its earlier agent-only scaffold.
            Message::assistant()
                .with_text("<older summary>")
                .with_visibility(false, false),
            Message::user()
                .with_text("Inspect the project")
                .with_visibility(false, false),
            Message::assistant()
                .with_text("<summary of earlier work>")
                .with_text("Continue from the compacted context")
                .agent_only(),
            Message::user()
                .with_text("Inspect the project")
                .agent_only(),
            Message::assistant().with_text("Inspection complete"),
        ]);
        assert_eq!(durable_assistant_turn_count(&compacted_tool_loop), 2);

        // Keep the hidden projected user message as a role boundary. Dropping
        // every agent-only message would merge the failed and retried replies.
        let compacted_retry = crate::conversation::Conversation::new_unvalidated(vec![
            Message::user()
                .with_text("Inspect the project")
                .with_visibility(true, false),
            Message::assistant()
                .with_text("Context window exceeded")
                .with_visibility(true, false),
            Message::assistant().with_text("<summary>").agent_only(),
            Message::user()
                .with_text("Inspect the project")
                .agent_only(),
            Message::assistant().with_text("Inspection complete"),
        ]);
        assert_eq!(durable_assistant_turn_count(&compacted_retry), 2);
    }

    #[tokio::test]
    async fn test_cancel_running_task() {
        let temp_dir = TempDir::new().unwrap();
        let session_manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().join("sessions"),
        ));
        let task_id = create_test_subagent_session(
            &session_manager,
            temp_dir.path(),
            &[Message::user().with_text("Analyse the project")],
        )
        .await;
        let task_session_manager = Arc::clone(&session_manager);
        let client =
            SummonClient::new(create_test_context_with_session_manager(session_manager)).unwrap();
        let token = CancellationToken::new();
        let notification_sink = buffered_notification_sink(Vec::new());
        let task_notification_sink = Arc::clone(&notification_sink);
        let task_token = token.clone();
        let task_notification_id = task_id.clone();

        let (handle, completion_token) = spawn_background_task(async move {
            task_token.cancelled().await;
            task_session_manager
                .add_message(
                    &task_notification_id,
                    &Message::assistant().with_text("Partial result"),
                )
                .await
                .unwrap();
            task_notification_sink
                .lock()
                .await
                .route(test_tool_notification("cancel", &task_notification_id));
            Ok("cancelled gracefully".to_string())
        });
        {
            let mut running = client.background_tasks.lock().await;
            running.insert(
                task_id.clone(),
                BackgroundTask {
                    id: task_id.clone(),
                    parent_session_id: String::new(),
                    non_blocking: false,
                    completion_delivery_error: Arc::new(Mutex::new(None)),
                    terminal_status: Arc::new(Mutex::new(None)),
                    description: "Cancellable task".to_string(),
                    started_at: Instant::now(),
                    // This stale event count must be replaced after cancellation.
                    turns: Arc::new(AtomicU32::new(3)),
                    last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                    handle,
                    cancellation_token: token.clone(),
                    completion_token,
                    notification_sink,
                },
            );
        }

        let (emitter, mut notifications) = notification_channel();
        let (result, notification) = tokio::join!(
            client.handle_load_task_result(&task_id, true, false, Some(emitter)),
            notifications.recv()
        );
        let result = result.unwrap();
        let text = extract_text(&result.content[0]);
        assert!(text.contains("Cancellation requested"));
        assert!(text.contains(&task_id));
        assert!(text.contains("Cancellable task"));
        assert!(text.contains("cancelled gracefully"));
        assert_eq!(result.status, "cancellation_requested");
        assert!(text.contains("automatic completion report is authoritative"));
        assert_eq!(result.turns, Some(1));
        assert_eq!(
            notification_subagent_id(&notification.unwrap()).as_deref(),
            Some(task_id.as_str())
        );
        assert!(token.is_cancelled());
        assert!(!client.background_tasks.lock().await.contains_key(&task_id));
    }

    #[tokio::test]
    async fn cancel_of_already_finished_task_returns_completion() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let task_id = "20260204_1";
        client.background_tasks.lock().await.insert(
            task_id.to_string(),
            BackgroundTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Finished task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(1)),
                last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                handle: tokio::spawn(async { Ok("finished output".to_string()) }),
                cancellation_token: CancellationToken::new(),
                completion_token: CancellationToken::new(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );
        while !client
            .background_tasks
            .lock()
            .await
            .get(task_id)
            .unwrap()
            .handle
            .is_finished()
        {
            tokio::task::yield_now().await;
        }

        let result = client
            .handle_load_task_result(task_id, true, false, None)
            .await
            .unwrap();

        assert_eq!(result.status, "completed");
        assert!(extract_text(&result.content[0]).contains("finished output"));
    }

    #[tokio::test]
    async fn cancellation_delivery_failure_remains_visible_to_parent() {
        let temp_dir = TempDir::new().unwrap();
        let session_manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().join("sessions"),
        ));
        let task_id = create_test_subagent_session(
            &session_manager,
            temp_dir.path(),
            &[Message::user().with_text("Analyse the project")],
        )
        .await;
        let client =
            SummonClient::new(create_test_context_with_session_manager(session_manager)).unwrap();
        let cancellation_token = CancellationToken::new();
        let wait_token = cancellation_token.clone();
        let (handle, completion_token) = spawn_background_task(async move {
            wait_token.cancelled().await;
            Ok("stopped".to_string())
        });
        client.background_tasks.lock().await.insert(
            task_id.clone(),
            BackgroundTask {
                id: task_id.clone(),
                parent_session_id: "parent".to_string(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Cancellable task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(0)),
                last_activity: Arc::new(AtomicU64::new(0)),
                handle,
                cancellation_token,
                completion_token,
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let error = client
            .handle_load_task_result(&task_id, true, false, None)
            .await
            .unwrap_err();
        assert!(error.contains("parent report could not be delivered"));
        assert!(client.completed_tasks.lock().await.contains_key(&task_id));
        assert!(client.has_active_tasks("parent").await.is_err());
    }

    #[tokio::test]
    async fn test_cancelled_running_load_remains_retrievable() {
        let client = Arc::new(SummonClient::new(create_test_context()).unwrap());
        let token = CancellationToken::new();
        let task_id = "20260204_1";
        let task_token = token.clone();

        let (handle, completion_token) = spawn_background_task(async move {
            task_token.cancelled().await;
            Ok("cancelled gracefully".to_string())
        });
        client.background_tasks.lock().await.insert(
            task_id.to_string(),
            BackgroundTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Cancellable task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(1)),
                last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                handle,
                cancellation_token: token.clone(),
                completion_token,
                notification_sink: buffered_notification_sink(vec![
                    test_tool_notification("inner-0", task_id),
                    test_tool_notification("inner-1", task_id),
                ]),
            },
        );

        let (emitter, mut notifications) = notification_channel();
        let load_client = Arc::clone(&client);
        let load = tokio::spawn(async move {
            load_client
                .handle_load_task_result(task_id, true, false, Some(emitter))
                .await
        });

        let first = notifications.recv().await.unwrap();
        assert_eq!(notification_command(&first).as_deref(), Some("inner-0"));
        load.abort();
        assert!(load.await.unwrap_err().is_cancelled());
        assert!(client.background_tasks.lock().await.contains_key(task_id));
        assert!(!token.is_cancelled());

        let (retry_emitter, mut retry_notifications) = notification_channel();
        let result = client
            .handle_load_task_result(task_id, true, false, Some(retry_emitter))
            .await
            .unwrap();

        assert_eq!(result.status, "cancellation_requested");
        assert!(token.is_cancelled());
        assert!(!client.background_tasks.lock().await.contains_key(task_id));

        let commands: Vec<String> = std::iter::from_fn(|| retry_notifications.try_recv().ok())
            .filter_map(|notification| notification_command(&notification))
            .collect();
        assert!(
            commands == ["inner-0", "inner-1"] || commands == ["inner-1"],
            "retry must replay the remaining notifications, with at-least-once delivery allowed"
        );
    }

    #[tokio::test]
    async fn test_cancelled_waiting_load_remains_retrievable() {
        let client = Arc::new(SummonClient::new(create_test_context()).unwrap());
        let task_id = "20260204_1";
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let (handle, completion_token) = spawn_background_task(async move {
            finish_rx.await.unwrap();
            Ok("done".to_string())
        });

        client.background_tasks.lock().await.insert(
            task_id.to_string(),
            BackgroundTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Running task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(1)),
                last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                handle,
                cancellation_token: CancellationToken::new(),
                completion_token,
                notification_sink: buffered_notification_sink(vec![
                    test_tool_notification("inner-0", task_id),
                    test_tool_notification("inner-1", task_id),
                ]),
            },
        );

        let (emitter, mut notifications) = notification_channel();
        let load_client = Arc::clone(&client);
        let load = tokio::spawn(async move {
            load_client
                .handle_load_task_result(task_id, false, false, Some(emitter))
                .await
        });

        let first = notifications.recv().await.unwrap();
        assert_eq!(notification_command(&first).as_deref(), Some("inner-0"));
        load.abort();
        assert!(load.await.unwrap_err().is_cancelled());
        assert!(client.background_tasks.lock().await.contains_key(task_id));

        finish_tx.send(()).unwrap();
        let (retry_emitter, mut retry_notifications) = notification_channel();
        let result = client
            .handle_load_task_result(task_id, false, false, Some(retry_emitter))
            .await
            .unwrap();

        assert_eq!(result.status, "completed");
        assert!(!client.background_tasks.lock().await.contains_key(task_id));

        let commands: Vec<String> = std::iter::from_fn(|| retry_notifications.try_recv().ok())
            .filter_map(|notification| notification_command(&notification))
            .collect();
        assert!(
            commands == ["inner-0", "inner-1"] || commands == ["inner-1"],
            "retry must replay the remaining notifications, with at-least-once delivery allowed"
        );
    }

    #[tokio::test]
    async fn test_dropped_waiting_load_during_turn_refresh_remains_retrievable() {
        let temp_dir = TempDir::new().unwrap();
        let session_manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().join("sessions"),
        ));
        let task_id = create_test_subagent_session(
            &session_manager,
            temp_dir.path(),
            &[
                Message::user().with_text("Analyse the project"),
                Message::assistant().with_text("done"),
            ],
        )
        .await;
        let client = SummonClient::new(create_test_context_with_session_manager(Arc::clone(
            &session_manager,
        )))
        .unwrap();

        let (handle, completion_token) = spawn_background_task(async { Ok("done".to_string()) });
        while !handle.is_finished() {
            tokio::task::yield_now().await;
        }
        client.background_tasks.lock().await.insert(
            task_id.clone(),
            BackgroundTask {
                id: task_id.clone(),
                parent_session_id: String::new(),
                non_blocking: false,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Finished task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(1)),
                last_activity: Arc::new(AtomicU64::new(current_epoch_millis())),
                handle,
                cancellation_token: CancellationToken::new(),
                completion_token,
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let pool = session_manager.storage().pool().await.unwrap().clone();
        let mut held_connections = Vec::new();
        for _ in 0..pool.options().get_max_connections() {
            held_connections.push(pool.acquire().await.unwrap());
        }
        tokio::task::yield_now().await;

        let mut load = Box::pin(client.handle_load_task_result(&task_id, false, false, None));
        let first_poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(load.as_mut(), cx))
        })
        .await;
        assert!(first_poll.is_pending());
        drop(load);

        assert!(
            client.completed_tasks.lock().await.contains_key(&task_id)
                || client.background_tasks.lock().await.contains_key(&task_id),
            "cancelling a load must not orphan a completed task"
        );

        drop(held_connections);
        let result = client
            .handle_load_task_result(&task_id, false, false, None)
            .await
            .unwrap();
        assert_eq!(result.status, "completed");
        assert_eq!(result.turns, Some(1));
        assert!(extract_text(&result.content[0]).contains("done"));
        assert!(!client.completed_tasks.lock().await.contains_key(&task_id));
        assert!(!client.background_tasks.lock().await.contains_key(&task_id));
    }

    #[tokio::test]
    async fn test_peek_running_task() {
        let temp_dir = TempDir::new().unwrap();
        let session_manager = Arc::new(crate::session::SessionManager::new(
            temp_dir.path().join("sessions"),
        ));
        let task_id = create_test_subagent_session(
            &session_manager,
            temp_dir.path(),
            &[Message::user().with_text("Analyse the project")],
        )
        .await;
        let client = SummonClient::new(create_test_context_with_session_manager(Arc::clone(
            &session_manager,
        )))
        .unwrap();
        let last_activity = Arc::new(AtomicU64::new(0));

        {
            let mut running = client.background_tasks.lock().await;
            running.insert(
                task_id.clone(),
                BackgroundTask {
                    id: task_id.clone(),
                    parent_session_id: String::new(),
                    non_blocking: false,
                    completion_delivery_error: Arc::new(Mutex::new(None)),
                    terminal_status: Arc::new(Mutex::new(None)),
                    description: "Long running analysis".to_string(),
                    started_at: Instant::now(),
                    // Simulate the old stream-event counter after seven fragments.
                    turns: Arc::new(AtomicU32::new(7)),
                    last_activity: Arc::clone(&last_activity),
                    handle: tokio::spawn(async {
                        tokio::time::sleep(Duration::from_secs(1000)).await;
                        Ok("eventual result".to_string())
                    }),
                    cancellation_token: CancellationToken::new(),
                    completion_token: CancellationToken::new(),
                    notification_sink: buffered_notification_sink(Vec::new()),
                },
            );
        }

        let result = client
            .handle_load_task_result(&task_id, false, true, None)
            .await
            .unwrap();
        assert!(extract_text(&result.content[0]).contains("Task is initialising"));

        // Activity can arrive before the assistant block is durably persisted.
        last_activity.store(current_epoch_millis(), Ordering::Relaxed);
        let result = client
            .handle_load_task_result(&task_id, false, true, None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        assert_eq!(result.turns, Some(0));
        assert!(!text.contains("Task is initialising"));

        for index in 0..7 {
            session_manager
                .add_message(
                    &task_id,
                    &Message::assistant().with_text(format!("fragment {index}")),
                )
                .await
                .unwrap();
        }

        // Peek should return status without removing the task
        let result = client
            .handle_load_task_result(&task_id, false, true, None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        assert!(text.contains("Running"));
        assert!(text.contains("Long running analysis"));
        assert!(text.contains("**Turns taken:** 1"));
        assert_eq!(result.turns, Some(1));

        let moim = client.get_moim("test").await.unwrap();
        assert!(moim.contains("1 turns"));

        // Task should still be in background_tasks (not consumed)
        assert!(client.background_tasks.lock().await.contains_key(&task_id));
    }

    #[tokio::test]
    async fn test_peek_nonexistent_task() {
        let client = SummonClient::new(create_test_context()).unwrap();

        let result = client
            .handle_load_task_result("20260204_999", false, true, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn non_blocking_task_load_returns_running_status_without_waiting() {
        let client = SummonClient::new(create_test_context()).unwrap();
        let task_id = "20260204_1";
        let cancellation_token = CancellationToken::new();
        let wait_token = cancellation_token.clone();
        client.background_tasks.lock().await.insert(
            task_id.to_string(),
            BackgroundTask {
                id: task_id.to_string(),
                parent_session_id: String::new(),
                non_blocking: true,
                completion_delivery_error: Arc::new(Mutex::new(None)),
                terminal_status: Arc::new(Mutex::new(None)),
                description: "Non-blocking task".to_string(),
                started_at: Instant::now(),
                turns: Arc::new(AtomicU32::new(0)),
                last_activity: Arc::new(AtomicU64::new(0)),
                handle: tokio::spawn(async move {
                    wait_token.cancelled().await;
                    Ok("done".to_string())
                }),
                cancellation_token,
                completion_token: CancellationToken::new(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let result = tokio::time::timeout(
            Duration::from_millis(100),
            client.handle_load_task_result(task_id, false, false, None),
        )
        .await
        .expect("non-blocking load should return immediately")
        .unwrap();
        assert_eq!(result.status, "running");
        assert!(extract_text(&result.content[0]).contains("Do not poll or sleep"));
        let moim = client.get_moim("test").await.unwrap();
        assert!(moim.contains("Do not poll or sleep"));
        assert!(!moim.contains("to wait for a task"));
        client
            .event_driven_parents
            .lock()
            .await
            .insert("test".to_string());
        let event_driven_moim = client.get_moim("test").await.unwrap();
        assert!(event_driven_moim.contains("you are resumed for each terminal report"));
        assert!(!event_driven_moim.contains("load(source:"));
        assert!(client.background_tasks.lock().await.contains_key(task_id));
    }

    #[tokio::test]
    async fn test_peek_completed_task_returns_result() {
        let client = SummonClient::new(create_test_context()).unwrap();

        {
            let mut completed = client.completed_tasks.lock().await;
            completed.insert(
                "20260204_1".to_string(),
                CompletedTask {
                    id: "20260204_1".to_string(),
                    parent_session_id: String::new(),
                    completion_delivery_error: None,
                    terminal_status: TaskTerminalStatus::Completed,
                    description: "Finished task".to_string(),
                    result: Ok("final output".to_string()),
                    turns_taken: 4,
                    duration: Duration::from_secs(30),
                    completed_at: Instant::now(),
                    notification_sink: buffered_notification_sink(Vec::new()),
                },
            );
        }

        // Peek on a completed task should return the full result (same as non-peek)
        let result = client
            .handle_load_task_result("20260204_1", false, true, None)
            .await
            .unwrap();
        let text = extract_text(&result.content[0]);
        assert!(text.contains("Completed"));
        assert!(text.contains("final output"));

        // Peek must be non-destructive: the result is still retrievable afterwards.
        assert!(client
            .completed_tasks
            .lock()
            .await
            .contains_key("20260204_1"));
        let result = client
            .handle_load_task_result("20260204_1", false, false, None)
            .await
            .unwrap();
        assert!(extract_text(&result.content[0]).contains("final output"));
    }

    #[tokio::test]
    async fn completion_delivery_failure_prevents_successful_idle_state() {
        let client = SummonClient::new(create_test_context()).unwrap();
        client.completed_tasks.lock().await.insert(
            "20260204_1".to_string(),
            CompletedTask {
                id: "20260204_1".to_string(),
                parent_session_id: "parent".to_string(),
                completion_delivery_error: Some("database unavailable".to_string()),
                terminal_status: TaskTerminalStatus::Completed,
                description: "Finished task".to_string(),
                result: Ok("final output".to_string()),
                turns_taken: 1,
                duration: Duration::from_secs(1),
                completed_at: Instant::now(),
                notification_sink: buffered_notification_sink(Vec::new()),
            },
        );

        let error = client.has_active_tasks("parent").await.unwrap_err();
        assert!(error.to_string().contains("database unavailable"));
        let load_error = client
            .handle_load_task_result("20260204_1", false, false, None)
            .await
            .unwrap_err();
        assert!(load_error.contains("parent report could not be delivered"));
        assert!(client
            .completed_tasks
            .lock()
            .await
            .contains_key("20260204_1"));
        assert!(client.has_active_tasks("parent").await.is_err());
        assert!(!client.has_active_tasks("other-parent").await.unwrap());
    }
}
