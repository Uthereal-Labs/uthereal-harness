# Uthereal harness

This fork keeps Goose's CLI, provider configuration, tools, and session storage.
It adds native specialist policies, durable parent/child messages, and session
tracing. There is no separate coordinator process, desktop bundle modification,
Quire dependency, or launcher wrapper.

## Build and run

From this checkout, with the Rust toolchain in `rust-toolchain.toml` installed:

```sh
df -h .
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
cargo build --locked -p goose-cli --bin goose --no-default-features \
  --features rustls-tls,otel,disable-update
./target/debug/goose configure
./target/debug/goose session
```

Allow at least 8–10 GiB of free space for building and testing; this is planning
headroom, not a guaranteed upper bound. If the disk is nearly full, use a larger
volume for `CARGO_TARGET_DIR` before building. The executable then lives under
`$CARGO_TARGET_DIR/debug/goose`. Reuse the same target directory and features for
subsequent builds and checks. `cargo build` and `cargo run` create local build
artifacts and populate Cargo's shared dependency cache; they do not install a
system-wide Goose executable. `cargo clean` removes this checkout's build
artifacts without removing its source or your installed Rust toolchain.

The executable remains named `goose`. Use its explicit path to distinguish it from
an upstream installation. `disable-update` prevents replacing this build with an
upstream release. Add provider/platform features such as `aws-providers` or
`system-keyring` when your deployment needs them. No local model runtime or
desktop dependencies are selected by the command above.

## Docker ACP server

The root `Dockerfile` packages the native HTTP/WebSocket ACP server without the
desktop application, local inference, AWS providers, updater, or OS keyring. It
builds the locked workspace with Rust 1.96.1 and enables `rustls-tls`, `otel`,
and `disable-update`. BuildKit caches Cargo downloads and build artifacts, then
copies the release binary out of the cached target mount.

Create a private `.env` file (excluded by `.dockerignore`). Compose passes it
to the container, including provider-specific variables that are not listed in
the Compose file. Secrets and telemetry headers are runtime configuration and
must not be passed as Docker build arguments:

```dotenv
GOOSE_SERVER__SECRET_KEY=a-long-random-secret
GOOSE_PROVIDER=openai
GOOSE_MODEL=gpt-5
OPENAI_API_KEY=YOUR_PROVIDER_KEY
```

```sh
docker compose up --build
```

Compose publishes the service only on `127.0.0.1:3284` by default. Containers
on the same Compose network reach it at `http://harness:3284`. The
unauthenticated health probe is `GET /health`. ACP traffic uses `/acp` and
authenticates with `X-Secret-Key`:

```sh
set -a
. ./.env
set +a
curl --fail http://127.0.0.1:3284/health
curl -i -H "X-Secret-Key: $GOOSE_SERVER__SECRET_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}' \
  http://127.0.0.1:3284/acp
```

The second command initializes an ACP connection and returns its
`Acp-Connection-Id` response header. ACP uses JSON-RPC over HTTP or WebSocket.
Applications should use an ACP client rather than treating
`/acp` as a REST chat endpoint. Browser WebSocket clients may put the secret
in the `token` query parameter and must add each non-loopback origin with
`--allowed-origin`.

An API backend can call this service over the Docker network; no interactive
terminal or application wrapper is required. The HTTP client lifecycle is:

1. POST a JSON-RPC `initialize` request to `/acp` with `X-Secret-Key` and
   `Content-Type: application/json`. Use `protocolVersion: 1` and
   `clientCapabilities: {}`. Save the response's `Acp-Connection-Id` header.
2. Open GET `/acp` with that connection header, authentication, and
   `Accept: text/event-stream` to receive connection events.
3. POST `session/new` with `params: {"cwd":"/workspace","mcpServers":[]}`
   and the connection header. Its response arrives on the connection stream;
   save `result.sessionId`.
4. Open a second event stream with `Acp-Session-Id` as well. POST
   `session/prompt` with both headers and
   `params: {"sessionId":"…","prompt":[{"type":"text","text":"…"}]}`.
   HTTP 202 means accepted; `session/update` events contain progress and text,
   and the JSON-RPC response matching the request ID marks completion.
5. Send the `session/cancel` notification to cancel an active prompt. Close the
   session when finished, or use `session/delete` to also remove its history.
   DELETE `/acp` with the connection header releases the transport connection.

All requests use JSON-RPC 2.0; requests carry a unique `id`, while notifications
do not. A backend must also handle agent-to-client requests such as permission
prompts if its configured Goose mode requires approval. The session working
directory is a path **inside the container**. ACP authentication grants access
to this harness and its mounted workspace; application tenant authorization and
workspace isolation belong in the calling service/deployment.

For a backend that assigns a persistent working directory to each conversation,
set `GOOSE_ACP_WORKSPACE_ROOT=/workspace/cortex`. With this opt-in setting,
`session/new` creates missing directories beneath that root and reuses their
files on later sessions. The requested `cwd` must be a strict descendant of the
root; traversal and symlink components are rejected. Without the setting, ACP
continues to require an existing directory. This provisions directories; it
does not sandbox the agent's shell or replace application authorization.

The `harness-state` volume backs the absolute `GOOSE_PATH_ROOT` and retains
configuration, sessions, mailbox records, agents, and skills. The separate
`harness-workspace` volume is the agent's working directory; replace it with a
bind mount when the agent must edit a host checkout. The image disables the OS
keyring, so inject provider secrets at runtime and do not copy credentials into
the image.

The runtime includes Git, a shell, curl, and ripgrep. Add your project's language
toolchains and build dependencies in a derived image when tasks require them;
the harness image does not install every possible project toolchain.

Add the OTLP variables from the tracing section below to `.env` before
`docker compose up` to export traces. Compose otherwise disables all three OTLP
signals. TLS is normally terminated by the deployment ingress; direct goose TLS remains
available because `rustls-tls` is compiled in.

The binary is the container entrypoint, so it receives Docker signals directly,
and Compose allows 30 seconds before sending `SIGKILL`. On SIGTERM or Ctrl-C,
the ACP server stops accepting connections, cancels active parent runs, and
allows up to 20 seconds for their child tasks to stop through the normal Summon
lifecycle. The server then returns through the CLI shutdown path so configured
telemetry exporters can flush before the container exits.

Enable the built-in **Summon** extension and the ordinary tools your main agent
needs. Use Goose's existing project working directory and `.agents/agents/`
discovery. Existing provider settings, project instructions, and approval modes
are preserved; this fork does not disable `AGENTS.md` ingestion.

## Specialist policies

Named-agent Markdown frontmatter supports these optional fields:

| Field | Behavior |
| --- | --- |
| `required_extensions: [names]` | A nonempty list selects the child's extensions from Goose's registry, including configured extensions disabled for the parent. Overrides caller extension selection. |
| `required_skills: [names]` | Loads the named skills directly into the child's instructions. Missing skills fail delegation. |
| `always_async: true` | Forces background execution even when the caller omits `async`. |
| `non_blocking: true` | Also forces background execution; loading a running task returns status instead of waiting. |
| `delegate_only: true` | Keeps the agent definition available for delegation but rejects loading its instructions into the parent. |

Without these fields, existing synchronous delegation and extension inheritance
continue to work. The child always receives the internal `message_parent` tool.

Set `delegate_only_extensions: [names]` in Goose's normal configuration to keep
specific extensions out of parent sessions, including restored sessions and
attempts to enable them through tools. Their registry entries remain available
to specialists. Set `metadata.delegate_only: true` in skill frontmatter to hide
that skill from the parent's skill tools while permitting child use. These are
tool/context routing policies, not filesystem or operating-system sandboxes.

## Task communication and lifetime

- `summon.delegate` returns the native child session ID for background work.
- `summon.send(task_id, message)` queues guidance for that running child.
- `summon.message_parent(message)` lets a child send an update or question. Its
  recipient is derived from the stored parent relationship, not model input.
- `summon.load(source: task_id, peek: true)` inspects progress; its existing
  cancellation option stops the task.

Guidance is consumed at model-turn checkpoints, including startup and after
tool execution. It cannot interrupt or undo an external operation already in
flight. A tool batch may finish before the next checkpoint. Both Goose agent
loops consume the same native mailbox; no skill has to remember to poll it.
Sending confirms that guidance was queued, not that the child consumed it. A
child finishing concurrently can leave late guidance pending without another
checkpoint; use its completion report to assess the result.

Completion, failure, and cancellation reports are automatically queued for the
parent. The main agent receives hidden internal reports, reviews them against
the conversation, and writes the user-facing answer. Reports do not impersonate
new user requests. Questions from children are asynchronous messages, not
blocking request/response calls.

The interactive CLI on macOS/Linux checks for reports while its prompt is idle.
Once the user begins typing, normal line editing owns the terminal until
submission; reports are handled after the user turn. External-editor prompts,
non-terminal input, and other platforms handle reports between submitted turns.
There is never a second simultaneous main-agent run or competing approval reader.

Headless runs keep the process alive while background tasks run and process
their reports before returning. JSON output is emitted once at the end;
stream-JSON emits a final `complete` event on success or `error` on failure.
A headless run cannot receive new
human input while running.

ACP `session/prompt` also waits for background tasks and processes their reports
before returning its final JSON-RPC response. Progress continues to stream during
that wait. Cancellation and session closure stop child tasks; deleting a session
waits for its active prompt to leave the run registry before deleting history.
Only one prompt can own a session at a time, including across ACP connections.

Messages live in the existing Goose SQLite database, under `session_mailbox`,
with an automatic schema migration. Child steering is acknowledged atomically
with its insertion into conversation history. Parent reports are acknowledged
only after a successful parent response; errors, cancellation, and action-limit
stops leave them pending. Interactive automatic delivery pauses after a failure
until the user submits another input. Delivery is at least once: failed retries
or a crash before acknowledgement can repeat evidence or a response.

Background execution itself is process-local. Exiting or crashing the CLI does
not turn children into resumable jobs. Persisted reports survive a restart;
unfinished tasks must be delegated again. Normal CLI exit requests cancellation
and awaits child cleanup before flushing telemetry. A forced process kill or
crash cannot guarantee a final report or trace. Upstream's existing delegated-agent
execution mode remains in effect (Summon currently runs children in Auto mode).

## Native OTLP tracing and Langfuse

Tracing uses Goose's existing OpenTelemetry exporter. Each in-memory main chat
has its own root trace; replies and detached children remain beneath that trace.
Subagent telemetry retains the parent `session.id` and records the child's
execution/conversation identity separately. Reopening a chat in a new process
starts a new trace. CLI shutdown flushes the configured exporters.

Configure an OTLP destination in the launching environment. For Langfuse:

```sh
export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='https://YOUR_LANGFUSE_HOST/api/public/otel/v1/traces'
export OTEL_EXPORTER_OTLP_TRACES_HEADERS='Authorization=Basic YOUR_BASE64_PUBLIC_KEY_COLON_SECRET_KEY'
./target/debug/goose session
```

Supply the header through your existing private environment/secrets mechanism.
Do not also enable Goose's legacy direct Langfuse adapter with
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (or the `LANGFUSE_INIT_PROJECT_*`
alternatives), unless duplicate export is intentional.

Content capture is off by default. Opt in with
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` to include the actual
assembled system prompt in generation input, conversation messages, and tool
arguments/results. Exported content is then stored at the configured destination.
Model pricing belongs in Langfuse's deployment configuration, not this source tree.

## Validation

```sh
cargo fmt --all -- --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
cargo test -p goose -p goose-cli --lib --no-default-features \
  --features goose-cli/rustls-tls,goose-cli/otel,goose-cli/disable-update
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
cargo clippy -p goose -p goose-cli --all-targets --no-default-features \
  --features goose-cli/rustls-tls,goose-cli/otel,goose-cli/disable-update -- -D warnings
```

The focused contracts cover mailbox scope and persistence, delivery checkpoints,
specialist policy, CLI acknowledgement, terminal input preservation, and exported
trace relationships. The provider-backed `goose-self-test.yaml` is an additional
manual smoke check when a model endpoint is configured.
