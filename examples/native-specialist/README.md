# Native specialist example

Enable Developer and Summon, then open this directory as the Goose working
directory. Ask main to delegate a multi-section document to `document-writer`.
While it works, send a revised title or section requirement. Main should use
`summon.send` with the existing task ID. Completion should trigger a new main-agent
response without a status request. Inspect the saved document and its Langfuse
trace, including the delegate tool, child generation, and file-writing tools.

This example uses local Markdown files. Applications can supply Quire or other
MCP extensions and domain skills through the same `required_extensions` and
`required_skills` fields.

## Application-owned specialists and MCP tools

Place agent definitions in the application's `.agents/agents/` directory and
skills in `.agents/skills/<name>/SKILL.md`. Open that application directory when
starting the session. The agent's description is the routing contract: describe
the user requests it owns (for example, creating reports and briefs), as well as
the tool it uses. A tool name alone can leave an ordinary request ambiguous.

Register each private MCP extension in the Goose configuration with
`enabled: false`, and add its registry name to `delegate_only_extensions`.
Reference that name in the specialist's `required_extensions`; Summon resolves
disabled registry entries for the child. Put private skills in `required_skills`
and mark them `metadata: { delegate-only: true }`. Use `always_async: true` and
`non_blocking: true` for specialists that keep working while the user steers them.

For the Quire setup from PR #4806, retain the external bridge and Quire skill,
and adapt the agent as described in `HARNESS.md`. Its description should explicitly
own requests to write reports. Main researches sources and hands off prepared
content; only the specialist creates and verifies the Quire document. Native
`send` replaces the Python coordinator's steering tools.

Validate with an ordinary compound request first, without naming the specialist.
Check the stored sessions and tool calls: main performs prerequisites, exactly
one child owns the document tools, and the child cannot delegate again. Then
send a revision while the child runs and confirm that the same child applies it.
Completion should produce one main-authored response without polling. Inspect
the actual document and Langfuse observations, not only the assistant's claim.

For the focused recipe check, from this directory run:

```sh
../../target/debug/goose run --recipe ../../goose-self-test.yaml \
  --params test_phases=native-parity --params test_depth=quick \
  --params parallel_tests=false --params cleanup_after=false
```

In Langfuse, verify that the specialist reply is a descendant of the `delegate`
tool, with its generations and tools beneath it. Main follow-ups share the same
trace; a separate chat has a separate trace. Failed tools should show `ERROR`
and their output. Compare generation response IDs to catch duplicate spans,
and compare usage details with your configured deployment prices. A downstream
MCP server must extract the forwarded W3C context to join the same trace.
