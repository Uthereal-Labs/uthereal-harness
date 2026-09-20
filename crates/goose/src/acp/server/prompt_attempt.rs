use super::*;
use sha2::{Digest, Sha256};

impl GooseAcpAgent {
    pub(super) async fn claim_acp_prompt_attempt(
        &self,
        args: &PromptRequest,
        use_state_machine: bool,
        await_background_tasks: bool,
    ) -> Result<Option<PromptAttemptLease>, agent_client_protocol::Error> {
        let Some(key) = args
            .meta
            .as_ref()
            .and_then(|meta| meta.get("goose"))
            .and_then(|goose| goose.get("attemptKey"))
        else {
            return Ok(None);
        };
        let key = key
            .as_str()
            .filter(|key| Uuid::parse_str(key).is_ok())
            .ok_or_else(|| {
                agent_client_protocol::Error::invalid_params().data("attemptKey must be a UUID")
            })?;
        if !await_background_tasks {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("durable attempts must await background tasks"));
        }
        let session = self
            .session_manager
            .get_session(&args.session_id.0, false)
            .await
            .internal_err_ctx("Failed to resolve prompt attempt configuration")?;
        if session.message_count != 0 {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("durable attempts require a new empty session"));
        }
        // Session IDs and tracing do not affect execution. Credentials remain part of the
        // digest: a changed tool authority must not silently reuse an accepted request.
        let request = serde_json::json!({
            "prompt": args.prompt,
            "provider": session.provider_name,
            "model": session.model_config,
            "mode": session.goose_mode,
            "extensions": session.extension_data,
            "recipe": session.recipe,
            "recipeValues": session.user_recipe_values,
            "stateMachine": use_state_machine,
            "awaitBackgroundTasks": await_background_tasks,
        });
        let digest: String = Sha256::digest(request.to_string().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let attempt = self
            .session_manager
            .claim_prompt_attempt(key, &digest, &args.session_id.0)
            .await
            .internal_err_ctx("Failed to claim prompt attempt")?;
        if attempt.is_none() {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("attempt already accepted or cancelled; inspect attempt/status"));
        }
        Ok(attempt)
    }
}
