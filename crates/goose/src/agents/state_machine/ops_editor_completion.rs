use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    applied, ends_turn, messages_since_kickoff, not_applicable, Emitter, GooseEffect, Operation,
    OperationResult,
};
use crate::agents::platform_extensions::summon::{running_editor_tasks, wait_for_editor_notice};
use crate::conversation::Conversation;
use crate::session::{Session, SessionManager, SessionType};

pub struct EditorCompletionOperation<'a> {
    manager: &'a SessionManager,
    cancel: CancellationToken,
}
impl<'a> EditorCompletionOperation<'a> {
    pub fn new(manager: &'a SessionManager, cancel: CancellationToken) -> Self {
        Self { manager, cancel }
    }
}
#[async_trait]
impl Operation<Session, GooseEffect> for EditorCompletionOperation<'_> {
    fn name(&self) -> &'static str {
        "editor_completion"
    }
    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        _emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if session.session_type != SessionType::SubAgent
            || !ends_turn(messages_since_kickoff(conversation)?)
        {
            return not_applicable();
        }
        if running_editor_tasks(session, conversation.messages()).is_empty() {
            return not_applicable();
        }
        wait_for_editor_notice(
            self.manager,
            &session.id,
            &self.cancel,
            std::time::Duration::from_secs(180),
        )
        .await?;
        // Restart the pipeline so mailbox delivery precedes the next inference.
        applied([])
    }
}
