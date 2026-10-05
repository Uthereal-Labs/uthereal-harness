use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::extension_manager::ExtensionManager;

use crate::agents::state_machine::{
    applied, ends_turn, last_effective_role, messages_since_kickoff, not_applicable, Emitter,
    GooseEffect, Operation, OperationResult,
};
use crate::conversation::{Conversation, EffectiveRole};
use crate::session::{Session, SessionManager, SessionType};

pub struct MailboxOperation<'a> {
    session_manager: &'a SessionManager,
    extension_manager: Arc<ExtensionManager>,
}

impl<'a> MailboxOperation<'a> {
    pub(crate) fn new(
        session_manager: &'a SessionManager,
        extension_manager: Arc<ExtensionManager>,
    ) -> Self {
        Self {
            session_manager,
            extension_manager,
        }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for MailboxOperation<'_> {
    fn name(&self) -> &'static str {
        "mailbox"
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        _emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let messages = messages_since_kickoff(conversation)?;
        let between_turns = ends_turn(messages)
            || matches!(
                last_effective_role(messages)?,
                EffectiveRole::User | EffectiveRole::Tool
            );
        if !between_turns {
            return not_applicable();
        }

        let pending = self
            .session_manager
            .pending_session_messages(&session.id)
            .await?;
        let pending: Vec<_> = pending
            .into_iter()
            .filter(|message| {
                session.session_type == SessionType::SubAgent
                    || message.kind == crate::session::MailboxMessageKind::Steering
            })
            .collect();
        if pending.is_empty() {
            return not_applicable();
        }

        if pending.iter().any(|message| message.refreshes_tools()) {
            // A channel join or leave changed this task's tools; the next
            // inference lists them again.
            self.extension_manager
                .invalidate_tools_cache_and_bump_version()
                .await;
        }
        let mut effects = Vec::with_capacity(pending.len());
        for mailbox_message in pending {
            effects.push(GooseEffect::DeliverMailboxMessage {
                mailbox_id: mailbox_message.id,
                message: mailbox_message.prompt()?,
            });
        }
        applied(effects)
    }
}
