use anyhow::Result;
use async_trait::async_trait;

use crate::agents::state_machine::{
    applied, ends_turn, last_effective_role, messages_since_kickoff, not_applicable, Emitter,
    GooseEffect, Operation, OperationResult,
};
use crate::conversation::message::Message;
use crate::conversation::{Conversation, EffectiveRole};
use crate::session::{Session, SessionManager, SessionType};

pub struct MailboxOperation<'a> {
    session_manager: &'a SessionManager,
}

impl<'a> MailboxOperation<'a> {
    pub(crate) fn new(session_manager: &'a SessionManager) -> Self {
        Self { session_manager }
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
        if session.session_type != SessionType::SubAgent {
            return not_applicable();
        }

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
        if pending.is_empty() {
            return not_applicable();
        }

        let mut effects = Vec::with_capacity(pending.len());
        for mailbox_message in pending {
            effects.push(GooseEffect::DeliverMailboxMessage {
                mailbox_id: mailbox_message.id,
                message: Message::user()
                    .with_text(format!(
                        "Message from parent task {}:\n\n{}",
                        mailbox_message.sender_session_id, mailbox_message.body
                    ))
                    .with_visibility(false, true)
                    .with_steer(),
            });
        }
        applied(effects)
    }
}
