//! Frozen session usage at a committed provider-response boundary.
//!
//! Restoring replaces the session's counters and existing ledger aggregate in the
//! snapshot import transaction. Repeated imports never add usage a second time.
//! Historical per-model/per-call ledger rows are replaced by one carried-forward
//! aggregate, even in the original database. Their original message usage and
//! inference evidence remain in the restored conversation.

use anyhow::{ensure, Result};
use goose_providers::conversation::token_usage::Usage;
use serde::{Deserialize, Deserializer, Serialize};
use sqlx::{Sqlite, Transaction};

use super::SessionManager;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DurableUsageReceipt {
    #[serde(deserialize_with = "deserialize_usage")]
    pub current: Usage,
    #[serde(deserialize_with = "deserialize_usage")]
    pub accumulated: Usage,
    #[serde(deserialize_with = "required_nullable")]
    pub accumulated_cost: Option<f64>,
}

fn required_nullable<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    Option::deserialize(deserializer)
}

fn deserialize_usage<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Usage, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    let fields = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
    ];
    if !value.as_object().is_some_and(|object| {
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field))
    }) {
        return Err(serde::de::Error::custom(
            "durable usage requires exactly five nullable token fields",
        ));
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

impl DurableUsageReceipt {
    pub(super) fn from_session(session: &super::Session) -> Result<Self> {
        let receipt = Self {
            current: session.usage,
            accumulated: session.accumulated_usage,
            accumulated_cost: session.accumulated_cost,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<()> {
        for (current, accumulated) in [
            (self.current.input_tokens, self.accumulated.input_tokens),
            (self.current.output_tokens, self.accumulated.output_tokens),
            (self.current.total_tokens, self.accumulated.total_tokens),
            (
                self.current.cache_read_input_tokens,
                self.accumulated.cache_read_input_tokens,
            ),
            (
                self.current.cache_write_input_tokens,
                self.accumulated.cache_write_input_tokens,
            ),
        ] {
            ensure!(
                current.is_none_or(|value| value >= 0)
                    && accumulated.is_none_or(|value| value >= 0)
                    && current.is_none_or(|value| accumulated.is_some_and(|sum| sum >= value)),
                "durable usage counters are negative or inconsistent"
            );
        }
        ensure!(
            self.accumulated_cost
                .is_none_or(|cost| cost.is_finite() && cost >= 0.0),
            "durable usage cost is invalid"
        );
        Ok(())
    }
}

pub(super) async fn capture(
    manager: &SessionManager,
    session_id: &str,
) -> Result<DurableUsageReceipt> {
    let session = manager.get_session(session_id, false).await?;
    DurableUsageReceipt::from_session(&session)
}

pub(super) async fn restore(
    tx: &mut Transaction<'_, Sqlite>,
    session_id: &str,
    receipt: &DurableUsageReceipt,
) -> Result<()> {
    receipt.validate()?;
    let updated = sqlx::query(
        "UPDATE sessions SET input_tokens=?, output_tokens=?, total_tokens=?, \
         cache_read_tokens=?, cache_write_tokens=?, accumulated_input_tokens=?, \
         accumulated_output_tokens=?, accumulated_total_tokens=?, \
         accumulated_cache_read_tokens=?, accumulated_cache_write_tokens=?, accumulated_cost=? \
         WHERE id=?",
    )
    .bind(receipt.current.input_tokens)
    .bind(receipt.current.output_tokens)
    .bind(receipt.current.total_tokens)
    .bind(receipt.current.cache_read_input_tokens)
    .bind(receipt.current.cache_write_input_tokens)
    .bind(receipt.accumulated.input_tokens)
    .bind(receipt.accumulated.output_tokens)
    .bind(receipt.accumulated.total_tokens)
    .bind(receipt.accumulated.cache_read_input_tokens)
    .bind(receipt.accumulated.cache_write_input_tokens)
    .bind(receipt.accumulated_cost)
    .bind(session_id)
    .execute(&mut **tx)
    .await?;
    ensure!(
        updated.rows_affected() == 1,
        "durable usage session is missing"
    );
    sqlx::query("DELETE FROM usage_ledger WHERE session_id=?")
        .bind(session_id)
        .execute(&mut **tx)
        .await?;
    if receipt.accumulated != Usage::default() || receipt.accumulated_cost.is_some() {
        // The receipt proves aggregate usage; it cannot recreate historical
        // per-model rows. Use the existing ledger's carried-forward convention.
        sqlx::query(
            "INSERT INTO usage_ledger(session_id,created_timestamp,input_tokens,output_tokens,\
             total_tokens,cache_read_tokens,cache_write_tokens,cost,cost_source,is_compaction) \
             VALUES (?,strftime('%s','now'),?,?,?,?,?,?, 'carried_forward',0)",
        )
        .bind(session_id)
        .bind(receipt.accumulated.input_tokens)
        .bind(receipt.accumulated.output_tokens)
        .bind(receipt.accumulated.total_tokens)
        .bind(receipt.accumulated.cache_read_input_tokens)
        .bind(receipt.accumulated.cache_write_input_tokens)
        .bind(receipt.accumulated_cost)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GooseMode;
    use crate::conversation::message::MessageUsage;
    use crate::session::SessionType;
    use goose_providers::conversation::token_usage::{CostSource, ProviderUsage};

    async fn session(manager: &SessionManager) -> Result<String> {
        Ok(manager
            .create_session(
                std::env::temp_dir(),
                "Durable usage".into(),
                SessionType::User,
                GooseMode::Auto,
            )
            .await?
            .id)
    }

    async fn record(manager: &SessionManager, id: &str, input: i32, output: i32) -> Result<()> {
        let usage = ProviderUsage::new(
            "original-model".into(),
            Usage::new(Some(input), Some(output), None).with_cache_tokens(Some(2), Some(1)),
        )
        .with_cost(0.25, CostSource::ProviderReported);
        manager
            .record_usage_metrics(
                id,
                None,
                usage.usage,
                &usage.model,
                &MessageUsage::from_provider_usage(&usage, false),
            )
            .await
    }

    #[tokio::test]
    async fn restore_is_idempotent_in_same_and_fresh_storage() -> Result<()> {
        let original_dir = tempfile::tempdir()?;
        let original = SessionManager::new(original_dir.path().to_path_buf());
        let original_id = session(&original).await?;
        record(&original, &original_id, 100, 20).await?;
        record(&original, &original_id, 50, 10).await?;
        let receipt = capture(&original, &original_id).await?;
        record(&original, &original_id, 40, 8).await?;

        let fresh_dir = tempfile::tempdir()?;
        let fresh = SessionManager::new(fresh_dir.path().to_path_buf());
        let fresh_id = session(&fresh).await?;
        for (manager, id) in [
            (&original, original_id.as_str()),
            (&fresh, fresh_id.as_str()),
        ] {
            let pool = manager.storage().pool().await?;
            for _ in 0..3 {
                let mut tx = pool.begin().await?;
                restore(&mut tx, id, &receipt).await?;
                tx.commit().await?;
                assert_eq!(capture(manager, id).await?, receipt);
                let totals = manager.get_session_usage_totals(id).await?;
                assert_eq!(totals.accumulated_usage, receipt.accumulated);
                assert_eq!(totals.accumulated_cost, receipt.accumulated_cost);
            }
            let rows: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM usage_ledger WHERE session_id=?")
                    .bind(id)
                    .fetch_one(pool)
                    .await?;
            assert_eq!(rows, 1);
            record(manager, id, 10, 2).await?;
            let after = capture(manager, id).await?;
            assert_eq!(after.accumulated.input_tokens, Some(160));
            assert_eq!(after.accumulated.output_tokens, Some(32));
            assert_eq!(after.accumulated_cost, Some(0.75));
        }
        Ok(())
    }

    #[tokio::test]
    async fn restore_rolls_back_with_snapshot_and_preserves_unknowns() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let manager = SessionManager::new(directory.path().to_path_buf());
        let id = session(&manager).await?;
        let unknown = capture(&manager, &id).await?;
        record(&manager, &id, 20, 4).await?;
        let original = capture(&manager, &id).await?;
        let pool = manager.storage().pool().await?;
        let mut tx = pool.begin().await?;
        restore(&mut tx, &id, &unknown).await?;
        tx.rollback().await?;
        assert_eq!(capture(&manager, &id).await?, original);
        let mut tx = pool.begin().await?;
        restore(&mut tx, &id, &unknown).await?;
        tx.commit().await?;
        assert_eq!(capture(&manager, &id).await?, unknown);
        assert!(unknown.accumulated_cost.is_none());
        assert_eq!(unknown.current, Usage::default());
        Ok(())
    }

    #[tokio::test]
    async fn zero_counters_remain_known_and_do_not_create_known_cost() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let manager = SessionManager::new(directory.path().to_path_buf());
        let id = session(&manager).await?;
        let zero = DurableUsageReceipt {
            current: Usage::new(Some(0), Some(0), Some(0)).with_cache_tokens(Some(0), Some(0)),
            accumulated: Usage::new(Some(0), Some(0), Some(0)).with_cache_tokens(Some(0), Some(0)),
            accumulated_cost: None,
        };
        let pool = manager.storage().pool().await?;
        for _ in 0..2 {
            let mut tx = pool.begin().await?;
            restore(&mut tx, &id, &zero).await?;
            tx.commit().await?;
            assert_eq!(capture(&manager, &id).await?, zero);
            assert!(manager
                .get_session_usage_totals(&id)
                .await?
                .accumulated_cost
                .is_none());
        }
        Ok(())
    }

    #[test]
    fn receipt_wire_requires_exact_nullable_fields() -> Result<()> {
        let receipt = DurableUsageReceipt {
            current: Usage::default(),
            accumulated: Usage::default(),
            accumulated_cost: None,
        };
        let encoded = serde_json::to_value(&receipt)?;
        assert_eq!(encoded["current"]["input_tokens"], serde_json::Value::Null);
        assert_eq!(encoded["accumulatedCost"], serde_json::Value::Null);
        assert_eq!(
            serde_json::from_value::<DurableUsageReceipt>(encoded.clone())?,
            receipt
        );
        for field in ["current", "accumulated", "accumulatedCost"] {
            let mut invalid = encoded.clone();
            invalid.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<DurableUsageReceipt>(invalid).is_err());
        }
        for field in [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        ] {
            let mut invalid = encoded.clone();
            invalid["current"].as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<DurableUsageReceipt>(invalid).is_err());
        }
        for value in [
            serde_json::json!(2_147_483_648_i64),
            serde_json::json!(true),
            serde_json::json!(1.5),
        ] {
            let mut invalid = encoded.clone();
            invalid["current"]["input_tokens"] = value;
            assert!(serde_json::from_value::<DurableUsageReceipt>(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn receipt_rejects_negative_nonfinite_and_non_cumulative_values() {
        let valid = DurableUsageReceipt {
            current: Usage::new(Some(10), Some(2), None),
            accumulated: Usage::new(Some(20), Some(4), None),
            accumulated_cost: Some(0.5),
        };
        assert!(valid.validate().is_ok());
        for cost in [-0.5, f64::NAN, f64::INFINITY] {
            let mut invalid = valid.clone();
            invalid.accumulated_cost = Some(cost);
            assert!(invalid.validate().is_err());
        }
        for accumulated in [None, Some(-1), Some(9)] {
            let mut invalid = valid.clone();
            invalid.accumulated.input_tokens = accumulated;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = valid;
        invalid.current.input_tokens = Some(-1);
        assert!(invalid.validate().is_err());
    }
}
