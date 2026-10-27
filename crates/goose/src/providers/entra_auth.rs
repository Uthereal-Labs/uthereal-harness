use std::time::{Duration, Instant};

use anyhow::{ensure, Result};
use async_trait::async_trait;
use goose_providers::api_client::AuthProvider;
use oauth2::{
    basic::{BasicClient, BasicTokenType},
    AccessToken, AuthType, ClientId, ClientSecret, Scope, TokenResponse, TokenUrl,
};
use tokio::sync::Mutex;

use crate::config::Config;

/// Service-principal credentials for an OpenAI-compatible Entra resource.
pub(crate) struct EntraAuthProvider {
    http: oauth2::reqwest::Client,
    client_id: ClientId,
    client_secret: ClientSecret,
    token_url: TokenUrl,
    scope: Scope,
    cached: Mutex<Option<(AccessToken, Instant)>>,
}

impl EntraAuthProvider {
    pub(crate) fn from_config(config: &Config) -> Result<Self> {
        let tenant: String = config.get_param("AZURE_TENANT_ID")?;
        let client_id: String = config.get_param("AZURE_CLIENT_ID")?;
        let client_secret: String = config.get_secret("AZURE_CLIENT_SECRET")?;
        let scope: String = config.get_param("OPENAI_ENTRA_SCOPE")?;
        uuid::Uuid::parse_str(&tenant)?;
        uuid::Uuid::parse_str(&client_id)?;
        ensure!(!client_secret.is_empty(), "AZURE_CLIENT_SECRET is empty");
        ensure!(
            scope.ends_with("/.default") && !scope.chars().any(char::is_whitespace),
            "OPENAI_ENTRA_SCOPE must name one resource's /.default scope"
        );
        Ok(Self {
            http: oauth2::reqwest::Client::builder()
                .https_only(true)
                .redirect(oauth2::reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(15))
                .build()?,
            client_id: ClientId::new(client_id),
            client_secret: ClientSecret::new(client_secret),
            token_url: TokenUrl::new(format!(
                "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
            ))?,
            scope: Scope::new(scope),
            cached: Mutex::new(None),
        })
    }
}

#[async_trait]
impl AuthProvider for EntraAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let mut cached = self.cached.lock().await;
        if let Some((token, expires_at)) = cached.as_ref() {
            if Instant::now() < *expires_at {
                return Ok(("Authorization".into(), format!("Bearer {}", token.secret())));
            }
        }

        // Keep the lock across acquisition so concurrent requests share one exchange.
        // Cancellation drops the exchange; model dispatch has not started yet.
        let started = Instant::now();
        let response = BasicClient::new(self.client_id.clone())
            .set_client_secret(self.client_secret.clone())
            .set_token_uri(self.token_url.clone())
            .set_auth_type(AuthType::RequestBody)
            .exchange_client_credentials()
            .add_scope(self.scope.clone())
            .request_async(&self.http)
            .await
            .map_err(|_| anyhow::anyhow!("Entra client-credentials exchange failed"))?;
        ensure!(
            response.token_type() == &BasicTokenType::Bearer,
            "Entra returned an unsupported token type"
        );
        let lifetime = response
            .expires_in()
            .ok_or_else(|| anyhow::anyhow!("Entra token has no expiry"))?;
        let expires_at = started + lifetime.saturating_sub(Duration::from_secs(60));
        ensure!(
            expires_at > Instant::now(),
            "Entra token lifetime is too short"
        );
        let token = response.access_token().clone();
        let header = ("Authorization".into(), format!("Bearer {}", token.secret()));
        *cached = Some((token, expires_at));
        Ok(header)
    }
}
