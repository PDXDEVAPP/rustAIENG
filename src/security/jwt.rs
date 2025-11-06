use crate::core::database::DatabaseManager;
use axum::{
    async_trait,
    extract::{FromRequestParts, State},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, warn};

const DEFAULT_JWT_SECRET: &str = "rust-ollama-jwt-secret-key-2025";
const JWT_EXPIRATION_HOURS: i64 = 24;

#[derive(Clone)]
pub struct JwtConfig {
    pub secret: String,
    pub encoding_key: EncodingKey,
    pub decoding_key: DecodingKey,
    pub expiration_hours: i64,
}

impl Default for JwtConfig {
    fn default() -> Self {
        let secret = std::env::var("JWT_SECRET").unwrap_or_else(|_| DEFAULT_JWT_SECRET.to_string());
        
        Self {
            secret: secret.clone(),
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            expiration_hours: JWT_EXPIRATION_HOURS,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sub: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    pub api_key: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_in: usize,
    pub token_type: String,
}

pub struct AuthenticatedUser {
    pub user_id: String,
    pub role: String,
    pub permissions: Vec<String>,
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut axum::http::RequestParts<S>) -> Result<Self, Self::Rejection> {
        let auth_header = parts.headers()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .filter(|value| value.starts_with("Bearer "));

        if let Some(token) = auth_header {
            let token = token.trim_start_matches("Bearer ");
            let jwt_config = parts.extensions()
                .get::<JwtConfig>()
                .cloned()
                .unwrap_or_default();

            match validate_jwt_token(token, &jwt_config) {
                Ok(claims) => Ok(AuthenticatedUser {
                    user_id: claims.sub,
                    role: claims.role,
                    permissions: claims.permissions,
                }),
                Err(e) => {
                    debug!("JWT validation failed: {}", e);
                    Err(AuthError::InvalidToken)
                }
            }
        } else {
            warn!("No authorization header provided");
            Err(AuthError::MissingToken)
        }
    }
}

pub struct JwtManager {
    config: JwtConfig,
    user_cache: RwLock<HashMap<String, AuthenticatedUser>>,
}

impl JwtManager {
    pub fn new(config: JwtConfig) -> Self {
        Self {
            config,
            user_cache: RwLock::new(HashMap::new()),
        }
    }

    pub async fn login(&self, request: LoginRequest) -> Result<LoginResponse, AuthError> {
        let (user_id, role, permissions) = self.validate_credentials(request).await?;

        let now = chrono::Utc::now();
        let exp = now + chrono::Duration::hours(self.config.expiration_hours);

        let claims = JwtClaims {
            sub: user_id.clone(),
            role: role.clone(),
            permissions: permissions.clone(),
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
        };

        let token = jsonwebtoken::encode(&Header::default(), &claims, &self.config.encoding_key)
            .map_err(|e| AuthError::TokenGeneration(e.to_string()))?;

        let user = AuthenticatedUser {
            user_id,
            role,
            permissions,
        };

        {
            let mut cache = self.user_cache.write().await;
            cache.insert(claims.sub.clone(), user);
        }

        Ok(LoginResponse {
            token,
            expires_in: (self.config.expiration_hours * 3600) as usize,
            token_type: "Bearer".to_string(),
        })
    }

    async fn validate_credentials(&self, request: LoginRequest) -> Result<(String, String, Vec<String>), AuthError> {
        if let Some(api_key) = request.api_key {
            self.validate_api_key(&api_key).await
        } else if let (Some(username), Some(password)) = (request.username, request.password) {
            self.validate_username_password(&username, &password).await
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }

    async fn validate_api_key(&self, api_key: &str) -> Result<(String, String, Vec<String>), AuthError> {
        // In a real implementation, this would validate against the database
        // For now, return a mock validation
        if api_key.starts_with("sk-") && api_key.len() > 10 {
            Ok((
                "user_api_key".to_string(),
                "user".to_string(),
                vec!["read".to_string(), "inference".to_string()],
            ))
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }

    async fn validate_username_password(&self, username: &str, password: &str) -> Result<(String, String, Vec<String>), AuthError> {
        // Mock authentication - in production, check against secure hash
        if username == "admin" && password == "admin" {
            Ok((
                "admin".to_string(),
                "admin".to_string(),
                vec!["read".to_string(), "write".to_string(), "admin".to_string(), "inference".to_string()],
            ))
        } else if username == "user" && password == "user" {
            Ok((
                "user".to_string(),
                "user".to_string(),
                vec!["read".to_string(), "inference".to_string()],
            ))
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }

    pub fn generate_token(&self, user: AuthenticatedUser) -> Result<String, AuthError> {
        let now = chrono::Utc::now();
        let exp = now + chrono::Duration::hours(self.config.expiration_hours);

        let claims = JwtClaims {
            sub: user.user_id,
            role: user.role,
            permissions: user.permissions,
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
        };

        jsonwebtoken::encode(&Header::default(), &claims, &self.config.encoding_key)
            .map_err(|e| AuthError::TokenGeneration(e.to_string()))
    }
}

pub fn validate_jwt_token(token: &str, config: &JwtConfig) -> Result<JwtClaims, AuthError> {
    let token_data = jsonwebtoken::decode::<JwtClaims>(
        token,
        &config.decoding_key,
        &Validation::default(),
    )
    .map_err(|e| {
        debug!("JWT decode error: {}", e);
        AuthError::InvalidToken
    })?;

    Ok(token_data.claims)
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Invalid token")]
    InvalidToken,
    #[error("Missing token")]
    MissingToken,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token generation failed: {0}")]
    TokenGeneration(String),
    #[error("Unauthorized")]
    Unauthorized,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        match self {
            AuthError::InvalidToken | AuthError::MissingToken => {
                (StatusCode::UNAUTHORIZED, "Invalid or missing authentication token").into_response()
            }
            AuthError::InvalidCredentials => {
                (StatusCode::UNAUTHORIZED, "Invalid credentials").into_response()
            }
            AuthError::TokenGeneration(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "Token generation failed").into_response()
            }
            AuthError::Unauthorized => {
                (StatusCode::FORBIDDEN, "Insufficient permissions").into_response()
            }
        }
    }
}