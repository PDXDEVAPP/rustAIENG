use crate::security::{JwtManager, ApiKeyManager, SlidingWindowRateLimiter};
use crate::security::auth::SecurityMiddleware;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

pub mod auth {
    pub use super::*;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    pub api_key: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyCreateRequest {
    pub name: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub expires_in_days: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RateLimitConfigRequest {
    pub requests_per_minute: Option<u64>,
    pub requests_per_hour: Option<u64>,
    pub requests_per_day: Option<u64>,
    pub burst_limit: Option<u64>,
}

pub struct AuthState {
    pub jwt_manager: JwtManager,
    pub api_key_manager: ApiKeyManager,
    pub rate_limiter: SlidingWindowRateLimiter,
}

impl AuthState {
    pub fn new(jwt_config: crate::security::JwtConfig) -> Self {
        Self {
            jwt_manager: JwtManager::new(jwt_config),
            api_key_manager: ApiKeyManager::new(),
            rate_limiter: SlidingWindowRateLimiter::new(chrono::Duration::minutes(1)),
        }
    }

    pub fn security_middleware(&self) -> SecurityMiddleware {
        SecurityMiddleware::new(self.jwt_manager.clone(), self.api_key_manager.clone())
    }
}

pub async fn login_handler(
    State(state): State<AuthState>,
    Json(request): Json<LoginRequest>,
) -> Result<Json<crate::security::LoginResponse>, axum::http::StatusCode> {
    match state.jwt_manager.login(request).await {
        Ok(response) => Ok(Json(response)),
        Err(_) => Err(StatusCode::UNAUTHORIZED),
    }
}

pub async fn create_api_key_handler(
    State(state): State<AuthState>,
    Json(request): Json<ApiKeyCreateRequest>,
) -> Result<Json<crate::security::CreateApiKeyResponse>, axum::http::StatusCode> {
    match state.api_key_manager.create_api_key(request).await {
        Ok(response) => Ok(Json(response)),
        Err(_) => Err(StatusCode::BAD_REQUEST),
    }
}

pub async fn list_api_keys_handler(
    State(state): State<AuthState>,
) -> Result<Json<Vec<crate::security::ApiKey>>, axum::http::StatusCode> {
    match state.api_key_manager.get_api_keys().await {
        Ok(keys) => Ok(Json(keys)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn revoke_api_key_handler(
    State(state): State<AuthState>,
    axum::extract::Path(key): axum::extract::Path<String>,
) -> Result<(), axum::http::StatusCode> {
    match state.api_key_manager.revoke_api_key(&key).await {
        Ok(_) => Ok(()),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn get_rate_limit_stats_handler(
    State(state): State<AuthState>,
    axum::extract::Path(identifier): axum::extract::Path<String>,
) -> Result<Json<Option<crate::security::RateLimitStats>>, axum::http::StatusCode> {
    match state.rate_limiter.get_usage_stats(&identifier).await {
        Some(stats) => Ok(Json(Some(stats))),
        None => Ok(Json(None)),
    }
}

pub async fn cleanup_expired_keys_handler(
    State(state): State<AuthState>,
) -> Result<Json<Vec<String>>, axum::http::StatusCode> {
    let expired_keys = state.api_key_manager.cleanup_expired_keys().await;
    Ok(Json(expired_keys))
}

pub fn auth_routes() -> Router {
    Router::new()
        .route("/auth/login", post(login_handler))
        .route("/auth/api-keys", post(create_api_key_handler))
        .route("/auth/api-keys", get(list_api_keys_handler))
        .route("/auth/api-keys/:key", delete(revoke_api_key_handler))
        .route("/auth/rate-limit/:identifier", get(get_rate_limit_stats_handler))
        .route("/auth/cleanup", post(cleanup_expired_keys_handler))
}

// Extension to extract auth state from request
pub struct AuthExtension(pub AuthState);

impl<S> axum::extract::FromRequestParts<S> for AuthExtension
where
    S: Send + Sync,
{
    type Rejection = axum::http::StatusCode;

    async fn from_request_parts(
        parts: &mut axum::http::RequestParts<S>,
    ) -> Result<Self, Self::Rejection> {
        let auth_state = parts
            .extensions
            .get::<AuthState>()
            .cloned()
            .ok_or_else(|| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        Ok(Self(auth_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;

    #[tokio::test]
    async fn test_login_with_api_key() {
        let jwt_config = crate::security::JwtConfig::default();
        let auth_state = AuthState::new(jwt_config);
        
        let login_request = LoginRequest {
            api_key: Some("sk-test123456789".to_string()),
            username: None,
            password: None,
        };

        let result = login_handler(State(auth_state), Json(login_request)).await;
        
        match result {
            Ok(response) => {
                assert!(!response.token.is_empty());
                assert_eq!(response.token_type, "Bearer");
            }
            Err(_) => panic!("Expected successful login"),
        }
    }

    #[tokio::test]
    async fn test_create_api_key() {
        let jwt_config = crate::security::JwtConfig::default();
        let auth_state = AuthState::new(jwt_config);
        
        let create_request = ApiKeyCreateRequest {
            name: "test-key".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string(), "inference".to_string()],
            expires_in_days: Some(30),
        };

        let result = create_api_key_handler(State(auth_state), Json(create_request)).await;
        
        match result {
            Ok(response) => {
                assert!(!response.key.is_empty());
                assert_eq!(response.name, "test-key");
                assert!(response.expires_at.is_some());
            }
            Err(_) => panic!("Expected successful API key creation"),
        }
    }
}