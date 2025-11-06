use crate::security::{JwtManager, ApiKeyManager, SlidingWindowRateLimiter, RateLimitConfig};
use axum::{
    extract::ConnectInfo,
    http::StatusCode,
    response::Response,
    middleware::Next,
    response::IntoResponse,
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::validate_request::ValidateRequest;
use tracing::{info, warn, debug, error};

pub struct SecurityMiddleware {
    pub jwt_manager: JwtManager,
    pub api_key_manager: ApiKeyManager,
    pub rate_limiter: SlidingWindowRateLimiter,
    pub cors_enabled: bool,
}

impl SecurityMiddleware {
    pub fn new(jwt_manager: JwtManager, api_key_manager: ApiKeyManager) -> Self {
        Self {
            jwt_manager,
            api_key_manager,
            rate_limiter: SlidingWindowRateLimiter::new(RateLimitConfig::default()),
            cors_enabled: true,
        }
    }

    pub fn with_cors_enabled(mut self, enabled: bool) -> Self {
        self.cors_enabled = enabled;
        self
    }

    pub fn cors_layer(&self) -> Option<CorsLayer> {
        if self.cors_enabled {
            Some(CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any))
        } else {
            None
        }
    }

    pub async fn authenticate_request<B>(
        &self,
        request: axum::extract::Request<B>,
        connect_info: Option<ConnectInfo<SocketAddr>>,
    ) -> Result<(axum::extract::Request<B>, SecurityContext), Response> {
        let (parts, body) = request.into_parts();
        let ip_address = connect_info
            .map(|connect_info| connect_info.0.ip().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        debug!("Processing request from IP: {}", ip_address);

        // Extract API key from headers
        let api_key = parts.headers
            .get("x-api-key")
            .and_then(|value| value.to_str().ok())
            .map(|s| s.to_string());

        // Extract authorization token
        let auth_header = parts.headers
            .get("authorization")
            .and_then(|value| value.to_str().ok());

        let mut authenticated_user = None;
        let mut api_key_info = None;

        // Try JWT authentication first
        if let Some(auth_header) = auth_header {
            if auth_header.starts_with("Bearer ") {
                let token = &auth_header[7..];
                match self.jwt_manager.config.validate_jwt_token(token, &self.jwt_manager.config) {
                    Ok(claims) => {
                        authenticated_user = Some(crate::security::AuthenticatedUser {
                            user_id: claims.sub,
                            role: claims.role,
                            permissions: claims.permissions,
                        });
                        info!("JWT authentication successful for user: {}", claims.sub);
                    }
                    Err(e) => {
                        warn!("JWT authentication failed: {}", e);
                    }
                }
            }
        }

        // Try API key authentication if JWT failed
        if authenticated_user.is_none() {
            if let Some(api_key) = api_key {
                match self.api_key_manager.validate_api_key(&api_key).await {
                    Ok(key_info) => {
                        api_key_info = Some(key_info);
                        info!("API key authentication successful for key: {}", key_info.id);
                    }
                    Err(e) => {
                        warn!("API key authentication failed: {}", e);
                    }
                }
            }
        }

        // Check rate limiting
        let identifier = authenticated_user.as_ref()
            .map(|u| format!("user:{}", u.user_id))
            .or_else(|| api_key_info.as_ref().map(|k| format!("key:{}", k.id)))
            .unwrap_or_else(|| format!("ip:{}", ip_address));

        if let Err(rate_limit_error) = self.rate_limiter.check_rate_limit(&identifier).await {
            warn!("Rate limit exceeded for identifier: {}", identifier);
            return Err(rate_limit_error.into_response());
        }

        let security_context = SecurityContext {
            ip_address,
            authenticated_user,
            api_key_info,
            rate_limit_identifier: identifier,
        };

        let request = axum::extract::Request::from_parts(parts, body);
        Ok((request, security_context))
    }

    pub async fn log_request(
        &self,
        method: &str,
        path: &str,
        status_code: StatusCode,
        response_time_ms: u64,
        security_context: &SecurityContext,
    ) {
        let user_info = security_context
            .authenticated_user
            .as_ref()
            .map(|u| format!("user:{}", u.user_id))
            .or_else(|| security_context.api_key_info.as_ref().map(|k| format!("key:{}", k.id)))
            .unwrap_or_else(|| "anonymous".to_string());

        info!(
            "Request: {} {} from {} - {} ({}ms)",
            method,
            path,
            user_info,
            status_code,
            response_time_ms
        );

        // Record API key usage if applicable
        if let Some(api_key_info) = &security_context.api_key_info {
            let endpoint = format!("{} {}", method, path);
            let status_code_u16 = status_code.as_u16();
            
            self.api_key_manager.record_usage(
                &api_key_info.id,
                &endpoint,
                method,
                status_code_u16,
                response_time_ms,
            ).await;
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub ip_address: String,
    pub authenticated_user: Option<crate::security::AuthenticatedUser>,
    pub api_key_info: Option<crate::security::ApiKey>,
    pub rate_limit_identifier: String,
}

impl SecurityContext {
    pub fn is_authenticated(&self) -> bool {
        self.authenticated_user.is_some() || self.api_key_info.is_some()
    }

    pub fn has_permission(&self, permission: &str) -> bool {
        if let Some(user) = &self.authenticated_user {
            user.permissions.contains(&permission.to_string())
        } else if let Some(api_key) = &self.api_key_info {
            api_key.permissions.contains(&permission.to_string())
        } else {
            false
        }
    }

    pub fn get_user_id(&self) -> Option<&str> {
        self.authenticated_user
            .as_ref()
            .map(|u| u.user_id.as_str())
            .or_else(|| self.api_key_info.as_ref().map(|k| k.id.as_str()))
    }

    pub fn get_role(&self) -> Option<&str> {
        self.authenticated_user
            .as_ref()
            .map(|u| u.role.as_str())
            .or_else(|| self.api_key_info.as_ref().map(|k| k.role.as_str()))
    }
}

pub struct SecurityValidator {
    security_context: Option<SecurityContext>,
}

impl SecurityValidator {
    pub fn new(security_context: SecurityContext) -> Self {
        Self {
            security_context: Some(security_context),
        }
    }

    pub fn require_authentication(&self) -> Result<(), Response> {
        if let Some(context) = &self.security_context {
            if context.is_authenticated() {
                Ok(())
            } else {
                Err((
                    StatusCode::UNAUTHORIZED,
                    "Authentication required"
                ).into_response())
            }
        } else {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Security context missing"
            ).into_response())
        }
    }

    pub fn require_permission(&self, permission: &str) -> Result<(), Response> {
        if let Some(context) = &self.security_context {
            if context.has_permission(permission) {
                Ok(())
            } else {
                Err((
                    StatusCode::FORBIDDEN,
                    format!("Permission '{}' required", permission)
                ).into_response())
            }
        } else {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Security context missing"
            ).into_response())
        }
    }

    pub fn get_security_context(&self) -> Option<&SecurityContext> {
        self.security_context.as_ref()
    }
}

impl ValidateRequest for SecurityValidator {
    fn validate(&mut self, request: &mut axum::http::Request<axum::body::Body>) -> Result<(), Response> {
        self.require_authentication()?;
        Ok(())
    }
}