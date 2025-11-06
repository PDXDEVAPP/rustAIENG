use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub key: String,
    pub name: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub expires_in_days: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyResponse {
    pub id: String,
    pub key: String,
    pub name: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyUsage {
    pub key_id: String,
    pub endpoint: String,
    pub method: String,
    pub timestamp: DateTime<Utc>,
    pub status_code: u16,
    pub response_time_ms: u64,
}

pub struct ApiKeyManager {
    keys: RwLock<HashMap<String, ApiKey>>,
    usage_history: RwLock<Vec<ApiKeyUsage>>,
    rate_limits: RwLock<HashMap<String, RateLimit>>,
}

#[derive(Debug, Clone)]
struct RateLimit {
    requests: u64,
    window_start: DateTime<Utc>,
    limit: u64,
    window_duration: chrono::Duration,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            requests: 0,
            window_start: Utc::now(),
            limit: 100, // Default: 100 requests per hour
            window_duration: chrono::Duration::hours(1),
        }
    }
}

impl ApiKeyManager {
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
            usage_history: RwLock::new(Vec::new()),
            rate_limits: RwLock::new(HashMap::new()),
        }
    }

    pub async fn create_api_key(&self, request: CreateApiKeyRequest) -> Result<CreateApiKeyResponse, ApiKeyError> {
        let id = uuid::Uuid::new_v4().to_string();
        let key = self.generate_api_key();
        let created_at = Utc::now();
        let expires_at = request.expires_in_days.map(|days| created_at + chrono::Duration::days(days as i64));

        let api_key = ApiKey {
            id: id.clone(),
            key: key.clone(),
            name: request.name,
            role: request.role,
            permissions: request.permissions,
            created_at,
            expires_at,
            last_used: None,
            is_active: true,
            usage_count: 0,
        };

        {
            let mut keys = self.keys.write().await;
            keys.insert(key.clone(), api_key.clone());
        }

        {
            let mut rate_limits = self.rate_limits.write().await;
            rate_limits.insert(key.clone(), RateLimit::default());
        }

        debug!("Created API key: {} for user", id);

        Ok(CreateApiKeyResponse {
            id,
            key,
            name: api_key.name,
            role: api_key.role,
            permissions: api_key.permissions,
            created_at,
            expires_at,
        })
    }

    pub async fn validate_api_key(&self, key: &str) -> Result<ApiKey, ApiKeyError> {
        let keys = self.keys.read().await;
        
        if let Some(api_key) = keys.get(key) {
            // Check if key is active
            if !api_key.is_active {
                return Err(ApiKeyError::InactiveKey);
            }

            // Check if key has expired
            if let Some(expires_at) = api_key.expires_at {
                if Utc::now() > expires_at {
                    return Err(ApiKeyError::ExpiredKey);
                }
            }

            // Update usage statistics
            drop(keys);
            let mut keys = self.keys.write().await;
            if let Some(key_entry) = keys.get_mut(key) {
                key_entry.last_used = Some(Utc::now());
                key_entry.usage_count += 1;
            }

            Ok(api_key.clone())
        } else {
            Err(ApiKeyError::InvalidKey)
        }
    }

    pub async fn check_rate_limit(&self, key: &str) -> Result<(), RateLimitError> {
        let mut rate_limits = self.rate_limits.write().await;
        let now = Utc::now();

        let rate_limit = rate_limits.entry(key.to_string()).or_insert_with(RateLimit::default);

        // Reset window if it's expired
        if now - rate_limit.window_start > rate_limit.window_duration {
            rate_limit.requests = 0;
            rate_limit.window_start = now;
        }

        if rate_limit.requests >= rate_limit.limit {
            return Err(RateLimitError::Exceeded {
                limit: rate_limit.limit,
                window_duration: rate_limit.window_duration,
            });
        }

        rate_limit.requests += 1;
        Ok(())
    }

    pub async fn record_usage(&self, key_id: &str, endpoint: &str, method: &str, status_code: u16, response_time_ms: u64) {
        let usage = ApiKeyUsage {
            key_id: key_id.to_string(),
            endpoint: endpoint.to_string(),
            method: method.to_string(),
            timestamp: Utc::now(),
            status_code,
            response_time_ms,
        };

        {
            let mut history = self.usage_history.write().await;
            history.push(usage);

            // Keep only last 10,000 entries
            if history.len() > 10_000 {
                history.drain(0..history.len() - 10_000);
            }
        }
    }

    pub async fn get_api_keys(&self) -> Result<Vec<ApiKey>, ApiKeyError> {
        let keys = self.keys.read().await;
        Ok(keys.values().cloned().collect())
    }

    pub async fn revoke_api_key(&self, key: &str) -> Result<(), ApiKeyError> {
        let mut keys = self.keys.write().await;
        if let Some(api_key) = keys.get_mut(key) {
            api_key.is_active = false;
            Ok(())
        } else {
            Err(ApiKeyError::InvalidKey)
        }
    }

    pub async fn get_usage_statistics(&self, key: &str) -> Result<ApiKeyUsageStats, ApiKeyError> {
        let keys = self.keys.read().await;
        let usage_history = self.usage_history.read().await;

        if let Some(api_key) = keys.get(key) {
            let key_usage: Vec<_> = usage_history
                .iter()
                .filter(|usage| usage.key_id == key)
                .collect();

            let total_requests = key_usage.len() as u64;
            let average_response_time = if !key_usage.is_empty() {
                key_usage.iter().map(|u| u.response_time_ms).sum::<u64>() / key_usage.len() as u64
            } else {
                0
            };

            let error_rate = if total_requests > 0 {
                (key_usage.iter().filter(|u| (400..600).contains(&u.status_code)).count() as f64 
                 / total_requests as f64) * 100.0
            } else {
                0.0
            };

            Ok(ApiKeyUsageStats {
                total_requests,
                average_response_time_ms: average_response_time,
                error_rate_percent: error_rate,
                last_request: key_usage.last().map(|u| u.timestamp),
                recent_usage: key_usage
                    .iter()
                    .rev()
                    .take(10)
                    .cloned()
                    .collect(),
            })
        } else {
            Err(ApiKeyError::InvalidKey)
        }
    }

    fn generate_api_key(&self) -> String {
        format!("sk-{}", uuid::Uuid::new_v4().to_string().replace("-", ""))
    }

    pub async fn cleanup_expired_keys(&self) -> Vec<String> {
        let mut keys = self.keys.write().await;
        let now = Utc::now();
        let mut expired_keys = Vec::new();

        for (key, api_key) in keys.iter_mut() {
            if let Some(expires_at) = api_key.expires_at {
                if now > expires_at {
                    api_key.is_active = false;
                    expired_keys.push(key.clone());
                }
            }
        }

        if !expired_keys.is_empty() {
            warn!("Cleaned up {} expired API keys", expired_keys.len());
        }

        expired_keys
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyUsageStats {
    pub total_requests: u64,
    pub average_response_time_ms: u64,
    pub error_rate_percent: f64,
    pub last_request: Option<DateTime<Utc>>,
    pub recent_usage: Vec<ApiKeyUsage>,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiKeyError {
    #[error("Invalid API key")]
    InvalidKey,
    #[error("API key is inactive")]
    InactiveKey,
    #[error("API key has expired")]
    ExpiredKey,
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("API key not found")]
    NotFound,
}

#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded: {limit} requests per {window_duration}")]
    Exceeded {
        limit: u64,
        window_duration: chrono::Duration,
    },
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> axum::response::Response {
        match self {
            RateLimitError::Exceeded { limit, window_duration } => {
                let retry_after = if window_duration.num_hours() > 0 {
                    window_duration.num_hours() * 3600
                } else if window_duration.num_minutes() > 0 {
                    window_duration.num_minutes() * 60
                } else {
                    window_duration.num_seconds()
                };

                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    [
                        ("X-RateLimit-Limit", limit.to_string()),
                        ("X-RateLimit-Reset", retry_after.to_string()),
                    ],
                    format!("Rate limit exceeded. Try again in {} seconds.", retry_after)
                ).into_response()
            }
        }
    }
}