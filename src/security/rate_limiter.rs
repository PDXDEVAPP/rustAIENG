use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u64,
    pub requests_per_hour: u64,
    pub requests_per_day: u64,
    pub burst_limit: u64,
    pub window_duration: chrono::Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 100,
            requests_per_hour: 1000,
            requests_per_day: 10000,
            burst_limit: 200,
            window_duration: chrono::Duration::minutes(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenBucket {
    tokens: Arc<AtomicU64>,
    last_refill: Arc<RwLock<DateTime<Utc>>>,
    config: RateLimitConfig,
}

impl TokenBucket {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            tokens: Arc::new(AtomicU64::new(config.burst_limit)),
            last_refill: Arc::new(RwLock::new(Utc::now())),
            config,
        }
    }

    pub async fn consume(&self) -> Result<(), RateLimitError> {
        let now = Utc::now();
        let mut last_refill = self.last_refill.write().await;
        let mut tokens = self.tokens.clone();

        // Calculate time elapsed since last refill
        let time_since_refill = now - *last_refill;

        // Refill tokens based on the time elapsed
        if time_since_refill >= chrono::Duration::seconds(1) {
            let seconds = time_since_refill.num_seconds() as u64;
            let refill_rate = self.config.requests_per_minute / 60; // requests per second
            let new_tokens = refill_rate * seconds;

            // Get current token count
            let current_tokens = tokens.load(Ordering::Relaxed);
            let new_token_count = std::cmp::min(
                current_tokens.saturating_add(new_tokens),
                self.config.burst_limit
            );

            tokens.store(new_token_count, Ordering::Relaxed);
            *last_refill = now;
        }

        // Check if we have tokens to consume
        let current_tokens = tokens.load(Ordering::Relaxed);
        if current_tokens > 0 {
            tokens.fetch_sub(1, Ordering::Relaxed);
            Ok(())
        } else {
            debug!("Rate limit exceeded for token bucket");
            Err(RateLimitError::TokenBucketExhausted)
        }
    }

    pub fn get_current_tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }

    pub async fn get_reset_time(&self) -> DateTime<Utc> {
        let last_refill = *self.last_refill.read().await;
        last_refill + self.config.window_duration
    }
}

#[derive(Debug, Clone)]
pub struct SlidingWindowRateLimiter {
    windows: HashMap<String, WindowCounter>,
    config: RateLimitConfig,
}

#[derive(Debug, Clone)]
struct WindowCounter {
    minute_counter: AtomicU64,
    hour_counter: AtomicU64,
    day_counter: AtomicU64,
    window_start_minute: DateTime<Utc>,
    window_start_hour: DateTime<Utc>,
    window_start_day: DateTime<Utc>,
}

impl SlidingWindowRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            windows: HashMap::new(),
            config,
        }
    }

    pub async fn check_rate_limit(&mut self, identifier: &str) -> Result<(), RateLimitError> {
        let now = Utc::now();
        
        let window = self.windows
            .entry(identifier.to_string())
            .or_insert_with(|| self.create_new_window(now));

        // Check if we need to reset windows
        self.check_and_reset_windows(window, now);

        // Check rate limits
        let minute_count = window.minute_counter.load(Ordering::Relaxed);
        if minute_count >= self.config.requests_per_minute {
            return Err(RateLimitError::MinuteLimitExceeded {
                limit: self.config.requests_per_minute,
                window_duration: chrono::Duration::minutes(1),
            });
        }

        let hour_count = window.hour_counter.load(Ordering::Relaxed);
        if hour_count >= self.config.requests_per_hour {
            return Err(RateLimitError::HourLimitExceeded {
                limit: self.config.requests_per_hour,
                window_duration: chrono::Duration::hours(1),
            });
        }

        let day_count = window.day_counter.load(Ordering::Relaxed);
        if day_count >= self.config.requests_per_day {
            return Err(RateLimitError::DayLimitExceeded {
                limit: self.config.requests_per_day,
                window_duration: chrono::Duration::days(1),
            });
        }

        // Increment counters
        window.minute_counter.fetch_add(1, Ordering::Relaxed);
        window.hour_counter.fetch_add(1, Ordering::Relaxed);
        window.day_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn create_new_window(&self, now: DateTime<Utc>) -> WindowCounter {
        WindowCounter {
            minute_counter: AtomicU64::new(0),
            hour_counter: AtomicU64::new(0),
            day_counter: AtomicU64::new(0),
            window_start_minute: now,
            window_start_hour: now,
            window_start_day: now,
        }
    }

    fn check_and_reset_windows(&self, window: &mut WindowCounter, now: DateTime<Utc>) {
        // Reset minute window
        if now - window.window_start_minute >= chrono::Duration::minutes(1) {
            window.minute_counter.store(0, Ordering::Relaxed);
            window.window_start_minute = now;
        }

        // Reset hour window
        if now - window.window_start_hour >= chrono::Duration::hours(1) {
            window.hour_counter.store(0, Ordering::Relaxed);
            window.window_start_hour = now;
        }

        // Reset day window
        if now - window.window_start_day >= chrono::Duration::days(1) {
            window.day_counter.store(0, Ordering::Relaxed);
            window.window_start_day = now;
        }
    }

    pub async fn get_usage_stats(&self, identifier: &str) -> Option<RateLimitStats> {
        self.windows.get(identifier).map(|window| RateLimitStats {
            minute_requests: window.minute_counter.load(Ordering::Relaxed),
            hour_requests: window.hour_counter.load(Ordering::Relaxed),
            day_requests: window.day_counter.load(Ordering::Relaxed),
            minute_limit: self.config.requests_per_minute,
            hour_limit: self.config.requests_per_hour,
            day_limit: self.config.requests_per_day,
            minute_reset: window.window_start_minute + chrono::Duration::minutes(1),
            hour_reset: window.window_start_hour + chrono::Duration::hours(1),
            day_reset: window.window_start_day + chrono::Duration::days(1),
        })
    }

    pub async fn cleanup_expired_windows(&mut self) {
        let now = Utc::now();
        let windows_to_remove: Vec<String> = self.windows
            .iter()
            .filter(|(_, window)| {
                now - window.window_start_day >= chrono::Duration::days(2)
            })
            .map(|(identifier, _)| identifier.clone())
            .collect();

        for identifier in windows_to_remove {
            self.windows.remove(&identifier);
        }

        if !windows_to_remove.is_empty() {
            debug!("Cleaned up {} expired rate limit windows", windows_to_remove.len());
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub minute_requests: u64,
    pub hour_requests: u64,
    pub day_requests: u64,
    pub minute_limit: u64,
    pub hour_limit: u64,
    pub day_limit: u64,
    pub minute_reset: DateTime<Utc>,
    pub hour_reset: DateTime<Utc>,
    pub day_reset: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    #[error("Token bucket exhausted")]
    TokenBucketExhausted,
    #[error("Minute rate limit exceeded: {limit} requests per {window_duration}")]
    MinuteLimitExceeded {
        limit: u64,
        window_duration: chrono::Duration,
    },
    #[error("Hour rate limit exceeded: {limit} requests per {window_duration}")]
    HourLimitExceeded {
        limit: u64,
        window_duration: chrono::Duration,
    },
    #[error("Day rate limit exceeded: {limit} requests per {window_duration}")]
    DayLimitExceeded {
        limit: u64,
        window_duration: chrono::Duration,
    },
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> axum::response::Response {
        match self {
            RateLimitError::TokenBucketExhausted => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please try again later."
            ).into_response(),
            RateLimitError::MinuteLimitExceeded { limit, window_duration } => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                [
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Reset", window_duration.num_seconds().to_string()),
                    ("X-RateLimit-Window", "1m".to_string()),
                ],
                format!("Rate limit exceeded: {} requests per minute", limit)
            ).into_response(),
            RateLimitError::HourLimitExceeded { limit, window_duration } => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                [
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Reset", window_duration.num_seconds().to_string()),
                    ("X-RateLimit-Window", "1h".to_string()),
                ],
                format!("Rate limit exceeded: {} requests per hour", limit)
            ).into_response(),
            RateLimitError::DayLimitExceeded { limit, window_duration } => (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                [
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Reset", window_duration.num_seconds().to_string()),
                    ("X-RateLimit-Window", "1d".to_string()),
                ],
                format!("Rate limit exceeded: {} requests per day", limit)
            ).into_response(),
        }
    }
}

// IP-based rate limiter for additional protection
pub struct IpRateLimiter {
    ip_windows: Arc<RwLock<HashMap<String, TokenBucket>>>,
    global_config: RateLimitConfig,
}

impl IpRateLimiter {
    pub fn new(global_config: RateLimitConfig) -> Self {
        Self {
            ip_windows: Arc::new(RwLock::new(HashMap::new())),
            global_config,
        }
    }

    pub async fn check_ip_rate_limit(&self, ip_address: &str) -> Result<(), RateLimitError> {
        let mut ip_windows = self.ip_windows.write().await;
        
        let bucket = ip_windows
            .entry(ip_address.to_string())
            .or_insert_with(|| TokenBucket::new(self.global_config.clone()));

        bucket.consume().await
    }

    pub async fn get_ip_stats(&self, ip_address: &str) -> Option<u64> {
        let ip_windows = self.ip_windows.read().await;
        ip_windows.get(ip_address).map(|bucket| bucket.get_current_tokens())
    }
}