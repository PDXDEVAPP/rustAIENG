use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error, debug, instrument, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_opentelemetry::OpenTelemetryLayer;
use opentelemetry::sdk::trace as sdktrace;
use opentelemetry::{KeyValue, global};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: String,
    pub message: String,
    pub logger: String,
    pub module: String,
    pub file: String,
    pub line: u32,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub user_id: Option<String>,
    pub request_id: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub performance: Option<PerformanceData>,
    pub security: Option<SecurityData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub duration_ms: u64,
    pub memory_usage_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
    pub database_queries: Option<u32>,
    pub cache_hits: Option<u32>,
    pub cache_misses: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityData {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub api_key_id: Option<String>,
    pub auth_method: Option<String>,
    pub risk_score: Option<f64>,
    pub anomalies: Option<Vec<String>>,
}

pub struct StructuredLogger {
    correlation_id: String,
    user_id: Option<String>,
    ip_address: Option<String>,
    request_size: Option<usize>,
    performance_data: PerformanceData,
}

impl StructuredLogger {
    pub fn new(correlation_id: String) -> Self {
        Self {
            correlation_id,
            user_id: None,
            ip_address: None,
            request_size: None,
            performance_data: PerformanceData {
                duration_ms: 0,
                memory_usage_mb: None,
                cpu_usage_percent: None,
                database_queries: None,
                cache_hits: None,
                cache_misses: None,
            },
        }
    }

    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_ip_address(mut self, ip: String) -> Self {
        self.ip_address = Some(ip);
        self
    }

    pub fn with_request_size(mut self, size: usize) -> Self {
        self.request_size = Some(size);
        self
    }

    #[instrument(skip(self))]
    pub fn info(&self, message: &str, metadata: HashMap<String, serde_json::Value>) {
        let log_event = self.create_log_event("INFO", message, metadata);
        info!("{} | {}", self.correlation_id, message);
        
        // In production, you would send this to your logging backend
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn warn(&self, message: &str, metadata: HashMap<String, serde_json::Value>) {
        let log_event = self.create_log_event("WARN", message, metadata);
        warn!("{} | {}", self.correlation_id, message);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn error(&self, message: &str, error: impl std::error::Error, metadata: HashMap<String, serde_json::Value>) {
        let mut enhanced_metadata = metadata;
        enhanced_metadata.insert("error_type".to_string(), error.type_id().to_string().into());
        enhanced_metadata.insert("error_message".to_string(), error.to_string().into());

        let log_event = self.create_log_event("ERROR", message, enhanced_metadata);
        error!("{} | {}: {}", self.correlation_id, message, error);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn debug(&self, message: &str, metadata: HashMap<String, serde_json::Value>) {
        let log_event = self.create_log_event("DEBUG", message, metadata);
        debug!("{} | {}", self.correlation_id, message);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn request_start(&self, method: &str, path: &str) {
        let metadata = HashMap::from([
            ("event_type".to_string(), "request_start".into()),
            ("method".to_string(), method.into()),
            ("path".to_string(), path.into()),
            ("request_size".to_string(), self.request_size.map(|s| s as i64).unwrap_or(0).into()),
        ]);

        let log_event = self.create_log_event("INFO", "Request started", metadata);
        info!("{} | {} {} started", self.correlation_id, method, path);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn request_end(&self, method: &str, path: &str, status_code: u16, duration_ms: u64) {
        let metadata = HashMap::from([
            ("event_type".to_string(), "request_end".into()),
            ("method".to_string(), method.into()),
            ("path".to_string(), path.into()),
            ("status_code".to_string(), status_code.into()),
            ("duration_ms".to_string(), duration_ms.into()),
            ("success".to_string(), (status_code < 400).into()),
        ]);

        let log_event = self.create_log_event("INFO", "Request completed", metadata);
        info!("{} | {} {} completed with {} ({}ms)", self.correlation_id, method, path, status_code, duration_ms);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn authentication_attempt(&self, success: bool, method: &str, api_key_id: Option<String>) {
        let mut metadata = HashMap::from([
            ("event_type".to_string(), "authentication".into()),
            ("success".to_string(), success.into()),
            ("method".to_string(), method.into()),
        ]);

        if let Some(key_id) = api_key_id {
            metadata.insert("api_key_id".to_string(), key_id.into());
        }

        let level = if success { "INFO" } else { "WARN" };
        let message = if success { "Authentication successful" } else { "Authentication failed" };
        
        let log_event = self.create_log_event(level, message, metadata);
        
        if success {
            info!("{} | {}: {}", self.correlation_id, message, method);
        } else {
            warn!("{} | {}: {}", self.correlation_id, message, method);
        }
        
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn rate_limit_exceeded(&self, identifier: &str, limit: u64) {
        let metadata = HashMap::from([
            ("event_type".to_string(), "rate_limit_exceeded".into()),
            ("identifier".to_string(), identifier.into()),
            ("limit".to_string(), limit.into()),
        ]);

        let log_event = self.create_log_event("WARN", "Rate limit exceeded", metadata);
        warn!("{} | Rate limit exceeded for {} (limit: {})", self.correlation_id, identifier, limit);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn model_inference(&self, model_name: &str, duration_ms: u64, tokens_generated: u64, cache_hit: bool) {
        let metadata = HashMap::from([
            ("event_type".to_string(), "model_inference".into()),
            ("model_name".to_string(), model_name.into()),
            ("duration_ms".to_string(), duration_ms.into()),
            ("tokens_generated".to_string(), tokens_generated.into()),
            ("cache_hit".to_string(), cache_hit.into()),
        ]);

        let log_event = self.create_log_event("INFO", "Model inference completed", metadata);
        info!("{} | Model inference: {} ({}ms, {} tokens, cache: {})", 
              self.correlation_id, model_name, duration_ms, tokens_generated, cache_hit);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn security_event(&self, event_type: &str, severity: &str, description: &str, details: HashMap<String, serde_json::Value>) {
        let mut metadata = details;
        metadata.insert("event_type".to_string(), event_type.into());
        metadata.insert("security_severity".to_string(), severity.into());

        let log_event = self.create_log_event("WARN", description, metadata);
        warn!("{} | Security event [{}]: {}", self.correlation_id, severity, description);
        self.send_to_logging_backend(&log_event);
    }

    #[instrument(skip(self))]
    pub fn performance_issue(&self, metric_name: &str, value: f64, threshold: f64) {
        let metadata = HashMap::from([
            ("event_type".to_string(), "performance_issue".into()),
            ("metric_name".to_string(), metric_name.into()),
            ("value".to_string(), value.into()),
            ("threshold".to_string(), threshold.into()),
            ("exceeded".to_string(), (value > threshold).into()),
        ]);

        let log_event = self.create_log_event("WARN", "Performance threshold exceeded", metadata);
        warn!("{} | Performance issue: {} = {} (threshold: {})", 
              self.correlation_id, metric_name, value, threshold);
        self.send_to_logging_backend(&log_event);
    }

    fn create_log_event(&self, level: &str, message: &str, metadata: HashMap<String, serde_json::Value>) -> LogEvent {
        LogEvent {
            timestamp: chrono::Utc::now(),
            level: level.to_string(),
            message: message.to_string(),
            logger: "rust_ollama".to_string(),
            module: module_path!().to_string(),
            file: file!().to_string(),
            line: line!(),
            trace_id: Some(self.correlation_id.clone()),
            span_id: None,
            user_id: self.user_id.clone(),
            request_id: Some(self.correlation_id.clone()),
            metadata,
            performance: Some(self.performance_data.clone()),
            security: Some(SecurityData {
                ip_address: self.ip_address.clone(),
                user_agent: None, // Would be extracted from request headers
                api_key_id: None, // Would be extracted from authenticated context
                auth_method: None,
                risk_score: None,
                anomalies: None,
            }),
        }
    }

    fn send_to_logging_backend(&self, log_event: &LogEvent) {
        // In production, this would send to your preferred logging backend
        // such as:
        // - ELK Stack (Elasticsearch + Logstash + Kibana)
        // - Splunk
        // - Datadog
        // - Grafana Loki
        // - CloudWatch Logs
        // etc.

        // For now, we'll just print in debug mode
        if log_event.level == "ERROR" || log_event.level == "WARN" {
            debug!("Structured log: {}", serde_json::to_string(log_event).unwrap_or_default());
        }
    }
}

pub struct LoggingConfig {
    pub structured_logging_enabled: bool,
    pub correlation_id_header: String,
    pub sensitive_fields: Vec<String>,
    pub log_level: String,
    pub output_format: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            structured_logging_enabled: true,
            correlation_id_header: "x-correlation-id".to_string(),
            sensitive_fields: vec!["password".to_string(), "token".to_string(), "api_key".to_string()],
            log_level: "INFO".to_string(),
            output_format: "json".to_string(),
            batch_size: 1000,
            flush_interval_ms: 5000,
        }
    }
}

pub fn initialize_structured_logging(config: &LoggingConfig) -> Result<(), LoggingError> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let stdout_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_level(true)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.log_level));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .init();

    info!("Structured logging initialized with level: {}", config.log_level);
    Ok(())
}

pub fn create_correlation_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub struct RequestContext {
    pub correlation_id: String,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub request_size: Option<usize>,
}

impl RequestContext {
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Self {
        let correlation_id = headers
            .get("x-correlation-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_else(|| {
                let id = create_correlation_id();
                info!("Generated correlation ID: {}", id);
                id
            })
            .to_string();

        let user_id = headers
            .get("x-user-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let ip_address = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .or_else(|| headers.get("x-real-ip").and_then(|v| v.to_str().ok()))
            .map(|s| s.to_string());

        let user_agent = headers
            .get("user-agent")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Self {
            correlation_id,
            user_id,
            ip_address,
            user_agent,
            request_size: None,
        }
    }

    pub fn create_logger(&self) -> StructuredLogger {
        let mut logger = StructuredLogger::new(self.correlation_id.clone());
        
        if let Some(user_id) = &self.user_id {
            logger = logger.with_user_id(user_id.clone());
        }
        
        if let Some(ip) = &self.ip_address {
            logger = logger.with_ip_address(ip.clone());
        }
        
        if let Some(size) = self.request_size {
            logger = logger.with_request_size(size);
        }
        
        logger
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    #[error("Failed to initialize logging: {0}")]
    InitializationError(String),
    
    #[error("Failed to serialize log event: {0}")]
    SerializationError(String),
    
    #[error("Failed to send to logging backend: {0}")]
    BackendError(String),
}

// Utility function for safe log field extraction
pub fn extract_safe_field<'a>(field_name: &str, data: &'a serde_json::Value, sensitive_fields: &[String]) -> Option<&'a str> {
    if sensitive_fields.iter().any(|s| s.eq_ignore_ascii_case(field_name)) {
        return Some("[REDACTED]");
    }
    
    match data {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(n) => Some(&n.to_string()),
        serde_json::Value::Bool(b) => Some(&b.to_string()),
        _ => None,
    }
}

// Health check for logging system
pub async fn logging_health_check() -> Result<(), LoggingError> {
    // Test if we can create and send a log event
    let test_logger = StructuredLogger::new(create_correlation_id());
    let metadata = HashMap::from([("test".to_string(), true.into())]);
    
    test_logger.info("Logging health check", metadata);
    
    Ok(())
}