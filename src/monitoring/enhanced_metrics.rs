use axum::{
    extract::State,
    response::Json,
    routing::get,
    Router,
};
use prometheus::{
    Counter, Gauge, Histogram, OptCounter, OptGauge, OptHistogram, Registry,
    TextEncoder, core::{AtomicI64, GenericGauge},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};

pub struct EnhancedMetricsCollector {
    registry: Registry,
    
    // Request metrics
    pub request_counter: Counter<u64>,
    pub request_duration: Histogram,
    pub active_requests: Gauge,
    pub request_size: Histogram,
    pub response_size: Histogram,
    
    // Model metrics
    pub model_inference_counter: Counter<u64>,
    pub model_inference_duration: Histogram,
    pub model_tokens_generated: Counter<u64>,
    pub model_cache_hits: Counter<u64>,
    pub model_cache_misses: Counter<u64>,
    pub model_memory_usage: Gauge,
    pub model_gpu_usage: Gauge,
    
    // System metrics
    pub cpu_usage: Gauge,
    pub memory_usage: Gauge,
    pub disk_usage: Gauge,
    pub network_io: Counter<u64>,
    pub active_connections: Gauge,
    
    // Security metrics
    pub auth_attempts: Counter<u64>,
    pub auth_failures: Counter<u64>,
    pub rate_limit_exceeded: Counter<u64>,
    pub api_key_usage: Counter<u64>,
    
    // Business metrics
    pub successful_requests: Counter<u64>,
    pub failed_requests: Counter<u64>,
    pub user_sessions: Gauge,
    pub peak_concurrent_users: Gauge,
    
    custom_metrics: Arc<RwLock<HashMap<String, GenericGauge<AtomicI64>>>>,
}

impl EnhancedMetricsCollector {
    pub fn new(registry: Registry) -> Self {
        // Request metrics
        let request_counter = Counter::new(
            "rust_ollama_requests_total",
            "Total number of HTTP requests"
        ).expect("Failed to create request counter");
        
        let request_duration = Histogram::new(
            prometheus::linear_buckets(0.01, 0.1, 20).unwrap(),
            "rust_ollama_request_duration_seconds",
            "Request duration in seconds"
        ).expect("Failed to create request duration histogram");
        
        let active_requests = Gauge::new(
            "rust_ollama_active_requests",
            "Number of currently active requests"
        ).expect("Failed to create active requests gauge");
        
        let request_size = Histogram::new(
            prometheus::linear_buckets(100.0, 1000.0, 20).unwrap(),
            "rust_ollama_request_size_bytes",
            "Request size in bytes"
        ).expect("Failed to create request size histogram");
        
        let response_size = Histogram::new(
            prometheus::linear_buckets(1000.0, 10000.0, 20).unwrap(),
            "rust_ollama_response_size_bytes",
            "Response size in bytes"
        ).expect("Failed to create response size histogram");
        
        // Model metrics
        let model_inference_counter = Counter::new(
            "rust_ollama_model_inferences_total",
            "Total number of model inferences"
        ).expect("Failed to create model inference counter");
        
        let model_inference_duration = Histogram::new(
            prometheus::linear_buckets(0.1, 0.5, 20).unwrap(),
            "rust_ollama_model_inference_duration_seconds",
            "Model inference duration in seconds"
        ).expect("Failed to create model inference duration histogram");
        
        let model_tokens_generated = Counter::new(
            "rust_ollama_tokens_generated_total",
            "Total number of tokens generated"
        ).expect("Failed to create tokens generated counter");
        
        let model_cache_hits = Counter::new(
            "rust_ollama_model_cache_hits_total",
            "Total number of model cache hits"
        ).expect("Failed to create model cache hits counter");
        
        let model_cache_misses = Counter::new(
            "rust_ollama_model_cache_misses_total",
            "Total number of model cache misses"
        ).expect("Failed to create model cache misses counter");
        
        let model_memory_usage = Gauge::new(
            "rust_ollama_model_memory_usage_bytes",
            "Memory usage by loaded models"
        ).expect("Failed to create model memory usage gauge");
        
        let model_gpu_usage = Gauge::new(
            "rust_ollama_model_gpu_usage_percent",
            "GPU usage by loaded models"
        ).expect("Failed to create model GPU usage gauge");
        
        // System metrics
        let cpu_usage = Gauge::new(
            "rust_ollama_cpu_usage_percent",
            "CPU usage percentage"
        ).expect("Failed to create CPU usage gauge");
        
        let memory_usage = Gauge::new(
            "rust_ollama_memory_usage_percent",
            "Memory usage percentage"
        ).expect("Failed to create memory usage gauge");
        
        let disk_usage = Gauge::new(
            "rust_ollama_disk_usage_percent",
            "Disk usage percentage"
        ).expect("Failed to create disk usage gauge");
        
        let network_io = Counter::new(
            "rust_ollama_network_io_bytes_total",
            "Total network I/O in bytes"
        ).expect("Failed to create network I/O counter");
        
        let active_connections = Gauge::new(
            "rust_ollama_active_connections",
            "Number of active connections"
        ).expect("Failed to create active connections gauge");
        
        // Security metrics
        let auth_attempts = Counter::new(
            "rust_ollama_auth_attempts_total",
            "Total number of authentication attempts"
        ).expect("Failed to create auth attempts counter");
        
        let auth_failures = Counter::new(
            "rust_ollama_auth_failures_total",
            "Total number of authentication failures"
        ).expect("Failed to create auth failures counter");
        
        let rate_limit_exceeded = Counter::new(
            "rust_ollama_rate_limit_exceeded_total",
            "Total number of rate limit exceeded events"
        ).expect("Failed to create rate limit exceeded counter");
        
        let api_key_usage = Counter::new(
            "rust_ollama_api_key_usage_total",
            "Total number of API key usage events"
        ).expect("Failed to create API key usage counter");
        
        // Business metrics
        let successful_requests = Counter::new(
            "rust_ollama_successful_requests_total",
            "Total number of successful requests"
        ).expect("Failed to create successful requests counter");
        
        let failed_requests = Counter::new(
            "rust_ollama_failed_requests_total",
            "Total number of failed requests"
        ).expect("Failed to create failed requests counter");
        
        let user_sessions = Gauge::new(
            "rust_ollama_user_sessions",
            "Number of active user sessions"
        ).expect("Failed to create user sessions gauge");
        
        let peak_concurrent_users = Gauge::new(
            "rust_ollama_peak_concurrent_users",
            "Peak number of concurrent users"
        ).expect("Failed to create peak concurrent users gauge");

        let collector = Self {
            registry: registry.clone(),
            request_counter,
            request_duration,
            active_requests,
            request_size,
            response_size,
            model_inference_counter,
            model_inference_duration,
            model_tokens_generated,
            model_cache_hits,
            model_cache_misses,
            model_memory_usage,
            model_gpu_usage,
            cpu_usage,
            memory_usage,
            disk_usage,
            network_io,
            active_connections,
            auth_attempts,
            auth_failures,
            rate_limit_exceeded,
            api_key_usage,
            successful_requests,
            failed_requests,
            user_sessions,
            peak_concurrent_users,
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register all metrics
        let _ = collector.registry.register(Box::new(collector.request_counter.clone()));
        let _ = collector.registry.register(Box::new(collector.request_duration.clone()));
        let _ = collector.registry.register(Box::new(collector.active_requests.clone()));
        let _ = collector.registry.register(Box::new(collector.request_size.clone()));
        let _ = collector.registry.register(Box::new(collector.response_size.clone()));
        
        let _ = collector.registry.register(Box::new(collector.model_inference_counter.clone()));
        let _ = collector.registry.register(Box::new(collector.model_inference_duration.clone()));
        let _ = collector.registry.register(Box::new(collector.model_tokens_generated.clone()));
        let _ = collector.registry.register(Box::new(collector.model_cache_hits.clone()));
        let _ = collector.registry.register(Box::new(collector.model_cache_misses.clone()));
        let _ = collector.registry.register(Box::new(collector.model_memory_usage.clone()));
        let _ = collector.registry.register(Box::new(collector.model_gpu_usage.clone()));
        
        let _ = collector.registry.register(Box::new(collector.cpu_usage.clone()));
        let _ = collector.registry.register(Box::new(collector.memory_usage.clone()));
        let _ = collector.registry.register(Box::new(collector.disk_usage.clone()));
        let _ = collector.registry.register(Box::new(collector.network_io.clone()));
        let _ = collector.registry.register(Box::new(collector.active_connections.clone()));
        
        let _ = collector.registry.register(Box::new(collector.auth_attempts.clone()));
        let _ = collector.registry.register(Box::new(collector.auth_failures.clone()));
        let _ = collector.registry.register(Box::new(collector.rate_limit_exceeded.clone()));
        let _ = collector.registry.register(Box::new(collector.api_key_usage.clone()));
        
        let _ = collector.registry.register(Box::new(collector.successful_requests.clone()));
        let _ = collector.registry.register(Box::new(collector.failed_requests.clone()));
        let _ = collector.registry.register(Box::new(collector.user_sessions.clone()));
        let _ = collector.registry.register(Box::new(collector.peak_concurrent_users.clone()));

        collector
    }

    #[instrument(skip(self))]
    pub fn record_request(&self, duration: Duration, success: bool, request_size: Option<usize>, response_size: Option<usize>) {
        self.request_counter.inc();
        self.request_duration.observe(duration.as_secs_f64());
        
        if let Some(size) = request_size {
            self.request_size.observe(size as f64);
        }
        
        if let Some(size) = response_size {
            self.response_size.observe(size as f64);
        }
        
        if success {
            self.successful_requests.inc();
        } else {
            self.failed_requests.inc();
        }
    }

    #[instrument(skip(self))]
    pub fn increment_active_requests(&self) {
        self.active_requests.inc();
    }

    #[instrument(skip(self))]
    pub fn decrement_active_requests(&self) {
        self.active_requests.dec();
    }

    #[instrument(skip(self))]
    pub fn record_model_inference(&self, duration: Duration, tokens_generated: Option<u64>, cache_hit: bool) {
        self.model_inference_counter.inc();
        self.model_inference_duration.observe(duration.as_secs_f64());
        
        if let Some(tokens) = tokens_generated {
            self.model_tokens_generated.inc_by(tokens);
        }
        
        if cache_hit {
            self.model_cache_hits.inc();
        } else {
            self.model_cache_misses.inc();
        }
    }

    #[instrument(skip(self))]
    pub fn record_auth_attempt(&self, success: bool) {
        self.auth_attempts.inc();
        
        if !success {
            self.auth_failures.inc();
        }
    }

    #[instrument(skip(self))]
    pub fn record_rate_limit_exceeded(&self) {
        self.rate_limit_exceeded.inc();
    }

    #[instrument(skip(self))]
    pub fn record_api_key_usage(&self) {
        self.api_key_usage.inc();
    }

    #[instrument(skip(self))]
    pub fn update_system_metrics(&self, cpu: f64, memory: f64, disk: f64) {
        self.cpu_usage.set(cpu);
        self.memory_usage.set(memory);
        self.disk_usage.set(disk);
    }

    #[instrument(skip(self))]
    pub fn update_network_io(&self, bytes: u64) {
        self.network_io.inc_by(bytes);
    }

    #[instrument(skip(self))]
    pub fn update_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }

    #[instrument(skip(self))]
    pub fn update_user_sessions(&self, count: i64) {
        self.user_sessions.set(count);
    }

    #[instrument(skip(self))]
    pub fn update_peak_concurrent_users(&self, count: i64) {
        self.peak_concurrent_users.set(count);
    }

    #[instrument(skip(self))]
    pub fn update_model_memory_usage(&self, bytes: i64) {
        self.model_memory_usage.set(bytes);
    }

    #[instrument(skip(self))]
    pub fn update_model_gpu_usage(&self, percent: f64) {
        self.model_gpu_usage.set(percent);
    }

    pub async fn create_custom_metric(&self, name: &str, help: &str) -> Result<(), MetricsError> {
        let mut custom_metrics = self.custom_metrics.write().await;
        
        if custom_metrics.contains_key(name) {
            return Err(MetricsError::MetricAlreadyExists(name.to_string()));
        }

        let gauge = GenericGauge::new(name, help)
            .map_err(|e| MetricsError::MetricCreationError(e.to_string()))?;
        
        let _ = self.registry.register(Box::new(gauge.clone()));
        custom_metrics.insert(name.to_string(), gauge);

        Ok(())
    }

    pub async fn set_custom_metric(&self, name: &str, value: i64) {
        let custom_metrics = self.custom_metrics.read().await;
        if let Some(gauge) = custom_metrics.get(name) {
            gauge.set(value);
        }
    }

    pub async fn increment_custom_metric(&self, name: &str, value: i64) {
        let custom_metrics = self.custom_metrics.read().await;
        if let Some(gauge) = custom_metrics.get(name) {
            gauge.add(value);
        }
    }

    pub async fn get_prometheus_metrics(&self) -> Result<String, MetricsError> {
        let metric_families = self.registry.gather();
        let encoder = TextEncoder::new();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)
            .map_err(|e| MetricsError::EncodingError(e.to_string()))?;
        
        String::from_utf8(buffer)
            .map_err(|e| MetricsError::EncodingError(e.to_string()))
    }

    pub async fn get_metrics_snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            request_counter: self.request_counter.get(),
            active_requests: self.active_requests.get(),
            model_inference_counter: self.model_inference_counter.get(),
            model_tokens_generated: self.model_tokens_generated.get(),
            cpu_usage: self.cpu_usage.get(),
            memory_usage: self.memory_usage.get(),
            disk_usage: self.disk_usage.get(),
            active_connections: self.active_connections.get(),
            user_sessions: self.user_sessions.get(),
            auth_attempts: self.auth_attempts.get(),
            auth_failures: self.auth_failures.get(),
            successful_requests: self.successful_requests.get(),
            failed_requests: self.failed_requests.get(),
            rate_limit_exceeded: self.rate_limit_exceeded.get(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub request_counter: u64,
    pub active_requests: i64,
    pub model_inference_counter: u64,
    pub model_tokens_generated: u64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub active_connections: i64,
    pub user_sessions: i64,
    pub auth_attempts: u64,
    pub auth_failures: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub rate_limit_exceeded: u64,
}

pub struct MetricsMiddleware {
    pub collector: Arc<EnhancedMetricsCollector>,
    start_time: Instant,
}

impl MetricsMiddleware {
    pub fn new(collector: Arc<EnhancedMetricsCollector>) -> Self {
        Self {
            collector,
            start_time: Instant::now(),
        }
    }

    pub fn on_request_start(&self) {
        self.collector.increment_active_requests();
    }

    pub fn on_request_end(&self, success: bool, request_size: Option<usize>, response_size: Option<usize>) {
        let duration = self.start_time.elapsed();
        self.collector.record_request(duration, success, request_size, response_size);
        self.collector.decrement_active_requests();
    }
}

pub fn metrics_routes(collector: Arc<EnhancedMetricsCollector>) -> Router {
    Router::new()
        .route("/metrics", get(prometheus_metrics_endpoint))
        .route("/metrics/snapshot", get(metrics_snapshot_endpoint))
        .route("/metrics/health", get(metrics_health_endpoint))
        .with_state(collector)
}

async fn prometheus_metrics_endpoint(
    State(collector): State<Arc<EnhancedMetricsCollector>>,
) -> Result<String, axum::http::StatusCode> {
    match collector.get_prometheus_metrics().await {
        Ok(metrics) => Ok(metrics),
        Err(e) => {
            error!("Failed to get Prometheus metrics: {}", e);
            Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn metrics_snapshot_endpoint(
    State(collector): State<Arc<EnhancedMetricsCollector>>,
) -> Result<Json<MetricsSnapshot>, axum::http::StatusCode> {
    let snapshot = collector.get_metrics_snapshot().await;
    Ok(Json(snapshot))
}

async fn metrics_health_endpoint(
    State(collector): State<Arc<EnhancedMetricsCollector>>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    let health = serde_json::json!({
        "status": "healthy",
        "registry_count": collector.registry.gather().len(),
        "timestamp": chrono::Utc::now()
    });
    Ok(Json(health))
}

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("Metric already exists: {0}")]
    MetricAlreadyExists(String),
    
    #[error("Metric creation error: {0}")]
    MetricCreationError(String),
    
    #[error("Encoding error: {0}")]
    EncodingError(String),
}

// Helper functions for easy metric recording
pub struct RequestRecorder {
    collector: Arc<EnhancedMetricsCollector>,
    start_time: Instant,
    request_size: Option<usize>,
}

impl RequestRecorder {
    pub fn new(collector: Arc<EnhancedMetricsCollector>, request_size: Option<usize>) -> Self {
        collector.increment_active_requests();
        
        Self {
            collector,
            start_time: Instant::now(),
            request_size,
        }
    }

    pub fn success(self, response_size: Option<usize>) {
        let duration = self.start_time.elapsed();
        self.collector.record_request(duration, true, self.request_size, response_size);
        self.collector.decrement_active_requests();
        drop(self); // Ensure proper cleanup
    }

    pub fn failure(self, response_size: Option<usize>) {
        let duration = self.start_time.elapsed();
        self.collector.record_request(duration, false, self.request_size, response_size);
        self.collector.decrement_active_requests();
        drop(self); // Ensure proper cleanup
    }
}

impl Drop for RequestRecorder {
    fn drop(&mut self) {
        // This ensures we always decrement active requests even if success/failure isn't called
        self.collector.decrement_active_requests();
    }
}