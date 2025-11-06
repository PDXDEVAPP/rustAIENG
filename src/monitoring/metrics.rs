use prometheus::{Counter, Gauge, Histogram, TextEncoder, Encoder};
use prometheus_core::Registry;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug)]
pub struct RequestMetrics {
    pub request_id: String,
    pub model_name: String,
    pub endpoint: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status_code: Option<u16>,
    pub tokens_generated: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModelMetrics {
    pub model_name: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub average_response_time_ms: f64,
    pub last_used: DateTime<Utc>,
    pub memory_usage_mb: u64,
}

#[derive(Clone, Debug)]
pub struct ServerMetrics {
    pub total_requests: u64,
    pub active_connections: u32,
    pub loaded_models: u32,
    pub uptime_seconds: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
}

pub struct MetricsCollector {
    registry: Registry,
    request_counter: Counter,
    request_duration: Histogram,
    active_connections: Gauge,
    loaded_models: Gauge,
    memory_usage: Gauge,
    request_metrics: Arc<RwLock<Vec<RequestMetrics>>>,
    model_metrics: Arc<RwLock<HashMap<String, ModelMetrics>>>,
    server_start_time: DateTime<Utc>,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        // Request metrics
        let request_counter = Counter::new(
            "rust_ollama_requests_total",
            "Total number of requests processed"
        )?.register(&registry);
        
        let request_duration = Histogram::new(
            prometheus::HistogramOpts::new(
                "rust_ollama_request_duration_seconds",
                "Request duration in seconds"
            ).buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        )?.register(&registry);
        
        // Connection metrics
        let active_connections = Gauge::new(
            "rust_ollama_active_connections",
            "Number of active connections"
        )?.register(&registry);
        
        let loaded_models = Gauge::new(
            "rust_ollama_loaded_models",
            "Number of models currently loaded"
        )?.register(&registry);
        
        let memory_usage = Gauge::new(
            "rust_ollama_memory_usage_bytes",
            "Memory usage in bytes"
        )?.register(&registry);

        Ok(Self {
            registry,
            request_counter,
            request_duration,
            active_connections,
            loaded_models,
            memory_usage,
            request_metrics: Arc::new(RwLock::new(Vec::new())),
            model_metrics: Arc::new(RwLock::new(HashMap::new())),
            server_start_time: Utc::now(),
        })
    }

    pub fn record_request(&self, metrics: RequestMetrics) {
        self.request_counter.inc();
        
        if let Some(end_time) = metrics.end_time {
            let duration = (end_time - metrics.start_time).num_milliseconds() as f64 / 1000.0;
            self.request_duration.observe(duration);
        }
        
        // Store metrics for detailed analysis
        tokio::spawn(async move {
            let mut metrics_vec = self.request_metrics.write().await;
            metrics_vec.push(metrics);
            
            // Keep only last 1000 requests
            if metrics_vec.len() > 1000 {
                metrics_vec.drain(0..(metrics_vec.len() - 1000));
            }
        });
    }

    pub fn update_connection_count(&self, count: u32) {
        self.active_connections.set(count as f64);
    }

    pub fn update_loaded_models(&self, count: u32) {
        self.loaded_models.set(count as f64);
    }

    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage.set(bytes as f64);
    }

    pub fn record_model_metrics(&self, metrics: ModelMetrics) {
        tokio::spawn(async move {
            let mut model_metrics_map = self.model_metrics.write().await;
            model_metrics_map.insert(metrics.model_name.clone(), metrics);
        });
    }

    pub async fn generate_prometheus_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut metric_families = self.registry.gather();
        
        // Add custom metrics
        self.add_custom_metrics(&mut metric_families);
        
        let encoder = TextEncoder::new();
        let encoded = encoder.encode_to_string(&metric_families)?;
        
        Ok(encoded)
    }

    fn add_custom_metrics(&self, families: &mut Vec<prometheus::proto::MetricFamily>) {
        // Add uptime metric
        let uptime_seconds = (Utc::now() - self.server_start_time).num_seconds() as f64;
        
        let mut uptime_family = prometheus::proto::MetricFamily::new();
        uptime_family.set_name("rust_ollama_uptime_seconds".to_string());
        uptime_family.set_help("Server uptime in seconds".to_string());
        uptime_family.set_r#type(prometheus::proto::MetricType::GAUGE);
        
        let mut uptime_metric = prometheus::proto::Metric::new();
        let mut gauge = prometheus::proto::Gauge::new();
        gauge.set_value(uptime_seconds);
        
        let mut label = prometheus::proto::Label::new();
        label.set_name("version".to_string());
        label.set_value(env!("CARGO_PKG_VERSION").to_string());
        
        let mut label_vec = prometheus::proto::LabelPair::new();
        label_vec.set_name("version".to_string());
        label_vec.set_value(env!("CARGO_PKG_VERSION").to_string());
        
        uptime_metric.mut_gauge().set_value(uptime_seconds);
        uptime_metric.mut_label().push(label_vec);
        uptime_family.mut_metric().push(uptime_metric);
        
        families.push(uptime_family);
    }

    pub async fn get_detailed_metrics(&self) -> ServerMetrics {
        let uptime_seconds = (Utc::now() - self.server_start_time).num_seconds() as u64;
        let active_connections = self.active_connections.get() as u32;
        let loaded_models = self.loaded_models.get() as u32;
        let memory_usage_mb = (self.memory_usage.get() / (1024 * 1024)) as u64;
        
        // Get CPU usage (placeholder implementation)
        let cpu_usage_percent = self.estimate_cpu_usage();
        
        ServerMetrics {
            total_requests: self.request_counter.get() as u64,
            active_connections,
            loaded_models,
            uptime_seconds,
            memory_usage_mb,
            cpu_usage_percent,
        }
    }

    fn estimate_cpu_usage(&self) -> f32 {
        // This is a placeholder - in a real implementation you'd get actual CPU usage
        // You could use sysinfo crate or read from /proc on Linux
        0.5 // 50% CPU usage as placeholder
    }

    pub async fn get_model_statistics(&self) -> HashMap<String, ModelMetrics> {
        self.model_metrics.read().await.clone()
    }

    pub async fn get_recent_requests(&self, limit: usize) -> Vec<RequestMetrics> {
        let metrics_vec = self.request_metrics.read().await;
        let start_idx = if metrics_vec.len() > limit {
            metrics_vec.len() - limit
        } else {
            0
        };
        metrics_vec[start_idx..].to_vec()
    }

    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let server_metrics = self.get_detailed_metrics().await;
        let model_metrics = self.get_model_statistics().await;
        let recent_requests = self.get_recent_requests(100).await;
        
        // Calculate performance metrics
        let avg_response_time = if !recent_requests.is_empty() {
            let total_duration: f64 = recent_requests
                .iter()
                .filter_map(|r| r.end_time.map(|end| (end - r.start_time).num_milliseconds() as f64))
                .sum();
            total_duration / recent_requests.len() as f64
        } else {
            0.0
        };
        
        let error_rate = if !recent_requests.is_empty() {
            let error_count = recent_requests
                .iter()
                .filter(|r| r.error_message.is_some() || r.status_code.map_or(false, |s| s >= 400))
                .count();
            (error_count as f64 / recent_requests.len() as f64) * 100.0
        } else {
            0.0
        };
        
        PerformanceSummary {
            server_metrics,
            model_count: model_metrics.len(),
            average_response_time_ms: avg_response_time,
            error_rate_percent: error_rate,
            total_requests: server_metrics.total_requests,
            uptime_hours: server_metrics.uptime_seconds as f64 / 3600.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerformanceSummary {
    pub server_metrics: ServerMetrics,
    pub model_count: usize,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub total_requests: u64,
    pub uptime_hours: f64,
}

pub struct PerformanceMonitor {
    metrics_collector: Arc<MetricsCollector>,
    monitor_interval: std::time::Duration,
}

impl PerformanceMonitor {
    pub fn new(metrics_collector: Arc<MetricsCollector>) -> Self {
        Self {
            metrics_collector,
            monitor_interval: std::time::Duration::from_secs(30),
        }
    }

    pub async fn start_monitoring(&self) {
        let collector = self.metrics_collector.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Update system metrics
                Self::update_system_metrics(&collector).await;
                
                // Clean up old metrics
                Self::cleanup_old_metrics(&collector).await;
            }
        });
        
        info!("Performance monitoring started");
    }

    async fn update_system_metrics(collector: &Arc<MetricsCollector>) {
        // Update memory usage
        if let Ok(memory_info) = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::new()
                .with_memory()
                .with_cpu(),
        ) {
            collector.update_memory_usage(memory_info.used_memory());
        }
        
        // Update active connections count
        // This would come from your WebSocket/connection manager
        let active_connections = 0; // Placeholder
        collector.update_connection_count(active_connections);
    }

    async fn cleanup_old_metrics(collector: &Arc<MetricsCollector>) {
        let cutoff_time = Utc::now() - chrono::Duration::hours(1);
        
        let mut metrics_vec = collector.request_metrics.write().await;
        metrics_vec.retain(|m| m.start_time > cutoff_time);
        
        info!("Cleaned up old metrics, {} requests remaining", metrics_vec.len());
    }
}

// Global metrics collector
use once_cell::sync::Lazy;

pub static METRICS_COLLECTOR: Lazy<Arc<MetricsCollector>> = Lazy::new(|| {
    Arc::new(MetricsCollector::new().unwrap_or_else(|e| {
        error!("Failed to initialize metrics collector: {}", e);
        panic!("Metrics collector initialization failed");
    }))
});

pub fn get_metrics_collector() -> Arc<MetricsCollector> {
    METRICS_COLLECTOR.clone()
}

pub struct RequestTimer {
    start_time: std::time::Instant,
    metrics_collector: Arc<MetricsCollector>,
    request_id: String,
    model_name: String,
    endpoint: String,
}

impl RequestTimer {
    pub fn new(
        metrics_collector: Arc<MetricsCollector>,
        request_id: String,
        model_name: String,
        endpoint: String,
    ) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            metrics_collector,
            request_id,
            model_name,
            endpoint,
        }
    }

    pub fn finish_with_success(
        self,
        status_code: u16,
        tokens_generated: Option<u32>,
        prompt_tokens: Option<u32>,
    ) {
        let metrics = RequestMetrics {
            request_id: self.request_id,
            model_name: self.model_name,
            endpoint: self.endpoint,
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(self.start_time.elapsed().as_millis() as i64),
            end_time: Some(Utc::now()),
            status_code: Some(status_code),
            tokens_generated,
            prompt_tokens,
            error_message: None,
        };
        
        self.metrics_collector.record_request(metrics);
    }

    pub fn finish_with_error(self, status_code: u16, error_message: String) {
        let metrics = RequestMetrics {
            request_id: self.request_id,
            model_name: self.model_name,
            endpoint: self.endpoint,
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(self.start_time.elapsed().as_millis() as i64),
            end_time: Some(Utc::now()),
            status_code: Some(status_code),
            tokens_generated: None,
            prompt_tokens: None,
            error_message: Some(error_message),
        };
        
        self.metrics_collector.record_request(metrics);
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        // If the request wasn't finished explicitly, record it as an error
        if std::thread::panicking() {
            self.finish_with_error(500, "Panic during request processing".to_string());
        }
    }
}