use opentelemetry::{
    global,
    trace::{TraceContextExt, Tracer, TracerProvider},
    baggage::BaggageExt,
    Context,
    KeyValue,
};
use opentelemetry_sdk::trace::{Config, TracerProvider as SdkTracerProvider};
use opentelemetry_otlp::TelemetryDriver;
use opentelemetry_semantic_conventions::trace::{self, http};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{instrument, span, Instrument};
use tracing_opentelemetry::OpenTelemetryLayer;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub attributes: HashMap<String, serde_json::Value>,
    pub events: Vec<TraceEvent>,
    pub status: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub name: String,
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct DistributedTracing {
    tracer: Arc<dyn Tracer + Send + Sync>,
    baggage: Arc<parking_lot::RwLock<HashMap<String, String>>>,
}

impl DistributedTracing {
    pub fn new() -> Result<Self, TracingError> {
        let tracer_provider = SdkTracerProvider::builder()
            .with_simple_exporter()
            .build();

        global::set_tracer_provider(tracer_provider);
        
        let tracer = global::tracer("rust_ollama");

        Ok(Self {
            tracer: Arc::new(tracer),
            baggage: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        })
    }

    pub fn get_tracer(&self) -> Arc<dyn Tracer + Send + Sync> {
        self.tracer.clone()
    }

    #[instrument(skip(self))]
    pub fn create_span(&self, operation_name: &str, attributes: HashMap<String, serde_json::Value>) -> Span {
        let tracer = self.get_tracer();
        let span = tracer.span_builder(operation_name)
            .with_attributes(
                attributes
                    .iter()
                    .map(|(k, v)| KeyValue::new(k.clone(), v.to_string()))
                    .collect()
            )
            .start(&tracer);

        Span {
            inner: span,
            operation_name: operation_name.to_string(),
            attributes,
            events: Vec::new(),
        }
    }

    #[instrument(skip(self))]
    pub fn create_root_span(&self, operation_name: &str, attributes: HashMap<String, serde_json::Value>) -> Span {
        let tracer = self.get_tracer();
        
        let mut builder = tracer.span_builder(operation_name)
            .with_attributes(
                attributes
                    .iter()
                    .map(|(k, v)| KeyValue::new(k.clone(), v.to_string()))
                    .collect()
            );

        // Add trace context attributes
        builder = builder.with_attributes(vec![
            KeyValue::new("service.name", "rust_ollama"),
            KeyValue::new("service.version", "0.2.0"),
            KeyValue::new("deployment.environment", "production"),
        ]);

        let span = builder.start(&tracer);

        Span {
            inner: span,
            operation_name: operation_name.to_string(),
            attributes,
            events: Vec::new(),
        }
    }

    #[instrument(skip(self))]
    pub fn create_child_span(&self, parent_span: &Span, operation_name: &str, attributes: HashMap<String, serde_json::Value>) -> Span {
        let tracer = self.get_tracer();
        
        let mut builder = tracer.span_builder(operation_name)
            .with_parent_context(&Context::current())
            .with_attributes(
                attributes
                    .iter()
                    .map(|(k, v)| KeyValue::new(k.clone(), v.to_string()))
                    .collect()
            );

        let span = builder.start(&tracer);

        Span {
            inner: span,
            operation_name: operation_name.to_string(),
            attributes,
            events: Vec::new(),
        }
    }

    #[instrument(skip(self))]
    pub fn add_baggage_item(&self, key: String, value: String) {
        let mut baggage = self.baggage.write();
        baggage.insert(key, value);
    }

    #[instrument(skip(self))]
    pub fn get_baggage_item(&self, key: &str) -> Option<String> {
        let baggage = self.baggage.read();
        baggage.get(key).cloned()
    }

    #[instrument(skip(self))]
    pub fn trace_database_query(&self, query: &str, duration_ms: u64, rows_affected: Option<u32>) -> Span {
        let mut attributes = HashMap::new();
        attributes.insert("db.statement".to_string(), query.into());
        attributes.insert("db.duration_ms".to_string(), duration_ms.into());
        
        if let Some(rows) = rows_affected {
            attributes.insert("db.rows_affected".to_string(), rows.into());
        }

        let mut span = self.create_span("database.query", attributes);
        span.add_event("Database query executed", HashMap::new());
        span
    }

    #[instrument(skip(self))]
    pub fn trace_model_inference(&self, model_name: &str, duration_ms: u64, tokens_generated: u64, cache_hit: bool) -> Span {
        let mut attributes = HashMap::new();
        attributes.insert("ai.model.name".to_string(), model_name.into());
        attributes.insert("ai.inference.duration_ms".to_string(), duration_ms.into());
        attributes.insert("ai.tokens.generated".to_string(), tokens_generated.into());
        attributes.insert("ai.cache.hit".to_string(), cache_hit.into());

        let mut span = self.create_span("ai.inference", attributes);
        span.add_event("Model inference completed", HashMap::new());
        span
    }

    #[instrument(skip(self))]
    pub fn trace_http_request(&self, method: &str, url: &str, status_code: u16, duration_ms: u64) -> Span {
        let mut attributes = HashMap::new();
        attributes.insert("http.method".to_string(), method.into());
        attributes.insert("http.url".to_string(), url.into());
        attributes.insert("http.status_code".to_string(), status_code.into());
        attributes.insert("http.duration_ms".to_string(), duration_ms.into());

        let mut span = self.create_span("http.request", attributes);
        span.add_event("HTTP request completed", HashMap::new());
        span
    }

    #[instrument(skip(self))]
    pub fn trace_authentication(&self, method: &str, success: bool, user_id: Option<&str>) -> Span {
        let mut attributes = HashMap::new();
        attributes.insert("auth.method".to_string(), method.into());
        attributes.insert("auth.success".to_string(), success.into());
        
        if let Some(user) = user_id {
            attributes.insert("user.id".to_string(), user.into());
        }

        let mut span = self.create_span("auth.attempt", attributes);
        span.add_event("Authentication attempt", HashMap::new());
        span
    }

    #[instrument(skip(self))]
    pub fn trace_external_api_call(&self, service: &str, endpoint: &str, duration_ms: u64, status_code: u16) -> Span {
        let mut attributes = HashMap::new();
        attributes.insert("external.service".to_string(), service.into());
        attributes.insert("external.endpoint".to_string(), endpoint.into());
        attributes.insert("external.duration_ms".to_string(), duration_ms.into());
        attributes.insert("external.status_code".to_string(), status_code.into());

        let mut span = self.create_span("external.api_call", attributes);
        span.add_event("External API call completed", HashMap::new());
        span
    }

    pub async fn export_traces_to_otlp(&self) -> Result<(), TracingError> {
        // This would export traces to an OTLP-compatible backend like Jaeger, Zipkin, or Grafana Tempo
        // Implementation depends on your tracing backend choice
        
        Ok(())
    }

    pub fn get_trace_summary(&self) -> Result<TraceSummary, TracingError> {
        // This would query the trace storage system to get summary statistics
        // For now, return mock data
        
        Ok(TraceSummary {
            total_traces: 1000,
            error_traces: 50,
            average_duration_ms: 250.0,
            slowest_operations: vec![
                "model.inference".to_string(),
                "database.query".to_string(),
                "auth.attempt".to_string(),
            ],
            trace_rate_per_minute: 10.0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Span {
    inner: opentelemetry::trace::Span,
    operation_name: String,
    attributes: HashMap<String, serde_json::Value>,
    events: Vec<TraceEvent>,
}

impl Span {
    #[instrument(skip(self))]
    pub fn add_event(&mut self, event_name: &str, attributes: HashMap<String, serde_json::Value>) {
        let event = opentelemetry::trace::Event::new(
            event_name,
            opentelemetry::trace::Severity::Info,
            SystemTime::now(),
            attributes
                .iter()
                .map(|(k, v)| KeyValue::new(k.clone(), v.to_string()))
                .collect(),
        );

        self.inner.add_event(event);
        
        self.events.push(TraceEvent {
            timestamp: chrono::Utc::now(),
            name: event_name.to_string(),
            attributes,
        });
    }

    #[instrument(skip(self))]
    pub fn set_attribute(&mut self, key: String, value: serde_json::Value) {
        self.attributes.insert(key.clone(), value.clone());
        self.inner.set_attribute(KeyValue::new(key, value.to_string()));
    }

    #[instrument(skip(self))]
    pub fn set_status(&mut self, status: opentelemetry::trace::Status) {
        self.inner.set_status(status);
    }

    #[instrument(skip(self))]
    pub fn end(self) {
        self.inner.end();
    }
}

// Helper function to create trace context from request
pub fn extract_trace_context_from_headers(headers: &axum::http::HeaderMap) -> (String, Option<String>) {
    let trace_id = headers
        .get("x-trace-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_else(|| {
            Uuid::new_v4().to_string().as_str()
        })
        .to_string();

    let parent_span_id = headers
        .get("x-span-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    (trace_id, parent_span_id)
}

pub fn create_trace_headers(trace_id: &str, span_id: &str) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("x-trace-id".to_string(), trace_id.to_string());
    headers.insert("x-span-id".to_string(), span_id.to_string());
    headers
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub total_traces: u64,
    pub error_traces: u64,
    pub average_duration_ms: f64,
    pub slowest_operations: Vec<String>,
    pub trace_rate_per_minute: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceQuery {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub operation_name: Option<String>,
    pub user_id: Option<String>,
    pub error_only: bool,
    pub min_duration_ms: Option<u64>,
    pub max_results: usize,
}

pub struct TraceQueryExecutor {
    tracing: Arc<DistributedTracing>,
}

impl TraceQueryExecutor {
    pub fn new(tracing: Arc<DistributedTracing>) -> Self {
        Self { tracing }
    }

    pub async fn execute_query(&self, query: TraceQuery) -> Result<Vec<TraceData>, TracingError> {
        // This would query the actual trace storage system
        // For now, return empty vector
        
        let mut results = Vec::new();
        
        // Mock trace data
        for i in 0..query.max_results.min(10) {
            let trace = TraceData {
                trace_id: format!("trace-{}", i),
                span_id: format!("span-{}", i),
                parent_span_id: if i > 0 { Some(format!("span-{}", i-1)) } else { None },
                operation_name: query.operation_name.as_ref().unwrap_or(&"unknown".to_string()).clone(),
                start_time: query.start_time,
                end_time: query.end_time,
                attributes: HashMap::new(),
                events: Vec::new(),
                status: "ok".to_string(),
                error_message: None,
            };
            results.push(trace);
        }
        
        Ok(results)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TracingError {
    #[error("Failed to initialize tracing: {0}")]
    InitializationError(String),
    
    #[error("Failed to export traces: {0}")]
    ExportError(String),
    
    #[error("Failed to query traces: {0}")]
    QueryError(String),
    
    #[error("Invalid trace context: {0}")]
    InvalidContext(String),
}

// Configuration for distributed tracing
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub otlp_endpoint: Option<String>,
    pub sampling_rate: f64,
    pub max_spans_per_trace: usize,
    pub span_timeout_ms: u64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "rust_ollama".to_string(),
            service_version: "0.2.0".to_string(),
            environment: "production".to_string(),
            otlp_endpoint: None,
            sampling_rate: 1.0,
            max_spans_per_trace: 100,
            span_timeout_ms: 30000,
        }
    }
}

pub async fn initialize_tracing(config: &TracingConfig) -> Result<DistributedTracing, TracingError> {
    let tracing = DistributedTracing::new()?;
    
    // Configure the tracer with the service information
    global::tracer_provider().add_span_processor(
        opentelemetry_sdk::trace::SimpleSpanProcessor::new(
            opentelemetry_stdout::SpanExporter::default()
        )
    );

    // Configure OTLP exporter if endpoint is provided
    if let Some(endpoint) = &config.otlp_endpoint {
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(endpoint);
        
        let tracer_provider = SdkTracerProvider::builder()
            .with_batch_exporter(opentelemetry_otlp::new_pipeline().tracing().trace_config(
                Config::default()
                    .with_service_name(&config.service_name)
                    .with_service_version(&config.service_version)
            ).with_exporter(otlp_exporter).install_batch(opentelemetry_sdk::runtime::Tokio))
            .expect("Failed to create tracer provider");
        
        global::set_tracer_provider(tracer_provider);
    }

    tracing::info!("Distributed tracing initialized with service: {}", config.service_name);
    Ok(tracing)
}

// Convenience macro for creating traced spans
#[macro_export]
macro_rules! with_trace {
    ($tracing:expr, $operation:expr, $attributes:expr, $code:block) => {{
        let span = $tracing.create_span($operation, $attributes);
        let span_guard = span;
        let result = async {
            let result = $code;
            result
        }.instrument(span_guard.inner.span());
        result.await
    }};
}

// Health check for tracing system
pub async fn tracing_health_check() -> Result<(), TracingError> {
    // Test if tracing is working by creating a test span
    let tracing = DistributedTracing::new()?;
    let mut test_span = tracing.create_span("health_check", HashMap::new());
    test_span.add_event("Tracing health check", HashMap::new());
    test_span.end();
    
    Ok(())
}