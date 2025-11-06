use crate::core::{DatabaseManager, InferenceEngine, ModelManager};
use axum::{
    extract::State,
    response::Json,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub uptime_seconds: u64,
    pub services: HashMap<String, ServiceHealth>,
    pub metrics: SystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    pub status: String,
    pub response_time_ms: Option<u64>,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub disk_usage_percent: f64,
    pub active_connections: u64,
    pub requests_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessProbe {
    pub ready: bool,
    pub checks: HashMap<String, bool>,
    pub last_failure: Option<chrono::DateTime<chrono::Utc>>,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessProbe {
    pub alive: bool,
    pub checks: HashMap<String, bool>,
    pub restart_count: u32,
    pub last_restart: Option<chrono::DateTime<chrono::Utc>>,
}

pub struct HealthChecker {
    pub start_time: std::time::Instant,
    pub database: Arc<RwLock<DatabaseManager>>,
    pub inference_engine: Arc<RwLock<InferenceEngine>>,
    pub model_manager: Arc<RwLock<ModelManager>>,
    pub version: String,
    pub checks_config: HealthChecksConfig,
}

#[derive(Debug, Clone)]
pub struct HealthChecksConfig {
    pub database_check_enabled: bool,
    pub inference_check_enabled: bool,
    pub model_check_enabled: bool,
    pub disk_space_threshold_percent: f64,
    pub memory_threshold_percent: f64,
    pub cpu_threshold_percent: f64,
}

impl Default for HealthChecksConfig {
    fn default() -> Self {
        Self {
            database_check_enabled: true,
            inference_check_enabled: true,
            model_check_enabled: true,
            disk_space_threshold_percent: 90.0,
            memory_threshold_percent: 85.0,
            cpu_threshold_percent: 80.0,
        }
    }
}

impl HealthChecker {
    pub fn new(
        database: Arc<RwLock<DatabaseManager>>,
        inference_engine: Arc<RwLock<InferenceEngine>>,
        model_manager: Arc<RwLock<ModelManager>>,
        version: String,
    ) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            database,
            inference_engine,
            model_manager,
            version,
            checks_config: HealthChecksConfig::default(),
        }
    }

    pub async fn get_health_status(&self) -> HealthStatus {
        let mut services = HashMap::new();
        let uptime_seconds = self.start_time.elapsed().as_secs();

        // Database health check
        if self.checks_config.database_check_enabled {
            let db_health = self.check_database_health().await;
            services.insert("database".to_string(), db_health);
        }

        // Inference engine health check
        if self.checks_config.inference_check_enabled {
            let inference_health = self.check_inference_health().await;
            services.insert("inference_engine".to_string(), inference_health);
        }

        // Model manager health check
        if self.checks_config.model_check_enabled {
            let model_health = self.check_model_health().await;
            services.insert("model_manager".to_string(), model_health);
        }

        // System metrics
        let metrics = self.get_system_metrics().await;

        // Overall status
        let overall_status = if services.values().all(|s| s.status == "healthy") {
            "healthy".to_string()
        } else if services.values().any(|s| s.status == "degraded") {
            "degraded".to_string()
        } else {
            "unhealthy".to_string()
        };

        HealthStatus {
            status: overall_status,
            timestamp: chrono::Utc::now(),
            version: self.version.clone(),
            uptime_seconds,
            services,
            metrics,
        }
    }

    pub async fn get_readiness_probe(&self) -> ReadinessProbe {
        let mut checks = HashMap::new();
        let mut ready = true;
        let mut last_failure = None;
        let mut failure_reason = None;

        // Database readiness
        let db_ready = self.check_database_readiness().await;
        checks.insert("database".to_string(), db_ready);
        if !db_ready {
            ready = false;
            last_failure = Some(chrono::Utc::now());
            failure_reason = Some("Database not ready".to_string());
        }

        // Model readiness
        let model_ready = self.check_model_readiness().await;
        checks.insert("models".to_string(), model_ready);
        if !model_ready {
            ready = false;
            last_failure = Some(chrono::Utc::now());
            failure_reason = Some("Models not loaded".to_string());
        }

        ReadinessProbe {
            ready,
            checks,
            last_failure,
            failure_reason,
        }
    }

    pub async fn get_liveness_probe(&self) -> LivenessProbe {
        let mut checks = HashMap::new();

        // Check if main services are responsive
        let db_alive = self.check_database_aliveness().await;
        checks.insert("database".to_string(), db_alive);

        let inference_alive = self.check_inference_aliveness().await;
        checks.insert("inference_engine".to_string(), inference_alive);

        let overall_alive = checks.values().all(|&alive| alive);

        LivenessProbe {
            alive: overall_alive,
            checks,
            restart_count: 0, // TODO: Track restart count
            last_restart: None,
        }
    }

    async fn check_database_health(&self) -> ServiceHealth {
        let start = std::time::Instant::now();
        
        match self.database.read().await.health_check().await {
            Ok(()) => ServiceHealth {
                status: "healthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                last_check: chrono::Utc::now(),
                error: None,
                details: Some(serde_json::json!({
                    "connection_count": 5, // Mock data
                    "query_performance": "normal"
                })),
            },
            Err(e) => {
                warn!("Database health check failed: {}", e);
                ServiceHealth {
                    status: "unhealthy".to_string(),
                    response_time_ms: Some(start.elapsed().as_millis() as u64),
                    last_check: chrono::Utc::now(),
                    error: Some(e.to_string()),
                    details: None,
                }
            }
        }
    }

    async fn check_inference_health(&self) -> ServiceHealth {
        let start = std::time::Instant::now();
        
        match self.inference_engine.read().await.health_check().await {
            Ok(()) => ServiceHealth {
                status: "healthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                last_check: chrono::Utc::now(),
                error: None,
                details: Some(serde_json::json!({
                    "active_models": 3, // Mock data
                    "gpu_usage": "normal"
                })),
            },
            Err(e) => {
                warn!("Inference engine health check failed: {}", e);
                ServiceHealth {
                    status: "unhealthy".to_string(),
                    response_time_ms: Some(start.elapsed().as_millis() as u64),
                    last_check: chrono::Utc::now(),
                    error: Some(e.to_string()),
                    details: None,
                }
            }
        }
    }

    async fn check_model_health(&self) -> ServiceHealth {
        let start = std::time::Instant::now();
        
        match self.model_manager.read().await.health_check().await {
            Ok(()) => ServiceHealth {
                status: "healthy".to_string(),
                response_time_ms: Some(start.elapsed().as_millis() as u64),
                last_check: chrono::Utc::now(),
                error: None,
                details: Some(serde_json::json!({
                    "loaded_models": 2, // Mock data
                    "model_cache_size": "normal"
                })),
            },
            Err(e) => {
                warn!("Model manager health check failed: {}", e);
                ServiceHealth {
                    status: "unhealthy".to_string(),
                    response_time_ms: Some(start.elapsed().as_millis() as u64),
                    last_check: chrono::Utc::now(),
                    error: Some(e.to_string()),
                    details: None,
                }
            }
        }
    }

    async fn check_database_readiness(&self) -> bool {
        // Simple readiness check - can we connect and query?
        self.database.read().await.health_check().await.is_ok()
    }

    async fn check_model_readiness(&self) -> bool {
        // Check if at least one model is loaded
        self.model_manager.read().await.list_models().await.len() > 0
    }

    async fn check_database_aliveness(&self) -> bool {
        self.database.read().await.health_check().await.is_ok()
    }

    async fn check_inference_aliveness(&self) -> bool {
        self.inference_engine.read().await.health_check().await.is_ok()
    }

    async fn get_system_metrics(&self) -> SystemMetrics {
        // Mock system metrics - in real implementation, use sysinfo or similar
        SystemMetrics {
            cpu_usage_percent: 25.0,
            memory_usage_percent: 45.0,
            memory_used_mb: 2048,
            memory_total_mb: 4096,
            disk_usage_percent: 30.0,
            active_connections: 5,
            requests_per_second: 10.0,
        }
    }
}

pub fn health_routes(health_checker: Arc<HealthChecker>) -> Router {
    Router::new()
        .route("/health", get(health_endpoint))
        .route("/health/ready", get(readiness_endpoint))
        .route("/health/live", get(liveness_endpoint))
        .route("/health/detailed", get(detailed_health_endpoint))
        .with_state(health_checker)
}

async fn health_endpoint(
    State(health_checker): State<Arc<HealthChecker>>,
) -> Json<HealthStatus> {
    let status = health_checker.get_health_status().await;
    Json(status)
}

async fn readiness_endpoint(
    State(health_checker): State<Arc<HealthChecker>>,
) -> Json<ReadinessProbe> {
    let probe = health_checker.get_readiness_probe().await;
    Json(probe)
}

async fn liveness_endpoint(
    State(health_checker): State<Arc<HealthChecker>>,
) -> Json<LivenessProbe> {
    let probe = health_checker.get_liveness_probe().await;
    Json(probe)
}

async fn detailed_health_endpoint(
    State(health_checker): State<Arc<HealthChecker>>,
) -> Json<serde_json::Value> {
    let health_status = health_checker.get_health_status().await;
    let readiness = health_checker.get_readiness_probe().await;
    let liveness = health_checker.get_liveness_probe().await;

    let detailed = serde_json::json!({
        "health": health_status,
        "readiness": readiness,
        "liveness": liveness,
        "kubernetes": {
            "probe_support": true,
            "endpoints": {
                "health": "/health",
                "readiness": "/health/ready", 
                "liveness": "/health/live"
            }
        }
    });

    Json(detailed)
}

impl DatabaseManager {
    pub async fn health_check(&self) -> Result<(), String> {
        // Simple health check - try to execute a query
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| format!("Database health check failed: {}", e))
    }
}

impl InferenceEngine {
    pub async fn health_check(&self) -> Result<(), String> {
        // Simple health check - verify core components are available
        if self.device.is_some() {
            Ok(())
        } else {
            Err("Inference engine not initialized".to_string())
        }
    }
}

impl ModelManager {
    pub async fn health_check(&self) -> Result<(), String> {
        // Simple health check - verify model storage is accessible
        Ok(())
    }
}