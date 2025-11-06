use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: String,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub labels: HashMap<String, String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetric {
    pub name: String,
    pub category: String,
    pub value: f64,
    pub previous_value: Option<f64>,
    pub change_percent: Option<f64>,
    pub trend: String,
    pub threshold_alerts: Vec<ThresholdAlert>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAlert {
    pub metric_name: String,
    pub threshold: f64,
    pub operator: String, // ">", ">=", "<", "<="
    pub is_triggered: bool,
    pub triggered_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub name: String,
    pub metric_type: String,
    pub unit: String,
    pub description: String,
    pub labels: Vec<String>,
    pub aggregation: String, // "sum", "avg", "max", "min", "count"
    pub retention_days: u32,
    pub alerting_enabled: bool,
}

pub struct CustomMetricsManager {
    metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
    business_metrics: Arc<RwLock<HashMap<String, BusinessMetric>>>,
    metric_definitions: Arc<RwLock<HashMap<String, MetricDefinition>>>,
    threshold_alerts: Arc<RwLock<Vec<ThresholdAlert>>>,
}

impl CustomMetricsManager {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            business_metrics: Arc::new(RwLock::new(HashMap::new())),
            metric_definitions: Arc::new(RwLock::new(HashMap::new())),
            threshold_alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }

    #[instrument(skip(self))]
    pub async fn register_metric(&self, definition: MetricDefinition) -> Result<(), MetricsError> {
        let mut definitions = self.metric_definitions.write().await;
        
        if definitions.contains_key(&definition.name) {
            return Err(MetricsError::MetricAlreadyRegistered(definition.name));
        }

        definitions.insert(definition.name.clone(), definition);
        info!("Registered custom metric: {}", definition.name);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn record_metric(&self, metric: CustomMetric) -> Result<(), MetricsError> {
        let definitions = self.metric_definitions.read().await;
        
        if !definitions.contains_key(&metric.metric_type) {
            return Err(MetricsError::UndefinedMetricType(metric.metric_type));
        }

        drop(definitions);
        
        let mut metrics = self.metrics.write().await;
        metrics.insert(metric.name.clone(), metric);
        
        // Check threshold alerts
        self.check_threshold_alerts(&metric).await;
        
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn record_business_metric(&self, metric: BusinessMetric) -> Result<(), MetricsError> {
        let mut metrics = self.business_metrics.write().await;
        
        // Calculate trend and change percentage
        let (trend, change_percent) = if let Some(previous) = metric.previous_value {
            let change = ((metric.value - previous) / previous) * 100.0;
            let trend = if change > 5.0 {
                "increasing"
            } else if change < -5.0 {
                "decreasing"
            } else {
                "stable"
            }.to_string();
            (trend, Some(change))
        } else {
            ("baseline".to_string(), None)
        };

        let mut threshold_alerts = Vec::new();
        for alert in &metric.threshold_alerts {
            let is_triggered = self.evaluate_threshold(alert, metric.value);
            if is_triggered {
                threshold_alerts.push(ThresholdAlert {
                    metric_name: metric.name.clone(),
                    threshold: alert.threshold,
                    operator: alert.operator.clone(),
                    is_triggered: true,
                    triggered_at: Some(chrono::Utc::now()),
                });
            }
        }

        let enhanced_metric = BusinessMetric {
            trend,
            change_percent,
            threshold_alerts,
            ..metric
        };

        metrics.insert(enhanced_metric.name.clone(), enhanced_metric);
        info!("Recorded business metric: {}", enhanced_metric.name);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn get_metric(&self, name: &str) -> Option<CustomMetric> {
        let metrics = self.metrics.read().await;
        metrics.get(name).cloned()
    }

    #[instrument(skip(self))]
    pub async fn get_business_metric(&self, name: &str) -> Option<BusinessMetric> {
        let metrics = self.business_metrics.read().await;
        metrics.get(name).cloned()
    }

    #[instrument(skip(self))]
    pub async fn list_metrics(&self) -> Vec<CustomMetric> {
        let metrics = self.metrics.read().await;
        metrics.values().cloned().collect()
    }

    #[instrument(skip(self))]
    pub async fn list_business_metrics(&self) -> Vec<BusinessMetric> {
        let metrics = self.business_metrics.read().await;
        metrics.values().cloned().collect()
    }

    #[instrument(skip(self))]
    pub async fn get_metric_history(&self, name: &str, hours: u32) -> Result<Vec<CustomMetric>, MetricsError> {
        // This would query a time-series database in a real implementation
        // For now, return the current metric if it exists
        
        if let Some(metric) = self.get_metric(name).await {
            Ok(vec![metric])
        } else {
            Err(MetricsError::MetricNotFound(name.to_string()))
        }
    }

    #[instrument(skip(self))]
    pub async fn aggregate_metrics(&self, metric_names: &[String], aggregation: &str) -> Result<f64, MetricsError> {
        let metrics = self.metrics.read().await;
        
        let values: Vec<f64> = metric_names
            .iter()
            .filter_map(|name| metrics.get(name).map(|m| m.value))
            .collect();

        if values.is_empty() {
            return Err(MetricsError::NoMetricsFound);
        }

        let result = match aggregation {
            "sum" => values.iter().sum::<f64>(),
            "avg" => values.iter().sum::<f64>() / values.len() as f64,
            "max" => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            "min" => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            "count" => values.len() as f64,
            _ => return Err(MetricsError::InvalidAggregation(aggregation.to_string())),
        };

        Ok(result)
    }

    async fn check_threshold_alerts(&self, metric: &CustomMetric) {
        // Check if any thresholds are defined for this metric
        let definitions = self.metric_definitions.read().await;
        let alerts = self.threshold_alerts.read().await;

        if let Some(definition) = definitions.get(&metric.metric_type) {
            if definition.alerting_enabled {
                debug!("Checking thresholds for metric: {}", metric.name);
                // In a real implementation, you would check configured thresholds
                // and trigger alerts if necessary
            }
        }

        drop(alerts);
        drop(definitions);
    }

    fn evaluate_threshold(&self, alert: &ThresholdAlert, value: f64) -> bool {
        match alert.operator.as_str() {
            ">" => value > alert.threshold,
            ">=" => value >= alert.threshold,
            "<" => value < alert.threshold,
            "<=" => value <= alert.threshold,
            _ => false,
        }
    }

    // Predefined business metrics
    pub async fn initialize_business_metrics(&self) {
        let definitions = vec![
            MetricDefinition {
                name: "user_satisfaction_score".to_string(),
                metric_type: "gauge".to_string(),
                unit: "score".to_string(),
                description: "User satisfaction score based on feedback".to_string(),
                labels: vec!["region".to_string()].into(),
                aggregation: "avg".to_string(),
                retention_days: 90,
                alerting_enabled: true,
            },
            MetricDefinition {
                name: "model_accuracy".to_string(),
                metric_type: "gauge".to_string(),
                unit: "percentage".to_string(),
                description: "Model accuracy percentage".to_string(),
                labels: vec!["model_name".to_string()].into(),
                aggregation: "avg".to_string(),
                retention_days: 30,
                alerting_enabled: true,
            },
            MetricDefinition {
                name: "cost_per_inference".to_string(),
                metric_type: "gauge".to_string(),
                unit: "usd".to_string(),
                description: "Cost per inference in USD".to_string(),
                labels: vec!["model_name".to_string(), "region".to_string()].into(),
                aggregation: "avg".to_string(),
                retention_days: 60,
                alerting_enabled: true,
            },
        ];

        for definition in definitions {
            let _ = self.register_metric(definition).await;
        }
    }
}

// Business metric recording helpers
pub struct BusinessMetricsRecorder {
    manager: Arc<CustomMetricsManager>,
}

impl BusinessMetricsRecorder {
    pub fn new(manager: Arc<CustomMetricsManager>) -> Self {
        Self { manager }
    }

    #[instrument(skip(self))]
    pub async fn record_user_satisfaction(&self, score: f64, region: &str, previous_score: Option<f64>) {
        let metric = BusinessMetric {
            name: "user_satisfaction_score".to_string(),
            category: "user_experience".to_string(),
            value: score,
            previous_value: previous_score,
            change_percent: None,
            trend: "stable".to_string(),
            threshold_alerts: vec![
                ThresholdAlert {
                    metric_name: "user_satisfaction_score".to_string(),
                    threshold: 3.0,
                    operator: "<".to_string(),
                    is_triggered: false,
                    triggered_at: None,
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        let _ = self.manager.record_business_metric(metric).await;
    }

    #[instrument(skip(self))]
    pub async fn record_model_accuracy(&self, model_name: &str, accuracy: f64) {
        let metric = BusinessMetric {
            name: format!("model_accuracy_{}", model_name),
            category: "model_performance".to_string(),
            value: accuracy,
            previous_value: None,
            change_percent: None,
            trend: "stable".to_string(),
            threshold_alerts: vec![
                ThresholdAlert {
                    metric_name: format!("model_accuracy_{}", model_name),
                    threshold: 0.85,
                    operator: "<".to_string(),
                    is_triggered: false,
                    triggered_at: None,
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        let _ = self.manager.record_business_metric(metric).await;
    }

    #[instrument(skip(self))]
    pub async fn record_inference_cost(&self, model_name: &str, cost: f64, region: &str) {
        let metric = BusinessMetric {
            name: format!("cost_per_inference_{}_{}", model_name, region),
            category: "cost_optimization".to_string(),
            value: cost,
            previous_value: None,
            change_percent: None,
            trend: "stable".to_string(),
            threshold_alerts: vec![
                ThresholdAlert {
                    metric_name: format!("cost_per_inference_{}_{}", model_name, region),
                    threshold: 0.10, // $0.10 threshold
                    operator: ">".to_string(),
                    is_triggered: false,
                    triggered_at: None,
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        let _ = self.manager.record_business_metric(metric).await;
    }
}

pub fn custom_metrics_routes(manager: Arc<CustomMetricsManager>) -> Router {
    Router::new()
        .route("/custom-metrics", get(list_custom_metrics))
        .route("/custom-metrics", post(record_custom_metric))
        .route("/custom-metrics/:name", get(get_custom_metric))
        .route("/business-metrics", get(list_business_metrics))
        .route("/business-metrics", post(record_business_metric))
        .route("/business-metrics/:name", get(get_business_metric))
        .route("/metric-definitions", get(list_metric_definitions))
        .route("/metric-definitions", post(register_metric_definition))
        .with_state(manager)
}

async fn list_custom_metrics(
    State(manager): State<Arc<CustomMetricsManager>>,
) -> Result<Json<Vec<CustomMetric>>, axum::http::StatusCode> {
    let metrics = manager.list_metrics().await;
    Ok(Json(metrics))
}

async fn record_custom_metric(
    State(manager): State<Arc<CustomMetricsManager>>,
    Json(metric): Json<CustomMetric>,
) -> Result<(), axum::http::StatusCode> {
    match manager.record_metric(metric).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

async fn get_custom_metric(
    State(manager): State<Arc<CustomMetricsManager>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<Option<CustomMetric>>, axum::http::StatusCode> {
    let metric = manager.get_metric(&name).await;
    Ok(Json(metric))
}

async fn list_business_metrics(
    State(manager): State<Arc<CustomMetricsManager>>,
) -> Result<Json<Vec<BusinessMetric>>, axum::http::StatusCode> {
    let metrics = manager.list_business_metrics().await;
    Ok(Json(metrics))
}

async fn record_business_metric(
    State(manager): State<Arc<CustomMetricsManager>>,
    Json(metric): Json<BusinessMetric>,
) -> Result<(), axum::http::StatusCode> {
    match manager.record_business_metric(metric).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

async fn get_business_metric(
    State(manager): State<Arc<CustomMetricsManager>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<Option<BusinessMetric>>, axum::http::StatusCode> {
    let metric = manager.get_business_metric(&name).await;
    Ok(Json(metric))
}

async fn list_metric_definitions(
    State(manager): State<Arc<CustomMetricsManager>>,
) -> Result<Json<Vec<MetricDefinition>>, axum::http::StatusCode> {
    let definitions = manager.metric_definitions.read().await;
    let defs = definitions.values().cloned().collect();
    Ok(Json(defs))
}

async fn register_metric_definition(
    State(manager): State<Arc<CustomMetricsManager>>,
    Json(definition): Json<MetricDefinition>,
) -> Result<(), axum::http::StatusCode> {
    match manager.register_metric(definition).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("Metric already registered: {0}")]
    MetricAlreadyRegistered(String),
    
    #[error("Undefined metric type: {0}")]
    UndefinedMetricType(String),
    
    #[error("Metric not found: {0}")]
    MetricNotFound(String),
    
    #[error("No metrics found")]
    NoMetricsFound,
    
    #[error("Invalid aggregation: {0}")]
    InvalidAggregation(String),
}

// Analytics and insights
pub struct MetricsAnalytics {
    manager: Arc<CustomMetricsManager>,
}

impl MetricsAnalytics {
    pub fn new(manager: Arc<CustomMetricsManager>) -> Self {
        Self { manager }
    }

    pub async fn generate_insights(&self) -> Result<MetricsInsights, MetricsError> {
        let business_metrics = self.manager.list_business_metrics().await;
        
        let mut insights = Vec::new();
        
        for metric in business_metrics {
            if let Some(change_percent) = metric.change_percent {
                if change_percent.abs() > 10.0 {
                    insights.push(format!(
                        "{} changed by {:.1}% ({})", 
                        metric.name, 
                        change_percent, 
                        metric.trend
                    ));
                }
            }
        }

        let triggered_alerts = self.get_triggered_alerts().await;
        
        Ok(MetricsInsights {
            total_metrics: business_metrics.len(),
            trending_up: business_metrics.iter().filter(|m| m.trend == "increasing").count(),
            trending_down: business_metrics.iter().filter(|m| m.trend == "decreasing").count(),
            triggered_alerts: triggered_alerts.len(),
            insights,
            generated_at: chrono::Utc::now(),
        })
    }

    async fn get_triggered_alerts(&self) -> Vec<ThresholdAlert> {
        let metrics = self.manager.business_metrics.read().await;
        metrics.values()
            .flat_map(|m| &m.threshold_alerts)
            .filter(|a| a.is_triggered)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsInsights {
    pub total_metrics: usize,
    pub trending_up: usize,
    pub trending_down: usize,
    pub triggered_alerts: usize,
    pub insights: Vec<String>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}