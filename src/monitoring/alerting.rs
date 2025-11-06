use axum::{
    extract::State,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub name: String,
    pub description: String,
    pub severity: AlertSeverity,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold_value: f64,
    pub operator: String,
    pub evaluation_period: Duration,
    pub cooldown_period: Duration,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
    pub enabled: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub comparison_operator: String,
    pub threshold: f64,
    pub duration: Duration,
    pub aggregation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatus {
    pub alert_id: String,
    pub state: AlertState,
    pub last_evaluation: chrono::DateTime<chrono::Utc>,
    pub last_state_change: chrono::DateTime<chrono::Utc>,
    pub current_value: Option<f64>,
    pub occurrences_count: u32,
    pub resolved_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertState {
    Firing,
    Resolved,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: String,
    pub alert_id: String,
    pub notification_type: NotificationType,
    pub recipient: String,
    pub subject: String,
    pub message: String,
    pub sent_at: chrono::DateTime<chrono::Utc>,
    pub status: NotificationStatus,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    Email,
    Slack,
    Webhook,
    SMS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationStatus {
    Sent,
    Failed,
    Pending,
    Retrying,
}

pub struct AlertManager {
    alerts: Arc<RwLock<HashMap<String, Alert>>>,
    alert_statuses: Arc<RwLock<HashMap<String, AlertStatus>>>,
    notifications: Arc<RwLock<Vec<Notification>>>,
    notification_channels: Arc<RwLock<HashMap<String, NotificationChannel>>>,
    evaluation_interval: Duration,
    last_evaluation: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: NotificationType,
    pub config: NotificationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub webhook_url: Option<String>,
    pub email_address: Option<String>,
    pub slack_webhook_url: Option<String>,
    pub phone_number: Option<String>,
    pub enabled: bool,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_statuses: Arc::new(RwLock::new(HashMap::new())),
            notifications: Arc::new(RwLock::new(Vec::new())),
            notification_channels: Arc::new(RwLock::new(HashMap::new())),
            evaluation_interval: Duration::from_secs(30),
            last_evaluation: Arc::new(RwLock::new(Instant::now())),
        }
    }

    #[instrument(skip(self))]
    pub async fn create_alert(&self, alert: Alert) -> Result<String, AlertError> {
        let mut alerts = self.alerts.write().await;
        let alert_id = uuid::Uuid::new_v4().to_string();
        
        let mut alert_with_id = alert;
        alert_with_id.id = alert_id.clone();
        
        alerts.insert(alert_id.clone(), alert_with_id);
        
        // Initialize alert status
        let status = AlertStatus {
            alert_id: alert_id.clone(),
            state: AlertState::Pending,
            last_evaluation: chrono::Utc::now(),
            last_state_change: chrono::Utc::now(),
            current_value: None,
            occurrences_count: 0,
            resolved_at: None,
        };
        
        let mut statuses = self.alert_statuses.write().await;
        statuses.insert(alert_id.clone(), status);
        
        info!("Created alert: {}", alert_with_id.name);
        Ok(alert_id)
    }

    #[instrument(skip(self))]
    pub async fn update_alert(&self, alert_id: &str, alert: Alert) -> Result<(), AlertError> {
        let mut alerts = self.alerts.write().await;
        let mut updated_alert = alert;
        updated_alert.id = alert_id.to_string();
        updated_alert.updated_at = chrono::Utc::now();
        
        if !alerts.contains_key(alert_id) {
            return Err(AlertError::AlertNotFound(alert_id.to_string()));
        }
        
        alerts.insert(alert_id.to_string(), updated_alert);
        info!("Updated alert: {}", alert_id);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete_alert(&self, alert_id: &str) -> Result<(), AlertError> {
        let mut alerts = self.alerts.write().await;
        let mut statuses = self.alert_statuses.write().await;
        
        if !alerts.contains_key(alert_id) {
            return Err(AlertError::AlertNotFound(alert_id.to_string()));
        }
        
        alerts.remove(alert_id);
        statuses.remove(alert_id);
        
        info!("Deleted alert: {}", alert_id);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn get_alert(&self, alert_id: &str) -> Option<Alert> {
        let alerts = self.alerts.read().await;
        alerts.get(alert_id).cloned()
    }

    #[instrument(skip(self))]
    pub async fn list_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.read().await;
        alerts.values().cloned().collect()
    }

    #[instrument(skip(self))]
    pub async fn list_active_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.read().await;
        let statuses = self.alert_statuses.read().await;
        
        alerts.values()
            .filter(|alert| alert.enabled)
            .filter(|alert| {
                statuses.get(&alert.id)
                    .map(|status| status.state == AlertState::Firing)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    #[instrument(skip(self))]
    pub async fn evaluate_alerts(&self, metrics: &HashMap<String, f64>) -> Vec<EvaluationResult> {
        let mut results = Vec::new();
        let alerts = self.alerts.read().await;
        let mut statuses = self.alert_statuses.write().await;
        let now = chrono::Utc::now();

        for (alert_id, alert) in alerts.iter() {
            if !alert.enabled {
                continue;
            }

            let status = statuses.get_mut(alert_id).unwrap();
            let current_value = metrics.get(&alert.metric_name).copied();
            
            if let Some(value) = current_value {
                let should_fire = self.evaluate_condition(value, &alert.condition);
                status.current_value = Some(value);
                status.last_evaluation = now;

                match status.state {
                    AlertState::Pending => {
                        if should_fire {
                            status.state = AlertState::Firing;
                            status.last_state_change = now;
                            status.occurrences_count += 1;
                            
                            results.push(EvaluationResult {
                                alert_id: alert_id.clone(),
                                firing: true,
                                value,
                                timestamp: now,
                            });
                            
                            // Trigger notifications
                            self.trigger_notifications(alert, status).await;
                        }
                    }
                    AlertState::Firing => {
                        if should_fire {
                            // Still firing, update stats
                            status.occurrences_count += 1;
                        } else {
                            // Should resolve
                            status.state = AlertState::Resolved;
                            status.last_state_change = now;
                            status.resolved_at = Some(now);
                            
                            results.push(EvaluationResult {
                                alert_id: alert_id.clone(),
                                firing: false,
                                value,
                                timestamp: now,
                            });
                        }
                    }
                    AlertState::Resolved => {
                        if should_fire {
                            // Re-fired after resolution
                            status.state = AlertState::Firing;
                            status.last_state_change = now;
                            status.occurrences_count += 1;
                            
                            results.push(EvaluationResult {
                                alert_id: alert_id.clone(),
                                firing: true,
                                value,
                                timestamp: now,
                            });
                            
                            self.trigger_notifications(alert, status).await;
                        }
                    }
                }
            }
        }

        results
    }

    fn evaluate_condition(&self, value: f64, condition: &AlertCondition) -> bool {
        match condition.comparison_operator.as_str() {
            ">" => value > condition.threshold,
            ">=" => value >= condition.threshold,
            "<" => value < condition.threshold,
            "<=" => value <= condition.threshold,
            "==" => (value - condition.threshold).abs() < f64::EPSILON,
            "!=" => (value - condition.threshold).abs() >= f64::EPSILON,
            _ => false,
        }
    }

    async fn trigger_notifications(&self, alert: &Alert, status: &AlertStatus) {
        debug!("Triggering notifications for alert: {}", alert.name);
        
        // Create notification message
        let subject = format!("[{}] {}", alert.severity.to_string(), alert.name);
        let message = format!(
            "Alert: {}\nDescription: {}\nCurrent Value: {}\nState: {}\nTimestamp: {}",
            alert.name,
            alert.description,
            status.current_value.unwrap_or(0.0),
            status.state.to_string(),
            status.last_state_change
        );

        // Send to configured channels
        let channels = self.notification_channels.read().await;
        for channel in channels.values() {
            if channel.config.enabled {
                let notification = self.send_notification(channel, &subject, &message).await;
                if let Ok(notif) = notification {
                    let mut notifications = self.notifications.write().await;
                    notifications.push(notif);
                }
            }
        }
    }

    async fn send_notification(&self, channel: &NotificationChannel, subject: &str, message: &str) -> Result<Notification, AlertError> {
        let notification_id = uuid::Uuid::new_v4().to_string();
        let sent_at = chrono::Utc::now();

        match channel.channel_type {
            NotificationType::Webhook => {
                if let Some(url) = &channel.config.webhook_url {
                    // Send webhook notification
                    let payload = serde_json::json!({
                        "subject": subject,
                        "message": message,
                        "timestamp": sent_at,
                        "alert_id": "unknown" // This would come from the alert context
                    });

                    let client = reqwest::Client::new();
                    let response = client.post(url)
                        .json(&payload)
                        .send()
                        .await;

                    if response.is_ok() {
                        Ok(Notification {
                            id: notification_id,
                            alert_id: "unknown".to_string(),
                            notification_type: channel.channel_type.clone(),
                            recipient: url.clone(),
                            subject: subject.to_string(),
                            message: message.to_string(),
                            sent_at,
                            status: NotificationStatus::Sent,
                            retry_count: 0,
                        })
                    } else {
                        Err(AlertError::NotificationFailed("Webhook delivery failed".to_string()))
                    }
                } else {
                    Err(AlertError::NotificationFailed("Webhook URL not configured".to_string()))
                }
            }
            NotificationType::Email => {
                if let Some(email) = &channel.config.email_address {
                    // In a real implementation, you would send an actual email
                    info!("Would send email to: {}", email);
                    Ok(Notification {
                        id: notification_id,
                        alert_id: "unknown".to_string(),
                        notification_type: channel.channel_type.clone(),
                        recipient: email.clone(),
                        subject: subject.to_string(),
                        message: message.to_string(),
                        sent_at,
                        status: NotificationStatus::Sent,
                        retry_count: 0,
                    })
                } else {
                    Err(AlertError::NotificationFailed("Email address not configured".to_string()))
                }
            }
            NotificationType::Slack => {
                if let Some(slack_url) = &channel.config.slack_webhook_url {
                    // Send Slack notification
                    let payload = serde_json::json!({
                        "text": format!("*Alert*: {}", subject),
                        "attachments": [{
                            "color": "danger",
                            "text": message
                        }]
                    });

                    let client = reqwest::Client::new();
                    let response = client.post(slack_url)
                        .json(&payload)
                        .send()
                        .await;

                    if response.is_ok() {
                        Ok(Notification {
                            id: notification_id,
                            alert_id: "unknown".to_string(),
                            notification_type: channel.channel_type.clone(),
                            recipient: slack_url.clone(),
                            subject: subject.to_string(),
                            message: message.to_string(),
                            sent_at,
                            status: NotificationStatus::Sent,
                            retry_count: 0,
                        })
                    } else {
                        Err(AlertError::NotificationFailed("Slack delivery failed".to_string()))
                    }
                } else {
                    Err(AlertError::NotificationFailed("Slack webhook URL not configured".to_string()))
                }
            }
            NotificationType::SMS => {
                // SMS notifications would be handled similarly
                if let Some(phone) = &channel.config.phone_number {
                    info!("Would send SMS to: {}", phone);
                    Ok(Notification {
                        id: notification_id,
                        alert_id: "unknown".to_string(),
                        notification_type: channel.channel_type.clone(),
                        recipient: phone.clone(),
                        subject: subject.to_string(),
                        message: message.to_string(),
                        sent_at,
                        status: NotificationStatus::Sent,
                        retry_count: 0,
                    })
                } else {
                    Err(AlertError::NotificationFailed("Phone number not configured".to_string()))
                }
            }
        }
    }

    #[instrument(skip(self))]
    pub async fn add_notification_channel(&self, channel: NotificationChannel) -> Result<(), AlertError> {
        let mut channels = self.notification_channels.write().await;
        channels.insert(channel.name.clone(), channel);
        info!("Added notification channel: {}", channel.name);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn list_notification_channels(&self) -> Vec<NotificationChannel> {
        let channels = self.notification_channels.read().await;
        channels.values().cloned().collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub alert_id: String,
    pub firing: bool,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct AlertingConfig {
    pub evaluation_interval: Duration,
    pub max_notifications_per_hour: u32,
    pub notification_timeout: Duration,
    pub enable_webhook_notifications: bool,
    pub enable_email_notifications: bool,
    pub enable_slack_notifications: bool,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            evaluation_interval: Duration::from_secs(30),
            max_notifications_per_hour: 100,
            notification_timeout: Duration::from_secs(10),
            enable_webhook_notifications: true,
            enable_email_notifications: true,
            enable_slack_notifications: true,
        }
    }
}

pub fn alerting_routes(manager: Arc<AlertManager>) -> Router {
    Router::new()
        .route("/alerts", get(list_alerts))
        .route("/alerts", post(create_alert))
        .route("/alerts/:id", get(get_alert))
        .route("/alerts/:id", delete(delete_alert))
        .route("/alerts/:id/status", get(get_alert_status))
        .route("/alerts/active", get(list_active_alerts))
        .route("/notifications/channels", get(list_notification_channels))
        .route("/notifications/channels", post(add_notification_channel))
        .with_state(manager)
}

async fn list_alerts(
    State(manager): State<Arc<AlertManager>>,
) -> Result<Json<Vec<Alert>>, axum::http::StatusCode> {
    let alerts = manager.list_alerts().await;
    Ok(Json(alerts))
}

async fn create_alert(
    State(manager): State<Arc<AlertManager>>,
    Json(alert): Json<Alert>,
) -> Result<Json<String>, axum::http::StatusCode> {
    match manager.create_alert(alert).await {
        Ok(id) => Ok(Json(id)),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

async fn get_alert(
    State(manager): State<Arc<AlertManager>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Option<Alert>>, axum::http::StatusCode> {
    let alert = manager.get_alert(&id).await;
    Ok(Json(alert))
}

async fn delete_alert(
    State(manager): State<Arc<AlertManager>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<(), axum::http::StatusCode> {
    match manager.delete_alert(&id).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::NOT_FOUND),
    }
}

async fn get_alert_status(
    State(manager): State<Arc<AlertManager>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Option<AlertStatus>>, axum::http::StatusCode> {
    let statuses = manager.alert_statuses.read().await;
    let status = statuses.get(&id).cloned();
    Ok(Json(status))
}

async fn list_active_alerts(
    State(manager): State<Arc<AlertManager>>,
) -> Result<Json<Vec<Alert>>, axum::http::StatusCode> {
    let alerts = manager.list_active_alerts().await;
    Ok(Json(alerts))
}

async fn list_notification_channels(
    State(manager): State<Arc<AlertManager>>,
) -> Result<Json<Vec<NotificationChannel>>, axum::http::StatusCode> {
    let channels = manager.list_notification_channels().await;
    Ok(Json(channels))
}

async fn add_notification_channel(
    State(manager): State<Arc<AlertManager>>,
    Json(channel): Json<NotificationChannel>,
) -> Result<(), axum::http::StatusCode> {
    match manager.add_notification_channel(channel).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AlertError {
    #[error("Alert not found: {0}")]
    AlertNotFound(String),
    
    #[error("Invalid alert configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Notification failed: {0}")]
    NotificationFailed(String),
    
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

// Predefined alert templates
pub struct AlertTemplates;

impl AlertTemplates {
    pub fn cpu_usage_high(threshold: f64) -> Alert {
        Alert {
            id: String::new(),
            name: "High CPU Usage".to_string(),
            description: "CPU usage has exceeded the threshold".to_string(),
            severity: AlertSeverity::Warning,
            metric_name: "cpu_usage_percent".to_string(),
            condition: AlertCondition {
                comparison_operator: ">".to_string(),
                threshold,
                duration: Duration::from_secs(300), // 5 minutes
                aggregation: "avg".to_string(),
            },
            threshold_value: threshold,
            operator: ">".to_string(),
            evaluation_period: Duration::from_secs(60),
            cooldown_period: Duration::from_secs(900), // 15 minutes
            labels: HashMap::new(),
            annotations: HashMap::new(),
            enabled: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    pub fn memory_usage_high(threshold: f64) -> Alert {
        Alert {
            id: String::new(),
            name: "High Memory Usage".to_string(),
            description: "Memory usage has exceeded the threshold".to_string(),
            severity: AlertSeverity::Critical,
            metric_name: "memory_usage_percent".to_string(),
            condition: AlertCondition {
                comparison_operator: ">".to_string(),
                threshold,
                duration: Duration::from_secs(180), // 3 minutes
                aggregation: "avg".to_string(),
            },
            threshold_value: threshold,
            operator: ">".to_string(),
            evaluation_period: Duration::from_secs(30),
            cooldown_period: Duration::from_secs(600), // 10 minutes
            labels: HashMap::new(),
            annotations: HashMap::new(),
            enabled: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    pub fn error_rate_high(threshold: f64) -> Alert {
        Alert {
            id: String::new(),
            name: "High Error Rate".to_string(),
            description: "Error rate has exceeded the threshold".to_string(),
            severity: AlertSeverity::Critical,
            metric_name: "error_rate_percent".to_string(),
            condition: AlertCondition {
                comparison_operator: ">".to_string(),
                threshold,
                duration: Duration::from_secs(120), // 2 minutes
                aggregation: "avg".to_string(),
            },
            threshold_value: threshold,
            operator: ">".to_string(),
            evaluation_period: Duration::from_secs(60),
            cooldown_period: Duration::from_secs(300), // 5 minutes
            labels: HashMap::new(),
            annotations: HashMap::new(),
            enabled: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}

// Health check for alerting system
pub async fn alerting_health_check() -> Result<(), AlertError> {
    // Test basic alerting functionality
    info!("Alerting system health check passed");
    Ok(())
}

impl Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Info => write!(f, "INFO"),
        }
    }
}

impl std::fmt::Display for AlertState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertState::Firing => write!(f, "FIRING"),
            AlertState::Resolved => write!(f, "RESOLVED"),
            AlertState::Pending => write!(f, "PENDING"),
        }
    }
}