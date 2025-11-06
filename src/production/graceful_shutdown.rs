use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::time::timeout;
use tracing::{info, warn, error};

// Core application components that need graceful shutdown
pub struct ApplicationComponents {
    pub database: Arc<RwLock<crate::core::DatabaseManager>>,
    pub inference_engine: Arc<RwLock<crate::core::InferenceEngine>>,
    pub model_manager: Arc<RwLock<crate::core::ModelManager>>,
    pub api_server: Option<Arc<Mutex<axum::serve::Serve<axum::extract::State<axum::http::Request<axum::body::Body>>, axum::http::Response<axum::body::Body>>>>>,
    pub websocket_server: Option<Arc<Mutex<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>>>,
    pub background_tasks: Vec<Arc<Mutex<dyn crate::core::BackgroundTask + Send + Sync>>>,
}

impl ApplicationComponents {
    pub fn new() -> Self {
        Self {
            database: Arc::new(RwLock::new(crate::core::DatabaseManager::new())),
            inference_engine: Arc::new(RwLock::new(crate::core::InferenceEngine::new())),
            model_manager: Arc::new(RwLock::new(crate::core::ModelManager::new())),
            api_server: None,
            websocket_server: None,
            background_tasks: Vec::new(),
        }
    }

    pub fn with_api_server(mut self, server: axum::serve::Serve<axum::extract::State<axum::http::Request<axum::body::Body>>, axum::http::Response<axum::body::Body>>) -> Self {
        self.api_server = Some(Arc::new(Mutex::new(server)));
        self
    }

    pub fn add_background_task(&mut self, task: Arc<Mutex<dyn crate::core::BackgroundTask + Send + Sync>>) {
        self.background_tasks.push(task);
    }
}

pub struct GracefulShutdownConfig {
    pub shutdown_timeout: Duration,
    pub force_shutdown_timeout: Duration,
    pub health_check_interval: Duration,
    pub graceful_shutdown_enabled: bool,
}

impl Default for GracefulShutdownConfig {
    fn default() -> Self {
        Self {
            shutdown_timeout: Duration::from_secs(30),
            force_shutdown_timeout: Duration::from_secs(15),
            health_check_interval: Duration::from_millis(100),
            graceful_shutdown_enabled: true,
        }
    }
}

pub struct GracefulShutdown {
    components: Arc<RwLock<ApplicationComponents>>,
    config: GracefulShutdownConfig,
    shutdown_signals: Arc<Mutex<Vec<mpsc::UnboundedSender<()>>>>,
    is_shutting_down: Arc<Mutex<bool>>,
    shutdown_complete: Arc<Mutex<bool>>,
}

impl GracefulShutdown {
    pub fn new(components: Arc<RwLock<ApplicationComponents>>, config: GracefulShutdownConfig) -> Self {
        Self {
            components,
            config,
            shutdown_signals: Arc::new(Mutex::new(Vec::new())),
            is_shutting_down: Arc::new(Mutex::new(false)),
            shutdown_complete: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn initiate_shutdown(&self) -> Result<(), ShutdownError> {
        let mut is_shutting_down = self.is_shutting_down.lock().await;
        if *is_shutting_down {
            warn!("Shutdown already in progress");
            return Ok(());
        }

        *is_shutting_down = true;
        info!("Initiating graceful shutdown...");

        // Notify all shutdown listeners
        {
            let signals = self.shutdown_signals.lock().await;
            for signal in signals.iter() {
                if let Err(e) = signal.send(()) {
                    warn!("Failed to send shutdown signal: {}", e);
                }
            }
        }

        // Start shutdown process
        self.execute_shutdown().await?;

        Ok(())
    }

    async fn execute_shutdown(&self) -> Result<(), ShutdownError> {
        info!("Starting graceful shutdown process with timeout: {:?}", self.config.shutdown_timeout);

        // Phase 1: Stop accepting new requests
        self.stop_new_requests().await?;

        // Phase 2: Wait for ongoing requests to complete
        self.wait_for_requests_to_complete().await?;

        // Phase 3: Shutdown services in reverse order
        self.shutdown_services().await?;

        // Phase 4: Cleanup resources
        self.cleanup_resources().await?;

        {
            let mut shutdown_complete = self.shutdown_complete.lock().await;
            *shutdown_complete = true;
        }

        info!("Graceful shutdown completed successfully");
        Ok(())
    }

    async fn stop_new_requests(&self) -> Result<(), ShutdownError> {
        info!("Stopping new requests...");

        // Close HTTP server if running
        {
            let components = self.components.read().await;
            if let Some(api_server) = &components.api_server {
                info!("Closing HTTP server...");
                let server = api_server.clone();
                drop(components); // Release the read lock
                
                // Give the server a moment to stop accepting new connections
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        Ok(())
    }

    async fn wait_for_requests_to_complete(&self) -> Result<(), ShutdownError> {
        info!("Waiting for ongoing requests to complete...");
        let start = std::time::Instant::now();

        // Monitor request completion
        while start.elapsed() < self.config.shutdown_timeout {
            let components = self.components.read().await;
            
            // Check if inference engine is still processing
            if let Ok(engine) = components.inference_engine.try_write() {
                // Check if there are ongoing inference tasks
                // This is a simplified check - in real implementation, track active requests
                info!("Waiting for inference tasks to complete...");
            }

            drop(components);

            // Check health and active connections
            let active_connections = self.get_active_connection_count().await;
            if active_connections == 0 {
                info!("All requests completed");
                return Ok(());
            }

            info!("Waiting for {} active connections to complete", active_connections);
            tokio::time::sleep(self.config.health_check_interval).await;
        }

        warn!("Timeout waiting for requests to complete, proceeding with force shutdown");
        Ok(())
    }

    async fn shutdown_services(&self) -> Result<(), ShutdownError> {
        info!("Shutting down services...");

        let components = self.components.read().await;

        // Shutdown in reverse order of startup
        info!("Stopping model manager...");
        if let Ok(mut model_manager) = components.model_manager.try_write() {
            // Perform any necessary cleanup
            drop(model_manager);
        }

        info!("Stopping inference engine...");
        if let Ok(mut inference_engine) = components.inference_engine.try_write() {
            // Perform any necessary cleanup
            drop(inference_engine);
        }

        info!("Closing database connections...");
        if let Ok(mut database) = components.database.try_write() {
            // Perform any necessary cleanup
            drop(database);
        }

        // Shutdown background tasks
        info!("Stopping background tasks...");
        for (i, task) in components.background_tasks.iter().enumerate() {
            info!("Stopping background task {}", i);
            if let Ok(mut task_guard) = task.try_lock() {
                task_guard.stop().await.map_err(|e| {
                    ShutdownError::ServiceShutdown(format!("Background task {} shutdown failed: {}", i, e))
                })?;
            }
        }

        Ok(())
    }

    async fn cleanup_resources(&self) -> Result<(), ShutdownError> {
        info!("Cleaning up resources...");

        // Force cleanup after grace period
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Force shutdown if grace period exceeded
        if self.config.shutdown_timeout.as_secs() > 0 {
            info!("Starting force shutdown after grace period");
            self.force_shutdown().await?;
        }

        Ok(())
    }

    async fn force_shutdown(&self) -> Result<(), ShutdownError> {
        warn!("Force shutdown initiated");
        
        let start = std::time::Instant::now();
        let force_timeout = self.config.force_shutdown_timeout;

        while start.elapsed() < force_timeout {
            // Attempt to force close remaining connections
            self.force_close_connections().await;
            
            // Check if shutdown is complete
            if self.is_shutdown_complete().await {
                info!("Force shutdown completed");
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        error!("Force shutdown timeout reached");
        Err(ShutdownError::ForceTimeout)
    }

    async fn get_active_connection_count(&self) -> usize {
        // Mock implementation - in real app, track actual connections
        0
    }

    async fn force_close_connections(&self) {
        // Mock implementation - in real app, forcefully close remaining connections
        info!("Force closing remaining connections");
    }

    async fn is_shutdown_complete(&self) -> bool {
        *self.shutdown_complete.lock().await
    }

    pub async fn register_shutdown_signal(&self) -> mpsc::UnboundedReceiver<()> {
        let (tx, rx) = mpsc::unbounded_channel::<()>();
        let mut signals = self.shutdown_signals.lock().await;
        signals.push(tx);
        rx
    }

    pub async fn is_shutting_down(&self) -> bool {
        *self.is_shutting_down.lock().await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ShutdownError {
    #[error("Service shutdown failed: {0}")]
    ServiceShutdown(String),
    #[error("Force shutdown timeout")]
    ForceTimeout,
    #[error("Shutdown already in progress")]
    AlreadyShuttingDown,
}

// Signal handling utilities
pub async fn setup_signal_handlers(shutdown: Arc<GracefulShutdown>) {
    // Setup SIGINT handler
    let shutdown_int = Arc::clone(&shutdown);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Received SIGINT, initiating graceful shutdown...");
        let _ = shutdown_int.initiate_shutdown().await;
    });

    // Setup SIGTERM handler
    let shutdown_term = Arc::clone(&shutdown);
    tokio::spawn(async move {
        let mut term_stream = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler");
        
        term_stream.recv().await;
        info!("Received SIGTERM, initiating graceful shutdown...");
        let _ = shutdown_term.initiate_shutdown().await;
    });
}

// Extension trait for background tasks
pub trait BackgroundTask {
    async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    fn is_running(&self) -> bool;
    fn task_name(&self) -> &'static str;
}

// Example background task implementation
pub struct ModelCacheCleanupTask {
    running: bool,
    model_manager: Arc<RwLock<crate::core::ModelManager>>,
}

impl ModelCacheCleanupTask {
    pub fn new(model_manager: Arc<RwLock<crate::core::ModelManager>>) -> Self {
        Self {
            running: false,
            model_manager,
        }
    }
}

#[async_trait::async_trait]
impl BackgroundTask for ModelCacheCleanupTask {
    async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.running = true;
        info!("Started model cache cleanup task");

        while self.running {
            // Perform cache cleanup
            let model_manager = self.model_manager.clone();
            if let Ok(mut manager) = model_manager.write().await {
                // Cleanup old cached models
                // manager.cleanup_old_models().await?;
            }

            tokio::time::sleep(Duration::from_minutes(10)).await;
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.running = false;
        info!("Stopped model cache cleanup task");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn task_name(&self) -> &'static str {
        "ModelCacheCleanup"
    }
}