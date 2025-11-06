use crate::core::{DatabaseManager, InferenceEngine, ModelManager};
use notify::{RecommendedWatcher, Watcher, RecursiveMode, Event};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFile {
    pub path: String,
    pub content: String,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigReloadEvent {
    pub file_path: String,
    pub event_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReloadableConfig {
    pub path: String,
    pub config: Arc<RwLock<serde_json::Value>>,
    pub last_modified: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
    pub checksum: Arc<RwLock<String>>,
}

pub struct ConfigHotReload {
    configs: HashMap<String, ReloadableConfig>,
    watchers: Arc<RwLock<HashMap<String, RecommendedWatcher>>>,
    event_sender: mpsc::UnboundedSender<ConfigReloadEvent>,
    config_dir: String,
    reload_enabled: bool,
}

impl ConfigHotReload {
    pub fn new(config_dir: &str) -> Self {
        let (event_sender, _) = mpsc::unbounded_channel::<ConfigReloadEvent>();
        
        Self {
            configs: HashMap::new(),
            watchers: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            config_dir: config_dir.to_string(),
            reload_enabled: true,
        }
    }

    pub async fn add_config_file(&mut self, name: &str, path: &str) -> Result<(), ConfigReloadError> {
        info!("Adding config file for hot reload: {} -> {}", name, path);

        let config_content = self.load_config_file(path).await?;
        let last_modified = self.get_file_last_modified(path).await?;
        let checksum = self.calculate_checksum(&config_content);

        let config = Arc::new(RwLock::new(
            serde_json::from_str(&config_content)
                .map_err(|e| ConfigReloadError::ConfigParse(format!("Failed to parse JSON: {}", e)))?
        ));

        let reloadable_config = ReloadableConfig {
            path: path.to_string(),
            config,
            last_modified: Arc::new(RwLock::new(last_modified)),
            checksum: Arc::new(RwLock::new(checksum)),
        };

        self.configs.insert(name.to_string(), reloadable_config);

        // Start watching this file
        if self.reload_enabled {
            self.start_file_watcher(name, path).await?;
        }

        Ok(())
    }

    pub async fn get_config(&self, name: &str) -> Option<Arc<RwLock<serde_json::Value>>> {
        self.configs.get(name).map(|c| c.config.clone())
    }

    pub async fn reload_all_configs(&mut self) -> Result<Vec<String>, ConfigReloadError> {
        info!("Manually reloading all configurations...");
        let mut reloaded_files = Vec::new();

        for (name, config) in self.configs.iter() {
            if let Err(e) = self.reload_config_file(name, config).await {
                error!("Failed to reload config '{}': {}", name, e);
                let _ = self.send_reload_event(name, "manual_reload", false, Some(e.to_string())).await;
            } else {
                reloaded_files.push(name.clone());
                info!("Successfully reloaded config: {}", name);
                let _ = self.send_reload_event(name, "manual_reload", true, None).await;
            }
        }

        Ok(reloaded_files)
    }

    pub async fn start_monitoring(&mut self) -> Result<(), ConfigReloadError> {
        if !self.reload_enabled {
            warn!("Configuration hot reload is disabled");
            return Ok(());
        }

        info!("Starting configuration file monitoring...");

        // Start watching all configured files
        for (name, config) in self.configs.iter() {
            self.start_file_watcher(name, &config.path).await?;
        }

        // Monitor config directory for new files
        let config_dir = Path::new(&self.config_dir).to_path_buf();
        let mut watcher = RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    if event.kind.is_modify() || event.kind.is_create() {
                        // This is a simplified implementation - in practice, you'd need better async handling
                    }
                }
            },
            notify::Config::default()
        )
        .map_err(|e| ConfigReloadError::WatcherError(format!("Failed to create watcher: {}", e)))?;

        watcher.watch(&config_dir, RecursiveMode::NonRecursive)
            .map_err(|e| ConfigReloadError::WatcherError(format!("Failed to watch config directory: {}", e)))?;

        let mut watchers = self.watchers.write().await;
        watchers.insert("config_dir".to_string(), watcher);

        info!("Configuration monitoring started");
        Ok(())
    }

    pub fn disable_reload(&mut self) {
        info!("Disabling configuration hot reload");
        self.reload_enabled = false;
    }

    pub fn enable_reload(&mut self) {
        info!("Enabling configuration hot reload");
        self.reload_enabled = true;
    }

    async fn load_config_file(&self, path: &str) -> Result<String, ConfigReloadError> {
        std::fs::read_to_string(path)
            .map_err(|e| ConfigReloadError::FileReadError(format!("Failed to read config file '{}': {}", path, e)))
    }

    async fn get_file_last_modified(&self, path: &str) -> Result<chrono::DateTime<chrono::Utc>, ConfigReloadError> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| ConfigReloadError::FileReadError(format!("Failed to get file metadata '{}': {}", path, e)))?;
        
        let modified = metadata.modified()
            .map_err(|e| ConfigReloadError::FileReadError(format!("Failed to get modification time '{}': {}", path, e)))?;
        
        let datetime = chrono::DateTime::from_system_time(modified, chrono::Utc);
        Ok(datetime)
    }

    fn calculate_checksum(&self, content: &str) -> String {
        format!("{:x}", md5::compute(content))
    }

    async fn reload_config_file(&self, name: &str, reloadable_config: &ReloadableConfig) -> Result<(), ConfigReloadError> {
        let path = &reloadable_config.path;
        
        debug!("Reloading config file: {}", path);

        // Check if file has actually changed
        let current_last_modified = self.get_file_last_modified(path).await?;
        let old_last_modified = *reloadable_config.last_modified.read().await;
        
        if current_last_modified <= old_last_modified {
            debug!("Config file {} hasn't changed, skipping reload", path);
            return Ok(());
        }

        // Load new content
        let new_content = self.load_config_file(path).await?;
        let new_checksum = self.calculate_checksum(&new_content);
        let old_checksum = *reloadable_config.checksum.read().await;

        if new_checksum == old_checksum {
            debug!("Config file {} content hasn't changed, skipping reload", path);
            return Ok(());
        }

        // Parse new configuration
        let new_config: serde_json::Value = serde_json::from_str(&new_content)
            .map_err(|e| ConfigReloadError::ConfigParse(format!("Failed to parse new config '{}': {}", path, e)))?;

        // Apply the new configuration
        let mut config_guard = reloadable_config.config.write().await;
        *config_guard = new_config;
        drop(config_guard);

        // Update metadata
        {
            let mut last_modified = reloadable_config.last_modified.write().await;
            *last_modified = current_last_modified;
        }
        {
            let mut checksum = reloadable_config.checksum.write().await;
            *checksum = new_checksum;
        }

        info!("Successfully reloaded configuration: {}", name);
        Ok(())
    }

    async fn start_file_watcher(&self, name: &str, path: &str) -> Result<(), ConfigReloadError> {
        let config_name = name.to_string();
        let config_path = path.to_string();
        let event_sender = self.event_sender.clone();
        let config = self.configs.get(name).unwrap().clone();

        let mut watcher = RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    if event.kind.is_modify() || event.kind.is_create() {
                        if let Some(paths) = event.paths.get(0) {
                            if paths.as_os_str().to_string_lossy() == config_path {
                                // Trigger reload in a blocking task since notify runs in sync context
                                let config_clone = config.clone();
                                let sender = event_sender.clone();
                                let name_clone = config_name.clone();
                                
                                std::thread::spawn(move || {
                                    let rt = tokio::runtime::Runtime::new().unwrap();
                                    rt.block_on(async {
                                        if let Err(e) = config_clone.config.read().await; // This will trigger a file reload check
                                        if let Err(e) = sender.send(ConfigReloadEvent {
                                            file_path: config_path.clone(),
                                            event_type: "file_changed".to_string(),
                                            timestamp: chrono::Utc::now(),
                                            success: true,
                                            error_message: None,
                                        }) {
                                            warn!("Failed to send reload event: {}", e);
                                        }
                                    });
                                });
                            }
                        }
                    }
                }
            },
            notify::Config::default()
        )
        .map_err(|e| ConfigReloadError::WatcherError(format!("Failed to create watcher for '{}': {}", path, e)))?;

        watcher.watch(Path::new(path), RecursiveMode::NonRecursive)
            .map_err(|e| ConfigReloadError::WatcherError(format!("Failed to watch file '{}': {}", path, e)))?;

        let mut watchers = self.watchers.write().await;
        watchers.insert(name.to_string(), watcher);

        info!("Started watching config file: {}", path);
        Ok(())
    }

    async fn send_reload_event(&self, name: &str, event_type: &str, success: bool, error_message: Option<String>) -> Result<(), ConfigReloadError> {
        self.event_sender.send(ConfigReloadEvent {
            file_path: name.to_string(),
            event_type: event_type.to_string(),
            timestamp: chrono::Utc::now(),
            success,
            error_message,
        }).map_err(|e| ConfigReloadError::EventSendError(e.to_string()))
    }

    pub fn get_config_status(&self) -> Vec<ConfigStatus> {
        self.configs.iter().map(|(name, config)| {
            ConfigStatus {
                name: name.clone(),
                path: config.path.clone(),
                last_modified: *config.last_modified.read().unwrap(),
                checksum: config.checksum.read().unwrap().clone(),
                is_loaded: true,
            }
        }).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStatus {
    pub name: String,
    pub path: String,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
    pub is_loaded: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigReloadError {
    #[error("File read error: {0}")]
    FileReadError(String),
    
    #[error("Configuration parse error: {0}")]
    ConfigParse(String),
    
    #[error("Watcher error: {0}")]
    WatcherError(String),
    
    #[error("Failed to send event: {0}")]
    EventSendError(String),
    
    #[error("Configuration not found: {0}")]
    ConfigNotFound(String),
}

// Configuration structures that can be reloaded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub timeout_seconds: u64,
    pub keep_alive_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_concurrent_requests: u32,
    pub default_model_timeout: u64,
    pub memory_limit_gb: u32,
    pub cache_size_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub jwt_secret: String,
    pub api_key_prefix: String,
    pub rate_limit_enabled: bool,
    pub cors_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub prometheus_port: u16,
    pub log_level: String,
    pub performance_tracking: bool,
}

pub struct ReloadableApplicationConfig {
    pub server: Arc<RwLock<ServerConfig>>,
    pub inference: Arc<RwLock<InferenceConfig>>,
    pub security: Arc<RwLock<SecurityConfig>>,
    pub monitoring: Arc<RwLock<MonitoringConfig>>,
}

impl ReloadableApplicationConfig {
    pub async fn load_initial_configs(config_dir: &str) -> Result<Self, ConfigReloadError> {
        let hot_reload = ConfigHotReload::new(config_dir);

        // Load main config file
        hot_reload.add_config_file("server", &format!("{}/server.toml", config_dir)).await?;
        hot_reload.add_config_file("inference", &format!("{}/inference.toml", config_dir)).await?;
        hot_reload.add_config_file("security", &format!("{}/security.toml", config_dir)).await?;
        hot_reload.add_config_file("monitoring", &format!("{}/monitoring.toml", config_dir)).await?;

        Ok(Self {
            server: hot_reload.get_config("server").unwrap(),
            inference: hot_reload.get_config("inference").unwrap(),
            security: hot_reload.get_config("security").unwrap(),
            monitoring: hot_reload.get_config("monitoring").unwrap(),
        })
    }
}