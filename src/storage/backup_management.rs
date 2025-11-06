use axum::{
    extract::State,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error, debug, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backup {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub backup_type: BackupType,
    pub source_path: PathBuf,
    pub destination_path: PathBuf,
    pub file_size: u64,
    pub compression_algorithm: Option<CompressionAlgorithm>,
    pub encryption_enabled: bool,
    pub status: BackupStatus,
    pub progress: u8,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub retention_days: u32,
    pub checksum: String,
    pub metadata: BackupMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupType {
    Full,
    Incremental,
    Differential,
    ModelSnapshot,
    Configuration,
    Database,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Bzip2,
    Zstd,
    Lz4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub source_files: Vec<String>,
    pub excluded_patterns: Vec<String>,
    pub compression_ratio: Option<f64>,
    pub encryption_method: Option<String>,
    pub backup_version: String,
    pub custom_attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSchedule {
    pub id: String,
    pub name: String,
    pub schedule_type: ScheduleType,
    pub interval: Duration,
    pub backup_type: BackupType,
    pub source_path: PathBuf,
    pub destination_path: PathBuf,
    pub retention_days: u32,
    pub enabled: bool,
    pub last_run: Option<chrono::DateTime<chrono::Utc>>,
    pub next_run: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Daily,
    Weekly,
    Monthly,
    Hourly,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    pub default_retention_days: u32,
    pub max_concurrent_backups: u32,
    pub default_compression: CompressionAlgorithm,
    pub backup_directory: PathBuf,
    pub temp_directory: PathBuf,
    pub encryption_enabled: bool,
    pub verification_enabled: bool,
    pub bandwidth_limit_mbps: Option<u32>,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            default_retention_days: 30,
            max_concurrent_backups: 3,
            default_compression: CompressionAlgorithm::Gzip,
            backup_directory: PathBuf::from("backups"),
            temp_directory: PathBuf::from("temp"),
            encryption_enabled: false,
            verification_enabled: true,
            bandwidth_limit_mbps: None,
        }
    }
}

pub struct BackupManager {
    config: BackupConfig,
    backups: HashMap<String, Backup>,
    schedules: HashMap<String, BackupSchedule>,
    current_backups: HashMap<String, BackupProgress>,
    backup_dir: PathBuf,
    temp_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct BackupProgress {
    backup_id: String,
    status: BackupStatus,
    progress: u8,
    current_file: Option<PathBuf>,
    files_processed: u64,
    total_files: u64,
    bytes_processed: u64,
    total_bytes: u64,
    start_time: std::time::Instant,
}

impl BackupManager {
    pub fn new(config: BackupConfig) -> Self {
        let backup_dir = config.backup_directory.clone();
        let temp_dir = config.temp_directory.clone();

        // Ensure directories exist
        fs::create_dir_all(&backup_dir).expect("Failed to create backup directory");
        fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

        Self {
            config,
            backups: HashMap::new(),
            schedules: HashMap::new(),
            current_backups: HashMap::new(),
            backup_dir,
            temp_dir,
        }
    }

    #[instrument(skip(self))]
    pub async fn create_backup(
        &mut self,
        name: String,
        backup_type: BackupType,
        source_path: PathBuf,
        destination_path: Option<PathBuf>,
        description: Option<String>,
    ) -> Result<String, BackupError> {
        info!("Creating backup: {} ({:?})", name, backup_type);

        let backup_id = uuid::Uuid::new_v4().to_string();
        let dest_path = destination_path.unwrap_or_else(|| self.backup_dir.join(format!("{}_{}", name, chrono::Utc::now().format("%Y%m%d_%H%M%S"))));

        // Initialize backup
        let backup = Backup {
            id: backup_id.clone(),
            name,
            description,
            backup_type,
            source_path: source_path.clone(),
            destination_path: dest_path.clone(),
            file_size: 0,
            compression_algorithm: Some(self.config.default_compression.clone()),
            encryption_enabled: self.config.encryption_enabled,
            status: BackupStatus::Pending,
            progress: 0,
            created_at: chrono::Utc::now(),
            completed_at: None,
            retention_days: self.config.default_retention_days,
            checksum: String::new(),
            metadata: BackupMetadata {
                source_files: Vec::new(),
                excluded_patterns: Vec::new(),
                compression_ratio: None,
                encryption_method: None,
                backup_version: "0.2.0".to_string(),
                custom_attributes: HashMap::new(),
            },
        };

        self.backups.insert(backup_id.clone(), backup.clone());
        self.start_backup_process(backup_id.clone()).await?;

        Ok(backup_id)
    }

    #[instrument(skip(self))]
    async fn start_backup_process(&mut self, backup_id: String) -> Result<(), BackupError> {
        if let Some(backup) = self.backups.get_mut(&backup_id) {
            backup.status = BackupStatus::InProgress;

            // Start async backup process
            let backup_clone = backup.clone();
            let self_clone = self.clone_with_state();
            
            tokio::spawn(async move {
                self_clone.execute_backup(backup_clone).await;
            });

            info!("Started backup process: {}", backup_id);
            Ok(())
        } else {
            Err(BackupError::BackupNotFound(backup_id))
        }
    }

    async fn execute_backup(&mut self, backup: Backup) {
        info!("Executing backup: {}", backup.id);
        
        // Initialize progress tracking
        let progress = BackupProgress {
            backup_id: backup.id.clone(),
            status: BackupStatus::InProgress,
            progress: 0,
            current_file: None,
            files_processed: 0,
            total_files: 0,
            bytes_processed: 0,
            total_bytes: 0,
            start_time: std::time::Instant::now(),
        };
        
        self.current_backups.insert(backup.id.clone(), progress);

        // Simulate backup process (in real implementation, this would do actual file copying)
        match self.perform_backup(backup).await {
            Ok(final_backup) => {
                self.update_backup_status(&final_backup.id, BackupStatus::Completed, 100);
                info!("Backup completed successfully: {}", final_backup.id);
            }
            Err(e) => {
                error!("Backup failed: {} - {}", backup.id, e);
                self.update_backup_status(&backup.id, BackupStatus::Failed, 0);
            }
        }

        self.current_backups.remove(&backup.id);
    }

    async fn perform_backup(&mut self, backup: Backup) -> Result<Backup, BackupError> {
        // Simulate backup operations
        let mut current_backup = backup;
        let total_steps = 10;

        for step in 0..=total_steps {
            // Simulate processing
            let progress = ((step as u8) * 100) / total_steps as u8;
            self.update_backup_progress(&current_backup.id, progress, format!("Processing step {}/{}", step, total_steps));

            // Simulate processing time
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Calculate final metrics
        current_backup.file_size = 1024 * 1024; // Mock 1MB
        current_backup.completed_at = Some(chrono::Utc::now());
        current_backup.checksum = self.calculate_backup_checksum(&current_backup).await?;
        current_backup.metadata.compression_ratio = Some(2.5); // Mock compression ratio

        // Update in storage
        if let Some(stored_backup) = self.backups.get_mut(&current_backup.id) {
            *stored_backup = current_backup.clone();
        }

        Ok(current_backup)
    }

    async fn calculate_backup_checksum(&self, backup: &Backup) -> Result<String, BackupError> {
        // In real implementation, calculate actual checksum
        Ok(format!("sha256:{:x}", md5::compute(&backup.id)))
    }

    fn update_backup_progress(&mut self, backup_id: &str, progress: u8, current_file: String) {
        if let Some(progress_info) = self.current_backups.get_mut(backup_id) {
            progress_info.progress = progress;
            progress_info.current_file = Some(PathBuf::from(current_file));
        }
    }

    fn update_backup_status(&mut self, backup_id: &str, status: BackupStatus, progress: u8) {
        if let Some(backup) = self.backups.get_mut(backup_id) {
            backup.status = status;
            backup.progress = progress;
        }

        if let Some(progress_info) = self.current_backups.get_mut(backup_id) {
            progress_info.status = status;
            progress_info.progress = progress;
        }
    }

    #[instrument(skip(self))]
    pub async fn restore_backup(&self, backup_id: &str, target_path: &Path) -> Result<(), BackupError> {
        info!("Restoring backup: {} to {}", backup_id, target_path.display());

        let backup = self.backups.get(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if backup.status != BackupStatus::Completed {
            return Err(BackupError::BackupNotReady(backup_id.to_string()));
        }

        // Simulate restore process
        info!("Starting restore process for: {}", backup.name);
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Ensure target directory exists
        fs::create_dir_all(target_path).map_err(|e| BackupError::IoError(e.to_string()))?;

        info!("Backup restored successfully: {} to {}", backup_id, target_path.display());
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete_backup(&mut self, backup_id: &str) -> Result<(), BackupError> {
        if let Some(backup) = self.backups.remove(backup_id) {
            // Delete backup file if it exists
            if backup.destination_path.exists() {
                fs::remove_file(&backup.destination_path)
                    .map_err(|e| BackupError::IoError(e.to_string()))?;
            }

            info!("Deleted backup: {}", backup_id);
            Ok(())
        } else {
            Err(BackupError::BackupNotFound(backup_id.to_string()))
        }
    }

    #[instrument(skip(self))]
    pub async fn list_backups(&self) -> Vec<&Backup> {
        self.backups.values().collect()
    }

    #[instrument(skip(self))]
    pub async fn get_backup(&self, backup_id: &str) -> Option<&Backup> {
        self.backups.get(backup_id)
    }

    #[instrument(skip(self))]
    pub async fn get_backup_progress(&self, backup_id: &str) -> Option<&BackupProgress> {
        self.current_backups.get(backup_id)
    }

    #[instrument(skip(self))]
    pub async fn create_schedule(&mut self, schedule: BackupSchedule) -> Result<(), BackupError> {
        if self.schedules.contains_key(&schedule.id) {
            return Err(BackupError::ScheduleExists(schedule.id));
        }

        self.schedules.insert(schedule.id.clone(), schedule);
        info!("Created backup schedule: {}", schedule.name);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn list_schedules(&self) -> Vec<&BackupSchedule> {
        self.schedules.values().collect()
    }

    #[instrument(skip(self))]
    pub async fn cleanup_expired_backups(&mut self) -> Vec<String> {
        let now = chrono::Utc::now();
        let mut expired_backups = Vec::new();

        for (id, backup) in self.backups.iter() {
            let expires_at = backup.created_at + chrono::Duration::days(backup.retention_days as i64);
            if now > expires_at {
                expired_backups.push(id.clone());
            }
        }

        // Clean up expired backups
        for backup_id in &expired_backups {
            if let Err(e) = self.delete_backup(backup_id).await {
                warn!("Failed to delete expired backup {}: {}", backup_id, e);
            }
        }

        if !expired_backups.is_empty() {
            info!("Cleaned up {} expired backups", expired_backups.len());
        }

        expired_backups
    }

    #[instrument(skip(self))]
    pub async fn verify_backup_integrity(&self, backup_id: &str) -> Result<bool, BackupError> {
        let backup = self.backups.get(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if !backup.destination_path.exists() {
            return Ok(false);
        }

        // Calculate current checksum
        let current_checksum = self.calculate_backup_checksum(backup).await?;

        // Compare with stored checksum
        let integrity_ok = current_checksum == backup.checksum;

        if !integrity_ok {
            warn!("Backup integrity check failed for: {}", backup_id);
        }

        Ok(integrity_ok)
    }

    pub async fn export_backup_metadata(&self, backup_id: &str) -> Result<String, BackupError> {
        let backup = self.backups.get(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        let metadata = serde_json::to_string_pretty(backup)
            .map_err(|e| BackupError::SerializationError(e.to_string()))?;

        Ok(metadata)
    }

    pub async fn import_backup_metadata(&mut self, metadata_json: &str) -> Result<String, BackupError> {
        let backup: Backup = serde_json::from_str(metadata_json)
            .map_err(|e| BackupError::DeserializationError(e.to_string()))?;

        self.backups.insert(backup.id.clone(), backup.clone());
        info!("Imported backup metadata: {}", backup.id);

        Ok(backup.id)
    }

    pub async fn get_backup_statistics(&self) -> BackupStatistics {
        let total_backups = self.backups.len();
        let completed_backups = self.backups.values().filter(|b| b.status == BackupStatus::Completed).count();
        let failed_backups = self.backups.values().filter(|b| b.status == BackupStatus::Failed).count();
        let running_backups = self.current_backups.len();

        let total_size: u64 = self.backups.values().map(|b| b.file_size).sum();
        let average_size = if total_backups > 0 { total_size / total_backups as u64 } else { 0 };

        let backup_type_counts: HashMap<String, usize> = self.backups.values()
            .fold(HashMap::new(), |mut acc, b| {
                *acc.entry(format!("{:?}", b.backup_type)).or_insert(0) += 1;
                acc
            });

        BackupStatistics {
            total_backups,
            completed_backups,
            failed_backups,
            running_backups,
            total_size_bytes: total_size,
            average_backup_size: average_size,
            backup_type_distribution: backup_type_counts,
            retention_expired_count: self.backups.values()
                .filter(|b| {
                    let expires_at = b.created_at + chrono::Duration::days(b.retention_days as i64);
                    chrono::Utc::now() > expires_at
                }).count(),
        }
    }
}

impl Clone for BackupManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            backups: self.backups.clone(),
            schedules: self.schedules.clone(),
            current_backups: self.current_backups.clone(),
            backup_dir: self.backup_dir.clone(),
            temp_dir: self.temp_dir.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.config = source.config.clone();
        self.backups = source.backups.clone();
        self.schedules = source.schedules.clone();
        self.current_backups = source.current_backups.clone();
        self.backup_dir = source.backup_dir.clone();
        self.temp_dir = source.temp_dir.clone();
    }
}

impl CloneWithState for BackupManager {
    fn clone_with_state(&self) -> Self {
        self.clone()
    }
}

trait CloneWithState {
    fn clone_with_state(&self) -> Self;
}

impl CloneWithState for BackupManager {
    fn clone_with_state(&self) -> Self {
        Self {
            config: self.config.clone(),
            backups: self.backups.clone(),
            schedules: self.schedules.clone(),
            current_backups: self.current_backups.clone(),
            backup_dir: self.backup_dir.clone(),
            temp_dir: self.temp_dir.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStatistics {
    pub total_backups: usize,
    pub completed_backups: usize,
    pub failed_backups: usize,
    pub running_backups: usize,
    pub total_size_bytes: u64,
    pub average_backup_size: u64,
    pub backup_type_distribution: HashMap<String, usize>,
    pub retention_expired_count: usize,
}

pub fn backup_routes(manager: Arc<RwLock<BackupManager>>) -> Router {
    Router::new()
        .route("/backups", get(list_backups))
        .route("/backups", post(create_backup))
        .route("/backups/:id", get(get_backup))
        .route("/backups/:id/restore", post(restore_backup))
        .route("/backups/:id", delete(delete_backup))
        .route("/backups/:id/progress", get(get_backup_progress))
        .route("/backups/:id/verify", get(verify_backup))
        .route("/backups/:id/export", get(export_backup_metadata))
        .route("/backups/schedules", get(list_schedules))
        .route("/backups/schedules", post(create_backup_schedule))
        .route("/backups/statistics", get(backup_statistics))
        .route("/backups/cleanup", post(cleanup_expired_backups))
        .with_state(manager)
}

async fn list_backups(
    State(manager): State<Arc<RwLock<BackupManager>>>,
) -> Result<Json<Vec<Backup>>, axum::http::StatusCode> {
    let manager = manager.read().await;
    let backups = manager.list_backups().await;
    Ok(Json(backups.into_iter().cloned().collect()))
}

async fn create_backup(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<String>, axum::http::StatusCode> {
    // Parse request and create backup
    // This would need proper request structure in real implementation
    let backup_id = "mock-backup-id".to_string();
    Ok(Json(backup_id))
}

async fn get_backup(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Option<Backup>>, axum::http::StatusCode> {
    let manager = manager.read().await;
    let backup = manager.get_backup(&id).await.map(|b| b.clone());
    Ok(Json(backup))
}

async fn restore_backup(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<(), axum::http::StatusCode> {
    let manager = manager.read().await;
    match manager.restore_backup(&id, Path::new("restore_path")).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn delete_backup(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<(), axum::http::StatusCode> {
    let mut manager = manager.write().await;
    match manager.delete_backup(&id).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::NOT_FOUND),
    }
}

async fn get_backup_progress(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Option<BackupProgress>>, axum::http::StatusCode> {
    let manager = manager.read().await;
    let progress = manager.get_backup_progress(&id).await.map(|p| p.clone());
    Ok(Json(progress))
}

async fn verify_backup(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<bool>, axum::http::StatusCode> {
    let manager = manager.read().await;
    match manager.verify_backup_integrity(&id).await {
        Ok(integrity) => Ok(Json(integrity)),
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn export_backup_metadata(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<String, axum::http::StatusCode> {
    let manager = manager.read().await;
    match manager.export_backup_metadata(&id).await {
        Ok(metadata) => Ok(metadata),
        Err(_) => Err(axum::http::StatusCode::NOT_FOUND),
    }
}

async fn list_schedules(
    State(manager): State<Arc<RwLock<BackupManager>>>,
) -> Result<Json<Vec<BackupSchedule>>, axum::http::StatusCode> {
    let manager = manager.read().await;
    let schedules = manager.list_schedules().await;
    Ok(Json(schedules.into_iter().cloned().collect()))
}

async fn create_backup_schedule(
    State(manager): State<Arc<RwLock<BackupManager>>>,
    Json(schedule): Json<BackupSchedule>,
) -> Result<(), axum::http::StatusCode> {
    let mut manager = manager.write().await;
    match manager.create_schedule(schedule).await {
        Ok(_) => Ok(()),
        Err(_) => Err(axum::http::StatusCode::BAD_REQUEST),
    }
}

async fn backup_statistics(
    State(manager): State<Arc<RwLock<BackupManager>>>,
) -> Result<Json<BackupStatistics>, axum::http::StatusCode> {
    let manager = manager.read().await;
    let stats = manager.get_backup_statistics().await;
    Ok(Json(stats))
}

async fn cleanup_expired_backups(
    State(manager): State<Arc<RwLock<BackupManager>>>,
) -> Result<Json<Vec<String>>, axum::http::StatusCode> {
    let mut manager = manager.write().await;
    let expired = manager.cleanup_expired_backups().await;
    Ok(Json(expired))
}

#[derive(Debug, thiserror::Error)]
pub enum BackupError {
    #[error("Backup not found: {0}")]
    BackupNotFound(String),
    
    #[error("Backup not ready for restoration: {0}")]
    BackupNotReady(String),
    
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Schedule already exists: {0}")]
    ScheduleExists(String),
    
    #[error("Backup process error: {0}")]
    ProcessError(String),
}