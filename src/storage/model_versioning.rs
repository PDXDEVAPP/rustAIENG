use crate::core::ModelManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn, error, debug, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_name: String,
    pub version: String,
    pub file_path: PathBuf,
    pub file_size: u64,
    pub sha256: String,
    pub description: Option<String>,
    pub metadata: ModelVersionMetadata,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub created_by: String,
    pub tags: Vec<String>,
    pub parent_version: Option<String>,
    pub is_active: bool,
    pub download_count: u64,
    pub usage_statistics: ModelUsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionMetadata {
    pub model_type: String,
    pub quantization: String,
    pub context_length: u32,
    pub parameters: String,
    pub gpu_layers: Option<u32>,
    pub supported_features: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub custom_attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsageStats {
    pub inference_count: u64,
    pub total_tokens_generated: u64,
    pub average_response_time_ms: f64,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub user_feedback_score: Option<f64>,
    pub success_rate: f64,
    pub error_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionTag {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub version_id: String,
    pub is_official: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct ModelVersionManager {
    models_dir: PathBuf,
    versions_file: PathBuf,
    tags_file: PathBuf,
    current_versions: HashMap<String, ModelVersion>,
    version_tags: Vec<VersionTag>,
}

impl ModelVersionManager {
    pub fn new(models_dir: &Path) -> Self {
        let versions_file = models_dir.join("versions.json");
        let tags_file = models_dir.join("tags.json");
        
        // Ensure directories exist
        if !models_dir.exists() {
            fs::create_dir_all(models_dir).expect("Failed to create models directory");
        }

        Self {
            models_dir: models_dir.to_path_buf(),
            versions_file,
            tags_file,
            current_versions: HashMap::new(),
            version_tags: Vec::new(),
        }
    }

    #[instrument(skip(self))]
    pub async fn initialize(&mut self) -> Result<(), VersioningError> {
        info!("Initializing model version manager...");

        // Load existing versions and tags
        self.load_versions().await?;
        self.load_tags().await?;

        // Verify file integrity
        self.verify_version_integrity().await?;

        info!("Model version manager initialized with {} versions", self.current_versions.len());
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn create_version(
        &mut self,
        model_name: String,
        version: String,
        file_path: PathBuf,
        description: Option<String>,
        metadata: ModelVersionMetadata,
        created_by: String,
    ) -> Result<ModelVersion, VersioningError> {
        info!("Creating new version: {}:{}", model_name, version);

        // Verify file exists and calculate hash
        if !file_path.exists() {
            return Err(VersioningError::FileNotFound(file_path.display().to_string()));
        }

        let file_size = fs::metadata(&file_path)?.len();
        let sha256 = self.calculate_file_hash(&file_path)?;

        // Check if version already exists
        if self.current_versions.contains_key(&format!("{}:{}", model_name, version)) {
            return Err(VersioningError::VersionExists(model_name, version));
        }

        // Find parent version if exists
        let parent_version = self.get_latest_version(&model_name)
            .map(|v| v.version.clone());

        let version_data = ModelVersion {
            id: format!("{}:{}", model_name, version),
            model_name: model_name.clone(),
            version,
            file_path,
            file_size,
            sha256,
            description,
            metadata,
            created_at: chrono::Utc::now(),
            created_by,
            tags: Vec::new(),
            parent_version,
            is_active: true,
            download_count: 0,
            usage_statistics: ModelUsageStats {
                inference_count: 0,
                total_tokens_generated: 0,
                average_response_time_ms: 0.0,
                last_used: None,
                user_feedback_score: None,
                success_rate: 100.0,
                error_count: 0,
            },
        };

        // Add to versions
        self.current_versions.insert(version_data.id.clone(), version_data.clone());

        // Save to disk
        self.save_versions().await?;

        info!("Created version: {}", version_data.id);
        Ok(version_data)
    }

    #[instrument(skip(self))]
    pub async fn get_version(&self, model_name: &str, version: &str) -> Option<&ModelVersion> {
        self.current_versions.get(&format!("{}:{}", model_name, version))
    }

    #[instrument(skip(self))]
    pub async fn get_latest_version(&self, model_name: &str) -> Option<&ModelVersion> {
        let versions: Vec<_> = self.current_versions.values()
            .filter(|v| v.model_name == model_name)
            .collect();

        versions.into_iter()
            .max_by(|a, b| a.created_at.cmp(&b.created_at))
    }

    #[instrument(skip(self))]
    pub async fn list_versions(&self, model_name: Option<&str>) -> Vec<&ModelVersion> {
        let versions = self.current_versions.values();
        
        if let Some(model) = model_name {
            versions.filter(|v| v.model_name == model).collect()
        } else {
            versions.collect()
        }
    }

    #[instrument(skip(self))]
    pub async fn list_versions_detailed(&self, model_name: Option<&str>) -> Vec<ModelVersion> {
        self.list_versions(model_name)
            .into_iter()
            .cloned()
            .collect()
    }

    #[instrument(skip(self))]
    pub async fn deactivate_version(&mut self, model_name: &str, version: &str) -> Result<(), VersioningError> {
        let key = format!("{}:{}", model_name, version);
        
        if let Some(v) = self.current_versions.get_mut(&key) {
            v.is_active = false;
            self.save_versions().await?;
            info!("Deactivated version: {}", key);
            Ok(())
        } else {
            Err(VersioningError::VersionNotFound(key))
        }
    }

    #[instrument(skip(self))]
    pub async fn update_usage_statistics(
        &mut self,
        model_name: &str,
        version: &str,
        inference_time_ms: u64,
        tokens_generated: u64,
        success: bool,
    ) -> Result<(), VersioningError> {
        let key = format!("{}:{}", model_name, version);
        
        if let Some(v) = self.current_versions.get_mut(&key) {
            v.usage_statistics.inference_count += 1;
            v.usage_statistics.total_tokens_generated += tokens_generated;
            
            // Update average response time
            let current_avg = v.usage_statistics.average_response_time_ms;
            let new_avg = (current_avg * (v.usage_statistics.inference_count - 1) as f64 + inference_time_ms as f64) 
                / v.usage_statistics.inference_count as f64;
            v.usage_statistics.average_response_time_ms = new_avg;
            
            v.usage_statistics.last_used = Some(chrono::Utc::now());
            
            if success {
                let success_rate = ((v.usage_statistics.inference_count - v.usage_statistics.error_count) as f64 
                    / v.usage_statistics.inference_count as f64) * 100.0;
                v.usage_statistics.success_rate = success_rate;
            } else {
                v.usage_statistics.error_count += 1;
                let success_rate = ((v.usage_statistics.inference_count - v.usage_statistics.error_count) as f64 
                    / v.usage_statistics.inference_count as f64) * 100.0;
                v.usage_statistics.success_rate = success_rate;
            }

            self.save_versions().await?;
            debug!("Updated usage statistics for version: {}", key);
            Ok(())
        } else {
            Err(VersioningError::VersionNotFound(key))
        }
    }

    #[instrument(skip(self))]
    pub async fn add_tag(&mut self, model_name: &str, version: &str, tag_name: String, description: Option<String>) -> Result<(), VersioningError> {
        let version_id = format!("{}:{}", model_name, version);
        
        if !self.current_versions.contains_key(&version_id) {
            return Err(VersioningError::VersionNotFound(version_id));
        }

        // Check if tag already exists
        if self.version_tags.iter().any(|t| t.name == tag_name) {
            return Err(VersioningError::TagExists(tag_name));
        }

        let tag = VersionTag {
            id: uuid::Uuid::new_v4().to_string(),
            name: tag_name.clone(),
            description,
            version_id,
            is_official: false,
            created_at: chrono::Utc::now(),
        };

        self.version_tags.push(tag);

        // Update model's tags
        if let Some(v) = self.current_versions.get_mut(&version_id) {
            v.tags.push(tag_name);
        }

        self.save_tags().await?;
        info!("Added tag '{}' to version: {}", tag_name, version_id);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn search_by_tag(&self, tag_name: &str) -> Vec<&ModelVersion> {
        self.version_tags
            .iter()
            .filter(|tag| tag.name == tag_name)
            .filter_map(|tag| self.current_versions.get(&tag.version_id))
            .collect()
    }

    #[instrument(skip(self))]
    pub async fn rollback_to_version(&mut self, model_name: &str, target_version: &str) -> Result<(), VersioningError> {
        let target_key = format!("{}:{}", model_name, target_version);
        
        if !self.current_versions.contains_key(&target_key) {
            return Err(VersioningError::VersionNotFound(target_key));
        }

        // Deactivate all other versions of this model
        for v in self.current_versions.values_mut() {
            if v.model_name == model_name {
                v.is_active = false;
            }
        }

        // Activate target version
        if let Some(v) = self.current_versions.get_mut(&target_key) {
            v.is_active = true;
        }

        self.save_versions().await?;
        info!("Rolled back {} to version: {}", model_name, target_version);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn compare_versions(&self, model_name: &str, version1: &str, version2: &str) -> Result<VersionComparison, VersioningError> {
        let v1 = self.get_version(model_name, version1)
            .ok_or_else(|| VersioningError::VersionNotFound(format!("{}:{}", model_name, version1)))?;
        let v2 = self.get_version(model_name, version2)
            .ok_or_else(|| VersioningError::VersionNotFound(format!("{}:{}", model_name, version2)))?;

        let comparison = VersionComparison {
            version1: v1.clone(),
            version2: v2.clone(),
            size_difference: (v1.file_size as i64) - (v2.file_size as i64),
            usage_comparison: VersionUsageComparison {
                inference_count_diff: (v1.usage_statistics.inference_count as i64) - (v2.usage_statistics.inference_count as i64),
                tokens_generated_diff: (v1.usage_statistics.total_tokens_generated as i64) - (v2.usage_statistics.total_tokens_generated as i64),
                success_rate_diff: v1.usage_statistics.success_rate - v2.usage_statistics.success_rate,
                avg_response_time_diff: v1.usage_statistics.average_response_time_ms - v2.usage_statistics.average_response_time_ms,
            },
            metadata_comparison: VersionMetadataComparison {
                context_length_diff: (v1.metadata.context_length as i64) - (v2.metadata.context_length as i64),
                parameter_count_diff: self.extract_parameter_count(&v1.metadata.parameters) - self.extract_parameter_count(&v2.metadata.parameters),
                quantization_diff: if v1.metadata.quantization == v2.metadata.quantization { 0 } else { 1 },
            },
            generated_at: chrono::Utc::now(),
        };

        Ok(comparison)
    }

    fn extract_parameter_count(&self, parameters: &str) -> i64 {
        // Extract numeric parameter count from string like "7B", "13B", etc.
        if let Some(num_str) = parameters.chars().take_while(|c| c.is_numeric()).collect::<String>().parse::<i64>().ok() {
            num_str
        } else {
            0
        }
    }

    #[instrument(skip(self))]
    pub async fn export_version_metadata(&self, model_name: &str, version: &str) -> Result<String, VersioningError> {
        let v = self.get_version(model_name, version)
            .ok_or_else(|| VersioningError::VersionNotFound(format!("{}:{}", model_name, version)))?;

        let metadata_export = VersionMetadataExport {
            version: v.clone(),
            tags: self.version_tags
                .iter()
                .filter(|tag| tag.version_id == v.id)
                .cloned()
                .collect(),
        };

        let json = serde_json::to_string_pretty(&metadata_export)
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;

        Ok(json)
    }

    #[instrument(skip(self))]
    pub async fn import_version_metadata(&mut self, metadata_json: &str) -> Result<ModelVersion, VersioningError> {
        let metadata_export: VersionMetadataExport = serde_json::from_str(metadata_json)
            .map_err(|e| VersioningError::DeserializationError(e.to_string()))?;

        // Add to current versions
        self.current_versions.insert(metadata_export.version.id.clone(), metadata_export.version.clone());

        // Add tags
        for tag in metadata_export.tags {
            self.version_tags.push(tag);
        }

        self.save_versions().await?;
        self.save_tags().await?;

        info!("Imported version metadata: {}", metadata_export.version.id);
        Ok(metadata_export.version)
    }

    async fn load_versions(&mut self) -> Result<(), VersioningError> {
        if self.versions_file.exists() {
            let content = fs::read_to_string(&self.versions_file)
                .map_err(|e| VersioningError::FileReadError(self.versions_file.display().to_string()))?;

            let versions_map: HashMap<String, ModelVersion> = serde_json::from_str(&content)
                .map_err(|e| VersioningError::DeserializationError(e.to_string()))?;

            self.current_versions = versions_map;
            debug!("Loaded {} versions from disk", self.current_versions.len());
        }

        Ok(())
    }

    async fn load_tags(&mut self) -> Result<(), VersioningError> {
        if self.tags_file.exists() {
            let content = fs::read_to_string(&self.tags_file)
                .map_err(|e| VersioningError::FileReadError(self.tags_file.display().to_string()))?;

            self.version_tags = serde_json::from_str(&content)
                .map_err(|e| VersioningError::DeserializationError(e.to_string()))?;

            debug!("Loaded {} tags from disk", self.version_tags.len());
        }

        Ok(())
    }

    async fn save_versions(&self) -> Result<(), VersioningError> {
        let content = serde_json::to_string_pretty(&self.current_versions)
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;

        fs::write(&self.versions_file, content)
            .map_err(|e| VersioningError::FileWriteError(self.versions_file.display().to_string()))?;

        Ok(())
    }

    async fn save_tags(&self) -> Result<(), VersioningError> {
        let content = serde_json::to_string_pretty(&self.version_tags)
            .map_err(|e| VersioningError::SerializationError(e.to_string()))?;

        fs::write(&self.tags_file, content)
            .map_err(|e| VersioningError::FileWriteError(self.tags_file.display().to_string()))?;

        Ok(())
    }

    async fn verify_version_integrity(&self) -> Result<(), VersioningError> {
        for version in self.current_versions.values() {
            if !version.file_path.exists() {
                warn!("Version file not found: {}", version.file_path.display());
                continue;
            }

            let current_hash = self.calculate_file_hash(&version.file_path)?;
            if current_hash != version.sha256 {
                warn!("Version integrity check failed for {}: expected {}, got {}", 
                      version.id, version.sha256, current_hash);
            }
        }

        Ok(())
    }

    fn calculate_file_hash(&self, file_path: &Path) -> Result<String, VersioningError> {
        use std::io::{BufRead, BufReader};

        let file = fs::File::open(file_path)
            .map_err(|e| VersioningError::FileReadError(file_path.display().to_string()))?;
        
        let reader = BufReader::new(file);
        let mut hasher = sha2::Sha256::new();

        for line in reader.lines() {
            hasher.update(line.map_err(|e| VersioningError::IoError(e.to_string()))?.as_bytes());
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    pub async fn get_version_statistics(&self) -> VersionStatistics {
        let total_versions = self.current_versions.len();
        let active_versions = self.current_versions.values().filter(|v| v.is_active).count();
        let total_size: u64 = self.current_versions.values().map(|v| v.file_size).sum();
        
        let model_counts: HashMap<String, usize> = self.current_versions.values()
            .fold(HashMap::new(), |mut acc, v| {
                *acc.entry(v.model_name.clone()).or_insert(0) += 1;
                acc
            });

        let most_used_versions: Vec<_> = self.current_versions.values()
            .filter(|v| v.is_active)
            .sorted_by_key(|v| std::cmp::Reverse(v.usage_statistics.inference_count))
            .take(5)
            .map(|v| v.id.clone())
            .collect();

        VersionStatistics {
            total_versions,
            active_versions,
            total_size_bytes: total_size,
            model_distribution: model_counts,
            most_used_versions,
            average_file_size: if total_versions > 0 { total_size / total_versions as u64 } else { 0 },
            total_tags: self.version_tags.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadataExport {
    pub version: ModelVersion,
    pub tags: Vec<VersionTag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    pub version1: ModelVersion,
    pub version2: ModelVersion,
    pub size_difference: i64,
    pub usage_comparison: VersionUsageComparison,
    pub metadata_comparison: VersionMetadataComparison,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionUsageComparison {
    pub inference_count_diff: i64,
    pub tokens_generated_diff: i64,
    pub success_rate_diff: f64,
    pub avg_response_time_diff: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadataComparison {
    pub context_length_diff: i64,
    pub parameter_count_diff: i64,
    pub quantization_diff: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStatistics {
    pub total_versions: usize,
    pub active_versions: usize,
    pub total_size_bytes: u64,
    pub model_distribution: HashMap<String, usize>,
    pub most_used_versions: Vec<String>,
    pub average_file_size: u64,
    pub total_tags: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum VersioningError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("File read error: {0}")]
    FileReadError(String),
    
    #[error("File write error: {0}")]
    FileWriteError(String),
    
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Version already exists: {0}:{1}")]
    VersionExists(String, String),
    
    #[error("Version not found: {0}")]
    VersionNotFound(String),
    
    #[error("Tag already exists: {0}")]
    TagExists(String),
    
    #[error("Invalid version comparison: {0}")]
    InvalidComparison(String),
}

// Extension trait for easier sorting
trait VersionSorter<T> {
    fn sorted_by_key<F: FnMut(&T) -> K, K: Ord>(&mut self, f: F);
}

impl<T> VersionSorter<T> for Vec<T> {
    fn sorted_by_key<F: FnMut(&T) -> K, K: Ord>(&mut self, mut f: F) {
        self.sort_by_key(f);
    }
}