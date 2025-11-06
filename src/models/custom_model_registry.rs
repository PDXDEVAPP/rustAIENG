use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, anyhow};
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, debug, error};
use uuid::Uuid;

use crate::inference::{ModelError, ModelMetadata};
use crate::models::multimodal_processor::{MultimodalInput, MultimodalOutput};

/// Custom model registry for managing user-defined models
#[derive(Debug)]
pub struct CustomModelRegistry {
    registry_path: PathBuf,
    models: RwLock<HashMap<String, RegisteredModel>>,
    model_templates: RwLock<HashMap<String, ModelTemplate>>,
    usage_statistics: RwLock<UsageStatistics>,
    index_lock: Mutex<()>,
}

/// Model registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub model_id: String,
    pub name: String,
    pub description: String,
    pub model_type: ModelType,
    pub file_path: PathBuf,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
    pub capabilities: Vec<ModelCapability>,
    pub performance_metrics: ModelPerformanceMetrics,
    pub registration_info: RegistrationInfo,
}

/// Model types supported in custom registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    ImageGeneration,
    SpeechToText,
    TextToSpeech,
    Translation,
    Classification,
    Embedding,
    Multimodal,
    Custom(String),
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub context_length: u32,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub stop_sequences: Vec<String>,
    pub device: String,
    pub dtype: String,
    pub use_cache: bool,
    pub trust_remote_code: bool,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

/// Model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelCapability {
    Streaming,
    ToolUse,
    FunctionCalling,
    Vision,
    Audio,
    Multimodal,
    Code,
    Math,
    Reasoning,
    Custom(String),
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub avg_inference_time_ms: f64,
    pub throughput_tokens_per_sec: f64,
    pub memory_usage_mb: f64,
    pub accuracy_score: Option<f32>,
    pub perplexity: Option<f32>,
    pub quality_score: f32,
    pub last_benchmark: Option<SystemTime>,
}

/// Registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationInfo {
    pub registered_at: SystemTime,
    pub registered_by: String,
    pub version: String,
    pub license: String,
    pub tags: Vec<String>,
    pub is_public: bool,
    pub is_active: bool,
    pub download_count: u64,
}

/// Model template for easy model creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTemplate {
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub model_type: ModelType,
    pub default_config: ModelConfig,
    pub required_files: Vec<String>,
    pub optional_files: Vec<String>,
    pub setup_instructions: String,
    pub example_usage: String,
    pub template_metadata: TemplateMetadata,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub difficulty: TemplateDifficulty,
    pub estimated_setup_time: Duration,
    pub required_expertise: Vec<String>,
    pub prerequisites: Vec<String>,
    pub supported_platforms: Vec<String>,
}

/// Template difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateDifficulty {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Usage statistics
#[derive(Debug, Clone, Default)]
pub struct UsageStatistics {
    pub total_models: u64,
    pub active_models: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub popular_models: VecDeque<PopularModel>,
    pub usage_by_type: HashMap<String, u64>,
    pub daily_stats: VecDeque<DailyStats>,
}

/// Popular model entry
#[derive(Debug, Clone)]
pub struct PopularModel {
    pub model_id: String,
    pub request_count: u64,
    pub last_used: SystemTime,
}

/// Daily usage statistics
#[derive(Debug, Clone)]
pub struct DailyStats {
    pub date: SystemTime,
    pub total_requests: u64,
    pub unique_models: u64,
    pub new_registrations: u64,
}

impl CustomModelRegistry {
    /// Create a new custom model registry
    pub fn new(registry_path: PathBuf) -> Self {
        Self {
            registry_path,
            models: RwLock::new(HashMap::new()),
            model_templates: RwLock::new(HashMap::new()),
            usage_statistics: RwLock::new(UsageStatistics::default()),
            index_lock: Mutex::new(()),
        }
    }

    /// Initialize the registry (load existing models)
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing custom model registry at: {:?}", self.registry_path);
        
        // Create registry directory if it doesn't exist
        tokio::fs::create_dir_all(&self.registry_path)
            .await
            .context("Failed to create registry directory")?;

        // Load existing models
        self.load_models().await?;
        
        // Load model templates
        self.load_templates().await?;
        
        // Load usage statistics
        self.load_usage_statistics().await?;

        info!("Model registry initialized with {} models", self.models.read().await.len());
        Ok(())
    }

    /// Register a new model
    pub async fn register_model(&self, model_info: NewModelInfo) -> Result<String> {
        let _lock = self.index_lock.lock().await;
        
        let model_id = if model_info.model_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            model_info.model_id.clone()
        };

        // Validate model files exist
        self.validate_model_files(&model_info).await?;

        // Create registered model entry
        let registered_model = RegisteredModel {
            model_id: model_id.clone(),
            name: model_info.name,
            description: model_info.description,
            model_type: model_info.model_type,
            file_path: model_info.file_path,
            config: model_info.config,
            metadata: model_info.metadata,
            capabilities: model_info.capabilities,
            performance_metrics: ModelPerformanceMetrics {
                avg_inference_time_ms: 0.0,
                throughput_tokens_per_sec: 0.0,
                memory_usage_mb: 0.0,
                accuracy_score: None,
                perplexity: None,
                quality_score: 0.0,
                last_benchmark: None,
            },
            registration_info: RegistrationInfo {
                registered_at: SystemTime::now(),
                registered_by: model_info.registered_by,
                version: model_info.version,
                license: model_info.license,
                tags: model_info.tags,
                is_public: model_info.is_public,
                is_active: true,
                download_count: 0,
            },
        };

        // Save model to registry
        {
            let mut models = self.models.write().await;
            models.insert(model_id.clone(), registered_model.clone());
        }

        // Save to disk
        self.save_model_to_disk(&registered_model).await?;

        // Update statistics
        self.update_registration_stats().await?;

        info!("Registered new model: {} ({})", model_id, registered_model.name);
        Ok(model_id)
    }

    /// Get model by ID
    pub async fn get_model(&self, model_id: &str) -> Result<RegisteredModel> {
        let models = self.models.read().await;
        models.get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))
    }

    /// List all models
    pub async fn list_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// List models by type
    pub async fn list_models_by_type(&self, model_type: &ModelType) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values()
            .filter(|model| &model.model_type == model_type)
            .cloned()
            .collect()
    }

    /// List public models
    pub async fn list_public_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values()
            .filter(|model| model.registration_info.is_public)
            .cloned()
            .collect()
    }

    /// Search models by name or description
    pub async fn search_models(&self, query: &str) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        let query_lower = query.to_lowercase();
        
        models.values()
            .filter(|model| {
                model.name.to_lowercase().contains(&query_lower) ||
                model.description.to_lowercase().contains(&query_lower) ||
                model.registration_info.tags.iter()
                    .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect()
    }

    /// Update model information
    pub async fn update_model(&self, model_id: &str, updates: ModelUpdates) -> Result<()> {
        let _lock = self.index_lock.lock().await;
        
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            // Apply updates
            if let Some(name) = updates.name {
                model.name = name;
            }
            if let Some(description) = updates.description {
                model.description = description;
            }
            if let Some(config) = updates.config {
                model.config = config;
            }
            if let Some(tags) = updates.tags {
                model.registration_info.tags = tags;
            }
            if let Some(is_active) = updates.is_active {
                model.registration_info.is_active = is_active;
            }
            
            // Save updated model
            self.save_model_to_disk(model).await?;
            
            info!("Updated model: {}", model_id);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Delete model from registry
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        let _lock = self.index_lock.lock().await;
        
        let mut models = self.models.write().await;
        
        if let Some(model) = models.remove(model_id) {
            // Delete model files if requested
            if model.registration_info.is_public {
                warn!("Attempted to delete public model: {}", model_id);
                return Err(anyhow!("Cannot delete public model"));
            }
            
            // Save updated registry
            self.save_registry_index().await?;
            
            info!("Deleted model: {}", model_id);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Get model performance metrics
    pub async fn get_model_performance(&self, model_id: &str) -> Result<ModelPerformanceMetrics> {
        let models = self.models.read().await;
        
        if let Some(model) = models.get(model_id) {
            Ok(model.performance_metrics.clone())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Update model performance metrics
    pub async fn update_model_performance(&self, model_id: &str, metrics: ModelPerformanceMetrics) -> Result<()> {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            model.performance_metrics = metrics;
            self.save_model_to_disk(model).await?;
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Register a model template
    pub async fn register_template(&self, template: ModelTemplate) -> Result<()> {
        let mut templates = self.model_templates.write().await;
        templates.insert(template.template_id.clone(), template);
        
        // Save templates to disk
        self.save_templates().await?;
        
        info!("Registered model template: {}", template.template_id);
        Ok(())
    }

    /// Get model template by ID
    pub async fn get_template(&self, template_id: &str) -> Result<ModelTemplate> {
        let templates = self.model_templates.read().await;
        templates.get(template_id)
            .cloned()
            .ok_or_else(|| anyhow!("Template not found: {}", template_id))
    }

    /// List all model templates
    pub async fn list_templates(&self) -> Vec<ModelTemplate> {
        let templates = self.model_templates.read().await;
        templates.values().cloned().collect()
    }

    /// Create model from template
    pub async fn create_model_from_template(&self, template_id: &str, customization: TemplateCustomization) -> Result<NewModelInfo> {
        let template = self.get_template(template_id).await?;
        
        // Apply customization
        let mut model_config = template.default_config.clone();
        if let Some(custom_params) = customization.custom_parameters {
            for (key, value) in custom_params {
                model_config.custom_parameters.insert(key, value);
            }
        }
        
        let model_info = NewModelInfo {
            model_id: customization.model_id.unwrap_or_default(),
            name: customization.name.unwrap_or_else(|| template.name.clone()),
            description: customization.description.unwrap_or_else(|| template.description.clone()),
            model_type: template.model_type.clone(),
            file_path: customization.file_path,
            config: model_config,
            metadata: ModelMetadata {
                model_id: "".to_string(), // Will be set during registration
                name: template.name.clone(),
                description: template.description.clone(),
                architecture: "custom".to_string(),
                parameters: 0,
                context_length: template.default_config.context_length,
                vocabulary_size: 0,
                quantization: None,
                tags: template.template_metadata.difficulty.to_string(),
            },
            capabilities: vec![], // Set during registration
            registered_by: customization.registered_by,
            version: "1.0.0".to_string(),
            license: "custom".to_string(),
            tags: template.template_metadata.difficulty.to_string().into(),
            is_public: false,
        };
        
        Ok(model_info)
    }

    /// Record model usage
    pub async fn record_usage(&self, model_id: &str, success: bool) -> Result<()> {
        let mut usage_stats = self.usage_statistics.write().await;
        
        usage_stats.total_requests += 1;
        if success {
            usage_stats.successful_requests += 1;
        } else {
            usage_stats.failed_requests += 1;
        }
        
        // Update popular models
        let model_id_str = model_id.to_string();
        if let Some(popular) = usage_stats.popular_models.iter_mut().find(|p| p.model_id == model_id_str) {
            popular.request_count += 1;
            popular.last_used = SystemTime::now();
        } else {
            usage_stats.popular_models.push_back(PopularModel {
                model_id: model_id_str,
                request_count: 1,
                last_used: SystemTime::now(),
            });
            
            // Keep only top 10
            while usage_stats.popular_models.len() > 10 {
                usage_stats.popular_models.pop_front();
            }
        }
        
        // Update usage by type
        if let Some(model) = self.models.read().await.get(model_id) {
            let model_type_str = format!("{:?}", model.model_type);
            *usage_stats.usage_by_type.entry(model_type_str).or_insert(0) += 1;
        }
        
        // Save usage statistics
        self.save_usage_statistics().await?;
        
        Ok(())
    }

    /// Get usage statistics
    pub async fn get_usage_statistics(&self) -> UsageStatistics {
        self.usage_statistics.read().await.clone()
    }

    /// Export model registry
    pub async fn export_registry(&self, export_path: &Path, include_models: bool) -> Result<()> {
        let export_data = if include_models {
            // Export with model data
            let models = self.list_models().await;
            serde_json::to_string_pretty(&ExportData {
                models,
                templates: self.list_templates().await,
                usage_statistics: self.get_usage_statistics().await,
                export_info: ExportInfo {
                    exported_at: SystemTime::now(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    include_models,
                },
            })
        } else {
            // Export without model data (index only)
            let usage_stats = self.get_usage_statistics().await;
            serde_json::to_string_pretty(&ExportData {
                models: vec![], // Empty
                templates: self.list_templates().await,
                usage_statistics: usage_stats,
                export_info: ExportInfo {
                    exported_at: SystemTime::now(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    include_models,
                },
            })
        }?;

        tokio::fs::write(export_path, export_data)
            .await
            .context("Failed to write export file")?;

        info!("Exported model registry to: {:?}", export_path);
        Ok(())
    }

    /// Import model registry
    pub async fn import_registry(&self, import_path: &Path) -> Result<ImportResult> {
        let import_data: ExportData = tokio::fs::read_to_string(import_path)
            .await
            .context("Failed to read import file")?
            .parse()
            .context("Failed to parse import data")?;

        let mut imported_count = 0;
        let mut skipped_count = 0;

        // Import models
        if import_data.export_info.include_models {
            for model in import_data.models {
                let model_id = format!("imported_{}", model.model_id);
                
                // Check if model already exists
                if self.models.read().await.contains_key(&model_id) {
                    skipped_count += 1;
                    continue;
                }

                // Register the model
                let mut imported_model = model;
                imported_model.model_id = model_id;
                
                {
                    let mut models = self.models.write().await;
                    models.insert(model_id.clone(), imported_model);
                }
                
                imported_count += 1;
            }
        }

        // Import templates
        for template in import_data.templates {
            let template_id = format!("imported_{}", template.template_id);
            let mut imported_template = template;
            imported_template.template_id = template_id;
            
            {
                let mut templates = self.model_templates.write().await;
                templates.insert(template_id, imported_template);
            }
        }

        // Save imported data
        self.save_registry_index().await?;
        self.save_templates().await?;

        Ok(ImportResult {
            imported_models: imported_count,
            skipped_models: skipped_count,
            imported_templates: import_data.templates.len(),
        })
    }

    /// Load models from disk
    async fn load_models(&self) -> Result<()> {
        let models_dir = self.registry_path.join("models");
        
        if !models_dir.exists() {
            return Ok(());
        }

        let mut entries = tokio::fs::read_dir(&models_dir).await
            .context("Failed to read models directory")?;

        while let Some(entry) = entries.next_entry().await
            .context("Failed to read model entry")? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let model_data = tokio::fs::read_to_string(&path)
                    .await
                    .context("Failed to read model file")?;
                
                let model: RegisteredModel = model_data.parse()
                    .context("Failed to parse model data")?;
                
                let mut models = self.models.write().await;
                models.insert(model.model_id.clone(), model);
            }
        }

        Ok(())
    }

    /// Load model templates from disk
    async fn load_templates(&self) -> Result<()> {
        let templates_file = self.registry_path.join("templates.json");
        
        if !templates_file.exists() {
            return Ok(());
        }

        let templates_data = tokio::fs::read_to_string(&templates_file)
            .await
            .context("Failed to read templates file")?;
        
        let templates: HashMap<String, ModelTemplate> = templates_data.parse()
            .context("Failed to parse templates data")?;

        let mut templates_lock = self.model_templates.write().await;
        *templates_lock = templates;

        Ok(())
    }

    /// Load usage statistics from disk
    async fn load_usage_statistics(&self) -> Result<()> {
        let stats_file = self.registry_path.join("usage_statistics.json");
        
        if !stats_file.exists() {
            return Ok(());
        }

        let stats_data = tokio::fs::read_to_string(&stats_file)
            .await
            .context("Failed to read usage statistics file")?;
        
        let stats: UsageStatistics = stats_data.parse()
            .context("Failed to parse usage statistics")?;

        let mut stats_lock = self.usage_statistics.write().await;
        *stats_lock = stats;

        Ok(())
    }

    /// Save model to disk
    async fn save_model_to_disk(&self, model: &RegisteredModel) -> Result<()> {
        let models_dir = self.registry_path.join("models");
        tokio::fs::create_dir_all(&models_dir).await?;
        
        let model_file = models_dir.join(format!("{}.json", model.model_id));
        let model_json = serde_json::to_string_pretty(model)
            .context("Failed to serialize model data")?;
        
        tokio::fs::write(&model_file, model_json)
            .await
            .context("Failed to write model file")?;
        
        Ok(())
    }

    /// Save registry index
    async fn save_registry_index(&self) -> Result<()> {
        let models = self.list_models().await;
        let index_data = RegistryIndex {
            total_models: models.len() as u64,
            last_updated: SystemTime::now(),
            models: models.into_iter().map(|m| (m.model_id.clone(), m.name)).collect(),
        };
        
        let index_file = self.registry_path.join("registry_index.json");
        let index_json = serde_json::to_string_pretty(&index_data)
            .context("Failed to serialize registry index")?;
        
        tokio::fs::write(&index_file, index_json)
            .await
            .context("Failed to write registry index")?;
        
        Ok(())
    }

    /// Save model templates
    async fn save_templates(&self) -> Result<()> {
        let templates = self.list_templates().await;
        let templates_data: HashMap<String, ModelTemplate> = templates
            .into_iter()
            .map(|t| (t.template_id.clone(), t))
            .collect();
        
        let templates_file = self.registry_path.join("templates.json");
        let templates_json = serde_json::to_string_pretty(&templates_data)
            .context("Failed to serialize templates data")?;
        
        tokio::fs::write(&templates_file, templates_json)
            .await
            .context("Failed to write templates file")?;
        
        Ok(())
    }

    /// Save usage statistics
    async fn save_usage_statistics(&self) -> Result<()> {
        let stats = self.get_usage_statistics().await;
        let stats_file = self.registry_path.join("usage_statistics.json");
        let stats_json = serde_json::to_string_pretty(&stats)
            .context("Failed to serialize usage statistics")?;
        
        tokio::fs::write(&stats_file, stats_json)
            .await
            .context("Failed to write usage statistics")?;
        
        Ok(())
    }

    /// Validate model files
    async fn validate_model_files(&self, model_info: &NewModelInfo) -> Result<()> {
        if !model_info.file_path.exists() {
            return Err(anyhow!("Model file not found: {:?}", model_info.file_path));
        }
        
        // Additional validation based on model type
        match model_info.model_type {
            ModelType::TextGeneration => {
                // Validate text generation model files
                debug!("Validating text generation model files");
            }
            ModelType::ImageGeneration => {
                // Validate image generation model files
                debug!("Validating image generation model files");
            }
            _ => {
                debug!("Validating custom model files");
            }
        }
        
        Ok(())
    }

    /// Update registration statistics
    async fn update_registration_stats(&self) -> Result<()> {
        let mut usage_stats = self.usage_statistics.write().await;
        let models = self.list_models().await;
        
        usage_stats.total_models = models.len() as u64;
        usage_stats.active_models = models.iter().filter(|m| m.registration_info.is_active).count() as u64;
        
        self.save_usage_statistics().await?;
        Ok(())
    }
}

/// Supporting data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewModelInfo {
    pub model_id: String,
    pub name: String,
    pub description: String,
    pub model_type: ModelType,
    pub file_path: PathBuf,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
    pub capabilities: Vec<ModelCapability>,
    pub registered_by: String,
    pub version: String,
    pub license: String,
    pub tags: Vec<String>,
    pub is_public: bool,
}

#[derive(Debug, Clone)]
pub struct ModelUpdates {
    pub name: Option<String>,
    pub description: Option<String>,
    pub config: Option<ModelConfig>,
    pub tags: Option<Vec<String>>,
    pub is_active: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct TemplateCustomization {
    pub model_id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub file_path: PathBuf,
    pub custom_parameters: Option<HashMap<String, serde_json::Value>>,
    pub registered_by: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistryIndex {
    pub total_models: u64,
    pub last_updated: SystemTime,
    pub models: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExportData {
    pub models: Vec<RegisteredModel>,
    pub templates: Vec<ModelTemplate>,
    pub usage_statistics: UsageStatistics,
    pub export_info: ExportInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportInfo {
    pub exported_at: SystemTime,
    pub version: String,
    pub include_models: bool,
}

#[derive(Debug)]
pub struct ImportResult {
    pub imported_models: usize,
    pub skipped_models: usize,
    pub imported_templates: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let registry = CustomModelRegistry::new(temp_dir.path().to_path_buf());
        
        let templates = registry.list_templates().await;
        assert_eq!(templates.len(), 0);
        
        let usage_stats = registry.get_usage_statistics().await;
        assert_eq!(usage_stats.total_models, 0);
    }

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            context_length: 2048,
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.1,
            do_sample: true,
            stop_sequences: vec!["<end>".to_string()],
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            use_cache: true,
            trust_remote_code: false,
            custom_parameters: HashMap::new(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.context_length, deserialized.context_length);
        assert_eq!(config.temperature, deserialized.temperature);
    }

    #[test]
    fn test_model_type_variants() {
        let types = vec![
            ModelType::TextGeneration,
            ModelType::ImageGeneration,
            ModelType::SpeechToText,
            ModelType::Custom("custom_type".to_string()),
        ];

        for model_type in types {
            let json = serde_json::to_string(&model_type).unwrap();
            let deserialized: ModelType = serde_json::from_str(&json).unwrap();
            assert_eq!(model_type, deserialized);
        }
    }

    #[test]
    fn test_template_difficulty_levels() {
        let difficulties = vec![
            TemplateDifficulty::Beginner,
            TemplateDifficulty::Intermediate,
            TemplateDifficulty::Advanced,
            TemplateDifficulty::Expert,
        ];

        for difficulty in difficulties {
            let json = serde_json::to_string(&difficulty).unwrap();
            let deserialized: TemplateDifficulty = serde_json::from_str(&json).unwrap();
            assert_eq!(difficulty, deserialized);
        }
    }

    #[test]
    fn test_capability_serialization() {
        let capabilities = vec![
            ModelCapability::Streaming,
            ModelCapability::Vision,
            ModelCapability::Custom("custom_capability".to_string()),
        ];

        for capability in capabilities {
            let json = serde_json::to_string(&capability).unwrap();
            let deserialized: ModelCapability = serde_json::from_str(&json).unwrap();
            assert_eq!(capability, deserialized);
        }
    }
}