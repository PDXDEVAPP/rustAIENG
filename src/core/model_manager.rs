use crate::core::database::{DatabaseManager, Model, ModelType};
use crate::core::inference_engine::{InferenceEngine, InferenceConfig};
use anyhow::{Context, Result};
use futures::stream::StreamExt;
use reqwest::Client;
use std::path::PathBuf;
use std::time::SystemTime;
use tokio::fs;
use tracing::{info, warn, error, progress};
use uuid::Uuid;

pub struct ModelManager {
    db: DatabaseManager,
    inference_engine: InferenceEngine,
    models_dir: PathBuf,
    client: Client,
}

impl ModelManager {
    pub fn new(db: DatabaseManager, inference_engine: InferenceEngine, models_dir: PathBuf) -> Self {
        Self {
            db,
            inference_engine,
            models_dir,
            client: Client::new(),
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        // Ensure models directory exists
        fs::create_dir_all(&self.models_dir).await
            .context("Failed to create models directory")?;
        
        info!("Model manager initialized with models directory: {:?}", self.models_dir);
        Ok(())
    }

    pub async fn list_local_models(&self) -> Result<Vec<Model>> {
        self.db.list_models().await
            .context("Failed to list models from database")
    }

    pub async fn get_model_details(&self, model_name: &str) -> Result<Option<Model>> {
        self.db.get_model(model_name).await
            .context("Failed to get model details")
    }

    pub async fn pull_model(&self, model_name: &str) -> Result<Model> {
        info!("Pulling model: {}", model_name);
        
        let model_id = Uuid::new_v4().to_string();
        let model_path = self.models_dir.join(format!("{}.gguf", model_name));
        
        // Check if model already exists
        if model_path.exists() {
            info!("Model {} already exists at {:?}", model_name, model_path);
        } else {
            // Download model from HuggingFace Hub
            self.download_model_from_hf(model_name, &model_path).await?;
        }
        
        let model = Model {
            id: model_id.clone(),
            name: model_name.to_string(),
            display_name: Some(model_name.to_string()),
            file_path: model_path,
            size_bytes: 1024 * 1024 * 100, // 100MB placeholder
            model_type: ModelType::from_str(model_name),
            quantization: Some("Q4_0".to_string()),
            context_length: Some(4096),
            max_tokens: Some(2048),
            parameters: None,
            description: Some(format!("Model: {}", model_name)),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            in_use: false,
        };

        self.db.add_model(model.clone()).await
            .context("Failed to save model to database")?;

        info!("Model pulled successfully: {}", model_name);
        Ok(model)
    }

    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        info!("Loading model: {}", model_name);
        
        let model = self.get_model_details(model_name).await?
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;

        let config = InferenceConfig::default();
        
        self.inference_engine.load_model(&model.file_path, model_name, config).await
            .context("Failed to load model into inference engine")?;
        
        self.db.mark_model_in_use(model_name, true).await
            .context("Failed to mark model as in use")?;

        info!("Model loaded successfully: {}", model_name);
        Ok(())
    }

    pub async fn unload_model(&self, model_name: &str) -> Result<bool> {
        info!("Unloading model: {}", model_name);
        
        let success = self.inference_engine.unload_model(model_name).await
            .context("Failed to unload model from inference engine")?;
        
        if success {
            self.db.mark_model_in_use(model_name, false).await
                .context("Failed to mark model as not in use")?;
            info!("Model unloaded successfully: {}", model_name);
        }
        
        Ok(success)
    }

    pub async fn remove_model(&self, model_name: &str) -> Result<bool> {
        info!("Removing model: {}", model_name);
        
        // Unload model if it's currently loaded
        let _ = self.unload_model(model_name).await;
        
        let model = self.get_model_details(model_name).await?
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;
        
        // Remove model file
        if model.file_path.exists() {
            fs::remove_file(&model.file_path).await
                .context("Failed to remove model file")?;
        }
        
        // Remove from database
        let removed = self.db.remove_model(model_name).await
            .context("Failed to remove model from database")?;
        
        if removed {
            info!("Model removed successfully: {}", model_name);
        }
        
        Ok(removed)
    }

    pub async fn copy_model(&self, source_name: &str, target_name: &str) -> Result<Model> {
        info!("Copying model from {} to {}", source_name, target_name);
        
        let source_model = self.get_model_details(source_name).await?
            .ok_or_else(|| anyhow::anyhow!("Source model not found: {}", source_name))?;
        
        // Create new model with different ID and name
        let new_model_id = Uuid::new_v4().to_string();
        let target_path = self.models_dir.join(format!("{}.gguf", new_model_id));
        
        // Copy file (simulate for now)
        self.simulate_model_copy(&source_model.file_path, &target_path).await?;
        
        let new_model = Model {
            id: new_model_id,
            name: target_name.to_string(),
            display_name: Some(format!("{} (copy)", source_model.display_name.unwrap_or_else(|| source_name.to_string()))),
            file_path: target_path,
            size_bytes: source_model.size_bytes,
            model_type: source_model.model_type,
            quantization: source_model.quantization,
            context_length: source_model.context_length,
            max_tokens: source_model.max_tokens,
            parameters: source_model.parameters,
            description: source_model.description.map(|d| format!("{} (copy)", d)),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            in_use: false,
        };

        self.db.add_model(new_model.clone()).await
            .context("Failed to save copied model to database")?;

        info!("Model copied successfully from {} to {}", source_name, target_name);
        Ok(new_model)
    }

    pub async fn list_running_models(&self) -> Result<Vec<String>> {
        self.inference_engine.list_loaded_models().await
            .context("Failed to list running models")
    }

    pub async fn stop_model(&self, model_name: &str) -> Result<bool> {
        self.unload_model(model_name).await
    }

    // Download model from HuggingFace Hub
    async fn download_model_from_hf(&self, model_name: &str, model_path: &PathBuf) -> Result<()> {
        info!("Downloading model from HuggingFace: {}", model_name);
        
        // Map common model names to HuggingFace repos
        let repo_id = match model_name.to_lowercase().as_str() {
            "llama3.2" => "meta-llama/Llama-3.2-1B-Instruct-GGUF",
            "llama3.1" => "meta-llama/Llama-3.1-8B-Instruct-GGUF", 
            "mistral" => "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            "codellama" => "TheBloke/CodeLlama-7B-Instruct-GGUF",
            "phi3" => "microsoft/Phi-3-mini-128k-instruct-gguf",
            _ => {
                // Default to a general model or return error for unknown models
                return Err(anyhow::anyhow!("Unknown model: {}. Please specify a supported model name.", model_name));
            }
        };
        
        // Create client for HuggingFace Hub
        let api = hf_hub::api::tokio::Api::new().await
            .context("Failed to initialize HuggingFace API")?;
        
        let repo = api.model(repo_id.to_string());
        
        // Download the quantized GGUF model file
        let model_file = match model_name.to_lowercase().as_str() {
            name if name.contains("llama3") => "Llama-3.2-1B-Instruct-Q4_0.gguf",
            name if name.contains("mistral") => "mistral-7b-instruct-v0.1.Q4_0.gguf",
            name if name.contains("codellama") => "codellama-7b-instruct.Q4_0.gguf", 
            name if name.contains("phi3") => "Phi-3-mini-128k-instruct-q4.gguf",
            _ => "model-q4_0.gguf" // fallback
        };
        
        info!("Downloading {} from HuggingFace Hub...", model_file);
        
        // Download file with progress
        let temp_dir = self.models_dir.join("temp");
        fs::create_dir_all(&temp_dir).await?;
        
        let file_path = repo.get(&model_file).await
            .context(format!("Failed to download model file: {}", model_file))?;
        
        // Move downloaded file to final location
        fs::rename(&file_path, model_path).await
            .context("Failed to move downloaded model to final location")?;
        
        // Clean up temp directory
        let _ = fs::remove_dir_all(&temp_dir).await;
        
        info!("Model downloaded successfully to: {:?}", model_path);
        Ok(())
    }

    // Simulate model copy (placeholder implementation)
    async fn simulate_model_copy(&self, source: &PathBuf, target: &PathBuf) -> Result<()> {
        info!("Simulating model copy from {:?} to {:?}", source, target);
        
        // Read source and write to target
        let content = fs::read(source).await
            .context("Failed to read source model file")?;
        fs::write(target, content).await
            .context("Failed to write target model file")?;
        
        Ok(())
    }

    // Search for models in local registry
    pub async fn search_local_models(&self, query: &str) -> Result<Vec<Model>> {
        let all_models = self.list_local_models().await?;
        
        let filtered_models: Vec<Model> = all_models.into_iter()
            .filter(|model| {
                model.name.to_lowercase().contains(&query.to_lowercase()) ||
                model.display_name.as_ref().map(|d| d.to_lowercase()).unwrap_or_default().contains(&query.to_lowercase())
            })
            .collect();

        Ok(filtered_models)
    }

    // Get model statistics
    pub async fn get_model_stats(&self) -> Result<ModelStats> {
        let all_models = self.list_local_models().await?;
        let running_models = self.list_running_models().await?;
        
        let total_size: u64 = all_models.iter()
            .map(|m| m.size_bytes)
            .sum();
        
        let models_by_type: std::collections::HashMap<String, usize> = all_models.iter()
            .fold(std::collections::HashMap::new(), |mut acc, model| {
                let model_type = format!("{:?}", model.model_type);
                *acc.entry(model_type).or_insert(0) += 1;
                acc
            });

        Ok(ModelStats {
            total_models: all_models.len(),
            running_models: running_models.len(),
            total_size_bytes: total_size,
            models_by_type,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStats {
    pub total_models: usize,
    pub running_models: usize,
    pub total_size_bytes: u64,
    pub models_by_type: std::collections::HashMap<String, usize>,
}

impl ModelStats {
    pub fn total_size_gb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}