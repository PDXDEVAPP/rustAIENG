use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, anyhow};
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use image::{DynamicImage, ImageFormat};
use candle_core::{Tensor, Device, DType};

use crate::inference::ModelError;
use crate::models::vision_models::{VisionModel, VisionCapability};

/// Multimodal input combining text and images
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: String,
    pub images: Vec<DynamicImage>,
    pub max_new_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub do_sample: bool,
    pub stop_sequences: Vec<String>,
    pub use_cache: bool,
}

/// Multimodal output containing generated text and metadata
#[derive(Debug, Clone)]
pub struct MultimodalOutput {
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_used: u32,
    pub confidence_score: Option<f32>,
    pub processing_time_ms: u64,
    pub metadata: HashMap<String, String>,
}

/// Processing result from multimodal model
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub success: bool,
    pub output: MultimodalOutput,
    pub processing_time_ms: u64,
    pub token_count: u32,
    pub metadata: HashMap<String, String>,
}

/// Multimodal processor for handling various input types
#[derive(Debug)]
pub struct MultimodalProcessor {
    supported_modalities: Vec<ModalityType>,
    default_model: Option<String>,
    models: RwLock<HashMap<String, Arc<VisionModel>>>,
    processing_stats: RwLock<ProcessingStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Document,
    Table,
    Code,
}

/// Processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_processing_time_ms: f64,
    pub modality_counts: HashMap<String, u64>,
    pub last_request_time: Option<SystemTime>,
}

impl MultimodalProcessor {
    /// Create a new multimodal processor
    pub fn new() -> Self {
        Self {
            supported_modalities: vec![
                ModalityType::Text,
                ModalityType::Image,
            ],
            default_model: None,
            models: RwLock::new(HashMap::new()),
            processing_stats: RwLock::new(ProcessingStatistics::default()),
        }
    }

    /// Register a vision model with the processor
    pub async fn register_model(&self, model_id: &str, model: Arc<VisionModel>) -> Result<()> {
        let mut models = self.models.write().await;
        models.insert(model_id.to_string(), model);
        
        if self.default_model.is_none() {
            self.default_model = Some(model_id.to_string());
        }
        
        info!("Registered multimodal model: {}", model_id);
        Ok(())
    }

    /// Process multimodal input
    pub async fn process(&self, input: &MultimodalInput) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Update statistics
        {
            let mut stats = self.processing_stats.write().await;
            stats.total_requests += 1;
            stats.last_request_time = Some(SystemTime::now());
            
            // Count modalities
            if !input.text.is_empty() {
                *stats.modality_counts.entry("text".to_string()).or_insert(0) += 1;
            }
            stats.modality_counts.insert("image".to_string(), input.images.len() as u64);
        }

        // Determine appropriate model
        let model = self.select_appropriate_model(input).await?;
        
        debug!("Processing multimodal input with model: {}", model.get_config().model_id);

        let result = match model.generate(input).await {
            Ok(output) => {
                let processing_time = start_time.elapsed();
                
                // Update statistics for successful request
                {
                    let mut stats = self.processing_stats.write().await;
                    stats.successful_requests += 1;
                    stats.average_processing_time_ms = self.calculate_average_time(&stats, processing_time);
                }

                ProcessingResult {
                    success: true,
                    output,
                    processing_time_ms: processing_time.as_millis() as u64,
                    token_count: output.tokens_generated,
                    metadata: HashMap::new(),
                }
            }
            Err(e) => {
                error!("Multimodal processing failed: {}", e);
                
                // Update statistics for failed request
                {
                    let mut stats = self.processing_stats.write().await;
                    stats.failed_requests += 1;
                }

                ProcessingResult {
                    success: false,
                    output: MultimodalOutput {
                        text: "Error: Failed to process multimodal input".to_string(),
                        tokens_generated: 0,
                        tokens_used: 0,
                        confidence_score: None,
                        processing_time_ms: start_time.elapsed().as_millis() as u64,
                        metadata: HashMap::new(),
                    },
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    token_count: 0,
                    metadata: HashMap::new(),
                }
            }
        };

        debug!("Processed multimodal input in {} ms", result.processing_time_ms);
        Ok(result)
    }

    /// Select the most appropriate model for the given input
    async fn select_appropriate_model(&self, input: &MultimodalInput) -> Result<Arc<VisionModel>> {
        // Determine required capabilities based on input
        let required_capabilities = self.determine_required_capabilities(input);
        
        let models = self.models.read().await;
        
        // Find model that supports required capabilities
        for (model_id, model) in models.iter() {
            let supports_all = required_capabilities.iter().all(|cap| model.supports_capability(cap));
            if supports_all {
                debug!("Selected model {} based on capability requirements", model_id);
                return Ok(model.clone());
            }
        }
        
        // If no perfect match, use default model
        if let Some(ref default_id) = self.default_model {
            if let Some(model) = models.get(default_id) {
                warn!("Using default model {} as no perfect capability match found", default_id);
                return Ok(model.clone());
            }
        }
        
        Err(anyhow!("No suitable multimodal model found for the given input"))
    }

    /// Determine required capabilities based on input
    fn determine_required_capabilities(&self, input: &MultimodalInput) -> Vec<VisionCapability> {
        let mut capabilities = Vec::new();
        
        if !input.text.is_empty() && !input.images.is_empty() {
            // Image + text input - need vision-to-text capability
            capabilities.push(VisionCapability::ImageToText);
        } else if !input.images.is_empty() {
            // Image only input
            capabilities.push(VisionCapability::ImageCaptioning);
        }
        
        // Add other capability detection based on text content
        if input.text.contains("analyze") || input.text.contains("explain") {
            capabilities.push(VisionCapability::VisualReasoning);
        }
        
        capabilities
    }

    /// Calculate running average of processing time
    fn calculate_average_time(&self, stats: &ProcessingStatistics, new_time: Duration) -> f64 {
        let total_time = stats.average_processing_time_ms * (stats.total_requests as f64 - 1.0);
        (total_time + new_time.as_millis() as f64) / stats.total_requests as f64
    }

    /// Get supported modalities
    pub fn get_supported_modalities(&self) -> &[ModalityType] {
        &self.supported_modalities
    }

    /// Get processing statistics
    pub async fn get_statistics(&self) -> ProcessingStatistics {
        self.processing_stats.read().await.clone()
    }

    /// Get list of registered models
    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    /// Check if processor supports a specific modality
    pub fn supports_modality(&self, modality: &ModalityType) -> bool {
        self.supported_modalities.contains(modality)
    }

    /// Set default model
    pub async fn set_default_model(&self, model_id: &str) -> Result<()> {
        let models = self.models.read().await;
        if models.contains_key(model_id) {
            self.default_model = Some(model_id.to_string());
            info!("Set default model to: {}", model_id);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Unregister a model
    pub async fn unregister_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        models.remove(model_id);
        
        if self.default_model.as_ref() == Some(&model_id.to_string()) {
            self.default_model = models.keys().next().map(|s| s.clone());
        }
        
        info!("Unregistered multimodal model: {}", model_id);
        Ok(())
    }

    /// Validate multimodal input
    pub fn validate_input(&self, input: &MultimodalInput) -> Result<()> {
        if input.text.is_empty() && input.images.is_empty() {
            return Err(anyhow!("Input must contain either text or images"));
        }
        
        if input.images.len() > 10 {
            return Err(anyhow!("Maximum 10 images supported per request"));
        }
        
        if input.temperature.is_some() && (input.temperature.unwrap() < 0.0 || input.temperature.unwrap() > 2.0) {
            return Err(anyhow!("Temperature must be between 0.0 and 2.0"));
        }
        
        if input.top_p.is_some() && (input.top_p.unwrap() < 0.0 || input.top_p.unwrap() > 1.0) {
            return Err(anyhow!("Top-p must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }

    /// Extract text from various input formats
    pub async fn extract_text(&self, input: &str, format: &TextFormat) -> Result<String> {
        match format {
            TextFormat::Plain => Ok(input.to_string()),
            TextFormat::Markdown => {
                // Basic markdown processing
                Ok(input.to_string())
            }
            TextFormat::HTML => {
                // Basic HTML text extraction (simplified)
                // In a real implementation, you'd use a proper HTML parser
                Ok(input.to_string())
            }
            TextFormat::JSON => {
                // Parse JSON and extract text fields
                let value: serde_json::Value = serde_json::from_str(input)?;
                self.extract_text_from_json(&value)
            }
        }
    }

    /// Extract text from JSON structure
    fn extract_text_from_json(&self, value: &serde_json::Value) -> Result<String> {
        match value {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Array(arr) => {
                let mut texts = Vec::new();
                for item in arr {
                    texts.push(self.extract_text_from_json(item)?);
                }
                Ok(texts.join(" "))
            }
            serde_json::Value::Object(obj) => {
                let mut texts = Vec::new();
                for (_key, val) in obj {
                    texts.push(self.extract_text_from_json(val)?);
                }
                Ok(texts.join(" "))
            }
            _ => Ok(value.to_string()),
        }
    }

    /// Process document input (PDF, DOCX, etc.)
    pub async fn process_document(&self, document_path: &str) -> Result<String> {
        // This would implement document processing for various formats
        warn!("Document processing not fully implemented");
        
        // For now, return a placeholder
        Ok("Document processing not available".to_string())
    }

    /// Create a batch processing request
    pub async fn create_batch_request(&self, inputs: Vec<MultimodalInput>) -> Result<BatchProcessingRequest> {
        // Validate all inputs
        for input in &inputs {
            self.validate_input(input)?;
        }
        
        let request_id = uuid::Uuid::new_v4().to_string();
        
        Ok(BatchProcessingRequest {
            request_id,
            inputs,
            created_at: SystemTime::now(),
            priority: BatchPriority::Normal,
        })
    }
}

/// Text format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextFormat {
    Plain,
    Markdown,
    HTML,
    JSON,
}

/// Batch processing request
#[derive(Debug, Clone)]
pub struct BatchProcessingRequest {
    pub request_id: String,
    pub inputs: Vec<MultimodalInput>,
    pub created_at: SystemTime,
    pub priority: BatchPriority,
}

/// Batch processing priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Helper functions for multimodal processing
pub mod helpers {
    use super::*;
    use image::{ImageBuffer, Rgba};

    /// Create a placeholder image
    pub fn create_placeholder_image(width: u32, height: u32) -> DynamicImage {
        let mut buffer = ImageBuffer::new(width, height);
        
        // Fill with a simple pattern
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let value = if (x + y) % 20 < 10 {
                [255, 255, 255, 255] // White
            } else {
                [0, 0, 0, 255] // Black
            };
            *pixel = Rgba(value);
        }
        
        DynamicImage::ImageRgba8(buffer)
    }

    /// Convert tensor to image
    pub fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage, ModelError> {
        let shape = tensor.dims();
        if shape.len() != 3 || shape[0] != 3 {
            return Err(ModelError::InvalidInput("Tensor must be in CHW format with 3 channels".to_string()));
        }
        
        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        
        // Convert tensor to bytes
        let data = tensor.to_vec::<f32>().map_err(|_| {
            ModelError::InvalidInput("Failed to convert tensor to f32".to_string())
        })?;
        
        let mut buffer = ImageBuffer::new(width as u32, height as u32);
        
        for y in 0..height {
            for x in 0..width {
                let r_idx = y * width + x;
                let g_idx = height * width + y * width + x;
                let b_idx = 2 * height * width + y * width + x;
                
                let r = (data[r_idx] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (data[g_idx] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (data[b_idx] * 255.0).clamp(0.0, 255.0) as u8;
                
                buffer.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, 255]));
            }
        }
        
        Ok(DynamicImage::ImageRgba8(buffer))
    }

    /// Resize image to target dimensions
    pub fn resize_image(image: &DynamicImage, target_width: u32, target_height: u32) -> DynamicImage {
        image.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3)
    }

    /// Normalize image pixel values
    pub fn normalize_image(image: &DynamicImage) -> Result<Tensor, ModelError> {
        let (width, height) = image.dimensions();
        let rgb_image = image.to_rgb8();
        let pixels = rgb_image.into_raw();
        
        let mut tensor_data = Vec::new();
        
        // Convert to CHW format and normalize to [0, 1]
        for c in 0..3 {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let idx = (y * width as usize + x) * 3 + c;
                    if idx < pixels.len() {
                        let normalized = pixels[idx] as f32 / 255.0;
                        tensor_data.push(normalized);
                    }
                }
            }
        }
        
        Tensor::from_slice(&tensor_data, &[3, height as usize, width as usize], &Device::Cpu)
            .map_err(|e| ModelError::TensorCreationFailed(e.to_string()))
    }
}

impl Default for MultimodalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_multimodal_processor_creation() {
        let processor = MultimodalProcessor::new();
        
        assert_eq!(processor.get_supported_modalities().len(), 2);
        assert!(processor.supports_modality(&ModalityType::Text));
        assert!(processor.supports_modality(&ModalityType::Image));
        assert!(!processor.supports_modality(&ModalityType::Audio));
    }

    #[tokio::test]
    async fn test_input_validation() {
        let processor = MultimodalProcessor::new();
        
        // Valid input
        let valid_input = MultimodalInput {
            text: "Test text".to_string(),
            images: vec![],
            max_new_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            repetition_penalty: None,
            do_sample: true,
            stop_sequences: vec![],
            use_cache: true,
        };
        
        assert!(processor.validate_input(&valid_input).is_ok());
        
        // Invalid input - empty text and no images
        let invalid_input = MultimodalInput {
            text: "".to_string(),
            images: vec![],
            max_new_tokens: Some(100),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            do_sample: false,
            stop_sequences: vec![],
            use_cache: true,
        };
        
        assert!(processor.validate_input(&invalid_input).is_err());
        
        // Invalid input - temperature out of range
        let invalid_temp_input = MultimodalInput {
            text: "Test text".to_string(),
            images: vec![],
            max_new_tokens: Some(100),
            temperature: Some(3.0), // Out of range
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            do_sample: false,
            stop_sequences: vec![],
            use_cache: true,
        };
        
        assert!(processor.validate_input(&invalid_temp_input).is_err());
    }

    #[tokio::test]
    async fn test_text_extraction() {
        let processor = MultimodalProcessor::new();
        
        // Test plain text
        let plain_text = "Hello, world!";
        let extracted = processor.extract_text(plain_text, &TextFormat::Plain).await.unwrap();
        assert_eq!(extracted, plain_text);
        
        // Test JSON extraction
        let json_text = r#"{"content": "Hello from JSON", "metadata": "test"}"#;
        let extracted_json = processor.extract_text(json_text, &TextFormat::JSON).await.unwrap();
        assert!(extracted_json.contains("Hello from JSON"));
    }

    #[test]
    fn test_multimodal_input_creation() {
        let input = MultimodalInput {
            text: "Test image analysis".to_string(),
            images: vec![],
            max_new_tokens: Some(150),
            temperature: Some(0.8),
            top_p: None,
            top_k: Some(50),
            repetition_penalty: Some(1.1),
            do_sample: true,
            stop_sequences: vec!["<end>".to_string()],
            use_cache: false,
        };
        
        assert_eq!(input.text, "Test image analysis");
        assert_eq!(input.max_new_tokens, Some(150));
        assert!(input.do_sample);
    }

    #[test]
    fn test_processing_result() {
        let output = MultimodalOutput {
            text: "Generated response".to_string(),
            tokens_generated: 50,
            tokens_used: 75,
            confidence_score: Some(0.85),
            processing_time_ms: 250,
            metadata: HashMap::new(),
        };
        
        let result = ProcessingResult {
            success: true,
            output: output.clone(),
            processing_time_ms: 250,
            token_count: 50,
            metadata: HashMap::new(),
        };
        
        assert!(result.success);
        assert_eq!(result.output.text, "Generated response");
        assert_eq!(result.token_count, 50);
    }
}