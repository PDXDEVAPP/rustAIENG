use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, Duration};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, anyhow};
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, debug, error};
use candle_core::{Tensor, Device, DType};
use rand::Rng;

use crate::inference::ModelError;
use crate::models::multimodal_processor::{MultimodalInput, MultimodalOutput};

/// Ensemble strategy for combining multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Simple majority voting or average
    Simple,
    /// Weighted voting based on model performance
    Weighted,
    /// Bayesian model averaging
    Bayesian,
    /// Stacking - use one model to learn how to combine others
    Stacking,
    /// Dynamic selection based on input characteristics
    Dynamic,
}

/// Individual model in an ensemble
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    pub model_id: String,
    pub model: Arc<dyn EnsembleModelInterface>,
    pub weight: f32,
    pub confidence_threshold: f32,
    pub last_used: SystemTime,
    pub success_count: u64,
    pub failure_count: u64,
}

#[async_trait]
pub trait EnsembleModelInterface: Send + Sync {
    async fn generate(&self, input: &MultimodalInput) -> Result<MultimodalOutput, ModelError>;
    fn get_model_info(&self) -> ModelInfo;
    async fn get_performance_metrics(&self) -> ModelPerformanceMetrics;
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_type: String,
    pub capabilities: Vec<String>,
    pub max_tokens: u32,
    pub average_response_time_ms: f64,
    pub success_rate: f64,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub average_confidence: f64,
    pub average_response_time: Duration,
    pub last_updated: SystemTime,
}

/// Model ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub ensemble_id: String,
    pub strategy: EnsembleStrategy,
    pub members: Vec<EnsembleMemberConfig>,
    pub timeout_ms: u64,
    pub min_responses: u32,
    pub confidence_threshold: f32,
    pub enable_failover: bool,
    pub max_retries: u32,
    pub enable_monitoring: bool,
    pub performance_window: u32, // Number of recent requests to consider
}

/// Configuration for ensemble member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMemberConfig {
    pub model_id: String,
    pub weight: f32,
    pub priority: u32,
    pub enabled: bool,
    pub timeout_multiplier: f32,
}

/// Model ensemble for combining multiple models
#[derive(Debug)]
pub struct ModelEnsemble {
    config: EnsembleConfig,
    members: RwLock<HashMap<String, Arc<EnsembleMember>>>,
    active_requests: Mutex<HashMap<String, ActiveRequest>>,
    performance_history: RwLock<PerformanceHistory>,
}

#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    start_time: SystemTime,
    responses: Vec<ModelResponse>,
    completed_members: std::collections::HashSet<String>,
}

#[derive(Debug, Clone)]
struct ModelResponse {
    model_id: String,
    output: MultimodalOutput,
    weight: f32,
    confidence: f32,
    response_time: Duration,
}

#[derive(Debug, Default)]
struct PerformanceHistory {
    requests: VecDeque<RequestRecord>,
    model_stats: HashMap<String, ModelStatistics>,
}

#[derive(Debug, Clone)]
struct RequestRecord {
    timestamp: SystemTime,
    models_used: Vec<String>,
    response_time: Duration,
    success: bool,
    confidence: f32,
}

#[derive(Debug, Clone, Default)]
struct ModelStatistics {
    total_requests: u64,
    successful_requests: u64,
    total_response_time: Duration,
    average_confidence: f32,
    last_updated: SystemTime,
}

impl ModelEnsemble {
    /// Create a new model ensemble
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            members: RwLock::new(HashMap::new()),
            active_requests: Mutex::new(HashMap::new()),
            performance_history: RwLock::new(PerformanceHistory::default()),
        }
    }

    /// Add a model member to the ensemble
    pub async fn add_member(&self, model: Arc<dyn EnsembleModelInterface>, config: EnsembleMemberConfig) -> Result<()> {
        let member = EnsembleMember {
            model_id: config.model_id.clone(),
            model: model.clone(),
            weight: config.weight,
            confidence_threshold: 0.5, // Default threshold
            last_used: SystemTime::now(),
            success_count: 0,
            failure_count: 0,
        };

        let mut members = self.members.write().await;
        members.insert(config.model_id.clone(), Arc::new(member));
        
        info!("Added model member to ensemble: {}", config.model_id);
        Ok(())
    }

    /// Remove a model member from the ensemble
    pub async fn remove_member(&self, model_id: &str) -> Result<()> {
        let mut members = self.members.write().await;
        members.remove(model_id);
        
        info!("Removed model member from ensemble: {}", model_id);
        Ok(())
    }

    /// Generate response using ensemble strategy
    pub async fn generate(&self, input: &MultimodalInput) -> Result<MultimodalOutput> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let start_time = SystemTime::now();
        
        debug!("Starting ensemble generation with request ID: {}", request_id);

        // Select appropriate models based on input
        let selected_models = self.select_models(input).await?;
        debug!("Selected {} models for ensemble", selected_models.len());

        // Set up active request tracking
        {
            let mut active_requests = self.active_requests.lock().await;
            active_requests.insert(request_id.clone(), ActiveRequest {
                request_id: request_id.clone(),
                start_time,
                responses: Vec::new(),
                completed_members: std::collections::HashSet::new(),
            });
        }

        // Generate responses from selected models
        let mut tasks = Vec::new();
        for model_id in &selected_models {
            let model_id = model_id.clone();
            let input = input.clone();
            let request_id = request_id.clone();
            
            let task = tokio::spawn(async move {
                let result = Self::generate_with_model(&model_id, &input).await;
                (model_id, result)
            });
            
            tasks.push(task);
        }

        // Wait for responses with timeout
        let mut responses = Vec::new();
        for task in tasks {
            if let Ok((model_id, result)) = task.await {
                match result {
                    Ok(output) => {
                        // Update active request
                        {
                            let mut active_requests = self.active_requests.lock().await;
                            if let Some(active_request) = active_requests.get_mut(&request_id) {
                                active_request.completed_members.insert(model_id.clone());
                            }
                        }

                        // Calculate confidence and weight
                        let weight = self.calculate_model_weight(&model_id).await;
                        let confidence = output.confidence_score.unwrap_or(0.5);
                        let response_time = start_time.elapsed();

                        responses.push(ModelResponse {
                            model_id,
                            output,
                            weight,
                            confidence,
                            response_time,
                        });

                        debug!("Received response from model: {} (confidence: {:.2})", model_id, confidence);
                    }
                    Err(e) => {
                        warn!("Model {} failed: {}", model_id, e);
                        
                        // Track failure
                        self.record_model_failure(&model_id).await;
                    }
                }
            }
        }

        // Clean up active request
        {
            let mut active_requests = self.active_requests.lock().await;
            active_requests.remove(&request_id);
        }

        // Check if we have enough responses
        if responses.len() < self.config.min_responses as usize {
            return Err(anyhow!("Insufficient responses from ensemble models: {} < {}", 
                             responses.len(), self.config.min_responses));
        }

        // Combine responses using ensemble strategy
        let combined_output = self.combine_responses(&responses, input).await?;

        // Record request statistics
        self.record_request_statistics(&selected_models, start_time.elapsed(), true, 
                                     combined_output.confidence_score.unwrap_or(0.5)).await;

        debug!("Ensemble generation completed in {} ms", start_time.elapsed().as_millis());
        Ok(combined_output)
    }

    /// Generate response with a specific model
    async fn generate_with_model(model_id: &str, input: &MultimodalInput) -> Result<MultimodalOutput, ModelError> {
        // This would be implemented by the specific ensemble implementation
        // For now, return a placeholder error
        Err(ModelError::ModelNotFound(format!("Model {} not found in ensemble", model_id)))
    }

    /// Select appropriate models for the given input
    async fn select_models(&self, input: &MultimodalInput) -> Result<Vec<String>> {
        let members = self.members.read().await;
        let mut eligible_models = Vec::new();

        for (model_id, member) in members.iter() {
            if member.model.get_model_info().capabilities.contains("text") {
                eligible_models.push((model_id.clone(), member.weight));
            }
        }

        // Sort by weight (descending)
        eligible_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top models (limit to reasonable number)
        let max_models = std::cmp::min(eligible_models.len(), 5);
        Ok(eligible_models.into_iter().take(max_models).map(|(id, _)| id).collect())
    }

    /// Calculate model weight based on performance
    async fn calculate_model_weight(&self, model_id: &str) -> f32 {
        let members = self.members.read().await;
        if let Some(member) = members.get(model_id) {
            // Use configured weight, but adjust based on recent performance
            let performance = self.get_model_performance(model_id).await;
            let performance_factor = performance.success_rate as f32;
            
            member.weight * performance_factor
        } else {
            1.0
        }
    }

    /// Combine multiple responses using ensemble strategy
    async fn combine_responses(&self, responses: &[ModelResponse], input: &MultimodalInput) -> Result<MultimodalOutput> {
        match self.config.strategy {
            EnsembleStrategy::Simple => self.combine_simple(responses).await,
            EnsembleStrategy::Weighted => self.combine_weighted(responses).await,
            EnsembleStrategy::Bayesian => self.combine_bayesian(responses).await,
            EnsembleStrategy::Stacking => self.combine_stacking(responses, input).await,
            EnsembleStrategy::Dynamic => self.combine_dynamic(responses, input).await,
        }
    }

    /// Simple combination: majority voting or averaging
    async fn combine_simple(&self, responses: &[ModelResponse]) -> Result<MultimodalOutput> {
        if responses.is_empty() {
            return Err(anyhow!("No responses to combine"));
        }

        // For text responses, use the most confident one
        let best_response = responses
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let total_response_time = responses.iter().map(|r| r.response_time.as_millis()).sum::<u128>() as u64 / responses.len() as u64;

        Ok(MultimodalOutput {
            text: best_response.output.text.clone(),
            tokens_generated: best_response.output.tokens_generated,
            tokens_used: best_response.output.tokens_used,
            confidence_score: Some(best_response.confidence),
            processing_time_ms: total_response_time,
            metadata: {
                let mut meta = best_response.output.metadata.clone();
                meta.insert("ensemble_strategy".to_string(), "simple".to_string());
                meta.insert("models_used".to_string(), responses.len().to_string());
                meta
            },
        })
    }

    /// Weighted combination based on model weights
    async fn combine_weighted(&self, responses: &[ModelResponse]) -> Result<MultimodalOutput> {
        let total_weight: f32 = responses.iter().map(|r| r.weight).sum();
        
        if total_weight == 0.0 {
            return self.combine_simple(responses).await;
        }

        // Weight the responses
        let mut weighted_responses = Vec::new();
        for response in responses {
            let normalized_weight = response.weight / total_weight;
            weighted_responses.push((response, normalized_weight));
        }

        // Use the response with highest weighted confidence
        let best_weighted_response = weighted_responses
            .iter()
            .max_by(|a, b| {
                (a.0.confidence * a.1).partial_cmp(&(b.0.confidence * b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let total_response_time = responses.iter().map(|r| r.response_time.as_millis()).sum::<u128>() as u64 / responses.len() as u64;

        Ok(MultimodalOutput {
            text: best_weighted_response.0.output.text.clone(),
            tokens_generated: best_weighted_response.0.output.tokens_generated,
            tokens_used: best_weighted_response.0.output.tokens_used,
            confidence_score: Some(best_weighted_response.0.confidence * best_weighted_response.1),
            processing_time_ms: total_response_time,
            metadata: {
                let mut meta = best_weighted_response.0.output.metadata.clone();
                meta.insert("ensemble_strategy".to_string(), "weighted".to_string());
                meta.insert("models_used".to_string(), responses.len().to_string());
                meta
            },
        })
    }

    /// Bayesian model averaging (simplified)
    async fn combine_bayesian(&self, responses: &[ModelResponse]) -> Result<MultimodalOutput> {
        // Simplified Bayesian averaging - in practice, this would be more sophisticated
        let confidence_sum: f32 = responses.iter().map(|r| r.confidence).sum();
        let avg_confidence = if !responses.is_empty() {
            confidence_sum / responses.len() as f32
        } else {
            0.0
        };

        // Select response with confidence closest to average
        let best_response = responses
            .iter()
            .min_by(|a, b| {
                ((a.confidence - avg_confidence).abs().partial_cmp(&((b.confidence - avg_confidence).abs()))
                    .unwrap_or(std::cmp::Ordering::Equal))
            })
            .unwrap();

        let total_response_time = responses.iter().map(|r| r.response_time.as_millis()).sum::<u128>() as u64 / responses.len() as u64;

        Ok(MultimodalOutput {
            text: best_response.output.text.clone(),
            tokens_generated: best_response.output.tokens_generated,
            tokens_used: best_response.output.tokens_used,
            confidence_score: Some(avg_confidence),
            processing_time_ms: total_response_time,
            metadata: {
                let mut meta = best_response.output.metadata.clone();
                meta.insert("ensemble_strategy".to_string(), "bayesian".to_string());
                meta.insert("models_used".to_string(), responses.len().to_string());
                meta.insert("bayesian_confidence".to_string(), avg_confidence.to_string());
                meta
            },
        })
    }

    /// Stacking approach - use a meta-model to combine responses
    async fn combine_stacking(&self, responses: &[ModelResponse], input: &MultimodalInput) -> Result<MultimodalOutput> {
        // For now, fall back to weighted approach
        // In a full implementation, you would train a meta-model
        warn!("Stacking ensemble strategy not fully implemented, falling back to weighted");
        self.combine_weighted(responses).await
    }

    /// Dynamic selection based on input characteristics
    async fn combine_dynamic(&self, responses: &[ModelResponse], input: &MultimodalInput) -> Result<MultimodalOutput> {
        // Analyze input characteristics and select best combination strategy
        let input_complexity = self.analyze_input_complexity(input);
        
        match input_complexity {
            InputComplexity::Simple => self.combine_simple(responses).await,
            InputComplexity::Moderate => self.combine_weighted(responses).await,
            InputComplexity::Complex => self.combine_bayesian(responses).await,
        }
    }

    /// Analyze input complexity to determine ensemble strategy
    fn analyze_input_complexity(&self, input: &MultimodalInput) -> InputComplexity {
        let text_length = input.text.len();
        let has_images = !input.images.is_empty();
        
        if text_length < 100 && !has_images {
            InputComplexity::Simple
        } else if text_length < 500 && has_images {
            InputComplexity::Moderate
        } else {
            InputComplexity::Complex
        }
    }

    /// Get model performance metrics
    async fn get_model_performance(&self, model_id: &str) -> ModelPerformanceMetrics {
        let performance_history = self.performance_history.read().await;
        
        if let Some(stats) = performance_history.model_stats.get(model_id) {
            ModelPerformanceMetrics {
                total_requests: stats.total_requests,
                successful_requests: stats.successful_requests,
                average_confidence: stats.average_confidence as f64,
                average_response_time: stats.total_response_time / std::cmp::max(1, stats.total_requests as usize),
                last_updated: stats.last_updated,
            }
        } else {
            ModelPerformanceMetrics {
                total_requests: 0,
                successful_requests: 0,
                average_confidence: 0.0,
                average_response_time: Duration::from_millis(0),
                last_updated: SystemTime::now(),
            }
        }
    }

    /// Record model failure
    async fn record_model_failure(&self, model_id: &str) {
        let mut performance_history = self.performance_history.write().await;
        
        let stats = performance_history.model_stats.entry(model_id.to_string())
            .or_insert_with(ModelStatistics::default);
        
        stats.total_requests += 1;
        stats.last_updated = SystemTime::now();
    }

    /// Record request statistics
    async fn record_request_statistics(&self, models_used: &[String], response_time: Duration, success: bool, confidence: f32) {
        let mut performance_history = self.performance_history.write().await;
        
        // Update overall performance history
        let record = RequestRecord {
            timestamp: SystemTime::now(),
            models_used: models_used.to_vec(),
            response_time,
            success,
            confidence,
        };
        
        performance_history.requests.push_back(record);
        
        // Keep only recent records
        while performance_history.requests.len() > self.config.performance_window as usize {
            performance_history.requests.pop_front();
        }
        
        // Update model statistics
        for model_id in models_used {
            let stats = performance_history.model_stats.entry(model_id.to_string())
                .or_insert_with(ModelStatistics::default);
            
            stats.total_requests += 1;
            if success {
                stats.successful_requests += 1;
            }
            
            // Update running average confidence
            let total_confidence = stats.average_confidence * (stats.total_requests as f32 - 1.0);
            stats.average_confidence = (total_confidence + confidence) / stats.total_requests as f32;
            
            stats.last_updated = SystemTime::now();
        }
    }

    /// Get ensemble statistics
    pub async fn get_statistics(&self) -> EnsembleStatistics {
        let performance_history = self.performance_history.read().await;
        let members = self.members.read().await;
        
        let total_requests = performance_history.requests.len() as u64;
        let successful_requests = performance_history.requests.iter().filter(|r| r.success).count() as u64;
        let average_response_time = if !performance_history.requests.is_empty() {
            let total_time: Duration = performance_history.requests.iter()
                .map(|r| r.response_time)
                .sum();
            total_time / performance_history.requests.len()
        } else {
            Duration::from_millis(0)
        };
        
        EnsembleStatistics {
            ensemble_id: self.config.ensemble_id.clone(),
            strategy: self.config.strategy.clone(),
            total_models: members.len() as u32,
            active_models: members.values().filter(|m| m.model.get_model_info().capabilities.contains("text")).count() as u32,
            total_requests,
            successful_requests,
            success_rate: if total_requests > 0 {
                successful_requests as f64 / total_requests as f64
            } else {
                0.0
            },
            average_response_time_ms: average_response_time.as_millis() as u64,
            model_statistics: members.keys().cloned().collect(),
        }
    }

    /// Update model weights based on recent performance
    pub async fn update_weights(&self) -> Result<()> {
        let mut members = self.members.write().await;
        
        for member in members.values_mut() {
            let performance = self.get_model_performance(&member.model_id).await;
            let success_rate = if performance.total_requests > 0 {
                performance.successful_requests as f32 / performance.total_requests as f32
            } else {
                1.0
            };
            
            // Adjust weight based on success rate
            member.weight = member.weight * (0.5 + 0.5 * success_rate);
            
            debug!("Updated weight for model {}: {}", member.model_id, member.weight);
        }
        
        info!("Updated ensemble model weights based on performance");
        Ok(())
    }
}

/// Input complexity levels
#[derive(Debug, Clone, PartialEq)]
enum InputComplexity {
    Simple,
    Moderate,
    Complex,
}

/// Ensemble statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleStatistics {
    pub ensemble_id: String,
    pub strategy: EnsembleStrategy,
    pub total_models: u32,
    pub active_models: u32,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub success_rate: f64,
    pub average_response_time_ms: u64,
    pub model_statistics: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ensemble_creation() {
        let config = EnsembleConfig {
            ensemble_id: "test-ensemble".to_string(),
            strategy: EnsembleStrategy::Weighted,
            members: vec![],
            timeout_ms: 10000,
            min_responses: 2,
            confidence_threshold: 0.7,
            enable_failover: true,
            max_retries: 3,
            enable_monitoring: true,
            performance_window: 100,
        };

        let ensemble = ModelEnsemble::new(config);
        let stats = ensemble.get_statistics().await;
        
        assert_eq!(stats.ensemble_id, "test-ensemble");
        assert_eq!(stats.total_models, 0);
    }

    #[tokio::test]
    async fn test_input_complexity_analysis() {
        let config = EnsembleConfig {
            ensemble_id: "test".to_string(),
            strategy: EnsembleStrategy::Dynamic,
            members: vec![],
            timeout_ms: 5000,
            min_responses: 1,
            confidence_threshold: 0.5,
            enable_failover: false,
            max_retries: 1,
            enable_monitoring: false,
            performance_window: 50,
        };

        let ensemble = ModelEnsemble::new(config);
        
        // Simple input
        let simple_input = MultimodalInput {
            text: "Short text".to_string(),
            images: vec![],
            max_new_tokens: Some(50),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            do_sample: false,
            stop_sequences: vec![],
            use_cache: true,
        };
        
        let complexity = ensemble.analyze_input_complexity(&simple_input);
        assert_eq!(complexity, InputComplexity::Simple);
    }

    #[test]
    fn test_ensemble_strategy_serialization() {
        let strategies = vec![
            EnsembleStrategy::Simple,
            EnsembleStrategy::Weighted,
            EnsembleStrategy::Bayesian,
            EnsembleStrategy::Stacking,
            EnsembleStrategy::Dynamic,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: EnsembleStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_ensemble_member_config() {
        let config = EnsembleMemberConfig {
            model_id: "test-model".to_string(),
            weight: 1.5,
            priority: 1,
            enabled: true,
            timeout_multiplier: 1.2,
        };

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.weight, 1.5);
        assert!(config.enabled);
    }
}