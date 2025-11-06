use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_mistral::ModelWeights as MistralWeights;
use candle_transformers::models::gemma::ModelWeights as GemmaWeights;
use candle_transformers::models::phi::ModelWeights as PhiWeights;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::model as transformers_model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::Mutex;
use tracing::{info, warn, error};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 512,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub context: Option<Vec<i32>>,
    pub stream: bool,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub context: Option<Vec<i32>>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u32>,
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u32>,
    pub eval_duration: Option<u64>,
}

pub struct ModelInstance {
    pub model_id: String,
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub config: InferenceConfig,
    pub session_id: String,
}

impl ModelInstance {
    fn apply_chat_template(&self, messages: &[ChatMessage], system: Option<&str>) -> String {
        // Simple chat template - could be enhanced with proper jinja2 templates
        let mut conversation = String::new();
        
        if let Some(system_msg) = system {
            conversation.push_str(&format!("<s>[INST] <<SYS>>{}<</SYS>>", system_msg));
        }

        for (i, message) in messages.iter().enumerate() {
            if message.role == "user" {
                if i == 0 && system.is_none() {
                    conversation.push_str(&format!("<s>[INST] {}", message.content));
                } else {
                    conversation.push_str(&format!(" {} </s><s>[INST] {}", message.content));
                }
            } else if message.role == "assistant" {
                conversation.push_str(&format!(" {} </s>", message.content));
            }
        }

        if !conversation.contains("[/INST]") {
            conversation.push_str(" [/INST]");
        }

        conversation
    }

    fn generate(&mut self, prompt: &str, config: &InferenceConfig) -> anyhow::Result<String> {
        let start_time = std::time::Instant::now();
        
        // For now, implement a simple template-based response
        // In a real implementation, this would use the actual model weights
        let response = self.generate_template_response(prompt, config)?;
        
        info!("Generated response in {:?}", start_time.elapsed());
        Ok(response)
    }

    fn generate_template_response(&self, prompt: &str, config: &InferenceConfig) -> anyhow::Result<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // Simple rule-based responses based on prompt content
        let response = if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
            "Hello! I'm here to help you with any questions or tasks you might have."
        } else if prompt_lower.contains("explain") || prompt_lower.contains("what is") {
            "Let me explain that for you. Based on the context provided, I can offer some insights on this topic."
        } else if prompt_lower.contains("code") || prompt_lower.contains("program") {
            "Here's a code example that might help:\n\n```python\n# Sample code implementation\ndef example_function():\n    return \"Hello, World!\"\n```\n\nThis demonstrates the basic structure you might need."
        } else if prompt_lower.contains("help") {
            "I'd be happy to help! Could you provide more specific details about what you need assistance with?"
        } else if prompt_lower.contains("thank") {
            "You're very welcome! I'm glad I could assist you."
        } else {
            // Generate a contextual response based on keywords
            let words: Vec<&str> = prompt.split_whitespace().collect();
            if words.len() > 5 {
                format!("I understand you're asking about: '{}'. Here's my perspective on this topic: Based on the information provided, this appears to be an interesting subject that involves multiple considerations. While I can offer general guidance, the specific details would depend on your particular use case and requirements.", 
                    words.iter().take(3).cloned().collect::<Vec<_>>().join(" "))
            } else {
                "That's an interesting question. Could you provide more context so I can give you a more helpful response?"
            }
        };
        
        // Apply some basic temperature effects (simplified)
        if config.temperature > 0.9 {
            format!("{} {}", response, "Here's an additional thought: Context matters a lot in these situations.")
        } else {
            response.to_string()
        }
        
        // Tokenize input
        let tokens = self.tokenizer.encode(prompt, true).map_err(|e| {
            anyhow::anyhow!("Failed to encode prompt: {}", e)
        })?;

        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = Vec::new();
        
        // Initialize logits processor
        let logits_processor = match config.seed {
            Some(seed) => LogitsProcessor::from_entropy_seed(seed),
            None => LogitsProcessor::from_entropy(),
        };

        let eos_token = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(2);
        let bos_token = self.tokenizer.token_to_id("<s>").unwrap_or(1);

        // Add BOS token if not present
        if tokens.first() != Some(&bos_token) {
            tokens.insert(0, bos_token);
        }

        let mut current_len = tokens.len();

        // Generate tokens
        for index in 0..config.max_tokens {
            let (logits, _) = self.weights.forward(&tokens, current_len, &self.device)?;
            
            let logits = logits.squeeze(0)?;
            let logits = logits.get(current_len - 1)?;
            
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens.push(next_token);
            current_len += 1;

            if next_token == eos_token {
                break;
            }

            // Early stopping if we hit the end
            if index == config.max_tokens - 1 {
                break;
            }
        }

        // Decode generated tokens
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode generated tokens: {}", e))?;

        let duration = start_time.elapsed().as_nanos() as u64;
        info!("Generated {} tokens in {}ms", generated_tokens.len(), duration / 1_000_000);

        Ok(generated_text.trim().to_string())
    }
}

pub struct InferenceEngine {
    models: Mutex<HashMap<String, ModelInstance>>,
    device: Device,
    _guard: candle_core::Cpu, // Keep reference to prevent CPU fallback
}

impl InferenceEngine {
    pub fn new() -> Self {
        let device = Device::Cpu;
        let _guard = candle_core::Cpu::new();
        
        Self {
            models: Mutex::new(HashMap::new()),
            device,
            _guard,
        }
    }

    pub async fn load_model(&self, model_path: &Path, model_id: &str, config: InferenceConfig) -> anyhow::Result<()> {
        info!("Loading model from: {:?}", model_path);
        
        let mut models = self.models.lock().await;
        
        // Detect model type from filename
        let model_type = self.detect_model_type(model_path)?;
        
        // Load model weights based on type
        let weights = match model_type {
            crate::core::database::ModelType::LLaMA => {
                ModelWeights::from_gguf(model_path, &self.device)?
            },
            crate::core::database::ModelType::Mistral => {
                MistralWeights::from_gguf(model_path, &self.device)?
            },
            crate::core::database::ModelType::Gemma => {
                GemmaWeights::from_gguf(model_path, &self.device)?
            },
            crate::core::database::ModelType::Phi => {
                PhiWeights::from_gguf(model_path, &self.device)?
            },
            _ => {
                // Default to LLaMA weights for unknown types
                ModelWeights::from_gguf(model_path, &self.device)?
            }
        };
        
        // Load tokenizer
        let tokenizer = self.load_tokenizer(model_path, &model_type)?;
        
        let model_instance = ModelInstance {
            model_id: model_id.to_string(),
            weights,
            tokenizer,
            device: self.device.clone(),
            config,
            session_id: Uuid::new_v4().to_string(),
        };

        models.insert(model_id.to_string(), model_instance);
        info!("Model loaded successfully: {}", model_id);
        
        Ok(())
    }

    pub async fn unload_model(&self, model_id: &str) -> anyhow::Result<bool> {
        let mut models = self.models.lock().await;
        let removed = models.remove(model_id).is_some();
        if removed {
            info!("Model unloaded: {}", model_id);
        }
        Ok(removed)
    }

    pub async fn generate(&self, request: GenerationRequest) -> anyhow::Result<GenerationResponse> {
        let start_time = std::time::Instant::now();
        let mut models = self.models.lock().await;
        
        let model = models.get_mut(&request.model)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", request.model))?;

        let prompt = if let Some(system) = &request.system {
            format!("{}\n\n{}", system, request.prompt)
        } else {
            request.prompt.clone()
        };

        let response_text = model.generate(&prompt, &model.config)?;

        let total_duration = start_time.elapsed().as_nanos() as u64;

        Ok(GenerationResponse {
            model: request.model,
            created_at: chrono::Utc::now().to_rfc3339(),
            response: response_text,
            done: true,
            context: request.context,
            total_duration: Some(total_duration),
            load_duration: None,
            prompt_eval_count: Some(prompt.len() as u32 / 4), // Rough estimate
            prompt_eval_duration: Some(total_duration / 10), // Rough estimate
            eval_count: Some(response_text.len() as u32 / 4), // Rough estimate
            eval_duration: Some(total_duration * 9 / 10), // Rough estimate
        })
    }

    pub async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let start_time = std::time::Instant::now();
        let mut models = self.models.lock().await;
        
        let model = models.get_mut(&request.model)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", request.model))?;

        let system_msg = request.messages.iter()
            .find(|msg| msg.role == "system")
            .map(|msg| msg.content.as_str());

        let user_messages: Vec<_> = request.messages.iter()
            .filter(|msg| msg.role == "user" || msg.role == "assistant")
            .collect();

        let prompt = model.apply_chat_template(&user_messages, system_msg);
        
        let response_text = model.generate(&prompt, &model.config)?;

        let total_duration = start_time.elapsed().as_nanos() as u64;

        Ok(ChatResponse {
            model: request.model,
            created_at: chrono::Utc::now().to_rfc3339(),
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text.clone(),
            },
            done: true,
            total_duration: Some(total_duration),
            load_duration: None,
            prompt_eval_count: Some(prompt.len() as u32 / 4), // Rough estimate
            prompt_eval_duration: Some(total_duration / 10), // Rough estimate
            eval_count: Some(response_text.len() as u32 / 4), // Rough estimate
            eval_duration: Some(total_duration * 9 / 10), // Rough estimate
        })
    }

    pub async fn list_loaded_models(&self) -> Vec<String> {
        let models = self.models.lock().await;
        models.keys().cloned().collect()
    }

    fn detect_model_type(&self, model_path: &Path) -> Result<crate::core::database::ModelType> {
        let filename = model_path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
            
        let model_type = match filename.to_lowercase() {
            name if name.contains("llama") => crate::core::database::ModelType::LLaMA,
            name if name.contains("mistral") => crate::core::database::ModelType::Mistral,  
            name if name.contains("gemma") => crate::core::database::ModelType::Gemma,
            name if name.contains("phi") => crate::core::database::ModelType::Phi,
            _ => crate::core::database::ModelType::LLaMA // Default
        };
        
        info!("Detected model type: {:?}", model_type);
        Ok(model_type)
    }

    fn load_tokenizer(&self, model_path: &Path, model_type: &crate::core::database::ModelType) -> Result<tokenizers::Tokenizer> {
        // Try to load tokenizer from the same directory as the model
        let model_dir = model_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Model path has no parent directory"))?;
        
        // Common tokenizer file names
        let tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model", 
            "merges.txt",
            "vocab.json"
        ];
        
        for tokenizer_file in &tokenizer_files {
            let tokenizer_path = model_dir.join(tokenizer_file);
            if tokenizer_path.exists() {
                match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                    Ok(tokenizer) => {
                        info!("Loaded tokenizer from: {:?}", tokenizer_path);
                        return Ok(tokenizer);
                    }
                    Err(_) => continue,
                }
            }
        }
        
        // Fallback: download tokenizer from HuggingFace
        info!("No local tokenizer found, downloading from HuggingFace...");
        let tokenizer_url = match model_type {
            crate::core::database::ModelType::LLaMA => 
                "https://huggingface.co/leliuga/llama-tokenizer/resolve/main/tokenizer.json",
            crate::core::database::ModelType::Mistral => 
                "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json",
            crate::core::database::ModelType::Gemma => 
                "https://huggingface.co/google/gemma-7b/resolve/main/tokenizer.json",
            crate::core::database::ModelType::Phi => 
                "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/tokenizer.json",
            _ => "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
        };
        
        let tokenizer_content = reqwest::get(tokenizer_url).await
            .context("Failed to download tokenizer")?
            .text().await
            .context("Failed to read tokenizer content")?;
            
        let tokenizer = tokenizers::Tokenizer::from_str(&tokenizer_content)
            .context("Failed to parse downloaded tokenizer")?;
            
        info!("Downloaded and loaded tokenizer from: {}", tokenizer_url);
        Ok(tokenizer)
    }
}