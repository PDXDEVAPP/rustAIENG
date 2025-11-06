use axum::{
    extract::{State, Path, Query},
    http::{StatusCode, HeaderValue},
    response::{IntoResponse, Json, sse::Event, Sse},
    routing::{get, post, delete},
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
};
use axum::body::Body;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use tracing::{info, warn, error};

use crate::core::model_manager::ModelManager;
use crate::core::inference_engine::{GenerationRequest, ChatRequest, ChatMessage, GenerationResponse, ChatResponse, InferenceConfig};
use crate::core::enhanced_inference::{EnhancedGenerationRequest, EnhancedGenerationResponse, GenerationConfig};
use crate::api::websocket::{WebSocketState, get_websocket_manager};
use crate::monitoring::metrics::{get_metrics_collector, RequestTimer};

#[derive(Clone)]
pub struct AppState {
    pub model_manager: Arc<ModelManager>,
    pub websocket_manager: Arc<get_websocket_manager>,
    pub metrics_collector: Arc<get_metrics_collector>,
}

pub struct ApiServer {
    model_manager: Arc<ModelManager>,
}

impl ApiServer {
    pub fn new(model_manager: ModelManager) -> Self {
        Self { 
            model_manager: Arc::new(model_manager)
        }
    }

    pub async fn create_routes(self) -> Router {
        Router::new()
            // Model Management Endpoints
            .route("/api/list", post(list_models))
            .route("/api/pull", post(pull_model))
            .route("/api/delete", post(delete_model))
            .route("/api/copy", post(copy_model))
            .route("/api/show", post(show_model))
            .route("/api/ps", post(list_running_models))
            .route("/api/stop", post(stop_model))
            
            // Generation Endpoints (Enhanced)
            .route("/api/generate", post(generate_text))
            .route("/api/chat", post(chat_completion))
            .route("/api/generate_stream", post(generate_text_stream))
            .route("/api/chat_stream", post(chat_completion_stream))
            
            // Embeddings Endpoints
            .route("/api/embed", post(generate_embeddings))
            
            // Performance and Monitoring
            .route("/api/metrics", get(get_metrics))
            .route("/api/cache_stats", post(get_cache_stats))
            .route("/api/preload", post(preload_models))
            
            // WebSocket Endpoints
            .route("/ws", get(websocket_handler))
            
            // Health and Info
            .route("/api/tags", get(list_model_tags))
            .route("/api/version", get(get_version))
            .route("/health", get(health_check))
    }

    pub async fn start(self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting Rust Ollama API server on port {}", port);
        
        let app_state = AppState {
            model_manager: self.model_manager,
        };
        
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
        
        axum::serve(listener, self.create_routes().with_state(app_state))
            .await?;
        
        Ok(())
    }
}

// API Request/Response Structures
#[derive(Debug, Deserialize)]
pub struct ListModelsRequest {
    // Ollama compatibility
}

#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    pub name: String,
    pub digest: Option<String>,
    pub insecure: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteModelRequest {
    pub name: String,
    pub insecure: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CopyModelRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Deserialize)]
pub struct ShowModelRequest {
    pub name: String,
    pub verbose: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct StopModelRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub status: String,
    pub message: Option<String>,
    pub data: Option<T>,
}

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
    pub build: String,
}

// API Route Handlers

async fn list_models(
    State(state): State<AppState>,
    Json(_request): Json<ListModelsRequest>,
) -> Result<Json<ListModelsResponse>, (StatusCode, String)> {
    info!("Listing all models");
    
    match state.model_manager.list_local_models().await {
        Ok(models) => {
            let model_infos: Vec<ModelInfo> = models.into_iter().map(|model| {
                ModelInfo {
                    name: model.name.clone(),
                    model: model.name,
                    modified_at: model.updated_at.to_rfc3339(),
                    size: model.size_bytes,
                    digest: format!("sha256:{}", model.id),
                    details: Some(ModelDetails {
                        format: "gguf".to_string(),
                        family: format!("{:?}", model.model_type),
                        families: vec![format!("{:?}", model.model_type)],
                        parameter_size: format!("{}B", model.size_bytes / (1024*1024*1024)),
                        quantization_level: model.quantization.unwrap_or_else(|| "Unknown".to_string()),
                    }),
                }
            }).collect();

            Ok(Json(ListModelsResponse { models: model_infos }))
        },
        Err(e) => {
            error!("Failed to list models: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn pull_model(
    State(state): State<AppState>,
    Json(request): Json<PullModelRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    info!("Pulling model: {}", request.name);
    
    let model_manager = state.model_manager.clone();
    let model_name = request.name.clone();
    
    // Create broadcast channel for progress updates
    let (tx, rx) = broadcast::channel(100);
    let mut rx = BroadcastStream::new(rx);
    
    tokio::spawn(async move {
        let _ = tx.send(Event::from_str("status", "pulling model")).await;
        
        match model_manager.pull_model(&model_name).await {
            Ok(model) => {
                let _ = tx.send(Event::from_str("status", "verifying model")).await;
                let _ = tx.send(Event::from_str("status", "success")).await;
                info!("Model pulled successfully: {}", model_name);
            }
            Err(e) => {
                let _ = tx.send(Event::from_str("error", &e.to_string())).await;
                error!("Failed to pull model {}: {}", model_name, e);
            }
        }
    });

    Ok(Sse::new(rx.map(|result| {
        result.map_err(|_| axum::Error::new("stream error"))
    })).keep_alive())
}

async fn delete_model(
    State(state): State<AppState>,
    Json(request): Json<DeleteModelRequest>,
) -> Result<Json<ApiResponse<()>>, (StatusCode, String)> {
    info!("Deleting model: {}", request.name);
    
    match state.model_manager.remove_model(&request.name).await {
        Ok(success) => {
            if success {
                Ok(Json(ApiResponse {
                    status: "success".to_string(),
                    message: Some("Model deleted successfully".to_string()),
                    data: None,
                }))
            } else {
                Err((StatusCode::NOT_FOUND, "Model not found".to_string()))
            }
        }
        Err(e) => {
            error!("Failed to delete model: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn copy_model(
    State(state): State<AppState>,
    Json(request): Json<CopyModelRequest>,
) -> Result<Json<ApiResponse<()>>, (StatusCode, String)> {
    info!("Copying model from {} to {}", request.source, request.destination);
    
    match state.model_manager.copy_model(&request.source, &request.destination).await {
        Ok(_) => {
            Ok(Json(ApiResponse {
                status: "success".to_string(),
                message: Some("Model copied successfully".to_string()),
                data: None,
            }))
        }
        Err(e) => {
            error!("Failed to copy model: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn show_model(
    State(state): State<AppState>,
    Json(request): Json<ShowModelRequest>,
) -> Result<Json<ApiResponse<ModelInfo>>, (StatusCode, String)> {
    info!("Showing model details: {}", request.name);
    
    match state.model_manager.get_model_details(&request.name).await {
        Ok(Some(model)) => {
            let model_info = ModelInfo {
                name: model.name.clone(),
                model: model.name,
                modified_at: model.updated_at.to_rfc3339(),
                size: model.size_bytes,
                digest: format!("sha256:{}", model.id),
                details: Some(ModelDetails {
                    format: "gguf".to_string(),
                    family: format!("{:?}", model.model_type),
                    families: vec![format!("{:?}", model.model_type)],
                    parameter_size: format!("{}B", model.size_bytes / (1024*1024*1024)),
                    quantization_level: model.quantization.unwrap_or_else(|| "Unknown".to_string()),
                }),
            };
            
            Ok(Json(ApiResponse {
                status: "success".to_string(),
                message: None,
                data: Some(model_info),
            }))
        }
        Ok(None) => Err((StatusCode::NOT_FOUND, "Model not found".to_string())),
        Err(e) => {
            error!("Failed to show model: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn list_running_models(
    State(state): State<AppState>,
    Json(_request): Json<ListModelsRequest>,
) -> Result<Json<ListModelsResponse>, (StatusCode, String)> {
    info!("Listing running models");
    
    match state.model_manager.list_running_models().await {
        Ok(running_models) => {
            let model_infos: Vec<ModelInfo> = running_models.into_iter().map(|model_name| {
                ModelInfo {
                    name: model_name.clone(),
                    model: model_name,
                    modified_at: chrono::Utc::now().to_rfc3339(),
                    size: 0, // Not available for running models
                    digest: format!("sha256:running-{}", model_name),
                    details: None,
                }
            }).collect();

            Ok(Json(ListModelsResponse { models: model_infos }))
        }
        Err(e) => {
            error!("Failed to list running models: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn stop_model(
    State(state): State<AppState>,
    Json(request): Json<StopModelRequest>,
) -> Result<Json<ApiResponse<()>>, (StatusCode, String)> {
    info!("Stopping model: {}", request.name);
    
    match state.model_manager.stop_model(&request.name).await {
        Ok(success) => {
            if success {
                Ok(Json(ApiResponse {
                    status: "success".to_string(),
                    message: Some("Model stopped successfully".to_string()),
                    data: None,
                }))
            } else {
                Err((StatusCode::NOT_FOUND, "Model not found or not running".to_string()))
            }
        }
        Err(e) => {
            error!("Failed to stop model: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn generate_text(
    State(state): State<AppState>,
    Json(request): Json<GenerationRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    info!("Generating text with model: {}", request.model);
    
    let model_manager = state.model_manager.clone();
    
    // For simplicity, we'll handle this without streaming for now
    // In practice, you'd implement proper streaming
    
    match model_manager.generate_text(request).await {
        Ok(response) => {
            let event = Event::from_data(response);
            let stream = tokio_stream::once(Ok(event));
            Ok(Sse::new(stream).keep_alive())
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn chat_completion(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    info!("Chat completion with model: {}", request.model);
    
    let model_manager = state.model_manager.clone();
    
    match model_manager.chat_completion(request).await {
        Ok(response) => {
            let event = Event::from_data(response);
            let stream = tokio_stream::once(Ok(event));
            Ok(Sse::new(stream).keep_alive())
        }
        Err(e) => {
            error!("Chat completion failed: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn list_model_tags() -> Json<HashMap<String, Vec<String>>> {
    // Placeholder implementation
    let tags = vec![
        "latest".to_string(),
        "3.2".to_string(),
        "3.1".to_string(),
    ];
    
    Json(HashMap::from([("latest".to_string(), tags)]))
}

async fn get_version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build: format!("{} {}", chrono::Utc::now().format("%Y-%m-%d"), env!("TARGET")),
    })
}

async fn health_check() -> &'static str {
    "OK"
}

// Extension trait for SSE event creation
trait SseEventExt {
    fn from_str(event_type: &str, data: &str) -> Self;
    fn from_data<T: Serialize>(data: T) -> Self;
}

impl SseEventExt for Event {
    fn from_str(event_type: &str, data: &str) -> Self {
        let mut event = Event::default();
        event = event.event_data(data);
        event = event.event(event_type.to_string());
        event
    }

    fn from_data<T: Serialize>(data: T) -> Self {
        let json_data = serde_json::to_string(&data).unwrap_or_default();
        let mut event = Event::default();
        event = event.event_data(json_data);
        event
    }
}

// Enhanced API Endpoints

async fn generate_text_stream(
    State(state): State<AppState>,
    Json(request): Json<EnhancedGenerationRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    info!("Streaming generation with model: {}", request.model);
    
    let model_manager = state.model_manager.clone();
    let metrics_collector = state.metrics_collector.clone();
    let request_id = uuid::Uuid::new_v4().to_string();
    
    let _timer = RequestTimer::new(
        metrics_collector,
        request_id.clone(),
        request.model.clone(),
        "/api/generate_stream".to_string(),
    );
    
    // Create broadcast channel for progress updates
    let (tx, rx) = broadcast::channel(100);
    let mut rx = BroadcastStream::new(rx);
    
    tokio::spawn(async move {
        let enhanced_request = EnhancedGenerationRequest {
            model: request.model.clone(),
            prompt: request.prompt.clone(),
            system: request.system,
            context: request.context,
            generation_config: request.generation_config.or(Some(GenerationConfig::default())),
            embeddings_only: false,
        };
        
        match crate::core::enhanced_inference::get_enhanced_inference_engine()
            .generate_with_config(enhanced_request)
            .await
        {
            Ok(response) => {
                if response.response.is_empty() {
                    let _ = tx.send(Event::from_str("error", "Empty response")).await;
                } else {
                    // Stream response in chunks
                    let chunks: Vec<&str> = response.response.split_whitespace().collect();
                    for chunk in chunks {
                        let chunk_event = Event::from_data(serde_json::json!({
                            "id": request_id,
                            "model": response.model,
                            "delta": chunk,
                            "done": false
                        }));
                        let _ = tx.send(chunk_event).await;
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                    
                    // Send final event
                    let final_event = Event::from_data(serde_json::json!({
                        "id": request_id,
                        "model": response.model,
                        "response": response.response,
                        "done": true,
                        "total_duration": response.total_duration,
                        "eval_count": response.eval_count
                    }));
                    let _ = tx.send(final_event).await;
                }
            }
            Err(e) => {
                let _ = tx.send(Event::from_str("error", &e.to_string())).await;
            }
        }
    });

    Ok(Sse::new(rx.map(|result| {
        result.map_err(|_| axum::Error::new("stream error"))
    })).keep_alive())
}

async fn chat_completion_stream(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    info!("Streaming chat with model: {}", request.model);
    
    // Convert ChatRequest to EnhancedGenerationRequest
    let enhanced_request = EnhancedGenerationRequest {
        model: request.model,
        prompt: format!("Chat messages: {:?}", request.messages),
        system: None,
        context: None,
        generation_config: Some(GenerationConfig {
            stream: true,
            ..GenerationConfig::default()
        }),
        embeddings_only: false,
    };
    
    // Delegate to enhanced generation endpoint
    generate_text_stream(State(state), Json(enhanced_request)).await
}

async fn generate_embeddings(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    info!("Generating embeddings");
    
    // Parse embedding request
    let model = request.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    let input = request.get("input")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![]);
    
    let enhanced_request = EnhancedGenerationRequest {
        model: model.to_string(),
        prompt: input.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        system: None,
        context: None,
        generation_config: None,
        embeddings_only: true,
    };
    
    match crate::core::enhanced_inference::get_enhanced_inference_engine()
        .generate_with_config(enhanced_request)
        .await
    {
        Ok(response) => {
            if let Some(embeddings) = response.embeddings {
                Ok(Json(serde_json::json!({
                    "model": response.model,
                    "embeddings": embeddings,
                    "total_duration": response.total_duration
                })))
            } else {
                Err((StatusCode::INTERNAL_SERVER_ERROR, "No embeddings generated".to_string()))
            }
        }
        Err(e) => {
            error!("Embeddings generation failed: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn get_metrics(
    State(state): State<AppState>,
) -> Result<String, (StatusCode, String)> {
    info!("Serving metrics");
    
    match state.metrics_collector.generate_prometheus_metrics().await {
        Ok(metrics) => Ok(metrics),
        Err(e) => {
            error!("Failed to generate metrics: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn get_cache_stats(
    State(_state): State<AppState>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    info!("Getting cache statistics");
    
    match crate::core::enhanced_inference::get_enhanced_inference_engine()
        .get_cache_statistics()
        .await
    {
        Ok(stats) => Ok(Json(serde_json::to_value(&stats).unwrap())),
        Err(e) => {
            error!("Failed to get cache stats: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn preload_models(
    State(_state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    info!("Preloading models");
    
    let model_ids = request.get("models")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect::<Vec<_>>();
    
    // Create default model config (this would be more sophisticated in practice)
    let config = crate::core::enhanced_inference::ModelConfig {
        max_sequence_length: 4096,
        vocab_size: 32000,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        pad_token_id: Some(0),
        model_type: crate::core::enhanced_inference::ModelType::LLaMA,
    };
    
    match crate::core::enhanced_inference::get_enhanced_inference_engine()
        .preload_models(model_ids, config)
        .await
    {
        Ok(_) => Ok(Json(serde_json::json!({
            "status": "success",
            "message": "Models preloading initiated"
        }))),
        Err(e) => {
            error!("Failed to preload models: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    info!("WebSocket connection requested");
    
    let ws_state = WebSocketState {
        model_manager: state.model_manager,
    };
    
    match state.websocket_manager.handle_connection(ws, State(ws_state)).await {
        Ok(response) => Ok(response),
        Err(e) => {
            error!("WebSocket connection failed: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

// Extension for ModelManager to add API-specific methods
trait ModelManagerApi {
    async fn generate_text(&self, request: GenerationRequest) -> Result<GenerationResponse>;
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse>;
}

impl ModelManagerApi for ModelManager {
    async fn generate_text(&self, request: GenerationRequest) -> Result<GenerationResponse> {
        // This would interface with the inference engine
        // For now, return a placeholder response
        Ok(GenerationResponse {
            model: request.model,
            created_at: chrono::Utc::now().to_rfc3339(),
            response: "This is a placeholder response. Full inference engine integration needed.".to_string(),
            done: true,
            context: request.context,
            total_duration: Some(1000),
            load_duration: Some(100),
            prompt_eval_count: Some(request.prompt.len() as u32 / 4),
            prompt_eval_duration: Some(100),
            eval_count: Some(20),
            eval_duration: Some(800),
        })
    }

    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        // This would interface with the inference engine
        // For now, return a placeholder response
        Ok(ChatResponse {
            model: request.model,
            created_at: chrono::Utc::now().to_rfc3339(),
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "This is a placeholder chat response. Full inference engine integration needed.".to_string(),
            },
            done: true,
            total_duration: Some(1200),
            load_duration: Some(100),
            prompt_eval_count: Some(50),
            prompt_eval_duration: Some(200),
            eval_count: Some(25),
            eval_duration: Some(900),
        })
    }
}