# Rust Ollama Backend Integration Guide

This guide explains how to integrate the React dashboard with your Rust Ollama backend, including WebSocket setup, API endpoints, and real-time communication.

## 🔌 WebSocket Integration

### **WebSocket Server Setup**

Add WebSocket support to your Rust Axum server:

```rust
// In src/main.rs or src/api/websocket.rs
use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::sync::broadcast;

pub fn websocket_routes() -> Router {
    Router::new().route("/ws", get(websocket_handler))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: AppState) {
    // Handle WebSocket connections
    let (mut sender, mut receiver) = socket.split();
    
    // Send initial data
    let init_data = serde_json::json!({
        "type": "init",
        "message": "Connected to Ollama WebSocket"
    });
    
    if sender.send(axum::extract::ws::Message::Text(init_data.to_string())).await.is_err() {
        return;
    }
    
    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            if msg.is_text() {
                handle_websocket_message(msg.to_text().unwrap(), &state).await;
            }
        } else {
            break;
        }
    }
}

async fn handle_websocket_message(message: &str, state: &AppState) {
    match serde_json::from_str::<WebSocketMessage>(message) {
        Ok(msg) => match msg.event.as_str() {
            "get_metrics" => send_metrics(state).await,
            "get_models" => send_models(state).await,
            "get_system_status" => send_system_status(state).await,
            "chat_request" => handle_chat_request(msg.data, state).await,
            "upload_model" => handle_model_upload(msg.data, state).await,
            "browse_storage" => handle_storage_browse(msg.data, state).await,
            // Add more event handlers as needed
            _ => println!("Unknown event: {}", msg.event),
        },
        Err(e) => println!("Failed to parse WebSocket message: {}", e),
    }
}
```

### **WebSocket Message Structure**

Define message types for type-safe communication:

```rust
// src/api/websocket_types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct WebSocketMessage {
    pub event: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct MetricsUpdate {
    pub active_models: u32,
    pub total_requests: u64,
    pub avg_response_time: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_connections: u32,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: u64,
    pub status: String,
    pub progress: Option<u8>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub message: String,
    pub model: String,
    pub tokens: u32,
    pub response_time: u64,
}
```

## 🌐 HTTP API Endpoints

### **Model Management Endpoints**

```rust
// src/api/models.rs
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, delete},
    Router,
};

pub fn model_routes() -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/models", post(upload_model))
        .route("/models/:id", delete(delete_model))
        .route("/models/:id/download", post(download_model))
}

async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    // Get list of available models
    let models = state.model_manager.list_models().await;
    Json(models)
}

async fn upload_model(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<ModelInfo>, StatusCode> {
    while let Some(field) = multipart.next_field().await {
        let field = field.map_err(|_| StatusCode::BAD_REQUEST)?;
        
        if field.name() == Some("file") {
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
            let filename = field.file_name().unwrap_or("unknown").to_string();
            
            // Process uploaded file
            let model_info = state.model_manager
                .load_model_from_bytes(&data, &filename)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            return Ok(Json(model_info));
        }
    }
    
    Err(StatusCode::BAD_REQUEST)
}
```

### **System Metrics Endpoint**

```rust
// src/api/metrics.rs
use axum::{extract::State, response::Json, routing::get, Router};

pub fn metrics_routes() -> Router {
    Router::new()
        .route("/metrics", get(get_metrics))
        .route("/metrics/realtime", get(get_realtime_metrics))
}

async fn get_metrics(State(state): State<AppState>) -> Json<SystemMetrics> {
    let metrics = state.metrics_collector.get_current_metrics().await;
    Json(metrics)
}

async fn get_realtime_metrics(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| realtime_metrics_stream(socket, state))
}
```

### **Chat Inference Endpoint**

```rust
// src/api/inference.rs
use axum::{
    extract::{State, Json as AxumJson},
    response::Json,
    routing::post,
    Router,
};

pub fn inference_routes() -> Router {
    Router::new()
        .route("/chat", post(handle_chat))
        .route("/chat/stream", post(handle_chat_stream))
}

async fn handle_chat(
    State(state): State<AppState>,
    AxumJson(request): AxumJson<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    let response = state.inference_engine
        .generate_response(&request.message, &request.model)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let response_time = start_time.elapsed().as_millis() as u64;
    
    // Update metrics
    state.metrics_collector.record_request(response_time).await;
    
    Ok(Json(ChatResponse {
        message: response,
        model: request.model,
        tokens: response.len() as u32, // Calculate actual tokens
        response_time,
    }))
}
```

## 📊 Real-time Data Streaming

### **Metrics Streaming**

```rust
// src/metrics/streaming.rs
use tokio::sync::broadcast;
use serde_json::json;

pub struct MetricsBroadcaster {
    sender: broadcast::Sender<serde_json::Value>,
}

impl MetricsBroadcaster {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self { sender }
    }
    
    pub async fn broadcast_metrics(&self, metrics: &MetricsUpdate) {
        let data = json!({
            "type": "metrics_update",
            "data": metrics
        });
        let _ = self.sender.send(data);
    }
    
    pub fn subscribe(&self) -> broadcast::Receiver<serde_json::Value> {
        self.sender.subscribe()
    }
}
```

### **Progress Tracking**

```rust
// src/models/progress.rs
use tokio::sync::broadcast;

pub struct ProgressTracker {
    sender: broadcast::Sender<ModelProgress>,
}

#[derive(Debug, Serialize)]
pub struct ModelProgress {
    pub model_id: String,
    pub progress: u8,
    pub status: String,
    pub message: String,
}

impl ProgressTracker {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self { sender }
    }
    
    pub async fn update_progress(&self, progress: ModelProgress) {
        let _ = self.sender.send(progress);
    }
}
```

## 🔧 Configuration Integration

### **Settings Management**

```rust
// src/config/settings.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub timeout: u64,
    pub log_level: String,
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub max_model_size: u64,
    pub cache_size: u64,
    pub quantization_enabled: bool,
    pub gpu_enabled: bool,
    pub batch_size: u32,
    pub max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub api_key_required: bool,
    pub allowed_ips: Vec<String>,
    pub rate_limit: u32,
    pub jwt_secret: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub prometheus_port: u16,
    pub tracing_enabled: bool,
    pub log_retention: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub models: ModelConfig,
    pub security: SecurityConfig,
    pub monitoring: MonitoringConfig,
}
```

### **Configuration Endpoints**

```rust
// src/api/config.rs
use axum::{extract::State, response::Json, routing::{get, post}, Router};

pub fn config_routes() -> Router {
    Router::new()
        .route("/config", get(get_config))
        .route("/config", post(save_config))
}

async fn get_config(State(state): State<AppState>) -> Json<AppConfig> {
    Json(state.config.clone())
}

async fn save_config(
    State(state): State<AppState>,
    AxumJson(config): AxumJson<AppConfig>,
) -> Result<Json<()>, StatusCode> {
    state.config_manager.save_config(config)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(()))
}
```

## 🗄️ Cloud Storage Integration

### **Storage Service Implementation**

```rust
// src/storage/cloud_service.rs
use async_trait::async_trait;

#[async_trait]
pub trait CloudStorage: Send + Sync {
    async fn upload_file(&self, path: &str, data: &[u8]) -> Result<String, StorageError>;
    async fn download_file(&self, path: &str) -> Result<Vec<u8>, StorageError>;
    async fn list_files(&self, prefix: &str) -> Result<Vec<StorageFile>, StorageError>;
    async fn delete_file(&self, path: &str) -> Result<(), StorageError>;
}

pub struct S3Storage {
    client: aws_sdk_s3::Client,
    bucket: String,
}

pub struct AzureBlobStorage {
    client: azure_storage_blobs::Client,
    container: String,
}

#[async_trait]
impl CloudStorage for S3Storage {
    async fn upload_file(&self, path: &str, data: &[u8]) -> Result<String, StorageError> {
        let result = self.client
            .put_object()
            .bucket(&self.bucket)
            .key(path)
            .body(data.into())
            .send()
            .await
            .map_err(|e| StorageError::UploadFailed(e.to_string()))?;
            
        Ok(format!("s3://{}/{}", self.bucket, path))
    }
    
    // Implement other methods...
}
```

## 🚀 Getting Started

### **1. Add Dependencies**

Add to your `Cargo.toml`:

```toml
[dependencies]
axum = { version = "0.7", features = ["json", "ws"] }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
aws-sdk-s3 = "1.0"
azure-storage-blobs = "0.17"
```

### **2. Setup WebSocket Handler**

```rust
// In your main.rs
use axum::Router;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .nest("/api", api_routes())
        .nest("/ws", websocket_routes());
    
    axum::Server::bind(&"0.0.0.0:11435".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### **3. Connect Dashboard**

The React dashboard automatically connects to:
- **WebSocket**: `ws://localhost:11435/ws`
- **HTTP API**: `http://localhost:11435/api`

Update the WebSocket connection URL in `src/contexts/WebSocketContext.tsx` if needed:

```typescript
const newSocket = io('ws://YOUR_HOST:YOUR_PORT', {
  path: '/ws',
  transports: ['websocket'],
})
```

## 📝 API Documentation

### **WebSocket Events**

**Client → Server:**
- `get_metrics` - Request system metrics
- `get_models` - Request model list
- `chat_request` - Send chat message
- `upload_model` - Upload model file
- `browse_storage` - List cloud storage files

**Server → Client:**
- `metrics_update` - System metrics update
- `models_list` - Model list response
- `chat_response` - Chat message response
- `model_progress` - Upload progress update
- `storage_files` - Cloud storage file list

### **HTTP Endpoints**

- `GET /api/models` - List models
- `POST /api/models` - Upload model
- `DELETE /api/models/:id` - Delete model
- `GET /api/metrics` - Get metrics
- `POST /api/chat` - Chat inference
- `GET /api/config` - Get configuration
- `POST /api/config` - Save configuration

## 🔍 Testing

Test the integration:

```bash
# Start Rust backend
cargo run

# In another terminal, start dashboard
cd dashboard && npm run dev

# Visit http://localhost:3000
```

The dashboard should connect to your Rust backend and display real-time data!