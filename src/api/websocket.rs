use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures_util::{stream::SplitStream, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio_stream::{Stream, StreamMap};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

use crate::core::model_manager::ModelManager;

#[derive(Clone)]
pub struct WebSocketState {
    pub model_manager: Arc<ModelManager>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "chat")]
    Chat {
        id: String,
        model: String,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    },
    #[serde(rename = "embed")]
    Embed {
        id: String,
        model: String,
        input: Vec<String>,
    },
    #[serde(rename = "generate")]
    Generate {
        id: String,
        model: String,
        prompt: String,
        options: GenerateOptions,
    },
    #[serde(rename = "ping")]
    Ping { id: String },
    #[serde(rename = "pingresp")]
    PingResp { id: String },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatOptions {
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateOptions {
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stop: Option<Vec<String>>,
    pub system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketResponse {
    #[serde(rename = "chat")]
    Chat {
        id: String,
        model: String,
        message: ChatMessage,
        done: bool,
        total_duration: Option<u64>,
        prompt_eval_count: Option<u32>,
        eval_count: Option<u32>,
    },
    #[serde(rename = "chunk")]
    Chunk {
        id: String,
        delta: String,
        model: String,
    },
    #[serde(rename = "embed")]
    Embed {
        id: String,
        embedding: Vec<f32>,
        total_duration: Option<u64>,
    },
    #[serde(rename = "generate")]
    Generate {
        id: String,
        model: String,
        response: String,
        done: bool,
        total_duration: Option<u64>,
        prompt_eval_count: Option<u32>,
        eval_count: Option<u32>,
    },
    #[serde(rename = "error")]
    Error {
        id: String,
        error: String,
    },
    #[serde(rename = "pingresp")]
    PingResp {
        id: String,
    },
    #[serde(rename = "metadata")]
    Metadata {
        version: String,
        models: Vec<String>,
    },
}

pub struct WebSocketManager {
    sessions: dashmap::DashMap<String, WebSocketSession>,
    broadcast_tx: broadcast::Sender<GlobalMessage>,
}

#[derive(Clone)]
pub struct GlobalMessage {
    pub message_type: String,
    pub data: serde_json::Value,
}

impl WebSocketManager {
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        Self {
            sessions: dashmap::DashMap::new(),
            broadcast_tx,
        }
    }

    pub fn get_broadcast_sender(&self) -> broadcast::Sender<GlobalMessage> {
        self.broadcast_tx.clone()
    }

    pub async fn handle_connection(
        &self,
        ws: WebSocketUpgrade,
        state: State<WebSocketState>,
    ) -> Response {
        ws.on_upgrade(move |socket| self.handle_socket(socket, state))
    }

    async fn handle_socket(
        &self,
        socket: WebSocket,
        State(state): State<WebSocketState>,
    ) {
        let session_id = Uuid::new_v4().to_string();
        info!("WebSocket connection established: {}", session_id);

        let (sender, receiver) = socket.split();
        
        // Create session
        let session = WebSocketSession::new(
            session_id.clone(),
            sender,
            state.model_manager.clone(),
        );
        self.sessions.insert(session_id.clone(), session.clone());

        // Send initial metadata
        let _ = session.send_message(WebSocketResponse::Metadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            models: vec!["placeholder".to_string()], // This would be populated from model manager
        }).await;

        // Handle incoming messages
        let (incoming, _) = receiver.split();
        let session_clone = session.clone();
        let sessions = self.sessions.clone();
        
        tokio::spawn(async move {
            if let Err(e) = Self::handle_incoming_messages(
                incoming,
                session_clone,
                sessions,
            ).await {
                warn!("Error handling incoming messages: {}", e);
            }
        });

        // Handle outgoing messages
        let session_clone = session.clone();
        tokio::spawn(async move {
            session_clone.handle_outgoing().await;
        });

        // Keep connection alive
        let session_clone = session.clone();
        tokio::spawn(async move {
            session_clone.keep_alive().await;
        });

        info!("WebSocket session started: {}", session_id);
    }

    async fn handle_incoming_messages(
        mut incoming: SplitStream<WebSocket>,
        session: WebSocketSession,
        sessions: dashmap::DashMap<String, WebSocketSession>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        while let Some(msg) = incoming.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("Received WebSocket message: {}", text);
                    
                    match serde_json::from_str::<WebSocketMessage>(&text) {
                        Ok(wsm) => {
                            if let Err(e) = session.handle_message(wsm).await {
                                error!("Error handling WebSocket message: {}", e);
                                let _ = session.send_error(wsm.get_id(), &e.to_string()).await;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse WebSocket message: {}", e);
                            let _ = session.send_error("unknown", &format!("Parse error: {}", e)).await;
                        }
                    }
                }
                Ok(Message::Binary(data)) => {
                    debug!("Received binary message: {} bytes", data.len());
                    // Handle binary messages if needed
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }
        
        // Clean up session
        sessions.remove(&session.session_id);
        info!("WebSocket session cleaned up: {}", session.session_id);
        
        Ok(())
    }

    pub fn broadcast(&self, message: GlobalMessage) {
        let _ = self.broadcast_tx.send(message);
    }

    pub fn get_session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[derive(Clone)]
pub struct WebSocketSession {
    pub session_id: String,
    sender: mpsc::UnboundedSender<Message>,
    model_manager: Arc<ModelManager>,
}

impl WebSocketSession {
    pub fn new(
        session_id: String,
        sender: futures_util::stream::SplitSink<WebSocket, Message>,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
        
        // Spawn task to forward messages to WebSocket
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if sender.send(msg).await.is_err() {
                    break;
                }
            }
        });

        Self {
            session_id,
            sender: tx,
            model_manager,
        }
    }

    pub async fn send_message(&self, response: WebSocketResponse) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&response)?;
        let message = Message::Text(json);
        
        if self.sender.send(message).is_err() {
            return Err("Failed to send WebSocket message".into());
        }
        
        Ok(())
    }

    pub async fn send_error(&self, id: &str, error: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.send_message(WebSocketResponse::Error {
            id: id.to_string(),
            error: error.to_string(),
        }).await
    }

    pub async fn handle_message(&self, message: WebSocketMessage) -> Result<(), Box<dyn std::error::Error>> {
        match message {
            WebSocketMessage::Ping { id } => {
                self.send_message(WebSocketResponse::PingResp { id }).await
            }
            WebSocketMessage::Chat { id, model, messages, options } => {
                self.handle_chat(id, model, messages, options).await
            }
            WebSocketMessage::Generate { id, model, prompt, options } => {
                self.handle_generate(id, model, prompt, options).await
            }
            WebSocketMessage::Embed { id, model, input } => {
                self.handle_embed(id, model, input).await
            }
            _ => Ok(()),
        }
    }

    async fn handle_chat(
        &self,
        id: String,
        model: String,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Handling chat request: {} (model: {})", id, model);

        // This would integrate with the actual inference engine
        // For now, send a placeholder response
        if options.stream {
            // Send chunked responses
            let chunks = vec!["Hello", " from", " the", " chat!"];
            for chunk in chunks {
                let chunk_msg = WebSocketResponse::Chunk {
                    id: id.clone(),
                    delta: chunk.to_string(),
                    model: model.clone(),
                };
                self.send_message(chunk_msg).await?;
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        let final_msg = WebSocketResponse::Chat {
            id: id.clone(),
            model: model.clone(),
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "This is a placeholder chat response. Full WebSocket integration would use the actual inference engine.".to_string(),
            },
            done: true,
            total_duration: Some(1000),
            prompt_eval_count: Some(50),
            eval_count: Some(25),
        };

        self.send_message(final_msg).await
    }

    async fn handle_generate(
        &self,
        id: String,
        model: String,
        prompt: String,
        options: GenerateOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Handling generate request: {} (model: {})", id, model);

        // This would integrate with the actual inference engine
        let response_text = format!("Generated response for prompt: '{}'", prompt);
        
        if options.stream {
            // Send chunked responses
            let words: Vec<&str> = response_text.split_whitespace().collect();
            for word in words {
                let chunk_msg = WebSocketResponse::Chunk {
                    id: id.clone(),
                    delta: format!(" {}" , word),
                    model: model.clone(),
                };
                self.send_message(chunk_msg).await?;
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        }

        let final_msg = WebSocketResponse::Generate {
            id: id.clone(),
            model: model.clone(),
            response: response_text,
            done: true,
            total_duration: Some(800),
            prompt_eval_count: Some(prompt.len() as u32 / 4),
            eval_count: Some(response_text.len() as u32 / 4),
        };

        self.send_message(final_msg).await
    }

    async fn handle_embed(
        &self,
        id: String,
        model: String,
        input: Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Handling embed request: {} (model: {}, {} inputs)", id, model, input.len());

        // Generate placeholder embedding
        let embedding = vec![0.1; 384]; // Standard embedding dimension
        
        let response = WebSocketResponse::Embed {
            id,
            embedding,
            total_duration: Some(100),
        };

        self.send_message(response).await
    }

    async fn handle_outgoing(&self) {
        // This handles any periodic outgoing messages, heartbeat, etc.
        // For now, just keep the session alive
    }

    async fn keep_alive(&self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            let ping_msg = WebSocketMessage::Ping {
                id: Uuid::new_v4().to_string(),
            };
            
            if let Err(e) = self.handle_message(ping_msg).await {
                warn!("Failed to send ping: {}", e);
                break;
            }
        }
    }
}

impl WebSocketMessage {
    pub fn get_id(&self) -> String {
        match self {
            WebSocketMessage::Chat { id, .. } => id.clone(),
            WebSocketMessage::Embed { id, .. } => id.clone(),
            WebSocketMessage::Generate { id, .. } => id.clone(),
            WebSocketMessage::Ping { id } => id.clone(),
            WebSocketMessage::PingResp { id } => id.clone(),
        }
    }
}

// Global WebSocket manager instance
use once_cell::sync::Lazy;
use std::sync::Arc;

pub static WEBSOCKET_MANAGER: Lazy<Arc<WebSocketManager>> = Lazy::new(|| {
    Arc::new(WebSocketManager::new())
});

pub fn get_websocket_manager() -> Arc<WebSocketManager> {
    WEBSOCKET_MANAGER.clone()
}