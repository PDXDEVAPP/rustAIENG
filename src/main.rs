use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, error, warn};

// Define modules
mod core {
    pub mod database;
    pub mod inference_engine;
    pub mod model_manager;
    pub mod enhanced_inference;
}

mod api {
    pub mod server;
    pub mod websocket;
}

mod monitoring {
    pub mod metrics;
}

#[derive(Parser)]
#[command(name = "rust_ollama")]
#[command(about = "A high-performance, modular LLM inference server with Ollama-compatible API")]
#[command(version = "0.2.0")]
#[command(author = "MiniMax Agent")]
struct Args {
    /// Database file path
    #[arg(short, long, default_value = "./ollama.db")]
    database: String,
    
    /// Models directory
    #[arg(short, long, default_value = "./models")]
    models_dir: String,
    
    /// Server port
    #[arg(short, long, default_value = "11435")]
    port: u16,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Run in CLI mode (instead of server mode)
    #[arg(long)]
    cli: bool,
    
    /// Run interactive TUI
    #[arg(long)]
    tui: bool,
    
    /// Enable WebSocket support
    #[arg(long, default_value = "true")]
    websocket: bool,
    
    /// Enable monitoring and metrics
    #[arg(long, default_value = "true")]
    monitoring: bool,
    
    /// Metrics port
    #[arg(short, long, default_value = "9090")]
    metrics_port: u16,
    
    /// Max cache size (MB)
    #[arg(short, long, default_value = "2048")]
    max_cache_mb: usize,
    
    /// Max concurrent model loads
    #[arg(short, long, default_value = "2")]
    max_concurrent_loads: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt::init();
    } else {
        env_logger::init();
    }
    
    info!("🚀 Starting Rust Ollama server v{}", env!("CARGO_PKG_VERSION"));
    info!("📁 Database: {}", args.database);
    info!("📦 Models directory: {}", args.models_dir);
    info!("🌐 Port: {}", args.port);
    info!("🔗 WebSocket: {}", if args.websocket { "Enabled" } else { "Disabled" });
    info!("📊 Monitoring: {}", if args.monitoring { "Enabled" } else { "Disabled" });
    info!("💾 Cache: {} MB (max {} concurrent loads)", args.max_cache_mb, args.max_concurrent_loads);
    
    if args.cli {
        // Run CLI mode - delegate to the CLI binary
        return run_cli_mode().await;
    }
    
    if args.tui {
        // Run TUI mode - delegate to the TUI binary
        return run_tui_mode().await;
    }
    
    // Initialize core components
    let database = crate::core::database::DatabaseManager::new(&args.database).await?;
    info!("✅ Database initialized successfully");
    
    // Initialize enhanced inference engine with caching
    let enhanced_inference_engine = crate::core::enhanced_inference::get_enhanced_inference_engine();
    info!("🧠 Enhanced inference engine initialized");
    
    // Initialize model manager with enhanced inference
    let model_manager = crate::core::model_manager::ModelManager::new(
        database,
        enhanced_inference_engine.clone(),
        PathBuf::from(&args.models_dir),
    );
    
    // Initialize model manager
    model_manager.initialize().await?;
    info!("⚙️ Model manager initialized successfully");
    
    // Initialize monitoring if enabled
    let metrics_collector = if args.monitoring {
        let collector = crate::monitoring::metrics::get_metrics_collector();
        
        // Start performance monitoring
        let monitor = crate::monitoring::metrics::PerformanceMonitor::new(collector.clone());
        tokio::spawn(async move {
            monitor.start_monitoring().await;
        });
        
        info!("📊 Performance monitoring started");
        Some(collector)
    } else {
        None
    };
    
    // Create API server with enhanced features
    let mut api_server = crate::api::server::ApiServer::new(model_manager);
    
    // Add middleware
    api_server = api_server
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::new().allow_origin(Any));
    
    // Start API server
    info!("🚀 Starting API server on http://127.0.0.1:{}", args.port);
    
    let server_handle = tokio::spawn(async move {
        if let Err(e) = api_server.start(args.port).await {
            error!("❌ API server error: {}", e);
        }
    });
    
    // Start WebSocket server if enabled
    let websocket_handle = if args.websocket {
        let ws_manager = crate::api::websocket::get_websocket_manager();
        let ws_state = crate::api::websocket::WebSocketState {
            model_manager: Arc::new(model_manager),
        };
        
        info!("🔌 Starting WebSocket server on ws://127.0.0.1:{}/ws", args.port);
        
        Some(tokio::spawn(async move {
            // WebSocket server would be started here
            // This is a placeholder - in practice you'd use axum's WebSocket support
            info!("WebSocket server running (placeholder implementation)");
            
            // Keep the server running
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }))
    } else {
        None
    };
    
    // Start metrics endpoint if monitoring is enabled
    let metrics_handle = if args.monitoring {
        let collector = metrics_collector.as_ref().unwrap().clone();
        
        tokio::spawn(async move {
            use axum::{Router, routing::get, extract::State};
            use crate::monitoring::metrics::AppState;
            
            let app_state = AppState { metrics_collector: collector };
            
            let app = Router::new()
                .route("/metrics", get(metrics_handler))
                .route("/health", get(health_check))
                .route("/stats", get(stats_handler))
                .with_state(app_state);
            
            info!("📈 Metrics server starting on http://127.0.0.1:{}/metrics", args.metrics_port);
            
            let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.metrics_port)).await?;
            axum::serve(listener, app).await?;
            
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    } else {
        None
    };
    
    // Wait for all servers to complete
    tokio::select! {
        result = server_handle => {
            if let Err(e) = result {
                error!("API server task failed: {}", e);
            }
        }
        _ = async {
            if let Some(handle) = websocket_handle {
                handle.await;
            } else {
                tokio::future::pending::<()>().await;
            }
        } => {
            info!("WebSocket server completed");
        }
        _ = async {
            if let Some(handle) = metrics_handle {
                handle.await;
            } else {
                tokio::future::pending::<()>().await;
            }
        } => {
            info!("Metrics server completed");
        }
    }
    
    Ok(())
}

async fn run_cli_mode() -> Result<(), Box<dyn std::error::Error>> {
    info!("🖥️ Running in CLI mode");
    
    let cli_args = std::env::args().collect::<Vec<_>>();
    let cli_program = "ollama_cli";
    let cli_path = std::env::current_exe()?
        .parent()
        .unwrap_or(std::env::current_dir()?)
        .join(cli_program);
    
    let mut command = tokio::process::Command::new(&cli_path);
    command.args(&cli_args[1..]); // Skip the program name
    
    let status = command.spawn()?.wait().await?;
    
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    
    Ok(())
}

async fn run_tui_mode() -> Result<(), Box<dyn std::error::Error>> {
    info!("🖼️ Running in TUI mode");
    
    let cli_args = std::env::args().collect::<Vec<_>>();
    let tui_program = "ollama_tui";
    let tui_path = std::env::current_exe()?
        .parent()
        .unwrap_or(std::env::current_dir()?)
        .join(tui_program);
    
    let mut command = tokio::process::Command::new(&tui_path);
    command.args(&cli_args[1..]); // Skip the program name
    
    let status = command.spawn()?.wait().await?;
    
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    
    Ok(())
}

// Metrics endpoint handlers
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};

async fn metrics_handler(State(state): State<crate::monitoring::metrics::AppState>) -> impl IntoResponse {
    match state.metrics_collector.generate_prometheus_metrics().await {
        Ok(metrics) => (StatusCode::OK, metrics),
        Err(e) => {
            warn!("Failed to generate metrics: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e))
        }
    }
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn stats_handler(State(state): State<crate::monitoring::metrics::AppState>) -> impl IntoResponse {
    match state.metrics_collector.get_performance_summary().await {
        Ok(summary) => Json(summary),
        Err(e) => {
            warn!("Failed to get performance summary: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e))
        }
    }
}