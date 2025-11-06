use clap::{Parser, Subcommand, Args};
use reqwest::{Client, header};
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::info;

#[derive(Parser)]
#[command(name = "rust_ollama")]
#[command(about = "A modular Rust-based LLM inference server with Ollama-compatible CLI")]
#[command(version = "0.1.0")]
#[command(author = "MiniMax Agent")]
struct Cli {
    /// Server URL (default: http://localhost:11434)
    #[arg(short, long, env = "OLLAMA_HOST")]
    host: Option<String>,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve the API server
    Serve(ServeArgs),
    
    /// Run a model interactively
    Run(RunArgs),
    
    /// Pull a model from a registry
    Pull(PullArgs),
    
    /// List local models
    List,
    
    /// Remove a model
    Rm(RmArgs),
    
    /// Copy a model
    Cp(CpArgs),
    
    /// Show model information
    Show(ShowArgs),
    
    /// List which models are currently loaded
    Ps,
    
    /// Stop a model which is currently running
    Stop(StopArgs),
    
    /// Generate text from a prompt
    Generate(GenerateArgs),
    
    /// Chat with a model
    Chat(ChatArgs),
}

#[derive(Args)]
struct ServeArgs {
    /// Port to serve on (default: 11434)
    #[arg(short, long, default_value = "11434")]
    port: u16,
    
    /// Models directory
    #[arg(short, long, default_value = "./models")]
    models_dir: String,
    
    /// Database path
    #[arg(short, long, default_value = "./ollama.db")]
    database: String,
}

#[derive(Args)]
struct RunArgs {
    /// Model to run
    model: String,
    
    /// Prompt to run
    #[arg(short, long)]
    prompt: Option<String>,
    
    /// System prompt
    #[arg(short, long)]
    system: Option<String>,
    
    /// Maximum tokens to generate
    #[arg(short, long, default_value = "512")]
    max_tokens: usize,
    
    /// Temperature (0.0 to 2.0)
    #[arg(short, long, default_value = "0.8")]
    temperature: f32,
}

#[derive(Args)]
struct PullArgs {
    /// Model to pull
    model: String,
}

#[derive(Args)]
struct RmArgs {
    /// Model to remove
    model: String,
}

#[derive(Args)]
struct CpArgs {
    /// Source model
    source: String,
    
    /// Destination model
    destination: String,
}

#[derive(Args)]
struct ShowArgs {
    /// Model to show
    model: String,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct StopArgs {
    /// Model to stop
    model: String,
}

#[derive(Args)]
struct GenerateArgs {
    /// Model to use
    model: String,
    
    /// Prompt
    prompt: String,
    
    /// System prompt
    #[arg(short, long)]
    system: Option<String>,
    
    /// Output format (json for JSON output)
    #[arg(short, long)]
    format: Option<String>,
    
    /// Stream output
    #[arg(short, long)]
    stream: bool,
}

#[derive(Args)]
struct ChatArgs {
    /// Model to use
    model: String,
    
    /// Stream output
    #[arg(short, long)]
    stream: bool,
}

impl Cli {
    fn get_host(&self) -> String {
        self.host.as_ref().cloned().unwrap_or_else(|| {
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
        })
    }

    async fn make_api_request<T>(&self, endpoint: &str, payload: Option<serde_json::Value>) -> Result<T, Box<dyn std::error::Error>>
    where
        T: serde::de::DeserializeOwned,
    {
        let client = Client::new();
        let url = format!("{}{}", self.get_host(), endpoint);
        
        let response = if let Some(payload) = payload {
            info!("POST {}: {}", url, payload);
            client.post(&url)
                .json(&payload)
                .header(header::CONTENT_TYPE, "application/json")
                .send()
                .await?
        } else {
            info!("GET {}", url);
            client.get(&url)
                .send()
                .await?
        };

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("API request failed: {} - {}", response.status(), error_text).into());
        }

        let result: T = response.json().await?;
        Ok(result)
    }

    async fn run_serve(&self, args: &ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::database::DatabaseManager;
        use crate::core::inference_engine::InferenceEngine;
        use crate::core::model_manager::ModelManager;
        use crate::api::server::ApiServer;
        use std::path::PathBuf;

        info!("Starting Rust Ollama server...");
        
        let db = DatabaseManager::new(&args.database).await?;
        let inference_engine = InferenceEngine::new();
        let model_manager = ModelManager::new(db, inference_engine, PathBuf::from(&args.models_dir));
        
        model_manager.initialize().await?;
        
        let api_server = ApiServer::new(model_manager);
        api_server.start(args.port).await?;
        
        Ok(())
    }

    async fn run_pull(&self, args: &PullArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "name": args.model
        });
        
        let response: serde_json::Value = self.make_api_request("/api/pull", Some(payload)).await?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        
        Ok(())
    }

    async fn run_list(&self) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Deserialize)]
        struct ListResponse {
            models: Vec<ModelInfo>,
        }
        
        #[derive(serde::Deserialize)]
        struct ModelInfo {
            name: String,
            model: String,
            modified_at: String,
            size: u64,
            digest: String,
        }
        
        let response: ListResponse = self.make_api_request("/api/list", None).await?;
        
        println!("NAME                    ID              SIZE      MODIFIED");
        for model in response.models {
            println!("{}  {}  {}  {}", 
                pad_to_width(&model.name, 20),
                pad_to_width(&model.id, 14),
                format_size(model.size),
                model.modified_at
            );
        }
        
        Ok(())
    }

    async fn run_rm(&self, args: &RmArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "name": args.model
        });
        
        let response: serde_json::Value = self.make_api_request("/api/delete", Some(payload)).await?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        
        Ok(())
    }

    async fn run_cp(&self, args: &CpArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "source": args.source,
            "destination": args.destination
        });
        
        let response: serde_json::Value = self.make_api_request("/api/copy", Some(payload)).await?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        
        Ok(())
    }

    async fn run_show(&self, args: &ShowArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "name": args.model,
            "verbose": args.verbose
        });
        
        let response: serde_json::Value = self.make_api_request("/api/show", Some(payload)).await?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        
        Ok(())
    }

    async fn run_ps(&self) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Deserialize)]
        struct PsResponse {
            models: Vec<ModelInfo>,
        }
        
        #[derive(serde::Deserialize)]
        struct ModelInfo {
            name: String,
            model: String,
        }
        
        let response: PsResponse = self.make_api_request("/api/ps", None).await?;
        
        if response.models.is_empty() {
            println!("No models are currently running");
        } else {
            println!("Running Models:");
            for model in response.models {
                println!("  {}", model.name);
            }
        }
        
        Ok(())
    }

    async fn run_stop(&self, args: &StopArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "name": args.model
        });
        
        let response: serde_json::Value = self.make_api_request("/api/stop", Some(payload)).await?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        
        Ok(())
    }

    async fn run_generate(&self, args: &GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
        let payload = json!({
            "model": args.model,
            "prompt": args.prompt,
            "system": args.system,
            "format": args.format,
            "stream": args.stream
        });
        
        let response: serde_json::Value = self.make_api_request("/api/generate", Some(payload)).await?;
        
        if args.format == Some("json".to_string()) {
            println!("{}", serde_json::to_string_pretty(&response)?);
        } else {
            // Extract the response text
            if let Some(response_text) = response.get("response") {
                println!("{}", response_text);
            }
        }
        
        Ok(())
    }

    async fn run_chat(&self, _args: &ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("Chat mode not fully implemented yet. This would provide an interactive chat interface.");
        Ok(())
    }

    async fn run_run(&self, args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
        // This combines loading and running a model
        // First, ensure the model is loaded
        println!("Loading model: {}", args.model);
        
        // Load the model
        let load_payload = json!({ "name": args.model });
        let _load_response: serde_json::Value = self.make_api_request("/api/stop", Some(load_payload.clone())).await.ok(); // Stop if running
        
        // Generate text
        let generate_args = GenerateArgs {
            model: args.model.clone(),
            prompt: args.prompt.clone().unwrap_or_else(|| {
                println!("Enter prompt (Ctrl+D to finish):");
                read_multiline_input()
            }),
            system: args.system.clone(),
            format: None,
            stream: false,
        };
        
        self.run_generate(&generate_args).await?;
        Ok(())
    }
}

async fn main_async() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Serve(args) => cli.run_serve(args).await?,
        Commands::Pull(args) => cli.run_pull(args).await?,
        Commands::List => cli.run_list().await?,
        Commands::Rm(args) => cli.run_rm(args).await?,
        Commands::Cp(args) => cli.run_cp(args).await?,
        Commands::Show(args) => cli.run_show(args).await?,
        Commands::Ps => cli.run_ps().await?,
        Commands::Stop(args) => cli.run_stop(args).await?,
        Commands::Generate(args) => cli.run_generate(args).await?,
        Commands::Chat(args) => cli.run_chat(args).await?,
        Commands::Run(args) => cli.run_run(args).await?,
    }
    
    Ok(())
}

fn pad_to_width(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.chars().take(width).collect()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

fn format_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = size as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.1}{}", size, UNITS[unit_index])
}

fn read_multiline_input() -> String {
    let mut input = String::new();
    loop {
        let mut buffer = String::new();
        match std::io::stdin().read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                if buffer.trim().is_empty() {
                    break;
                }
                input.push_str(&buffer);
            }
            Err(_) => break,
        }
    }
    input.trim().to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let result = main_async().await;
    
    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}