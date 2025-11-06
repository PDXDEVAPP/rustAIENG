use clap::{Parser, ValueEnum};
use futures::future::join_all;
use reqwest::{Client, header};
use serde_json::json;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "stress_test")]
#[command(about = "Stress testing tool for Rust Ollama")]
#[command(version = "0.2.0")]
struct Args {
    /// Server URL
    #[arg(short, long, default_value = "http://localhost:11434")]
    server_url: String,
    
    /// Test duration (seconds)
    #[arg(short, long, default_value = "60")]
    duration: u64,
    
    /// Number of concurrent workers
    #[arg(short, long, default_value = "10")]
    workers: usize,
    
    /// Requests per second per worker
    #[arg(short, long, default_value = "5")]
    rps: usize,
    
    /// Test type
    #[arg(short, long, value_enum)]
    test_type: TestType,
    
    /// Model to use for testing
    #[arg(short, long, default_value = "llama3.2")]
    model: String,
    
    /// Output file for results
    #[arg(short, long, default_value = "stress_test_results.json")]
    output: String,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Prompts to use (if applicable)
    #[arg(short, long)]
    prompt: Option<String>,
}

#[derive(ValueEnum, Clone)]
enum TestType {
    Generate,
    Chat,
    Embeddings,
    Mixed,
    ModelList,
    ModelLoad,
}

#[derive(Debug)]
struct StressTestResult {
    test_type: TestType,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_duration: Duration,
    min_response_time: Duration,
    max_response_time: Duration,
    avg_response_time: Duration,
    requests_per_second: f64,
    error_rate: f64,
    response_times: Vec<Duration>,
    error_messages: Vec<String>,
}

impl Default for StressTestResult {
    fn default() -> Self {
        Self {
            test_type: TestType::Mixed,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::from_secs(0),
            min_response_time: Duration::from_secs(999999),
            max_response_time: Duration::from_secs(0),
            avg_response_time: Duration::from_secs(0),
            requests_per_second: 0.0,
            error_rate: 0.0,
            response_times: Vec::new(),
            error_messages: Vec::new(),
        }
    }
}

struct StressTestClient {
    client: Client,
    server_url: String,
    model: String,
}

impl StressTestClient {
    fn new(server_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            server_url,
            model,
        }
    }

    async fn health_check(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let response = self.client
            .get(&format!("{}/health", self.server_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        
        Ok(response.status().is_success())
    }

    async fn test_generation(&self, prompt: &str) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let payload = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.8,
                "max_tokens": 100
            }
        });

        let response = self.client
            .post(&format!("{}/api/generate", self.server_url))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let response_time = start_time.elapsed();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Generation failed: {} - {}", response.status(), error_text).into());
        }

        let _response_data: serde_json::Value = response.json().await?;
        
        Ok(response_time)
    }

    async fn test_chat(&self, messages: Vec<serde_json::Value>) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let payload = json!({
            "model": self.model,
            "messages": messages,
            "stream": false
        });

        let response = self.client
            .post(&format!("{}/api/chat", self.server_url))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let response_time = start_time.elapsed();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Chat failed: {} - {}", response.status(), error_text).into());
        }

        Ok(response_time)
    }

    async fn test_embeddings(&self, texts: Vec<String>) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let payload = json!({
            "model": self.model,
            "input": texts
        });

        let response = self.client
            .post(&format!("{}/api/embed", self.server_url))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let response_time = start_time.elapsed();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Embeddings failed: {} - {}", response.status(), error_text).into());
        }

        Ok(response_time)
    }

    async fn test_model_list(&self) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let response = self.client
            .post(&format!("{}/api/list", self.server_url))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&json!({}))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        let response_time = start_time.elapsed();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Model list failed: {} - {}", response.status(), error_text).into());
        }

        Ok(response_time)
    }

    async fn test_model_load(&self) -> Result<Duration, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let payload = json!({
            "name": self.model
        });

        let response = self.client
            .post(&format!("{}/api/pull", self.server_url))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .timeout(Duration::from_secs(60))
            .send()
            .await?;

        let response_time = start_time.elapsed();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Model load failed: {} - {}", response.status(), error_text).into());
        }

        Ok(response_time)
    }
}

async fn run_stress_test(
    test_type: TestType,
    server_url: String,
    model: String,
    duration: Duration,
    workers: usize,
    rps: usize,
    output_file: String,
    verbose: bool,
    prompt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting stress test");
    info!("Test type: {:?}", test_type);
    info!("Server: {}", server_url);
    info!("Model: {}", model);
    info!("Duration: {:?}", duration);
    info!("Workers: {}", workers);
    info!("RPS per worker: {}", rps);

    // Check server health
    let client = StressTestClient::new(server_url.clone(), model.clone());
    if let Err(e) = client.health_check().await {
        error!("Server health check failed: {}", e);
        return Err(e);
    }

    info!("Server is healthy, starting test...");

    // Set up test coordination
    let (result_tx, mut result_rx) = mpsc::channel::<StressTestResult>(workers);
    let start_time = Instant::now();
    let mut worker_handles = Vec::new();

    // Start workers
    for worker_id in 0..workers {
        let test_client = StressTestClient::new(server_url.clone(), model.clone());
        let test_type_clone = test_type.clone();
        let result_sender = result_tx.clone();
        let prompt = prompt.clone();
        
        let handle = tokio::spawn(async move {
            run_worker_worker(
                worker_id,
                test_client,
                test_type_clone,
                duration,
                rps,
                prompt,
            ).await
        });
        
        worker_handles.push(handle);
    }

    // Wait for all workers to complete
    let results = join_all(worker_handles).await;

    // Aggregate results
    let mut aggregated_result = StressTestResult::default();
    aggregated_result.test_type = test_type;

    for result in results {
        match result {
            Ok(worker_result) => {
                aggregated_result.total_requests += worker_result.total_requests;
                aggregated_result.successful_requests += worker_result.successful_requests;
                aggregated_result.failed_requests += worker_result.failed_requests;
                aggregated_result.response_times.extend(worker_result.response_times);
                aggregated_result.error_messages.extend(worker_result.error_messages);
            }
            Err(e) => error!("Worker failed: {}", e),
        }
    }

    // Calculate aggregate statistics
    aggregated_result.total_duration = start_time.elapsed();
    aggregated_result.requests_per_second = aggregated_result.total_requests as f64 / aggregated_result.total_duration.as_secs_f64();
    aggregated_result.error_rate = (aggregated_result.failed_requests as f64 / aggregated_result.total_requests as f64) * 100.0;

    if !aggregated_result.response_times.is_empty() {
        let total_response_time: Duration = aggregated_result.response_times.iter().sum();
        aggregated_result.avg_response_time = total_response_time / aggregated_result.response_times.len() as u32;
        aggregated_result.min_response_time = *aggregated_result.response_times.iter().min().unwrap();
        aggregated_result.max_response_time = *aggregated_result.response_times.iter().max().unwrap();
    }

    // Print results
    print_stress_test_results(&aggregated_result);

    // Save results to file
    save_results_to_file(&aggregated_result, &output_file).await?;

    Ok(())
}

async fn run_worker_worker(
    worker_id: usize,
    client: StressTestClient,
    test_type: TestType,
    duration: Duration,
    rps: usize,
    prompt: Option<String>,
) -> StressTestResult {
    let mut result = StressTestResult::default();
    result.test_type = test_type;
    
    let request_interval = Duration::from_secs_f64(1.0 / rps as f64);
    let mut last_request_time = Instant::now();

    while last_request_time.elapsed() < duration {
        let request_start = Instant::now();
        
        let (success, response_time, error_msg) = match test_type {
            TestType::Generate => {
                let test_prompt = prompt.as_deref().unwrap_or("Explain the concept of artificial intelligence in simple terms.");
                match client.test_generation(test_prompt).await {
                    Ok(rt) => (true, rt, None),
                    Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                }
            }
            TestType::Chat => {
                let messages = vec![
                    json!({"role": "user", "content": prompt.as_deref().unwrap_or("Hello, how are you?")}),
                ];
                match client.test_chat(messages).await {
                    Ok(rt) => (true, rt, None),
                    Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                }
            }
            TestType::Embeddings => {
                let texts = vec![prompt.as_deref().unwrap_or("This is a test sentence for embeddings.").to_string()];
                match client.test_embeddings(texts).await {
                    Ok(rt) => (true, rt, None),
                    Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                }
            }
            TestType::ModelList => {
                match client.test_model_list().await {
                    Ok(rt) => (true, rt, None),
                    Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                }
            }
            TestType::ModelLoad => {
                match client.test_model_load().await {
                    Ok(rt) => (true, rt, None),
                    Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                }
            }
            TestType::Mixed => {
                // Randomly select test type
                let random_type = match rand::random::<u8>() % 4 {
                    0 => TestType::Generate,
                    1 => TestType::Chat,
                    2 => TestType::Embeddings,
                    _ => TestType::ModelList,
                };
                
                let test_prompt = prompt.as_deref().unwrap_or("Test prompt for mixed workload.");
                
                match random_type {
                    TestType::Generate => match client.test_generation(test_prompt).await {
                        Ok(rt) => (true, rt, None),
                        Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                    },
                    TestType::Chat => {
                        let messages = vec![json!({"role": "user", "content": test_prompt})];
                        match client.test_chat(messages).await {
                            Ok(rt) => (true, rt, None),
                            Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                        }
                    },
                    TestType::Embeddings => {
                        let texts = vec![test_prompt.to_string()];
                        match client.test_embeddings(texts).await {
                            Ok(rt) => (true, rt, None),
                            Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                        }
                    },
                    _ => match client.test_model_list().await {
                        Ok(rt) => (true, rt, None),
                        Err(e) => (false, Duration::from_secs(0), Some(e.to_string())),
                    },
                }
            }
        };

        result.total_requests += 1;
        if success {
            result.successful_requests += 1;
            result.response_times.push(response_time);
        } else {
            result.failed_requests += 1;
            if let Some(msg) = error_msg {
                result.error_messages.push(format!("Worker {}: {}", worker_id, msg));
            }
        }

        // Rate limiting
        let elapsed = request_start.elapsed();
        if elapsed < request_interval {
            sleep(request_interval - elapsed).await;
        }
        
        last_request_time = request_start;
        
        // Progress reporting
        if worker_id == 0 && result.total_requests % 10 == 0 {
            info!("Worker {} completed {} requests", worker_id, result.total_requests);
        }
    }

    result
}

fn print_stress_test_results(result: &StressTestResult) {
    println!("\n🎯 STRESS TEST RESULTS");
    println!("========================");
    println!("Test Type: {:?}", result.test_type);
    println!("Total Duration: {:.2} seconds", result.total_duration.as_secs_f64());
    println!("Total Requests: {}", result.total_requests);
    println!("Successful: {}", result.successful_requests);
    println!("Failed: {}", result.failed_requests);
    println!("Success Rate: {:.2}%", ((result.successful_requests as f64 / result.total_requests as f64) * 100.0));
    println!("Error Rate: {:.2}%", result.error_rate);
    println!("Requests/Second: {:.2}", result.requests_per_second);
    
    if !result.response_times.is_empty() {
        println!("Response Time Statistics:");
        println!("  Average: {:.2}ms", result.avg_response_time.as_secs_f64() * 1000.0);
        println!("  Minimum: {:.2}ms", result.min_response_time.as_secs_f64() * 1000.0);
        println!("  Maximum: {:.2}ms", result.max_response_time.as_secs_f64() * 1000.0);
    }
    
    if !result.error_messages.is_empty() {
        println!("\n❌ Errors encountered:");
        for error in &result.error_messages[..10] { // Show first 10 errors
            println!("  {}", error);
        }
        if result.error_messages.len() > 10 {
            println!("  ... and {} more errors", result.error_messages.len() - 10);
        }
    }
    
    println!("\n📊 Performance Assessment:");
    if result.error_rate < 1.0 {
        println!("  ✅ Excellent - Error rate < 1%");
    } else if result.error_rate < 5.0 {
        println!("  ✅ Good - Error rate < 5%");
    } else if result.error_rate < 10.0 {
        println!("  ⚠️  Fair - Error rate < 10%");
    } else {
        println!("  ❌ Poor - Error rate >= 10%");
    }
    
    if result.requests_per_second > 100.0 {
        println!("  ✅ High throughput - > 100 RPS");
    } else if result.requests_per_second > 10.0 {
        println!("  ✅ Medium throughput - > 10 RPS");
    } else {
        println!("  ⚠️  Low throughput - < 10 RPS");
    }
}

async fn save_results_to_file(result: &StressTestResult, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json_result = serde_json::to_string_pretty(result)?;
    tokio::fs::write(filename, json_result).await?;
    info!("Results saved to {}", filename);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();

    let test_duration = Duration::from_secs(args.duration);
    let output_file = args.output.clone();
    
    run_stress_test(
        args.test_type,
        args.server_url,
        args.model,
        test_duration,
        args.workers,
        args.rps,
        output_file,
        args.verbose,
        args.prompt,
    ).await?;

    Ok(())
}