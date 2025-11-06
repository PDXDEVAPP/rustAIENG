# Rust Ollama Enhanced v0.2.0 🚀

A **high-performance, modular LLM inference server** built in Rust with **WebSocket support**, **real-time monitoring**, **model fine-tuning**, and **interactive TUI** - designed as an enhanced drop-in replacement for Ollama.

## 🌟 Enhanced Features

### Core Infrastructure
- **Modular Architecture**: Clean separation with database, inference, model management, API, monitoring modules
- **SQLite Database**: Robust metadata storage with ACID transactions and performance metrics
- **Candle-Powered Inference**: High-performance LLM inference with Hugging Face's Candle framework
- **GGUF Model Support**: Full support for quantized models with intelligent caching

### 🚀 **NEW**: WebSocket & Real-time Features
- **WebSocket Support**: Real-time chat and streaming responses
- **Server-Sent Events**: Enhanced streaming for API responses
- **Live Monitoring**: Real-time connection and performance tracking
- **Interactive Sessions**: Persistent chat sessions with context

### 📊 **NEW**: Advanced Monitoring & Metrics
- **Prometheus Metrics**: Comprehensive performance monitoring
- **Real-time Dashboards**: Interactive TUI for system management
- **Performance Analytics**: Request timing, error rates, throughput analysis
- **Resource Monitoring**: Memory usage, CPU utilization, cache statistics
- **OpenTelemetry Integration**: Distributed tracing and observability

### 🧠 **NEW**: Enhanced Inference Engine
- **Intelligent Caching**: LRU-based model caching with memory pressure detection
- **Batch Processing**: Optimize inference for multiple requests
- **Model Preloading**: Predictive model loading based on usage patterns
- **Enhanced Streaming**: Better response streaming with chunk management
- **Embeddings Generation**: Built-in text embedding capabilities

### 🎯 **NEW**: Model Fine-tuning & Training
- **Custom Training**: Fine-tune models with your own data
- **LoRA Support**: Parameter-efficient fine-tuning
- **Training Analytics**: Loss tracking and validation metrics
- **Model Merging**: Combine base models with fine-tuned adapters

### 🖼️ **NEW**: Interactive TUI
- **Real-time Dashboard**: Monitor system performance and model status
- **Interactive Management**: Start/stop models, view logs, configure settings
- **Performance Visualization**: Charts and gauges for resource usage
- **Model Browser**: Browse and manage local models with details

### ⚡ **NEW**: Advanced Testing & Benchmarking
- **Stress Testing**: Comprehensive load testing and performance evaluation
- **Benchmarking Suite**: Automated performance testing with detailed reports
- **Response Time Analysis**: P50, P95, P99 latency metrics
- **Error Rate Monitoring**: Detailed error tracking and analysis

### 🌐 Enhanced API
- **Ollama Compatibility**: Drop-in replacement with enhanced features
- **Rate Limiting**: Configurable request throttling
- **CORS Support**: Cross-origin resource sharing configuration
- **API Documentation**: Comprehensive OpenAPI specification
- **Batch Endpoints**: Process multiple requests efficiently

## 🚀 Quick Start

### Installation

1. **Clone and build**:
   ```bash
   git clone <repository>
   cd rust_ollama
   cargo build --release
   ```

2. **Start the enhanced server**:
   ```bash
   ./target/release/rust_ollama serve --port 11434 --monitoring --websocket
   ```

3. **Use the enhanced CLI**:
   ```bash
   ./target/release/ollama_cli list
   ./target/release/ollama_cli pull llama3.2
   ./target/release/ollama_cli run llama3.2 "Explain quantum computing"
   ```

4. **Interactive TUI** (NEW!):
   ```bash
   ./target/release/ollama_tui --server-url http://localhost:11434
   ```

5. **View metrics** (NEW!):
   ```bash
   curl http://localhost:9090/metrics
   curl http://localhost:11434/api/metrics
   ```

6. **Run stress tests** (NEW!):
   ```bash
   ./target/release/stress_test --server-url http://localhost:11434 --workers 20 --rps 10
   ```

7. **Fine-tune models** (NEW!):
   ```bash
   ./target/release/model_finetuner --model llama3.2 --data training_data.jsonl --output my_model
   ```

## 📖 Architecture

### Core Components

```
rust_ollama/
├── src/
│   ├── main.rs              # Application entry point
│   ├── core/
│   │   ├── database.rs      # SQLite database layer
│   │   ├── inference_engine.rs # LLM inference with Candle
│   │   └── model_manager.rs # Model lifecycle management
│   ├── api/
│   │   └── server.rs        # REST API server (Axum)
│   └── bin/
│       └── ollama_cli.rs    # Command-line interface
├── Cargo.toml
├── build.rs                 # Build configuration
└── config.toml             # Default configuration
```

### Database Schema

The SQLite database stores:
- **Models**: Metadata, file paths, sizes, parameters
- **Running Models**: Active model instances and statistics
- **Sessions**: User sessions and context history
- **Performance Metrics**: Request timing and resource usage

### API Endpoints

#### Model Management
- `POST /api/list` - List all models
- `POST /api/pull` - Download/pull a model
- `POST /api/delete` - Remove a model
- `POST /api/copy` - Copy a model
- `POST /api/show` - Show model details
- `POST /api/ps` - List running models
- `POST /api/stop` - Stop a model

#### LLM Inference
- `POST /api/generate` - Generate text from prompt
- `POST /api/chat` - Chat completion

#### System
- `GET /api/version` - Server version
- `GET /health` - Health check

## 🛠️ Enhanced CLI Commands

### Core Server Commands
```bash
# Start enhanced server
rust_ollama serve --port 11434 --models-dir ./models --monitoring --websocket
rust_ollama serve --port 11434 --cli  # Use CLI mode
rust_ollama serve --port 11434 --tui  # Use TUI mode
```

### Enhanced Model Management
```bash
ollama_cli list              # List local models with stats
ollama_cli pull llama3.2     # Pull a model
ollama_cli rm llama3.2       # Remove a model
ollama_cli cp llama3.2 custom # Copy a model
ollama_cli show llama3.2     # Show detailed model info
ollama_cli preload llama3.2  # Preload model for faster access
```

### Runtime Operations
```bash
ollama_cli ps                # List running models with metrics
ollama_cli stop llama3.2     # Stop a model
ollama_cli run llama3.2 "Hello" # Run model interactively
ollama_cli chat --model llama3.2 --stream # Enhanced streaming chat
ollama_cli embed --model llama3.2 "text to embed" # Generate embeddings
```

### Enhanced Direct Inference
```bash
ollama_cli generate --model llama3.2 "Explain AI" --format json --stream
ollama_cli generate --model llama3.2 "Generate embeddings" --embeddings-only
ollama_cli batch-generate --prompt-file prompts.txt --model llama3.2
```

### Interactive TUI (NEW!)
```bash
ollama_tui --server-url http://localhost:11434 --refresh-interval 2
```

### Stress Testing & Benchmarking (NEW!)
```bash
stress_test --server-url http://localhost:11434 --workers 20 --rps 10 --duration 60
stress_test --server-url http://localhost:11434 --test-type chat --model mistral
stress_test --server-url http://localhost:11434 --output results.json
```

### Model Fine-tuning (NEW!)
```bash
model_finetuner train --model llama3.2 --data training.jsonl --output fine_tuned
model_finetuner evaluate --model fine_tuned --data test.jsonl
model_finetuner sample --model llama3.2 --prompt "Hello world" --num-samples 5
```

## ⚙️ Configuration

### Configuration File (config.toml)
```toml
[server]
host = "127.0.0.1"
port = 11434

[storage]
database_path = "./ollama.db"
models_directory = "./models"

[inference]
default_temperature = 0.8
default_max_tokens = 512

[performance]
enable_caching = true
gpu_memory_fraction = 0.8
```

### Environment Variables
- `OLLAMA_HOST` - Server host (default: http://localhost:11434)
- `RUST_LOG` - Logging level (default: info)

## 📊 Performance Features

- **Memory Efficient**: Optimized model loading and caching
- **GPU Acceleration**: CUDA and Metal support for faster inference
- **Concurrent Requests**: Handle multiple requests simultaneously
- **Streaming Support**: Real-time response streaming
- **Resource Monitoring**: Track memory usage and performance metrics

## 🔧 Development

### Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# With CUDA support
cargo build --release --features candle-cuda

# With Metal support (macOS)
cargo build --release --features metal
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test database

# Integration tests
cargo test --test integration
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint with clippy
cargo clippy

# Check for security vulnerabilities
cargo audit
```

## 🌐 Model Support

### Supported Formats
- **GGUF**: Primary format with full support
- **Quantization**: Q4_0, Q8_0, and other llama.cpp quantization levels

### Supported Model Families
- LLaMA/LLaMA 2/LLaMA 3
- Mistral/Mixtral
- CodeLLaMA
- Gemma
- Phi
- Custom GGUF models

### Model Lifecycle
1. **Pull**: Download from registry
2. **Load**: Load into memory for inference
3. **Cache**: Keep frequently used models in memory
4. **Unload**: Free memory when not needed
5. **Remove**: Delete from storage

## 🔍 API Examples

### Generate Text
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Explain quantum computing in simple terms",
    "system": "You are a helpful AI assistant",
    "stream": false
  }'
```

### Chat Completion
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you!"},
      {"role": "user", "content": "Can you help me with coding?"}
    ],
    "stream": false
  }'
```

### Model Management
```bash
# List models
curl -X POST http://localhost:11434/api/list

# Pull model
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'

# Delete model
curl -X POST http://localhost:11434/api/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'
```

## 🚧 Roadmap

### Version 0.2.0
- [ ] Complete Candle model loading integration
- [ ] Add multimodal model support (vision)
- [ ] Implement streaming responses
- [ ] Add Docker containerization

### Version 0.3.0
- [ ] WebSocket support for real-time chat
- [ ] Model fine-tuning capabilities
- [ ] Advanced caching strategies
- [ ] Kubernetes deployment manifests

### Version 1.0.0
- [ ] Full Ollama API compatibility
- [ ] Production-ready performance
- [ ] Comprehensive test suite
- [ ] Official documentation site

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ollama](https://github.com/ollama/ollama) for the inspiration and API design
- [Hugging Face](https://github.com/huggingface/candle) for the Candle ML framework
- [Rust Community](https://www.rust-lang.org/) for the excellent ecosystem

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/example/rust_ollama/issues)
- 📖 Documentation: [docs.rs](https://docs.rs/rust_ollama)