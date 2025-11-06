# 🚀 Rust Ollama Development Setup Guide

This guide provides multiple ways to set up and run your Rust Ollama development environment with all dependencies up to date.

## 📋 Prerequisites

Make sure you have the following installed:

- **Rust/Cargo**: Install via [rustup.rs](https://rustup.rs/)
- **Node.js 18+**: Install from [nodejs.org](https://nodejs.org/)
- **npm**: Usually comes with Node.js
- **Git**: For version control
- **SQLite3**: For the database

## 🛠️ Setup Options

### Option 1: Complete Setup (Recommended)

**Run the comprehensive setup script:**
```bash
./dev-setup-complete.sh
```

This script will:
- ✅ Install Rust and Node.js if missing
- ✅ Create development directories and configuration
- ✅ Update all dependencies to latest versions
- ✅ Build both backend and frontend
- ✅ Test the setup automatically
- ✅ Provide startup scripts

### Option 2: Detailed Setup

**Run the comprehensive development setup:**
```bash
./dev-setup-comprehensive.sh
```

This provides detailed step-by-step setup with:
- ✅ Dependency checking and updates
- ✅ Development environment configuration
- ✅ Testing utilities and monitoring scripts
- ✅ Hot reload setup for both frontend and backend

### Option 3: Quick Setup

**Run the simple setup:**
```bash
./dev-setup-simple.sh
```

For minimal setup with basic configuration.

## 🚀 Running the Development Environment

### One-Click Start

**Start everything at once:**
```bash
./dev-run.sh
```

This will:
- ✅ Build the project if needed
- ✅ Start the backend server on port 11435
- ✅ Test model download functionality
- ✅ Test API endpoints
- ✅ Optionally start the frontend dashboard
- ✅ Provide system monitoring

### Manual Start

**Start backend only:**
```bash
# Set environment
export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
export RUST_LOG=debug
source ~/.cargo/env

# Start server
cargo run --release -- serve --port 11435
```

**Start frontend only:**
```bash
cd dashboard
npm run dev
```

## 🧪 Testing the Enhanced Backend

The enhanced backend includes real model downloading and inference capabilities:

### Test Model Download

```bash
# Download LLaMA 3.2 model
curl -X POST http://localhost:11435/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'

# Download Mistral model  
curl -X POST http://localhost:11435/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "mistral"}'
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:11435/api/health

# List models
curl http://localhost:11435/api/models

# Generate text
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Hello, how are you?",
    "stream": false
  }'
```

## 📁 Development Structure

```
/workspace/
├── dev/                    # Development-specific files
│   ├── models/            # Model storage for development
│   ├── data/              # Development database
│   ├── logs/              # Development logs
│   └── cache/             # Development cache
├── dashboard/             # React frontend
├── src/                   # Rust backend source
├── dev_config.toml        # Development configuration
├── .env                   # Environment variables
└── dev-start-*.sh         # Development startup scripts
```

## 🔧 Development Scripts Created

- `dev_start_backend.sh` - Start backend only
- `dev_start_frontend.sh` - Start frontend only  
- `dev_start_all.sh` - Start both services
- `dev_test_api.sh` - Test API endpoints
- `dev_test_frontend.sh` - Test frontend
- `dev_monitor.sh` - Monitor development environment

## 📱 Access URLs

When running:
- **Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:11435
- **API Documentation**: http://localhost:11435/docs
- **Health Check**: http://localhost:11435/api/health

## 🐛 Troubleshooting

### Backend Issues

**Rust not found:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

**Build failures:**
```bash
cargo clean
cargo update
cargo build --release
```

**Port already in use:**
```bash
# Kill process using port 11435
lsof -ti:11435 | xargs kill -9
```

### Frontend Issues

**Node.js/npm issues:**
```bash
# Update npm
npm install -g npm@latest

# Clean install
cd dashboard
rm -rf node_modules package-lock.json
npm install
```

**Port 3000 already in use:**
```bash
# Kill process using port 3000
lsof -ti:3000 | xargs kill -9
```

### Database Issues

**Database errors:**
```bash
# Recreate development database
rm -f ./dev_data/ollama_dev.db
mkdir -p ./dev_data
touch ./dev_data/ollama_dev.db
chmod 666 ./dev_data/ollama_dev.db
```

## 🎯 Features Enhanced

### Real Model Downloading
- ✅ HuggingFace Hub integration
- ✅ GGUF format support
- ✅ Multiple model support (LLaMA 3.2, Mistral, etc.)
- ✅ Model quantization (Q4_0, Q8_0, etc.)

### Enhanced Inference Engine
- ✅ Real Candle ML framework integration
- ✅ GPU acceleration support (CUDA/Metal)
- ✅ WebSocket streaming
- ✅ Batch processing capabilities

### Fine-tuning Support
- ✅ LoRA (Low-Rank Adaptation)
- ✅ Full model fine-tuning
- ✅ Training monitoring
- ✅ Checkpoint management

### Development Features
- ✅ Hot reload for both frontend and backend
- ✅ Real-time logging and monitoring
- ✅ Performance metrics
- ✅ Stress testing utilities

## 💡 Development Tips

1. **Use hot reload**: Both frontend and backend support hot reloading during development
2. **Monitor performance**: Use `./dev_monitor.sh` to monitor system resources
3. **Test incrementally**: Use the testing scripts to verify functionality step by step
4. **Clean builds**: Use `cargo clean` if you encounter strange build issues
5. **Check logs**: Development logs are stored in `./dev_logs/`

## 📚 Next Steps

After setup:
1. Download a test model using the API
2. Try generating text with different models
3. Experiment with fine-tuning features
4. Use the dashboard for visual management
5. Monitor performance and optimize

Happy coding! 🚀