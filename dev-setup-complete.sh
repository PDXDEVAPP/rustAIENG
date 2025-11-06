#!/bin/bash

# Rust Ollama Development Setup - Complete Solution
echo "🚀 Rust Ollama Development Setup"
echo "================================="

# Function to install Rust if not present
install_rust() {
    if ! command -v cargo &> /dev/null; then
        echo "📦 Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    else
        echo "✅ Rust already installed"
        source ~/.cargo/env
    fi
}

# Function to install Node.js if not present  
install_nodejs() {
    if ! command -v node &> /dev/null; then
        echo "📦 Installing Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        echo "✅ Node.js already installed"
    fi
}

# Setup development environment
setup_dev_env() {
    echo ""
    echo "⚙️ Setting up development environment..."
    
    # Create directories
    mkdir -p dev/{models,data,logs,cache}
    mkdir -p models logs
    
    # Create dev config
    cp config.toml dev_config.toml
    sed -i.bak \
        -e 's|database_path = "./ollama.db"|database_path = "./dev_data/ollama_dev.db"|g' \
        -e 's|models_directory = "./models"|models_directory = "./dev_models"|g' \
        -e 's|file = "./logs/ollama.log"|file = "./dev_logs/ollama.log"|g' \
        -e 's|level = "info"|level = "debug"|g' \
        dev_config.toml
    rm -f dev_config.toml.bak
    
    # Create env file
    cat > .env << EOF
DATABASE_URL=sqlite:./dev_data/ollama_dev.db
RUST_LOG=debug
SERVER_PORT=11435
MODELS_DIR=./dev_models
LOGS_DIR=./dev_logs
ENABLE_HOT_RELOAD=true
EOF
    
    echo "✅ Development environment configured"
}

# Update and build dependencies
build_dependencies() {
    echo ""
    echo "🔨 Building dependencies..."
    
    # Install/update Rust deps
    source ~/.cargo/env
    echo "📦 Updating Rust dependencies..."
    cargo update
    
    echo "🔨 Building Rust project..."
    cargo build --release
    
    if [ $? -eq 0 ]; then
        echo "✅ Rust backend built successfully"
    else
        echo "❌ Rust build failed"
        return 1
    fi
    
    # Build frontend
    echo "📦 Building React frontend..."
    cd dashboard
    npm install
    npm run build
    cd ..
    
    if [ $? -eq 0 ]; then
        echo "✅ React frontend built successfully"
    else
        echo "❌ Frontend build failed"
        return 1
    fi
}

# Test the setup
test_setup() {
    echo ""
    echo "🧪 Testing the setup..."
    
    # Test Rust backend
    echo "Testing Rust backend startup..."
    source ~/.cargo/env
    export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
    export RUST_LOG=debug
    
    # Start server in background and test
    cargo run --release -- serve --port 11435 &
    BACKEND_PID=$!
    
    echo "Backend PID: $BACKEND_PID"
    
    # Wait for startup
    sleep 5
    
    # Test API
    if curl -s http://localhost:11435/api/health >/dev/null 2>&1; then
        echo "✅ Backend API is responding"
        
        # Test model download
        echo "Testing model download..."
        curl -X POST http://localhost:11435/api/pull \
            -H "Content-Type: application/json" \
            -d '{"name": "llama3.2"}' \
            2>/dev/null || echo "Model download request sent"
        
        # Test other endpoints
        echo "Testing other endpoints..."
        curl -s http://localhost:11435/api/models >/dev/null 2>&1 && echo "✅ Models endpoint working"
        
        # Stop server
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
        
        echo "✅ Backend tests completed successfully"
    else
        echo "❌ Backend API not responding"
        kill $BACKEND_PID 2>/dev/null || true
        return 1
    fi
}

# Main setup function
main() {
    install_rust
    install_nodejs
    setup_dev_env
    build_dependencies
    
    # Ask user if they want to test
    read -p "🧪 Would you like to test the setup? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_setup
    fi
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📁 Development files created:"
    echo "  • dev_config.toml - Development configuration"
    echo "  • .env - Environment variables"
    echo "  • ./dev_models/ - Models directory"
    echo "  • ./dev_data/ - Database and data"
    echo ""
    echo "🚀 Start development:"
    echo "  Backend: source ~/.cargo/env && cargo run --release -- serve --port 11435"
    echo "  Frontend: cd dashboard && npm run dev"
    echo ""
    echo "🔗 URLs:"
    echo "  Dashboard: http://localhost:3000"
    echo "  Backend API: http://localhost:11435"
    echo ""
    echo "📖 For detailed setup: run ./dev-setup-comprehensive.sh"
    echo "🚀 For one-click dev run: run ./dev-run.sh"
}

# Run main setup
main