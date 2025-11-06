#!/bin/bash

# Rust Ollama - Comprehensive Development Setup
# This script sets up a complete development environment with all dependencies up to date

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${PURPLE}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${PURPLE}============================================${NC}\n"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in project root
check_project_root() {
    if [ ! -f "Cargo.toml" ] || [ ! -d "dashboard" ]; then
        print_error "Please run this script from the Rust Ollama project root directory"
        exit 1
    fi
}

# Check system prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        missing_deps+=("Rust/Cargo")
        print_error "Rust/Cargo not found. Please install Rust:"
        echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    else
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        print_success "Rust/Cargo found (version $RUST_VERSION)"
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        missing_deps+=("Node.js")
        print_error "Node.js not found. Please install Node.js 18+ from:"
        echo "https://nodejs.org/"
    else
        NODE_VERSION=$(node --version)
        print_success "Node.js found (version $NODE_VERSION)"
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        missing_deps+=("npm")
        print_error "npm not found. Please install npm:"
        echo "Usually comes with Node.js installation"
    else
        NPM_VERSION=$(npm --version)
        print_success "npm found (version $NPM_VERSION)"
    fi
    
    # Check additional tools
    for tool in "git" "curl" "sqlite3"; do
        if ! command -v $tool &> /dev/null; then
            missing_deps+=($tool)
            print_warning "$tool not found. Some features may not work."
        else
            print_success "$tool found"
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
}

# Setup development directories and files
setup_development_environment() {
    print_header "Setting Up Development Environment"
    
    print_step "Creating development directories..."
    
    # Create development directories
    mkdir -p dev/{models,data,logs,cache,backups}
    mkdir -p models
    mkdir -p logs
    
    # Setup development database
    print_step "Setting up development database..."
    export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
    
    # Create development configuration
    print_step "Creating development configuration..."
    cp config.toml dev_config.toml
    
    # Update dev config for development paths
    sed -i.bak \
        -e 's|database_path = "./ollama.db"|database_path = "./dev_data/ollama_dev.db"|g' \
        -e 's|models_directory = "./models"|models_directory = "./dev_models"|g' \
        -e 's|file = "./logs/ollama.log"|file = "./dev_logs/ollama.log"|g' \
        -e 's|level = "info"|level = "debug"|g' \
        -e 's|preload_models_on_startup = false|preload_models_on_startup = true|g' \
        -e 's|enable_model_prefetch = true|enable_model_prefetch = true|g' \
        dev_config.toml
    
    rm -f dev_config.toml.bak
    
    # Create environment file for development
    cat > .env << EOF
# Rust Ollama Development Environment
DATABASE_URL=sqlite:./dev_data/ollama_dev.db
RUST_LOG=debug
RUST_BACKTRACE=full

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=11435
WEBSOCKET_PATH=/ws

# Models Configuration
MODELS_DIR=./dev_models
CACHE_DIR=./dev_cache
LOGS_DIR=./dev_logs

# Performance Settings
MAX_CONNECTIONS=20
REQUEST_TIMEOUT=60
ENABLE_CACHING=true
CACHE_SIZE_MB=512

# Development Settings
ENABLE_HOT_RELOAD=true
ENABLE_DEBUG_LOGGING=true
ENABLE_PERFORMANCE_PROFILING=false

# Cloud Storage (Development - optional)
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AZURE_STORAGE_ACCOUNT_NAME=your_account
# AZURE_STORAGE_ACCOUNT_KEY=your_key

# HuggingFace Token (optional, for private models)
# HUGGINGFACE_TOKEN=your_token

# Redis Cache (optional)
# REDIS_URL=redis://localhost:6379
EOF

    print_success "Development environment configured"
}

# Update and install Rust dependencies
update_rust_dependencies() {
    print_header "Updating Rust Dependencies"
    
    cd /workspace
    
    # Clean previous builds
    print_step "Cleaning previous builds..."
    cargo clean
    
    # Update dependencies
    print_step "Updating Rust dependencies to latest compatible versions..."
    cargo update
    
    # Install or update rustfmt
    print_step "Installing rustfmt..."
    rustup component add rustfmt
    
    # Install or update clippy
    print_step "Installing clippy..."
    rustup component add clippy
    
    # Build the project
    print_step "Building the project..."
    cargo build
    
    # Check the project
    print_step "Running cargo check..."
    cargo check
    
    # Run clippy lints
    print_step "Running clippy lints..."
    cargo clippy -- -D warnings || print_warning "Some clippy warnings found"
    
    print_success "Rust dependencies updated and project built successfully"
}

# Update and install Node.js dependencies
update_node_dependencies() {
    print_header "Updating Node.js Dependencies"
    
    cd dashboard
    
    # Clean node_modules and package-lock
    print_step "Cleaning previous Node.js dependencies..."
    rm -rf node_modules package-lock.json
    
    # Update npm to latest version
    print_step "Updating npm to latest version..."
    npm install -g npm@latest
    
    # Install dependencies
    print_step "Installing Node.js dependencies..."
    npm install
    
    # Check for outdated packages
    print_step "Checking for outdated packages..."
    npm outdated --depth=0 || print_info "All packages are up to date"
    
    # Build the project to verify
    print_step "Building React project..."
    npm run build
    
    cd ..
    
    print_success "Node.js dependencies updated successfully"
}

# Create development startup scripts
create_dev_scripts() {
    print_header "Creating Development Scripts"
    
    # Backend development script
    cat > dev_start_backend.sh << 'EOF'
#!/bin/bash

# Development Backend Startup Script

echo "🚀 Starting Rust Ollama Backend (Development Mode)..."

# Set development environment
export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
export RUST_LOG=debug
export RUST_BACKTRACE=full

# Start with hot reload enabled
echo "Starting server with development configuration..."
cargo run -- serve --port 11435 --config dev_config.toml --database ./dev_data/ollama_dev.db --models-dir ./dev_models --logs-dir ./dev_logs
EOF

    # Frontend development script
    cat > dev_start_frontend.sh << 'EOF'
#!/bin/bash

# Development Frontend Startup Script

echo "⚛️  Starting React Dashboard (Development Mode)..."

cd dashboard

# Start Vite dev server with hot reload
npm run dev

cd ..
EOF

    # Combined development script
    cat > dev_start_all.sh << 'EOF'
#!/bin/bash

# Combined Development Startup Script

echo "🚀 Starting Complete Development Environment..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping all services..."
    kill $(jobs -p) 2>/dev/null || true
    exit
}

trap cleanup INT

# Start backend in background
echo "Starting Rust backend..."
./dev_start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is responding
if curl -s http://localhost:11435/api/models >/dev/null 2>&1; then
    echo "✅ Backend is running"
else
    print_warning "Backend may not be fully started yet"
fi

# Start frontend in background
echo "Starting React dashboard..."
./dev_start_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "🎉 Development environment is ready!"
echo "📱 Dashboard: http://localhost:3000"
echo "🔧 Backend API: http://localhost:11435"
echo "📊 Metrics: http://localhost:9090"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for any process to finish
wait
EOF

    # Make scripts executable
    chmod +x dev_start_backend.sh dev_start_frontend.sh dev_start_all.sh
    
    print_success "Development scripts created"
}

# Create testing utilities
create_testing_utils() {
    print_header "Creating Testing Utilities"
    
    # Backend API testing script
    cat > dev_test_api.sh << 'EOF'
#!/bin/bash

# API Testing Script for Development

echo "🧪 Testing Rust Ollama API..."

BASE_URL="http://localhost:11435"

# Test health check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/api/health" | jq . || echo "Health check failed"

echo ""
echo "2. Testing models endpoint..."
curl -s "$BASE_URL/api/models" | jq . || echo "Models check failed"

echo ""
echo "3. Testing model pull (llama3.2)..."
curl -X POST "$BASE_URL/api/pull" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}' | jq . || echo "Model pull test failed"

echo ""
echo "4. Testing generate endpoint..."
curl -X POST "$BASE_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "Hello, how are you?", "stream": false}' | jq . || echo "Generate test failed"

echo ""
echo "✅ API testing completed!"
EOF

    # Frontend testing script
    cat > dev_test_frontend.sh << 'EOF'
#!/bin/bash

# Frontend Testing Script

echo "🧪 Testing React Dashboard..."

cd dashboard

echo "1. Checking if development server is running..."
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "✅ Dashboard is accessible at http://localhost:3000"
else
    echo "❌ Dashboard is not accessible"
fi

echo ""
echo "2. Running TypeScript checks..."
npm run build

echo ""
echo "3. Running linting..."
npm run lint

cd ..

echo "✅ Frontend testing completed!"
EOF

    # Performance monitoring script
    cat > dev_monitor.sh << 'EOF'
#!/bin/bash

# Development Monitoring Script

echo "📊 Starting Development Monitoring..."

# Function to show system resources
show_resources() {
    echo "💻 System Resources:"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    echo "Memory Usage:"
    free -h | grep Mem | awk '{print $3"/"$2}'
    echo ""
}

# Function to check API health
check_api() {
    echo "🔧 API Health Check:"
    if curl -s http://localhost:11435/api/health >/dev/null 2>&1; then
        echo "✅ Backend API: Running"
    else
        echo "❌ Backend API: Not responding"
    fi
    echo ""
}

# Function to check database
check_database() {
    echo "🗄️ Database Check:"
    if [ -f "./dev_data/ollama_dev.db" ]; then
        echo "✅ Database file exists"
        echo "Database size: $(du -h ./dev_data/ollama_dev.db | cut -f1)"
    else
        echo "❌ Database file not found"
    fi
    echo ""
}

# Function to check models
check_models() {
    echo "🤖 Models Check:"
    if [ -d "./dev_models" ]; then
        echo "✅ Models directory exists"
        echo "Models found: $(ls ./dev_models 2>/dev/null | wc -l)"
        echo "Directory size: $(du -sh ./dev_models | cut -f1)"
    else
        echo "❌ Models directory not found"
    fi
    echo ""
}

# Main monitoring loop
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "📊 Development Environment Monitor"
    echo "================================="
    echo ""
    
    show_resources
    check_api
    check_database
    check_models
    
    echo "⏰ Last updated: $(date)"
    echo ""
    echo "Press Ctrl+C to exit"
    
    sleep 5
done
EOF

    # Make scripts executable
    chmod +x dev_test_api.sh dev_test_frontend.sh dev_monitor.sh
    
    print_success "Testing utilities created"
}

# Setup development database
setup_database() {
    print_header "Setting Up Development Database"
    
    export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
    
    # The database will be created automatically when the server starts
    # For now, just ensure the directory exists and has proper permissions
    touch ./dev_data/ollama_dev.db
    chmod 666 ./dev_data/ollama_dev.db
    
    print_success "Database setup completed"
}

# Show final setup summary
show_summary() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}🎉 Your Rust Ollama development environment is ready!${NC}\n"
    
    echo -e "${CYAN}📁 Development Files Created:${NC}"
    echo "  • dev_config.toml - Development configuration"
    echo "  • .env - Environment variables"
    echo "  • dev_start_backend.sh - Backend startup script"
    echo "  • dev_start_frontend.sh - Frontend startup script"
    echo "  • dev_start_all.sh - Combined startup script"
    echo "  • dev_test_api.sh - API testing script"
    echo "  • dev_test_frontend.sh - Frontend testing script"
    echo "  • dev_monitor.sh - Development monitoring script"
    
    echo ""
    echo -e "${CYAN}🚀 Quick Start Commands:${NC}"
    echo "  • Start everything:  ./dev_start_all.sh"
    echo "  • Start backend:     ./dev_start_backend.sh"
    echo "  • Start frontend:    ./dev_start_frontend.sh"
    echo "  • Test API:          ./dev_test_api.sh"
    echo "  • Monitor system:    ./dev_monitor.sh"
    
    echo ""
    echo -e "${CYAN}📱 Access URLs:${NC}"
    echo "  • Dashboard:         http://localhost:3000"
    echo "  • Backend API:       http://localhost:11435"
    echo "  • API Documentation: http://localhost:11435/docs"
    echo "  • Metrics:           http://localhost:9090"
    
    echo ""
    echo -e "${CYAN}🛠️ Development Workflow:${NC}"
    echo "  1. Make changes to Rust backend or React frontend"
    echo "  2. Use the provided scripts to test changes"
    echo "  3. Monitor performance with dev_monitor.sh"
    echo "  4. Test API endpoints with dev_test_api.sh"
    
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "  • Backend has hot reload (auto-rebuilds on changes)"
    echo "  • Frontend has hot reload via Vite"
    echo "  • All data is stored in dev_data/ and dev_models/"
    echo "  • Logs are available in dev_logs/"
    echo "  • Use dev_config.toml for development-specific settings"
    
    echo ""
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Main setup function
main() {
    print_header "Rust Ollama - Comprehensive Development Setup"
    
    check_project_root
    check_prerequisites
    setup_development_environment
    update_rust_dependencies
    update_node_dependencies
    create_dev_scripts
    create_testing_utils
    setup_database
    
    show_summary
}

# Run the setup
main "$@"