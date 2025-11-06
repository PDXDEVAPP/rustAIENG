#!/bin/bash

# Rust Ollama - Development Run Script
# This script runs the complete development environment and tests the enhanced backend

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if setup has been run
check_setup() {
    if [ ! -f ".env" ] || [ ! -f "dev_config.toml" ]; then
        print_warning "Development environment not set up yet"
        echo "Please run the setup script first:"
        echo "  ./dev-setup-comprehensive.sh"
        exit 1
    fi
}

# Build the project
build_project() {
    print_header "Building Project"
    
    print_step "Building Rust backend..."
    cargo build --release
    
    if [ $? -eq 0 ]; then
        print_success "Rust backend built successfully"
    else
        print_error "Failed to build Rust backend"
        exit 1
    fi
    
    print_step "Building React frontend..."
    cd dashboard
    npm run build
    
    if [ $? -eq 0 ]; then
        print_success "React frontend built successfully"
    else
        print_error "Failed to build React frontend"
        exit 1
    fi
    
    cd ..
}

# Start the backend server
start_backend() {
    print_header "Starting Backend Server"
    
    print_step "Setting development environment..."
    export DATABASE_URL="sqlite:./dev_data/ollama_dev.db"
    export RUST_LOG=debug
    export RUST_BACKTRACE=full
    
    print_step "Starting Rust Ollama server on port 11435..."
    echo "Command: ./target/release/rust_ollama serve --port 11435"
    echo "Configuration: dev_config.toml"
    echo ""
    
    # Start server in background
    ./target/release/rust_ollama serve --port 11435 &
    BACKEND_PID=$!
    
    echo "Backend PID: $BACKEND_PID"
    
    # Wait for server to start
    print_step "Waiting for server to start..."
    local retries=0
    local max_retries=30
    
    while [ $retries -lt $max_retries ]; do
        if curl -s http://localhost:11435/api/health >/dev/null 2>&1; then
            print_success "Backend server is running!"
            return 0
        fi
        
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            print_error "Backend server died unexpectedly"
            return 1
        fi
        
        sleep 1
        retries=$((retries + 1))
        echo -n "."
    done
    
    echo ""
    print_error "Backend server failed to start within 30 seconds"
    kill $BACKEND_PID 2>/dev/null || true
    return 1
}

# Test model download functionality
test_model_download() {
    print_header "Testing Model Download"
    
    print_step "Testing model download endpoint..."
    echo "Request: POST http://localhost:11435/api/pull"
    echo "Body: {\"name\": \"llama3.2\"}"
    echo ""
    
    # Test model download
    RESPONSE=$(curl -s -X POST http://localhost:11435/api/pull \
        -H "Content-Type: application/json" \
        -d '{"name": "llama3.2"}')
    
    echo "Response:"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    echo ""
    
    # Check if download started
    if echo "$RESPONSE" | grep -q '"status"' || echo "$RESPONSE" | grep -q 'downloading'; then
        print_success "Model download request accepted"
    else
        print_warning "Model download response unclear"
    fi
    
    # Wait a moment for download to potentially start
    sleep 2
}

# Test API endpoints
test_api_endpoints() {
    print_header "Testing API Endpoints"
    
    # Test health endpoint
    print_step "Testing health endpoint..."
    HEALTH=$(curl -s http://localhost:11435/api/health)
    echo "Health: $HEALTH"
    echo ""
    
    # Test models list
    print_step "Testing models list..."
    MODELS=$(curl -s http://localhost:11435/api/models)
    echo "Models: $MODELS"
    echo ""
    
    # Test generation (if model is available)
    print_step "Testing generation endpoint..."
    GENERATE=$(curl -s -X POST http://localhost:11435/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "llama3.2", "prompt": "Hello, how are you?", "stream": false}')
    
    echo "Generation: $GENERATE"
    echo ""
}

# Start frontend (optional)
start_frontend() {
    print_header "Starting Frontend"
    
    print_step "Starting React dashboard..."
    cd dashboard
    
    # Start Vite dev server in background
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
    
    echo "Frontend PID: $FRONTEND_PID"
    
    # Wait for frontend to start
    print_step "Waiting for frontend to start..."
    local retries=0
    local max_retries=10
    
    while [ $retries -lt $max_retries ]; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            print_success "Frontend dashboard is running!"
            return 0
        fi
        
        sleep 1
        retries=$((retries + 1))
        echo -n "."
    done
    
    echo ""
    print_warning "Frontend may not be fully started yet"
}

# Show system information
show_system_info() {
    print_header "System Information"
    
    echo -e "${CYAN}Backend:${NC}"
    echo "  • URL: http://localhost:11435"
    echo "  • Status: $(curl -s http://localhost:11435/api/health | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")"
    echo "  • PID: $BACKEND_PID"
    echo ""
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo -e "${CYAN}Frontend:${NC}"
        echo "  • URL: http://localhost:3000"
        echo "  • Status: $(curl -s http://localhost:3000 >/dev/null 2>&1 && echo "running" || echo "not accessible")"
        echo "  • PID: $FRONTEND_PID"
        echo ""
    fi
    
    echo -e "${CYAN}Development Directories:${NC}"
    echo "  • Database: ./dev_data/ollama_dev.db"
    echo "  • Models: ./dev_models/"
    echo "  • Logs: ./dev_logs/"
    echo "  • Config: dev_config.toml"
    echo ""
    
    echo -e "${CYAN}Available Endpoints:${NC}"
    echo "  • Health: http://localhost:11435/api/health"
    echo "  • Models: http://localhost:11435/api/models"
    echo "  • Generate: http://localhost:11435/api/generate"
    echo "  • Pull: http://localhost:11435/api/pull"
    echo ""
}

# Cleanup function
cleanup() {
    echo ""
    print_header "Shutting Down Services"
    
    if [ ! -z "$BACKEND_PID" ]; then
        print_step "Stopping backend server..."
        kill $BACKEND_PID 2>/dev/null || true
        print_success "Backend stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        print_step "Stopping frontend server..."
        kill $FRONTEND_PID 2>/dev/null || true
        print_success "Frontend stopped"
    fi
    
    echo ""
    print_success "All services stopped"
    exit 0
}

# Main execution
main() {
    print_header "Rust Ollama Development Run"
    
    # Set up cleanup trap
    trap cleanup INT TERM
    
    # Check setup
    check_setup
    
    # Build project
    build_project
    
    # Start backend
    if start_backend; then
        # Test model download
        test_model_download
        
        # Test API endpoints
        test_api_endpoints
        
        # Start frontend (optional)
        start_frontend
        
        # Show system information
        show_system_info
        
        print_header "Development Environment Ready!"
        
        echo -e "${GREEN}🎉 Your development environment is now running!${NC}\n"
        
        echo -e "${CYAN}Access URLs:${NC}"
        echo "  • Dashboard: http://localhost:3000"
        echo "  • Backend API: http://localhost:11435"
        echo "  • API Docs: http://localhost:11435/docs"
        echo ""
        
        echo -e "${YELLOW}Test Commands:${NC}"
        echo "  • Download model: curl -X POST http://localhost:11435/api/pull -H 'Content-Type: application/json' -d '{\"name\": \"mistral\"}'"
        echo "  • Generate text: curl -X POST http://localhost:11435/api/generate -H 'Content-Type: application/json' -d '{\"model\": \"llama3.2\", \"prompt\": \"Hello!\"}'"
        echo "  • List models: curl http://localhost:11435/api/models"
        echo ""
        
        echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
        echo ""
        
        # Wait for user interrupt
        wait
    else
        print_error "Failed to start backend server"
        exit 1
    fi
}

# Run the development environment
main "$@"