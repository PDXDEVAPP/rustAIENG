#!/bin/bash
# Deployment script for Rust Ollama

set -e

ENVIRONMENT=${1:-"production"}
BUILD_TYPE=${2:-"release"}

echo "🚀 Deploying Rust Ollama to $ENVIRONMENT environment"

# Validate environment
case $ENVIRONMENT in
    "production"|"staging"|"development")
        echo "   Environment: $ENVIRONMENT"
        ;;
    *)
        echo "❌ Invalid environment. Use: production, staging, or development"
        exit 1
        ;;
esac

# Build the application
echo "🔨 Building application ($BUILD_TYPE)..."
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release --bin rust_ollama
    cargo build --release --bin ollama_cli
else
    cargo build --bin rust_ollama
    cargo build --bin ollama_cli
fi

# Create deployment directories
DEPLOY_DIR="./deploy/$ENVIRONMENT"
mkdir -p "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR/data"
mkdir -p "$DEPLOY_DIR/models"
mkdir -p "$DEPLOY_DIR/logs"
mkdir -p "$DEPLOY_DIR/config"

# Copy binaries
echo "📦 Copying binaries..."
cp "./target/$BUILD_TYPE/rust_ollama" "$DEPLOY_DIR/"
cp "./target/$BUILD_TYPE/ollama_cli" "$DEPLOY_DIR/"

# Copy configuration
echo "⚙️  Setting up configuration..."
case $ENVIRONMENT in
    "production")
        cp config.toml "$DEPLOY_DIR/config/"
        sed -i.bak 's/level = "info"/level = "warn"/g' "$DEPLOY_DIR/config/config.toml"
        sed -i.bak 's/auto_cleanup = false/auto_cleanup = true/g' "$DEPLOY_DIR/config/config.toml"
        rm "$DEPLOY_DIR/config/config.toml.bak"
        ;;
    "staging")
        cp config.toml "$DEPLOY_DIR/config/"
        sed -i.bak 's/level = "info"/level = "debug"/g' "$DEPLOY_DIR/config/config.toml"
        rm "$DEPLOY_DIR/config/config.toml.bak"
        ;;
    "development")
        cp .env.example "$DEPLOY_DIR/config/.env"
        ;;
esac

# Create systemd service file (for Linux production)
if [ "$ENVIRONMENT" = "production" ] && [ "$(uname)" = "Linux" ]; then
    echo "🔧 Creating systemd service..."
    cat > "$DEPLOY_DIR/rust_ollama.service" << EOF
[Unit]
Description=Rust Ollama LLM Server
After=network.target

[Service]
Type=simple
User=ollama
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/rust_ollama serve --port 11434 --models-dir ./models
Restart=always
RestartSec=10

# Environment
Environment=RUST_LOG=info
Environment=DATABASE_PATH=$DEPLOY_DIR/data/ollama.db

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DEPLOY_DIR

[Install]
WantedBy=multi-user.target
EOF
fi

# Create Docker compose file
echo "🐳 Creating Docker configuration..."
cat > "$DEPLOY_DIR/docker-compose.yml" << EOF
version: '3.8'
services:
  rust_ollama:
    build: ../../
    ports:
      - "11434:11434"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DATABASE_PATH=/app/data/ollama.db
      - MODELS_DIR=/app/models
      - RUST_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "./ollama_cli", "health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  data:
  models:
  logs:
EOF

# Create startup script
echo "📝 Creating startup scripts..."
cat > "$DEPLOY_DIR/start.sh" << 'EOF'
#!/bin/bash
set -e

ENVIRONMENT=$(basename "$(pwd)")

echo "🚀 Starting Rust Ollama ($ENVIRONMENT)..."
echo "   Database: ./data/ollama.db"
echo "   Models: ./models"
echo "   Logs: ./logs"
echo ""

# Check if models directory is empty
if [ ! "$(ls -A ./models 2>/dev/null)" ]; then
    echo "⚠️  Models directory is empty. You may want to pull some models:"
    echo "   ./ollama_cli pull llama3.2"
    echo ""
fi

# Start the server
exec ./rust_ollama serve --port 11434 --database ./data/ollama.db --models-dir ./models
EOF

chmod +x "$DEPLOY_DIR/start.sh"

cat > "$DEPLOY_DIR/stop.sh" << 'EOF'
#!/bin/bash
echo "🛑 Stopping Rust Ollama..."
pkill -f "rust_ollama serve" || echo "No processes found to stop"
echo "Stopped"
EOF

chmod +x "$DEPLOY_DIR/stop.sh"

# Create health check script
cat > "$DEPLOY_DIR/health_check.sh" << EOF
#!/bin/bash

SERVER_URL="http://localhost:11434"

echo "🏥 Health check for Rust Ollama"

# Check if server is responding
if curl -s "\$SERVER_URL/health" > /dev/null; then
    echo "✅ Server is healthy"
    echo "   Version: \$(curl -s \$SERVER_URL/api/version | jq -r '.version' 2>/dev/null || echo 'unknown')"
    echo "   Models: \$(curl -s -X POST \$SERVER_URL/api/list -H 'Content-Type: application/json' -d '{}' | jq -r '.models | length' 2>/dev/null || echo 'unknown')"
    exit 0
else
    echo "❌ Server is not responding"
    exit 1
fi
EOF

chmod +x "$DEPLOY_DIR/health_check.sh"

# Create README for deployment
cat > "$DEPLOY_DIR/README.md" << EOF
# Rust Ollama Deployment ($ENVIRONMENT)

This directory contains a complete deployment of Rust Ollama for the $ENVIRONMENT environment.

## Quick Start

1. **Start the server:**
   \`\`\`bash
   ./start.sh
   \`\`\`

2. **Verify installation:**
   \`\`\`bash
   ./health_check.sh
   \`\`\`

3. **Pull models:**
   \`\`\`bash
   ./ollama_cli pull llama3.2
   \`\`\`

## Files

- \`rust_ollama\` - Main server binary
- \`ollama_cli\` - CLI tool
- \`config/config.toml\` - Configuration
- \`start.sh\` - Startup script
- \`stop.sh\` - Stop script
- \`health_check.sh\` - Health check
- \`data/\` - Database files
- \`models/\` - Downloaded models
- \`logs/\` - Application logs

## Management

\`\`\`bash
# Start server
./start.sh

# Stop server
./stop.sh

# Check health
./health_check.sh

# Pull models
./ollama_cli pull llama3.2

# List models
./ollama_cli list

# Run model
./ollama_cli run llama3.2 "Hello world"
\`\`\`

## Docker

\`\`\`bash
docker-compose up -d
\`\`\`

EOF

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📁 Deployment directory: $DEPLOY_DIR"
echo ""
echo "🚀 To start the deployment:"
echo "   cd $DEPLOY_DIR"
echo "   ./start.sh"
echo ""
echo "📋 To verify deployment:"
echo "   cd $DEPLOY_DIR"
echo "   ./health_check.sh"
echo ""
echo "🐳 To use Docker:"
echo "   cd $DEPLOY_DIR"
echo "   docker-compose up -d"