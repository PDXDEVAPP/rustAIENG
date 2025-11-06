#!/bin/bash
# Development setup script for Rust Ollama

set -e

echo "🔧 Setting up Rust Ollama development environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p dev_models
mkdir -p dev_logs
mkdir -p dev_data

# Copy environment file
echo "📋 Setting up environment..."
cp .env.example .env

# Create example config
echo "⚙️  Setting up configuration..."
cp config.toml dev_config.toml

# Update dev config paths
sed -i.bak 's|./ollama.db|./dev_data/dev_ollama.db|g' dev_config.toml
sed -i.bak 's|./models|./dev_models|g' dev_config.toml
sed -i.bak 's|./logs/ollama.log|./dev_logs/ollama.log|g' dev_config.toml
rm dev_config.toml.bak

# Install Rust dependencies
echo "📦 Installing Rust dependencies..."
cargo update

# Run database migrations
echo "🗄️  Running database setup..."
export DATABASE_URL=sqlite:./dev_data/dev_ollama.db
export RUST_LOG=debug

# Build the project
echo "🔨 Building project..."
cargo build

echo "✅ Setup complete!"
echo ""
echo "🚀 To start development:"
echo "   export DATABASE_URL=sqlite:./dev_data/dev_ollama.db"
echo "   cargo run -- serve --port 11435 --database ./dev_data/dev_ollama.db --models-dir ./dev_models"
echo ""
echo "📝 Development configuration saved to:"
echo "   - .env"
echo "   - dev_config.toml"
echo "   - dev_models/ (models directory)"
echo "   - dev_data/ (database)"
echo "   - dev_logs/ (logs)"