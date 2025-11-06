#!/bin/bash

echo "🚀 Setting up Rust Ollama Development Environment..."

# Create development directories
echo "📁 Creating development directories..."
mkdir -p dev/{models,data,logs,cache}
mkdir -p models logs

# Create development configuration
echo "⚙️ Creating development configuration..."
cp config.toml dev_config.toml
sed -i.bak \
    -e 's|database_path = "./ollama.db"|database_path = "./dev_data/ollama_dev.db"|g' \
    -e 's|models_directory = "./models"|models_directory = "./dev_models"|g' \
    -e 's|file = "./logs/ollama.log"|file = "./dev_logs/ollama.log"|g' \
    -e 's|level = "info"|level = "debug"|g' \
    dev_config.toml
rm -f dev_config.toml.bak

# Create environment file
echo "📋 Setting up environment..."
cat > .env << EOF
DATABASE_URL=sqlite:./dev_data/ollama_dev.db
RUST_LOG=debug
SERVER_PORT=11435
MODELS_DIR=./dev_models
EOF

# Update Rust dependencies and build
echo "📦 Updating Rust dependencies..."
cargo update
echo "🔨 Building project..."
cargo build --release

# Setup dashboard
echo "⚛️ Setting up dashboard..."
cd dashboard
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "🚀 To start development:"
echo "1. Backend: cargo run --release -- serve --port 11435"
echo "2. Frontend: cd dashboard && npm run dev"
echo ""
echo "Or run both: ./dev-run.sh"