#!/bin/bash

# Rust Ollama Dashboard - Quick Start Script
# This script sets up both the Rust backend and React dashboard

set -e

echo "🚀 Rust Ollama Dashboard - Quick Start"
echo "======================================"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ from:"
    echo "   https://nodejs.org/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm:"
    echo "   Usually comes with Node.js installation"
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Setup Rust Backend
echo ""
echo "🔧 Setting up Rust Backend..."
echo "==============================="

if [ ! -f "Cargo.toml" ]; then
    echo "❌ Cargo.toml not found. Please run this script from the project root."
    exit 1
fi

echo "📦 Installing Rust dependencies..."
cargo check

if [ $? -eq 0 ]; then
    echo "✅ Rust dependencies installed successfully!"
else
    echo "❌ Failed to install Rust dependencies"
    exit 1
fi

# Setup React Dashboard
echo ""
echo "⚛️  Setting up React Dashboard..."
echo "================================="

if [ ! -d "dashboard" ]; then
    echo "❌ Dashboard directory not found. Please run this script from the project root."
    exit 1
fi

cd dashboard

echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Node.js dependencies installed successfully!"
else
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

cd ..

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📖 Next Steps:"
echo "1. Start the Rust backend:"
echo "   cargo run --release"
echo ""
echo "2. In a new terminal, start the dashboard:"
echo "   cd dashboard && npm run dev"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:3000"
echo ""
echo "🔗 Integration Guide:"
echo "   See INTEGRATION.md for detailed API documentation"
echo ""
echo "📚 Documentation:"
echo "   See dashboard/README.md for full feature guide"
echo ""

# Ask if user wants to start the services
read -p "🚀 Would you like to start the services now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting Rust Backend..."
    echo "============================"
    echo "Press Ctrl+C to stop the server"
    echo ""
    cargo run --release &
    RUST_PID=$!
    
    echo ""
    echo "⏳ Waiting for Rust server to start..."
    sleep 5
    
    echo ""
    echo "⚛️  Starting React Dashboard..."
    echo "==============================="
    cd dashboard
    npm run dev &
    DASHBOARD_PID=$!
    
    echo ""
    echo "🎉 Both services are starting!"
    echo "🔗 Dashboard will be available at: http://localhost:3000"
    echo "🔧 Rust API will be available at: http://localhost:11435"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Wait for user interrupt
    trap 'kill $RUST_PID $DASHBOARD_PID 2>/dev/null; echo ""; echo "🛑 Services stopped"; exit 0' INT
    wait
fi

echo ""
echo "✅ Quick start setup completed!"
echo "Happy coding! 🎯"