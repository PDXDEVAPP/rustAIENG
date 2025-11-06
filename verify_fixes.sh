#!/bin/bash

echo "=== Rust Ollama Compilation Verification ==="
echo "Verifying fixes applied to resolve compilation errors..."
echo

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust/Cargo first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "   source ~/.cargo/env"
    exit 1
fi

echo "✅ Cargo found: $(cargo --version)"
echo

# Check if Cargo.toml exists
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Cargo.toml not found. Run this script from the project root."
    exit 1
fi

echo "=== Step 1: Overall Compilation Check ==="
cargo check

echo
echo "=== Step 2: Binary Compilation Tests ==="

echo "Checking model_finetuner binary..."
if cargo check --bin model_finetuner; then
    echo "✅ model_finetuner compiles successfully"
else
    echo "❌ model_finetuner has compilation errors"
fi

echo
echo "Checking ollama_cli binary..."
if cargo check --bin ollama_cli; then
    echo "✅ ollama_cli compiles successfully"  
else
    echo "❌ ollama_cli has compilation errors"
fi

echo
echo "=== Step 3: Build Test ==="
echo "Building all binaries..."
cargo build

echo
echo "=== Step 4: Dependency Verification ==="
echo "Checking dependency tree..."
cargo tree | grep -E "(candle|rand)" | head -10

echo
echo "=== Verification Complete ==="
echo "If all steps passed, the compilation fixes have been successfully applied."
echo "You can now proceed with local development setup."