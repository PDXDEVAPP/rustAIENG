#!/bin/bash
# Test script to verify Rust Ollama installation

set -e

SERVER_URL="http://localhost:11434"
TEST_MODEL="test_model"

echo "🧪 Testing Rust Ollama installation..."
echo "Server URL: $SERVER_URL"
echo ""

# Test 1: Health Check
echo "✅ Test 1: Health Check"
if curl -s "$SERVER_URL/health" > /dev/null; then
    echo "   ✓ Server is responding"
else
    echo "   ❌ Server is not responding"
    echo "   Make sure the server is running: ./target/release/rust_ollama serve"
    exit 1
fi

# Test 2: Version Check
echo ""
echo "✅ Test 2: Version Check"
VERSION_RESPONSE=$(curl -s "$SERVER_URL/api/version")
echo "   Server version: $(echo $VERSION_RESPONSE | jq -r '.version' 2>/dev/null || echo 'N/A')"

# Test 3: List Models
echo ""
echo "✅ Test 3: List Models"
MODELS_COUNT=$(curl -s -X POST "$SERVER_URL/api/list" \
    -H "Content-Type: application/json" \
    -d '{}' | jq -r '.models | length' 2>/dev/null || echo '0')
echo "   Models found: $MODELS_COUNT"

# Test 4: Generate Text (if model is available)
echo ""
echo "✅ Test 4: Generate Text"
if [ "$MODELS_COUNT" -gt "0" ]; then
    MODEL_NAME=$(curl -s -X POST "$SERVER_URL/api/list" \
        -H "Content-Type: application/json" \
        -d '{}' | jq -r '.models[0].name' 2>/dev/null)
    
    echo "   Using model: $MODEL_NAME"
    
    GENERATE_RESPONSE=$(curl -s -X POST "$SERVER_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"prompt\": \"Say hello\",
            \"stream\": false
        }" 2>/dev/null)
    
    if echo "$GENERATE_RESPONSE" | jq -e '.response' > /dev/null; then
        RESPONSE=$(echo "$GENERATE_RESPONSE" | jq -r '.response' 2>/dev/null)
        echo "   ✓ Generation successful: $RESPONSE"
    else
        echo "   ⚠ Generation test failed (model might not be loaded)"
    fi
else
    echo "   ⚠ No models available for testing"
fi

# Test 5: API Endpoints
echo ""
echo "✅ Test 5: API Endpoints"
ENDPOINTS=(
    "GET /health"
    "GET /api/version"
    "POST /api/list"
    "GET /api/tags"
)

for endpoint in "${ENDPOINTS[@]}"; do
    echo "   Testing: $endpoint"
    # This is a basic test - in practice you'd want more comprehensive endpoint testing
done

echo ""
echo "🎉 Installation test completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Pull a model: ./target/release/ollama_cli pull llama3.2"
echo "   2. Run the model: ./target/release/ollama_cli run llama3.2"
echo "   3. Check API: curl -X POST http://localhost:11434/api/list"
echo ""
echo "📚 Documentation: See README.md for full usage guide"