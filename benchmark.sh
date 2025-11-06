#!/bin/bash
# Benchmark script for Rust Ollama

set -e

SERVER_URL="http://localhost:11434"
BENCHMARK_MODEL=""
PROMPT="Explain the concept of artificial intelligence in 50 words or less."
ITERATIONS=10
OUTPUT_FILE="benchmark_results.txt"

echo "🏃 Rust Ollama Performance Benchmark"
echo "Server: $SERVER_URL"
echo "Prompt: $PROMPT"
echo "Iterations: $ITERATIONS"
echo ""

# Get list of available models
echo "📋 Checking available models..."
MODELS_RESPONSE=$(curl -s -X POST "$SERVER_URL/api/list" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo '{"models":[]}')

MODELS_COUNT=$(echo "$MODELS_RESPONSE" | jq -r '.models | length' 2>/dev/null || echo '0')

if [ "$MODELS_COUNT" -eq "0" ]; then
    echo "❌ No models available. Please pull a model first:"
    echo "   ./ollama_cli pull llama3.2"
    exit 1
fi

# Use the first available model
BENCHMARK_MODEL=$(echo "$MODELS_RESPONSE" | jq -r '.models[0].name' 2>/dev/null)
echo "🎯 Using model: $BENCHMARK_MODEL"
echo ""

# Initialize results
echo "Running benchmark..." > "$OUTPUT_FILE"
echo "Model: $BENCHMARK_MODEL" >> "$OUTPUT_FILE"
echo "Prompt: $PROMPT" >> "$OUTPUT_FILE"
echo "Iterations: $ITERATIONS" >> "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Warm up the model
echo "🔥 Warming up model..."
curl -s -X POST "$SERVER_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$BENCHMARK_MODEL\",
        \"prompt\": \"$PROMPT\",
        \"stream\": false
    }" > /dev/null

echo "Running $ITERATIONS iterations..."
echo ""

# Benchmark runs
TOTAL_TIME=0
TOTAL_TOKENS=0
FAILED_REQUESTS=0

for i in $(seq 1 $ITERATIONS); do
    echo -n "Iteration $i/$ITERATIONS: "
    
    # Measure response time
    START_TIME=$(date +%s.%N)
    
    RESPONSE=$(curl -s -X POST "$SERVER_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$BENCHMARK_MODEL\",
            \"prompt\": \"$PROMPT\",
            \"stream\": false
        }" 2>/dev/null)
    
    END_TIME=$(date +%s.%N)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | jq -e '.response' > /dev/null 2>&1; then
        RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        RESPONSE_TEXT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
        TOKENS=$(echo "$RESPONSE_TEXT" | wc -w)
        TOTAL_TIME=$(echo "$TOTAL_TIME + $RESPONSE_TIME" | bc)
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
        
        printf "%.3fs, %d tokens\n" "$RESPONSE_TIME" "$TOKENS"
        
        # Log iteration result
        echo "Iteration $i:" >> "$OUTPUT_FILE"
        echo "  Response time: ${RESPONSE_TIME}s" >> "$OUTPUT_FILE"
        echo "  Tokens generated: $TOKENS" >> "$OUTPUT_FILE"
        echo "  Response: $RESPONSE_TEXT" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        echo "FAILED"
        FAILED_REQUESTS=$((FAILED_REQUESTS + 1))
        echo "Iteration $i: FAILED" >> "$OUTPUT_FILE"
    fi
    
    # Small delay between requests
    sleep 0.5
done

# Calculate statistics
if [ $FAILED_REQUESTS -lt $ITERATIONS ]; then
    AVG_TIME=$(echo "scale=3; $TOTAL_TIME / ($ITERATIONS - $FAILED_REQUESTS)" | bc)
    AVG_TOKENS=$(echo "scale=1; $TOTAL_TOKENS / ($ITERATIONS - $FAILED_REQUESTS)" | bc)
    TOKENS_PER_SECOND=$(echo "scale=2; $AVG_TOKENS / $AVG_TIME" | bc)
    
    echo ""
    echo "📊 Benchmark Results"
    echo "===================="
    echo "Average response time: ${AVG_TIME}s"
    echo "Average tokens per response: $AVG_TOKENS"
    echo "Tokens per second: $TOKENS_PER_SECOND"
    echo "Failed requests: $FAILED_REQUESTS/$ITERATIONS"
    echo "Success rate: $(( (ITERATIONS - FAILED_REQUESTS) * 100 / ITERATIONS ))%"
    
    # Save summary to file
    echo "" >> "$OUTPUT_FILE"
    echo "Summary:" >> "$OUTPUT_FILE"
    echo "Average response time: ${AVG_TIME}s" >> "$OUTPUT_FILE"
    echo "Average tokens per response: $AVG_TOKENS" >> "$OUTPUT_FILE"
    echo "Tokens per second: $TOKENS_PER_SECOND" >> "$OUTPUT_FILE"
    echo "Success rate: $(( (ITERATIONS - FAILED_REQUESTS) * 100 / ITERATIONS ))%" >> "$OUTPUT_FILE"
else
    echo ""
    echo "❌ All requests failed. Check server logs."
fi

echo ""
echo "📄 Results saved to: $OUTPUT_FILE"