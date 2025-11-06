# Docker setup for Rust Ollama

FROM rust:1.70 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy cargo files
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY build.rs ./

# Build application
RUN cargo build --release

# Runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -u 1000 -s /bin/bash ollama

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/ollama ./
COPY --from=builder /app/target/release/ollama_cli ./

# Create data directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R ollama:ollama /app

USER ollama

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ./ollama_cli health || exit 1

# Expose port
EXPOSE 11434

# Environment variables
ENV RUST_LOG=info
ENV DATABASE_PATH=/app/data/ollama.db
ENV MODELS_DIR=/app/models

# Default command
CMD ["./ollama", "serve"]