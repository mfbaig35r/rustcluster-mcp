#!/usr/bin/env bash
# Build the rustcluster-mcp Docker image.
#
# This script creates a temporary build context with both rustcluster (Rust source)
# and rustcluster-mcp (Python MCP server), then builds the multi-stage image.
#
# Usage:
#   ./build-docker.sh                    # builds with default tag
#   ./build-docker.sh my-tag             # builds with custom tag

set -euo pipefail

TAG="${1:-rustcluster-mcp:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUSTCLUSTER_DIR="${SCRIPT_DIR}/../rustcluster"

if [ ! -d "$RUSTCLUSTER_DIR/src" ]; then
    echo "Error: rustcluster source not found at $RUSTCLUSTER_DIR"
    echo "Expected sibling directory: ../rustcluster/"
    exit 1
fi

# Create temporary build context
BUILD_CTX=$(mktemp -d)
trap 'rm -rf "$BUILD_CTX"' EXIT

echo "Preparing build context..."

# Copy rustcluster source (only what's needed for the build)
mkdir -p "$BUILD_CTX/rustcluster"
cp "$RUSTCLUSTER_DIR/Cargo.toml" "$BUILD_CTX/rustcluster/"
cp "$RUSTCLUSTER_DIR/Cargo.lock" "$BUILD_CTX/rustcluster/"
cp "$RUSTCLUSTER_DIR/pyproject.toml" "$BUILD_CTX/rustcluster/"
cp "$RUSTCLUSTER_DIR/README.md" "$BUILD_CTX/rustcluster/"
cp -r "$RUSTCLUSTER_DIR/src" "$BUILD_CTX/rustcluster/src"
cp -r "$RUSTCLUSTER_DIR/python" "$BUILD_CTX/rustcluster/python"

# Copy rustcluster-mcp source
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_CTX/"
cp "$SCRIPT_DIR/pyproject.toml" "$BUILD_CTX/"
cp "$SCRIPT_DIR/README.md" "$BUILD_CTX/"
cp -r "$SCRIPT_DIR/src" "$BUILD_CTX/src"

echo "Building image: $TAG"
docker build -t "$TAG" "$BUILD_CTX"

echo ""
echo "Done! Run with:"
echo "  docker run --rm -i -v /path/to/data:/data $TAG"
