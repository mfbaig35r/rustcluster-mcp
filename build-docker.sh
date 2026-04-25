#!/usr/bin/env bash
# Build the rustcluster-mcp Docker image.
#
# This script creates a temporary build context with both rustcluster (Rust source)
# and rustcluster-mcp (Python MCP server), then builds the multi-stage image.
#
# Usage:
#   ./build-docker.sh                          # local build for current arch
#   ./build-docker.sh --push                   # multi-arch build + push to registry
#   ./build-docker.sh --tag ghcr.io/user/img   # custom tag
#   ./build-docker.sh --push --tag ghcr.io/mfbaig35r/rustcluster-mcp

set -euo pipefail

TAG="rustcluster-mcp:latest"
PUSH=false
PLATFORMS="linux/amd64,linux/arm64"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH=true; shift ;;
        --tag)  TAG="$2"; shift 2 ;;
        *)      TAG="$1"; shift ;;
    esac
done

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

if [ "$PUSH" = true ]; then
    echo "Building multi-arch image: $TAG"
    echo "Platforms: $PLATFORMS"
    echo ""

    # Ensure buildx builder exists
    docker buildx inspect multiarch >/dev/null 2>&1 || \
        docker buildx create --name multiarch --use

    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "$TAG" \
        --push \
        "$BUILD_CTX"
else
    echo "Building image for current architecture: $TAG"
    docker build -t "$TAG" "$BUILD_CTX"
fi

echo ""
echo "Done! Run with:"
echo "  docker run --rm -i -v /path/to/data:/data $TAG"
