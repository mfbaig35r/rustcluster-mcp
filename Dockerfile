# Multi-stage build for rustcluster-mcp
#
# Build: docker build -t rustcluster-mcp .
#   (requires rustcluster source at ../rustcluster — see docker-bake or build script)
#
# Run:   docker run --rm -i -v /path/to/data:/data rustcluster-mcp

# ── Stage 1: Build rustcluster from source ────────────────────────────────
FROM python:3.11-slim AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc g++ make pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

# Copy rustcluster source (Rust + Python)
COPY rustcluster/ /build/rustcluster/
WORKDIR /build/rustcluster
RUN maturin build --release --strip -o /wheels


# ── Stage 2: Build rustcluster-mcp wheel ──────────────────────────────────
FROM python:3.11-slim AS mcp-builder

COPY pyproject.toml README.md /build/mcp/
COPY src/ /build/mcp/src/
WORKDIR /build/mcp
RUN pip wheel --no-deps -w /wheels .


# ── Stage 3: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim

# Combine wheels from both build stages
COPY --from=rust-builder /wheels /wheels/rustcluster/
COPY --from=mcp-builder /wheels /wheels/mcp/

# Install everything
RUN pip install --no-cache-dir \
        /wheels/rustcluster/*.whl \
        /wheels/mcp/*.whl \
    && rm -rf /wheels

# Mount point for user data
RUN mkdir /data
VOLUME /data

# MCP stdio transport
ENTRYPOINT ["rustcluster-mcp"]
