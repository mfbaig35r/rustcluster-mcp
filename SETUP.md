# Setup Guide

Three ways to run rustcluster-mcp, from easiest to most flexible.

## Option 1: Docker (recommended)

No Python, Rust, or dependency management needed. Works on macOS, Windows, and Linux.

### Install

```bash
docker pull fbaig4/rustcluster-mcp:latest
```

### Configure Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "rustcluster": {
    "type": "stdio",
    "command": "docker",
    "args": [
      "run", "--rm", "-i",
      "-v", "/path/to/your/data:/data",
      "fbaig4/rustcluster-mcp:latest"
    ]
  }
}
```

Replace `/path/to/your/data` with the directory containing your data files.
Inside Claude, reference files as `/data/filename.npy`.

### Multiple data directories

Mount as many directories as you need:

```json
{
  "rustcluster": {
    "type": "stdio",
    "command": "docker",
    "args": [
      "run", "--rm", "-i",
      "-v", "/Users/me/embeddings:/data/embeddings",
      "-v", "/Users/me/results:/data/results",
      "fbaig4/rustcluster-mcp:latest"
    ]
  }
}
```

### Update

```bash
docker pull fbaig4/rustcluster-mcp:latest
```

---

## Option 2: pip / uv install

Requires Python 3.11+ and a working rustcluster installation (Rust-backed,
needs the native extension compiled for your platform).

### Install

```bash
pip install rustcluster-mcp
```

Or with uv:

```bash
uv pip install rustcluster-mcp
```

### Optional extras

```bash
# Sandbox: code execution in Marimo notebooks
pip install rustcluster-mcp[sandbox]

# Visualization: UMAP, t-SNE for snapshot plots
pip install rustcluster-mcp[viz]

# Everything
pip install rustcluster-mcp[all]
```

### Configure Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "rustcluster": {
    "type": "stdio",
    "command": "rustcluster-mcp"
  }
}
```

Or if using uv with a project directory:

```json
{
  "rustcluster": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "run",
      "--directory", "/path/to/rustcluster-mcp",
      "rustcluster-mcp"
    ]
  }
}
```

### Data access

With a local install, the server has access to your full filesystem. Use
absolute paths: `fit(data_path="/Users/me/embeddings.npy", ...)`.

---

## Option 3: From source (development)

For contributing or modifying the server.

### Prerequisites

- Python 3.11+
- Rust toolchain ([rustup.rs](https://rustup.rs)) — only if rustcluster
  isn't already installed
- uv (recommended) or pip

### Install

```bash
git clone https://github.com/mfbaig35r/rustcluster-mcp
cd rustcluster-mcp
uv venv && uv pip install -e ".[dev]"
```

If rustcluster isn't on PyPI for your platform, build it from source:

```bash
git clone https://github.com/mfbaig35r/rustcluster ../rustcluster
cd ../rustcluster
pip install maturin
maturin develop --release
cd ../rustcluster-mcp
```

### Run tests

```bash
# Fast tests (no rustcluster needed — mocked)
uv run pytest tests/ -m "not slow" -v

# Full suite (needs rustcluster installed)
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/rustcluster_mcp/ --ignore-missing-imports
```

### Configure Claude Code

```json
{
  "rustcluster": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "run",
      "--directory", "/path/to/rustcluster-mcp",
      "rustcluster-mcp"
    ]
  }
}
```

### Build Docker image locally

```bash
./build-docker.sh
```

Requires the rustcluster source at `../rustcluster/` (sibling directory).
Builds a multi-stage image: Rust compilation in a builder stage, slim
Python runtime in the final image.

---

## Data preparation

The server accepts `.npy`, `.npz`, and `.parquet` files. If your embeddings
live elsewhere, extract them first:

### From a Python script

```python
import numpy as np

# From a list of embeddings
embeddings = [...]  # list of float lists
np.save("embeddings.npy", np.array(embeddings))

# From a pandas DataFrame
df = pd.read_parquet("data.parquet")
np.save("embeddings.npy", df[["dim_0", "dim_1", ..., "dim_1535"]].to_numpy())
```

### From a database

```python
import numpy as np

# DuckDB
import duckdb
conn = duckdb.connect("my.db")
df = conn.execute("SELECT embedding FROM items").fetchdf()
np.save("embeddings.npy", np.vstack(df["embedding"].values))

# Snowflake (via snowbox or connector)
# Databricks (via ASI or spark)
# Extract → DataFrame → np.save
```

### Parquet files

If your data is already in parquet format with numeric columns, the server
reads it directly — no conversion needed. It selects all numeric columns
automatically.

```
> analyze my data at /data/embeddings.parquet
```

---

## Troubleshooting

### "rustcluster not found" (pip install)

rustcluster is a Rust-backed native extension. If no pre-built wheel exists
for your platform, you'll need to build from source (see Option 3). The Docker
image avoids this entirely.

### Docker: "no matching manifest for linux/amd64"

The published image is currently arm64 (Apple Silicon). For Intel/Windows,
build locally:

```bash
./build-docker.sh
```

Or use the multi-arch build:

```bash
./build-docker.sh --push --tag fbaig4/rustcluster-mcp:latest
```

### "marimo-sandbox not installed" when using run_python

The sandbox tools are optional. Install with:

```bash
pip install rustcluster-mcp[sandbox]
```

Or use the `fit` tool instead — it handles most clustering tasks without
needing the sandbox.

### Data file not found in Docker

Make sure the directory containing your data is mounted with `-v`:

```bash
docker run --rm -i -v /Users/me/data:/data fbaig4/rustcluster-mcp:latest
```

Then reference files as `/data/filename.npy`, not the local path.
