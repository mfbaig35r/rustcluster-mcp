# rustcluster-mcp

**rustcluster-mcp** is a FastMCP server that acts as a clustering advisor — it
doesn't just expose [rustcluster](https://github.com/mfbaig35r/rustcluster)'s
API as tools, it brings domain intelligence to help you converge on the right
clustering configuration.

Under the hood is a knowledge graph encoding 7 algorithms, 30+ parameters,
their interactions, decision rules, anti-patterns, and diagnostic recipes.
The server queries this graph to analyze your data, recommend algorithms,
suggest justified configurations, run clustering directly, and diagnose problems.

## Quick start

The fastest way to get started — no Python or Rust install needed:

```bash
docker pull fbaig4/rustcluster-mcp:latest
```

Add to your Claude Code config (`~/.claude.json`):

```json
{
  "mcpServers": {
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
}
```

Replace `/path/to/your/data` with the directory containing your `.npy`, `.npz`,
or `.parquet` files. Inside Claude, your data is accessible at `/data/`.

For other setup options (pip install, from source), see [SETUP.md](SETUP.md).

## How it works

### Phase 1 — Plan (advisory tools, no clustering runs)

```
> analyze my embeddings at /data/embeddings.npy
```
Profiles dimensionality, normalization, density, intrinsic dimensionality.

```
> what algorithm should I use? 50K OpenAI embeddings, unknown cluster count
```
Returns ranked recommendations: "EmbeddingCluster with reduction_dim=128,
or HDBSCAN if you want auto-k discovery."

```
> give me a config for embedding_cluster
```
Every parameter with a suggested value and data-specific rationale.

### Phase 2 — Execute (run clustering directly)

```
> cluster my data with embedding_cluster, 20 clusters
```
The `fit` tool runs clustering and returns labels, metrics (silhouette, CH, DB),
and cluster size distribution. No code writing needed.

```
> find the best k between 5 and 100
```
Sweeps k values with all three metrics, recommends the best.

### Phase 3 — Refine

```
> my clusters look uneven — one cluster has 80% of the points
```
Returns pathology signature, likely causes, and specific fix recipes.

## Features

- **Knowledge graph** — 7 algorithms, 30+ parameters, 9 decision rules, 8
  anti-patterns, 6 pathology signatures with fix recipes
- **Direct clustering** — `fit` tool runs clustering on data files without
  needing a sandbox or writing code
- **Advisor tools** — analyze data, recommend algorithms, suggest configs,
  optimize k, compare configs, evaluate quality, diagnose issues
- **Knowledge tools** — explain any algorithm or parameter, validate configs,
  list algorithms by capability
- **Sandbox tools** — execute custom Python in auditable Marimo notebooks with
  a pre-injected `__cluster__` helper (optional, requires marimo-sandbox)
- **Snapshot visualizations** — 2D projections, confidence heatmaps, drift radar
  charts, hierarchical sunbursts
- **Data formats** — `.npy`, `.npz`, and `.parquet` (parquet requires pyarrow)

## Tools reference

### Execution tools

| Tool | Description |
|------|-------------|
| `fit(data_path, algorithm, ...)` | Run clustering directly — returns labels, metrics, sizes |
| `optimize_k(data_path, algorithm, k_range, ...)` | Sweep k values with silhouette/CH/DB scores |
| `compare_configs(data_path, configs)` | Head-to-head comparison of 2+ configurations |
| `evaluate_clusters(data_path, labels_path)` | Quality assessment with pathology detection |

### Advisor tools

| Tool | Description |
|------|-------------|
| `analyze_data(data_path, source_type)` | Profile data: dims, normalization, density |
| `recommend_algorithm(n_samples, n_features, ...)` | Ranked recommendations from data profile + requirements |
| `suggest_config(algorithm, n_samples, n_features, ...)` | Justified config with per-parameter rationale |
| `diagnose(description, metrics, cluster_sizes, ...)` | Symptom-based diagnosis with fix recipes |

### Knowledge tools

| Tool | Description |
|------|-------------|
| `explain_algorithm(algorithm)` | How it works, params, strengths, relationships |
| `explain_parameter(algorithm, parameter)` | Semantics, interactions, tuning advice |
| `list_algorithms(category, supports_noise, ...)` | Browse with capability filtering |
| `check_config(algorithm, params, ...)` | Validate against anti-patterns |

### Sandbox tools (optional, requires marimo-sandbox)

| Tool | Description |
|------|-------------|
| `run_python(code, ...)` | Execute Python with `__cluster__` pre-injected |
| `get_run` / `list_runs` | Browse run history |
| `list_artifacts` / `read_artifact` | Access plots, labels, CSVs from runs |
| `open_notebook(run_id)` | Open in Marimo UI for interactive editing |
| `rerun` / `diff_runs` / `approve_run` / `delete_run` | Run lifecycle |

## Knowledge graph

| Layer | What it encodes |
|-------|----------------|
| **Algorithms** | KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Agglomerative, EmbeddingCluster, SphericalKMeans |
| **Parameters** | 30+ parameters with types, defaults, ranges, sensitivity, interactions |
| **Data characteristics** | Dimensionality, scale, normalization, density profile, source type |
| **Decision rules** | 9 condition-based rules mapping data profiles to recommendations |
| **Diagnostics** | 3 metric interpretations, 8 anti-patterns, 6 pathology signatures |

## Development

See [SETUP.md](SETUP.md) for detailed setup instructions. Quick version:

```bash
git clone https://github.com/mfbaig35r/rustcluster-mcp
cd rustcluster-mcp
uv venv && uv pip install -e ".[dev]"
uv run ruff check src/ tests/
uv run pytest tests/ -m "not slow" -v
```

## License

MIT
