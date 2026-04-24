# rustcluster-mcp

**rustcluster-mcp** is a FastMCP server that acts as a clustering advisor — it
doesn't just expose [rustcluster](https://github.com/mfbaig35r/rustcluster)'s
API as tools, it brings domain intelligence to help you converge on the right
clustering configuration.

Under the hood is a knowledge graph encoding 7 algorithms, 30+ parameters,
their interactions, decision rules, anti-patterns, and diagnostic recipes.
The server queries this graph to analyze your data, recommend algorithms,
suggest justified configurations, run parameter sweeps, and diagnose problems.

## Features

- **Knowledge graph** — structured ontology of algorithms, parameters,
  interactions, data characteristics, and decision logic
- **Advisor tools** — analyze data, recommend algorithms, suggest configs with
  per-parameter rationale, optimize k, compare configurations, diagnose issues
- **Knowledge tools** — explain any algorithm or parameter in depth, validate
  configs against anti-patterns, list algorithms by capability
- **Sandbox tools** — execute clustering code in auditable Marimo notebooks via
  [marimo-sandbox](https://github.com/mfbaig35r/marimo-sandbox) with a
  pre-injected `__cluster__` context
- **Snapshot visualizations** — 2D projection maps (UMAP/t-SNE/PCA), confidence
  heatmaps, drift radar charts, hierarchical sunbursts — all as notebook artifacts

## Installation

```bash
pip install rustcluster-mcp
```

With sandbox (code execution) support:

```bash
pip install rustcluster-mcp[sandbox]
```

With visualization support (UMAP, scikit-learn):

```bash
pip install rustcluster-mcp[viz]
```

Everything:

```bash
pip install rustcluster-mcp[all]
```

## Quick start

### MCP configuration

Add to your Claude Desktop / MCP client config:

```json
{
  "mcpServers": {
    "rustcluster": {
      "command": "rustcluster-mcp"
    }
  }
}
```

### Typical workflow

**1. Analyze your data**

```
> analyze my embeddings at /data/embeddings.npy
```

The server profiles dimensionality, normalization state, density, intrinsic
dimensionality, and flags issues (NaN, mixed scale, etc.).

**2. Get a recommendation**

```
> what algorithm should I use? I have 50K OpenAI embeddings and I don't know how many clusters
```

Returns ranked recommendations with rationale — e.g., "EmbeddingCluster with
reduction_dim=128, or HDBSCAN if you want auto-k discovery."

**3. Get a justified config**

```
> give me a config for embedding_cluster on this data
```

Returns every parameter with a suggested value, the default, and a
data-specific rationale for why this value was chosen.

**4. Find optimal k**

```
> find the best k between 5 and 100
```

Runs a parameter sweep, computes silhouette/CH/DB at each k, and recommends the
best with rationale.

**5. Diagnose issues**

```
> my clusters look uneven — one cluster has 80% of the points
```

Returns the pathology signature ("One Dominant Cluster"), likely causes, and
specific fix recipes with parameter changes.

### Using `__cluster__` in sandbox runs

```python
# In any run_python call, __cluster__ is pre-injected:
data = __cluster__.load("/data/embeddings.npy")

# Fit and get metrics in one call
result = __cluster__.fit("embedding_cluster", data, n_clusters=20, reduction_dim=128)
print(result["metrics"])
# {'silhouette': 0.34, 'calinski_harabasz': 412.5, 'davies_bouldin': 0.89}

# Parameter sweep with visualization
sweep = __cluster__.sweep_k(data, "embedding_cluster", reduction_dim=128)
__cluster__.plot_sweep(sweep)  # -> sweep.png artifact

# Snapshot visualization
snap = __cluster__.snapshot()
snap.calibrate(data)
__cluster__.visualize_snapshot(data, snap, method="umap", color_by="cluster",
                               adaptive_threshold=True)  # -> snapshot_map.png artifact

# Drift monitoring
report = snap.drift_report(new_data)
__cluster__.visualize_drift(snap, report)  # -> drift_radar.png artifact
```

## Tools reference

### Advisor tools (data-driven)

| Tool | Description |
|------|-------------|
| `analyze_data(data_path, source_type)` | Profile data: dims, normalization, density, intrinsic dimensionality |
| `recommend_algorithm(n_samples, n_features, ...)` | Ranked algorithm recommendations from data profile + requirements |
| `suggest_config(algorithm, n_samples, n_features, ...)` | Full justified config with per-parameter rationale |
| `optimize_k(data_path, algorithm, k_range, ...)` | Parameter sweep across k values with silhouette/CH/DB |
| `compare_configs(data_path, configs)` | Head-to-head comparison of 2+ configurations |
| `evaluate_clusters(data_path, labels_path)` | Full quality assessment with pathology detection |
| `diagnose(description, algorithm, params, metrics, ...)` | Symptom-based diagnosis with fix recipes |

### Knowledge tools (no data needed)

| Tool | Description |
|------|-------------|
| `explain_algorithm(algorithm)` | Deep dive: how it works, params, strengths, relationships |
| `explain_parameter(algorithm, parameter)` | Semantics, interactions, tuning advice |
| `list_algorithms(category, supports_noise, ...)` | Browse algorithms with capability filtering |
| `check_config(algorithm, params, n_features, ...)` | Validate a config against anti-patterns |

### Sandbox tools (via marimo-sandbox)

| Tool | Description |
|------|-------------|
| `run_python(code, ...)` | Execute Python with `__cluster__` pre-injected |
| `get_run` / `list_runs` | Browse run history |
| `get_run_outputs(run_id)` | Structured `__outputs__` dict from a run |
| `list_artifacts` / `read_artifact` | Access plots, labels, CSVs from runs |
| `open_notebook(run_id)` | Open a run in Marimo UI for interactive editing |
| `rerun` / `diff_runs` | Re-execute or compare runs |
| `approve_run` / `delete_run` / `check_setup` | Lifecycle management |

## Knowledge graph

The server's intelligence comes from a structured knowledge graph with 5 layers:

| Layer | What it encodes |
|-------|----------------|
| **Algorithms** | 7 algorithms: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Agglomerative, EmbeddingCluster, SphericalKMeans |
| **Parameters** | 30+ parameters with types, defaults, ranges, sensitivity ratings, interactions |
| **Data characteristics** | Dimensionality, scale, normalization, density profile, source type |
| **Decision rules** | 9 condition-based rules mapping data profiles to algorithm recommendations |
| **Diagnostics** | 3 metric interpretations, 8 anti-patterns, 6 pathology signatures with fix recipes |

### Anti-patterns detected

- Euclidean distance on high-dimensional embeddings
- KMeans without PCA on 1000+ dimensions
- DBSCAN on data above 16 dimensions
- DBSCAN on variable-density data
- HDBSCAN on datasets above 50K points
- Ward linkage with non-Euclidean metric
- Panicking at low silhouette scores on high-dimensional data
- Matryoshka reduction on non-Matryoshka models

### Pathologies diagnosed

- One dominant cluster (>60% of points)
- Excessive singleton clusters
- Everything labeled as noise
- Unstable clustering across runs
- Highly uneven cluster sizes
- Slow convergence / hitting max_iter

## Snapshot visualizations

The `__cluster__` context includes 4 visualization methods, all saved as
notebook artifacts:

| Method | What it shows |
|--------|--------------|
| `visualize_snapshot(X, snap)` | 2D UMAP/t-SNE/PCA projection colored by cluster, opacity by confidence, rejected points as gray X markers |
| `visualize_confidence(X, snap)` | Per-cluster confidence distribution heatmap |
| `visualize_drift(snap, report)` | Radar chart of per-cluster relative drift + direction drift |
| `visualize_hierarchical(X, hier_snap)` | Sunburst chart: inner ring = root clusters, outer ring = child clusters |

## Development

```bash
git clone https://github.com/mfbaig35r/rustcluster-mcp
cd rustcluster-mcp
pip install -e ".[all]"
```

```bash
# Lint
ruff check src/ tests/

# Type-check
mypy src/rustcluster_mcp/ --ignore-missing-imports

# Tests
pytest tests/ -v
```

## License

MIT
