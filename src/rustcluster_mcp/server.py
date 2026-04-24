"""
rustcluster MCP server — Clustering advisor powered by a domain knowledge graph.

Not just API wrappers: this server *reasons* about your clustering problem.
It analyzes data, recommends algorithms, suggests configurations, runs
parameter sweeps, diagnoses issues, and explains why.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from .context import CLUSTER_CONTEXT_CODE, CLUSTER_REQUIRED_PACKAGES
from .knowledge_graph import (
    ALGORITHMS,
    METRIC_INTERPRETATIONS,
    PATHOLOGY_SIGNATURES,
    AlgorithmCategory,
    DataProfile,
    DataScale,
    DensityProfile,
    Dimensionality,
    NormalizationState,
    check_anti_patterns,
    diagnose_results,
    get_algorithm,
    get_parameters_for,
    get_relationships_for,
    match_decision_rules,
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "rustcluster",
    instructions="""\
Clustering advisor for the rustcluster library. Deep knowledge of 7 algorithms,
30+ parameters, their interactions, and when to use what.

## Data prep

Data must be in .npy, .npz, or .parquet format. If embeddings live in a database
(DuckDB, Snowflake, Databricks), extract them first:
    import numpy as np
    np.save("embeddings.npy", df[embedding_columns].to_numpy())

## Workflow: Plan → Execute → Refine

### Phase 1 — Plan (advisory tools, no clustering runs)
1. analyze_data — profile your data (dims, normalization, density, quality checks)
2. recommend_algorithm — get ranked algorithm suggestions for your data + requirements
3. suggest_config — get a justified config with per-parameter rationale

### Phase 2 — Execute (run clustering directly)
4. fit — run clustering on your data. Takes a file path + algorithm + params,
   returns labels, metrics, and cluster sizes. This is the primary execution tool.
   The output of suggest_config maps directly to fit's parameters.
5. optimize_k — sweep k values to find the optimum (silhouette, CH, DB scores)

### Phase 3 — Refine
6. evaluate_clusters — assess quality with all metrics + pathology detection
7. diagnose — describe what looks wrong, get specific fix recipes
8. compare_configs — run two configs head-to-head

## When to use run_python instead of fit

Use fit for standard clustering. Use run_python (sandbox) for custom analysis:
visualization, snapshot operations, hierarchical clustering pipelines, or anything
that needs multiple steps. run_python injects a __cluster__ helper with methods:
- __cluster__.load(path) / .fit(algorithm, X, **params) / .sweep_k(X, ...)
- __cluster__.snapshot() / .visualize_snapshot(X, snap) / .visualize_confidence(X, snap)
- __cluster__.visualize_drift(snap, report) / .visualize_hierarchical(X, hier_snap)
- __cluster__.plot_sweep(results) / .plot_sizes(sizes) / .profile(X)

## Knowledge tools (no data needed)
- explain_algorithm — deep dive into any algorithm
- explain_parameter — understand what a parameter does and how to tune it
- list_algorithms — browse all algorithms with capability filtering
- check_config — validate a configuration against known anti-patterns
""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_dimensionality(d: int) -> Dimensionality:
    """Classify feature count into dimensionality tier."""
    if d <= 10:
        return Dimensionality.LOW
    elif d <= 100:
        return Dimensionality.MEDIUM
    elif d <= 768:
        return Dimensionality.HIGH
    return Dimensionality.VERY_HIGH


def _classify_scale(n: int) -> DataScale:
    """Classify sample count into scale tier."""
    if n < 1_000:
        return DataScale.SMALL
    elif n < 50_000:
        return DataScale.MEDIUM
    elif n < 500_000:
        return DataScale.LARGE
    return DataScale.VERY_LARGE


def success_response(**data: Any) -> dict[str, Any]:
    return {"status": "success", **data}


def error_response(message: str, error_type: str = "error") -> dict[str, Any]:
    return {"status": "error", "error_type": error_type, "error": message}


def _load_array(path: str) -> np.ndarray:
    """Load a numpy array from .npy, .npz, or .parquet file."""
    if path.endswith(".npz"):
        with np.load(path) as f:
            key = list(f.keys())[0]
            return f[key]
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for parquet support. "
                "Install with: pip install pyarrow"
            )
        table = pq.read_table(path)
        return table.to_pandas().select_dtypes(include=["number"]).to_numpy()
    return np.load(path)


def _profile_data(X: np.ndarray) -> tuple[dict[str, Any], DataProfile]:
    """Compute data characteristics for the knowledge graph."""
    n, d = X.shape
    dim = _classify_dimensionality(d)
    scale = _classify_scale(n)

    # L2 normalization check
    norms = np.linalg.norm(X, axis=1)
    norm_std = float(np.std(norms))
    norm_mean = float(np.mean(norms))
    is_l2 = bool(np.allclose(norms, 1.0, atol=1e-3))

    if is_l2:
        norm_state = NormalizationState.L2_NORMALIZED
    elif norm_std / max(norm_mean, 1e-10) > 0.5:
        norm_state = NormalizationState.MIXED_SCALE
    else:
        norm_state = NormalizationState.RAW

    # Density profile (heuristic via NN distances on a sample)
    density = DensityProfile.UNIFORM  # default
    if n > 50:
        sample_size = min(n, 2000)
        idx = np.random.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
        sample = X[idx]

        # Compute pairwise distances for a small subset to estimate density variation
        from scipy.spatial.distance import cdist
        sub = sample[:min(500, len(sample))]
        dists = cdist(sub, sub)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)

        nn_cv = float(np.std(nn_dists) / max(np.mean(nn_dists), 1e-10))
        if nn_cv > 1.5:
            density = DensityProfile.CLUSTERED_WITH_NOISE
        elif nn_cv > 0.7:
            density = DensityProfile.VARIABLE
        else:
            density = DensityProfile.UNIFORM

    # Intrinsic dimensionality estimate (via PCA explained variance)
    intrinsic_dim = None
    if d > 10 and n > d:
        try:
            sample_size = min(n, 5000)
            idx = np.random.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
            sample = X[idx].astype(np.float64)
            sample -= sample.mean(axis=0)
            cov = np.cov(sample, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            intrinsic_dim = int(np.searchsorted(cumvar, 0.95) + 1)
        except Exception:
            pass

    profile_dict = {
        "n_samples": n,
        "n_features": d,
        "dimensionality": dim.value,
        "scale": scale.value,
        "normalization": norm_state.value,
        "is_l2_normalized": is_l2,
        "norm_mean": round(norm_mean, 4),
        "norm_std": round(norm_std, 4),
        "density_profile": density.value,
        "intrinsic_dimensionality": intrinsic_dim,
        "dtype": str(X.dtype),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
    }
    data_profile = DataProfile(
        dimensionality=dim,
        scale=scale,
        normalization=norm_state,
        density=density,
        source_type=None,
        intrinsic_dim_estimate=intrinsic_dim,
    )
    return profile_dict, data_profile


# ===========================================================================
# Phase 1 — Understand the data
# ===========================================================================

@mcp.tool()
async def analyze_data(
    data_path: str,
    source_type: str | None = None,
) -> dict[str, Any]:
    """Analyze data characteristics to inform algorithm and parameter selection.

    Computes: dimensionality, scale, normalization state, density profile,
    intrinsic dimensionality estimate, and data quality checks (NaN, Inf).

    Args:
        data_path: Path to .npy, .npz, or .parquet file containing the data matrix (n_samples, n_features).
        source_type: Optional hint about the embedding source. One of:
            "openai", "cohere", "voyage", "sentence_transformers", "tfidf", "tabular", or None.
            Helps the advisor give more specific recommendations.

    Returns:
        Data profile with characteristics and initial recommendations.
    """
    try:
        X = _load_array(data_path)
        if X.ndim != 2:
            return error_response(f"Expected 2D array, got {X.ndim}D", "validation_error")

        profile, data_profile = _profile_data(X)
        data_profile.source_type = source_type
        profile["source_type"] = source_type

        # Get initial recommendations
        rules = match_decision_rules(data_profile)
        if rules:
            top = rules[0]
            profile["initial_recommendation"] = {
                "algorithm": top.recommended_algorithm,
                "confidence": top.confidence,
                "rationale": top.rationale,
                "suggested_params": top.suggested_params,
            }
            if len(rules) > 1:
                profile["alternatives"] = [
                    {
                        "algorithm": r.recommended_algorithm,
                        "confidence": r.confidence,
                        "rationale": r.rationale,
                    }
                    for r in rules[1:3]
                ]

        # Check for obvious anti-patterns based on data alone
        warnings = []
        if profile["has_nan"]:
            warnings.append(
                "Data contains NaN values — all rustcluster algorithms will reject this. "
                "Clean your data first."
            )
        if profile["has_inf"]:
            warnings.append(
                "Data contains Inf values — all rustcluster algorithms will reject this. "
                "Clean your data first."
            )
        if profile["normalization"] == "mixed_scale" and source_type in ("openai", "cohere", "voyage"):
            warnings.append(
                f"Data from {source_type} is expected to be L2-normalized "
                "but appears mixed-scale. Check for preprocessing issues."
            )
        if profile["intrinsic_dimensionality"] and profile["n_features"] > 100:
            ratio = profile["intrinsic_dimensionality"] / profile["n_features"]
            if ratio < 0.2:
                warnings.append(
                    f"Intrinsic dimensionality ({profile['intrinsic_dimensionality']}) is much lower than "
                    f"ambient dimensionality ({profile['n_features']}). PCA reduction will be very effective."
                )
        if warnings:
            profile["warnings"] = warnings

        return success_response(**profile)

    except FileNotFoundError:
        return error_response(f"File not found: {data_path}", "file_error")
    except Exception as exc:
        return error_response(f"Analysis failed: {exc}", "internal_error")


# ===========================================================================
# Direct clustering — no sandbox needed
# ===========================================================================

@mcp.tool()
async def fit(
    data_path: str,
    algorithm: str = "embedding_cluster",
    n_clusters: int | None = None,
    reduction_dim: int | None = 128,
    metric: str = "euclidean",
    min_cluster_size: int = 15,
    eps: float = 0.5,
    min_samples: int = 5,
    linkage: str = "ward",
    n_init: int = 5,
    random_state: int = 0,
    save_labels: str | None = None,
    save_centers: str | None = None,
) -> dict[str, Any]:
    """Run clustering directly on a data file and return results with metrics.

    This is the primary tool for actually clustering data — no sandbox needed.
    Pass a data file, pick an algorithm, and get back labels, metrics, and
    cluster size distribution.

    Args:
        data_path: Path to .npy, .npz, or .parquet file.
        algorithm: Algorithm to use. One of:
            "kmeans", "minibatch_kmeans", "dbscan", "hdbscan",
            "agglomerative", "embedding_cluster".
        n_clusters: Number of clusters (required for kmeans, minibatch_kmeans,
            agglomerative, embedding_cluster). Ignored for dbscan/hdbscan.
        reduction_dim: PCA reduction dimension (embedding_cluster only). None to skip.
        metric: Distance metric: "euclidean", "cosine", "manhattan" (algorithm-dependent).
        min_cluster_size: Minimum cluster size (hdbscan only).
        eps: Neighborhood radius (dbscan only).
        min_samples: Core point threshold (dbscan/hdbscan only).
        linkage: Merge criterion (agglomerative only): "ward", "complete", "average", "single".
        n_init: Number of random restarts (kmeans/embedding_cluster only).
        random_state: Random seed for reproducibility.
        save_labels: If set, save cluster labels to this .npy path.
        save_centers: If set, save cluster centers to this .npy path (centroid-based only).

    Returns:
        Clustering results with labels (as list), metrics (silhouette, CH, DB),
        cluster sizes, and optional saved file paths.
    """
    try:
        X = _load_array(data_path)
        if X.ndim != 2:
            return error_response(f"Expected 2D array, got {X.ndim}D", "validation_error")

        import rustcluster
        from rustcluster.experimental import EmbeddingCluster

        # Build model
        if algorithm == "kmeans":
            if n_clusters is None:
                return error_response("n_clusters is required for kmeans", "validation_error")
            model = rustcluster.KMeans(
                n_clusters=n_clusters, metric=metric, n_init=n_init,
                random_state=random_state,
            )
        elif algorithm == "minibatch_kmeans":
            if n_clusters is None:
                return error_response("n_clusters is required for minibatch_kmeans", "validation_error")
            model = rustcluster.MiniBatchKMeans(
                n_clusters=n_clusters, metric=metric, random_state=random_state,
            )
        elif algorithm == "dbscan":
            model = rustcluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        elif algorithm == "hdbscan":
            model = rustcluster.HDBSCAN(
                min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
            )
        elif algorithm == "agglomerative":
            if n_clusters is None:
                return error_response("n_clusters is required for agglomerative", "validation_error")
            model = rustcluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage, metric=metric,
            )
        elif algorithm == "embedding_cluster":
            if n_clusters is None:
                return error_response("n_clusters is required for embedding_cluster", "validation_error")
            model = EmbeddingCluster(
                n_clusters=n_clusters, reduction_dim=reduction_dim,
                n_init=n_init, random_state=random_state,
            )
        else:
            valid = "kmeans, minibatch_kmeans, dbscan, hdbscan, agglomerative, embedding_cluster"
            return error_response(f"Unknown algorithm '{algorithm}'. Valid: {valid}", "validation_error")

        # Fit
        model.fit(X)
        labels = model.labels_

        # Eval data (use reduced for EmbeddingCluster)
        eval_data = X
        if hasattr(model, "reduced_data_") and model.reduced_data_ is not None:
            eval_data = model.reduced_data_

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        # Metrics
        metrics = {}
        if n_clusters_found >= 2:
            sil = float(rustcluster.silhouette_score(eval_data, labels))
            ch = float(rustcluster.calinski_harabasz_score(eval_data, labels))
            db = float(rustcluster.davies_bouldin_score(eval_data, labels))
            metrics = {
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "davies_bouldin": round(db, 4),
            }

        # Cluster sizes
        non_noise = labels[labels >= 0]
        _, counts = np.unique(non_noise, return_counts=True)
        sizes = sorted(counts.tolist(), reverse=True)

        result_data: dict[str, Any] = {
            "algorithm": algorithm,
            "n_clusters": n_clusters_found,
            "n_noise": n_noise,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "metrics": metrics,
            "cluster_sizes": sizes,
        }

        # Labels: include inline for small datasets, auto-save for large ones
        _LABEL_INLINE_LIMIT = 10_000
        if len(labels) <= _LABEL_INLINE_LIMIT:
            result_data["labels"] = labels.tolist()
        else:
            result_data["labels_truncated"] = labels[:_LABEL_INLINE_LIMIT].tolist()
            result_data["labels_count"] = len(labels)
            # Auto-save to temp file if caller didn't specify save_labels
            if not save_labels:
                import tempfile
                auto_path = tempfile.mktemp(suffix="_labels.npy", prefix="rustcluster_")
                np.save(auto_path, labels)
                save_labels = auto_path
                result_data["labels_auto_saved"] = True

        # Optional model attributes
        if hasattr(model, "inertia_"):
            result_data["inertia"] = round(float(model.inertia_), 2)
        if hasattr(model, "objective_"):
            result_data["objective"] = round(float(model.objective_), 4)
        if hasattr(model, "n_iter_"):
            result_data["n_iter"] = int(model.n_iter_)

        # Save artifacts
        if save_labels:
            if "labels_path" not in result_data:  # don't double-save
                np.save(save_labels, labels)
            result_data["labels_path"] = save_labels
        if save_centers and hasattr(model, "cluster_centers_"):
            np.save(save_centers, model.cluster_centers_)
            result_data["centers_path"] = save_centers

        return success_response(**result_data)

    except FileNotFoundError:
        return error_response(f"File not found: {data_path}", "file_error")
    except ImportError as exc:
        return error_response(str(exc), "dependency_error")
    except Exception as exc:
        return error_response(f"Clustering failed: {exc}", "internal_error")


# ===========================================================================
# Phase 2 — Recommend algorithm & configuration
# ===========================================================================

@mcp.tool()
async def recommend_algorithm(
    n_samples: int,
    n_features: int,
    normalization: str = "unknown",
    density: str = "uniform",
    source_type: str | None = None,
    know_k: bool | None = None,
    need_noise_detection: bool = False,
    need_soft_membership: bool = False,
    need_dendrogram: bool = False,
    need_predict: bool = False,
    need_snapshot: bool = False,
    max_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Recommend the best clustering algorithm for your problem.

    Combines data characteristics with your requirements to select the right
    algorithm. Returns ranked recommendations with rationale.

    Args:
        n_samples: Number of data points.
        n_features: Number of features/dimensions per point.
        normalization: Data normalization state. One of:
            "l2_normalized", "raw", "mixed_scale", "unknown".
        density: Density profile of the data. One of:
            "uniform", "variable", "clustered_with_noise", "hierarchical".
        source_type: Embedding source hint (e.g., "openai", "cohere", "tabular").
        know_k: Whether you know the desired number of clusters.
            True = you'll specify k. False = want auto-discovery. None = unsure.
        need_noise_detection: Whether you need outlier/noise labeling.
        need_soft_membership: Whether you need probability-based membership.
        need_dendrogram: Whether you need a hierarchical tree structure.
        need_predict: Whether you need to assign new points without refitting.
        need_snapshot: Whether you need to save/load cluster state for production.
        max_memory_gb: Memory budget in GB (helps avoid O(n^2) algorithms on large data).

    Returns:
        Ranked algorithm recommendations with rationale and suggested parameters.
    """
    try:
        # Build data profile
        dim = _classify_dimensionality(n_features)
        scale = _classify_scale(n_samples)
        norm_state = NormalizationState(normalization) if normalization != "unknown" else NormalizationState.UNKNOWN

        profile = DataProfile(
            dimensionality=dim,
            scale=scale,
            normalization=norm_state,
            density=DensityProfile(density),
            source_type=source_type,
            intrinsic_dim_estimate=None,
        )

        # Get rule-based recommendations
        rules = match_decision_rules(profile, requires_k=know_k)

        # Filter by hard requirements
        candidates = []
        for rule in rules:
            algo = ALGORITHMS.get(rule.recommended_algorithm)
            if not algo:
                continue

            disqualified = False
            reasons = []

            if need_noise_detection and not algo.supports_noise:
                disqualified = True
                reasons.append("does not support noise detection")
            if need_soft_membership and not algo.supports_soft:
                disqualified = True
                reasons.append("does not support soft membership")
            if need_dendrogram and algo.category != AlgorithmCategory.HIERARCHICAL:
                disqualified = True
                reasons.append("does not produce a dendrogram")
            if need_predict and not algo.supports_predict:
                disqualified = True
                reasons.append("does not support predict()")
            if need_snapshot and not algo.supports_snapshot:
                disqualified = True
                reasons.append("does not support snapshots")

            # Memory check for O(n^2) algorithms
            if max_memory_gb and algo.id in ("hdbscan", "agglomerative"):
                mem_gb = (n_samples ** 2 * 8) / (1024 ** 3)  # float64 distance matrix
                if mem_gb > max_memory_gb:
                    disqualified = True
                    reasons.append(f"requires ~{mem_gb:.1f}GB for distance matrix (budget: {max_memory_gb}GB)")

            candidates.append({
                "algorithm": algo.id,
                "name": algo.name,
                "category": algo.category.value,
                "confidence": rule.confidence,
                "rationale": rule.rationale,
                "suggested_params": rule.suggested_params,
                "disqualified": disqualified,
                "disqualification_reasons": reasons,
            })

        # If all rule-based candidates are disqualified, add fallbacks
        viable = [c for c in candidates if not c["disqualified"]]
        if not viable:
            # Build fallback based on hard requirements
            for algo_id, algo in ALGORITHMS.items():
                if algo_id in [c["algorithm"] for c in candidates]:
                    continue
                meets = True
                if need_noise_detection and not algo.supports_noise:
                    meets = False
                if need_soft_membership and not algo.supports_soft:
                    meets = False
                if need_predict and not algo.supports_predict:
                    meets = False
                if need_snapshot and not algo.supports_snapshot:
                    meets = False
                if know_k is False and algo.requires_k:
                    meets = False
                if meets:
                    candidates.append({
                        "algorithm": algo.id,
                        "name": algo.name,
                        "category": algo.category.value,
                        "confidence": 0.5,
                        "rationale": f"Fallback: meets hard requirements ({algo.name})",
                        "suggested_params": {},
                        "disqualified": False,
                        "disqualification_reasons": [],
                    })

        # Sort: viable first (by confidence), then disqualified
        candidates.sort(key=lambda c: (not c["disqualified"], c["confidence"]), reverse=True)

        return success_response(
            recommendations=candidates[:5],
            data_profile={
                "dimensionality": dim.value,
                "scale": scale.value,
                "normalization": norm_state.value,
                "density": density,
            },
        )

    except Exception as exc:
        return error_response(f"Recommendation failed: {exc}", "internal_error")


@mcp.tool()
async def suggest_config(
    algorithm: str,
    n_samples: int,
    n_features: int,
    normalization: str = "unknown",
    source_type: str | None = None,
    target_k: int | None = None,
) -> dict[str, Any]:
    """Get a detailed, justified configuration for a specific algorithm.

    Returns every parameter with its suggested value, the default, valid range,
    and a rationale for why this value was chosen for YOUR data.

    Args:
        algorithm: Algorithm ID (e.g., "kmeans", "embedding_cluster", "hdbscan").
        n_samples: Number of data points.
        n_features: Number of features/dimensions.
        normalization: Data normalization state.
        source_type: Embedding source hint.
        target_k: Desired number of clusters (if known).

    Returns:
        Full parameter configuration with rationale for each choice.
    """
    try:
        algo = get_algorithm(algorithm)
        if not algo:
            valid = ", ".join(ALGORITHMS.keys())
            return error_response(f"Unknown algorithm '{algorithm}'. Valid: {valid}", "validation_error")

        params = get_parameters_for(algorithm)
        config = {}

        for p in params:
            suggestion = {
                "parameter": p.name,
                "default": p.default,
                "valid_range": p.valid_range,
                "sensitivity": p.sensitivity,
                "description": p.description,
            }

            # Generate data-aware suggestion
            suggested_value = p.default
            rationale = "Default value — suitable for most cases."

            if p.name == "n_clusters" and target_k is not None:
                suggested_value = target_k
                rationale = f"User-specified target: {target_k} clusters."
            elif p.name == "n_clusters" and target_k is None:
                heuristic_k = max(2, int(np.sqrt(n_samples / 2)))
                suggested_value = heuristic_k
                rationale = f"Heuristic k=sqrt(n/2) = {heuristic_k}. Refine with optimize_k tool."

            elif p.name == "reduction_dim":
                if n_features <= 256:
                    suggested_value = None
                    rationale = f"Data is only {n_features}d — PCA reduction not needed."
                elif n_features <= 768:
                    suggested_value = 64
                    rationale = f"For {n_features}d embeddings, 64 dims captures most variance."
                elif n_features <= 1536:
                    suggested_value = 128
                    rationale = f"For {n_features}d embeddings (e.g., OpenAI), 128 is the sweet spot."
                else:
                    suggested_value = 256
                    rationale = f"For very high-dim ({n_features}d) embeddings, 256 dims preserves detail."

            elif p.name == "batch_size":
                if target_k:
                    suggested_value = max(1024, target_k * 10)
                    rationale = f"batch_size should be >= 10*k ({target_k * 10}). Using {suggested_value}."
                else:
                    suggested_value = min(1024, n_samples)
                    rationale = f"Default 1024, capped at n_samples ({n_samples})."

            elif p.name == "min_cluster_size":
                if n_samples < 500:
                    suggested_value = 5
                    rationale = "Small dataset — use low min_cluster_size to find structure."
                elif n_samples < 10_000:
                    suggested_value = 15
                    rationale = "Medium dataset — 15 balances noise filtering and cluster discovery."
                else:
                    suggested_value = max(25, n_samples // 500)
                    suggested_value = min(suggested_value, 200)
                    rationale = f"Large dataset — {suggested_value} filters noise while preserving structure."

            elif p.name == "eps":
                rationale = (
                    "eps is data-dependent and cannot be set heuristically. "
                    "Use the optimize_k tool with algorithm='dbscan' to find the right eps "
                    "via k-distance plot analysis."
                )

            elif p.name == "metric":
                if source_type in ("openai", "cohere", "voyage", "sentence_transformers", "embedding"):
                    suggested_value = "cosine"
                    rationale = "Embedding data — cosine similarity is the correct metric."
                elif n_features > 100:
                    suggested_value = "cosine"
                    rationale = "High-dimensional data — cosine avoids distance concentration."
                else:
                    rationale = "Low-to-medium dimensional data — Euclidean is appropriate."

            elif p.name == "n_init":
                if target_k and target_k > 50:
                    suggested_value = 20
                    rationale = f"Large k ({target_k}) makes the landscape complex — more restarts help."
                elif n_samples > 100_000:
                    suggested_value = 3
                    rationale = "Large dataset — fewer restarts to keep runtime manageable."

            elif p.name == "reduction":
                if source_type in ("openai", "cohere"):
                    suggested_value = "matryoshka"
                    rationale = f"{source_type} models support Matryoshka — instant reduction, no PCA cost."
                else:
                    rationale = "PCA is the safe default for any embedding model."

            elif p.name == "linkage":
                rationale = "Ward is best for compact clusters. Use 'complete' for non-Euclidean metrics."

            suggestion["suggested_value"] = suggested_value
            suggestion["rationale"] = rationale

            if p.tips:
                suggestion["tips"] = p.tips
            if p.interactions:
                suggestion["interactions"] = [
                    {"with": i.other_param, "effect": i.description}
                    for i in p.interactions
                ]

            config[p.name] = suggestion

        # Check for anti-patterns in the suggested config
        param_values = {name: s["suggested_value"] for name, s in config.items()}

        profile = DataProfile(
            dimensionality=_classify_dimensionality(n_features),
            scale=DataScale.SMALL,  # not critical for anti-pattern check
            normalization=NormalizationState(normalization) if normalization != "unknown" else NormalizationState.RAW,
            density=DensityProfile.UNIFORM,
            source_type=source_type,
            intrinsic_dim_estimate=None,
        )
        anti = check_anti_patterns(algorithm, param_values, profile)
        warnings = [{"id": a.id, "description": a.description, "fix": a.fix, "severity": a.severity} for a in anti]

        return success_response(
            algorithm=algorithm,
            algorithm_name=algo.name,
            config=config,
            warnings=warnings,
            usage_example=_build_usage_example(algorithm, param_values),
        )

    except Exception as exc:
        return error_response(f"Config suggestion failed: {exc}", "internal_error")


def _build_usage_example(algorithm: str, params: dict[str, Any]) -> str:
    """Build a Python usage example for the suggested config."""
    algo = ALGORITHMS[algorithm]

    if algorithm == "embedding_cluster":
        module = "rustcluster.experimental"
    else:
        module = "rustcluster"

    # Filter to non-default params
    defaults = {p.name: p.default for p in get_parameters_for(algorithm)}
    non_default = {k: v for k, v in params.items() if v != defaults.get(k) and v is not None}

    param_str = ", ".join(f"{k}={v!r}" for k, v in non_default.items())
    class_name = algo.name

    return f"""\
from {module} import {class_name}

model = {class_name}({param_str})
model.fit(X)

labels = model.labels_
print(f"Found {{len(set(labels))}} clusters")
"""


# ===========================================================================
# Phase 3 — Optimize
# ===========================================================================

@mcp.tool()
async def optimize_k(
    data_path: str,
    algorithm: str = "embedding_cluster",
    k_range: list[int] | None = None,
    reduction_dim: int | None = 128,
    metric: str = "euclidean",
    random_state: int = 0,
    max_k_values: int | None = None,
) -> dict[str, Any]:
    """Find the optimal number of clusters by evaluating multiple k values.

    Runs the specified algorithm at each k and computes silhouette, Calinski-Harabasz,
    and Davies-Bouldin scores. Returns scores, the recommended k, and rationale.

    Args:
        data_path: Path to .npy, .npz, or .parquet file.
        algorithm: Algorithm to use. One of: "kmeans", "minibatch_kmeans", "embedding_cluster".
        k_range: List of k values to evaluate (e.g., [5, 10, 15, 20, 30, 50]).
            If None, auto-generates based on data size.
        reduction_dim: PCA reduction dimension (only for embedding_cluster). None to skip.
        metric: Distance metric (for kmeans/minibatch_kmeans).
        random_state: Random seed.
        max_k_values: Maximum number of k values to evaluate. Limits runtime on large datasets.
            If None, evaluates all values in k_range.

    Returns:
        Per-k scores, recommended k with rationale, and score curves for visualization.
    """
    try:
        X = _load_array(data_path)
        if X.ndim != 2:
            return error_response(f"Expected 2D array, got {X.ndim}D", "validation_error")

        n, d = X.shape

        if algorithm not in ("kmeans", "minibatch_kmeans", "embedding_cluster"):
            return error_response(
                f"optimize_k only supports centroid-based algorithms. Got '{algorithm}'. "
                "For density-based (DBSCAN/HDBSCAN), cluster count is discovered automatically.",
                "validation_error",
            )

        # Auto k_range
        if k_range is None:
            max_k = min(n // 2, 200)
            if max_k <= 20:
                k_range = list(range(2, max_k + 1))
            else:
                k_range = sorted(set([2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]) & set(range(2, max_k + 1)))
                if not k_range:
                    k_range = list(range(2, min(max_k + 1, 21)))

        if max_k_values is not None and len(k_range) > max_k_values:
            # Evenly sample from the range to stay within budget
            step = max(1, len(k_range) // max_k_values)
            k_range = k_range[::step][:max_k_values]

        import rustcluster
        from rustcluster.experimental import EmbeddingCluster

        results = []
        for k in k_range:
            if k >= n:
                continue

            try:
                if algorithm == "kmeans":
                    model = rustcluster.KMeans(n_clusters=k, n_init=3, metric=metric, random_state=random_state)
                    model.fit(X)
                    labels = model.labels_
                    eval_data = X
                elif algorithm == "minibatch_kmeans":
                    model = rustcluster.MiniBatchKMeans(n_clusters=k, metric=metric, random_state=random_state)
                    model.fit(X)
                    labels = model.labels_
                    eval_data = X
                elif algorithm == "embedding_cluster":
                    model = EmbeddingCluster(
                        n_clusters=k, reduction_dim=reduction_dim,
                        n_init=3, random_state=random_state,
                    )
                    model.fit(X)
                    labels = model.labels_
                    eval_data = model.reduced_data_ if model.reduced_data_ is not None else X

                n_actual = len(set(labels)) - (1 if -1 in labels else 0)
                if n_actual < 2:
                    continue

                sil = float(rustcluster.silhouette_score(eval_data, labels))
                ch = float(rustcluster.calinski_harabasz_score(eval_data, labels))
                db = float(rustcluster.davies_bouldin_score(eval_data, labels))

                inertia = float(model.inertia_) if hasattr(model, "inertia_") else None
                objective = float(model.objective_) if hasattr(model, "objective_") else None

                results.append({
                    "k": k,
                    "silhouette": round(sil, 4),
                    "calinski_harabasz": round(ch, 2),
                    "davies_bouldin": round(db, 4),
                    "inertia": round(inertia, 2) if inertia is not None else None,
                    "objective": round(objective, 4) if objective is not None else None,
                    "n_actual_clusters": n_actual,
                })
            except Exception as exc:
                results.append({"k": k, "error": str(exc)})

        if not results:
            return error_response("No valid k values produced results.", "computation_error")

        # Find optimal k
        valid = [r for r in results if "error" not in r]
        if valid:
            best_sil = max(valid, key=lambda r: r["silhouette"])
            best_db = min(valid, key=lambda r: r["davies_bouldin"])

            # Weighted recommendation: prefer silhouette, check DB agrees
            recommended_k = best_sil["k"]
            rationale = f"k={recommended_k} has the best silhouette score ({best_sil['silhouette']})."
            if best_db["k"] != recommended_k:
                rationale += f" Note: Davies-Bouldin prefers k={best_db['k']} ({best_db['davies_bouldin']})."
                rationale += " Consider evaluating both."
        else:
            recommended_k = None
            rationale = "All k values produced errors."

        return success_response(
            results=results,
            recommended_k=recommended_k,
            rationale=rationale,
            algorithm=algorithm,
            k_range_evaluated=k_range,
        )

    except FileNotFoundError:
        return error_response(f"File not found: {data_path}", "file_error")
    except Exception as exc:
        return error_response(f"Optimization failed: {exc}", "internal_error")


@mcp.tool()
async def compare_configs(
    data_path: str,
    configs: list[dict[str, Any]],
    random_state: int = 0,
) -> dict[str, Any]:
    """Compare two or more clustering configurations head-to-head on the same data.

    Runs each configuration and computes quality metrics for direct comparison.

    Args:
        data_path: Path to .npy or .npz file.
        configs: List of configuration dicts, each with:
            - algorithm (str): "kmeans", "minibatch_kmeans", "embedding_cluster", etc.
            - params (dict): Algorithm parameters (n_clusters, metric, etc.)
            - label (str, optional): A friendly name for this config.
        random_state: Random seed (overrides per-config seeds for fairness).

    Returns:
        Side-by-side comparison with metrics, cluster size distributions, and a winner.
    """
    try:
        X = _load_array(data_path)
        if X.ndim != 2:
            return error_response(f"Expected 2D array, got {X.ndim}D", "validation_error")

        if len(configs) < 2:
            return error_response("Need at least 2 configs to compare.", "validation_error")

        import rustcluster
        from rustcluster.experimental import EmbeddingCluster

        comparison = []
        for i, cfg in enumerate(configs):
            algo_id = cfg.get("algorithm", "kmeans")
            params = cfg.get("params", {})
            label = cfg.get("label", f"config_{i}")
            params["random_state"] = random_state

            try:
                if algo_id == "kmeans":
                    model = rustcluster.KMeans(**params)
                elif algo_id == "minibatch_kmeans":
                    model = rustcluster.MiniBatchKMeans(**params)
                elif algo_id == "embedding_cluster":
                    model = EmbeddingCluster(**params)
                elif algo_id == "dbscan":
                    model = rustcluster.DBSCAN(**{k: v for k, v in params.items() if k != "random_state"})
                elif algo_id == "hdbscan":
                    model = rustcluster.HDBSCAN(**{k: v for k, v in params.items() if k != "random_state"})
                elif algo_id == "agglomerative":
                    filtered = {k: v for k, v in params.items() if k != "random_state"}
                    model = rustcluster.AgglomerativeClustering(**filtered)
                else:
                    comparison.append({"label": label, "error": f"Unknown algorithm: {algo_id}"})
                    continue

                model.fit(X)
                labels = model.labels_

                # Choose eval data
                eval_data = X
                if hasattr(model, "reduced_data_") and model.reduced_data_ is not None:
                    eval_data = model.reduced_data_

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int(np.sum(labels == -1))

                metrics = {}
                if n_clusters >= 2:
                    sil = float(rustcluster.silhouette_score(eval_data, labels))
                    ch = float(rustcluster.calinski_harabasz_score(eval_data, labels))
                    db = float(rustcluster.davies_bouldin_score(eval_data, labels))
                    metrics["silhouette"] = round(sil, 4)
                    metrics["calinski_harabasz"] = round(ch, 2)
                    metrics["davies_bouldin"] = round(db, 4)

                # Cluster size distribution
                unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                sizes = sorted(counts.tolist(), reverse=True)

                result = {
                    "label": label,
                    "algorithm": algo_id,
                    "params": {k: v for k, v in params.items() if k != "random_state"},
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "metrics": metrics,
                    "cluster_sizes": {
                        "min": int(min(sizes)) if sizes else 0,
                        "max": int(max(sizes)) if sizes else 0,
                        "median": int(np.median(sizes)) if sizes else 0,
                        "std": round(float(np.std(sizes)), 1) if sizes else 0,
                    },
                    "n_iter": int(model.n_iter_) if hasattr(model, "n_iter_") else None,
                }
                if hasattr(model, "inertia_"):
                    result["inertia"] = round(float(model.inertia_), 2)

                comparison.append(result)

            except Exception as exc:
                comparison.append({"label": label, "error": str(exc)})

        # Determine winner (by silhouette if available)
        scored = [c for c in comparison if "error" not in c and "silhouette" in c.get("metrics", {})]
        winner = None
        if scored:
            winner = max(scored, key=lambda c: c["metrics"]["silhouette"])["label"]

        return success_response(
            comparison=comparison,
            winner=winner,
            winner_criterion="highest silhouette score",
        )

    except FileNotFoundError:
        return error_response(f"File not found: {data_path}", "file_error")
    except Exception as exc:
        return error_response(f"Comparison failed: {exc}", "internal_error")


# ===========================================================================
# Phase 4 — Evaluate & diagnose
# ===========================================================================

@mcp.tool()
async def evaluate_clusters(
    data_path: str,
    labels_path: str,
) -> dict[str, Any]:
    """Evaluate clustering quality with all available metrics and diagnostics.

    Computes silhouette, Calinski-Harabasz, and Davies-Bouldin scores,
    plus cluster size distribution analysis and pathology detection.

    Args:
        data_path: Path to .npy, .npz, or .parquet file with the data matrix.
        labels_path: Path to .npy file with cluster labels (int array, -1 for noise).

    Returns:
        Quality metrics with interpretation, cluster size analysis, and any
        detected pathologies with fix suggestions.
    """
    try:
        X = _load_array(data_path)
        labels = np.load(labels_path)

        if X.ndim != 2:
            return error_response(f"Data must be 2D, got {X.ndim}D", "validation_error")
        if labels.ndim != 1:
            return error_response(f"Labels must be 1D, got {labels.ndim}D", "validation_error")
        if len(labels) != len(X):
            return error_response(f"Shape mismatch: {len(X)} samples but {len(labels)} labels", "validation_error")

        import rustcluster

        n_noise = int(np.sum(labels == -1))
        non_noise = labels[labels >= 0]
        unique, counts = np.unique(non_noise, return_counts=True)
        n_clusters = len(unique)
        sizes = sorted(counts.tolist(), reverse=True)

        # Metrics
        metrics = {}
        interpretations = {}
        if n_clusters >= 2:
            sil = float(rustcluster.silhouette_score(X, labels))
            ch = float(rustcluster.calinski_harabasz_score(X, labels))
            db = float(rustcluster.davies_bouldin_score(X, labels))

            metrics = {
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "davies_bouldin": round(db, 4),
            }

            # Interpret each metric
            for mi in METRIC_INTERPRETATIONS:
                val = metrics.get(mi.metric_name)
                if val is None:
                    continue
                if mi.direction == "higher_better":
                    if val >= mi.thresholds.get("excellent", float("inf")):
                        quality = "excellent"
                    elif val >= mi.thresholds.get("good", float("inf")):
                        quality = "good"
                    elif val >= mi.thresholds.get("fair", float("inf")):
                        quality = "fair"
                    else:
                        quality = "poor"
                else:
                    if val <= mi.thresholds.get("good", 0):
                        quality = "good"
                    elif val <= mi.thresholds.get("fair", 0):
                        quality = "fair"
                    else:
                        quality = "poor"

                interpretations[mi.metric_name] = {
                    "value": val,
                    "quality": quality,
                    "direction": mi.direction,
                    "caveats": mi.caveats,
                }
        else:
            interpretations["warning"] = "Need >= 2 clusters to compute metrics."

        # Cluster size analysis
        size_analysis = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(n_noise / len(labels) * 100, 1) if len(labels) > 0 else 0,
            "sizes": sizes,
            "min_size": int(min(sizes)) if sizes else 0,
            "max_size": int(max(sizes)) if sizes else 0,
            "median_size": int(np.median(sizes)) if sizes else 0,
            "size_ratio": round(max(sizes) / max(min(sizes), 1), 1) if sizes else 0,
        }

        # Pathology detection
        pathologies = diagnose_results(
            metric_scores=metrics,
            cluster_sizes=sizes,
            n_noise=n_noise,
            n_total=len(labels),
        )
        detected = []
        for p in pathologies:
            detected.append({
                "id": p.id,
                "name": p.name,
                "symptoms": p.symptoms,
                "likely_causes": p.likely_causes,
                "fixes": [
                    {
                        "description": f.description,
                        "parameter_changes": f.parameter_changes,
                        "expected_effect": f.expected_effect,
                    }
                    for f in p.fixes
                ],
            })

        return success_response(
            metrics=metrics,
            interpretations=interpretations,
            cluster_sizes=size_analysis,
            pathologies=detected,
        )

    except FileNotFoundError as exc:
        return error_response(f"File not found: {exc}", "file_error")
    except Exception as exc:
        return error_response(f"Evaluation failed: {exc}", "internal_error")


@mcp.tool()
async def diagnose(
    description: str,
    algorithm: str | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    cluster_sizes: list[int] | None = None,
    n_noise: int = 0,
    n_total: int = 0,
    n_iter: int | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    """Diagnose a clustering problem from symptoms, metrics, or description.

    Describe what's wrong ("my clusters look uneven", "everything is noise",
    "silhouette is very low") and get specific fix suggestions.

    Args:
        description: Natural language description of the problem.
        algorithm: Algorithm being used (if known).
        params: Current parameter values (if known).
        metrics: Metric scores (e.g., {"silhouette": 0.12, "davies_bouldin": 2.3}).
        cluster_sizes: List of cluster sizes (largest to smallest).
        n_noise: Number of noise points (for DBSCAN/HDBSCAN).
        n_total: Total number of data points.
        n_iter: Number of iterations the algorithm ran.
        max_iter: Maximum iterations configured.

    Returns:
        Detected pathologies with fix recipes and explanations.
    """
    try:
        detected = []

        # Automatic detection from metrics/sizes
        if metrics or cluster_sizes:
            pathologies = diagnose_results(
                metric_scores=metrics or {},
                cluster_sizes=cluster_sizes or [],
                n_iter=n_iter,
                max_iter=max_iter,
                n_noise=n_noise,
                n_total=n_total,
            )
            for p in pathologies:
                detected.append({
                    "id": p.id,
                    "name": p.name,
                    "symptoms": p.symptoms,
                    "likely_causes": p.likely_causes,
                    "fixes": [
                        {
                            "description": f.description,
                            "parameter_changes": f.parameter_changes,
                            "expected_effect": f.expected_effect,
                        }
                        for f in p.fixes
                    ],
                })

        # Keyword-based detection from description
        desc_lower = description.lower()
        keyword_map = {
            "one big cluster": "one_giant_cluster",
            "one giant": "one_giant_cluster",
            "dominant cluster": "one_giant_cluster",
            "singleton": "too_many_singletons",
            "too many small": "too_many_singletons",
            "all noise": "all_noise",
            "everything noise": "all_noise",
            "no clusters": "all_noise",
            "unstable": "unstable_across_runs",
            "different results": "unstable_across_runs",
            "inconsistent": "unstable_across_runs",
            "uneven": "uneven_cluster_sizes",
            "imbalanced": "uneven_cluster_sizes",
            "slow": "slow_convergence",
            "max_iter": "slow_convergence",
            "not converging": "slow_convergence",
        }

        already_ids = {d["id"] for d in detected}
        for keyword, pathology_id in keyword_map.items():
            if keyword in desc_lower and pathology_id not in already_ids:
                p = next((p for p in PATHOLOGY_SIGNATURES if p.id == pathology_id), None)
                if p:
                    detected.append({
                        "id": p.id,
                        "name": p.name,
                        "symptoms": p.symptoms,
                        "likely_causes": p.likely_causes,
                        "fixes": [
                            {
                                "description": f.description,
                                "parameter_changes": f.parameter_changes,
                                "expected_effect": f.expected_effect,
                            }
                            for f in p.fixes
                        ],
                    })
                    already_ids.add(pathology_id)

        # Check anti-patterns on current config
        anti_warnings = []
        if algorithm and params:
            anti = check_anti_patterns(algorithm, params)
            anti_warnings = [
                {"id": a.id, "description": a.description, "fix": a.fix, "severity": a.severity}
                for a in anti
            ]

        return success_response(
            description=description,
            pathologies_detected=detected,
            anti_pattern_warnings=anti_warnings,
            general_advice=(
                "If none of the above matches your issue, try: "
                "(1) run optimize_k to verify your k is reasonable, "
                "(2) check if your data needs PCA reduction (use analyze_data), "
                "(3) try a different algorithm family (use recommend_algorithm)."
            ),
        )

    except Exception as exc:
        return error_response(f"Diagnosis failed: {exc}", "internal_error")


# ===========================================================================
# Knowledge tools (no data needed)
# ===========================================================================

@mcp.tool()
async def explain_algorithm(algorithm: str) -> dict[str, Any]:
    """Get a deep explanation of a clustering algorithm.

    Returns how it works, strengths/weaknesses, all parameters, relationships
    to other algorithms, and when to use (or not use) it.

    Args:
        algorithm: Algorithm ID. One of:
            "kmeans", "minibatch_kmeans", "dbscan", "hdbscan",
            "agglomerative", "embedding_cluster", "spherical_kmeans".

    Returns:
        Comprehensive algorithm explanation with parameters, relationships,
        and usage guidance.
    """
    try:
        algo = get_algorithm(algorithm)
        if not algo:
            valid = ", ".join(ALGORITHMS.keys())
            return error_response(f"Unknown algorithm '{algorithm}'. Valid: {valid}", "validation_error")

        params = get_parameters_for(algorithm)
        rels = get_relationships_for(algorithm)

        return success_response(
            id=algo.id,
            name=algo.name,
            category=algo.category.value,
            description=algo.description,
            how_it_works=algo.how_it_works,
            time_complexity=algo.time_complexity,
            space_complexity=algo.space_complexity,
            convergence=algo.convergence_mechanism,
            capabilities={
                "requires_k": algo.requires_k,
                "supports_predict": algo.supports_predict,
                "supports_soft_membership": algo.supports_soft,
                "supports_snapshot": algo.supports_snapshot,
                "supports_noise_detection": algo.supports_noise,
                "deterministic": algo.deterministic,
                "distance_metrics": algo.metrics,
            },
            strengths=algo.strengths,
            weaknesses=algo.weaknesses,
            internal_constants=algo.internal_constants,
            parameters=[
                {
                    "name": p.name,
                    "type": p.param_type.value,
                    "default": p.default,
                    "valid_range": p.valid_range,
                    "description": p.description,
                    "semantic_meaning": p.semantic_meaning,
                    "sensitivity": p.sensitivity,
                    "tips": p.tips,
                }
                for p in params
            ],
            relationships=[
                {
                    "direction": "outgoing" if r.source == algorithm else "incoming",
                    "other": r.target if r.source == algorithm else r.source,
                    "type": r.relationship.value,
                    "description": r.description,
                }
                for r in rels
            ],
        )

    except Exception as exc:
        return error_response(f"Explanation failed: {exc}", "internal_error")


@mcp.tool()
async def explain_parameter(algorithm: str, parameter: str) -> dict[str, Any]:
    """Get a deep explanation of a specific algorithm parameter.

    Returns what it does conceptually, how to tune it, what it interacts with,
    and practical tips.

    Args:
        algorithm: Algorithm ID (e.g., "kmeans", "embedding_cluster").
        parameter: Parameter name (e.g., "n_clusters", "reduction_dim", "eps").

    Returns:
        Full parameter semantics, interactions, constraints, and tuning advice.
    """
    try:
        algo = get_algorithm(algorithm)
        if not algo:
            valid = ", ".join(ALGORITHMS.keys())
            return error_response(f"Unknown algorithm '{algorithm}'. Valid: {valid}", "validation_error")

        params = get_parameters_for(algorithm)
        param = next((p for p in params if p.name == parameter), None)
        if not param:
            valid = ", ".join(p.name for p in params)
            msg = f"Unknown parameter '{parameter}' for {algorithm}. Valid: {valid}"
            return error_response(msg, "validation_error")

        return success_response(
            algorithm=algorithm,
            parameter=parameter,
            type=param.param_type.value,
            default=param.default,
            valid_range=param.valid_range,
            description=param.description,
            semantic_meaning=param.semantic_meaning,
            sensitivity=param.sensitivity,
            constraints=param.constraints,
            tips=param.tips,
            interactions=[
                {
                    "with": i.other_param,
                    "type": i.interaction_type,
                    "description": i.description,
                }
                for i in param.interactions
            ],
        )

    except Exception as exc:
        return error_response(f"Explanation failed: {exc}", "internal_error")


@mcp.tool()
async def list_algorithms(
    category: str | None = None,
    supports_noise: bool | None = None,
    supports_predict: bool | None = None,
    supports_snapshot: bool | None = None,
    requires_k: bool | None = None,
) -> dict[str, Any]:
    """List all available clustering algorithms with optional filtering.

    Args:
        category: Filter by category. One of:
            "centroid-based", "density-based", "hierarchical", "pipeline", or None for all.
        supports_noise: Filter to algorithms that detect noise/outliers.
        supports_predict: Filter to algorithms with predict() for new data.
        supports_snapshot: Filter to algorithms with snapshot support.
        requires_k: Filter by whether k must be specified (True) or is auto-discovered (False).

    Returns:
        List of algorithms with key capabilities.
    """
    try:
        result = []
        for algo in ALGORITHMS.values():
            if category and algo.category.value != category:
                continue
            if supports_noise is not None and algo.supports_noise != supports_noise:
                continue
            if supports_predict is not None and algo.supports_predict != supports_predict:
                continue
            if supports_snapshot is not None and algo.supports_snapshot != supports_snapshot:
                continue
            if requires_k is not None and algo.requires_k != requires_k:
                continue

            result.append({
                "id": algo.id,
                "name": algo.name,
                "category": algo.category.value,
                "description": algo.description,
                "requires_k": algo.requires_k,
                "supports_predict": algo.supports_predict,
                "supports_noise": algo.supports_noise,
                "supports_soft_membership": algo.supports_soft,
                "supports_snapshot": algo.supports_snapshot,
                "metrics": algo.metrics,
            })

        return success_response(algorithms=result, count=len(result))

    except Exception as exc:
        return error_response(f"Listing failed: {exc}", "internal_error")


@mcp.tool()
async def check_config(
    algorithm: str,
    params: dict[str, Any],
    n_features: int | None = None,
    normalization: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    """Validate a clustering configuration for anti-patterns and issues.

    Checks for common mistakes like using Euclidean on high-dim embeddings,
    Ward linkage with non-Euclidean metric, DBSCAN on high-d data, etc.

    Args:
        algorithm: Algorithm ID.
        params: Parameter dict to validate.
        n_features: Number of features (helps detect dimensionality issues).
        normalization: Data normalization state.
        source_type: Embedding source hint.

    Returns:
        List of warnings/errors with fix suggestions, or confirmation that
        the config looks good.
    """
    try:
        algo = get_algorithm(algorithm)
        if not algo:
            valid = ", ".join(ALGORITHMS.keys())
            return error_response(f"Unknown algorithm '{algorithm}'. Valid: {valid}", "validation_error")

        profile = None
        if n_features is not None:
            norm_state = NormalizationState(normalization) if normalization else NormalizationState.UNKNOWN
            profile = DataProfile(
                dimensionality=_classify_dimensionality(n_features),
                scale=DataScale.MEDIUM,
                normalization=norm_state,
                density=DensityProfile.UNIFORM,
                source_type=source_type,
                intrinsic_dim_estimate=None,
            )

        # Check anti-patterns
        anti = check_anti_patterns(algorithm, params, profile)
        warnings = [
            {
                "id": a.id,
                "severity": a.severity,
                "description": a.description,
                "why_bad": a.why_bad,
                "fix": a.fix,
            }
            for a in anti
        ]

        # Check parameter validity
        algo_params = get_parameters_for(algorithm)
        param_issues = []
        for name, value in params.items():
            p = next((p for p in algo_params if p.name == name), None)
            if not p:
                param_issues.append(f"Unknown parameter '{name}' for {algorithm}")
                continue
            if p.valid_range:
                if "min" in p.valid_range and value is not None:
                    if p.valid_range.get("min_exclusive"):
                        if value <= p.valid_range["min"]:
                            param_issues.append(f"{name}={value} must be > {p.valid_range['min']}")
                    elif value < p.valid_range["min"]:
                        param_issues.append(f"{name}={value} must be >= {p.valid_range['min']}")
                if "values" in p.valid_range and value not in p.valid_range["values"]:
                    param_issues.append(f"{name}='{value}' must be one of {p.valid_range['values']}")

        if not warnings and not param_issues:
            return success_response(
                valid=True,
                message=f"Configuration looks good for {algo.name}.",
            )

        return success_response(
            valid=len([w for w in warnings if w["severity"] == "error"]) == 0 and not param_issues,
            warnings=warnings,
            parameter_issues=param_issues,
        )

    except Exception as exc:
        return error_response(f"Validation failed: {exc}", "internal_error")


# ===========================================================================
# Sandbox tools — code execution via marimo-sandbox
# ===========================================================================

# Import marimo-sandbox internals (same pattern as snowbox)
try:
    from marimo_sandbox.server import (  # noqa: F401
        _impl_approve_run,
        _impl_cancel_run,
        _impl_check_setup,
        _impl_clean_environments,
        _impl_delete_run,
        _impl_diff_runs,
        _impl_get_run,
        _impl_get_run_outputs,
        _impl_list_artifacts,
        _impl_list_environments,
        _impl_list_pending_approvals,
        _impl_list_runs,
        _impl_open_notebook,
        _impl_purge_runs,
        _impl_read_artifact,
        _impl_rerun,
        _impl_run_python,
        _inject_pep723_header,
    )

    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False


def _impl_cluster_run_python(
    code: str,
    description: str | None = None,
    packages: list[str] | None = None,
    dry_run: bool = False,
    require_approval: bool = False,
    async_mode: bool = False,
    sandbox: bool = False,
) -> dict[str, Any]:
    """Run Python code with rustcluster context injected."""
    if not _SANDBOX_AVAILABLE:
        return error_response(
            "marimo-sandbox is not installed. Install it to enable code execution: "
            "pip install marimo-sandbox",
            "dependency_error",
        )

    # Merge packages
    all_packages = list(packages or [])
    for pkg in CLUSTER_REQUIRED_PACKAGES:
        if pkg not in all_packages:
            all_packages.append(pkg)

    # Inject clustering context
    injected_code = CLUSTER_CONTEXT_CODE + code

    result = _impl_run_python(
        code=injected_code,
        description=description or "rustcluster analysis",
        timeout_seconds=120,
        packages=all_packages,
        dry_run=dry_run,
        require_approval=require_approval,
        async_mode=async_mode,
        sandbox=sandbox,
    )

    # Inject PEP 723 header for notebook compatibility
    if notebook_path := result.get("notebook_path"):
        _inject_pep723_header(notebook_path, all_packages)

    return result


@mcp.tool()
async def run_python(
    code: str,
    description: str | None = None,
    packages: list[str] | None = None,
    dry_run: bool = False,
    require_approval: bool = False,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed Marimo notebook with rustcluster pre-loaded.

    The `__cluster__` helper is automatically available — no imports needed.
    Every run is saved as a reproducible notebook file.

    Common patterns:
        # Load and cluster
        data = __cluster__.load("embeddings.npy")
        result = __cluster__.fit("embedding_cluster", data, n_clusters=20)
        print(result["metrics"])

        # Parameter sweep with plot
        sweep = __cluster__.sweep_k(data, "embedding_cluster", reduction_dim=128)
        __cluster__.plot_sweep(sweep)

        # Direct rustcluster usage (also available)
        from rustcluster import KMeans
        model = KMeans(n_clusters=10)
        model.fit(data)

    Args:
        code: Python code to execute. `__cluster__` is pre-injected.
        description: Optional description for the run log.
        packages: Additional PyPI packages to install (rustcluster, numpy,
            matplotlib are always included).
        dry_run: If True, analyze code for risks without executing.
        require_approval: If True, block on critical risk findings until approved.
        async_mode: If True, launch in background. Poll with get_run().

    Returns:
        Run result with stdout, stderr, status, artifacts, and notebook path.
    """
    return _impl_cluster_run_python(
        code=code,
        description=description,
        packages=packages,
        dry_run=dry_run,
        require_approval=require_approval,
        async_mode=async_mode,
    )


@mcp.tool()
async def get_run(run_id: str) -> dict[str, Any]:
    """Get details of a previous clustering run.

    Args:
        run_id: The run ID returned by run_python.

    Returns:
        Run record with status, code, stdout/stderr, artifacts, and notebook path.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_get_run(run_id)


@mcp.tool()
async def get_run_outputs(run_id: str) -> dict[str, Any]:
    """Get the structured __outputs__ dict from a run.

    In your code, populate `__outputs__` to return structured data:
        __outputs__ = {"best_k": 15, "silhouette": 0.42}

    Args:
        run_id: The run ID.

    Returns:
        The __outputs__ dict as JSON.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_get_run_outputs(run_id)


@mcp.tool()
async def list_runs(
    limit: int = 20,
    status: str | None = None,
) -> dict[str, Any]:
    """List recent clustering runs.

    Args:
        limit: Max runs to return.
        status: Filter by status ("completed", "failed", "running").

    Returns:
        List of run summaries with IDs, descriptions, and statuses.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_list_runs(limit=limit, status=status)


@mcp.tool()
async def list_artifacts(run_id: str) -> dict[str, Any]:
    """List files created by a clustering run (plots, labels, CSVs, etc.).

    Args:
        run_id: The run ID.

    Returns:
        List of artifact paths and metadata.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_list_artifacts(run_id)


@mcp.tool()
async def read_artifact(run_id: str, artifact_name: str) -> dict[str, Any]:
    """Read an artifact file from a clustering run.

    Args:
        run_id: The run ID.
        artifact_name: Filename of the artifact (e.g., "sweep.png", "labels.npy").

    Returns:
        Artifact content (text) or base64-encoded binary.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_read_artifact(run_id, artifact_name)


@mcp.tool()
async def open_notebook(run_id: str) -> dict[str, Any]:
    """Open a clustering run in the Marimo notebook UI for interactive editing.

    Args:
        run_id: The run ID.

    Returns:
        URL to the Marimo notebook interface.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_open_notebook(run_id)


@mcp.tool()
async def rerun(run_id: str) -> dict[str, Any]:
    """Re-execute a previous clustering run's code.

    Args:
        run_id: The run ID to re-execute.

    Returns:
        New run result.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_rerun(run_id)


@mcp.tool()
async def diff_runs(
    run_id: str,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """Compare two clustering runs: code changes, output changes, metric deltas.

    Args:
        run_id: The run ID.
        compare_to: Run ID to compare against. If None, compares to previous run.

    Returns:
        Diff of code, environment, status, artifacts, and outputs.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_diff_runs(run_id, compare_to=compare_to)


@mcp.tool()
async def approve_run(approval_token: str) -> dict[str, Any]:
    """Approve a run that was blocked by risk analysis.

    Args:
        approval_token: Token from the blocked run's response.

    Returns:
        Approved run result.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_approve_run(approval_token)


@mcp.tool()
async def delete_run(run_id: str) -> dict[str, Any]:
    """Delete a clustering run's record and files.

    Args:
        run_id: The run ID to delete.

    Returns:
        Confirmation.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    return _impl_delete_run(run_id)


@mcp.tool()
async def check_setup() -> dict[str, Any]:
    """Verify that the sandbox runtime is properly configured.

    Returns:
        Setup status for marimo-sandbox, rustcluster, and dependencies.
    """
    if not _SANDBOX_AVAILABLE:
        return error_response("marimo-sandbox not installed", "dependency_error")
    result = _impl_check_setup()
    # Add rustcluster-specific checks
    try:
        import rustcluster as _rc_check  # noqa: F401
        result["rustcluster"] = {
            "installed": True,
            "algorithms": list(ALGORITHMS.keys()),
        }
    except ImportError:
        result["rustcluster"] = {"installed": False}
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
