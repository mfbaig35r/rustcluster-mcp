"""
Clustering context injected into every sandbox run.

Provides `__cluster__` — a pre-configured helper for clustering operations
with rustcluster. Users write code against this context and get back results,
metrics, and artifacts automatically.
"""

CLUSTER_CONTEXT_CODE: str = """\
# ── __cluster__ context (injected by rustcluster-mcp) ──────────────────────
import numpy as _rc_np
import rustcluster as _rc
from rustcluster.experimental import EmbeddingCluster as _rc_EmbeddingCluster
from rustcluster.experimental import EmbeddingReducer as _rc_EmbeddingReducer
from rustcluster.snapshot import ClusterSnapshot as _rc_ClusterSnapshot


class _ClusterContext:
    \"\"\"Pre-configured clustering helper available as `__cluster__`.

    Usage:
        data = __cluster__.load("embeddings.npy")
        result = __cluster__.fit("embedding_cluster", data, n_clusters=20, reduction_dim=128)
        print(result["metrics"])
        __cluster__.save_labels(result["labels"], "labels.npy")
    \"\"\"

    _ALGO_MAP = {
        "kmeans": _rc.KMeans,
        "minibatch_kmeans": _rc.MiniBatchKMeans,
        "dbscan": _rc.DBSCAN,
        "hdbscan": _rc.HDBSCAN,
        "agglomerative": _rc.AgglomerativeClustering,
        "embedding_cluster": _rc_EmbeddingCluster,
    }

    def __init__(self) -> None:
        self._last_model = None
        self._last_data = None
        self._last_labels = None

    # ── Data I/O ──────────────────────────────────────────────────────────

    def load(self, path: str) -> "_rc_np.ndarray":
        \"\"\"Load data from .npy or .npz file.\"\"\"
        if path.endswith(".npz"):
            f = _rc_np.load(path)
            key = list(f.keys())[0]
            return f[key]
        return _rc_np.load(path)

    def save_labels(self, labels: "_rc_np.ndarray", path: str = "labels.npy") -> str:
        \"\"\"Save cluster labels to .npy file (captured as artifact).\"\"\"
        _rc_np.save(path, labels)
        return path

    def save_centers(self, centers: "_rc_np.ndarray", path: str = "centers.npy") -> str:
        \"\"\"Save cluster centers to .npy file (captured as artifact).\"\"\"
        _rc_np.save(path, centers)
        return path

    # ── Profiling ─────────────────────────────────────────────────────────

    def profile(self, X: "_rc_np.ndarray") -> dict:
        \"\"\"Quick data profile: shape, norms, normalization state.\"\"\"
        n, d = X.shape
        norms = _rc_np.linalg.norm(X, axis=1)
        is_l2 = bool(_rc_np.allclose(norms, 1.0, atol=1e-3))
        return {
            "n_samples": n,
            "n_features": d,
            "dtype": str(X.dtype),
            "is_l2_normalized": is_l2,
            "norm_mean": round(float(_rc_np.mean(norms)), 4),
            "norm_std": round(float(_rc_np.std(norms)), 4),
            "has_nan": bool(_rc_np.isnan(X).any()),
            "has_inf": bool(_rc_np.isinf(X).any()),
        }

    # ── Clustering ────────────────────────────────────────────────────────

    def fit(self, algorithm: str, X: "_rc_np.ndarray", **params) -> dict:
        \"\"\"Fit a clustering model and return results with metrics.

        Args:
            algorithm: One of 'kmeans', 'minibatch_kmeans', 'dbscan',
                'hdbscan', 'agglomerative', 'embedding_cluster'.
            X: Data matrix (n_samples, n_features).
            **params: Algorithm parameters (n_clusters, metric, etc.).

        Returns:
            Dict with 'labels', 'metrics', 'cluster_sizes', 'model', etc.
        \"\"\"
        cls = self._ALGO_MAP.get(algorithm)
        if cls is None:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Valid: {list(self._ALGO_MAP)}")

        model = cls(**params)
        model.fit(X)
        labels = model.labels_
        self._last_model = model
        self._last_data = X
        self._last_labels = labels

        result = {
            "labels": labels,
            "model": model,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        }

        # Eval data (use reduced for EmbeddingCluster)
        eval_data = X
        if hasattr(model, "reduced_data_") and model.reduced_data_ is not None:
            eval_data = model.reduced_data_
            result["reduced_data"] = model.reduced_data_

        # Metrics
        n_clusters = result["n_clusters"]
        if n_clusters >= 2:
            result["metrics"] = {
                "silhouette": round(float(_rc.silhouette_score(eval_data, labels)), 4),
                "calinski_harabasz": round(float(_rc.calinski_harabasz_score(eval_data, labels)), 2),
                "davies_bouldin": round(float(_rc.davies_bouldin_score(eval_data, labels)), 4),
            }

        # Cluster sizes
        non_noise = labels[labels >= 0]
        _, counts = _rc_np.unique(non_noise, return_counts=True)
        sizes = sorted(counts.tolist(), reverse=True)
        result["cluster_sizes"] = sizes
        result["n_noise"] = int(_rc_np.sum(labels == -1))

        # Model-specific attributes
        if hasattr(model, "cluster_centers_"):
            result["centers"] = model.cluster_centers_
        if hasattr(model, "inertia_"):
            result["inertia"] = float(model.inertia_)
        if hasattr(model, "n_iter_"):
            result["n_iter"] = int(model.n_iter_)
        if hasattr(model, "objective_"):
            result["objective"] = float(model.objective_)

        return result

    # ── Parameter sweeps ──────────────────────────────────────────────────

    def sweep_k(
        self,
        X: "_rc_np.ndarray",
        algorithm: str = "embedding_cluster",
        k_values: "list[int] | None" = None,
        **params,
    ) -> "list[dict]":
        \"\"\"Run a parameter sweep over k values, returning metrics for each.

        Args:
            X: Data matrix.
            algorithm: Algorithm to use.
            k_values: List of k to try. Auto-generated if None.
            **params: Other algorithm parameters (reduction_dim, metric, etc.).

        Returns:
            List of dicts with k, metrics, and cluster sizes.
        \"\"\"
        n = len(X)
        if k_values is None:
            max_k = min(n // 2, 200)
            if max_k <= 20:
                k_values = list(range(2, max_k + 1))
            else:
                candidates = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
                k_values = [k for k in candidates if k < max_k]

        results = []
        for k in k_values:
            if k >= n:
                continue
            try:
                r = self.fit(algorithm, X, n_clusters=k, **params)
                entry = {"k": k}
                if "metrics" in r:
                    entry.update(r["metrics"])
                entry["n_clusters"] = r["n_clusters"]
                entry["cluster_sizes"] = r["cluster_sizes"]
                if "inertia" in r:
                    entry["inertia"] = r["inertia"]
                if "objective" in r:
                    entry["objective"] = r["objective"]
                results.append(entry)
            except Exception as e:
                results.append({"k": k, "error": str(e)})

        return results

    # ── Visualization helpers ─────────────────────────────────────────────

    def plot_sweep(self, sweep_results: "list[dict]", save_path: str = "sweep.png") -> str:
        \"\"\"Plot k-sweep results (silhouette, CH, DB curves). Saved as artifact.\"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in sweep_results if "error" not in r and "silhouette" in r]
        if not valid:
            raise ValueError("No valid sweep results to plot.")

        ks = [r["k"] for r in valid]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(ks, [r["silhouette"] for r in valid], "o-")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Silhouette")
        axes[0].set_title("Silhouette Score")

        axes[1].plot(ks, [r["calinski_harabasz"] for r in valid], "o-")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("CH Index")
        axes[1].set_title("Calinski-Harabasz")

        axes[2].plot(ks, [r["davies_bouldin"] for r in valid], "o-")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("DB Index")
        axes[2].set_title("Davies-Bouldin")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_sizes(self, cluster_sizes: "list[int]", save_path: str = "sizes.png") -> str:
        \"\"\"Plot cluster size distribution as a bar chart. Saved as artifact.\"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(cluster_sizes)), cluster_sizes)
        ax.set_xlabel("Cluster (sorted by size)")
        ax.set_ylabel("Size")
        ax.set_title(f"Cluster Size Distribution (n={len(cluster_sizes)})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self, model=None):
        \"\"\"Create a ClusterSnapshot from the last fitted model (or a given model).\"\"\"
        model = model or self._last_model
        if model is None:
            raise RuntimeError("No model fitted yet. Call fit() first.")
        if not hasattr(model, "snapshot"):
            raise RuntimeError(f"{type(model).__name__} does not support snapshots.")
        return model.snapshot()

    # ── Snapshot Visualization ────────────────────────────────────────────

    def _reduce_2d(self, X, method="umap", random_state=42):
        \"\"\"Reduce data to 2D for visualization.\"\"\"
        if method == "umap":
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(X)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(X)
        elif method == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=random_state).fit_transform(X)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'umap', 'tsne', or 'pca'.")

    def visualize_snapshot(
        self,
        X,
        snapshot,
        method="umap",
        color_by="cluster",
        show_rejected=True,
        show_centroids=True,
        confidence_threshold=None,
        adaptive_threshold=False,
        adaptive_percentile="p10",
        title=None,
        save_path="snapshot_map.png",
    ) -> str:
        \"\"\"Visualize a ClusterSnapshot as a 2D scatter plot.

        Projects high-dimensional data to 2D via UMAP/t-SNE/PCA, then overlays
        cluster assignments, confidence (opacity), and rejected points.

        Args:
            X: Data matrix (n_samples, n_features) — the original vectors.
            snapshot: ClusterSnapshot (from model.snapshot() or __cluster__.snapshot()).
            method: Projection method: 'umap' (best), 'tsne', or 'pca' (fastest).
            color_by: What determines point color:
                'cluster' — assigned cluster label (default)
                'confidence' — confidence score (blue=high, red=low)
            show_rejected: If True, show rejected points as gray X markers.
            show_centroids: If True, project centroids and show as bold markers.
            confidence_threshold: Optional rejection threshold for assign_with_scores.
            adaptive_threshold: Use per-cluster adaptive thresholds (requires calibration).
            adaptive_percentile: Percentile for adaptive threshold: 'p5','p10','p25','p50'.
            title: Plot title. Auto-generated if None.
            save_path: Output path (captured as artifact).

        Returns:
            Path to saved PNG.

        Requires:
            packages=['umap-learn'] for UMAP, or ['scikit-learn'] for t-SNE/PCA.
        \"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import colormaps

        # Get assignment with scores
        assign_kwargs = {}
        if confidence_threshold is not None:
            assign_kwargs["confidence_threshold"] = confidence_threshold
        if adaptive_threshold:
            assign_kwargs["adaptive_threshold"] = True
            assign_kwargs["adaptive_percentile"] = adaptive_percentile
        result = snapshot.assign_with_scores(X, **assign_kwargs)

        labels = result.labels_
        confidences = result.confidences_
        rejected = result.rejected_

        # 2D projection
        X_arr = _rc_np.asarray(X, dtype=_rc_np.float64)
        coords = self._reduce_2d(X_arr, method=method)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Separate accepted and rejected
        accepted_mask = ~rejected
        rejected_mask = rejected

        k = snapshot.k
        cmap = colormaps.get_cmap("tab20" if k <= 20 else "hsv")

        if color_by == "cluster":
            # Vectorized: build RGBA array with per-point alpha from confidence
            colors_rgba = _rc_np.zeros((len(coords), 4))
            for cluster_id in range(k):
                mask = accepted_mask & (labels == cluster_id)
                if not mask.any():
                    continue
                color = cmap(cluster_id / max(k - 1, 1))
                colors_rgba[mask, :3] = color[:3]
                colors_rgba[mask, 3] = _rc_np.clip(confidences[mask], 0.15, 1.0)
                # Legend entry (empty scatter for label only)
                ax.scatter([], [], c=[color], s=20,
                           label=f"Cluster {cluster_id} (n={int(mask.sum())})")
            # Single draw call for all accepted points
            if accepted_mask.any():
                ax.scatter(
                    coords[accepted_mask, 0], coords[accepted_mask, 1],
                    c=colors_rgba[accepted_mask], s=15, edgecolors="none",
                )

        elif color_by == "confidence":
            sc = ax.scatter(
                coords[accepted_mask, 0], coords[accepted_mask, 1],
                c=confidences[accepted_mask], cmap="RdYlGn", vmin=0, vmax=1,
                s=20, edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, label="Confidence", shrink=0.8)

        # Rejected points
        if show_rejected and rejected_mask.any():
            ax.scatter(
                coords[rejected_mask, 0], coords[rejected_mask, 1],
                c="gray", marker="x", s=25, alpha=0.4,
                label=f"Rejected (n={int(rejected_mask.sum())})",
                linewidths=0.8,
            )

        # Centroids
        if show_centroids:
            try:
                # Project centroids through the same 2D reduction
                # For EmbeddingCluster snapshots, centroids are in reduced space
                centroid_data = _rc_np.array(snapshot._snap.centroids()).reshape(k, snapshot.d)
                # Combine centroids with data, project together, take centroid positions
                combined = _rc_np.vstack([X_arr, centroid_data]) if snapshot.d == X_arr.shape[1] else None
                if combined is not None:
                    combined_2d = self._reduce_2d(combined, method=method)
                    centroid_coords = combined_2d[len(X_arr):]
                    for i in range(k):
                        color = cmap(i / max(k - 1, 1))
                        ax.scatter(
                            centroid_coords[i, 0], centroid_coords[i, 1],
                            c=[color], marker="*", s=200, edgecolors="black",
                            linewidths=1.0, zorder=10,
                        )
            except Exception:
                pass  # Centroids not available or dim mismatch

        n_accepted = int(accepted_mask.sum())
        n_rejected = int(rejected_mask.sum())
        auto_title = (
            f"Cluster Snapshot ({method.upper()}) — "
            f"{k} clusters, {n_accepted} accepted, {n_rejected} rejected"
        )
        ax.set_title(title or auto_title, fontsize=13)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

        # Legend (limit to avoid clutter)
        handles, legend_labels = ax.get_legend_handles_labels()
        if len(handles) <= 25:
            ax.legend(fontsize=7, loc="best", ncol=2 if len(handles) > 12 else 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def visualize_confidence(
        self,
        X,
        snapshot,
        save_path="confidence_heatmap.png",
    ) -> str:
        \"\"\"Visualize per-cluster confidence distributions as a heatmap.

        Shows the spread of confidence scores within each cluster — tight clusters
        have narrow, high-confidence distributions; diffuse clusters spread wider.
        No 2D projection needed.

        Args:
            X: Data matrix.
            snapshot: ClusterSnapshot.
            save_path: Output path.

        Returns:
            Path to saved PNG.
        \"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = snapshot.assign_with_scores(X)
        labels = result.labels_
        confidences = result.confidences_
        k = snapshot.k

        # Build per-cluster confidence distributions
        bins = _rc_np.linspace(0, 1, 51)
        heatmap_data = []
        cluster_labels = []

        for cid in range(k):
            mask = labels == cid
            if not mask.any():
                heatmap_data.append(_rc_np.zeros(50))
                cluster_labels.append(f"C{cid} (n=0)")
                continue
            hist, _ = _rc_np.histogram(confidences[mask], bins=bins, density=True)
            heatmap_data.append(hist)
            mean_conf = float(_rc_np.mean(confidences[mask]))
            cluster_labels.append(f"C{cid} (n={int(mask.sum())}, \\u03bc={mean_conf:.2f})")

        heatmap = _rc_np.array(heatmap_data)

        fig, ax = plt.subplots(figsize=(14, max(4, k * 0.35)))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd",
                       extent=[0, 1, k - 0.5, -0.5])
        ax.set_yticks(range(k))
        ax.set_yticklabels(cluster_labels, fontsize=7)
        ax.set_xlabel("Confidence Score")
        ax.set_title("Per-Cluster Confidence Distribution")
        plt.colorbar(im, ax=ax, label="Density", shrink=0.8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def visualize_drift(
        self,
        snapshot,
        drift_report,
        save_path="drift_radar.png",
    ) -> str:
        \"\"\"Visualize cluster drift as a radar chart.

        One axis per cluster showing relative drift magnitude. Quick visual for
        'which clusters are degrading?' after deploying a snapshot to production.

        Args:
            snapshot: ClusterSnapshot.
            drift_report: DriftReport from snapshot.drift_report(X_new).
            save_path: Output path.

        Returns:
            Path to saved PNG.
        \"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = snapshot.k
        rel_drift = _rc_np.array(drift_report.relative_drift_)

        # Cap extreme values for visualization
        rel_drift = _rc_np.clip(_rc_np.nan_to_num(rel_drift, nan=0.0), -5.0, 5.0)
        abs_drift = _rc_np.abs(rel_drift)

        # Radar chart
        angles = _rc_np.linspace(0, 2 * _rc_np.pi, k, endpoint=False).tolist()
        angles += angles[:1]
        values = abs_drift.tolist() + [abs_drift[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        ax.plot(angles, values, "o-", linewidth=2, color="#e74c3c")
        ax.fill(angles, values, alpha=0.2, color="#e74c3c")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=8)
        ax.set_title(
            f"Cluster Drift Radar — global mean dist: "
            f"{drift_report.global_mean_distance_:.3f}",
            pad=20, fontsize=12,
        )

        # Add direction drift if available (spherical snapshots)
        dir_drift = getattr(drift_report, "direction_drift_", None)
        if dir_drift is not None:
            dir_vals = _rc_np.clip(_rc_np.nan_to_num(_rc_np.array(dir_drift), nan=0.0), 0, 1)
            dir_values = dir_vals.tolist() + [dir_vals[0]]
            ax.plot(angles, dir_values, "s--", linewidth=1.5, color="#3498db", alpha=0.7,
                    label="Direction drift")
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    def visualize_hierarchical(
        self,
        X,
        hierarchical_snapshot,
        method="umap",
        save_path="hierarchical_sunburst.png",
    ) -> str:
        \"\"\"Visualize a HierarchicalSnapshot as a sunburst chart.

        Inner ring = root clusters, outer ring = child clusters, sized by count.

        Args:
            X: Data matrix.
            hierarchical_snapshot: HierarchicalSnapshot.
            method: Projection method (only used for fallback scatter).
            save_path: Output path.

        Returns:
            Path to saved PNG.
        \"\"\"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        from matplotlib import colormaps

        root_labels, child_labels = hierarchical_snapshot.assign(X)
        k_root = hierarchical_snapshot.k_root

        # Count per root cluster
        root_sizes = []
        for rid in range(k_root):
            root_sizes.append(int((root_labels == rid).sum()))
        total = sum(root_sizes)

        cmap = colormaps.get_cmap("tab20" if k_root <= 20 else "hsv")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Inner ring: root clusters
        inner_r = 0.5
        inner_width = 0.35
        angle = 0
        root_angles = []

        for rid in range(k_root):
            if total == 0:
                continue
            sweep = 360 * root_sizes[rid] / total
            color = cmap(rid / max(k_root - 1, 1))
            wedge = Wedge((0, 0), inner_r + inner_width, angle, angle + sweep,
                          width=inner_width, facecolor=color, edgecolor="white", linewidth=1.5)
            ax.add_patch(wedge)
            # Label
            mid_angle = _rc_np.radians(angle + sweep / 2)
            lx = (inner_r + inner_width / 2) * _rc_np.cos(mid_angle)
            ly = (inner_r + inner_width / 2) * _rc_np.sin(mid_angle)
            if sweep > 10:
                ax.text(lx, ly, f"R{rid}\\n{root_sizes[rid]}", ha="center", va="center",
                        fontsize=7, fontweight="bold")
            root_angles.append((angle, sweep, color))
            angle += sweep

        # Outer ring: child clusters
        outer_r = inner_r + inner_width
        outer_width = 0.25
        angle = 0

        for rid in range(k_root):
            if total == 0:
                continue
            sweep = 360 * root_sizes[rid] / total
            root_mask = root_labels == rid
            children = child_labels[root_mask]
            unique_children = _rc_np.unique(children[children >= 0])

            if len(unique_children) == 0:
                # No children — single outer wedge
                _, _, base_color = root_angles[rid]
                wedge = Wedge((0, 0), outer_r + outer_width, angle, angle + sweep,
                              width=outer_width, facecolor=base_color, edgecolor="white",
                              linewidth=0.5, alpha=0.5)
                ax.add_patch(wedge)
            else:
                child_counts = []
                for cid in unique_children:
                    child_counts.append((cid, int((children == cid).sum())))
                child_total = sum(c for _, c in child_counts)

                child_angle = angle
                _, _, base_color = root_angles[rid]
                base_rgb = matplotlib.colors.to_rgb(base_color)

                for i, (cid, count) in enumerate(child_counts):
                    child_sweep = sweep * count / max(child_total, 1)
                    # Vary lightness
                    factor = 0.6 + 0.4 * (i / max(len(child_counts) - 1, 1))
                    child_color = tuple(min(1, c * factor) for c in base_rgb)
                    wedge = Wedge((0, 0), outer_r + outer_width, child_angle,
                                  child_angle + child_sweep, width=outer_width,
                                  facecolor=child_color, edgecolor="white", linewidth=0.5)
                    ax.add_patch(wedge)
                    child_angle += child_sweep

            angle += sweep

        ax.set_title(f"Hierarchical Clustering — {k_root} root clusters", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path


__cluster__ = _ClusterContext()
"""

CLUSTER_REQUIRED_PACKAGES: list[str] = [
    "rustcluster",
    "numpy",
    "matplotlib",
    "marimo",
]
