"""
Clustering Knowledge Graph for rustcluster MCP server.

This module encodes the complete domain knowledge about clustering algorithms,
their parameters, data characteristics, decision logic, and diagnostics.
The MCP server queries this graph to provide intelligent clustering advice.

Architecture:
    Layer 1 — Algorithms: identity, category, relationships
    Layer 2 — Parameters: types, ranges, defaults, semantics, interactions
    Layer 3 — Data Characteristics: dimensions, density, normalization, scale
    Layer 4 — Decision Edges: algorithm selection rules, anti-patterns, tradeoffs
    Layer 5 — Diagnostics: metric interpretation, pathology signatures, fix recipes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Layer 0 — Enums & primitives
# ---------------------------------------------------------------------------

class AlgorithmCategory(str, Enum):
    CENTROID = "centroid-based"
    DENSITY = "density-based"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"


class ParameterType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    OPTIONAL_INT = "optional_int"


class Relationship(str, Enum):
    COMPOSES = "composes"          # A uses B internally
    APPROXIMATES = "approximates"  # A is a faster version of B
    EXTENDS = "extends"            # A generalizes B
    ACCELERATES = "accelerates"    # A is an optimization of B
    REFINES = "refines"            # A adds soft membership to B


class DataScale(str, Enum):
    SMALL = "small"        # < 1K
    MEDIUM = "medium"      # 1K - 50K
    LARGE = "large"        # 50K - 500K
    VERY_LARGE = "very_large"  # > 500K


class Dimensionality(str, Enum):
    LOW = "low"            # 2-10
    MEDIUM = "medium"      # 10-100
    HIGH = "high"          # 100-768
    VERY_HIGH = "very_high"  # 768-4096+


class DensityProfile(str, Enum):
    UNIFORM = "uniform"
    VARIABLE = "variable"
    CLUSTERED_WITH_NOISE = "clustered_with_noise"
    HIERARCHICAL = "hierarchical"


class NormalizationState(str, Enum):
    L2_NORMALIZED = "l2_normalized"
    RAW = "raw"
    MIXED_SCALE = "mixed_scale"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Layer 1 — Algorithms
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmRelationship:
    """Directed relationship between two algorithms."""
    source: str
    target: str
    relationship: Relationship
    description: str


@dataclass
class Algorithm:
    """A clustering algorithm with its identity and behavior."""
    id: str
    name: str
    category: AlgorithmCategory
    description: str
    how_it_works: str
    time_complexity: str
    space_complexity: str
    deterministic: bool
    requires_k: bool            # does user need to specify number of clusters?
    supports_predict: bool      # can assign new points without refitting?
    supports_soft: bool         # soft membership / probabilities?
    supports_snapshot: bool     # can create ClusterSnapshot?
    supports_noise: bool        # can label points as noise (-1)?
    metrics: list[str]          # supported distance metrics
    convergence_mechanism: str
    internal_constants: dict[str, Any] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class AlgorithmParameter:
    """A parameter for a clustering algorithm, with full semantics."""
    algorithm_id: str
    name: str
    param_type: ParameterType
    default: Any
    valid_range: dict[str, Any] | None  # {"min": ..., "max": ..., "values": [...]}
    description: str
    semantic_meaning: str       # what does this *conceptually* control?
    sensitivity: str            # "high", "medium", "low" — how much does it affect results?
    interactions: list[ParameterInteraction] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)


@dataclass
class ParameterInteraction:
    """How two parameters affect each other."""
    other_param: str            # "algorithm.param" format
    interaction_type: str       # "increases_optimal", "requires", "conflicts", "modulates"
    description: str


# ---------------------------------------------------------------------------
# Layer 3 — Data characteristics
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """A characterization of input data that drives algorithm selection."""
    dimensionality: Dimensionality
    scale: DataScale
    normalization: NormalizationState
    density: DensityProfile
    source_type: str | None     # "openai", "cohere", "tfidf", "tabular", etc.
    intrinsic_dim_estimate: int | None  # estimated intrinsic dimensionality


# ---------------------------------------------------------------------------
# Layer 4 — Decision edges
# ---------------------------------------------------------------------------

@dataclass
class DecisionRule:
    """A rule mapping data characteristics to algorithm recommendations."""
    id: str
    conditions: dict[str, Any]  # data profile conditions
    recommended_algorithm: str
    confidence: float           # 0-1, how strongly we recommend this
    rationale: str
    suggested_params: dict[str, Any] = field(default_factory=dict)
    anti_patterns: list[str] = field(default_factory=list)


@dataclass
class AntiPattern:
    """A configuration that should be avoided."""
    id: str
    description: str
    why_bad: str
    fix: str
    severity: str               # "error", "warning", "info"


# ---------------------------------------------------------------------------
# Layer 5 — Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MetricInterpretation:
    """How to interpret a clustering quality metric."""
    metric_name: str
    range: tuple[float, float]
    direction: str              # "higher_better" or "lower_better"
    thresholds: dict[str, float]  # "good", "ok", "poor"
    caveats: list[str]


@dataclass
class PathologySignature:
    """A recognizable pattern indicating a clustering problem."""
    id: str
    name: str
    symptoms: list[str]
    likely_causes: list[str]
    fixes: list[FixRecipe]


@dataclass
class FixRecipe:
    """A parameter adjustment to fix a clustering pathology."""
    description: str
    parameter_changes: dict[str, str]  # param -> direction or value
    expected_effect: str


# ===========================================================================
# KNOWLEDGE GRAPH INSTANCE
# ===========================================================================

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, Algorithm] = {

    "kmeans": Algorithm(
        id="kmeans",
        name="KMeans",
        category=AlgorithmCategory.CENTROID,
        description="Lloyd's K-means clustering with k-means++ initialization.",
        how_it_works=(
            "Iteratively assigns points to nearest centroid, then recomputes centroids "
            "as cluster means. Uses k-means++ for smart initialization (points selected "
            "with probability proportional to squared distance to nearest existing center). "
            "Runs n_init independent starts and keeps the lowest-inertia result."
        ),
        time_complexity="O(n * k * d * max_iter * n_init)",
        space_complexity="O(n * d + k * d)",
        deterministic=True,  # seeded
        requires_k=True,
        supports_predict=True,
        supports_soft=False,
        supports_snapshot=True,
        supports_noise=False,
        metrics=["euclidean", "cosine"],
        convergence_mechanism="Centroid shift < tol (max movement of any centroid below threshold)",
        internal_constants={
            "hamerly_auto_threshold": "d <= 16 and k <= 32 -> Lloyd, else Hamerly",
            "cosine_forces_lloyd": True,
        },
        strengths=[
            "Well-understood, fast, predictable",
            "Multiple restarts (n_init) for robustness",
            "Hamerly acceleration for large k",
            "Supports snapshot for incremental assignment",
        ],
        weaknesses=[
            "Requires specifying k upfront",
            "Assumes spherical, equally-sized clusters",
            "Sensitive to outliers (mean-based centroids)",
            "Struggles with high-dimensional raw embeddings (curse of dimensionality)",
        ],
    ),

    "minibatch_kmeans": Algorithm(
        id="minibatch_kmeans",
        name="MiniBatchKMeans",
        category=AlgorithmCategory.CENTROID,
        description="Stochastic K-means using random mini-batches for scalability.",
        how_it_works=(
            "Samples random mini-batches per iteration instead of scanning all data. "
            "Centroids updated via exponentially weighted averaging (EWA, alpha=0.1) to "
            "smooth stochastic variance. Early stopping via max_no_improvement counter. "
            "Trades solution quality for speed — typically within 1-2% of full K-means inertia."
        ),
        time_complexity="O(batch_size * k * d * max_iter)",
        space_complexity="O(n * d + k * d)",
        deterministic=True,
        requires_k=True,
        supports_predict=True,
        supports_soft=False,
        supports_snapshot=True,
        supports_noise=False,
        metrics=["euclidean", "cosine"],
        convergence_mechanism="EWA inertia change < tol, or max_no_improvement exceeded",
        internal_constants={
            "ewa_alpha": 0.1,
            "learning_rate": "1.0 / centroid_counts[cluster] (adaptive per-cluster)",
        },
        strengths=[
            "Much faster than KMeans on large datasets",
            "Constant memory per iteration (batch_size, not n)",
            "Good enough solution for most practical use cases",
        ],
        weaknesses=[
            "Slightly worse inertia than full KMeans",
            "No n_init (single online pass)",
            "Stochastic — convergence behavior less predictable",
        ],
    ),

    "dbscan": Algorithm(
        id="dbscan",
        name="DBSCAN",
        category=AlgorithmCategory.DENSITY,
        description="Density-Based Spatial Clustering of Applications with Noise.",
        how_it_works=(
            "Identifies core points (>= min_samples neighbors within eps radius). "
            "Clusters are connected components of core points. Non-core points within eps "
            "of a core point are border points; all others are noise (-1). Uses KD-tree "
            "acceleration when d <= 16 for O(n log n) neighbor queries."
        ),
        time_complexity="O(n log n) with KD-tree, O(n^2) without",
        space_complexity="O(n * d)",
        deterministic=True,
        requires_k=False,
        supports_predict=False,
        supports_soft=False,
        supports_snapshot=False,
        supports_noise=True,
        metrics=["euclidean", "cosine"],
        convergence_mechanism="Single pass — no iteration",
        internal_constants={
            "kdtree_dim_threshold": 16,
            "eps_squared_optimization": "Uses eps^2 for Euclidean to avoid sqrt",
        },
        strengths=[
            "No need to specify k",
            "Finds arbitrarily-shaped clusters",
            "Natural noise/outlier detection",
            "Deterministic (no randomness)",
        ],
        weaknesses=[
            "eps is hard to tune — very sensitive",
            "Struggles with varying density across clusters",
            "Cannot assign new points (no predict())",
            "KD-tree ineffective above ~16 dimensions",
        ],
    ),

    "hdbscan": Algorithm(
        id="hdbscan",
        name="HDBSCAN",
        category=AlgorithmCategory.DENSITY,
        description="Hierarchical DBSCAN — density-based clustering that handles variable density.",
        how_it_works=(
            "6-stage pipeline: (1) compute core distances (k-th NN distance), "
            "(2) build mutual reachability graph (max of raw dist and both core dists), "
            "(3) construct MST via Prim's algorithm, (4) build hierarchy by sorting MST edges, "
            "(5) extract clusters via stability analysis (EOM) or leaf method, "
            "(6) compute soft membership probabilities. Uses Union-Find for hierarchy."
        ),
        time_complexity="O(n^2) for distance matrix, O(n^2 log n) overall",
        space_complexity="O(n^2) for distance matrix",
        deterministic=True,
        requires_k=False,
        supports_predict=False,
        supports_soft=True,
        supports_snapshot=False,
        supports_noise=True,
        metrics=["euclidean", "cosine"],
        convergence_mechanism="Single pass — hierarchy built once, clusters extracted from it",
        internal_constants={
            "lambda_calculation": "1.0 / distance",
            "stability_formula": "sum of (lambda - lambda_birth) for each point",
        },
        strengths=[
            "No need to specify k",
            "Handles varying density across clusters",
            "Soft membership probabilities",
            "Cluster persistence scores for interpretability",
            "More robust than DBSCAN (no eps sensitivity)",
        ],
        weaknesses=[
            "O(n^2) memory — limited to ~50K points",
            "Slower than DBSCAN",
            "Cannot assign new points (no predict())",
            "min_cluster_size still needs tuning",
        ],
    ),

    "agglomerative": Algorithm(
        id="agglomerative",
        name="AgglomerativeClustering",
        category=AlgorithmCategory.HIERARCHICAL,
        description="Bottom-up hierarchical clustering with multiple linkage methods.",
        how_it_works=(
            "Starts with each point as its own cluster. Iteratively merges the two closest "
            "clusters according to linkage criterion. Uses Lance-Williams formula for efficient "
            "distance updates. Produces a full dendrogram (children_ + distances_), then cuts "
            "at n_clusters."
        ),
        time_complexity="O(n^2 log n) with priority queue",
        space_complexity="O(n^2) for distance matrix",
        deterministic=True,
        requires_k=True,
        supports_predict=False,
        supports_soft=False,
        supports_snapshot=False,
        supports_noise=False,
        metrics=["euclidean", "manhattan", "cosine"],
        convergence_mechanism="Merge until n_clusters remain",
        internal_constants={
            "ward_requires_euclidean": True,
        },
        strengths=[
            "Produces interpretable dendrogram",
            "Multiple linkage options for different cluster shapes",
            "Deterministic, no randomness",
            "Can visualize cluster hierarchy",
        ],
        weaknesses=[
            "O(n^2) memory — limited to ~50K points",
            "Cannot assign new points",
            "Ward linkage assumes spherical clusters",
            "Single linkage prone to chaining",
        ],
    ),

    "embedding_cluster": Algorithm(
        id="embedding_cluster",
        name="EmbeddingCluster",
        category=AlgorithmCategory.PIPELINE,
        description=(
            "Purpose-built pipeline for dense embedding vectors: "
            "L2 normalize -> optional PCA -> spherical K-means. "
            "Optimized for vectors from LLM embedding APIs."
        ),
        how_it_works=(
            "Three-stage pipeline: (1) L2-normalize all vectors to unit sphere, "
            "(2) optionally reduce dimensionality via randomized PCA (Halko-Martinsson-Tropp), "
            "(3) run spherical K-means (maximizes dot product / cosine similarity on unit sphere). "
            "Optional VMF refinement adds soft probabilities via von Mises-Fisher mixture model "
            "with EM algorithm. Concentration parameter kappa estimated via Banerjee approximation "
            "with Newton iteration."
        ),
        time_complexity="O(n * d * reduction_dim) for PCA + O(n * k * reduction_dim * max_iter) for clustering",
        space_complexity="O(n * d) for PCA, O(n * reduction_dim) for clustering",
        deterministic=True,
        requires_k=True,
        supports_predict=False,
        supports_soft=True,  # via refine_vmf()
        supports_snapshot=True,
        supports_noise=False,
        metrics=["cosine"],  # implicit — always cosine on unit sphere
        convergence_mechanism=(
            "Angular shift: arccos(dot(old_center, new_center)) < tol, "
            "plus churn tracking (label stability) with patience"
        ),
        internal_constants={
            "spherical_rel_obj_tol": 1e-4,
            "spherical_churn_tol": 0.001,
            "spherical_patience": 2,
            "pca_oversampling": 10,
            "pca_faer_threshold": "n >= 1000 uses faer GEMM",
            "l2_zero_norm_threshold": 1e-30,
            "vmf_bic_formula": "-2 * LL + n_params * ln(n)",
        },
        strengths=[
            "Purpose-built for embedding vectors — handles the full pipeline",
            "PCA removes noise dimensions, dramatically improves cluster quality",
            "Spherical K-means respects cosine geometry (right metric for embeddings)",
            "VMF refinement gives calibrated soft probabilities + concentration",
            "Snapshot support for production incremental assignment",
            "Matryoshka shortcut for compatible models (instant reduction)",
        ],
        weaknesses=[
            "Requires specifying k",
            "PCA is the bottleneck (~99% of runtime on first fit)",
            "Assumes embeddings are roughly uniformly distributed on sphere",
            "Not suitable for non-embedding data (tabular, etc.)",
        ],
    ),

    "spherical_kmeans": Algorithm(
        id="spherical_kmeans",
        name="SphericalKMeans",
        category=AlgorithmCategory.CENTROID,
        description="K-means variant on the unit hypersphere, maximizing cosine similarity.",
        how_it_works=(
            "Like standard K-means but on the unit sphere. Assignment maximizes dot product "
            "(equivalent to minimizing angular distance). Centroid update: average assigned "
            "points then L2-normalize back to unit sphere. Supports Hamerly acceleration "
            "using angular distance bounds."
        ),
        time_complexity="O(n * k * d * max_iter)",
        space_complexity="O(n * d + k * d)",
        deterministic=True,
        requires_k=True,
        supports_predict=False,  # internal, used via EmbeddingCluster
        supports_soft=False,
        supports_snapshot=False,  # used via EmbeddingCluster
        supports_noise=False,
        metrics=["cosine"],
        convergence_mechanism="Angular shift + relative objective change + churn threshold with patience",
        internal_constants={
            "initial_upper_bound": "PI",
            "initial_lower_bound": 0.0,
            "angular_distance": "arccos(dot(a, b))",
        },
        strengths=[
            "Correct geometry for normalized embeddings",
            "Hamerly acceleration available",
            "Multiple convergence criteria for robustness",
        ],
        weaknesses=[
            "Only meaningful on L2-normalized data",
            "Internal algorithm — use EmbeddingCluster for the full pipeline",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Algorithm relationships
# ---------------------------------------------------------------------------

ALGORITHM_RELATIONSHIPS: list[AlgorithmRelationship] = [
    AlgorithmRelationship(
        source="embedding_cluster",
        target="spherical_kmeans",
        relationship=Relationship.COMPOSES,
        description="EmbeddingCluster uses SphericalKMeans as its clustering stage after normalization + PCA",
    ),
    AlgorithmRelationship(
        source="minibatch_kmeans",
        target="kmeans",
        relationship=Relationship.APPROXIMATES,
        description="MiniBatchKMeans approximates KMeans using stochastic mini-batches; typically within 1-2% of optimal inertia",
    ),
    AlgorithmRelationship(
        source="hdbscan",
        target="dbscan",
        relationship=Relationship.EXTENDS,
        description="HDBSCAN generalizes DBSCAN by building a full density hierarchy, removing the eps parameter sensitivity",
    ),
    AlgorithmRelationship(
        source="spherical_kmeans",
        target="kmeans",
        relationship=Relationship.EXTENDS,
        description="SphericalKMeans adapts KMeans to the unit hypersphere, using cosine similarity instead of Euclidean distance",
    ),
    AlgorithmRelationship(
        source="embedding_cluster",
        target="kmeans",
        relationship=Relationship.REFINES,
        description="EmbeddingCluster refines the K-means concept for embedding data via normalization, PCA, and spherical geometry",
    ),
]

# ---------------------------------------------------------------------------
# Parameters (Layer 2)
# ---------------------------------------------------------------------------

PARAMETERS: list[AlgorithmParameter] = [

    # === KMeans ===
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="n_clusters",
        param_type=ParameterType.INT,
        default=None,  # required
        valid_range={"min": 1, "max": "n_samples"},
        description="Number of clusters to find.",
        semantic_meaning="Controls the granularity of grouping. Higher k = finer-grained clusters.",
        sensitivity="high",
        interactions=[
            ParameterInteraction("kmeans.n_init", "modulates",
                "Higher k benefits more from multiple restarts — initialization matters more"),
            ParameterInteraction("kmeans.algorithm", "modulates",
                "Hamerly acceleration benefit increases with k (more distance computations to skip)"),
        ],
        constraints=["Must be <= n_samples", "Must be >= 1"],
        tips=[
            "Use silhouette analysis or elbow method to find optimal k",
            "For embeddings, typical range is 5-200 depending on corpus diversity",
            "k=sqrt(n/2) is a reasonable starting heuristic",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="max_iter",
        param_type=ParameterType.INT,
        default=300,
        valid_range={"min": 1},
        description="Maximum iterations per initialization run.",
        semantic_meaning="Safety limit on convergence. Most runs converge well before 300 iterations.",
        sensitivity="low",
        tips=["Rarely needs tuning. If hitting max_iter, the data may have issues (mixed scales, outliers)."],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="tol",
        param_type=ParameterType.FLOAT,
        default=1e-4,
        valid_range={"min": 0.0},
        description="Convergence tolerance on max centroid shift.",
        semantic_meaning="How precisely centroids must stabilize before stopping. Lower = more precise but slower.",
        sensitivity="low",
        tips=["Default 1e-4 is fine for virtually all use cases."],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="n_init",
        param_type=ParameterType.INT,
        default=10,
        valid_range={"min": 1},
        description="Number of independent runs with different seeds. Best (lowest inertia) wins.",
        semantic_meaning="Robustness against bad initialization. More runs = higher chance of finding global optimum.",
        sensitivity="medium",
        interactions=[
            ParameterInteraction("kmeans.n_clusters", "modulates",
                "Higher k makes the optimization landscape more complex — more n_init helps"),
        ],
        tips=[
            "n_init=10 is a good default for most cases",
            "For production with large k (>50), consider n_init=20",
            "For quick exploration, n_init=3 is often sufficient",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="algorithm",
        param_type=ParameterType.STR,
        default="auto",
        valid_range={"values": ["auto", "lloyd", "hamerly"]},
        description="Which K-means variant to use internally.",
        semantic_meaning=(
            "Lloyd: standard, always works. Hamerly: uses triangle inequality to skip distance "
            "computations — faster for large k on Euclidean data. Auto selects based on d and k."
        ),
        sensitivity="low",  # affects speed, not results
        interactions=[
            ParameterInteraction("kmeans.metric", "conflicts",
                "Cosine metric forces Lloyd — Hamerly bounds assume Euclidean geometry"),
        ],
        constraints=["Cosine metric forces Lloyd regardless of this setting"],
        tips=["Leave on 'auto' unless benchmarking specific configurations."],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="metric",
        param_type=ParameterType.STR,
        default="euclidean",
        valid_range={"values": ["euclidean", "cosine"]},
        description="Distance metric for centroid assignment.",
        semantic_meaning=(
            "Euclidean: straight-line distance, good for tabular/spatial data. "
            "Cosine: angular similarity, good for normalized vectors / text embeddings."
        ),
        sensitivity="high",
        interactions=[
            ParameterInteraction("kmeans.algorithm", "conflicts",
                "Cosine forces Lloyd algorithm (Hamerly bounds don't hold for cosine)"),
        ],
        tips=[
            "Use cosine for text/document embeddings",
            "Use euclidean for spatial, tabular, or image feature data",
            "If using cosine, consider EmbeddingCluster instead — it's purpose-built for this",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="kmeans",
        name="random_state",
        param_type=ParameterType.INT,
        default=0,
        valid_range=None,
        description="Random seed for reproducibility.",
        semantic_meaning="Controls k-means++ initialization and run ordering. Same seed = same results.",
        sensitivity="low",
    ),

    # === MiniBatchKMeans ===
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="n_clusters",
        param_type=ParameterType.INT,
        default=None,
        valid_range={"min": 1, "max": "n_samples"},
        description="Number of clusters.",
        semantic_meaning="Same as KMeans — controls grouping granularity.",
        sensitivity="high",
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="batch_size",
        param_type=ParameterType.INT,
        default=1024,
        valid_range={"min": 1},
        description="Number of samples per mini-batch.",
        semantic_meaning=(
            "Tradeoff between speed and solution quality. Larger batches = better centroids "
            "but slower. When batch_size >= n, degrades to standard K-means."
        ),
        sensitivity="medium",
        interactions=[
            ParameterInteraction("minibatch_kmeans.n_clusters", "modulates",
                "batch_size should be >> n_clusters for stable centroid updates (aim for 10x k)"),
        ],
        tips=[
            "Default 1024 works well for 10K-500K points",
            "For large k (>100), increase batch_size to at least 10*k",
            "If batch_size >= n_samples, effectively becomes standard K-means in one pass",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="max_iter",
        param_type=ParameterType.INT,
        default=100,
        valid_range={"min": 1},
        description="Maximum iterations.",
        semantic_meaning="Lower default (100 vs 300) because mini-batches converge faster with early stopping.",
        sensitivity="low",
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="tol",
        param_type=ParameterType.FLOAT,
        default=0.0,
        valid_range={"min": 0.0},
        description="Early stopping tolerance on EWA inertia change. 0.0 disables early stopping.",
        semantic_meaning=(
            "Default 0 because stochastic updates don't converge smoothly — "
            "max_no_improvement is the preferred early stopping mechanism."
        ),
        sensitivity="low",
        interactions=[
            ParameterInteraction("minibatch_kmeans.max_no_improvement", "modulates",
                "tol > 0 enables the max_no_improvement counter"),
        ],
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="max_no_improvement",
        param_type=ParameterType.INT,
        default=10,
        valid_range={"min": 1},
        description="Stop after this many iterations with no inertia improvement.",
        semantic_meaning="Patience counter — prevents wasting compute when centroids have stabilized.",
        sensitivity="low",
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="metric",
        param_type=ParameterType.STR,
        default="euclidean",
        valid_range={"values": ["euclidean", "cosine"]},
        description="Distance metric.",
        semantic_meaning="Same as KMeans.",
        sensitivity="high",
    ),
    AlgorithmParameter(
        algorithm_id="minibatch_kmeans",
        name="random_state",
        param_type=ParameterType.INT,
        default=0,
        valid_range=None,
        description="Random seed.",
        semantic_meaning="Controls batch sampling and initialization.",
        sensitivity="low",
    ),

    # === DBSCAN ===
    AlgorithmParameter(
        algorithm_id="dbscan",
        name="eps",
        param_type=ParameterType.FLOAT,
        default=0.5,
        valid_range={"min": 0.0, "min_exclusive": True},
        description="Maximum distance between two samples to be considered neighbors.",
        semantic_meaning=(
            "The fundamental density threshold. Points within eps of a core point belong "
            "to its cluster. Too small = everything is noise. Too large = everything is one cluster."
        ),
        sensitivity="high",
        interactions=[
            ParameterInteraction("dbscan.min_samples", "modulates",
                "eps and min_samples jointly define density — larger eps with larger min_samples can give similar results"),
            ParameterInteraction("dbscan.metric", "modulates",
                "Cosine distance range is [0, 2], so eps must be in that range. Euclidean has no upper bound."),
        ],
        tips=[
            "Use k-distance plot (sorted k-th nearest neighbor distances) to find the 'elbow'",
            "For cosine distance, eps is typically in [0.1, 0.5]",
            "For euclidean, eps depends entirely on data scale — normalize first or use standardized data",
            "eps is the single hardest parameter to tune in all of clustering",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="dbscan",
        name="min_samples",
        param_type=ParameterType.INT,
        default=5,
        valid_range={"min": 1},
        description="Minimum points in eps-neighborhood to qualify as a core point.",
        semantic_meaning=(
            "Controls noise sensitivity. Higher values = stricter density requirement = more noise points. "
            "min_samples=1 means every point is a core point (no noise, every isolated point is its own cluster)."
        ),
        sensitivity="medium",
        interactions=[
            ParameterInteraction("dbscan.eps", "modulates",
                "Jointly defines density with eps"),
        ],
        tips=[
            "Rule of thumb: min_samples >= dimensionality + 1",
            "For noisy data, increase min_samples to filter more aggressively",
            "min_samples=1 disables noise detection entirely",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="dbscan",
        name="metric",
        param_type=ParameterType.STR,
        default="euclidean",
        valid_range={"values": ["euclidean", "cosine"]},
        description="Distance metric.",
        semantic_meaning="Determines what 'close' means. Affects valid eps range.",
        sensitivity="high",
    ),

    # === HDBSCAN ===
    AlgorithmParameter(
        algorithm_id="hdbscan",
        name="min_cluster_size",
        param_type=ParameterType.INT,
        default=5,
        valid_range={"min": 2},
        description="Minimum number of points to form a cluster.",
        semantic_meaning=(
            "The primary tuning knob. Smaller = more fine-grained clusters. "
            "Larger = only keeps robust, well-populated clusters."
        ),
        sensitivity="high",
        tips=[
            "Start with min_cluster_size=15 for most datasets",
            "For small datasets (<500), try 5-10",
            "For large datasets (>10K), try 25-100",
            "This is much easier to tune than DBSCAN's eps",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="hdbscan",
        name="min_samples",
        param_type=ParameterType.OPTIONAL_INT,
        default=None,
        valid_range={"min": 1},
        description="Core distance parameter. Defaults to min_cluster_size if None.",
        semantic_meaning=(
            "Controls how conservative the density estimate is. Higher = more conservative "
            "(smooths out local density spikes). Usually fine at default."
        ),
        sensitivity="low",
        tips=[
            "Leave as None (= min_cluster_size) unless you have specific density requirements",
            "Lower min_samples makes clusters more 'ragged' at borders",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="hdbscan",
        name="cluster_selection_method",
        param_type=ParameterType.STR,
        default="eom",
        valid_range={"values": ["eom", "leaf"]},
        description="How to extract flat clusters from the hierarchy.",
        semantic_meaning=(
            "EOM (Excess of Mass): stability-based, finds clusters at mixed scales. "
            "Leaf: uses leaf nodes, tends to produce more evenly-sized clusters."
        ),
        sensitivity="medium",
        tips=[
            "EOM is usually better — it finds the most stable clusters at any scale",
            "Leaf can be better when you want uniformly-sized clusters",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="hdbscan",
        name="metric",
        param_type=ParameterType.STR,
        default="euclidean",
        valid_range={"values": ["euclidean", "cosine"]},
        description="Distance metric.",
        semantic_meaning="Same as DBSCAN.",
        sensitivity="high",
    ),

    # === AgglomerativeClustering ===
    AlgorithmParameter(
        algorithm_id="agglomerative",
        name="n_clusters",
        param_type=ParameterType.INT,
        default=2,
        valid_range={"min": 1, "max": "n_samples"},
        description="Target number of clusters.",
        semantic_meaning="Where to cut the dendrogram.",
        sensitivity="high",
    ),
    AlgorithmParameter(
        algorithm_id="agglomerative",
        name="linkage",
        param_type=ParameterType.STR,
        default="ward",
        valid_range={"values": ["ward", "complete", "average", "single"]},
        description="Cluster merge criterion.",
        semantic_meaning=(
            "Ward: minimizes within-cluster variance — produces compact, spherical clusters. "
            "Complete: max distance — produces compact clusters but not necessarily spherical. "
            "Average: mean distance — balanced approach. "
            "Single: min distance — can find elongated/chain-like clusters but prone to chaining artifacts."
        ),
        sensitivity="high",
        interactions=[
            ParameterInteraction("agglomerative.metric", "conflicts",
                "Ward linkage requires Euclidean metric — raises ValueError otherwise"),
        ],
        constraints=["Ward linkage requires metric='euclidean'"],
        tips=[
            "Ward is best for compact, spherical clusters (most common case)",
            "Complete is good when clusters have similar diameters",
            "Average is a safe middle ground",
            "Single is useful for detecting elongated structures but can produce artifacts",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="agglomerative",
        name="metric",
        param_type=ParameterType.STR,
        default="euclidean",
        valid_range={"values": ["euclidean", "manhattan", "cosine"]},
        description="Distance metric. Note: ward linkage requires euclidean.",
        semantic_meaning="What 'distance' means for merge decisions.",
        sensitivity="high",
        constraints=["Must be 'euclidean' when linkage='ward'"],
    ),

    # === EmbeddingCluster ===
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="n_clusters",
        param_type=ParameterType.INT,
        default=50,
        valid_range={"min": 1, "max": "n_samples"},
        description="Target number of clusters for spherical K-means.",
        semantic_meaning="Grouping granularity for embeddings. Default 50 reflects typical embedding corpus diversity.",
        sensitivity="high",
        tips=[
            "Start with k=sqrt(n/2) as a heuristic",
            "For OpenAI embeddings of documents, 20-100 is typical",
            "For fine-grained semantic grouping, go higher (100-500)",
            "Use silhouette analysis on the REDUCED data to tune",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="reduction_dim",
        param_type=ParameterType.OPTIONAL_INT,
        default=128,
        valid_range={"min": 1},
        description="Target dimensionality after PCA. None to skip reduction.",
        semantic_meaning=(
            "Removes noise dimensions from embeddings, dramatically improving cluster quality. "
            "128 is the sweet spot for most embedding models (captures 90%+ variance from 1536d). "
            "None skips PCA — use for already-reduced data or dimensions <= 256."
        ),
        sensitivity="high",
        interactions=[
            ParameterInteraction("embedding_cluster.n_clusters", "modulates",
                "Higher reduction_dim shifts optimal k upward — more preserved dimensions = more distinguishable clusters"),
        ],
        tips=[
            "128 for 1536d embeddings (OpenAI text-embedding-3-small)",
            "64-128 for 768d embeddings (many open-source models)",
            "256 for 3072d embeddings (OpenAI text-embedding-3-large)",
            "None if data is already <= 256 dimensions",
            "PCA is ~99% of runtime on first fit — use EmbeddingReducer to pay this cost once",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="max_iter",
        param_type=ParameterType.INT,
        default=100,
        valid_range={"min": 1},
        description="Max iterations for spherical K-means.",
        semantic_meaning="Usually converges in 10-30 iterations. 100 is a generous safety margin.",
        sensitivity="low",
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="tol",
        param_type=ParameterType.FLOAT,
        default=1e-6,
        valid_range={"min": 0.0},
        description="Convergence tolerance on angular shift.",
        semantic_meaning="Tighter than standard K-means (1e-6 vs 1e-4) because spherical geometry needs more precision.",
        sensitivity="low",
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="n_init",
        param_type=ParameterType.INT,
        default=5,
        valid_range={"min": 1},
        description="Number of independent starts.",
        semantic_meaning="Lower default (5 vs 10) because embedding data is typically cleaner than tabular data.",
        sensitivity="medium",
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="reduction",
        param_type=ParameterType.STR,
        default="pca",
        valid_range={"values": ["pca", "matryoshka"]},
        description="Dimensionality reduction method.",
        semantic_meaning=(
            "PCA: randomized PCA, works with any embeddings. "
            "Matryoshka: prefix truncation (instant, no fitting), only for Matryoshka-trained models "
            "(OpenAI text-embedding-3-*, Cohere embed-v3). Matryoshka is 1000x faster but model-specific."
        ),
        sensitivity="medium",
        tips=[
            "Use 'pca' unless you know your model is Matryoshka-trained",
            "Matryoshka models: OpenAI text-embedding-3-small/large, Cohere embed-v3",
            "If using matryoshka, reduction_dim should match a supported truncation point (e.g., 256, 512)",
        ],
    ),
    AlgorithmParameter(
        algorithm_id="embedding_cluster",
        name="random_state",
        param_type=ParameterType.INT,
        default=0,
        valid_range=None,
        description="Random seed for PCA and initialization.",
        semantic_meaning="Controls both PCA random matrix and K-means++ seeding.",
        sensitivity="low",
    ),
]

# ---------------------------------------------------------------------------
# Decision rules (Layer 4)
# ---------------------------------------------------------------------------

DECISION_RULES: list[DecisionRule] = [

    # --- High-dimensional embeddings ---
    DecisionRule(
        id="highd_normalized_unknown_k",
        conditions={
            "dimensionality": ["high", "very_high"],
            "normalization": "l2_normalized",
            "requires_k": False,
        },
        recommended_algorithm="hdbscan",
        confidence=0.8,
        rationale=(
            "High-dimensional normalized embeddings with unknown k: HDBSCAN discovers "
            "cluster count automatically and handles variable density. Use cosine metric."
        ),
        suggested_params={"min_cluster_size": 15, "metric": "cosine"},
        anti_patterns=["dbscan_highd"],
    ),
    DecisionRule(
        id="highd_normalized_known_k",
        conditions={
            "dimensionality": ["high", "very_high"],
            "normalization": ["l2_normalized", "raw"],
            "requires_k": True,
            "source_type": ["openai", "cohere", "voyage", "embedding"],
        },
        recommended_algorithm="embedding_cluster",
        confidence=0.95,
        rationale=(
            "High-dimensional embedding vectors with known k: EmbeddingCluster is purpose-built. "
            "Normalizes, reduces via PCA, and clusters on the unit sphere. "
            "The PCA step is critical — raw high-dim embeddings cluster poorly."
        ),
        suggested_params={"reduction_dim": 128, "n_init": 5},
    ),
    DecisionRule(
        id="highd_raw_unknown_k",
        conditions={
            "dimensionality": ["high", "very_high"],
            "normalization": "raw",
            "requires_k": False,
        },
        recommended_algorithm="hdbscan",
        confidence=0.7,
        rationale=(
            "High-dimensional raw data with unknown k: HDBSCAN is the safest bet. "
            "Consider whether the data should be normalized first."
        ),
        suggested_params={"min_cluster_size": 15, "metric": "euclidean"},
    ),

    # --- Medium-dimensional ---
    DecisionRule(
        id="medd_known_k_large_n",
        conditions={
            "dimensionality": ["medium"],
            "scale": ["large", "very_large"],
            "requires_k": True,
        },
        recommended_algorithm="minibatch_kmeans",
        confidence=0.85,
        rationale=(
            "Medium-dimensional data, large n, known k: MiniBatchKMeans gives near-optimal "
            "results in a fraction of the time. Full KMeans would be too slow."
        ),
        suggested_params={"batch_size": 1024},
    ),
    DecisionRule(
        id="medd_known_k_small_n",
        conditions={
            "dimensionality": ["low", "medium"],
            "scale": ["small", "medium"],
            "requires_k": True,
        },
        recommended_algorithm="kmeans",
        confidence=0.9,
        rationale=(
            "Low-to-medium dimensional data, manageable size, known k: standard KMeans "
            "with n_init=10 will find a good solution. No need for approximations."
        ),
        suggested_params={"n_init": 10, "algorithm": "auto"},
    ),

    # --- Density/noise scenarios ---
    DecisionRule(
        id="variable_density_with_noise",
        conditions={
            "density": "variable",
            "requires_k": False,
        },
        recommended_algorithm="hdbscan",
        confidence=0.9,
        rationale=(
            "Variable density with noise: HDBSCAN is the clear winner. DBSCAN would need "
            "different eps values for different density regions. HDBSCAN handles this natively."
        ),
        suggested_params={"min_cluster_size": 15},
        anti_patterns=["dbscan_variable_density"],
    ),
    DecisionRule(
        id="uniform_density_with_noise",
        conditions={
            "density": "uniform",
            "requires_k": False,
            "dimensionality": ["low", "medium"],
        },
        recommended_algorithm="dbscan",
        confidence=0.8,
        rationale=(
            "Uniform density, noise present, low-to-medium dimensions: DBSCAN works well. "
            "KD-tree acceleration kicks in for d <= 16. Uniform density means a single eps works."
        ),
        suggested_params={"min_samples": 5},
    ),

    # --- Hierarchical needs ---
    DecisionRule(
        id="need_dendrogram",
        conditions={
            "scale": ["small", "medium"],
            "requires_k": True,
            "need_dendrogram": True,
        },
        recommended_algorithm="agglomerative",
        confidence=0.85,
        rationale=(
            "When you need a hierarchical structure (dendrogram), AgglomerativeClustering "
            "is the right choice. Limited to ~50K points due to O(n^2) memory."
        ),
        suggested_params={"linkage": "ward"},
    ),

    # --- Very large datasets ---
    DecisionRule(
        id="very_large_known_k",
        conditions={
            "scale": "very_large",
            "requires_k": True,
        },
        recommended_algorithm="minibatch_kmeans",
        confidence=0.9,
        rationale=(
            "For >500K points with known k, MiniBatchKMeans is the only practical centroid-based "
            "option. Full KMeans and HDBSCAN/Agglomerative are too slow."
        ),
        suggested_params={"batch_size": 2048},
    ),
]

# ---------------------------------------------------------------------------
# Anti-patterns
# ---------------------------------------------------------------------------

ANTI_PATTERNS: list[AntiPattern] = [

    AntiPattern(
        id="euclidean_highd_embeddings",
        description="Using Euclidean distance on raw high-dimensional embeddings (>100d)",
        why_bad=(
            "In high dimensions, Euclidean distances concentrate — all pairwise distances "
            "become similar. Cosine similarity (or equivalently, Euclidean after L2 normalization) "
            "preserves angular relationships which is what matters for semantic similarity."
        ),
        fix="Use EmbeddingCluster (handles normalization + PCA automatically) or manually L2-normalize and use cosine metric.",
        severity="error",
    ),
    AntiPattern(
        id="kmeans_no_pca_highd",
        description="Running KMeans directly on 1000+ dimensional embeddings without dimensionality reduction",
        why_bad=(
            "High-dimensional spaces have noise dimensions that degrade cluster quality. "
            "PCA to 64-256 dims removes this noise and dramatically improves clustering. "
            "Silhouette scores typically jump from ~0.05 to ~0.25 after PCA."
        ),
        fix="Use EmbeddingCluster (includes PCA) or apply EmbeddingReducer first, then cluster the reduced data.",
        severity="error",
    ),
    AntiPattern(
        id="dbscan_highd",
        description="Using DBSCAN on data with more than ~16 dimensions",
        why_bad=(
            "KD-tree acceleration only works for d <= 16, falling back to O(n^2). "
            "Additionally, the eps parameter becomes nearly impossible to tune in high dimensions "
            "due to distance concentration."
        ),
        fix="Reduce dimensionality first (PCA to d <= 16), or use HDBSCAN which is more robust to high-d.",
        severity="warning",
    ),
    AntiPattern(
        id="dbscan_variable_density",
        description="Using DBSCAN on data with varying cluster densities",
        why_bad=(
            "A single eps value cannot capture clusters at different density scales. "
            "Dense clusters will be found but sparse clusters will be split or marked as noise."
        ),
        fix="Use HDBSCAN — it builds a hierarchy of densities and extracts clusters at their natural scale.",
        severity="warning",
    ),
    AntiPattern(
        id="hdbscan_large_n",
        description="Using HDBSCAN on datasets with more than ~50K points",
        why_bad=(
            "HDBSCAN computes a full O(n^2) distance matrix. At 50K points that's ~10GB of memory. "
            "At 100K it's ~40GB."
        ),
        fix="Subsample, or use MiniBatchKMeans/KMeans with a predetermined k. For embeddings, use EmbeddingCluster.",
        severity="warning",
    ),
    AntiPattern(
        id="ward_non_euclidean",
        description="Using Ward linkage with non-Euclidean metric",
        why_bad="Ward linkage minimizes variance, which is only meaningful in Euclidean space. The code will raise ValueError.",
        fix="Use metric='euclidean' with Ward, or switch to 'complete'/'average' linkage for non-Euclidean metrics.",
        severity="error",
    ),
    AntiPattern(
        id="silhouette_panic_highd",
        description="Interpreting low silhouette scores (<0.25) on high-dimensional data as 'bad clustering'",
        why_bad=(
            "Silhouette scores are systematically lower in high dimensions due to distance concentration. "
            "A score of 0.15 in 1536d might be excellent. Always compute silhouette on PCA-reduced data."
        ),
        fix="Apply PCA first, then compute silhouette on the reduced data. Compare scores at different k values relatively, not absolutely.",
        severity="info",
    ),
    AntiPattern(
        id="matryoshka_wrong_model",
        description="Using matryoshka reduction with non-Matryoshka-trained embedding models",
        why_bad=(
            "Matryoshka reduction simply truncates to the first N dimensions. This only works if the model "
            "was trained to pack information into earlier dimensions (Matryoshka training). For standard models, "
            "the first N dimensions are arbitrary and truncation destroys information."
        ),
        fix="Use reduction='pca' for non-Matryoshka models. Known Matryoshka models: OpenAI text-embedding-3-*, Cohere embed-v3.",
        severity="error",
    ),
]

# ---------------------------------------------------------------------------
# Diagnostics (Layer 5)
# ---------------------------------------------------------------------------

METRIC_INTERPRETATIONS: list[MetricInterpretation] = [

    MetricInterpretation(
        metric_name="silhouette_score",
        range=(-1.0, 1.0),
        direction="higher_better",
        thresholds={
            "excellent": 0.7,
            "good": 0.5,
            "fair": 0.25,
            "poor": 0.0,
        },
        caveats=[
            "Systematically lower in high dimensions — always evaluate on PCA-reduced data",
            "Biased toward spherical, equally-sized clusters (favors KMeans-like solutions)",
            "Noise points (label=-1) are excluded from computation",
            "Meaningless with only 1 cluster — requires >= 2",
            "Compare relative values across k, not absolute thresholds",
        ],
    ),
    MetricInterpretation(
        metric_name="calinski_harabasz_score",
        range=(0.0, float("inf")),
        direction="higher_better",
        thresholds={
            "good": 500.0,    # very data-dependent
            "fair": 100.0,
            "poor": 50.0,
        },
        caveats=[
            "Absolute values are not comparable across datasets — only compare across k for same data",
            "Strongly biased toward spherical, well-separated clusters",
            "Tends to favor larger k (more clusters) — use elbow on the CH curve",
            "Ratio of between-cluster to within-cluster variance",
        ],
    ),
    MetricInterpretation(
        metric_name="davies_bouldin_score",
        range=(0.0, float("inf")),
        direction="lower_better",
        thresholds={
            "good": 0.5,
            "fair": 1.0,
            "poor": 2.0,
        },
        caveats=[
            "Average worst-case similarity ratio — sensitive to outlier clusters",
            "Only uses cluster centroids and scatter — misses non-convex structure",
            "Lower is better (opposite of silhouette and CH)",
            "Good for comparing configurations on same dataset",
        ],
    ),
]

PATHOLOGY_SIGNATURES: list[PathologySignature] = [

    PathologySignature(
        id="one_giant_cluster",
        name="One Dominant Cluster",
        symptoms=[
            "One cluster contains >60% of all points",
            "Silhouette score is very low (<0.1)",
            "Other clusters are tiny fragments",
        ],
        likely_causes=[
            "k is too low — the data has more structure than k allows",
            "Distance metric mismatch (e.g., euclidean on unnormalized embeddings)",
            "Data needs dimensionality reduction (noise dims dominate distances)",
        ],
        fixes=[
            FixRecipe(
                description="Increase k",
                parameter_changes={"n_clusters": "increase by 2-3x"},
                expected_effect="Giant cluster splits into meaningful sub-clusters",
            ),
            FixRecipe(
                description="Add PCA / use EmbeddingCluster",
                parameter_changes={"reduction_dim": "128 (for 1536d embeddings)"},
                expected_effect="Noise dimensions removed, distance metric becomes meaningful",
            ),
            FixRecipe(
                description="Switch to cosine metric",
                parameter_changes={"metric": "cosine"},
                expected_effect="Angular separation used instead of magnitude, better for embeddings",
            ),
        ],
    ),

    PathologySignature(
        id="too_many_singletons",
        name="Excessive Singleton Clusters",
        symptoms=[
            "Many clusters with 1-3 points",
            "High k relative to n",
            "CH score may be misleadingly high",
        ],
        likely_causes=[
            "k is too high for the data's natural structure",
            "Outliers are being treated as their own clusters (KMeans can't filter noise)",
        ],
        fixes=[
            FixRecipe(
                description="Decrease k",
                parameter_changes={"n_clusters": "decrease by 30-50%"},
                expected_effect="Small clusters merge into coherent groups",
            ),
            FixRecipe(
                description="Switch to HDBSCAN for automatic k discovery",
                parameter_changes={"algorithm": "hdbscan", "min_cluster_size": "10-25"},
                expected_effect="Algorithm finds natural cluster count, labels outliers as noise",
            ),
        ],
    ),

    PathologySignature(
        id="all_noise",
        name="Everything Labeled as Noise",
        symptoms=[
            "All or nearly all labels are -1 (DBSCAN/HDBSCAN)",
            "No meaningful clusters found",
        ],
        likely_causes=[
            "eps is too small (DBSCAN) — density threshold too strict",
            "min_cluster_size is too large (HDBSCAN)",
            "Data is genuinely uniform / not clustered",
            "Wrong distance metric (euclidean on high-d embeddings)",
        ],
        fixes=[
            FixRecipe(
                description="Increase eps (DBSCAN)",
                parameter_changes={"eps": "increase 2-5x, use k-distance plot"},
                expected_effect="More points become core points, clusters form",
            ),
            FixRecipe(
                description="Decrease min_cluster_size (HDBSCAN)",
                parameter_changes={"min_cluster_size": "decrease to 5-10"},
                expected_effect="Smaller groups recognized as clusters",
            ),
            FixRecipe(
                description="Reduce dimensionality first",
                parameter_changes={"reduction_dim": "32-128"},
                expected_effect="Distance metric becomes more discriminative",
            ),
        ],
    ),

    PathologySignature(
        id="unstable_across_runs",
        name="Unstable Clustering (Different Results Per Run)",
        symptoms=[
            "Different random seeds produce very different cluster assignments",
            "ARI between runs is low (<0.7)",
            "Inertia varies significantly across n_init runs",
        ],
        likely_causes=[
            "k is in an ambiguous range — data supports multiple valid partitions",
            "Clusters are not well-separated",
            "Too few n_init runs — not enough exploration",
        ],
        fixes=[
            FixRecipe(
                description="Increase n_init",
                parameter_changes={"n_init": "20-50"},
                expected_effect="More exploration of initialization space, more consistent best solution",
            ),
            FixRecipe(
                description="Try different k values",
                parameter_changes={"n_clusters": "try k-2, k-1, k+1, k+2"},
                expected_effect="Find the k where clustering stabilizes",
            ),
            FixRecipe(
                description="Use HDBSCAN to discover natural k",
                parameter_changes={"algorithm": "hdbscan"},
                expected_effect="Let the data tell you how many clusters exist",
            ),
        ],
    ),

    PathologySignature(
        id="uneven_cluster_sizes",
        name="Highly Uneven Cluster Sizes",
        symptoms=[
            "Cluster sizes vary by >10x (e.g., sizes [5000, 50, 20, 15, 10])",
            "A few large clusters and many small ones",
            "Not the same as 'one giant cluster' — multiple large clusters exist",
        ],
        likely_causes=[
            "Natural data distribution is skewed (some topics/categories are more common)",
            "K-means assumes equal-variance clusters — real data often violates this",
            "Initialization bias toward dense regions",
        ],
        fixes=[
            FixRecipe(
                description="Accept it — uneven sizes may be correct",
                parameter_changes={},
                expected_effect="If the large clusters are coherent, the clustering is fine",
            ),
            FixRecipe(
                description="Use HDBSCAN with 'leaf' selection for more even sizes",
                parameter_changes={"algorithm": "hdbscan", "cluster_selection_method": "leaf"},
                expected_effect="Leaf selection tends to produce more evenly-sized clusters",
            ),
            FixRecipe(
                description="Increase k to split large clusters",
                parameter_changes={"n_clusters": "increase 1.5-2x"},
                expected_effect="Large clusters subdivide into more granular groups",
            ),
        ],
    ),

    PathologySignature(
        id="slow_convergence",
        name="Slow Convergence / Hitting max_iter",
        symptoms=[
            "n_iter_ equals max_iter",
            "Inertia is still decreasing when iterations stop",
            "Runtime is unexpectedly long",
        ],
        likely_causes=[
            "Data has mixed scales (some features dominate distance)",
            "k is near a bifurcation point — centroids oscillate between two configurations",
            "Very high dimensionality without PCA",
        ],
        fixes=[
            FixRecipe(
                description="Increase max_iter",
                parameter_changes={"max_iter": "500-1000"},
                expected_effect="Allows convergence to complete",
            ),
            FixRecipe(
                description="Normalize/standardize data",
                parameter_changes={},
                expected_effect="All features contribute equally to distance",
            ),
            FixRecipe(
                description="Apply PCA to reduce dimensionality",
                parameter_changes={"reduction_dim": "64-256"},
                expected_effect="Faster per-iteration and typically fewer iterations needed",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

def get_algorithm(algorithm_id: str) -> Algorithm | None:
    """Get algorithm by ID."""
    return ALGORITHMS.get(algorithm_id)


def get_parameters_for(algorithm_id: str) -> list[AlgorithmParameter]:
    """Get all parameters for an algorithm."""
    return [p for p in PARAMETERS if p.algorithm_id == algorithm_id]


def get_relationships_for(algorithm_id: str) -> list[AlgorithmRelationship]:
    """Get all relationships involving an algorithm."""
    return [r for r in ALGORITHM_RELATIONSHIPS
            if r.source == algorithm_id or r.target == algorithm_id]


def match_decision_rules(profile: DataProfile, requires_k: bool | None = None) -> list[DecisionRule]:
    """Find applicable decision rules for a data profile."""
    matches = []
    for rule in DECISION_RULES:
        conds = rule.conditions
        match = True

        # Check dimensionality
        if "dimensionality" in conds:
            allowed = conds["dimensionality"]
            if isinstance(allowed, list):
                if profile.dimensionality.value not in allowed:
                    match = False
            elif profile.dimensionality.value != allowed:
                match = False

        # Check scale
        if "scale" in conds:
            allowed = conds["scale"]
            if isinstance(allowed, list):
                if profile.scale.value not in allowed:
                    match = False
            elif profile.scale.value != allowed:
                match = False

        # Check normalization
        if "normalization" in conds:
            allowed = conds["normalization"]
            if isinstance(allowed, list):
                if profile.normalization.value not in allowed:
                    match = False
            elif profile.normalization.value != allowed:
                match = False

        # Check density
        if "density" in conds:
            if profile.density.value != conds["density"]:
                match = False

        # Check source type
        if "source_type" in conds:
            if profile.source_type not in conds["source_type"]:
                match = False

        # Check requires_k
        if "requires_k" in conds and requires_k is not None:
            if conds["requires_k"] != requires_k:
                match = False

        if match:
            matches.append(rule)

    # Sort by confidence descending
    matches.sort(key=lambda r: r.confidence, reverse=True)
    return matches


def check_anti_patterns(
    algorithm_id: str,
    params: dict[str, Any],
    profile: DataProfile | None = None,
) -> list[AntiPattern]:
    """Check for anti-patterns in a configuration."""
    triggered = []

    algo = ALGORITHMS.get(algorithm_id)
    if not algo:
        return triggered

    # Euclidean on high-d embeddings
    if profile and profile.dimensionality in (Dimensionality.HIGH, Dimensionality.VERY_HIGH):
        metric = params.get("metric", "euclidean")
        if metric == "euclidean" and profile.source_type in ("openai", "cohere", "voyage", "embedding"):
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "euclidean_highd_embeddings"))

    # KMeans without PCA on high-d
    if algorithm_id == "kmeans" and profile:
        if profile.dimensionality in (Dimensionality.HIGH, Dimensionality.VERY_HIGH):
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "kmeans_no_pca_highd"))

    # DBSCAN on high-d
    if algorithm_id == "dbscan" and profile:
        if profile.dimensionality in (Dimensionality.HIGH, Dimensionality.VERY_HIGH):
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "dbscan_highd"))

    # DBSCAN with variable density
    if algorithm_id == "dbscan" and profile:
        if profile.density == DensityProfile.VARIABLE:
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "dbscan_variable_density"))

    # HDBSCAN on very large data
    if algorithm_id == "hdbscan" and profile:
        if profile.scale == DataScale.VERY_LARGE:
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "hdbscan_large_n"))

    # Ward + non-euclidean
    if algorithm_id == "agglomerative":
        if params.get("linkage") == "ward" and params.get("metric", "euclidean") != "euclidean":
            triggered.append(next(a for a in ANTI_PATTERNS if a.id == "ward_non_euclidean"))

    # Matryoshka with wrong model
    if algorithm_id == "embedding_cluster":
        if params.get("reduction") == "matryoshka":
            if profile and profile.source_type not in ("openai", "cohere"):
                triggered.append(next(a for a in ANTI_PATTERNS if a.id == "matryoshka_wrong_model"))

    return triggered


def diagnose_results(
    metric_scores: dict[str, float],
    cluster_sizes: list[int],
    n_iter: int | None = None,
    max_iter: int | None = None,
    n_noise: int = 0,
    n_total: int = 0,
) -> list[PathologySignature]:
    """Diagnose clustering results for pathologies."""
    triggered = []

    if cluster_sizes:
        total_points = sum(cluster_sizes)
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)

        # One giant cluster
        if total_points > 0 and max_size / total_points > 0.6:
            triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "one_giant_cluster"))

        # Too many singletons
        singletons = sum(1 for s in cluster_sizes if s <= 3)
        if len(cluster_sizes) > 3 and singletons / len(cluster_sizes) > 0.3:
            triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "too_many_singletons"))

        # Highly uneven
        if max_size > 0 and min_size > 0 and max_size / min_size > 10 and len(cluster_sizes) > 2:
            triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "uneven_cluster_sizes"))

    # All noise
    if n_total > 0 and n_noise / n_total > 0.8:
        triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "all_noise"))

    # Slow convergence
    if n_iter is not None and max_iter is not None and n_iter >= max_iter:
        triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "slow_convergence"))

    # Low silhouette
    sil = metric_scores.get("silhouette_score")
    if sil is not None and sil < 0.1:
        # Could be one_giant_cluster or just bad clustering — check if already flagged
        if not any(p.id == "one_giant_cluster" for p in triggered):
            triggered.append(next(p for p in PATHOLOGY_SIGNATURES if p.id == "unstable_across_runs"))

    return triggered
