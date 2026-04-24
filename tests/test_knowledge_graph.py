"""Tests for the clustering knowledge graph — pure logic, no external deps."""


from rustcluster_mcp.knowledge_graph import (
    ALGORITHM_RELATIONSHIPS,
    ALGORITHMS,
    ANTI_PATTERNS,
    METRIC_INTERPRETATIONS,
    PARAMETERS,
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
# Algorithm registry
# ---------------------------------------------------------------------------


class TestAlgorithmRegistry:
    def test_all_seven_algorithms_registered(self):
        assert len(ALGORITHMS) == 7
        expected = {
            "kmeans", "minibatch_kmeans", "dbscan", "hdbscan",
            "agglomerative", "embedding_cluster", "spherical_kmeans",
        }
        assert set(ALGORITHMS.keys()) == expected

    def test_algorithm_categories(self):
        assert ALGORITHMS["kmeans"].category == AlgorithmCategory.CENTROID
        assert ALGORITHMS["minibatch_kmeans"].category == AlgorithmCategory.CENTROID
        assert ALGORITHMS["spherical_kmeans"].category == AlgorithmCategory.CENTROID
        assert ALGORITHMS["dbscan"].category == AlgorithmCategory.DENSITY
        assert ALGORITHMS["hdbscan"].category == AlgorithmCategory.DENSITY
        assert ALGORITHMS["agglomerative"].category == AlgorithmCategory.HIERARCHICAL
        assert ALGORITHMS["embedding_cluster"].category == AlgorithmCategory.PIPELINE

    def test_capability_supports_predict(self):
        assert ALGORITHMS["kmeans"].supports_predict is True
        assert ALGORITHMS["minibatch_kmeans"].supports_predict is True
        assert ALGORITHMS["dbscan"].supports_predict is False
        assert ALGORITHMS["hdbscan"].supports_predict is False
        assert ALGORITHMS["agglomerative"].supports_predict is False

    def test_capability_supports_noise(self):
        assert ALGORITHMS["dbscan"].supports_noise is True
        assert ALGORITHMS["hdbscan"].supports_noise is True
        assert ALGORITHMS["kmeans"].supports_noise is False
        assert ALGORITHMS["embedding_cluster"].supports_noise is False

    def test_capability_supports_soft(self):
        assert ALGORITHMS["hdbscan"].supports_soft is True
        assert ALGORITHMS["embedding_cluster"].supports_soft is True
        assert ALGORITHMS["kmeans"].supports_soft is False

    def test_capability_supports_snapshot(self):
        assert ALGORITHMS["kmeans"].supports_snapshot is True
        assert ALGORITHMS["minibatch_kmeans"].supports_snapshot is True
        assert ALGORITHMS["embedding_cluster"].supports_snapshot is True
        assert ALGORITHMS["dbscan"].supports_snapshot is False
        assert ALGORITHMS["hdbscan"].supports_snapshot is False

    def test_capability_requires_k(self):
        assert ALGORITHMS["kmeans"].requires_k is True
        assert ALGORITHMS["embedding_cluster"].requires_k is True
        assert ALGORITHMS["dbscan"].requires_k is False
        assert ALGORITHMS["hdbscan"].requires_k is False

    def test_get_algorithm_valid(self):
        algo = get_algorithm("kmeans")
        assert algo is not None
        assert algo.id == "kmeans"
        assert algo.name == "KMeans"

    def test_get_algorithm_invalid(self):
        assert get_algorithm("nonexistent") is None

    def test_algorithm_relationships_count(self):
        assert len(ALGORITHM_RELATIONSHIPS) == 5

    def test_get_relationships_for_embedding_cluster(self):
        rels = get_relationships_for("embedding_cluster")
        assert len(rels) >= 2
        other_ids = {r.target if r.source == "embedding_cluster" else r.source for r in rels}
        assert "spherical_kmeans" in other_ids
        assert "kmeans" in other_ids

    def test_all_algorithms_have_metrics(self):
        for algo in ALGORITHMS.values():
            assert len(algo.metrics) >= 1

    def test_all_algorithms_have_descriptions(self):
        for algo in ALGORITHMS.values():
            assert algo.description
            assert algo.how_it_works
            assert algo.time_complexity
            assert algo.convergence_mechanism


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestParameters:
    def test_parameters_linked_to_valid_algorithms(self):
        for p in PARAMETERS:
            assert p.algorithm_id in ALGORITHMS, (
                f"Parameter '{p.name}' references unknown algorithm '{p.algorithm_id}'"
            )

    def test_get_parameters_for_kmeans(self):
        params = get_parameters_for("kmeans")
        names = {p.name for p in params}
        assert "n_clusters" in names
        assert "max_iter" in names
        assert "tol" in names
        assert "n_init" in names
        assert "algorithm" in names
        assert "metric" in names
        assert "random_state" in names
        assert len(params) == 7

    def test_get_parameters_for_embedding_cluster(self):
        params = get_parameters_for("embedding_cluster")
        names = {p.name for p in params}
        assert "n_clusters" in names
        assert "reduction_dim" in names
        assert "reduction" in names
        assert "n_init" in names

    def test_get_parameters_for_unknown(self):
        assert get_parameters_for("fake") == []

    def test_parameter_defaults_required_are_none(self):
        """n_clusters params that are required should have default=None."""
        for p in PARAMETERS:
            if p.name == "n_clusters" and p.algorithm_id in ("kmeans", "minibatch_kmeans"):
                assert p.default is None

    def test_parameter_valid_range_types(self):
        for p in PARAMETERS:
            if p.valid_range is not None:
                assert isinstance(p.valid_range, dict)
                assert any(k in p.valid_range for k in ("min", "max", "values", "min_exclusive"))

    def test_embedding_cluster_defaults(self):
        params = {p.name: p.default for p in get_parameters_for("embedding_cluster")}
        assert params["n_clusters"] == 50
        assert params["reduction_dim"] == 128
        assert params["reduction"] == "pca"
        assert params["n_init"] == 5
        assert params["tol"] == 1e-6


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


def _make_profile(**kwargs):
    defaults = dict(
        dimensionality=Dimensionality.MEDIUM,
        scale=DataScale.MEDIUM,
        normalization=NormalizationState.RAW,
        density=DensityProfile.UNIFORM,
        source_type=None,
        intrinsic_dim_estimate=None,
    )
    defaults.update(kwargs)
    return DataProfile(**defaults)


class TestDecisionRules:
    def test_highd_normalized_known_k(self):
        profile = _make_profile(
            dimensionality=Dimensionality.VERY_HIGH,
            normalization=NormalizationState.L2_NORMALIZED,
            source_type="openai",
        )
        rules = match_decision_rules(profile, requires_k=True)
        assert rules
        assert rules[0].recommended_algorithm == "embedding_cluster"

    def test_highd_normalized_unknown_k(self):
        profile = _make_profile(
            dimensionality=Dimensionality.HIGH,
            normalization=NormalizationState.L2_NORMALIZED,
        )
        rules = match_decision_rules(profile, requires_k=False)
        algos = [r.recommended_algorithm for r in rules]
        assert "hdbscan" in algos

    def test_low_dim_small_known_k(self):
        profile = _make_profile(
            dimensionality=Dimensionality.LOW,
            scale=DataScale.SMALL,
        )
        rules = match_decision_rules(profile, requires_k=True)
        algos = [r.recommended_algorithm for r in rules]
        assert "kmeans" in algos

    def test_large_scale_known_k(self):
        profile = _make_profile(scale=DataScale.VERY_LARGE)
        rules = match_decision_rules(profile, requires_k=True)
        algos = [r.recommended_algorithm for r in rules]
        assert "minibatch_kmeans" in algos

    def test_variable_density_unknown_k(self):
        profile = _make_profile(density=DensityProfile.VARIABLE)
        rules = match_decision_rules(profile, requires_k=False)
        assert rules
        assert rules[0].recommended_algorithm == "hdbscan"

    def test_uniform_density_low_dim_unknown_k(self):
        profile = _make_profile(
            density=DensityProfile.UNIFORM,
            dimensionality=Dimensionality.LOW,
        )
        rules = match_decision_rules(profile, requires_k=False)
        algos = [r.recommended_algorithm for r in rules]
        assert "dbscan" in algos

    def test_sorted_by_confidence(self):
        profile = _make_profile(dimensionality=Dimensionality.HIGH)
        rules = match_decision_rules(profile)
        for i in range(len(rules) - 1):
            assert rules[i].confidence >= rules[i + 1].confidence

    def test_no_matches_returns_empty(self):
        """A very specific profile may match nothing — that's valid."""
        profile = _make_profile(
            dimensionality=Dimensionality.LOW,
            scale=DataScale.VERY_LARGE,
            density=DensityProfile.HIERARCHICAL,
        )
        # May or may not match, but should not raise
        rules = match_decision_rules(profile)
        assert isinstance(rules, list)


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------


class TestAntiPatterns:
    def test_euclidean_highd_embeddings(self):
        profile = _make_profile(
            dimensionality=Dimensionality.HIGH,
            source_type="openai",
        )
        results = check_anti_patterns("kmeans", {"metric": "euclidean"}, profile)
        ids = {a.id for a in results}
        assert "euclidean_highd_embeddings" in ids

    def test_kmeans_no_pca_highd(self):
        profile = _make_profile(dimensionality=Dimensionality.VERY_HIGH)
        results = check_anti_patterns("kmeans", {}, profile)
        ids = {a.id for a in results}
        assert "kmeans_no_pca_highd" in ids

    def test_dbscan_highd(self):
        profile = _make_profile(dimensionality=Dimensionality.HIGH)
        results = check_anti_patterns("dbscan", {}, profile)
        ids = {a.id for a in results}
        assert "dbscan_highd" in ids

    def test_dbscan_variable_density(self):
        profile = _make_profile(density=DensityProfile.VARIABLE)
        results = check_anti_patterns("dbscan", {}, profile)
        ids = {a.id for a in results}
        assert "dbscan_variable_density" in ids

    def test_hdbscan_large_n(self):
        profile = _make_profile(scale=DataScale.VERY_LARGE)
        results = check_anti_patterns("hdbscan", {}, profile)
        ids = {a.id for a in results}
        assert "hdbscan_large_n" in ids

    def test_ward_non_euclidean(self):
        results = check_anti_patterns(
            "agglomerative", {"linkage": "ward", "metric": "cosine"}
        )
        ids = {a.id for a in results}
        assert "ward_non_euclidean" in ids

    def test_matryoshka_wrong_model(self):
        profile = _make_profile(source_type="sentence_transformers")
        results = check_anti_patterns(
            "embedding_cluster", {"reduction": "matryoshka"}, profile
        )
        ids = {a.id for a in results}
        assert "matryoshka_wrong_model" in ids

    def test_matryoshka_correct_model_no_trigger(self):
        profile = _make_profile(source_type="openai")
        results = check_anti_patterns(
            "embedding_cluster", {"reduction": "matryoshka"}, profile
        )
        ids = {a.id for a in results}
        assert "matryoshka_wrong_model" not in ids

    def test_no_profile(self):
        """Without profile, only param-based checks fire."""
        results = check_anti_patterns(
            "agglomerative", {"linkage": "ward", "metric": "cosine"}, None
        )
        ids = {a.id for a in results}
        assert "ward_non_euclidean" in ids

    def test_unknown_algorithm(self):
        results = check_anti_patterns("nonexistent", {})
        assert results == []


# ---------------------------------------------------------------------------
# Pathology diagnosis
# ---------------------------------------------------------------------------


class TestDiagnosis:
    def test_one_giant_cluster(self):
        results = diagnose_results(
            metric_scores={}, cluster_sizes=[800, 50, 50, 50, 50]
        )
        ids = {p.id for p in results}
        assert "one_giant_cluster" in ids

    def test_too_many_singletons(self):
        # >30% of clusters have <= 3 points
        sizes = [100, 2, 1, 1, 1, 1, 1]
        results = diagnose_results(metric_scores={}, cluster_sizes=sizes)
        ids = {p.id for p in results}
        assert "too_many_singletons" in ids

    def test_all_noise(self):
        results = diagnose_results(
            metric_scores={}, cluster_sizes=[],
            n_noise=900, n_total=1000,
        )
        ids = {p.id for p in results}
        assert "all_noise" in ids

    def test_slow_convergence(self):
        results = diagnose_results(
            metric_scores={}, cluster_sizes=[100, 100],
            n_iter=300, max_iter=300,
        )
        ids = {p.id for p in results}
        assert "slow_convergence" in ids

    def test_uneven_cluster_sizes(self):
        results = diagnose_results(
            metric_scores={}, cluster_sizes=[1000, 500, 10]
        )
        ids = {p.id for p in results}
        assert "uneven_cluster_sizes" in ids

    def test_low_silhouette_unstable(self):
        results = diagnose_results(
            metric_scores={"silhouette_score": 0.05},
            cluster_sizes=[100, 100, 100],
        )
        ids = {p.id for p in results}
        assert "unstable_across_runs" in ids

    def test_no_pathology(self):
        results = diagnose_results(
            metric_scores={"silhouette_score": 0.5},
            cluster_sizes=[100, 100, 100],
        )
        assert results == []

    def test_empty_inputs(self):
        results = diagnose_results(metric_scores={}, cluster_sizes=[])
        assert results == []


# ---------------------------------------------------------------------------
# Metric interpretations
# ---------------------------------------------------------------------------


class TestMetricInterpretations:
    def test_count(self):
        assert len(METRIC_INTERPRETATIONS) == 3

    def test_directions(self):
        by_name = {m.metric_name: m for m in METRIC_INTERPRETATIONS}
        assert by_name["silhouette_score"].direction == "higher_better"
        assert by_name["calinski_harabasz_score"].direction == "higher_better"
        assert by_name["davies_bouldin_score"].direction == "lower_better"

    def test_all_have_thresholds(self):
        for m in METRIC_INTERPRETATIONS:
            assert len(m.thresholds) >= 2

    def test_all_have_caveats(self):
        for m in METRIC_INTERPRETATIONS:
            assert len(m.caveats) >= 1


# ---------------------------------------------------------------------------
# Anti-patterns & pathologies data completeness
# ---------------------------------------------------------------------------


class TestDataCompleteness:
    def test_anti_patterns_count(self):
        assert len(ANTI_PATTERNS) == 8

    def test_pathology_signatures_count(self):
        assert len(PATHOLOGY_SIGNATURES) == 6

    def test_all_anti_patterns_have_fix(self):
        for ap in ANTI_PATTERNS:
            assert ap.fix
            assert ap.severity in ("error", "warning", "info")

    def test_all_pathologies_have_fixes(self):
        for p in PATHOLOGY_SIGNATURES:
            assert len(p.fixes) >= 1
            assert len(p.symptoms) >= 1
            assert len(p.likely_causes) >= 1
