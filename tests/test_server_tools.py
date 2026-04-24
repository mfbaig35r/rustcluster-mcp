"""Tests for MCP server tools — knowledge tools (direct) + advisor tools (mocked + integration)."""

import pytest

from rustcluster_mcp.server import (
    analyze_data,
    check_config,
    compare_configs,
    diagnose,
    evaluate_clusters,
    explain_algorithm,
    explain_parameter,
    list_algorithms,
    optimize_k,
    recommend_algorithm,
    suggest_config,
)

# ===========================================================================
# Knowledge tools — pure lookups, no mocks needed
# ===========================================================================


class TestExplainAlgorithm:
    async def test_valid(self):
        result = await explain_algorithm("kmeans")
        assert result["status"] == "success"
        assert result["id"] == "kmeans"
        assert result["name"] == "KMeans"
        assert "parameters" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "capabilities" in result

    async def test_embedding_cluster(self):
        result = await explain_algorithm("embedding_cluster")
        assert result["status"] == "success"
        assert result["category"] == "pipeline"
        assert len(result["relationships"]) >= 2

    async def test_invalid(self):
        result = await explain_algorithm("nonexistent")
        assert result["status"] == "error"
        assert "nonexistent" in result["error"]
        assert "kmeans" in result["error"]  # suggests valid algorithms


class TestExplainParameter:
    async def test_valid(self):
        result = await explain_parameter("kmeans", "n_clusters")
        assert result["status"] == "success"
        assert result["parameter"] == "n_clusters"
        assert result["sensitivity"] == "high"
        assert "tips" in result

    async def test_with_interactions(self):
        result = await explain_parameter("kmeans", "n_init")
        assert result["status"] == "success"
        assert len(result["interactions"]) >= 1

    async def test_invalid_algorithm(self):
        result = await explain_parameter("fake", "n_clusters")
        assert result["status"] == "error"

    async def test_invalid_parameter(self):
        result = await explain_parameter("kmeans", "fake_param")
        assert result["status"] == "error"
        assert "n_clusters" in result["error"]  # suggests valid params


class TestListAlgorithms:
    async def test_no_filter(self):
        result = await list_algorithms()
        assert result["status"] == "success"
        assert result["count"] == 7

    async def test_filter_density(self):
        result = await list_algorithms(category="density-based")
        assert result["status"] == "success"
        assert result["count"] == 2
        ids = {a["id"] for a in result["algorithms"]}
        assert ids == {"dbscan", "hdbscan"}

    async def test_filter_supports_noise(self):
        result = await list_algorithms(supports_noise=True)
        assert result["count"] == 2
        ids = {a["id"] for a in result["algorithms"]}
        assert ids == {"dbscan", "hdbscan"}

    async def test_filter_supports_predict(self):
        result = await list_algorithms(supports_predict=True)
        ids = {a["id"] for a in result["algorithms"]}
        assert ids == {"kmeans", "minibatch_kmeans"}

    async def test_filter_supports_snapshot(self):
        result = await list_algorithms(supports_snapshot=True)
        ids = {a["id"] for a in result["algorithms"]}
        assert ids == {"kmeans", "minibatch_kmeans", "embedding_cluster"}

    async def test_filter_requires_k_false(self):
        result = await list_algorithms(requires_k=False)
        ids = {a["id"] for a in result["algorithms"]}
        assert "dbscan" in ids
        assert "hdbscan" in ids
        assert "kmeans" not in ids

    async def test_multiple_filters(self):
        result = await list_algorithms(supports_noise=True, requires_k=False)
        ids = {a["id"] for a in result["algorithms"]}
        assert ids == {"dbscan", "hdbscan"}

    async def test_filter_no_results(self):
        """No algorithm supports both predict and noise."""
        result = await list_algorithms(supports_predict=True, supports_noise=True)
        assert result["count"] == 0


class TestCheckConfig:
    async def test_valid_config(self):
        result = await check_config("kmeans", {"n_clusters": 10, "metric": "euclidean"})
        assert result["status"] == "success"
        assert result["valid"] is True

    async def test_invalid_algorithm(self):
        result = await check_config("nonexistent", {"n_clusters": 10})
        assert result["status"] == "error"

    async def test_unknown_param(self):
        result = await check_config("kmeans", {"fake_param": 5})
        assert result["status"] == "success"
        assert len(result["parameter_issues"]) > 0
        assert "fake_param" in result["parameter_issues"][0]

    async def test_out_of_range(self):
        result = await check_config("hdbscan", {"min_cluster_size": 1})
        assert result["status"] == "success"
        assert len(result["parameter_issues"]) > 0

    async def test_invalid_value(self):
        result = await check_config("kmeans", {"algorithm": "invalid_algo"})
        assert result["status"] == "success"
        assert len(result["parameter_issues"]) > 0

    async def test_ward_cosine_antipattern(self):
        result = await check_config(
            "agglomerative",
            {"linkage": "ward", "metric": "cosine"},
        )
        assert result["status"] == "success"
        assert result["valid"] is False
        assert any(w["id"] == "ward_non_euclidean" for w in result["warnings"])

    async def test_with_n_features_highd(self):
        result = await check_config(
            "kmeans", {"metric": "euclidean"},
            n_features=1536, source_type="openai",
        )
        assert result["status"] == "success"
        assert len(result["warnings"]) > 0


# ===========================================================================
# Advisor tools — analyze_data
# ===========================================================================


class TestAnalyzeData:
    async def test_basic(self, tmp_data_2d):
        result = await analyze_data(tmp_data_2d)
        assert result["status"] == "success"
        assert result["n_samples"] == 100
        assert result["n_features"] == 50
        assert result["dimensionality"] == "medium"
        assert result["has_nan"] is False
        assert result["has_inf"] is False

    async def test_l2_normalized(self, tmp_data_l2_normalized):
        result = await analyze_data(tmp_data_l2_normalized)
        assert result["status"] == "success"
        assert result["is_l2_normalized"] is True
        assert result["normalization"] == "l2_normalized"

    async def test_initial_recommendation(self, tmp_data_2d):
        result = await analyze_data(tmp_data_2d)
        assert result["status"] == "success"
        assert "initial_recommendation" in result
        rec = result["initial_recommendation"]
        assert "algorithm" in rec
        assert "confidence" in rec
        assert "rationale" in rec

    async def test_1d_error(self, tmp_data_1d):
        result = await analyze_data(tmp_data_1d)
        assert result["status"] == "error"
        assert "2D" in result["error"]

    async def test_file_not_found(self):
        result = await analyze_data("/nonexistent/path.npy")
        assert result["status"] == "error"
        assert result["error_type"] == "file_error"

    async def test_nan_warning(self, tmp_data_with_nan):
        result = await analyze_data(tmp_data_with_nan)
        assert result["status"] == "success"
        assert result["has_nan"] is True
        assert any("NaN" in w for w in result.get("warnings", []))

    async def test_npz_loading(self, tmp_data_npz):
        result = await analyze_data(tmp_data_npz)
        assert result["status"] == "success"
        assert result["n_samples"] == 100
        assert result["n_features"] == 50

    async def test_with_source_type(self, tmp_data_2d):
        result = await analyze_data(tmp_data_2d, source_type="openai")
        assert result["status"] == "success"
        assert result["source_type"] == "openai"


# ===========================================================================
# Advisor tools — recommend_algorithm
# ===========================================================================


class TestRecommendAlgorithm:
    async def test_highd_embeddings_known_k(self):
        result = await recommend_algorithm(
            n_samples=50000, n_features=1536,
            normalization="l2_normalized", source_type="openai",
            know_k=True,
        )
        assert result["status"] == "success"
        top = result["recommendations"][0]
        assert top["algorithm"] == "embedding_cluster"
        assert not top["disqualified"]

    async def test_lowdim_small_known_k(self):
        result = await recommend_algorithm(
            n_samples=500, n_features=10,
            know_k=True,
        )
        assert result["status"] == "success"
        algos = [r["algorithm"] for r in result["recommendations"] if not r["disqualified"]]
        assert "kmeans" in algos

    async def test_need_noise_disqualifies(self):
        result = await recommend_algorithm(
            n_samples=1000, n_features=10,
            need_noise_detection=True,
        )
        assert result["status"] == "success"
        for rec in result["recommendations"]:
            if rec["algorithm"] in ("kmeans", "minibatch_kmeans", "agglomerative"):
                assert rec["disqualified"]

    async def test_need_predict(self):
        result = await recommend_algorithm(
            n_samples=1000, n_features=10,
            need_predict=True,
        )
        viable = [r for r in result["recommendations"] if not r["disqualified"]]
        viable_algos = {r["algorithm"] for r in viable}
        # Only kmeans and minibatch_kmeans support predict
        assert viable_algos <= {"kmeans", "minibatch_kmeans"}

    async def test_need_snapshot(self):
        result = await recommend_algorithm(
            n_samples=1000, n_features=10,
            need_snapshot=True,
        )
        viable = [r for r in result["recommendations"] if not r["disqualified"]]
        viable_algos = {r["algorithm"] for r in viable}
        assert viable_algos <= {"kmeans", "minibatch_kmeans", "embedding_cluster"}

    async def test_memory_budget_filters_hdbscan(self):
        result = await recommend_algorithm(
            n_samples=100000, n_features=10,
            max_memory_gb=1.0,
        )
        for rec in result["recommendations"]:
            if rec["algorithm"] in ("hdbscan", "agglomerative"):
                assert rec["disqualified"]
                assert any("memory" in r.lower() for r in rec["disqualification_reasons"])

    async def test_returns_data_profile(self):
        result = await recommend_algorithm(
            n_samples=5000, n_features=768,
            normalization="l2_normalized",
        )
        assert "data_profile" in result
        assert result["data_profile"]["dimensionality"] == "high"

    async def test_max_five_recommendations(self):
        result = await recommend_algorithm(
            n_samples=1000, n_features=50,
        )
        assert len(result["recommendations"]) <= 5


# ===========================================================================
# Advisor tools — suggest_config
# ===========================================================================


class TestSuggestConfig:
    async def test_kmeans_basic(self):
        result = await suggest_config("kmeans", n_samples=1000, n_features=50)
        assert result["status"] == "success"
        assert result["algorithm"] == "kmeans"
        assert "n_clusters" in result["config"]
        assert "metric" in result["config"]
        assert "n_init" in result["config"]

    async def test_embedding_cluster_reduction_dim(self):
        result = await suggest_config(
            "embedding_cluster", n_samples=50000, n_features=1536,
        )
        config = result["config"]
        assert config["reduction_dim"]["suggested_value"] == 128

    async def test_embedding_cluster_low_dim_no_reduction(self):
        result = await suggest_config(
            "embedding_cluster", n_samples=1000, n_features=128,
        )
        config = result["config"]
        assert config["reduction_dim"]["suggested_value"] is None

    async def test_with_target_k(self):
        result = await suggest_config(
            "kmeans", n_samples=1000, n_features=50, target_k=15,
        )
        config = result["config"]
        assert config["n_clusters"]["suggested_value"] == 15

    async def test_heuristic_k_without_target(self):
        result = await suggest_config(
            "kmeans", n_samples=1000, n_features=50,
        )
        config = result["config"]
        # sqrt(1000/2) ~ 22
        suggested_k = config["n_clusters"]["suggested_value"]
        assert 10 <= suggested_k <= 30

    async def test_usage_example(self):
        result = await suggest_config("kmeans", n_samples=1000, n_features=50)
        assert "usage_example" in result
        assert "KMeans" in result["usage_example"]
        assert "fit" in result["usage_example"]

    async def test_invalid_algorithm(self):
        result = await suggest_config("nonexistent", n_samples=100, n_features=10)
        assert result["status"] == "error"

    async def test_cosine_for_embeddings(self):
        result = await suggest_config(
            "kmeans", n_samples=1000, n_features=768,
            source_type="openai",
        )
        config = result["config"]
        assert config["metric"]["suggested_value"] == "cosine"

    async def test_matryoshka_for_openai(self):
        result = await suggest_config(
            "embedding_cluster", n_samples=1000, n_features=1536,
            source_type="openai",
        )
        config = result["config"]
        assert config["reduction"]["suggested_value"] == "matryoshka"

    async def test_anti_pattern_warnings(self):
        result = await suggest_config(
            "kmeans", n_samples=1000, n_features=1536,
            source_type="openai",
        )
        assert len(result["warnings"]) > 0


# ===========================================================================
# Advisor tools — diagnose
# ===========================================================================


class TestDiagnose:
    async def test_keyword_one_giant(self):
        result = await diagnose(description="I have one big cluster with most points")
        assert result["status"] == "success"
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "one_giant_cluster" in ids

    async def test_keyword_all_noise(self):
        result = await diagnose(description="everything is noise, no clusters found")
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "all_noise" in ids

    async def test_keyword_slow(self):
        result = await diagnose(description="the algorithm is not converging")
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "slow_convergence" in ids

    async def test_keyword_unstable(self):
        result = await diagnose(description="getting different results each run")
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "unstable_across_runs" in ids

    async def test_from_metrics(self):
        result = await diagnose(
            description="clusters don't look right",
            metrics={"silhouette": 0.05},
            cluster_sizes=[800, 50, 50, 50, 50],
        )
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "one_giant_cluster" in ids

    async def test_from_convergence(self):
        result = await diagnose(
            description="slow",
            n_iter=300, max_iter=300,
            cluster_sizes=[100, 100],
        )
        ids = {p["id"] for p in result["pathologies_detected"]}
        assert "slow_convergence" in ids

    async def test_no_match(self):
        result = await diagnose(description="everything looks great and perfect")
        assert result["status"] == "success"
        assert len(result["pathologies_detected"]) == 0

    async def test_general_advice_present(self):
        result = await diagnose(description="something is wrong")
        assert "general_advice" in result


# ===========================================================================
# Advisor tools — data-dependent (integration, @pytest.mark.slow)
# ===========================================================================


class TestOptimizeK:
    async def test_density_algorithm_rejected(self, tmp_data_2d):
        result = await optimize_k(tmp_data_2d, algorithm="dbscan")
        assert result["status"] == "error"
        assert "density-based" in result["error"]

    async def test_1d_rejected(self, tmp_data_1d):
        result = await optimize_k(tmp_data_1d)
        assert result["status"] == "error"

    async def test_file_not_found(self):
        result = await optimize_k("/nonexistent/path.npy")
        assert result["status"] == "error"
        assert result["error_type"] == "file_error"

    @pytest.mark.slow
    async def test_integration(self, tmp_data_2d):
        result = await optimize_k(
            tmp_data_2d, algorithm="kmeans",
            k_range=[2, 3, 5, 8],
        )
        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert result["recommended_k"] is not None
        assert result["recommended_k"] in [2, 3, 5, 8]
        # Check result structure
        for r in result["results"]:
            if "error" not in r:
                assert "silhouette" in r
                assert "calinski_harabasz" in r
                assert "davies_bouldin" in r


class TestCompareConfigs:
    async def test_too_few_configs(self, tmp_data_2d):
        result = await compare_configs(tmp_data_2d, configs=[{"algorithm": "kmeans"}])
        assert result["status"] == "error"
        assert "at least 2" in result["error"]

    async def test_file_not_found(self):
        result = await compare_configs("/nonexistent.npy", configs=[{}, {}])
        assert result["status"] == "error"

    async def test_1d_rejected(self, tmp_data_1d):
        result = await compare_configs(
            tmp_data_1d,
            configs=[
                {"algorithm": "kmeans", "params": {"n_clusters": 3}},
                {"algorithm": "kmeans", "params": {"n_clusters": 5}},
            ],
        )
        assert result["status"] == "error"

    @pytest.mark.slow
    async def test_integration(self, tmp_data_2d):
        result = await compare_configs(
            tmp_data_2d,
            configs=[
                {"algorithm": "kmeans", "params": {"n_clusters": 3}, "label": "k3"},
                {"algorithm": "kmeans", "params": {"n_clusters": 5}, "label": "k5"},
            ],
        )
        assert result["status"] == "success"
        assert len(result["comparison"]) == 2
        assert result["winner"] in ("k3", "k5")


class TestEvaluateClusters:
    async def test_file_not_found(self):
        result = await evaluate_clusters("/nonexistent.npy", "/nonexistent_labels.npy")
        assert result["status"] == "error"

    @pytest.mark.slow
    async def test_integration(self, tmp_data_2d, tmp_labels):
        result = await evaluate_clusters(tmp_data_2d, tmp_labels)
        assert result["status"] == "success"
        assert "metrics" in result
        assert "silhouette" in result["metrics"]
        assert "calinski_harabasz" in result["metrics"]
        assert "davies_bouldin" in result["metrics"]
        assert result["cluster_sizes"]["n_clusters"] == 3
        assert "interpretations" in result
