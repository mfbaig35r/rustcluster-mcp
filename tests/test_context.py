"""Tests for context injection code — string validation, no execution."""

from rustcluster_mcp.context import CLUSTER_CONTEXT_CODE, CLUSTER_REQUIRED_PACKAGES


class TestContextCodeContent:
    def test_is_string(self):
        assert isinstance(CLUSTER_CONTEXT_CODE, str)
        assert len(CLUSTER_CONTEXT_CODE) > 100

    def test_defines_cluster_helper(self):
        assert "__cluster__ = _ClusterContext()" in CLUSTER_CONTEXT_CODE

    def test_defines_class(self):
        assert "class _ClusterContext:" in CLUSTER_CONTEXT_CODE

    def test_has_fit_method(self):
        assert "def fit(" in CLUSTER_CONTEXT_CODE

    def test_has_sweep_k_method(self):
        assert "def sweep_k(" in CLUSTER_CONTEXT_CODE

    def test_has_plot_sweep_method(self):
        assert "def plot_sweep(" in CLUSTER_CONTEXT_CODE

    def test_has_snapshot_method(self):
        assert "def snapshot(" in CLUSTER_CONTEXT_CODE

    def test_has_visualize_snapshot(self):
        assert "def visualize_snapshot(" in CLUSTER_CONTEXT_CODE

    def test_has_visualize_confidence(self):
        assert "def visualize_confidence(" in CLUSTER_CONTEXT_CODE

    def test_has_visualize_drift(self):
        assert "def visualize_drift(" in CLUSTER_CONTEXT_CODE

    def test_has_visualize_hierarchical(self):
        assert "def visualize_hierarchical(" in CLUSTER_CONTEXT_CODE

    def test_imports_rustcluster(self):
        assert "import rustcluster" in CLUSTER_CONTEXT_CODE

    def test_has_all_user_facing_algorithms(self):
        for algo in ["kmeans", "minibatch_kmeans", "dbscan", "hdbscan",
                     "agglomerative", "embedding_cluster"]:
            assert f'"{algo}"' in CLUSTER_CONTEXT_CODE

    def test_has_load_method(self):
        assert "def load(" in CLUSTER_CONTEXT_CODE

    def test_has_profile_method(self):
        assert "def profile(" in CLUSTER_CONTEXT_CODE

    def test_syntactically_valid(self):
        """Context code must compile without SyntaxError."""
        compile(CLUSTER_CONTEXT_CODE, "<context>", "exec")


class TestRequiredPackages:
    def test_is_list(self):
        assert isinstance(CLUSTER_REQUIRED_PACKAGES, list)

    def test_contains_rustcluster(self):
        assert "rustcluster" in CLUSTER_REQUIRED_PACKAGES

    def test_contains_numpy(self):
        assert "numpy" in CLUSTER_REQUIRED_PACKAGES

    def test_contains_matplotlib(self):
        assert "matplotlib" in CLUSTER_REQUIRED_PACKAGES

    def test_contains_marimo(self):
        assert "marimo" in CLUSTER_REQUIRED_PACKAGES
