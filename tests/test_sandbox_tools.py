"""Tests for sandbox tool delegation layer — mocks marimo-sandbox internals."""

from unittest.mock import patch

from rustcluster_mcp.context import CLUSTER_CONTEXT_CODE, CLUSTER_REQUIRED_PACKAGES

# ===========================================================================
# Context injection
# ===========================================================================


class TestContextInjection:
    def test_run_python_prepends_context_code(self):
        """Verify context code is prepended to user code."""
        from rustcluster_mcp.server import _impl_cluster_run_python

        user_code = "print('hello')"
        fake_result = {"status": "success", "run_id": "123"}

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_run_python", create=True, return_value=fake_result) as mock_run, \
             patch("rustcluster_mcp.server._inject_pep723_header", create=True):
            _impl_cluster_run_python(code=user_code)

            # Check that the code passed to _impl_run_python starts with context
            call_kwargs = mock_run.call_args
            injected_code = call_kwargs.kwargs.get("code") or call_kwargs[1].get("code")
            if injected_code is None:
                injected_code = call_kwargs[0][0]
            assert injected_code.startswith(CLUSTER_CONTEXT_CODE)
            assert user_code in injected_code

    def test_run_python_merges_packages(self):
        """Extra packages are merged with required packages."""
        from rustcluster_mcp.server import _impl_cluster_run_python

        fake_result = {"status": "success", "run_id": "123"}

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_run_python", create=True, return_value=fake_result) as mock_run, \
             patch("rustcluster_mcp.server._inject_pep723_header", create=True):
            _impl_cluster_run_python(code="x", packages=["pandas"])

            call_kwargs = mock_run.call_args
            packages = call_kwargs.kwargs.get("packages") or call_kwargs[1].get("packages")
            if packages is None:
                # positional
                packages = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else []

            assert "pandas" in packages
            for pkg in CLUSTER_REQUIRED_PACKAGES:
                assert pkg in packages

    def test_run_python_no_duplicate_packages(self):
        """numpy already in required list — should not appear twice."""
        from rustcluster_mcp.server import _impl_cluster_run_python

        fake_result = {"status": "success", "run_id": "123"}

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_run_python", create=True, return_value=fake_result) as mock_run, \
             patch("rustcluster_mcp.server._inject_pep723_header", create=True):
            _impl_cluster_run_python(code="x", packages=["numpy"])

            call_kwargs = mock_run.call_args
            packages = call_kwargs.kwargs.get("packages") or call_kwargs[1].get("packages")
            if packages is None:
                packages = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else []

            assert packages.count("numpy") == 1


# ===========================================================================
# Graceful degradation
# ===========================================================================


class TestGracefulDegradation:
    async def test_run_python_no_sandbox(self):
        from rustcluster_mcp.server import run_python

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", False):
            result = await run_python("print('hi')")
            assert result["status"] == "error"
            assert "marimo-sandbox" in result["error"]

    async def test_get_run_no_sandbox(self):
        from rustcluster_mcp.server import get_run

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", False):
            result = await get_run("abc")
            assert result["status"] == "error"
            assert "marimo-sandbox" in result["error"]

    async def test_list_runs_no_sandbox(self):
        from rustcluster_mcp.server import list_runs

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", False):
            result = await list_runs()
            assert result["status"] == "error"

    async def test_check_setup_no_sandbox(self):
        from rustcluster_mcp.server import check_setup

        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", False):
            result = await check_setup()
            assert result["status"] == "error"


# ===========================================================================
# Delegation passthrough
# ===========================================================================


class TestDelegation:
    async def test_get_run_delegates(self):
        from rustcluster_mcp.server import get_run

        fake_result = {"status": "success", "run_id": "abc", "code": "x"}
        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_get_run", create=True, return_value=fake_result) as mock:
            result = await get_run("abc")
            mock.assert_called_once_with("abc")
            assert result == fake_result

    async def test_list_runs_delegates(self):
        from rustcluster_mcp.server import list_runs

        fake_result = {"status": "success", "runs": []}
        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_list_runs", create=True, return_value=fake_result) as mock:
            result = await list_runs(limit=5, status="completed")
            mock.assert_called_once_with(limit=5, status="completed")
            assert result == fake_result

    async def test_check_setup_adds_rustcluster_info(self):
        from rustcluster_mcp.server import check_setup

        fake_result = {"sandbox": "ok"}
        with patch("rustcluster_mcp.server._SANDBOX_AVAILABLE", True), \
             patch("rustcluster_mcp.server._impl_check_setup", create=True, return_value=fake_result):
            result = await check_setup()
            assert "rustcluster" in result
            assert result["rustcluster"]["installed"] is True
            assert "algorithms" in result["rustcluster"]
