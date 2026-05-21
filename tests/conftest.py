"""
Pytest configuration for the multi-agent customer support system.

Provides shared fixtures, pytest plugins, and test environment setup.
"""

import pytest
import os
import sys
import tempfile

# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Auto-use fixture to set up test environment variables
    so tests don't depend on real API keys.
    """
    # Set dummy values for required env vars
    env_vars = {
        "OPENAI_API_KEY": "sk-test-dummy-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "test-qdrant-key",
    }
    for key, value in env_vars.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)


@pytest.fixture
def temp_file():
    """Create a temporary file that auto-cleans up."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory that auto-cleans up."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "kg: marks tests for the knowledge graph module")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless --integration flag is used."""
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (requires running services)",
    )
