from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""

    class MockChoice:
        def __init__(self, content="# Test Markdown", finish_reason="stop"):
            self.message = type("Message", (), {"content": content})()
            self.finish_reason = finish_reason

    class MockResponse:
        def __init__(self, content="# Test Markdown", finish_reason="stop"):
            self.choices = [MockChoice(content, finish_reason)]

    return MockResponse


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow E2E tests")
