"""Test configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def test_database_url():
    """Database URL for testing."""
    return "postgresql://postgres:postgres@localhost:5432/pds_stress_test_test"
