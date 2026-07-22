import pytest

from llm_cost_router import settings


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point every test at a throwaway SQLite file instead of data/router.db."""
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "test_router.db")
