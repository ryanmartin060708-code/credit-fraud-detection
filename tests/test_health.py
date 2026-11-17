from unittest.mock import MagicMock
from urllib.error import URLError
import pytest
from ops import health_service


def test_health_checks_streamlit(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.status = 200
    probe = MagicMock(return_value=response)
    monkeypatch.setattr(health_service, "urlopen", probe)
    result = health_service.create_app().test_client().get("/health")
    assert result.status_code == 200
    probe.assert_called_once_with("http://127.0.0.1:8501/_stcore/health", timeout=2)


@pytest.mark.parametrize("error", [URLError("down"), TimeoutError(), OSError()])
def test_unhealthy_dependency_returns_503(monkeypatch, error):
    monkeypatch.setattr(health_service, "urlopen", MagicMock(side_effect=error))
    client = health_service.create_app().test_client()
    assert client.get("/health").status_code == 503
    assert 'http_errors_total{route="/health"} 1.0' in client.get("/metrics").text
