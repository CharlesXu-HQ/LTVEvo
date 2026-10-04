from unittest.mock import patch

from ltvevo.agent import reflect_experiment
from ltvevo.provider import ApiProvider


def test_reflection_retries_once_when_provider_omits_required_field():
    provider = ApiProvider("https://example.com", "model", "secret")
    invalid = {"verdict": "inconclusive", "evidence": "metric change is small", "lesson": "compare again"}
    valid = {**invalid, "next_direction": "try a calibrated model"}
    with patch("ltvevo.agent._call", side_effect=[invalid, valid]) as request:
        result = reflect_experiment(provider, {"status": "evaluated"})
    assert result == valid
    assert request.call_count == 2
