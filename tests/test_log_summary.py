from ops.summarize_logs import summarize


def test_summary_ignores_non_request_logs_and_handles_bad_lines():
    result = summarize(['{"event":"http_request","status":200,"duration_seconds":0.2}', '{"event":"http_request","status":500,"duration_seconds":0.4}', 'bad', '{"event":"other"}', '{"event":"http_request","status":200,"duration_seconds":-1}'])
    assert result["requests"] == 2
    assert result["errors"] == 1
    assert abs(result["mean_latency_seconds"] - 0.3) < 0.00001
    assert result["malformed_lines"] == 2


def test_empty_summary():
    assert summarize([])["mean_latency_seconds"] == 0
