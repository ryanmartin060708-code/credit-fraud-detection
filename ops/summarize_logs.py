"""Summarize JSON request logs from stdin without forwarding raw data anywhere."""
import json
import sys


def summarize(lines):
    count = errors = malformed = 0
    total = 0.0
    for line in lines:
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("event") != "http_request":
                continue
            status = int(event["status"])
            duration = float(event["duration_seconds"])
            if not 100 <= status <= 599 or not 0 <= duration < float("inf"):
                raise ValueError("Invalid request record")
            count += 1
            errors += status >= 400
            total += duration
        except (ValueError, TypeError, KeyError):
            malformed += 1
    return {"requests": count, "errors": errors, "mean_latency_seconds": total / count if count else 0, "malformed_lines": malformed}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.stdin)))
