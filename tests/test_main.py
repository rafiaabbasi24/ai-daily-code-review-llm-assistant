"""tests/test_main.py
Unit tests for the FastAPI endpoints of the LLM-Powered Code Review Assistant.

The tests cover:
- Health check endpoint (GET /health)
- Code review endpoint (POST /review) with valid payload
- Validation error when required fields are missing
- Basic sanity checks on the response structure
"""

import json
from typing import Any, Dict

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Import the FastAPI application instance.
# Adjust the import path according to the actual project layout.
# It is assumed that the main FastAPI app is defined in `app/main.py` as `app`.
from app.main import app  # type: ignore

# Create a TestClient instance that will be reused across tests.
client = TestClient(app)


def _load_fixture(name: str) -> Dict[str, Any]:
    """Utility to load JSON fixtures from the `tests/fixtures` directory.

    Args:
        name: The filename (without extension) of the fixture.

    Returns:
        The parsed JSON content as a dictionary.
    """
    path = f"tests/fixtures/{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_health_endpoint() -> None:
    """The health endpoint should return HTTP 200 and a simple JSON payload."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK, "Health check should be reachable"
    data = response.json()
    assert isinstance(data, dict), "Response should be a JSON object"
    # The exact shape may vary; we assert the presence of a `status` key.
    assert data.get("status") == "ok", "Health status should be 'ok'"


def test_review_endpoint_success() -> None:
    """A valid code payload should return a review with expected fields."""
    # Example payload – replace with realistic code snippet if needed.
    payload = {
        "code": "def add(a, b):\n    return a + b",
        "language": "python"
    }
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_200_OK, "Valid review request should succeed"

    data = response.json()
    assert isinstance(data, dict), "Response must be a JSON object"
    # Expected keys in a successful review response.
    expected_keys = {"review", "suggestions", "metadata"}
    missing = expected_keys - data.keys()
    assert not missing, f"Missing keys in response: {missing}"

    # Basic sanity checks on the content.
    assert isinstance(data["review"], str) and data["review"], "Review text should be non‑empty"
    assert isinstance(data["suggestions"], list), "Suggestions should be a list"
    assert isinstance(data["metadata"], dict), "Metadata should be a dictionary"


def test_review_endpoint_missing_code() -> None:
    """Submitting a payload without the required `code` field should raise a validation error."""
    payload = {"language": "python"}  # `code` is omitted intentionally.
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, (
        "Missing required fields must trigger a 422 response"
    )
    data = response.json()
    # FastAPI returns a list of errors under the "detail" key.
    assert "detail" in data, "Error response must contain a 'detail' field"
    errors = data["detail"]
    assert isinstance(errors, list) and any(
        err.get("loc") == ["body", "code"] for err in errors
    ), "Error details should indicate the missing 'code' field"


def test_review_endpoint_empty_code() -> None:
    """An empty string for `code` should be rejected with a 422 error."""
    payload = {"code": "", "language": "python"}
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, (
        "Empty code should not be accepted"
    )
    data = response.json()
    assert "detail" in data, "Error response must contain a 'detail' field"


@pytest.mark.parametrize(
    "language,code_snippet",
    [
        ("python", "def foo():\n    pass"),
        ("javascript", "function foo() { return; }"),
        ("java", "public class Foo {}"),
    ],
)
def test_review_endpoint_multiple_languages(language: str, code_snippet: str) -> None:
    """The review endpoint should handle different supported languages."""
    payload = {"code": code_snippet, "language": language}
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_200_OK, f"Review failed for language {language}"
    data = response.json()
    assert data.get("metadata", {}).get("language") == language, "Returned metadata must echo the language"


def test_review_endpoint_invalid_language() -> None:
    """Submitting an unsupported language should result in a 400 Bad Request."""
    payload = {"code": "print('hello')", "language": "brainfuck"}
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST, "Unsupported language should be rejected"
    data = response.json()
    assert "detail" in data, "Error response must contain a 'detail' field"
    assert "unsupported language" in data["detail"].lower(), "Error message should mention unsupported language"
