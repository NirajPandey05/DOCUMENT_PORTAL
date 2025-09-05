import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text

def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"].lower() == "ok"

def test_analyze_route():
    # Test with no file (should fail)
    response = client.post("/analyze")
    assert response.status_code == 422  # Unprocessable Entity (missing file)
    # Optionally, add a valid file test if test files are available

def test_compare_route():
    # Test with no files (should fail)
    response = client.post("/compare")
    assert response.status_code == 422  # Unprocessable Entity (missing files)
    # Optionally, add a valid file test if test files are available

def test_chat_index_route():
    # Test with no files (should fail)
    response = client.post("/chat/index")
    assert response.status_code in (400, 422)  # Expecting error due to missing files
    # Optionally, add a valid file test if test files are available

def test_chat_query_route():
    # Test with no question (should fail)
    response = client.post("/chat/query")
    assert response.status_code in (400, 422)  # Expecting error due to missing question
    # Optionally, add a valid question test if test index and data are available
