import pytest

pytest.importorskip("jinja2")
from fastapi.testclient import TestClient
from review_studio.review_studio import app, registry

client = TestClient(app)

def test_studio_page_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Azozo Review Studio" in response.text

def test_list_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "counts" in data
    assert data["total"] > 0
    assert len(data["documents"]) > 0

def test_audit_endpoint():
    valid_xml = "<section># EXAM</section><question_label>1.</question_label><stem>Stem</stem><option_label>A.</option_label><option_text>Opt</option_text>"
    response = client.post(
        "/api/documents/exam_test/audit",
        json={"xml_content": valid_xml},
    )
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert data["overall_score"] > 80.0
    assert data["decision"] in ["PASS", "NEEDS_REVISION"]

def test_auto_fix_endpoint():
    broken_xml = "<section># EXAM</section><stimulus>Dựa vào đoạn thông tin sau để trả lời: Nội dung đoạn văn ở đây.</stimulus><question_label>1.</question_label><stem>Stem</stem><option_label>A.</option_label><option_text>Opt</option_text>"
    response = client.post(
        "/api/documents/exam_test/auto-fix",
        json={"xml_content": broken_xml},
    )
    assert response.status_code == 200
    data = response.json()
    assert "cleaned_xml" in data
    assert "</stimulus>" not in data["cleaned_xml"]
    assert '<stimulus id="stim_1"' in data["cleaned_xml"]
    assert "fixes" in data
    assert len(data["fixes"]) > 0
