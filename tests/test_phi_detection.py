"""Tests for PHI detection — Layers 1 and 6."""

import pytest


class TestPHIDeidentifier:
    @pytest.fixture
    def deidentifier(self):
        try:
            from src.ingestion.deidentifier import PHIDeidentifier
            d = PHIDeidentifier(threshold=0.5)
            d._ensure_loaded()
            return d
        except (ImportError, OSError):
            pytest.skip("Presidio or spacy model not installed")

    def test_detect_person(self, deidentifier):
        detections = deidentifier.detect("Patient John Smith was admitted.")
        assert any(d["entity_type"] == "PERSON" for d in detections)

    def test_clean_text(self, deidentifier):
        assert deidentifier.is_clean("Metformin treats type 2 diabetes.")

    def test_redacts(self, deidentifier):
        result = deidentifier.deidentify("Dr. Jane Doe prescribed metformin. Call 555-0100.")
        assert result.entities_found > 0
        assert "Jane Doe" not in result.cleaned_text


class TestPHIOutputScanner:
    @pytest.fixture
    def scanner(self):
        try:
            from src.filtering.phi_scanner import PHIOutputScanner
            s = PHIOutputScanner(threshold=0.5)
            s._phi._ensure_loaded()
            return s
        except (ImportError, OSError):
            pytest.skip("Presidio or spacy model not installed")

    def test_clean(self, scanner):
        assert scanner.is_clean("Metformin has GI side effects.")

    def test_phi_detected(self, scanner):
        assert len(scanner.scan("John Smith at 123-456-7890")) > 0
