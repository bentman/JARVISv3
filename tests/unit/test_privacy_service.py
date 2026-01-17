"""
Unit tests for Privacy Service
"""
import pytest
import asyncio
from datetime import datetime, UTC
from backend.core.privacy import privacy_service


def test_privacy_service_enhancements():
    """Test privacy service enhancements"""
    # Test data classification
    classification = privacy_service.classify_data("My email is test@example.com")
    assert str(classification) == "personal"

    # Test PII redaction
    redacted = privacy_service.redact_sensitive_data("My email is test@example.com and phone is 123-456-7890")
    assert "[EMAIL_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted

    # Test local processing decision
    should_process_locally = privacy_service.should_process_locally("Sensitive data", "high")
    assert should_process_locally is True

    # Test data hashing
    data_hash = privacy_service.generate_data_hash("test data")
    assert len(data_hash) == 64  # SHA256 hash length

    # Test data retention check
    retention_ok = privacy_service.check_data_retention("test data", datetime.now(UTC))
    assert retention_ok is True

    # Test privacy audit log
    audit_log = privacy_service.create_privacy_audit_log("data_access", "personal", "test_user")
    assert "timestamp" in audit_log
    assert "compliance" in audit_log

    # Test anonymization
    anonymized = privacy_service.anonymize_data("John Doe lives at 123 Main St", "high")
    assert "[NAME_REDACTED]" in anonymized

    # Test consent requirements
    consent_reqs = privacy_service.get_consent_requirements("personal")
    assert "consent_required" in consent_reqs