"""Privacy Service for JARVISv3
Ports JARVISv2 privacy patterns and classification to JARVISv3.
"""
import re
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
try:
    from enum import StrEnum
except ImportError:
    # Fallback for Python < 3.11
    class StrEnum(str, Enum):
        pass
from pydantic import BaseModel
from datetime import datetime, timedelta, UTC
import hashlib

logger = logging.getLogger(__name__)

class DataClassification(StrEnum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"

class PrivacyService:
    """
    Privacy service for data classification and PII redaction.
    """

    def __init__(self):
        # Regex patterns from JARVISv2
        self.patterns = {
            DataClassification.SENSITIVE: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # US SSN
                r'\b(?:\d{4}[ -]?){3}\d{4}\b',  # Credit card (generic 16-digit)
                r'\b[A-Z]{2}[0-9A-Z]{2}[0-9A-Z]{1,30}\b',  # IBAN (generic)
            ],
            DataClassification.PERSONAL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # Email
                r'\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}\b',  # Intl/US phone
                r'\b\d{9,19}\b',  # Bank acct (generic long digit sequences)
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IPv4
            ]
        }

        self.sensitive_keywords = [
            "password", "social security", "medical record", "financial",
            "ssn", "credit card", "bank account", "tax id", "national insurance",
            "sensitive", "restricted", "confidential"
        ]

        self.personal_keywords = [
            "name", "address", "phone", "email", "birthday", "birth date",
            "passport", "driver's license", "national id", "personal", "pii"
        ]

        # GDPR/CCPA compliance settings
        self.data_retention_policies = {
            "public": timedelta(days=365),  # 1 year
            "personal": timedelta(days=180),  # 6 months
            "sensitive": timedelta(days=90),  # 3 months
            "restricted": timedelta(days=30)  # 1 month
        }

    def classify_data(self, content: str) -> DataClassification:
        """
        Classify data based on sensitivity patterns and keywords.
        """
        content_lower = content.lower()

        # Check patterns first
        for classification, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, content):
                    return classification

        # Check keywords
        for keyword in self.sensitive_keywords:
            if keyword in content_lower:
                return DataClassification.SENSITIVE

        for keyword in self.personal_keywords:
            if keyword in content_lower:
                return DataClassification.PERSONAL

        return DataClassification.PUBLIC

    def redact_sensitive_data(self, content: str) -> str:
        """
        Redact sensitive information from content.
        """
        # Emails
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]', content)
        # Phones (intl + US)
        content = re.sub(r'\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{3,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}\b', '[PHONE_REDACTED]', content)
        # Credit cards
        content = re.sub(r'\b(?:\d{4}[ -]?){3}\d{4}\b', '[CREDIT_CARD_REDACTED]', content)
        # SSNs
        content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', content)
        # IBAN
        content = re.sub(r'\b[A-Z]{2}[0-9A-Z]{2}[0-9A-Z]{1,30}\b', '[IBAN_REDACTED]', content)
        # IPv4
        content = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', content)

        return content

    def should_process_locally(self, content: str, privacy_level: str = "medium") -> bool:
        """
        Determine if data should be processed locally based on classification and privacy level.
        """
        classification = self.classify_data(content)

        if privacy_level == "high":
            # In high privacy, almost everything should be local if it's personal/sensitive
            return classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE, DataClassification.RESTRICTED]
        elif privacy_level == "medium":
            # In medium, sensitive/restricted must be local
            return classification in [DataClassification.SENSITIVE, DataClassification.RESTRICTED]
        else: # low
            return classification == DataClassification.RESTRICTED

    def generate_data_hash(self, content: str) -> str:
        """
        Generate a hash for data tracking while preserving privacy
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def check_data_retention(self, content: str, creation_date: datetime) -> bool:
        """
        Check if data should be retained based on classification and retention policies
        """
        classification = self.classify_data(content)
        retention_period = self.data_retention_policies.get(classification, timedelta(days=365))

        return datetime.now(UTC) - creation_date <= retention_period

    def create_privacy_audit_log(self, action: str, data_type: str,
                               user_id: str, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Create a privacy audit log entry for compliance tracking
        """
        if not timestamp:
            timestamp = datetime.now(UTC)

        return {
            "timestamp": timestamp.isoformat(),
            "action": action,
            "data_type": data_type,
            "user_id": user_id,
            "compliance": "GDPR/CCPA",
            "status": "compliant"
        }

    def anonymize_data(self, content: str, level: str = "medium") -> str:
        """
        Anonymize data based on privacy level
        """
        if level == "high":
            # Full anonymization - remove all personal and sensitive data
            content = self.redact_sensitive_data(content)
            # Remove names (simple pattern)
            content = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME_REDACTED]', content)
            # Remove locations
            content = re.sub(r'\b\d{1,5} [A-Za-z ]+, [A-Z]{2} \d{5}\b', '[LOCATION_REDACTED]', content)
        elif level == "medium":
            # Standard anonymization - redact sensitive data only
            content = self.redact_sensitive_data(content)
        # low level - no anonymization

        return content

    def get_consent_requirements(self, data_type: str) -> Dict[str, Any]:
        """
        Get consent requirements based on data type for GDPR/CCPA compliance
        """
        requirements = {
            "public": {
                "consent_required": False,
                "retention_disclosure": False,
                "processing_disclosure": False
            },
            "personal": {
                "consent_required": True,
                "retention_disclosure": True,
                "processing_disclosure": True,
                "right_to_delete": True,
                "right_to_access": True
            },
            "sensitive": {
                "consent_required": True,
                "explicit_consent": True,
                "retention_disclosure": True,
                "processing_disclosure": True,
                "right_to_delete": True,
                "right_to_access": True,
                "data_minimization": True
            },
            "restricted": {
                "consent_required": True,
                "explicit_consent": True,
                "special_protection": True,
                "limited_processing": True,
                "encryption_required": True
            }
        }

        return requirements.get(data_type, requirements["public"])

# Global instance
privacy_service = PrivacyService()
