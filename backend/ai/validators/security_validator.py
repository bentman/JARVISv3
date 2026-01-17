"""
Security Validator for JARVISv3
"""
import re
import json
import logging
from typing import Any, Optional
from .base import BaseValidator, ValidationResult
from ..context.schemas import TaskContext
from ...core.privacy import privacy_service, DataClassification

logger = logging.getLogger(__name__)

class SecurityValidator(BaseValidator):
    """Validates security and privacy constraints (PII, dangerous patterns)"""
    
    @property
    def name(self) -> str:
        return "security_validator"
        
    async def validate(self, target: Any, context: Optional[TaskContext] = None, **kwargs) -> ValidationResult:
        errors = []
        warnings = []
        
        # Determine privacy level
        privacy_level = "medium"
        if context:
            privacy_level = context.system_context.user_preferences.privacy_level
            
        # Serialize target to string for pattern checking
        target_str = str(target)
        if isinstance(target, (dict, list)):
            try:
                target_str = json.dumps(target)
            except:
                pass
                
        # 1. PII Detection using PrivacyService
        classification = privacy_service.classify_data(target_str)
        if classification == DataClassification.SENSITIVE:
            msg = "Sensitive data (PII) detected"
            if privacy_level == "high":
                errors.append(msg)
            else:
                warnings.append(msg)
        
        # 2. Check for internal system secrets
        secret_patterns = [
            (r'password["\']?\s*[:=]\s*["\']?\w+', "Password pattern detected"),
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?\w+', "API Key pattern detected"),
            (r'token["\']?\s*[:=]\s*["\']?\w+', "Token pattern detected")
        ]
        
        for pattern, warning in secret_patterns:
            if re.search(pattern, target_str, re.IGNORECASE):
                warnings.append(warning)
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
