"""
Security module for JARVISv3
Implements comprehensive input validation and sanitization
"""
import re
import html
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """
    Comprehensive security validator for input sanitization and validation
    """
    
    def __init__(self):
        # Regex patterns for PII detection
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'iban': r'\b[A-Z]{2}[0-9A-Z]{2}[0-9A-Z]{1,30}\b',
        }
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
            r'(\'|")\s*(--|#|/\*|;)',  # Comment indicators
            r'(\bOR\b|\bAND\b)\s*[\'"][^\'"]*[\'"]\s*=\s*[\'"][^\'"]*[\'"]',  # Basic SQLi
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>',
        ]
    
    async def validate_input(self, input_text: str) -> Dict[str, Any]:
        """
        Validate input for security issues
        Returns a dictionary with validation results
        """
        issues = []
        sanitized_text = input_text # Start with original text
        
        # Check for PII
        for name, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, input_text, re.IGNORECASE)
            if matches:
                issues.append({
                    'type': 'PII_DETECTED',
                    'category': name,
                    'matches': matches,
                    'severity': 'high'
                })
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append({
                    'type': 'SQL_INJECTION',
                    'pattern': pattern,
                    'severity': 'critical'
                })
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, input_text, re.IGNORECASE | re.DOTALL):
                issues.append({
                    'type': 'XSS_DETECTED',
                    'pattern': pattern,
                    'severity': 'critical'
                })
        
        # Check for command injection
        cmd_injection_patterns = [
            r'[;&|]',
            r'\$\(',
            r'`.*?`',
            r'exec\(',
            r'system\(',
        ]
        for pattern in cmd_injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append({
                    'type': 'COMMAND_INJECTION',
                    'pattern': pattern,
                    'severity': 'critical'
                })
        
        # Determine if input is valid based on issues (any issue makes it invalid for strict validation)
        has_critical = any(issue['severity'] == 'critical' for issue in issues)
        has_issues = len(issues) > 0
        is_valid = not has_issues # Strict validation: any issue (PII, etc.) makes it invalid
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'has_critical': has_critical,
            'sanitized_text': sanitized_text
        }
    
    async def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize input by escaping HTML and removing potentially dangerous content
        """
        # HTML escape to prevent XSS
        sanitized = html.escape(input_text)
        
        # Remove potential script tags (additional protection)
        sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized

class InputValidator:
    """
    Additional input validation for specific data types and formats
    """
    
    @staticmethod
    async def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    async def validate_url(url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    async def validate_file_path(path: str) -> bool:
        """Validate file path to prevent directory traversal"""
        # Check for directory traversal attempts
        if '..' in path or './' in path or '../' in path:
            return False
        return True

# Global instances
security_validator = SecurityValidator()
input_validator = InputValidator()
