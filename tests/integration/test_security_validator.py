#!/usr/bin/env python3
"""
End-to-end test for security validator functionality
Tests the implemented security validation features locally
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.security import security_validator


async def test_security_validation():
    """Test the security validator with various input types"""
    
    print("Testing Security Validator...")
    
    # Test 1: Clean input (should pass)
    clean_input = "This is a test."
    result1 = await security_validator.validate_input(clean_input)
    print(f"✓ Clean input: valid={result1['is_valid']}, issues={len(result1['issues'])}")
    if result1['issues']:
        print(f"  Issues found: {result1['issues']}")
    assert result1['is_valid'] == True
    assert len(result1['issues']) == 0
    
    # Test 2: PII detection (SSN)
    ssn_input = "My SSN is 123-45-6789 and I need help."
    result2 = await security_validator.validate_input(ssn_input)
    print(f"✓ PII detection: valid={result2['is_valid']}, issues={len(result2['issues'])}")
    assert result2['is_valid'] == False
    assert any(issue['type'] == 'PII_DETECTED' and issue['category'] == 'ssn' for issue in result2['issues'])
    
    # Test 3: SQL injection detection
    sql_input = "SELECT * FROM users WHERE id = 1 OR 1=1; DROP TABLE users;"
    result3 = await security_validator.validate_input(sql_input)
    print(f"✓ SQL injection: valid={result3['is_valid']}, issues={len(result3['issues'])}")
    assert result3['is_valid'] == False
    assert any(issue['type'] == 'SQL_INJECTION' for issue in result3['issues'])
    
    # Test 4: XSS detection
    xss_input = '<script>alert("XSS");</script> Hello world'
    result4 = await security_validator.validate_input(xss_input)
    print(f"✓ XSS detection: valid={result4['is_valid']}, issues={len(result4['issues'])}")
    assert result4['is_valid'] == False
    assert any(issue['type'] == 'XSS_DETECTED' for issue in result4['issues'])
    
    # Test 5: Email detection
    email_input = "Please contact me at test@example.com for more info"
    result5 = await security_validator.validate_input(email_input)
    print(f"✓ Email detection: valid={result5['is_valid']}, issues={len(result5['issues'])}")
    assert result5['is_valid'] == False
    assert any(issue['type'] == 'PII_DETECTED' and issue['category'] == 'email' for issue in result5['issues'])
    
    # Test sanitization
    xss_input = '<script>alert("XSS");</script> Hello world'
    sanitized = await security_validator.sanitize_input(xss_input)
    print(f"✓ Sanitization: original='{xss_input}', sanitized='{sanitized}'")
    assert '<script>' not in sanitized
    assert 'Hello world' in sanitized
    
    print("\nAll security validation tests passed!")
    print("Security validator is fully functional and locally exercised.")


if __name__ == "__main__":
    asyncio.run(test_security_validation())
