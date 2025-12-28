#!/usr/bin/env python3
"""
JARVISv3 Backend Validation Suite
Validates Backend Core, Feature Parity, Integration, and AI Intelligence.
Generates a timestamped report in the reports/ directory.
"""

import asyncio
import sys
import os
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the root directory to the path so we can import the backend package
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the validation function from the correct location
# Removed imports that cause errors

class ValidationLogger:
    """Handles terminal output and file-based reporting"""
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = ROOT_DIR / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.report_file = self.report_dir / f"validate_backend-{self.timestamp}.txt"
        self.buffer = []
        
        self.log(f"JARVISv3 Validation Session started at {datetime.now().isoformat()}")
        self.log(f"Report File: {self.report_file}")
        self.log("="*60)

    def log(self, message: str):
        print(message)
        self.buffer.append(message)

    def save(self):
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer))
        print(f"\n✓ Report saved to {self.report_file}")

    def header(self, title: str):
        self.log("\n" + "="*60)
        self.log(title.upper())
        self.log("="*60)

# Frontend validation removed - this is a backend-focused validation script

async def main():
    """Main validation function"""
    logger = ValidationLogger()
    
    # Run all tests recursively
    logger.header("Running All Tests in backend/tests/")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "backend/tests/", "--tb=short"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True
    )
    
    logger.log("Pytest Output:")
    logger.log(result.stdout)
    if result.stderr:
        logger.log("Stderr:")
        logger.log(result.stderr)
    
    # Parse summary for status
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    lines = result.stdout.split('\n')
    summary_found = False
    for line in lines:
        if '===' in line and 'passed' in line:
            # e.g. === 10 passed, 2 failed, 3 skipped in 1.23s ===
            match = re.search(r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) errors?)?', line)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2)) if match.group(2) else 0
                skipped = int(match.group(3)) if match.group(3) else 0
                errors = int(match.group(4)) if match.group(4) else 0
                summary_found = True
            break
    
    if summary_found:
        if failed > 0 or errors > 0:
            test_status = "FAIL"
        elif skipped > 0:
            test_status = "WARN"
        else:
            test_status = "SUCCESS"
    else:
        if result.returncode == 0:
            test_status = "SUCCESS"
        else:
            test_status = "FAIL"
    
    logger.log(f"Test Status: {test_status}")
    test_success = (result.returncode == 0)  # or based on parsed, but since if returncode !=0, FAIL, test_success = False
    
    # Summary
    logger.header("Backend Validation Summary")
    logger.log(f"All Tests:       {'✓ PASS' if test_success else ('⚠ WARN' if test_status == 'WARN' else '✗ FAIL')}")
    logger.log("="*60)

    if test_success:
        logger.log("\n✅ JARVISv3 Backend Tests PASSED!")
        status = 0
    else:
        logger.log("\n❌ Validation failed - see test failures above")
        status = 1
        
    logger.save()
    return status

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
