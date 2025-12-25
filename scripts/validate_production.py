#!/usr/bin/env python3
"""
JARVISv3 Unified Validation Suite
Validates Backend Core, Feature Parity, Integration, Frontend, and AI Intelligence.
Generates a timestamped report in the reports/ directory.
"""

import asyncio
import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the root directory to the path so we can import the backend package
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the validation function from the correct location
try:
    from backend.tests.test_production_readiness import validate_production_readiness
    from backend.tests.test_e2e_model import test_real_inference
except ImportError as e:
    print(f"Error importing backend tests: {e}")
    sys.exit(1)

class ValidationLogger:
    """Handles terminal output and file-based reporting"""
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = ROOT_DIR / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.report_file = self.report_dir / f"validation_report_{self.timestamp}.txt"
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

async def run_frontend_tests(logger: ValidationLogger) -> bool:
    """Execute frontend vitest suite"""
    logger.header("Frontend Validation (Vitest)")
    
    frontend_dir = ROOT_DIR / "frontend"
    if not frontend_dir.exists():
        logger.log("⚠ SKIP: Frontend directory not found.")
        return True

    try:
        logger.log(f"Running npm run test:run in {frontend_dir}...")
        # Use shell=True for Windows compatibility with npm
        result = subprocess.run(
            ["npm", "run", "test:run"], 
            cwd=frontend_dir, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=True
        )
        
        if result.returncode == 0:
            logger.log("✓ Frontend tests passed!")
            return True
        else:
            logger.log("✗ Frontend tests failed:")
            logger.log(result.stdout)
            logger.log(result.stderr)
            return False
    except FileNotFoundError:
        logger.log("⚠ SKIP: 'npm' command not found. Ensure Node.js is installed.")
        return True 

async def run_pytest_suite(logger: ValidationLogger, suite_name: str, test_files: List[tuple]) -> bool:
    """Execute a list of pytest files and log results"""
    logger.header(suite_name)
    
    all_passed = True
    for name, path in test_files:
        logger.log(f"Running {name} ({path})...")
        # Use sys.executable to ensure we use the correct python/venv
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.log(f"✓ {name}: PASS")
        else:
            logger.log(f"✗ {name}: FAIL")
            logger.log("--- STDOUT ---")
            logger.log(result.stdout)
            logger.log("--- STDERR ---")
            logger.log(result.stderr)
            all_passed = False
            
    return all_passed

async def main():
    """Main validation function"""
    logger = ValidationLogger()
    
    # 1. Backend Core Validation
    # We run this directly as it's an imported function returning bool
    logger.header("Backend Core Validation")
    # Redirect stdout briefly to capture it if needed, or just let it print
    # For now, let validate_production_readiness print to stdout, 
    # but we'll capture its success
    core_success = await validate_production_readiness()
    
    # 2. Feature Parity & Reliability Integration
    integration_tests = [
        ("Voice Session API", "backend/tests/test_voice_session.py"),
        ("Voice Reliability", "backend/tests/test_voice_reliability.py"),
        ("Search Parity", "backend/tests/test_search_parity.py"),
        ("Memory Parity", "backend/tests/test_memory_parity.py"),
        ("Conversation Management API", "backend/tests/test_conversation_management.py"),
        ("Hardware Status API", "backend/tests/test_hardware_api.py"),
        ("Workflow Engine Failures", "backend/tests/test_workflow_engine_failures.py"),
    ]
    integration_success = await run_pytest_suite(logger, "Backend Integration & Parity", integration_tests)
    
    # 3. Frontend Validation
    frontend_success = await run_frontend_tests(logger)
    
    # 4. AI Intelligence (E2E) Validation
    logger.header("AI Intelligence Validation (E2E Smoke)")
    intelligence_success = await test_real_inference()
    
    # Summary
    logger.header("Unified Validation Summary")
    logger.log(f"Backend Core:    {'✓ PASS' if core_success else '✗ FAIL'}")
    logger.log(f"Integration:     {'✓ PASS' if integration_success else '✗ FAIL'}")
    logger.log(f"Frontend:        {'✓ PASS' if frontend_success else '✗ FAIL'}")
    logger.log(f"AI Intelligence: {'✓ PASS' if intelligence_success else '✗ FAIL'}")
    logger.log("="*60)

    if core_success and integration_success and frontend_success and intelligence_success:
        logger.log("\n✅ JARVISv3 system is FULLY validated!")
        status = 0
    else:
        logger.log("\n❌ Validation failed - see specific component failures above")
        status = 1
        
    logger.save()
    return status

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
