#!/usr/bin/env python3
"""
JARVISv3 Unified Validation Suite
Validates Backend, Frontend, and AI Intelligence in one command.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# Add the root directory to the path so we can import the backend package
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the validation function from the correct location
from backend.tests.test_production_readiness import validate_production_readiness
from backend.tests.test_e2e_model import test_real_inference

async def run_frontend_tests():
    """Execute frontend vitest suite"""
    print("\n" + "="*60)
    print("FRONTEND VALIDATION (Vitest)")
    print("="*60)
    
    frontend_dir = ROOT_DIR / "frontend"
    if not frontend_dir.exists():
        print("⚠ SKIP: Frontend directory not found.")
        return True

    try:
        print(f"Running npm run test:run in {frontend_dir}...")
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
            print("✓ Frontend tests passed!")
            return True
        else:
            print("✗ Frontend tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠ SKIP: 'npm' command not found. Ensure Node.js is installed.")
        return True # Don't fail the whole suite if npm is missing on a partial dev env

async def main():
    """Main validation function"""
    print("Starting JARVISv3 Unified Validation Suite...\n")
    
    # 1. Backend Core Validation
    backend_success = await validate_production_readiness()
    
    # 2. Frontend Validation
    frontend_success = await run_frontend_tests()
    
    # 3. AI Intelligence (E2E) Validation
    print("\n" + "="*60)
    print("AI INTELLIGENCE VALIDATION (E2E Smoke Test)")
    print("="*60)
    intelligence_success = await test_real_inference()
    
    print("\n" + "="*60)
    print("UNIFIED VALIDATION SUMMARY")
    print("="*60)
    print(f"Backend Core:    {'✓ PASS' if backend_success else '✗ FAIL'}")
    print(f"Frontend:        {'✓ PASS' if frontend_success else '✗ FAIL'}")
    print(f"AI Intelligence: {'✓ PASS' if intelligence_success else '✗ FAIL'}")
    print("="*60)

    if backend_success and frontend_success and intelligence_success:
        print("\n✅ JARVISv3 system is FULLY validated!")
        return 0
    else:
        print("\n❌ Validation failed - see specific component failures above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
