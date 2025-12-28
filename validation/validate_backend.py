"""
JARVISv3 Backend Validation Suite
Single comprehensive validation that runs all backend tests and generates timestamped reports.
"""
import asyncio
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def get_venv_python_path() -> Path:
    """Get the path to the Python executable in the virtual environment"""
    base_path = Path("backend")
    venv_path = base_path / ".venv"
    
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    return python_exe


class ValidationLogger:
    """Handles terminal output and file-based reporting"""
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        self.report_file = self.report_dir / f"backend_validation_report_{self.timestamp}.txt"
        self.buffer = []
        
        self.log(f"JARVISv3 Backend Validation Session started at {datetime.now().isoformat()}")
        self.log(f"Report File: {self.report_file}")
        self.log("="*60)

    def log(self, message: str):
        print(message)
        self.buffer.append(message)
    
    def header(self, title: str):
        self.log("\n" + "="*60)
        self.log(title.upper())
        self.log("="*60)

    def save(self):
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer))
        print(f"\n[SUCCESS] Report saved to {self.report_file}")


def validate_venv(logger) -> bool:
    """Validate that the virtual environment exists and is valid"""
    venv_python = get_venv_python_path()
    
    if not venv_python.exists():
        logger.log(f"ERROR: Virtual environment not found: {venv_python}")
        logger.log("Please create the virtual environment first using: python -m venv backend/.venv")
        return False
    
    # Test that the venv python can run and import required modules
    try:
        result = subprocess.run(
            [str(venv_python), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            logger.log(f"ERROR: Virtual environment Python is not working: {result.stderr}")
            return False
        else:
            logger.log(f"SUCCESS: Using Python interpreter: {venv_python} ({result.stdout.strip()})")
            return True
    except subprocess.TimeoutExpired:
        logger.log("ERROR: Virtual environment Python timed out during version check")
        return False
    except Exception as e:
        logger.log(f"ERROR: Error accessing virtual environment Python: {e}")
        return False


def run_pytest_suite(logger, suite_name: str, test_files: list) -> bool:
    """Execute a list of pytest files and log results using venv Python"""
    logger.header(suite_name)
    
    # Get the venv Python path
    venv_python = get_venv_python_path()
    
    # Validate the venv
    if not validate_venv(logger):
        logger.log(f"ERROR: Skipping {suite_name} - Virtual environment not available")
        return False
    
    all_passed = True
    for test_path in test_files:
        logger.log(f"Running pytest: {test_path}...")
        result = subprocess.run(
            [str(venv_python), "-m", "pytest", test_path, "-v"],
            cwd=Path.cwd(),  # Use current working directory
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.log(f"SUCCESS: {test_path}: PASS")
        else:
            logger.log(f"FAILED: {test_path}: FAIL")
            logger.log("--- STDOUT ---")
            logger.log(result.stdout)
            logger.log("--- STDERR ---")
            logger.log(result.stderr)
            all_passed = False
            
    return all_passed


def run_existing_backend_tests(logger) -> bool:
    """Run all existing backend tests"""
    logger.header("Backend Integration Tests")
    
    # These are the actual test files that exist
    existing_test_files = [
        "backend/tests/test_conversation_management.py",
        "backend/tests/test_e2e_model.py",
        "backend/tests/test_voice_session.py",
        "backend/tests/test_workflow_engine_failures.py"
    ]
    
    return run_pytest_suite(logger, "Backend Integration Tests", existing_test_files)


def run_workflow_engine_failures_test(logger) -> bool:
    """Run workflow engine failures test specifically using venv Python"""
    logger.header("Workflow Engine Failures Test")
    
    # Get the venv Python path
    venv_python = get_venv_python_path()
    
    # Validate the venv
    if not validate_venv(logger):
        logger.log("ERROR: Skipping Workflow Engine Failures test - Virtual environment not available")
        return False
    
    result = subprocess.run(
        [str(venv_python), "-m", "pytest", "backend/tests/test_workflow_engine_failures.py", "-v"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.log("SUCCESS: Workflow Engine Failures: PASS")
        return True
    else:
        logger.log("FAILED: Workflow Engine Failures: FAIL")
        logger.log("--- STDOUT ---")
        logger.log(result.stdout)
        logger.log("--- STDERR ---")
        logger.log(result.stderr)
        return False


async def run_e2e_model_test(logger) -> bool:
    """Run end-to-end model test using venv Python"""
    logger.header("AI Intelligence Validation (E2E)")
    
    # Get the venv Python path
    venv_python = get_venv_python_path()
    
    # Validate the venv
    if not validate_venv(logger):
        logger.log("ERROR: Skipping AI Intelligence test - Virtual environment not available")
        return False
    
    # Run pytest on the specific e2e model test functions
    result = subprocess.run(
        [str(venv_python), "-m", "pytest", "backend/tests/test_e2e_model.py", "-v", "-k", "test_model_provider_availability or test_real_inference_execution", "--tb=short"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout for model tests
    )
    
    if result.returncode == 0:
        # Check if tests were skipped (which should be treated as failure in validation context)
        if "SKIPPED" in result.stdout:
            logger.log("WARNING: AI Intelligence: SOME TESTS WERE SKIPPED (no models available)")
            logger.log("FAILED: AI Intelligence: FAIL (validation requires actual model connectivity)")
            return False
        else:
            logger.log("SUCCESS: AI Intelligence: PASS")
            return True
    else:
        logger.log("FAILED: AI Intelligence: FAIL")
        logger.log("--- STDOUT ---")
        logger.log(result.stdout)
        logger.log("--- STDERR ---")
        logger.log(result.stderr)
        return False


def main():
    """Main validation function"""
    logger = ValidationLogger()
    
    # Check if venv is available before running tests
    venv_python = get_venv_python_path()
    if not venv_python.exists():
        logger.log(f"WARNING: Virtual environment not found at: {venv_python}")
        logger.log("Some tests may fail if dependencies are not installed globally.")
        logger.log("Recommended: Create venv with 'python -m venv backend/.venv'")
    else:
        logger.log(f"SUCCESS: Virtual environment found at: {venv_python}")
    
    # Run existing backend tests
    integration_success = run_existing_backend_tests(logger)
    
    # Run specific workflow engine tests
    workflow_success = run_workflow_engine_failures_test(logger)
    
    # Run async tests
    e2e_success = asyncio.run(run_e2e_model_test(logger))
    
    # Summary
    logger.header("Backend Validation Summary")
    logger.log(f"Backend Tests:     {'PASS' if integration_success else 'FAIL'}")
    logger.log(f"Workflow Failures: {'PASS' if workflow_success else 'FAIL'}")
    logger.log(f"AI Intelligence:   {'PASS' if e2e_success else 'FAIL'}")
    logger.log("="*60)

    total_success = integration_success and workflow_success and e2e_success
    
    if total_success:
        logger.log("\n✅ JARVISv3 Backend is FULLY validated!")
        status = 0
    else:
        logger.log("\n!!! Validation failed - see specific component failures above")
        status = 1
        
    logger.save()
    return status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)