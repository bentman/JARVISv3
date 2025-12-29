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


def discover_test_files(test_dir: Path, category: str) -> list[str]:
    """Discover test files in a given directory category"""
    if not test_dir.exists():
        return []

    test_files = []
    # Find all test_*.py files in this directory and subdirectories
    for pattern in ["test_*.py", "**/test_*.py"]:
        for test_file in test_dir.glob(pattern):
            if test_file.is_file():
                test_files.append(str(test_file))

    # Sort for consistent ordering
    return sorted(set(test_files))


def run_pytest_on_directory(logger, category_name: str, test_dir: str) -> bool:
    """Run pytest on an entire directory of tests with per-test visibility"""
    logger.header(f"{category_name} Tests")

    venv_python = get_venv_python_path()

    # Validate the venv
    if not validate_venv(logger):
        logger.log(f"ERROR: Skipping {category_name} tests - Virtual environment not available")
        return False

    # Run pytest on the entire directory with verbose output
    result = subprocess.run(
        [str(venv_python), "-m", "pytest", test_dir, "-v", "--tb=no"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )

    # Parse individual test results
    test_results = []
    stdout_lines = result.stdout.strip().split('\n')

    for line in stdout_lines:
        line = line.strip()
        if line.startswith('tests/') or line.startswith('::'):
            if 'PASSED' in line:
                test_results.append(('PASS', line.split(' PASSED')[0]))
            elif 'FAILED' in line:
                test_results.append(('FAIL', line.split(' FAILED')[0]))
            elif 'SKIPPED' in line:
                test_results.append(('SKIP', line.split(' SKIPPED')[0]))
            elif 'ERROR' in line:
                test_results.append(('ERROR', line.split(' ERROR')[0]))

    # Display per-test results
    for status, test_name in test_results:
        status_icon = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○', 'ERROR': '✗'}.get(status, '?')
        logger.log(f"  {status_icon} {status}: {test_name}")

    # Extract and display summary
    summary_line = None
    for line in reversed(stdout_lines):
        if 'passed' in line and ('failed' in line or 'skipped' in line or 'error' in line):
            summary_line = line
            break

    if summary_line:
        logger.log(f"SUCCESS: {category_name}: {summary_line}")
    else:
        logger.log(f"SUCCESS: {category_name} tests completed")

    return result.returncode == 0


async def run_e2e_model_test(logger) -> bool:
    """Run end-to-end model test using venv Python with per-test visibility"""
    logger.header("AI Intelligence Validation (E2E)")

    # Get the venv Python path
    venv_python = get_venv_python_path()

    # Validate the venv
    if not validate_venv(logger):
        logger.log("ERROR: Skipping AI Intelligence test - Virtual environment not available")
        return False

    # Run pytest on the specific e2e model test functions
    result = subprocess.run(
        [str(venv_python), "-m", "pytest", "tests/integration/test_model_execution.py", "-v", "-k", "test_model_provider_availability or test_real_inference_execution", "--tb=no"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout for model tests
    )

    # Parse individual test results
    test_results = []
    stdout_lines = result.stdout.strip().split('\n')

    for line in stdout_lines:
        line = line.strip()
        if 'test_model_provider_availability' in line or 'test_real_inference_execution' in line:
            if 'PASSED' in line:
                test_results.append(('PASS', line.split(' PASSED')[0]))
            elif 'FAILED' in line:
                test_results.append(('FAIL', line.split(' FAILED')[0]))
            elif 'SKIPPED' in line:
                test_results.append(('SKIP', line.split(' SKIPPED')[0]))

    # Display per-test results
    for status, test_name in test_results:
        status_icon = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○'}.get(status, '?')
        logger.log(f"  {status_icon} {status}: {test_name}")

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
        if result.stderr:
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

    # Run tests by category using dynamic discovery
    unit_success = run_pytest_on_directory(logger, "Unit", "tests/unit")
    integration_success = run_pytest_on_directory(logger, "Integration", "tests/integration")
    agentic_success = run_pytest_on_directory(logger, "Agentic", "tests/agentic")

    # Run async E2E model tests
    e2e_success = asyncio.run(run_e2e_model_test(logger))

    # Summary
    logger.header("Backend Validation Summary")
    logger.log(f"Unit Tests:        {'PASS' if unit_success else 'FAIL'}")
    logger.log(f"Integration Tests: {'PASS' if integration_success else 'FAIL'}")
    logger.log(f"Agentic Tests:     {'PASS' if agentic_success else 'FAIL'}")
    logger.log(f"AI Intelligence:   {'PASS' if e2e_success else 'FAIL'}")
    logger.log("="*60)

    total_success = unit_success and integration_success and agentic_success and e2e_success

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
