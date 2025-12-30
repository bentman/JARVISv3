"""
JARVISv3 Backend Validation Suite
Single comprehensive validation that runs all backend tests and generates timestamped reports.
"""
import asyncio
import sys
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
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


def extract_deprecation_warnings(stderr: str) -> list[str]:
    """Extract deprecation warnings from stderr output"""
    warnings = []
    if not stderr:
        return warnings

    lines = stderr.split('\n')
    for line in lines:
        line = line.strip()
        if any(w in line for w in ['DeprecationWarning', 'FutureWarning', 'PendingDeprecationWarning']):
            warnings.append(line)
    return warnings


def parse_junit_xml(xml_file: Path) -> tuple[list[tuple[str, str]], str, bool, bool]:
    """Parse JUnit XML file and return (test_results, summary, success, has_skips)"""
    if not xml_file.exists():
        return [], "No XML report generated", False, False

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        test_results = []
        total_tests = 0
        failures = 0
        errors = 0
        skipped = 0

        # Parse test cases
        for testsuite in root:
            for testcase in testsuite:
                total_tests += 1
                test_name = f"{testcase.get('classname', '')}::{testcase.get('name', '')}"

                if testcase.find('failure') is not None:
                    test_results.append(('FAIL', test_name))
                    failures += 1
                elif testcase.find('error') is not None:
                    test_results.append(('ERROR', test_name))
                    errors += 1
                elif testcase.find('skipped') is not None:
                    test_results.append(('SKIP', test_name))
                    skipped += 1
                else:
                    test_results.append(('PASS', test_name))

        # Check for collection errors in system-out/system-err and error elements
        collection_errors = []
        for testsuite in root:
            # Check error elements in testcases (collection errors)
            for testcase in testsuite:
                error_elem = testcase.find('error')
                if error_elem is not None and error_elem.text:
                    error_msg = error_elem.text.strip()
                    if 'ModuleNotFoundError' in error_msg or 'ImportError' in error_msg:
                        collection_errors.append(f"{testcase.get('name', 'unknown')}: {error_msg[:200]}...")

            # Check system-out/system-err for additional errors
            system_out = testsuite.find('system-out')
            system_err = testsuite.find('system-err')
            if system_out is not None and system_out.text:
                out_text = system_out.text.strip()
                if 'ERROR' in out_text or 'ModuleNotFoundError' in out_text or 'ImportError' in out_text:
                    # Extract first error line
                    lines = out_text.split('\n')
                    for line in lines:
                        if 'ERROR' in line or 'ModuleNotFoundError' in line or 'ImportError' in line:
                            collection_errors.append(line.strip()[:200])
                            break
            if system_err is not None and system_err.text:
                err_text = system_err.text.strip()
                if 'ERROR' in err_text or 'ModuleNotFoundError' in err_text or 'ImportError' in err_text:
                    # Extract first error line
                    lines = err_text.split('\n')
                    for line in lines:
                        if 'ERROR' in line or 'ModuleNotFoundError' in line or 'ImportError' in line:
                            collection_errors.append(line.strip()[:200])
                            break

        # Build summary
        summary_parts = []
        if total_tests > 0:
            summary_parts.append(f"{total_tests} tests")
        if failures > 0:
            summary_parts.append(f"{failures} failed")
        if errors > 0:
            summary_parts.append(f"{errors} errors")
        if skipped > 0:
            summary_parts.append(f"{skipped} skipped")

        summary = ", ".join(summary_parts) if summary_parts else "No tests collected"

        # Determine success (no failures, errors, or collection errors)
        success = (failures == 0 and errors == 0 and len(collection_errors) == 0)
        has_skips = skipped > 0

        # Add collection errors to results
        for error in collection_errors:
            test_results.append(('COLLECTION_ERROR', error[:100] + "..." if len(error) > 100 else error))

        return test_results, summary, success, has_skips

    except Exception as e:
        return [], f"XML parsing error: {e}", False, False


def run_pytest_on_directory(logger, category_name: str, test_dir: str) -> tuple[str, str]:
    """Run pytest on an entire directory of tests with per-test visibility using JUnit XML"""
    logger.header(f"{category_name} Tests")

    venv_python = get_venv_python_path()

    # Validate the venv
    if not validate_venv(logger):
        logger.log(f"ERROR: Skipping {category_name} tests - Virtual environment not available")
        return 'FAIL', 'Virtual environment not available'

    # Generate unique XML file path
    xml_file = Path(f"test_results_{category_name.lower()}_{datetime.now().strftime('%H%M%S')}.xml")

    try:
        # Run pytest with JUnit XML output
        result = subprocess.run(
            [str(venv_python), "-m", "pytest", test_dir, "--junitxml", str(xml_file), "--tb=no", "-q"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Parse XML results
        test_results, summary, xml_success, has_skips = parse_junit_xml(xml_file)

        # Display per-test results
        for status, test_name in test_results:
            if status == 'COLLECTION_ERROR':
                status_icon = '✗'
                logger.log(f"  {status_icon} COLLECTION ERROR: {test_name}")
            else:
                status_icon = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○', 'ERROR': '✗'}.get(status, '?')
                logger.log(f"  {status_icon} {status}: {test_name}")

        # Check for deprecation warnings
        warnings = extract_deprecation_warnings(result.stderr or "")
        if warnings:
            logger.log("DEPRECATION WARNINGS DETECTED:")
            for w in warnings:
                logger.log(f"  ⚠️ {w}")
        else:
            logger.log("No deprecation warnings detected.")

        # Display summary and determine status
        if xml_success and result.returncode == 0:
            if has_skips:
                status = 'PASS_WITH_SKIPS'
                logger.log(f"PASS WITH SKIPS: {category_name}: {summary}")
            else:
                status = 'PASS'
                logger.log(f"SUCCESS: {category_name}: {summary}")
            return status, summary
        else:
            logger.log(f"FAILED: {category_name}: {summary}")
            # Show stderr if available and not already captured in XML
            if result.stderr and not test_results:
                logger.log("--- STDERR ---")
                logger.log(result.stderr.strip())
            return 'FAIL', summary

    except subprocess.TimeoutExpired:
        logger.log(f"FAILED: {category_name}: Test execution timed out")
        return 'FAIL', 'Test execution timed out'
    except Exception as e:
        logger.log(f"FAILED: {category_name}: Unexpected error - {e}")
        return 'FAIL', f"Unexpected error - {e}"
    finally:
        # Clean up XML file
        if xml_file.exists():
            xml_file.unlink()


async def run_e2e_model_test(logger) -> str:
    """Run end-to-end model test using venv Python with per-test visibility"""
    logger.header("AI Intelligence Validation (E2E)")

    # Get the venv Python path
    venv_python = get_venv_python_path()

    # Validate the venv
    if not validate_venv(logger):
        logger.log("ERROR: Skipping AI Intelligence test - Virtual environment not available")
        return 'FAIL'

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

    # Check for deprecation warnings
    warnings = extract_deprecation_warnings(result.stderr or "")
    if warnings:
        logger.log("DEPRECATION WARNINGS DETECTED:")
        for w in warnings:
            logger.log(f"  ⚠️ {w}")
    else:
        logger.log("No deprecation warnings detected.")

    if result.returncode == 0:
        # Check if tests were skipped (expected skips for external models)
        if "SKIPPED" in result.stdout:
            logger.log("WARNING: AI Intelligence: SOME TESTS WERE SKIPPED (no models available)")
            logger.log("PASS WITH SKIPS: AI Intelligence: PASS WITH SKIPS (expected skips for external models)")
            return 'PASS_WITH_SKIPS'
        else:
            logger.log("SUCCESS: AI Intelligence: PASS")
            return 'PASS'
    else:
        logger.log("FAILED: AI Intelligence: FAIL")
        logger.log("--- STDOUT ---")
        logger.log(result.stdout)
        if result.stderr:
            logger.log("--- STDERR ---")
            logger.log(result.stderr)
        return 'FAIL'


def cleanup_old_reports(logger):
    """Remove validation reports older than 7 days to prevent accumulation"""
    report_dir = Path("reports")
    if not report_dir.exists():
        return

    now = datetime.now()
    cutoff = now - timedelta(days=7)
    removed_count = 0

    for report_file in report_dir.glob("backend_validation_report_*.txt"):
        try:
            # Extract timestamp from filename: backend_validation_report_YYYYMMDD_HHMMSS.txt
            filename = report_file.name
            timestamp_str = filename.replace("backend_validation_report_", "").replace(".txt", "")
            file_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            if file_datetime < cutoff:
                report_file.unlink()
                removed_count += 1
                logger.log(f"Cleaned up old report: {report_file.name}")
        except (ValueError, OSError) as e:
            # Skip files that don't match expected pattern or can't be deleted
            logger.log(f"Skipped cleanup for {report_file.name}: {e}")

    if removed_count > 0:
        logger.log(f"Report cleanup: Removed {removed_count} reports older than 7 days")
    else:
        logger.log("Report cleanup: No old reports to remove")


def main():
    """Main validation function"""
    logger = ValidationLogger()

    # Clean up old validation reports to prevent accumulation
    cleanup_old_reports(logger)

    # Check if venv is available before running tests
    venv_python = get_venv_python_path()
    if not venv_python.exists():
        logger.log(f"WARNING: Virtual environment not found at: {venv_python}")
        logger.log("Some tests may fail if dependencies are not installed globally.")
        logger.log("Recommended: Create venv with 'python -m venv backend/.venv'")
    else:
        logger.log(f"SUCCESS: Virtual environment found at: {venv_python}")

    # Run tests by category using dynamic discovery
    unit_status, _ = run_pytest_on_directory(logger, "Unit", "tests/unit")
    integration_status, _ = run_pytest_on_directory(logger, "Integration", "tests/integration")
    agentic_status, _ = run_pytest_on_directory(logger, "Agentic", "tests/agentic")

    # Run async E2E model tests
    e2e_status = asyncio.run(run_e2e_model_test(logger))

    # Summary
    logger.header("Backend Validation Summary")
    logger.log(f"Unit Tests:        {unit_status}")
    logger.log(f"Integration Tests: {integration_status}")
    logger.log(f"Agentic Tests:     {agentic_status}")
    logger.log(f"AI Intelligence:   {e2e_status}")
    logger.log("="*60)

    statuses = [unit_status, integration_status, agentic_status, e2e_status]
    has_any_fail = any(s == 'FAIL' for s in statuses)
    has_any_skips = any(s == 'PASS_WITH_SKIPS' for s in statuses)

    if not has_any_fail:
        if has_any_skips:
            logger.log("\n✅ JARVISv3 Backend is VALIDATED WITH EXPECTED SKIPS!")
        else:
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
