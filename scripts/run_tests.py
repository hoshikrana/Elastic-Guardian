#!/usr/bin/env python3
"""
EGX Testing Phase Coordinator

Orchestrates comprehensive testing across all framework components.
Supports unit, integration, GPU validation, and end-to-end training tests.

Usage:
    python run_tests.py --suite full              # All tests
    python run_tests.py --suite unit              # Unit tests only
    python run_tests.py --suite integration       # Integration pipelines
    python run_tests.py --suite gpu-validation    # GPU tests (requires GPU)
    python run_tests.py --suite monitoring        # Monitoring/metrics
    python run_tests.py --suite e2e               # End-to-end training
    python run_tests.py --report coverage         # Generate coverage report
"""

import argparse
import subprocess
import sys
from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Suite Definitions
# ============================================================================

TEST_SUITES = {
    "unit": {
        "description": "Unit tests (fast, no GPU needed)",
        "command": "pytest tests/unit/ -v --tb=short",
        "timeout": 300,
        "requires_gpu": False,
    },
    "integration": {
        "description": "Integration test pipelines",
        "command": "pytest tests/integration/ -v --tb=short -k 'not large_models'",
        "timeout": 600,
        "requires_gpu": False,
    },
    "large-models": {
        "description": "Large model integration tests (memory estimation, recovery)",
        "command": "pytest tests/integration/test_large_models.py -v --tb=short",
        "timeout": 300,
        "requires_gpu": False,
    },
    "gpu-validation": {
        "description": "GPU validation tests (requires NVIDIA GPU)",
        "command": "pytest tests/gpu_validation/ -v --tb=short",
        "timeout": 1200,
        "requires_gpu": True,
    },
    "monitoring": {
        "description": "Monitoring and metrics tests",
        "command": "pytest tests/unit/monitoring/ -v --tb=short",
        "timeout": 300,
        "requires_gpu": False,
    },
    "e2e": {
        "description": "End-to-end training tests",
        "command": "pytest tests/integration/test_lifecycle.py -v --tb=short -s",
        "timeout": 900,
        "requires_gpu": False,
    },
    "full": {
        "description": "All tests (unit + integration + large-models)",
        "command": "pytest tests/ -v --tb=short -k 'not gpu_validation' --co -q",
        "timeout": 2400,
        "requires_gpu": False,
    },
}


# ============================================================================
# Test Runner
# ============================================================================


def run_test_suite(suite_name: str, verbose: bool = False) -> Dict[str, any]:
    """
    Run a test suite and return results.

    Returns: Dictionary with test results
    """
    if suite_name not in TEST_SUITES:
        logger.error(f"Unknown test suite: {suite_name}")
        logger.info(f"Available suites: {', '.join(TEST_SUITES.keys())}")
        return {"success": False, "error": "Unknown suite"}

    suite = TEST_SUITES[suite_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {suite['description']}")
    logger.info(f"Command: {suite['command']}")
    logger.info(f"{'='*60}\n")

    try:
        result = subprocess.run(
            suite["command"],
            shell=True,
            timeout=suite["timeout"],
            capture_output=not verbose,
            text=True,
        )

        success = result.returncode == 0

        return {
            "success": success,
            "suite": suite_name,
            "returncode": result.returncode,
            "timeout": suite["timeout"],
            "stdout": result.stdout if verbose else result.stdout[-500:],
            "stderr": (
                result.stderr
                if verbose
                else result.stderr[-500:] if result.stderr else ""
            ),
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Test suite '{suite_name}' timed out after {suite['timeout']}s")
        return {
            "success": False,
            "suite": suite_name,
            "error": "TIMEOUT",
            "timeout": suite["timeout"],
        }
    except Exception as e:
        logger.error(f"Error running test suite: {e}")
        return {
            "success": False,
            "suite": suite_name,
            "error": str(e),
        }


# ============================================================================
# Coverage Report Generation
# ============================================================================


def generate_coverage_report() -> Dict[str, any]:
    """Generate coverage report with detailed breakdown."""
    logger.info("\n" + "=" * 60)
    logger.info("Generating Coverage Report")
    logger.info("=" * 60 + "\n")

    try:
        cmd = (
            "pytest tests/unit/ tests/integration/ "
            "--cov=egx --cov-report=html --cov-report=term-missing "
            "-q"
        )

        result = subprocess.run(
            cmd,
            shell=True,
            timeout=1200,
            capture_output=True,
            text=True,
        )

        logger.info(result.stdout)

        return {
            "success": result.returncode == 0,
            "report_type": "coverage",
            "html_report": "htmlcov/index.html",
            "summary": result.stdout,
        }

    except Exception as e:
        logger.error(f"Error generating coverage report: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Test Plan Executor
# ============================================================================


def run_testing_plan() -> Dict[str, any]:
    """
    Execute comprehensive testing plan for new testing phase.

    Plan:
    1. Unit tests (baseline)
    2. Integration tests (pipelines)
    3. Large model tests (memory validation)
    4. Monitoring tests (if implemented)
    5. Coverage analysis
    """
    logger.info("\n" + "=" * 70)
    logger.info(" TESTING PHASE - COMPREHENSIVE TEST RUN")
    logger.info("=" * 70)

    suites_to_run = [
        "unit",
        "integration",
        "large-models",
        "monitoring",
    ]

    results = {}
    passed_count = 0
    failed_count = 0

    for suite in suites_to_run:
        logger.info(
            f"\n[{suites_to_run.index(suite) + 1}/{len(suites_to_run)}] Running {suite}..."
        )
        result = run_test_suite(suite, verbose=False)
        results[suite] = result

        if result.get("success"):
            passed_count += 1
            logger.info(f"✅ {suite} PASSED")
        else:
            failed_count += 1
            logger.error(f"❌ {suite} FAILED")

    # Generate coverage
    logger.info("\n[Final] Generating coverage report...")
    coverage_result = generate_coverage_report()
    results["coverage"] = coverage_result

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info(" TEST EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Passed: {passed_count}/{len(suites_to_run)}")
    logger.info(f"❌ Failed: {failed_count}/{len(suites_to_run)}")
    logger.info(f"Coverage Report: {coverage_result.get('html_report', 'N/A')}")
    logger.info("=" * 70 + "\n")

    return {
        "total_suites": len(suites_to_run),
        "passed": passed_count,
        "failed": failed_count,
        "suite_results": results,
        "overall_success": failed_count == 0,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="EGX Testing Phase Runner")
    parser.add_argument(
        "--suite",
        choices=list(TEST_SUITES.keys()),
        default="full",
        help="Test suite to run",
    )
    parser.add_argument(
        "--report",
        choices=["coverage", "results"],
        help="Generate a report",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Run comprehensive testing plan",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Include GPU validation tests",
    )

    args = parser.parse_args()

    try:
        if args.plan:
            # Run full testing plan
            result = run_testing_plan()
            exit_code = 0 if result["overall_success"] else 1

        elif args.report:
            if args.report == "coverage":
                result = generate_coverage_report()
                exit_code = 0 if result.get("success") else 1
            else:
                result = run_test_suite("full", verbose=args.verbose)
                exit_code = 0 if result.get("success") else 1

        else:
            # Run single suite
            result = run_test_suite(args.suite, verbose=args.verbose)
            exit_code = 0 if result.get("success") else 1

            # Print result
            if result.get("success"):
                logger.info(f"\n✅ {args.suite} tests PASSED")
            else:
                logger.error(f"\n❌ {args.suite} tests FAILED")
                if "error" in result:
                    logger.error(f"Error: {result['error']}")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
