#!/usr/bin/env python3
"""
Test runner for Bitcoin Options Analysis unit tests.

This script runs all unit tests and provides a summary of results.
Use this to verify that changes to the codebase don't break functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all unit tests and display results."""
    print("🧪 Bitcoin Options Analysis - Unit Test Runner")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run pytest with verbose output and coverage if available
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_bitcoin_options_analysis.py",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--no-header",  # Remove pytest header
    ]
    
    # Try to add coverage if pytest-cov is available
    try:
        import pytest_cov
        cmd.extend([
            "--cov=btc_options",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("📊 Running tests with coverage analysis...")
    except ImportError:
        print("📊 Running tests (install pytest-cov for coverage analysis)...")
    
    print()
    
    # Run the tests
    result = subprocess.run(cmd, capture_output=False)
    
    print("\n" + "=" * 60)
    
    if result.returncode == 0:
        print("✅ All tests passed successfully!")
        print("🎉 Your code changes haven't broken any existing functionality.")
    else:
        print("❌ Some tests failed.")
        print("🔧 Please review the test output above and fix any issues.")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())