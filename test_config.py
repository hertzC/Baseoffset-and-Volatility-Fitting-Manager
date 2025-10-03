"""
Advanced Testing Configuration

This module provides different test configurations for different types of changes,
allowing you to run appropriate tests based on what you're modifying.
"""

import pytest
import sys
from pathlib import Path

# Test markers for different categories
pytestmark = [
    pytest.mark.regression,  # All tests are regression tests by default
]

class TestCategories:
    """Test categories for different types of changes."""
    
    # Core algorithm tests - run for any math/fitting changes
    CORE_ALGORITHM = [
        "TestWLSRegression::test_wls_fitting_consistency",
        "TestNonlinearMinimization::test_nonlinear_fitting_consistency", 
        "TestRegressionValues::test_wls_baseline_values"
    ]
    
    # Data handling tests - run for data processing changes
    DATA_PROCESSING = [
        "TestSyntheticCreation::test_synthetic_creation_consistency"
    ]
    
    # Parameter management tests - run for configuration changes
    PARAMETER_MANAGEMENT = [
        "TestNonlinearMinimization::test_parameter_management",
        "TestNonlinearMinimization::test_results_management"
    ]
    
    # Edge case tests - run for robustness improvements
    EDGE_CASES = [
        "TestWLSRegression::test_wls_parameter_validation"
    ]
    
    # All tests
    ALL_TESTS = CORE_ALGORITHM + DATA_PROCESSING + PARAMETER_MANAGEMENT + EDGE_CASES

def run_targeted_tests(change_type="all"):
    """Run tests appropriate for the type of change being made."""
    
    test_mapping = {
        "algorithm": TestCategories.CORE_ALGORITHM,
        "data": TestCategories.DATA_PROCESSING, 
        "parameters": TestCategories.PARAMETER_MANAGEMENT,
        "edge_cases": TestCategories.EDGE_CASES,
        "all": TestCategories.ALL_TESTS
    }
    
    if change_type not in test_mapping:
        print(f"❌ Unknown change type: {change_type}")
        print(f"Available types: {list(test_mapping.keys())}")
        return False
        
    tests = test_mapping[change_type]
    print(f"🧪 Running {len(tests)} tests for '{change_type}' changes...")
    
    # Convert test names to pytest format
    test_args = []
    for test in tests:
        test_args.extend(["-k", test.replace("::", " and ")])
    
    # Run pytest with the selected tests
    exit_code = pytest.main([
        "tests/test_bitcoin_options_analysis.py",
        "-v",
        "--tb=short"
    ] + test_args)
    
    return exit_code == 0

def create_test_plan(changes_description):
    """Create a test plan based on description of changes."""
    changes = changes_description.lower()
    
    plan = []
    
    if any(word in changes for word in ["algorithm", "formula", "math", "regression", "optimization"]):
        plan.append("algorithm")
        
    if any(word in changes for word in ["data", "synthetic", "option", "processing"]):
        plan.append("data")
        
    if any(word in changes for word in ["parameter", "config", "setting"]):
        plan.append("parameters")
        
    if any(word in changes for word in ["edge", "error", "validation", "robustness"]):
        plan.append("edge_cases")
        
    if not plan:
        plan = ["all"]  # Default to all tests if unclear
        
    return plan

if __name__ == "__main__":
    if len(sys.argv) > 1:
        change_type = sys.argv[1]
        success = run_targeted_tests(change_type)
        sys.exit(0 if success else 1)
    else:
        print("🎯 Targeted Test Runner")
        print("Usage: python test_config.py <change_type>")
        print("")
        print("Change types:")
        print("  algorithm    - Core fitting algorithm changes")
        print("  data         - Data processing changes") 
        print("  parameters   - Parameter management changes")
        print("  edge_cases   - Edge case and validation changes")
        print("  all          - Run all tests (default)")
        print("")
        print("Example: python test_config.py algorithm")