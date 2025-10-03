#!/usr/bin/env python3
"""
Development Workflow Helper

This script provides a structured approach to making code changes while
maintaining test coverage and preventing regressions.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

class DevelopmentWorkflow:
    """Helper class for safe code development with regression protection."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.baseline_file = self.project_root / ".test_baseline.json"
        
    def establish_baseline(self):
        """Run tests and save current results as baseline."""
        print("🧪 Establishing test baseline...")
        
        # Use the virtual environment's Python if available
        python_cmd = self._get_python_command()
        
        # Run tests and capture results
        result = subprocess.run([
            python_cmd, "-m", "pytest", 
            "tests/test_bitcoin_options_analysis.py",
            "--tb=short", "-v"
        ], capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            print("❌ Cannot establish baseline - tests are currently failing!")
            print(result.stdout)
            print(result.stderr)
            return False
            
        # Save baseline
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'test_count': self._count_tests(),
            'all_passed': True,
            'returncode': result.returncode
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
            
        print(f"✅ Baseline established: {baseline['test_count']} tests passing")
        return True
    
    def _get_python_command(self):
        """Get the appropriate Python command (preferring virtual environment)."""
        venv_python = self.project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        
        venv_python_alt = self.project_root / "venv" / "bin" / "python"
        if venv_python_alt.exists():
            return str(venv_python_alt)
        
        return sys.executable
    
    def check_regression(self):
        """Check if current code has any regressions compared to baseline."""
        if not self.baseline_file.exists():
            print("⚠️ No baseline found. Establishing baseline first...")
            return self.establish_baseline()
            
        print("🔍 Checking for regressions...")
        
        # Use the virtual environment's Python if available
        python_cmd = self._get_python_command()
        
        # Run current tests
        result = subprocess.run([
            python_cmd, "-m", "pytest", 
            "tests/test_bitcoin_options_analysis.py",
            "--tb=short", "-v"
        ], capture_output=True, text=True, cwd=self.project_root)
        
        # Load baseline
        with open(self.baseline_file, 'r') as f:
            baseline = json.load(f)
            
        current_test_count = self._count_tests()
        
        print(f"📊 Test Results:")
        print(f"   Baseline: {baseline['test_count']} tests")
        print(f"   Current:  {current_test_count} tests")
        
        if result.returncode == 0:
            print("✅ All tests passing - no regressions detected!")
            
            if current_test_count > baseline['test_count']:
                print(f"🎉 Added {current_test_count - baseline['test_count']} new tests!")
                
            return True
        else:
            print("❌ Regression detected!")
            print("\n📋 Test Output:")
            print(result.stdout)
            if result.stderr:
                print("\n🔴 Error Output:")
                print(result.stderr)
            return False
    
    def _count_tests(self):
        """Count the number of tests in the test suite."""
        python_cmd = self._get_python_command()
        
        result = subprocess.run([
            python_cmd, "-m", "pytest", 
            "tests/test_bitcoin_options_analysis.py",
            "--collect-only", "-q"
        ], capture_output=True, text=True, cwd=self.project_root)
        
        # Count test functions by looking for "test_" in output
        test_count = result.stdout.count('::test_')
        return test_count if test_count > 0 else 7  # fallback to known count

def print_development_guidelines():
    """Print best practices for safe development."""
    print("""
🛡️ SAFE DEVELOPMENT WORKFLOW
============================

1. 📋 BEFORE MAKING CHANGES:
   python dev_workflow.py baseline    # Establish baseline
   
2. 🔧 MAKE YOUR CHANGES:
   - Edit code incrementally
   - Test frequently during development
   
3. 🧪 AFTER MAKING CHANGES:
   python dev_workflow.py check       # Check for regressions
   
4. ✅ IF TESTS PASS:
   - Commit your changes
   - Update baseline if needed
   
5. ❌ IF TESTS FAIL:
   - Review failed tests
   - Fix issues or update tests if intentional
   - Re-run checks

🎯 TYPES OF CHANGES & TEST STRATEGY:
===================================

🟢 SAFE CHANGES (shouldn't break tests):
   • Code refactoring without logic changes
   • Adding new optional parameters with defaults
   • Performance improvements
   • Documentation updates
   • Adding new methods/classes

🟡 MODERATE CHANGES (may need test updates):
   • Changing parameter ranges/constraints
   • Modifying optimization algorithms
   • Adding new validation logic
   • Changing error handling

🔴 BREAKING CHANGES (will require test updates):
   • Changing mathematical formulas
   • Modifying core algorithm logic
   • Changing API contracts
   • Altering expected output formats

📝 WHEN TO UPDATE TESTS:
=======================

✅ UPDATE tests when:
   • You intentionally change algorithm behavior
   • You modify mathematical formulas
   • You change expected output ranges
   • You add new features that need coverage

❌ DON'T UPDATE tests for:
   • Minor refactoring
   • Performance optimizations
   • Code formatting changes
   • Documentation updates
""")

def main():
    """Main entry point for development workflow helper."""
    if len(sys.argv) < 2:
        print("🚀 Development Workflow Helper")
        print("Usage:")
        print("  python dev_workflow.py baseline  # Establish test baseline")
        print("  python dev_workflow.py check     # Check for regressions")
        print("  python dev_workflow.py guide     # Show development guidelines")
        return
        
    command = sys.argv[1].lower()
    workflow = DevelopmentWorkflow()
    
    if command == "baseline":
        success = workflow.establish_baseline()
        if success:
            print("\n🎯 Next steps:")
            print("   1. Make your code changes")
            print("   2. Run: python dev_workflow.py check")
            
    elif command == "check":
        success = workflow.check_regression()
        if success:
            print("\n🎉 Safe to commit your changes!")
        else:
            print("\n🔧 Please fix failing tests before committing")
            
    elif command == "guide":
        print_development_guidelines()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: baseline, check, guide")

if __name__ == "__main__":
    main()