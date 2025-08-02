#!/usr/bin/env python3
"""
Test runner script for MLX distributed inference system.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    else:
        print(f"✅ Passed: {description}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run MLX distributed inference tests")
    parser.add_argument("--suite", choices=["unit", "integration", "e2e", "performance", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--fast", action="store_true", 
                       help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--markers", type=str, 
                       help="Custom pytest markers (e.g., 'not slow')")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["uv", "run", "python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add markers
    markers = []
    
    if args.fast:
        markers.append("not slow")
    
    if args.markers:
        markers.append(args.markers)
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add test paths based on suite
    if args.suite == "unit":
        cmd.append("tests/unit")
    elif args.suite == "integration":
        cmd.append("tests/integration")
    elif args.suite == "e2e":
        cmd.append("tests/e2e")
    elif args.suite == "performance":
        cmd.append("tests/performance")
    else:  # all
        cmd.append("tests")
    
    # Run the tests
    success = run_command(cmd, f"{args.suite.upper()} Tests")
    
    if success:
        print(f"\n🎉 All {args.suite} tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n💥 Some {args.suite} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()