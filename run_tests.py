#!/usr/bin/env python3
"""Test runner script."""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests() -> int:
    """Discover and run all tests."""
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code based on success
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
