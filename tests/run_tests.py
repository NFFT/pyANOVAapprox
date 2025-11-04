#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import sys


# Script to run all tests and return any failures
def run_tests():
    test_files = [
        #        "tests/wav_lsqr.py",
        "tests/cheb_fista.py",
        "tests/cheb_lsqr.py",
        "tests/per_fista.py",
        "tests/per_lsqr.py",
    ]

    for test_file in test_files:
        print("Testing ", test_file, "...")
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Test failed for ", test_file)
            print(result.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    run_tests()
