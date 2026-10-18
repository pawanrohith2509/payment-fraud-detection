"""
CLI script to run the real-time inference showcase demonstration.
"""

import os
import sys

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from payment_fraud_detection.showcase import run_showcase


if __name__ == "__main__":
    run_showcase()
