#!/usr/bin/env python
"""
Entry point for the Crowd Management System.
Run this file from the root directory.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crowd_counter import main

if __name__ == "__main__":
    main()
