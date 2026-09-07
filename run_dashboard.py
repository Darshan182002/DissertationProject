"""
Launcher for the Streamlit dashboard.

Run from project root with:
    streamlit run run_dashboard.py
"""

import os
import sys

# Ensure project root is in path so 'from src.dashboard.app import' works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import triggers the Streamlit page rendering
from src.dashboard import app