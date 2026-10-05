# conftest.py — project root
# Makes src/ importable during test collection without requiring an editable install.
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
