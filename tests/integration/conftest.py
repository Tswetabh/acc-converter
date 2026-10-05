# tests/integration/conftest.py
# Ensures tests/integration is a proper pytest package and
# adds the src/ path so accent_converter is importable even when
# pytest is invoked directly against this subdirectory.

import sys
import os

# Add src/ to path (mirrors root conftest.py)
_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))
