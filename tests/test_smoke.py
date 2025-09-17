import os
import sys
import importlib


# Ensure repository root is on sys.path so `main.py` is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_import_main_module():
    module = importlib.import_module("main")
    assert hasattr(module, "run_dashboard")


def test_generate_ai_text_demo_mode():
    module = importlib.import_module("main")
    text = module.generate_ai_text("hello world")
    assert isinstance(text, str)
    assert "AI Generated Content" in text

