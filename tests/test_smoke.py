import importlib


def test_import_main_module():
    module = importlib.import_module("main")
    assert hasattr(module, "run_dashboard")


def test_generate_ai_text_demo_mode():
    module = importlib.import_module("main")
    text = module.generate_ai_text("hello world")
    assert isinstance(text, str)
    assert "AI Generated Content" in text

