import os
import sys
import types


# Ensure repository root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_run_dashboard_with_mocked_streamlit():
    import importlib
    main = importlib.import_module("main")

    # Build a minimal mock for Streamlit used APIs
    class MockStreamlit:
        def __init__(self):
            self.titles = []
            self.subheaders = []
            self.messages = []
            self.success_called = False

        def title(self, text):
            self.titles.append(text)

        def button(self, _label):
            # Simulate button click to exercise simulation path
            return True

        def success(self, _msg):
            self.success_called = True
            self.messages.append(_msg)

        def dataframe(self, _df):
            # accept any DataFrame
            pass

        def bar_chart(self, _df):
            # accept any DataFrame
            pass

        def subheader(self, text):
            self.subheaders.append(text)

        def write(self, _msg):
            self.messages.append(str(_msg))

    mock_st = MockStreamlit()

    # Swap the streamlit module inside main
    original_st = getattr(main, "st", None)
    try:
        main.st = mock_st  # type: ignore
        # Run the dashboard logic; should complete without exceptions
        main.run_dashboard()
    finally:
        if original_st is not None:
            main.st = original_st  # type: ignore

    # Validate that success was called and profits updated
    assert mock_st.success_called is True
    assert any(layer.get("actual_profit", 0) > 0 for layer in main.investment_tracker)


def test_simulate_all_layers_no_exceptions():
    import importlib
    main = importlib.import_module("main")

    for layer in main.investment_tracker:
        profit = main.simulate_layer(layer)
        assert isinstance(profit, (int, float))
        assert profit == layer["actual_profit"]
        assert profit >= 0

