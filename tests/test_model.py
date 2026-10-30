"""Small tests for the dashboard's core probability calculation."""

import ast
from pathlib import Path


def load_expected_score():
    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "expected_score"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["expected_score"]


expected_score = load_expected_score()


def test_equal_ratings_give_even_probability():
    assert expected_score(1500, 1500) == 0.5


def test_higher_rating_has_higher_expected_score():
    assert expected_score(1600, 1500) > 0.5
    assert expected_score(1400, 1500) < 0.5


def test_opponent_probabilities_sum_to_one():
    a = expected_score(1575, 1490)
    b = expected_score(1490, 1575)
    assert abs((a + b) - 1.0) < 1e-12


def test_probability_stays_between_zero_and_one():
    probability = expected_score(1800, 1200)
    assert 0 < probability < 1

def load_backtest_model():
    import math
    import pandas as pd

    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"expected_score", "backtest_model"}
    ]
    module = ast.Module(body=function_nodes, type_ignores=[])
    namespace = {
        "math": math,
        "pd": pd,
        "SPORTS": {
            "NFL": {"home_adv": 55, "k": 24},
            "NBA": {"home_adv": 65, "k": 20},
            "MLB": {"home_adv": 45, "k": 18},
            "NHL": {"home_adv": 50, "k": 20},
        },
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["backtest_model"]


backtest_model = load_backtest_model()


def test_backtest_uses_warmup_then_scores_later_games_only():
    history = []
    for index in range(10):
        history.append({
            "Date": f"2026-01-{index + 1:02d}",
            "Away Team": "Away",
            "Home Team": "Home",
            "Away Score": 1 if index % 2 == 0 else 0,
            "Home Score": 0 if index % 2 == 0 else 1,
        })

    results, metrics = backtest_model("NFL", history)

    assert metrics["training_games"] == 3
    assert metrics["games"] == 7
    assert len(results) == 7
    assert "Home Win Probability (%)" in results.columns
    assert 0 <= metrics["accuracy"] <= 100
    assert 0 <= metrics["baseline_accuracy"] <= 100
    assert 0 <= metrics["brier"] <= 1


def test_backtest_handles_too_little_history():
    results, metrics = backtest_model("NFL", [{
        "Date": "2026-01-01",
        "Away Team": "Away",
        "Home Team": "Home",
        "Away Score": 0,
        "Home Score": 1,
    }])

    assert results.empty
    assert metrics["games"] == 0
    assert metrics["accuracy"] is None



def load_score_backtest():
    import math
    import pandas as pd

    source_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_project_scores", "backtest_score_model"}
    ]
    module = ast.Module(body=function_nodes, type_ignores=[])
    namespace = {"math": math, "pd": pd}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["backtest_score_model"]


backtest_score_model = load_score_backtest()


def test_score_backtest_is_walk_forward_and_reports_error_metrics():
    history = []
    for index in range(12):
        history.append({
            "Date": f"2026-01-{index + 1:02d}",
            "Away Team": "Away",
            "Home Team": "Home",
            "Away Score": 10 + index,
            "Home Score": 14 + index,
        })

    results, metrics = backtest_score_model(history)

    assert metrics["training_games"] == 3
    assert metrics["games"] == 9
    assert len(results) == 9
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["home_mae"] >= 0
    assert metrics["away_mae"] >= 0


def test_score_backtest_handles_empty_history():
    results, metrics = backtest_score_model([])
    assert results.empty
    assert metrics["games"] == 0
    assert metrics["mae"] is None
