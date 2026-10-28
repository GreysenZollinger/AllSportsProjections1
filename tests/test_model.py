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
