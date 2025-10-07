import sys
from pathlib import Path
import pytest

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from recommender_core import content_recommend, hybrid_recommend, get_product_choices


def test_content_recommend_no_artifacts():
    # In this workspace there may be no products.pkl; ensure function returns a list (possibly empty) and doesn't raise
    res = content_recommend('nonexistent-product', top_n=5)
    assert isinstance(res, list)


def test_hybrid_recommend_no_artifacts():
    res = hybrid_recommend(None, top_n=5)
    assert isinstance(res, list)


def test_get_product_choices_type():
    choices = get_product_choices(10)
    assert isinstance(choices, list)
