import pytest
from recommender_utils import synthesize_title


def test_synthesize_generic_product():
    prod = {'name': 'Product 7', 'brand': 'BrandX', 'category': 'electronics', 'price': 12.5}
    out = synthesize_title(prod, pid='P7')
    assert 'BrandX' in out and '$12.50' in out


def test_synthesize_already_friendly():
    prod = {'name': 'SuperWidget', 'brand': 'BrandX', 'price': 99}
    out = synthesize_title(prod, pid='P9')
    assert out == 'SuperWidget'
