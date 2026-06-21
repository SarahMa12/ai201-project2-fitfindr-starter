"""
Pytest tests for all three FitFindr tools.

Run with:
    pytest tests/test_tools.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0

def test_search_size_mismatch():
    results = search_listings("vintage", size="XXXL", max_price=None)
    assert results == []

def test_search_price_out_of_bounds():
    results = search_listings("jacket", size=None, max_price=3.00)
    assert results == []

def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)

def test_search_no_listings_found():
    results = search_listings("xyznotarealitem", size=None, max_price=None)
    assert results == []


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def test_suggest_outfit_populated_wardrobe():
    new_item = search_listings("vintage graphic tee")[0]
    wardrobe = get_example_wardrobe()
    result = suggest_outfit(new_item=new_item, wardrobe=wardrobe)
    assert isinstance(result, str)
    assert len(result) > 0

def test_suggest_outfit_empty_wardrobe():
    new_item = search_listings("vintage graphic tee")[0]
    empty_wardrobe = get_empty_wardrobe()
    result = suggest_outfit(new_item=new_item, wardrobe=empty_wardrobe)
    assert isinstance(result, str)
    assert len(result) > 0

def test_suggest_outfit_no_new_item():
    wardrobe = get_example_wardrobe()
    result = suggest_outfit(new_item=None, wardrobe=wardrobe)
    assert isinstance(result, str)
    assert "no item was provided" in result.lower()


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def test_create_fit_card_valid():
    new_item = search_listings("vintage graphic tee")[0]
    wardrobe = get_example_wardrobe()
    outfit = suggest_outfit(new_item=new_item, wardrobe=wardrobe)
    fit_card = create_fit_card(outfit=outfit, new_item=new_item)
    assert isinstance(fit_card, str)
    assert len(fit_card) > 0

def test_create_fit_card_empty_outfit():
    new_item = search_listings("vintage graphic tee")[0]
    fit_card = create_fit_card(outfit="", new_item=new_item)
    assert isinstance(fit_card, str)
    assert "no outfit suggestion was provided" in fit_card.lower()
