import pytest
import ferrmion


def test_sum_as_string():
    assert ferrmion.sum_as_string(1, 1) == "2"
