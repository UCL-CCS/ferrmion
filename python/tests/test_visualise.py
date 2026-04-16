"""Tests for visualisation functions."""

import sys

import pytest

try:
    import matplotlib
    import matplotlib.figure
    from ferrmion.visualise import symplectic_matshow
except ImportError:
    matplotlib = None

from ferrmion.encode.ternary_tree import JordanWigner

pytestmark = pytest.mark.skipif(
    matplotlib is None, reason="Extra dependency `ferrmion[viz]` not installed."
)


def test_symplectic_matshow_returns_figure():
    import matplotlib.pyplot as plt

    enc = JordanWigner(4)
    _, symplectics = enc._build_symplectic_matrix()
    fig = symplectic_matshow(symplectics)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_symplectic_matshow_with_title():
    import matplotlib.pyplot as plt

    enc = JordanWigner(4)
    _, symplectics = enc._build_symplectic_matrix()
    fig = symplectic_matshow(symplectics, title="Test Title")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_visualise_operators_returns_figure():
    import matplotlib.pyplot as plt

    enc = JordanWigner(4)
    fig = enc.visualise_operators()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_visualise_operators_raises_without_viz(monkeypatch):
    """ImportError with install hint when viz extra is absent."""
    monkeypatch.setitem(sys.modules, "ferrmion.visualise", None)
    enc = JordanWigner(4)
    with pytest.raises(ImportError, match="ferrmion\\[viz\\]"):
        enc.visualise_operators()
