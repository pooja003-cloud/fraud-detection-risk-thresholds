"""Shared chart style so every figure in reports/figures looks like one system.

Palette: a colour-blind-validated categorical order (blue, orange, aqua, yellow,
magenta, green, violet, red). Colours are assigned by entity in a fixed order and
never cycled. Thin marks and recessive hairline grids let the data stand out.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#a3a29c"

# Fixed colour per model so the same model is the same colour in every chart.
MODEL_COLORS = {
    "Rules baseline": NEUTRAL,
    "Logistic regression": BLUE,
    "Random forest": AQUA,
    "LightGBM": ORANGE,
    "LightGBM (undersampled)": VIOLET,
}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig: plt.Figure, name: str) -> str:
    """Save to reports/figures/<name>.png and return the relative path."""
    path = config.FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return f"reports/figures/{name}.png"


def title(ax, finding: str, subtitle: str | None = None) -> None:
    """Title states the finding; optional grey subtitle states what is plotted."""
    ax.set_title(finding, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_2, fontsize=9, va="bottom")


def takeaway(text: str) -> None:
    """Display a one-line, data-driven takeaway under a chart."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"**Takeaway:** {text}"))
    except ImportError:  # pragma: no cover
        print(f"Takeaway: {text}")


def rate_bars(ax, labels, rates, overall=None, horizontal=False, highlight=None, fmt="{:.2f}%"):
    """Bar chart of a rate by segment, with the overall rate as a reference line."""
    labels = [str(l) for l in labels]
    colors = [ORANGE if (highlight is not None and highlight(l, r)) else BLUE for l, r in zip(labels, rates)]
    if horizontal:
        ax.barh(labels, rates, color=colors, height=0.6)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if overall is not None:
            ax.axvline(overall, color=TEXT_2, lw=1)
            ax.text(overall, -0.9, f" overall {fmt.format(overall)}", color=TEXT_2, fontsize=8, va="bottom")
    else:
        ax.bar(labels, rates, color=colors, width=0.6)
        ax.grid(axis="x", visible=False)
        if overall is not None:
            ax.axhline(overall, color=TEXT_2, lw=1)
            ax.text(len(labels) - 0.5, overall, f"overall {fmt.format(overall)}", color=TEXT_2,
                    fontsize=8, ha="right", va="bottom")
