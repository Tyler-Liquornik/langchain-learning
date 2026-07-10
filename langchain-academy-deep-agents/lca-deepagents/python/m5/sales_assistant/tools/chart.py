# python/m5/tools/chart.py
"""Render a chart from structured data — no code execution required.

A fixed-purpose tool instead of a code-execution sandbox: the model supplies
a title and two parallel lists (never code), so there is no execution
surface to secure in the first place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from langchain_core.tools import tool

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


@tool
def render_pie_chart(title: str, labels: list[str], values: list[float], filename: str) -> str:
    """Render a labeled pie chart and save it as a PNG in outputs/.

    Args:
        title: Chart title.
        labels: One label per slice (e.g. genre names).
        values: One numeric value per label (e.g. revenue), same length as labels.
        filename: Output filename, e.g. "territory_chart.png". Any directory
            components are stripped — the file is always saved into outputs/.
    """
    if len(labels) != len(values):
        return "Error: labels and values must be the same length."

    # Strip any directory components so a crafted filename can't write outside outputs/.
    safe_name = Path(filename).name
    if not safe_name:
        return "Error: filename must not be empty."

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.0f%%")
    ax.set_title(title)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / safe_name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return f"Saved to outputs/{safe_name}"
