from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _save_or_return(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig


def plot_loss_history(
    history: dict[str, list[float]] | Any,
    save_path: str | Path | None = None,
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    values = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(figsize=(8, 5))
    if "loss" in values:
        ax.plot(values["loss"], label="training loss")
    if "val_loss" in values:
        ax.plot(values["val_loss"], label="validation loss")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_return(fig, save_path)


def plot_metric_bar(
    records: list[dict[str, Any]] | pd.DataFrame,
    x: str,
    y: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    rotation: int = 45,
) -> plt.Figure:
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame[x].astype(str), frame[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(True, axis="y", alpha=0.3)
    return _save_or_return(fig, save_path)


def plot_metric_lines(
    records: list[dict[str, Any]] | pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    if hue is None:
        ax.plot(frame[x], frame[y], marker="o")
    else:
        for label, group in frame.groupby(hue):
            ax.plot(group[x], group[y], marker="o", label=str(label))
        ax.legend(title=hue)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, save_path)
