"""Matplotlib charts written to assets/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1f2933"
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#a0aec0"
GREEN = "#2f855a"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight",
})


def model_comparison(cv: pd.DataFrame, shipped: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cv = cv.sort_values("cv_log_loss", ascending=False)
    colors = [ORANGE if m == shipped else BLUE for m in cv["model"]]
    ax.barh(cv["model"], cv["cv_log_loss"], xerr=cv["cv_se"], color=colors, capsize=3)
    lo = cv["cv_log_loss"].min() - 0.02
    ax.set_xlim(lo, cv["cv_log_loss"].max() + 0.01)
    ax.set_xlabel("Cross-validated log loss on training matches (lower is better)")
    ax.set_title("Model comparison, orange is the shipped model", loc="left")
    fig.savefig(path)
    plt.close(fig)


def calibration(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 0.6], [0, 0.6], color=GREY, ls="--", lw=1)
    colors = [BLUE, ORANGE, GREEN]
    for (label, t), c in zip(tables.items(), colors, strict=False):
        ax.plot(t["mean_pred"], t["observed"], marker="o", ms=4, color=c, label=label)
    ax.set_xlabel("Predicted xG (decile mean)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("Calibration on shots from matches the model never saw", loc="left")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def pitch_map(grid_xg: np.ndarray, xs: np.ndarray, ys: np.ndarray, path: Path) -> None:
    from mplsoccer import VerticalPitch

    pitch = VerticalPitch(half=True, pitch_color="white", line_color=GREY, pad_bottom=0.5)
    fig, ax = pitch.draw(figsize=(6.2, 5.6))
    mesh = ax.pcolormesh(ys, xs, grid_xg, cmap="YlOrRd", vmin=0, vmax=0.5, alpha=0.9, zorder=0.5)
    pitch.draw(ax=ax)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="xG")
    ax.set_title("xG of a right-footed open-play shot after regular play", loc="left", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def shrinkage(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(table["xg"], table["raw_ratio"], s=14, color=GREY, label="raw goals / xG")
    ax.scatter(table["xg"], table["odds_multiplier"], s=14, color=BLUE, label="shrunk estimate")
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 3)
    ax.set_xlabel("Season xG (log scale)")
    ax.set_ylabel("Finishing multiplier")
    ax.set_title("Few shots, wild ratios: shrinkage pulls them back", loc="left")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def split_half(both: pd.DataFrame, r_raw: float, r_shrunk: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, col, r, title in zip(axes, ["raw_a", "effect"], [r_raw, r_shrunk],
                                 ["Raw goals minus xG per shot", "Shrunk effect"], strict=True):
        ax.scatter(both[col], both["excess_b"], s=14, color=BLUE, alpha=0.7)
        ax.set_title(f"{title}, r = {r:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("Estimated from first half of matches")
        ax.set_ylabel("Goals minus xG per shot, second half")
    fig.suptitle("Does a player's finishing in one half predict the other?", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def transfer(rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, rows["xg_total"], 0.4, color=BLUE, label="Model xG")
    ax.bar(x + 0.2, rows["goals"], 0.4, color=ORANGE, label="Goals scored")
    ax.set_xticks(x, rows["set"])
    ax.set_title("Totals on data from other leagues and competitions", loc="left")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)
