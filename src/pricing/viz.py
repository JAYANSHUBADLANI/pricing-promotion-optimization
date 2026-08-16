"""Figures for the write up. Every one is generated from produced output."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing import features as F
from pricing import schema
from pricing.config import Config, load_config

PALETTE = ["#2a4d69", "#4b86b4", "#adcbe3", "#63ace5", "#7e8d85", "#b5651d"]


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def figure_promo_states(frame: pd.DataFrame, path, dpi: int) -> None:
    counts = frame[F.PROMO_STATE].value_counts()
    depth = frame.groupby(F.PROMO_STATE)[F.DISCOUNT_DEPTH].mean()
    order = counts.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(order, counts[order].to_numpy(), color=PALETTE[1])
    _style(axes[0], "Store weeks by promotional state", "store weeks")
    axes[0].invert_yaxis()

    axes[1].barh(order, (depth[order] * 100).to_numpy(), color=PALETTE[0])
    _style(axes[1], "Mean discount depth by state", "percent off base price")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_elasticity(table: pd.DataFrame, path, dpi: int) -> None:
    ordered = table.sort_values("elasticity_mean").reset_index(drop=True)
    colours = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(table["category"].unique()))}
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(ordered))
    ax.hlines(y, ordered["hdi_low"], ordered["hdi_high"], color="#cccccc", linewidth=2)
    ax.scatter(
        ordered["elasticity_mean"], y, s=22,
        color=[colours[c] for c in ordered["category"]], zorder=3,
    )
    ax.axvline(-1.0, color="#b5651d", linestyle="--", linewidth=1)
    ax.text(-1.0, len(ordered) + 0.5, " unit elastic", color="#b5651d", fontsize=8)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["upc"].astype(str), fontsize=6)
    _style(ax, "Own price elasticity by product, posterior mean and 94 percent interval",
           "elasticity")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=v, label=k)
               for k, v in colours.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_lift_decomposition(table: pd.DataFrame, path, dpi: int) -> None:
    pivot = table.pivot_table(
        index="mechanic", columns="category",
        values=["price_lift_pct", "mechanic_lift_pct"], aggfunc="mean",
    )
    mechanics = pivot.index.tolist()
    categories = sorted(table["category"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(categories)
    for i, category in enumerate(categories):
        offset = (i - len(categories) / 2) * width + width / 2
        x = np.arange(len(mechanics)) + offset
        price = pivot[("price_lift_pct", category)].to_numpy()
        mech = pivot[("mechanic_lift_pct", category)].to_numpy()
        ax.bar(x, price, width, color=PALETTE[i % len(PALETTE)], label=f"{category}: price")
        ax.bar(x, mech, width, bottom=np.clip(price, 0, None),
               color=PALETTE[i % len(PALETTE)], alpha=0.45,
               label=f"{category}: merchandising")
    ax.set_xticks(np.arange(len(mechanics)))
    ax.set_xticklabels(mechanics, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    _style(ax, "Where promotional lift comes from: price cut versus merchandising",
           "", "percent lift")
    ax.legend(fontsize=6, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_cannibalisation(table: pd.DataFrame, path, dpi: int) -> None:
    ordered = table.sort_values("diversion_ratio")
    fig, ax = plt.subplots(figsize=(9, 4))
    colours = ["#b5651d" if v > 1 else PALETTE[1] for v in ordered["diversion_ratio"]]
    ax.barh(ordered["sub_category"], ordered["diversion_ratio"], color=colours)
    ax.axvline(1.0, color="#b5651d", linestyle="--", linewidth=1)
    ax.text(1.0, -0.7, " all gain taken from the shelf", color="#b5651d", fontsize=8)
    _style(ax, "Diversion ratio: share of promotional gain taken from neighbouring products",
           "diversion ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_backtest(report: dict, path, dpi: int) -> None:
    states = sorted(report["by_promo_state"].keys())
    model = [report["by_promo_state"][s]["model_log_rmse"] for s in states]
    base = [report["by_promo_state"][s]["baseline_log_rmse"] for s in states]
    x = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, base, 0.4, label="store product average", color=PALETTE[2])
    ax.bar(x + 0.2, model, 0.4, label="demand model", color=PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=8)
    _style(ax, "Out of sample error on held out weeks, by promotional state",
           "", "log scale RMSE, lower is better")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg: Config = load_config()
    cfg.ensure_dirs()
    figures = cfg.path("reporting", "figures_dir")
    tables = cfg.path("reporting", "tables_dir")
    dpi = int(cfg["reporting"]["figure_dpi"])

    frame = pd.read_parquet(cfg.path("data", "processed_dir") / "panel_features.parquet")
    figure_promo_states(frame, figures / "promotional_states.png", dpi)
    figure_elasticity(
        pd.read_csv(tables / "elasticity_by_product.csv"),
        figures / "elasticity_by_product.png", dpi,
    )
    figure_lift_decomposition(
        pd.read_csv(tables / "lift_decomposition.csv"),
        figures / "lift_decomposition.png", dpi,
    )
    figure_cannibalisation(
        pd.read_csv(tables / "cannibalisation.csv"),
        figures / "cannibalisation.png", dpi,
    )
    backtest_path = tables / "backtest.json"
    if backtest_path.exists():
        figure_backtest(
            json.loads(backtest_path.read_text(encoding="utf-8")),
            figures / "backtest_by_promo_state.png", dpi,
        )
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    main()
