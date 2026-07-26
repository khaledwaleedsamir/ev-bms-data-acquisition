"""
visualization.py
=================

Shared plotting helpers used by every analysis module. Centralizing this
here means:
  - every figure gets saved the same way (consistent naming, consistent DPI)
  - swapping the plotting backend (e.g. disabling Plotly) is a one-file change
  - analysis modules stay focused on *computing* things, not on plot styling

Two families of helpers:
  - interactive (Plotly): time series, scatter, histograms - anything you'd
    want to zoom/pan/hover on while exploring the data.
  - static (Matplotlib/Seaborn): correlation heatmaps, pair plots, and the
    final "publication-quality" dashboard figures meant for a report/thesis.

Plotly is optional - if it isn't installed, `interactive=True` calls
transparently fall back to the Matplotlib equivalent so the rest of the
pipeline keeps working.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd

from . import config

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without plotly
    PLOTLY_AVAILABLE = False

try:
    plt.style.use(config.PLOT_STYLE)
except (OSError, ValueError):
    pass  # style name not available in this matplotlib version - use default


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_matplotlib_fig(fig, outdir: str, name: str) -> str:
    ensure_outdir(outdir)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_plotly_fig(fig, outdir: str, name: str) -> str:
    ensure_outdir(outdir)
    path = os.path.join(outdir, f"{name}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, y_cols: list[str], outdir: str, name: str,
                     time_col: str = "elapsed_s", color_col: str = "run_name",
                     title: str = "", interactive: bool = True) -> str:
    """
    Plot one or more signals against time. With multiple y_cols, each gets
    its own stacked subplot (shared x-axis) so signals with different units
    (e.g. voltage in V, current in A) stay readable on their own scale.
    """
    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        return ""

    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure()
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=len(y_cols), cols=1, shared_xaxes=True,
                             subplot_titles=y_cols, vertical_spacing=0.05)
        runs = df[color_col].unique() if color_col in df.columns else [None]
        colors = px.colors.qualitative.Plotly

        for row, y in enumerate(y_cols, start=1):
            for i, run in enumerate(runs):
                sub = df[df[color_col] == run] if run is not None else df
                fig.add_trace(
                    go.Scattergl(
                        x=sub[time_col], y=sub[y], mode="lines", name=str(run),
                        legendgroup=str(run), showlegend=(row == 1),
                        line=dict(color=colors[i % len(colors)], width=1),
                    ),
                    row=row, col=1,
                )
        fig.update_layout(title=title or "Time series", height=280 * len(y_cols),
                           template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    # Matplotlib fallback: one subplot per signal.
    fig, axes = plt.subplots(len(y_cols), 1, sharex=True, figsize=(11, 3 * len(y_cols)))
    axes = [axes] if len(y_cols) == 1 else axes
    for ax, y in zip(axes, y_cols):
        if color_col in df.columns:
            for run, sub in df.groupby(color_col):
                ax.plot(sub[time_col], sub[y], linewidth=0.8, label=str(run))
        else:
            ax.plot(df[time_col], df[y], linewidth=0.8)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(time_col)
    fig.suptitle(title or "Time series")
    return save_matplotlib_fig(fig, outdir, name)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame, col: str, outdir: str, name: str,
                    bins: int = 60, title: str = "", interactive: bool = True) -> str:
    if col not in df.columns:
        return ""
    if interactive and PLOTLY_AVAILABLE:
        fig = px.histogram(df, x=col, nbins=bins, title=title or f"{col} histogram",
                            template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=bins, color="steelblue", edgecolor="white")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title or f"{col} histogram")
    return save_matplotlib_fig(fig, outdir, name)


def plot_box(df: pd.DataFrame, col: str, outdir: str, name: str, by: str | None = None,
             title: str = "", interactive: bool = True) -> str:
    if col not in df.columns:
        return ""
    if interactive and PLOTLY_AVAILABLE:
        fig = px.box(df, x=by, y=col, title=title or f"{col} box plot", template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    fig, ax = plt.subplots(figsize=(8, 4))
    if by and by in df.columns:
        df.boxplot(column=col, by=by, ax=ax, rot=45)
    else:
        ax.boxplot(df[col].dropna())
    ax.set_title(title or f"{col} box plot")
    return save_matplotlib_fig(fig, outdir, name)


def plot_violin(df: pd.DataFrame, col: str, outdir: str, name: str, by: str | None = None,
                 title: str = "", interactive: bool = True) -> str:
    if col not in df.columns:
        return ""
    if interactive and PLOTLY_AVAILABLE:
        fig = px.violin(df, x=by, y=col, box=True, title=title or f"{col} violin plot",
                         template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=df, x=by, y=col, ax=ax)
    ax.set_title(title or f"{col} violin plot")
    return save_matplotlib_fig(fig, outdir, name)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------
def plot_scatter(df: pd.DataFrame, x: str, y: str, outdir: str, name: str,
                  color: str | None = None, title: str = "", interactive: bool = True) -> str:
    if x not in df.columns or y not in df.columns:
        return ""
    if interactive and PLOTLY_AVAILABLE:
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None,
                          opacity=0.5, title=title or f"{y} vs {x}", template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    fig, ax = plt.subplots(figsize=(7, 5))
    if color and color in df.columns:
        for val, sub in df.groupby(color):
            ax.scatter(sub[x], sub[y], s=4, alpha=0.4, label=str(val))
        ax.legend(fontsize=7, markerscale=3)
    else:
        ax.scatter(df[x], df[y], s=4, alpha=0.4)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    return save_matplotlib_fig(fig, outdir, name)


# ---------------------------------------------------------------------------
# Correlation / multi-variable
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(corr: pd.DataFrame, outdir: str, name: str, title: str = "") -> str:
    """Always static (Matplotlib/Seaborn) - a heatmap is a "final figure", not something to explore."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(0.5 * len(corr.columns) + 3, 0.5 * len(corr.columns) + 2))
    sns.heatmap(corr, annot=len(corr.columns) <= 15, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    return save_matplotlib_fig(fig, outdir, name)


def plot_pairplot(df: pd.DataFrame, columns: list[str], outdir: str, name: str,
                   color: str | None = None, sample: int = 3000) -> str:
    """Static scatter-matrix over a small set of key columns (kept small on purpose - see config.MAX_PAIRPLOT_COLUMNS)."""
    import seaborn as sns

    columns = [c for c in columns if c in df.columns][: config.MAX_PAIRPLOT_COLUMNS]
    if len(columns) < 2:
        return ""
    plot_df = df[columns + ([color] if color and color in df.columns else [])]
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    grid = sns.pairplot(plot_df, hue=color if color in plot_df.columns else None,
                         diag_kind="kde", plot_kws={"s": 6, "alpha": 0.4})
    grid.figure.suptitle("Pair plot", y=1.02)
    ensure_outdir(outdir)
    path = os.path.join(outdir, f"{name}.png")
    grid.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(grid.figure)
    return path


def plot_line(x, y, outdir: str, name: str, xlabel: str = "", ylabel: str = "",
              title: str = "", color_values=None, interactive: bool = True) -> str:
    """Generic x/y line or scatter plot for one-off derived series (e.g. resistance over time)."""
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=4, opacity=0.6,
                                      color=color_values, colorscale="Viridis",
                                      showscale=color_values is not None)))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                           template="plotly_white")
        return save_plotly_fig(fig, outdir, name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(x, y, s=6, c=color_values, cmap="viridis", alpha=0.6)
    if color_values is not None:
        fig.colorbar(sc, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save_matplotlib_fig(fig, outdir, name)
