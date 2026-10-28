"""Shared plotting style and the EDA figures saved to ``reports/figures/``.

Every figure follows one style: a headline title that states the insight, a
muted subtitle that says what is plotted, thin marks, hairline gridlines and
text in neutral ink (never in the series colour). Numbers come from
``src/eda.py`` so the web app shows exactly the same values.
"""

import logging
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src import config, eda
from src.load_data import missing_summary

logger = logging.getLogger(__name__)

# Validated categorical order (blue, orange) + neutral chrome, light theme.
COLORS: dict[str, str] = {
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "muted_series": "#c3c2b7",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
# Diverging scale for correlations: blue (negative) <- neutral gray -> red (positive).
DIVERGING: tuple[str, str, str] = ("#256abf", "#f0efec", "#e34948")
HOTEL_COLORS: dict[str, str] = {
    "City Hotel": COLORS["series_1"],
    "Resort Hotel": COLORS["series_2"],
}


def apply_style() -> None:
    """Set matplotlib defaults for every project figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["surface"],
            "axes.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["surface"],
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink_secondary"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": COLORS["ink_muted"],
            "ytick.color": COLORS["ink_muted"],
            "xtick.labelcolor": COLORS["ink_secondary"],
            "ytick.labelcolor": COLORS["ink_secondary"],
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
        }
    )


def new_figure(
    title: str, subtitle: str, ncols: int = 1, figsize: tuple[float, float] = (9, 4.8)
) -> tuple[Figure, list[Axes]]:
    """Create a figure with a left-aligned headline and muted subtitle.

    Args:
        title: Headline stating the insight.
        subtitle: What is plotted (measure, population).
        ncols: Number of side-by-side panels.
        figsize: Figure size in inches.

    Returns:
        The figure and a list of its axes.
    """
    apply_style()
    # Object-oriented Figure (no pyplot): no GUI backend or global state needed.
    fig = Figure(figsize=figsize)
    axes = fig.subplots(1, ncols, squeeze=False)
    fig.text(0.01, 0.98, title, fontsize=14, fontweight=600, va="top")
    fig.text(0.01, 0.915, subtitle, fontsize=10, color=COLORS["ink_secondary"], va="top")
    fig.subplots_adjust(top=0.8, left=0.12, right=0.97, bottom=0.12, wspace=0.35)
    return fig, list(axes[0])


def save_figure(fig: Figure, name: str, folder: Path = config.FIGURES_DIR) -> Path:
    """Save a figure as PNG.

    Args:
        fig: Figure to save.
        name: File name without extension.
        folder: Output folder.

    Returns:
        Path of the saved PNG.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    logger.info("Saved figure %s", path.name)
    return path


def _hbar(ax: Axes, values: pd.Series, color: str, fmt: str = "{:,.0f}") -> None:
    """Draw a horizontal bar chart (largest at top) with values at the bar tips.

    Args:
        ax: Target axes.
        values: Series indexed by label.
        color: Bar colour.
        fmt: Format string for the value labels.
    """
    labels = [str(label) for label in values.index]
    ax.barh(labels, values.to_numpy(), color=color, height=0.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for y, value in enumerate(values.to_numpy()):
        ax.annotate(
            fmt.format(value),
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )
    ax.margins(x=0.15)


def _reference_line(ax: Axes, value: float, label: str, horizontal: bool = False) -> None:
    """Draw a thin muted reference line (e.g. the overall average) with a label.

    Args:
        ax: Target axes.
        value: Position of the line on the value axis.
        label: Text shown next to the line.
        horizontal: Draw a horizontal line (vertical bars) instead of a vertical one.
    """
    style = {"color": COLORS["ink_muted"], "linewidth": 1}
    if horizontal:
        ax.axhline(value, **style)
        ax.annotate(
            label,
            (1, value),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLORS["ink_muted"],
        )
    else:
        ax.axvline(value, **style)
        ax.annotate(
            label,
            (value, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["ink_muted"],
        )


def _overall_cancel_rate(df: pd.DataFrame) -> float:
    """Return the overall cancellation rate in percent.

    Args:
        df: Bookings DataFrame with the target column.

    Returns:
        Cancellation rate (%).
    """
    return float(df[config.TARGET].mean() * 100)


# ---------------------------------------------------------------------------
# Data-quality figures (raw vs clean)
# ---------------------------------------------------------------------------
def plot_missing_values(raw: pd.DataFrame) -> Figure:
    """Bar chart of missing-value share per column in the raw data.

    Args:
        raw: Raw bookings DataFrame.

    Returns:
        The figure.
    """
    summary = missing_summary(raw)["missing_pct"]
    fig, (ax,) = new_figure(
        "The company column is almost empty; agent misses 1 in 7",
        'Share of missing values per column, raw data (after reading "NULL" as missing)',
        figsize=(9, 3.6),
    )
    _hbar(ax, summary, COLORS["series_1"], fmt="{:.2f}%")
    ax.set_xlabel("Missing (%)")
    return fig


def plot_cleaning_steps(step_report: pd.DataFrame) -> Figure:
    """Rows removed by each cleaning step that removed rows.

    Args:
        step_report: Report returned by ``clean_data``.

    Returns:
        The figure.
    """
    removed = step_report.loc[step_report["rows_removed"] > 0, "rows_removed"]
    fig, (ax,) = new_figure(
        "Exact duplicates account for almost every removed row",
        f"Rows removed per cleaning step (total {removed.sum():,} of "
        f"{int(step_report['rows_before'].iloc[0]):,})",
        figsize=(9, 3.4),
    )
    _hbar(ax, removed, COLORS["series_1"])
    ax.set_xlabel("Rows removed")
    return fig


def plot_duplicates_profile(raw: pd.DataFrame) -> Figure:
    """Compare removed duplicates with kept rows on three shares.

    Args:
        raw: Raw bookings DataFrame.

    Returns:
        The figure.
    """
    profile = eda.duplicates_profile(raw)
    metrics = {
        "cancel_rate_pct": "Canceled",
        "groups_segment_pct": "Groups segment",
        "non_refund_pct": "Non-refund deposit",
    }
    fig, (ax,) = new_figure(
        "Duplicates are mostly canceled group blocks",
        "Share of rows (%) - removed exact duplicates vs rows kept",
        figsize=(9, 3.8),
    )
    height = 0.36
    groups = [("Removed duplicates", COLORS["series_2"]), ("Kept rows", COLORS["series_1"])]
    for offset, (group, color) in zip((-height / 2, height / 2), groups, strict=True):
        values = profile.loc[group, list(metrics)].to_numpy(dtype=float)
        positions = [i + offset for i in range(len(metrics))]
        ax.barh(
            positions,
            values,
            height=height - 0.04,
            color=color,
            label=f"{group} ({int(profile.loc[group, 'rows']):,} rows)",
        )
        for y, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:.1f}%",
                (value, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=COLORS["ink_secondary"],
            )
    ax.set_yticks(range(len(metrics)), list(metrics.values()))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of rows (%)")
    ax.margins(x=0.12)
    ax.legend(loc="lower right")
    return fig


def plot_adr_before_after(raw: pd.DataFrame, clean: pd.DataFrame) -> Figure:
    """Side-by-side boxplots of ADR before and after cleaning.

    Args:
        raw: Raw bookings DataFrame.
        clean: Cleaned bookings DataFrame.

    Returns:
        The figure.
    """
    fig, axes = new_figure(
        "One 5,400 booking stretched the ADR scale; capping fixes it",
        "Average daily rate (EUR) - raw vs cleaned (negative removed, capped at 99.9th pct)",
        ncols=2,
        figsize=(10, 3.4),
    )
    for ax, data, label in zip(axes, (raw, clean), ("Raw", "Cleaned"), strict=True):
        ax.boxplot(
            data["adr"],
            orientation="horizontal",
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": COLORS["grid"], "edgecolor": COLORS["ink_muted"]},
            medianprops={"color": COLORS["series_1"], "linewidth": 2},
            whiskerprops={"color": COLORS["ink_muted"]},
            capprops={"color": COLORS["ink_muted"]},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": COLORS["ink_muted"],
                "markeredgecolor": "none",
                "alpha": 0.4,
            },
        )
        ax.set_yticks([1], [label])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ADR (EUR)")
        max_value = data["adr"].max()
        ax.set_title(
            f"max {max_value:,.0f}  |  min {data['adr'].min():,.2f}",
            loc="left",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )
    return fig


# ---------------------------------------------------------------------------
# Insight figures (cleaned + engineered data)
# ---------------------------------------------------------------------------
def plot_cancellation_by_lead_time(df: pd.DataFrame) -> Figure:
    """Cancellation rate per lead-time group.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        The figure.
    """
    table = eda.cancellation_by_lead_time(df)
    fig, (ax,) = new_figure(
        "The earlier the booking, the more likely it is canceled",
        "Cancellation rate (%) by lead time (days between booking and arrival)",
    )
    ax.bar(table.index.astype(str), table["cancel_rate_pct"], color=COLORS["series_1"], width=0.6)
    for x, value in enumerate(table["cancel_rate_pct"]):
        ax.annotate(
            f"{value:.0f}%",
            (x, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )
    overall = _overall_cancel_rate(df)
    _reference_line(ax, overall, f"overall {overall:.1f}%", horizontal=True)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Canceled (%)")
    return fig


def _monthly_lines(ax: Axes, table: pd.DataFrame, fmt: str) -> None:
    """Draw one line per hotel with an end-of-line value label.

    Args:
        ax: Target axes.
        table: Output of ``eda.monthly_by_hotel``.
        fmt: Format for the end labels.
    """
    for hotel in table.columns:
        color = HOTEL_COLORS.get(hotel, COLORS["muted_series"])
        ax.plot(
            table.index,
            table[hotel],
            color=color,
            label=hotel,
            marker="o",
            markersize=4,
            markeredgecolor=COLORS["surface"],
            markeredgewidth=1.5,
        )
        ax.annotate(
            fmt.format(table[hotel].iloc[-1]),
            (len(table) - 1, table[hotel].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)


def plot_monthly_cancellation(df: pd.DataFrame) -> Figure:
    """Monthly cancellation rate per hotel.

    Args:
        df: Cleaned + engineered bookings DataFrame.

    Returns:
        The figure.
    """
    table = eda.monthly_by_hotel(df, "cancel_rate_pct")
    fig, (ax,) = new_figure(
        "Resort cancellations climb in summer; City stays high all year",
        "Cancellation rate (%) by arrival month, 2015-2017 combined",
    )
    _monthly_lines(ax, table, "{:.0f}%")
    ax.set_ylabel("Canceled (%)")
    ax.set_ylim(0, table.max().max() * 1.25)
    return fig


def plot_monthly_adr(df: pd.DataFrame) -> Figure:
    """Monthly average ADR per hotel.

    Args:
        df: Cleaned + engineered bookings DataFrame.

    Returns:
        The figure.
    """
    table = eda.monthly_by_hotel(df, "adr")
    fig, (ax,) = new_figure(
        "Resort prices peak sharply in summer; City prices stay flatter",
        "Average daily rate (EUR) by arrival month",
    )
    _monthly_lines(ax, table, "{:.0f}")
    ax.set_ylabel("Average ADR (EUR)")
    ax.set_ylim(0, table.max().max() * 1.2)
    return fig


def plot_cancellation_by_deposit_and_segment(df: pd.DataFrame) -> Figure:
    """Cancellation rate by deposit type and by market segment (two panels).

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        The figure.
    """
    fig, axes = new_figure(
        "Non-refundable deposits and online travel agents cancel most",
        "Cancellation rate (%) - by deposit type (left) and market segment (right)",
        ncols=2,
        figsize=(11, 4.4),
    )
    overall = _overall_cancel_rate(df)
    for ax, column in zip(axes, ("deposit_type", "market_segment"), strict=True):
        table = eda.cancellation_rate_ranked(df, column)
        _hbar(ax, table["cancel_rate_pct"], COLORS["series_1"], fmt="{:.1f}%")
        _reference_line(ax, overall, f"overall {overall:.1f}%")
        ax.set_xlabel("Canceled (%)")
    fig.subplots_adjust(left=0.13, wspace=0.55)
    return fig


def plot_top_countries(df: pd.DataFrame) -> Figure:
    """Bookings and cancellation rate for the top countries (two panels).

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        The figure.
    """
    table = eda.top_countries(df)
    fig, axes = new_figure(
        "Portugal dominates bookings and cancels above average",
        f"Top {len(table)} countries by bookings - volume (left) and cancellation rate (right)",
        ncols=2,
        figsize=(11, 4.4),
    )
    _hbar(axes[0], table["bookings"], COLORS["series_1"])
    axes[0].set_xlabel("Bookings")
    _hbar(axes[1], table["cancel_rate_pct"], COLORS["series_1"], fmt="{:.1f}%")
    overall = _overall_cancel_rate(df)
    _reference_line(axes[1], overall, f"overall {overall:.1f}%")
    axes[1].set_xlabel("Canceled (%)")
    return fig


def plot_cancellation_by_special_requests(df: pd.DataFrame) -> Figure:
    """Cancellation rate by number of special requests.

    Args:
        df: Cleaned bookings DataFrame.

    Returns:
        The figure.
    """
    table = eda.cancellation_rate_by(df, "total_of_special_requests").sort_index()
    fig, (ax,) = new_figure(
        "The more special requests, the less likely a cancellation",
        "Cancellation rate (%) by number of special requests (bookings in label)",
    )
    labels = [
        f"{count}\n(n={n:,})" for count, n in zip(table.index, table["bookings"], strict=True)
    ]
    ax.bar(labels, table["cancel_rate_pct"], color=COLORS["series_1"], width=0.6)
    for x, value in enumerate(table["cancel_rate_pct"]):
        ax.annotate(
            f"{value:.0f}%",
            (x, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["ink_secondary"],
        )
    overall = _overall_cancel_rate(df)
    _reference_line(ax, overall, f"overall {overall:.1f}%", horizontal=True)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Special requests")
    ax.set_ylabel("Canceled (%)")
    return fig


def plot_correlation_heatmap(
    train_rows: pd.DataFrame, columns: list[str], label_threshold: float = 0.7
) -> Figure:
    """Correlation heatmap of candidate numeric features (training rows).

    Only strong correlations (|r| >= ``label_threshold``) are labelled, so the
    redundant pairs stand out instead of drowning in numbers.

    Args:
        train_rows: Training split.
        columns: Numeric/binary candidate features.
        label_threshold: Minimum |r| for an in-cell label.

    Returns:
        The figure.
    """
    corr = train_rows[columns].corr()
    fig, (ax,) = new_figure(
        "Two pairs are near-duplicates: week number vs month, week nights vs total nights",
        "Pearson correlation between candidate numeric features (train); |r| >= 0.7 shown",
        figsize=(11, 10),
    )
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    ax.grid(visible=False)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.iat[i, j]
            if i != j and abs(value) >= label_threshold:
                # White text only on the darkest cells; dark ink elsewhere for contrast.
                ink = "#ffffff" if abs(value) >= 0.85 else COLORS["ink"]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=ink)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.6)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=0.9, left=0.25)
    return fig


def generate_eda_figures(
    raw: pd.DataFrame, clean: pd.DataFrame, featured: pd.DataFrame, step_report: pd.DataFrame
) -> list[Path]:
    """Create and save every EDA figure.

    Args:
        raw: Raw bookings DataFrame.
        clean: Cleaned bookings DataFrame.
        featured: Cleaned + engineered DataFrame.
        step_report: Cleaning step report.

    Returns:
        Paths of the saved figures, in order.
    """
    figures = {
        "01_missing_values": plot_missing_values(raw),
        "02_cleaning_steps": plot_cleaning_steps(step_report),
        "03_duplicates_profile": plot_duplicates_profile(raw),
        "04_adr_before_after": plot_adr_before_after(raw, clean),
        "05_cancel_by_lead_time": plot_cancellation_by_lead_time(featured),
        "06_monthly_cancellation": plot_monthly_cancellation(featured),
        "07_monthly_adr": plot_monthly_adr(featured),
        "08_cancel_by_deposit_and_segment": plot_cancellation_by_deposit_and_segment(featured),
        "09_top_countries": plot_top_countries(featured),
        "10_cancel_by_special_requests": plot_cancellation_by_special_requests(featured),
    }
    return [save_figure(fig, name) for name, fig in figures.items()]
