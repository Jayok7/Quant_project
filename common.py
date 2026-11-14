"""
common.py - data loading, cleaning and plot styling shared by both notebooks.

Everything in here is deliberately boring. The interesting code lives in the
notebooks; this file exists so that both projects start from *exactly* the same
cleaned return panel, the same way both of the Week 7 scripts shared one
training core.

DATA
----
Daily OHLCV for the S&P 500 constituents, 2013-02-08 to 2018-02-07 (the public
"all_stocks_5yr" Kaggle dataset, mirrored on GitHub). Three things about it
matter and are handled or flagged below:

  1. SURVIVORSHIP BIAS. The ticker list is the index membership at the END of
     the sample. Companies that went bust or were deleted before 2018 are simply
     absent. This inflates average returns and makes any "strategy" look better
     than it would have in real time. It cannot be fixed with this file; it
     needs point-in-time constituents (CRSP via WRDS). It is the first thing to
     say in the limitations section.

  2. UNADJUSTED CORPORATE ACTIONS. Closes are not adjusted for several spin-offs
     and share distributions, so e.g. eBay "falls 57%" on the day PayPal was
     spun out. Shareholders lost nothing that day. Left in, these are the
     largest "returns" in the whole panel and they are entirely fictitious.

  3. BAD TICKS. A handful of prices are simply wrong for one day (a stock halves
     and fully recovers the next session). Also fictitious.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib as mpl

DATA_URL = ("https://raw.githubusercontent.com/plotly/datasets/master/"
            "all_stocks_5yr.csv")
SECTOR_URL = ("https://raw.githubusercontent.com/datasets/"
              "s-and-p-500-companies/main/data/constituents.csv")
DATA_DIR = Path(__file__).resolve().parent / "data"

TRADING_DAYS = 252  # annualisation constant used everywhere

# Days on which the raw close-to-close "return" is a corporate action, not a
# price move. Each was checked by hand against the price series: the level
# shifts permanently on the day (unlike a bad tick, which reverts).
CORPORATE_ACTIONS = {
    "DISCA": "2014-08-07",  # 2-for-1 share distribution (Series C shares)
    "DISCK": "2014-08-07",  # same event, other share class
    "A":     "2014-11-03",  # Agilent spins off Keysight
    "BAX":   "2015-07-01",  # Baxter spins off Baxalta
    "NI":    "2015-07-02",  # NiSource spins off Columbia Pipeline
    "EBAY":  "2015-07-20",  # eBay spins off PayPal
    "ARNC":  "2016-11-01",  # Alcoa separation into Arconic / Alcoa Corp
    "YUM":   "2016-11-01",  # Yum! spins off Yum China
    "CAG":   "2016-11-10",  # Conagra spins off Lamb Weston
}


def _fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once; later calls reuse the local copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_prices() -> pd.DataFrame:
    """Wide (date x ticker) frame of closing prices, complete histories only.

    Tickers with ANY missing day are dropped rather than filled. Most of them
    are companies that listed or were spun out mid-sample, so there is nothing
    honest to fill the gap with, and a correlation matrix needs every column to
    cover the same dates.
    """
    raw = pd.read_csv(_fetch(DATA_URL, DATA_DIR / "all_stocks_5yr.csv"),
                      parse_dates=["date"])
    prices = raw.pivot(index="date", columns="Name", values="close")
    complete = prices.columns[prices.isna().sum() == 0]
    return prices[complete]


def clean_returns(prices: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Simple daily returns with corporate actions and bad ticks neutralised.

    Both kinds of artefact are set to a return of ZERO for the affected day.
    Zero is the conservative choice: it removes the fictitious move without
    inventing a replacement.
    """
    rets = prices.pct_change().iloc[1:]

    # --- 1. known corporate actions ------------------------------------------
    for ticker, day in CORPORATE_ACTIONS.items():
        if ticker in rets.columns:
            rets.loc[day, ticker] = 0.0

    # --- 2. bad ticks: a big move that the NEXT day almost exactly undoes ----
    # (1 + r_t)(1 + r_{t+1}) ~ 1 means the price went away and came straight
    # back. A genuine 20% earnings move does not do that.
    gross = 1.0 + rets
    round_trip = (gross * gross.shift(-1) - 1.0).abs() < 0.05
    spike = (rets.abs() > 0.20) & round_trip
    # zero both the spike day and the day after (the reversal)
    bad = spike | spike.shift(1, fill_value=False)
    n_ticks = int(spike.sum().sum())
    rets = rets.mask(bad, 0.0)

    if verbose:
        print(f"panel: {rets.shape[0]} days x {rets.shape[1]} tickers "
              f"({rets.index[0].date()} to {rets.index[-1].date()})")
        print(f"neutralised {len(CORPORATE_ACTIONS)} corporate-action days "
              f"and {n_ticks} bad ticks")
        print(f"largest remaining one-day move: {rets.abs().max().max():.1%}")
    return rets


def load_sectors() -> pd.Series:
    """ticker -> GICS sector. CAVEAT: this is TODAY'S constituent list, so
    tickers that have since been renamed, merged or deleted come back missing,
    and sector definitions have changed since 2018 (Communication Services did
    not exist yet). Good enough to label eigenvectors; not good enough to build
    a sector-neutral strategy on."""
    df = pd.read_csv(_fetch(SECTOR_URL, DATA_DIR / "constituents.csv"))
    return df.set_index("Symbol")["GICS Sector"]


# ---------------------------------------------------------------- plot style
# One fixed palette, assigned by ROLE rather than by plotting order, so the
# same thing is the same colour in every figure of both notebooks.
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VIOLET = "#e87ba4", "#6250d6"
NULL_GREY = "#b9b8b1"      # anything that is a null / noise reference


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": "#52514e",
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
        "lines.linewidth": 2.0, "legend.frameon": False,
    })
