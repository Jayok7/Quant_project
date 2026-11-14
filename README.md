# Two quantitative-research projects

Both projects use the same data and the same habit: work out what *nothing*
looks like before believing anything.

| file | what it is |
|---|---|
| `01_backtest_overfitting.ipynb` | How good does the best of 666 trading rules look when none of them work, and does the best real one clear that bar? |
| `02_rmt_correlation.ipynb` | How many independent directions does a 470-stock correlation matrix really have, how does that change through time, and does knowing it improve a portfolio? |
| `common.py` | Data download, cleaning (corporate actions, bad ticks) and plot style, shared by both. |
| `*.py` next to each notebook | The same notebook as a plain Python file (jupytext "percent" format) — easier to diff and review in Git. |
| `*.html` | Executed notebooks rendered for reading, with all outputs and figures. |

## Running

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
jupyter notebook            # then open either notebook and Run All
```

The data (about 30 MB) downloads itself into `data/` on first run.
Notebook 1 takes about three minutes; notebook 2 about twenty seconds.

## Data, and its main limitation

Daily prices for S&P 500 constituents, 2013-02-08 to 2018-02-07 (the public
"all_stocks_5yr" dataset). The ticker list is index membership **at the end**
of the sample, so it has survivorship bias, and five years contains no real
crisis. Both notebooks say where this limits the conclusions. Re-running on
CRSP data via WRDS would remove both problems and is the first extension to do.
