# Faults and Austin chalk productivity

In spite of several thousand horizontal wells drilled in the Austin Chalk, regional productivity analyses are limited. So we wrote a paper on it, called "Regional Productivity in the Austin Chalk with Emphasis on Fault Zone Production in the Karnes Trough Area."

It will eventually appear in the AAPG Bulletin. In the meantime, this is the code used to generate the analysis and figures in that paper.

We found that faults are key to understanding how productive wells are.

## Getting started

Please have [`git`](https://git-scm.com/) and [`uv`](https://docs.astral.sh/uv/) installed. `uv` can install python, if needed, with

```sh
uv python install 3.12
```

The python precise project requirements are stored in `uv.lock`. To get them, run the following commands

```sh
git clone https://github.com/frank1010111/austin-chalk-faults.git
cd austin-chalk-faults
uv venv
uv sync
source venv/bin/activate
```

The exploratory data analysis is in `EDA Austin chalk.ipynb`. The results of the XGBoost and LightGBM gradient boosting regressors are at `Fault impact-Karnes.ipynb` and `Fault impact-Karnes-lightgbm.ipynb`.

Want the data? I'm sorry, it came from IHS. Have an IHS or S&P-Global oil and gas data subscription? Drop me a line and I can show you how to replicate the data and analysis.
