# Pipeline examples

End-to-end workflows that chain **stationarity tests** with an estimation model. Each notebook is a single narrative: every step is motivated by the previous diagnostic, not by a feature checklist.

| Notebook | Pipeline | Data |
|----------|----------|------|
| `Stationarity_to_ARDL.ipynb` | Sequential ADF / KPSS → lag selection → ARDL → bounds → long run / ECM | Quarterly inflation & interest rate (`../ARDL/data/interest_inflation.csv`) |
| `Stationarity_to_ARMA_GARCH.ipynb` | Sequential ADF / KPSS → ARMA mean → GARCH(1,1) → Student-t | Daily Fama–French Mkt–RF (`../ARMA_GARCH/data/ff_mkt_rf_daily.csv`), 2015–2023 |

```bash
pip install "makystats[plot]" matplotlib pandas
jupyter notebook exemples/pipelines/
```

Requires **makystats ≥ 0.1.5**.

## Design principles

1. **Stationarity before estimation** — integration order constrains which models are valid (ARDL bounds need no I(2); returns GARCH needs I(0)).
2. **Honest test outcomes** — inconclusive bounds or residual non-normality are reported, not hidden.
3. **Parsimony** — BIC-oriented lag choices; GARCH(1,1) before more exotic volatility forms.
4. **One question per notebook** — conditional inflation equation *or* equity-return mean/variance, not both.
