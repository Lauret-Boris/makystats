# ARMA–GARCH examples

| Notebook | Series | What it shows |
|----------|--------|----------------|
| `Market_Excess_Returns_ARMA_GARCH.ipynb` | Fama–French Mkt–RF daily | Diagnostics → GJR + *t*; final **`select_order`** |
| `Simulated_ARMA_GARCH.ipynb` | ARMA(1,1)–GARCH(1,1) DGP | Joint recovery of **AR+MA**; **`select_order`** finds (1,1) |
| `Fed_Funds_ARMA_GARCH.ipynb` | Δ federal funds (FRED DFF) | Real data with AR **and** MA; grid search |
| `EURUSD_ARMA_GARCH.ipynb` | EUR/USD log-returns | When **not** to keep ARMA; BIC grid |
| `Inflation_ARMA_GARCH.ipynb` | Quarterly Dp | Seasonal **AR(4)**; targeted `p_range=[1,2,4]` |

## `select_order` conventions (all notebooks)

- Use **`mean="AR"`** so that `p_range` / `q_range` vary. Do **not** pass `mean="ARMA"` into `select_order` (ARMAX path without exog fails).
- Keep grids **theory-shaped** (often `r=s=1`; small p,q).
- Prefer **`return_best=True`** for the final diagnostics object.
- Criterion default in examples: **BIC**.

Requires **makystats ≥ 0.1.4**.

Data under `data/`.
