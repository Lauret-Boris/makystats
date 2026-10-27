# ARDL examples

| Notebook | Topic |
|----------|--------|
| `Danish_Money_Demand_ARDL.ipynb` | Money demand (Denmark) · bounds, long run, ECM |
| `Inflation_Interest_ARDL.ipynb` | Inflation `Dp` and rate `R` |
| `Consumption_Income_ARDL.ipynb` | US log C on log Y (PSS case 2) |
| `Unemployment_Inflation_ARDL.ipynb` | US unemployment and inflation |

## Data

| File | Source |
|------|--------|
| `data/danish_money.csv` | Classic Danish money-demand series |
| `data/interest_inflation.csv` | Quarterly Dp, R |
| `data/us_macro_quarterly.csv` | US macro (consumption, income, unemp, infl, …) |

```bash
pip install "makystats[plot]"
jupyter notebook exemples/ARDL/
```

Requires **makystats ≥ 0.1.4**.
