# ARMA / SARIMA example — Airline passengers

Classic Box–Jenkins monthly airline passenger series (1949–1960).

| File | Role |
|------|------|
| `Airline_Passengers_SARIMA.ipynb` | Walkthrough: log transform, SARIMA(0,1,1)×(1,1,1)₁₂, diagnostics, `select_arma_order` |
| `data/airline_passengers.csv` | Source data |

```bash
pip install makystats matplotlib pandas
jupyter notebook Airline_Passengers_SARIMA.ipynb
```

Requires **makystats ≥ 0.1.3**.

| `Inflation_ARMA.ipynb` | Quarterly inflation (`Dp`) · AR(4), selection, forecast |
| `data/interest_inflation.csv` | year, quarter, Dp, R |

