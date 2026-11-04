# makystats.arma — ARMA / ARIMA / SARIMA (variance constante)

Estimation MLE d’un **ARMA**, **ARIMA** ou **SARIMA** **sans** GARCH :
un seul paramètre de variance \(\sigma^2\).

L’**index** temporel de `y` (DatetimeIndex, PeriodIndex, …) est préservé
sur résidus, fitted values et prévisions. Si `d` ou `D` > 0, des NaN
apparaissent en tête (alignement sur l’échantillon complet).

## Import

```python
from makystats import arma, arima, sarima, select_arma_order
# ou : from makystats.arma import ARMA, arma, ...
```

## ARMA (série stationnaire, `d=0`)

```python
res = arma(y, mean="ARMA", ar=1, ma=1, trend="c").fit()
print(res.coef_table)
print(res.params["AR(1)"])       # Series nommée
print(res.bse, res.pvalues)
print(res.cov_params())
print(res.forecast(steps=5, conf_int=True))
```

## ARIMA (série en niveau, souvent I(1))

L’utilisateur **n’a pas** à différencier à la main :

```python
res = arma(prices, mean="ARMA", ar=1, ma=1, d=1, trend="c").fit()
res = arima(prices, order=(1, 1, 1), trend="c").fit()

res.forecast(steps=10)                # niveaux (défaut si d>0)
res.forecast(steps=10, levels=False)  # sur Δy
res.fitted_values_level()             # fitted en niveau
```

| `d` | Rôle |
|-----|------|
| `0` | ARMA sur `y` |
| `1` | ARIMA sur `y` non stationnaire en niveau |
| `2` | Double différenciation (rare) |

Sur Δy, `trend='c'` correspond à une **dérive** en niveau (constante sur les différences).

## SARIMA (saisonnalité)

```python
# Forme multiplicative via SARIMAX (ordres denses)
res = arma(
    ylog, mean="ARMA", ar=0, ma=1, d=1,
    seasonal_order=(0, 1, 1, 12), trend="n",
).fit()

# Alias
res = sarima(ylog, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend="n").fit()

# Lags « troués » + saison → moteur natif (lags saisonniers additifs)
res = arma(
    ylog, mean="ARMA", ar=[4], ma=1, d=1,
    seasonal_order=(1, 1, 1, 12), trend="c",
).fit()
```

| `seasonal_order` | `(P, D, Q, s)` |
|------------------|----------------|
| `P`, `Q` | AR / MA saisonniers |
| `D` | différences saisonnières \((1-L^s)^D\) |
| `s` | période (12 = mensuel, 4 = trimestriel) |

## Récupération des résultats (après `fit`)

| Attribut / méthode | Contenu |
|--------------------|---------|
| `params`, `coef`, `coefficients` | Series des coefficients nommés |
| `bse`, `se`, `pvalues`, `tvalues` | Series d’inférence |
| `coef_table` | Tableau Coeff / SE / z / Prob. / IC |
| `cov_params()` | Matrice de covariance \(k\times k\) |
| `conf_int(alpha=0.05)` | IC des coefficients |
| `aic`, `bic`, `hqic`, `llf`, `loglik` | Critères et log-vraisemblance |
| `nobs`, `df_model`, `df_resid` | Tailles d’échantillon |
| `residuals`, `resid`, `fitted_values` | Séries indexées |
| `arparams`, `maparams` | Sous-ensembles AR/SAR et MA/SMA |
| `arroots`, `maroots` | Racines des polynômes (complexes) ; \(\|r\| > 1\) souhaité |
| `sigma2` | Variance innovatrice |

## Diagnostics

```python
res.test_serial_correlation(lags=10)   # Ljung-Box
res.test_heteroskedasticity(lags=5)    # ARCH-LM
res.test_normality()                   # Jarque-Bera
res.plot_diagnostics()
```

## Sélection d’ordres

```python
tab = select_arma_order(y, p_range=range(0, 4), q_range=range(0, 3), criterion="bic")
tab = select_arma_order(prices, d=1, p_range=range(0, 3), q_range=range(0, 2))
tab = select_arma_order(
    ylog, d=1, seasonal_order=(0, 1, 1, 12),
    p_range=range(0, 3), q_range=range(0, 2), criterion="aic",
)
tab, best = select_arma_order(..., return_best=True)
```

`d` et `seasonal_order` sont **fixes** pour toute la grille ; seuls `p` et `q` varient.

## `arma` vs `arma_garch`

| Besoin | Fonction |
|--------|----------|
| Moyenne seule, éventuellement intégrée / saisonnière | **`arma` / `arima` / `sarima`** |
| Volatilité conditionnelle / VaR | **`arma_garch`** |

## Notes

- IC de **prévision** en niveau pour `d>0` : approximation indicative (pas la variance exacte état-espace).
- Coefficients estimés sur l’échelle de travail (après `d` / `D`) ; `forecast` et `fitted_values_level` recomposent les niveaux.
- SARIMA dense → SARIMAX multiplicatif ; ARMA « troué » + saison → lags saisonniers **additifs**.
