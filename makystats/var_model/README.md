# makystats.var_model — VAR (OLS natif)

Vector Autoregression en **niveau**, estimation **OLS native** (coefficients,
IC, IRF, FEVD, Granger, JB, whiteness, diagnostics graphiques).

## Import

```python
from makystats import VAR, select_var_order
```

## Déterministes

| `det` | Contenu |
|-------|---------|
| `'n'` | aucun |
| `'c'` | constante (défaut) |
| `'t'` | tendance linéaire **seule** |
| `'ct'` | constante + tendance |
| `'ctt'` | constante + tendance + quadratique |

## Exemple

```python
from makystats import VAR

res = VAR(df[["DTS", "FMG"]], k_ar=1, det="c").fit()
print(res.summary())
print(res.aic, res.bic)                 # IC système (−2ℓ/T)
print(res.fittedvalues.head())          # DataFrame, index d'estimation
print(res.resid.head())
print(res.forecast(5))                  # toutes les variables
print(res.forecast(5, variables="DTS")) # une variable
print(res.forecast(5, conf_int=True, variables=["DTS", "FMG"]))
print(res.irf(10).to_wide()["DTS"])
print(res.fevd(10).to_wide()["DTS"])
print(res.test_causality("DTS", "FMG"))
print(res.test_normality())
print(res.test_whiteness(nlags=2))
res.plot_diagnostics(lags=12)
```

## Sélection d'ordre

```python
order = select_var_order(data, maxlags=4, det="c", criterion="bic")
print(order.table)       # triée par criterion (meilleur en haut), index = k_ar
print(order.best)        # p* selon criterion
print(order.selected)    # {'aic': …, 'bic': …, 'hqic': …}

order = select_var_order(data, maxlags=4, return_best=True, criterion="aic")
order.best_model.summary()
order.best_model.forecast(3, variables="DTS")
```

Options alignées sur `select_order` (ARMA-GARCH) : `criterion`, `return_best`,
`verbose`, `common_sample` (IC comparables sur le même échantillon T−maxlags).

### Pont vers le VECM

```python
from makystats import select_coint_rank, VECM

p = order.best                 # ordre VAR en niveau
k_ar_diff = p - 1              # retards en différences (peut être 0 si p=1)
rank = select_coint_rank(data, k_ar_diff=k_ar_diff, test_case="3")
if rank.rank >= 1:
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank.rank, det_case="3a").fit()
else:
    # pas de cointégration détectée → rester en VAR
    res = VAR(data, k_ar=p, det="c").fit()
```

## API `VARResults`

| Méthode / propriété | Rôle |
|---------------------|------|
| `summary()` | en-tête + tables par équation + diagnostics + critères système |
| `params` / `stderr` / `tvalues` / `pvalues` | coefficients |
| `fittedvalues` / `resid` | DataFrames, **index d'estimation** (`data.index[k_ar:]`) |
| `coef_table()` / `equation_stats()` | coeffs et diagnostics par équation |
| `forecast(steps, conf_int=…, variables=…)` | prévision ; défaut = toutes les variables |
| `irf(periods)` | IRF ; `.to_frame()`, `.to_wide()`, `.plot()` |
| `fevd(periods)` | FEVD ; `.to_frame()`, `.to_wide()`, `.plot()` |
| `test_causality(caused, causing)` | Granger (F) |
| `test_normality()` | Jarque-Bera multivarié |
| `test_whiteness(nlags)` | Portmanteau multivarié |
| `plot_diagnostics(lags=…, equations=…)` | 1 ligne/équation : résidus std, hist, Q-Q, ACF, ACF² + corr. en bas |

## `forecast` et filtre de variables

```python
res.forecast(5)                                    # toutes
res.forecast(5, variables="DTS")
res.forecast(5, variables=["FMG", "DTS"])          # ordre conservé
out = res.forecast(5, conf_int=True, variables="DTS")
# out["mean"], out["lower"], out["upper"]  (colonnes filtrées)
```

La prévision reste **système multivarié** en interne ; seul le résultat
exposé est filtré.

## IRF et FEVD

```python
irf = res.irf(10)
irf.to_frame()          # long : horizon, response, impulse, value
irf.to_wide()["DTS"]    # large : index=horizon, colonnes=chocs
irf.plot()

fevd = res.fevd(10)
fevd.to_wide()["DTS"]   # parts (lignes → somme 1)
fevd.plot()             # barres / aires empilées par équation
```

**Lecture FEVD.** Pour l'équation `DTS` à l'horizon 5, `shock[FMG] = 0.38`
signifie qu'environ 38 % de l'erreur de prévision de DTS à 5 pas est
attribuable aux chocs sur FMG (ordre de Cholesky = ordre des colonnes).

## Diagnostics résiduels

```python
res.plot_diagnostics(lags=12)
res.plot_diagnostics(equations=["DTS"], lags=8)
```

Une ligne par équation (style aligné ARMA-GARCH) : résidus standardisés,
histogramme + N(0,1)/KDE, Q-Q, ACF, ACF des carrés. Heatmap de corrélation
des résidus **en bas**, centrée (compatible `plt.style.use('fivethirtyeight')`).

## IC système (−2ℓ/T)

AIC = −2ℓ/T + 2k/T ; BIC = −2ℓ/T + k log(T)/T

(`summary()` rapporte aussi l'échelle log|Σ̂|.)
