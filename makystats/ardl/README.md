# makystats.ardl — ARDL natif

Moteur **OLS natif** (pas de délégation statsmodels pour l’estimation) :
ARDL en niveaux, UECM, relation de long terme, test aux bornes
(Pesaran / Narayan), ECM, CUSUM, prévisions conditionnelles.

## API recommandée : `case=` (1…5)

| case | Libellé tendance  | Contenu              |
|------|-------------------|----------------------|
| **1** | None             | pas d’intercept      |
| **2** | Rest. Constant   | intercept restreint  |
| **3** | Constant         | intercept libre (**défaut**) |
| **4** | Rest. Trend      | tendance restreinte  |
| **5** | Trend            | tendance libre       |

`trend="Rest. Constant"` (etc.) reste accepté ; `case=` est plus court.

```python
from makystats import ardl, select_ardl_order

res = ardl(y, x=x, lags=2, order=2, case=3).fit(verbose=False)

res.summary()           # ARDL + Normalité, ARCH-LM, Ljung-Box
res.summary_uecm()      # UECM + long run
res.summary_ecm()       # ECM (α)
res.summary_bounds()    # Pesaran / Narayan

res.bounds["f_stat"]
res.long_run_table      # θ : Coeff, Std.Err, t, Prob.
res.ecm()["alpha"]
```

## Régresseurs hors cointégration (`fixed`)

Comme statsmodels `fixed` : entrent en contemporain,
**pas** dans le long terme ni dans le F aux bornes (`k` = nombre de `x` seulement).

```python
res = ardl(y, x=df[["R"]], lags=2, order=2, case=3, fixed=df[["Z"]]).fit(verbose=False)
# Z dans res.params / UECM ; absent de res.long_run
```

## Sélection d’ordres

```python
tab = select_ardl_order(
    y, x=x, maxlag=4, maxorder=2, case=3, ic="bic", verbose=False
)
# maxorder hétérogène :
tab = select_ardl_order(
    y, x=df[["HR_s", "Vis", "spays", "oil"]],
    maxlag=3,
    maxorder={"HR_s": 2, "Vis": 3, "spays": 0, "oil": 1},
    case=3,
)
tab.best_model.summary()
tab.best_model.bounds
```

## Long terme : fitted values et graphiques

```python
df_lr = res.long_run_fitted()   # y, fitted_lr (y*), ect
res.plot_long_run()             # barres des θ
res.plot_long_run_fitted()      # y vs y* + ECT
```

## ECM restreint

Retrait prudent de coefficients (souvent non significatifs).  
**Attention :** sélection general-to-specific → biais de pré-test.  
Le **F aux bornes** reste celui du modèle **non restreint**.

```python
res.restrict_ecm(drop=["D.R.L1"])
# ou
res.restrict_ecm(pvalue=0.10, keep_levels=True)  # protège α et x.L1
```

## Prévisions

Prévisions **conditionnelles** aux trajectoires futures des `x` (obligatoires si le modèle a des x).

```python
exog_f = np.full((10, 1), x.iloc[-1])  # scénario

# ARDL en niveaux
fc = res.forecast(steps=10, exog_future=exog_f, conf_int=True)

# UECM / ECM (niveaux reconstruits : y_t = y_{t-1} + Δy_t)
res.restrict_ecm(drop=["D.R.L1"])
fc_e = res.forecast_ecm(
    steps=10, exog_future=exog_f, restricted=True, level=True, conf_int=True
)
# forecast, forecast_level, forecast_diff [, lower, upper, se]
```

| Méthode | Contenu de `forecast` |
|---------|------------------------|
| `forecast(...)` | niveaux (ARDL) |
| `forecast_ecm(..., level=True)` | **niveaux** (défaut) |
| `forecast_ecm(..., level=False)` | différences Δy |

## Récupération des résultats (après `fit`)

| Attribut / méthode | Contenu |
|--------------------|---------|
| `params`, `coef`, `coefficients` | Series des coefficients ARDL |
| `bse`, `se`, `pvalues`, `tvalues` | Inférence (Series) |
| `coef_table` | Coeff / Std.Err / t / Prob. |
| `cov_params()` | Matrice de covariance OLS |
| `conf_int(alpha=0.05)` | IC des coefficients |
| `aic`, `bic`, `hqic`, `llf`, `loglik` | Critères et log-vraisemblance |
| `nobs`, `df_model`, `df_resid`, `sigma2` | Tailles et variance |
| `rsquared`, `rsquared_adj` | Qualité d’ajustement |
| `residuals`, `resid`, `fitted_values` | Séries indexées |
| `long_run`, `long_run_table` | Multiplicateurs de long terme |

## Diagnostics

```python
# Tableau récapitulatif (Jarque-Bera + Ljung-Box + ARCH-LM)
res.diagnostic_summary(lags_lb=10, nlags_arch=5)

# Tests individuels
res.test_normality()
res.test_heteroskedasticity(5)
res.test_serial_correlation(10)

res.plot_diagnostics()
res.plot_cusum(alpha=0.05)   # alpha=0.05 ou 0.10
```

## Fichier technique

Voir [DOCUMENTATION.md](DOCUMENTATION.md) pour le détail des formules
(UECM, multiplicateurs, F aux bornes, IC de prévision).
