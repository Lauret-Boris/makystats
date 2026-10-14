<p align="center">
  <img src="https://raw.githubusercontent.com/Lauret-Boris/makystats/main/docs/logo_makystats.png" alt="makystats" width="160"/>
</p>

# makystats.armagarch — ARMA + GARCH (MLE)

Sous-package de **makystats** pour l’estimation conjointe d’une moyenne
linéaire (ARMA / ARX / HAR) et d’une variance conditionnelle
(GARCH, GJR, EGARCH, APARCH, FIGARCH).

**Documentation technique complète :** [`DOCUMENTATION.md`](DOCUMENTATION.md)

---

## Installation

```bash
pip install makystats
pip install "makystats[numba]"   # JIT optionnel
pip install "makystats[plot]"    # plot_diagnostics
```

---

## Imports

```python
from makystats import arma_garch, select_order, ARMAGarch
# ou : from makystats.armagarch import arma_garch, ARMAGarch, select_order
```

---

## Workflow minimal

Après `fit()`, **`y` et `x` sont mémorisés** (`res._y`, `res._x_user`) :
`forecast()`, `var()`, `results_frame()`, `walk_forward_calibration()`, etc.
peuvent omettre `y` / `x`. Pour `forecast`/`var` multi-pas, si `x` futur
est omis, la **dernière ligne** des régresseurs est répétée (avec warning).

## Workflow minimal (détail)

```python
from makystats import arma_garch, select_order

res = arma_garch(y, mean='AR', ar=1, ma=1, vol='GARCH', dist='normal', trend='c').fit()
print(res.coef_table)   # const, AR(1), MA(1), omega, ...
print(res.loglik, res.aic, res.bic)
print(res.forecast(y, h=5))
fig = res.plot_diagnostics(lags=20)

# Constante + tendance
res_ct = arma_garch(y, mean='AR', ar=1, vol='GARCH', trend='ct').fit()

# ARMA à moyenne nulle (pas de constante)
res0 = arma_garch(y, mean='Zero', ar=1, ma=1, vol='GARCH').fit()

tab = select_order(y, p_range=range(0, 3), q_range=range(0, 3),
                   criterion='bic', n_jobs=1)
print(tab.attrs['best'])
```

---

## Fitted values & `results_frame`

```python
res.fitted_values          # prédiction 1-pas de la moyenne
res.residuals              # y - fitted_values
res.results_               # DataFrame in-sample créé à la fin de fit()
res.results_frame()        # sans re-passer y
res.results_frame(include_x=True, index=dates)
```

Identité du filtre : `y = fitted_values + residuals`.

## API en bref

| Argument | Valeurs / rôle |
|----------|----------------|
| `x` | `DataFrame` / array (n×k) ; noms de colonnes → labels |
| `mean` | `Constant`, `Zero`, `AR`, `ARX`, `ARMAX`, `HAR`, `HARX`, `LS` — structure dynamique. `Zero` = pas de constante (AR/MA restent possibles). |
| `in_mean` | `None` / `sigma` / `var` / `log` — **GARCH-M** : λ·f(σ_t) dans la moyenne ; coef `archm` |
| `trend` | `n` / `c` (défaut) / `t` / `ct` — déterministes : aucun / constante / tendance / les deux. Labels : `const`, `trend`. |
| `vol` | `GARCH`, `ARCH`, `GJR`/`TARCH`, `EGARCH`, `APARCH`, `FIGARCH` |
| `dist` | `normal`, `t`, `skewt`, `ged` |
| `ar` / `ma` | ordres ou listes de lags (ex. `[1, 3]`, `ma=[5]`) |
| `r` / `s` ou `p` / `q` | ordres ARCH / GARCH |
| `o` | ordre d’asymétrie (GJR / EGARCH / APARCH) |
| `backcast` | λ pré-échantillon variance (souvent `0.7`) |
| `adjust_sample` | `True` : burn-in hors LL ; `False` : échantillon complet |
| `egarch_form` | `centered` (défaut) ou `raw` |
| `center` | déméanisation numérique ; désactivé automatiquement si `trend` ∈ {`t`,`ct`} |

---

## Régresseurs exogènes (`x`) — noms dans `coef_table`

| Forme de `x` | Labels des coefficients |
|--------------|-------------------------|
| `pandas.DataFrame` | noms des **colonnes** |
| `pandas.Series` | `Series.name` |
| `ndarray` + `fit(..., x_names=[...])` | noms fournis |
| `ndarray` seul | `x1`, `x2`, … |

```python
# Recommandé : DataFrame
x = df[["Visiteurs", "Ouverture", "Crise", "Change"]]
res = arma_garch(y, x=x, mean="LS", vol="GARCH", r=1, s=1).fit()
print(res.coef_table)  # const, Visiteurs, Ouverture, …

# ndarray : passer les noms explicitement
res = arma_garch(y, x=x.values, mean="LS", vol="GARCH").fit(
    x_names=["Visiteurs", "Ouverture", "Crise", "Change"]
)
```


## GARCH-M (ARCH-M)

```python
res = arma_garch(y, mean="AR", ar=1, ma=1, vol="GARCH", in_mean="var").fit()
# in_mean: None | "sigma" | "var" (EViews Variance) | "log"
# coefficient label: archm (λ)
```

Avec `select_order` (même `in_mean` pour toute la grille) :

```python
tab, best = select_order(y, p_range=range(0, 3), q_range=range(0, 2),
                         in_mean="var", return_best=True)
```

Si σ² varie peu, `const` et `archm` peuvent échanger du niveau (quasi
non-identification). Préférer la significativité de `archm` et le niveau
`c + λ·E[σ²]`, pas `const` seul.

## Value-at-Risk / Expected Shortfall

- **Prévision** : `var` / `expected_shortfall` — `aggregation="marginal"` (défaut) ou `"cumulative"` (MC).
- **In-sample 1-pas** : `var_series` → `var_fitted_`, `var_backtest`, `plot_var` (pas de cumul multi-jours).

```python
res = arma_garch(y, mean="AR", ar=1, ma=1, vol="GARCH").fit()

# VaR / ES horizon h (prévision)
res.var(alpha=0.05, h=1, method="parametric")              # marginale (défaut)
res.var(alpha=0.05, h=10, aggregation="cumulative", n_sims=10000, seed=1)  # somme sur 10 pas
res.expected_shortfall(alpha=0.05)

# Fitted VaR in-sample (DataFrame) → aussi res.var_fitted_
df = res.var_series(alpha=0.05)
df = res.var_series(alphas=(0.05, 0.01), index=dates)

# Backtest (Kupiec)
res.var_backtest(alpha=0.05)

# Graphique des dépassements
res.plot_var(alphas=(0.05, 0.01), method="parametric")
res.plot_var(alphas=(0.05, 0.01), window=300)          # dernières 300 obs.
res.plot_var(alphas=(0.05, 0.01), window=(100, 400))   # intervalle
res.plot_var(alphas=(0.05, 0.01), window=slice(-200, None))
res.plot_var(alphas=(0.05, 0.01), style={
    "var_lines": {0.05: {"color": "darkorange", "lw": 2}},
    "exceedance": {0.01: {"marker": "x", "c": "crimson", "s": 30}},
    "no_exceedance": {"s": 8},
})
ax, data = res.plot_var(alphas=(0.05, 0.01), return_data=True)
```

- VaR = μ̂ + σ̂ · q_α(Z) ; `as_loss=True` → −VaR_raw (convention perte).
- Horizon h = quantile de **y_{T+h}** (pas du cumul multi-périodes).
- Styles par défaut : `ARMAGarch._default_plot_var_style()`.


## Fichiers

| Fichier | Rôle |
|---------|------|
| `DOCUMENTATION.md` | Documentation technique complète |
| `__init__.py` | Exports publics |
| `core.py` | `ARMAGarch`, `arma_garch`, `select_order` |


---


