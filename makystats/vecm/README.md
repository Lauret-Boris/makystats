<p align="center">
  <img src="https://raw.githubusercontent.com/Lauret-Boris/makystats/main/docs/logo_makystats.png" alt="makystats" width="160"/>
</p>

# makystats.vecm — VECM et cointégration de Johansen

Sous-package de **makystats** pour l’estimation de modèles vectoriels à
correction d’erreur (VECM) et les tests de rang de Johansen.

- Régression *reduced-rank* (Johansen / Lütkepohl)
- Huit spécifications déterministes, dont les variantes **JHJ**
- P-values et valeurs critiques **MacKinnon–Haug–Michelis (1999)**
- IRF, FEVD, Granger, JB, whiteness, diagnostics et prévision **natifs**

**Documentation technique complète :** [`DOCUMENTATION.md`](DOCUMENTATION.md)

Paramètres (API makystats, **pas** statsmodels) :

| Paramètre | Valeurs | Rôle |
|-----------|---------|------|
| `test_case` | `'1'`…`'5'` | Spécification déterministe du **test de rang** |
| `det_case` | `'1'`, `'2'`, `'3a'`, `'3b'`, `'4a'`, `'4b'`, `'5a'`, `'5b'` | Spécification de l’**estimation** VECM (défaut `'3a'`) |
| `k_ar_diff` | int (≥ 0) | Retards en différences (= ordre du VAR en niveau − 1) |
| `coint_rank` | int | Rang de cointégration \(r\) |

---

## Installation

```bash
pip install makystats
```

Les tables `mhm_tables.npz` sont incluses dans le package.

---

## Imports

```python
from makystats import VECM, select_coint_rank

from makystats.vecm import (
    VECMModel, JohansenTest, vecm, VECM,
    select_coint_rank, summarize_johansen,
    information_criteria_table, mhm_pvalue, mhm_critical,
)
```

> Au niveau racine, seul `VECM` est exporté (pas `vecm`), pour ne pas
> masquer le sous-package `makystats.vecm`.

---

## Workflow minimal

```python
from makystats import VECM, select_coint_rank, select_var_order
from makystats.vecm import summarize_johansen, information_criteria_table

# data : DataFrame ou ndarray (T × n) en NIVEAU

# (optionnel) ordre VAR → k_ar_diff = p - 1
order = select_var_order(data, maxlags=4, criterion="bic")
k_ar_diff = order.best - 1

summarize_johansen(data, k_ar_diff=k_ar_diff, signif=0.05)
rank = select_coint_rank(data, k_ar_diff=k_ar_diff, test_case="3")
rank.summary()

model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=max(rank.rank, 1),
             det_case="3a").fit()
model.summary()

# Prévision (toutes les variables ou un sous-ensemble)
print(model.forecast(steps=5))
print(model.forecast(steps=5, variables="DTS"))
print(model.forecast(steps=5, conf_int=True, method="bootstrap",
                     sim_reps=1000, variables=["DTS", "FMG"]))

# Fitted / résidus
print(model.fitted_values(level=True).head())   # niveaux
print(model.fitted_values(level=False).head())  # différences
print(model.fittedvalues.head())                # Δ̂ seul
print(model.resid.head())                       # résidus (DataFrame + index)

# IRF / FEVD / tests / diagnostics (natifs)
model.irf(10).plot()
model.fevd(10).to_wide()["DTS"]
print(model.test_normality())
print(model.test_whiteness(nlags=4))
print(model.test_causality("DTS", "FMG"))
model.plot_diagnostics(lags=12)
model.plot_forecast(steps=5, variables="DTS")

information_criteria_table(data, k_ar_diff=k_ar_diff)
```

---

## API essentielle

| Méthode / propriété | Rôle |
|---------------------|------|
| `summary()` | tables de coefficients (style regression) |
| `resid` / `fittedvalues` | DataFrames Δ, **index d'estimation** |
| `fitted_values(level=True\|False)` | bascule **niveau** / **différences** (actual, fitted, resid) |
| `forecast(steps, conf_int=…, method=…, variables=…)` | prévision ; défaut = toutes les variables |
| `plot_forecast(…, variables=…)` | graphique historique + prévision (+ IC) |
| `irf` / `fevd` | natifs ; `.to_frame()`, `.to_wide()`, `.plot()` |
| `test_normality` / `test_whiteness` / `test_causality` | natifs |
| `plot_diagnostics` | 1 ligne/équation + corr. résidus en bas |

### `fitted_values(level=…)`

| `level` | Contenu |
|---------|---------|
| `True` (défaut) | `{c}_actual`, `{c}_fitted`, `{c}_resid` en **niveau** |
| `False` | `D({c})_actual`, `D({c})_fitted`, `D({c})_resid` en **différences** |

### `forecast(variables=…)`

```python
model.forecast(5)                         # toutes
model.forecast(5, variables="DTS")
model.forecast(5, conf_int=True, variables=["DTS", "FMG"])
```

Prévision système en interne ; seul le DataFrame renvoyé est filtré.

### IRF / FEVD / tests

Natifs (même philosophie que `makystats.VAR`), via le VAR en niveau
équivalent et les résidus reduced-rank. Granger : test F sur le VAR en
niveau (peut différer du Wald statsmodels en paramétrisation VECM).

### `information_criteria_table`

Un `*` marque le **minimum global** du critère (précision d’affichage
5 décimales). Plusieurs `*` = **ex aequo** (souvent Case *k* JHJ et Case *k*).

---

## Rang : p-values vs valeurs critiques

| Fonction | Critère |
|----------|---------|
| `select_coint_rank` | **P-values** continues MHM |
| `summarize_johansen` | **Valeurs critiques** tabulées MHM |

Près de la frontière, les rangs peuvent différer d’une unité. Détails :
[DOCUMENTATION.md §5.4](DOCUMENTATION.md#54-p-values-vs-valeurs-critiques-rang-affiché).

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `DOCUMENTATION.md` | Documentation technique complète |
| `__init__.py` | Exports publics |
| `core.py` | `select_coint_rank`, `vecm` / `VECM` |
| `vecm_model.py` | `VECMModel`, `JohansenTest`, MHM, tables IC |
| `mhm_tables.npz` | Tables MacKinnon–Haug–Michelis 1999 |
