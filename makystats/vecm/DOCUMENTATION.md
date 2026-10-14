# makystats.vecm — Documentation technique

**Estimation et inférence pour les modèles à correction d’erreur vectoriels (VECM) et le test de cointégration de Johansen.**

Sous-package de **makystats** (`makystats.vecm`).  
Backend : `vecm_model.py` + tables `mhm_tables.npz` (MacKinnon–Haug–Michelis 1999).  
Dépendances : `numpy`, `pandas`, `scipy`, `statsmodels`.

---

## Table des matières

1. [Vue d’ensemble](#1-vue-densemble)
2. [Installation et prérequis](#2-installation-et-prérequis)
3. [Cadre théorique](#3-cadre-théorique)
4. [Spécifications déterministes](#4-spécifications-déterministes)
5. [Test de Johansen](#5-test-de-johansen)
6. [Estimation du VECM](#6-estimation-du-vecm)
7. [Approche JHJ (constantes / tendances doubles)](#7-approche-jhj)
8. [Critères d’information et sélection de rang](#8-critères-dinformation-et-sélection-de-rang)
9. [Référence de l’API](#9-référence-de-lapi)
10. [Exemples d’utilisation](#10-exemples-dutilisation)
11. [Conventions numériques](#11-conventions-numériques)
12. [Limitations et périmètre](#12-limitations-et-périmètre)
13. [Références](#13-références)

---

## 1. Vue d’ensemble

| Composant | Rôle |
|-----------|------|
| `JohansenTest` | Test de rang de cointégration (trace et maximum d’eigenvalue) |
| `select_coint_rank` | Test de rang + rang sélectionné (p-values MHM) |
| `mhm_pvalue` / `mhm_critical` | P-values et quantiles asymptotiques MacKinnon–Haug–Michelis (1999) |
| `summarize_johansen` | Comparaison des 5 cas déterministes côte à côte (sélection par CV) |
| `VECMModel` / `vecm` / `VECM` | Estimation VECM (α, β, Γ, déterministes), diagnostics, prévision |
| `information_criteria_table` | AIC / BIC / log-vraisemblance par rang et par cas |

**Public cible** : économistes appliqués, statisticiens, chercheurs en séries temporelles multivariées.

**Principes de conception** :

- Alignement sur la théorie classique (Johansen 1988, 1995 ; Lütkepohl ; Johansen–Hendry–Juselius pour les termes doubles).
- Noms de paramètres **makystats** : `test_case`, `det_case`, `k_ar_diff`, `coint_rank`, `columns` — **pas** la notation statsmodels (`det_order`, `deterministic`).
- Erreurs-types de court terme en OLS équation par équation avec degrés de liberté \(T - k\) (convention petit-échantillon).
- P-values de rang fondées sur les tables et l’algorithme officiel de MacKinnon (régression GLS locale), et non sur une interpolation ad hoc.

---

## 2. Installation et prérequis

### 2.1 Package

```bash
pip install makystats
```

Les tables MHM (`mhm_tables.npz`) sont livrées avec le sous-package
`makystats/vecm/`. Aucun fichier à placer manuellement.

### 2.2 Dépendances Python

```text
numpy >= 1.20
pandas >= 1.3
scipy >= 1.7
statsmodels >= 0.13
```

### 2.3 Imports typiques

```python
# Racine (recommandé pour l’estimation et la sélection de rang)
from makystats import VECM, select_coint_rank

# Sous-package (API complète)
from makystats.vecm import (
    VECMModel,
    JohansenTest,
    select_coint_rank,
    vecm,               # constructeur fluide (= VECMModel)
    VECM,               # alias de vecm
    summarize_johansen,
    information_criteria_table,
    mhm_pvalue,
    mhm_critical,
)
```

> Au niveau racine, seul l’alias `VECM` est exporté (pas `vecm`), afin de ne
> pas masquer le sous-package `makystats.vecm`.

---

## 3. Cadre théorique

### 3.1 Représentation VECM

Soit \(Y_t\) un processus \(n \times 1\) intégré d’ordre 1. La représentation à correction d’erreur d’un VAR d’ordre \(p+1\) en niveau s’écrit :

\[
\Delta Y_t = \alpha\beta' Y_{t-1} + \sum_{i=1}^{p} \Gamma_i \Delta Y_{t-i} + \Phi D_t + \varepsilon_t
\]

où :

- \(\alpha\) (\(n \times r\)) : coefficients d’ajustement (loading) ;
- \(\beta\) (\(n \times r\)) : vecteurs de cointégration ;
- \(\Gamma_i\) : dynamiques de court terme ;
- \(D_t\) : déterministes (constante, tendance) selon le cas ;
- \(r = \mathrm{rang}(\Pi)\) avec \(\Pi = \alpha\beta'\) ;
- \(p =\) `k_ar_diff` (nombre de retards en différences = ordre du VAR en niveau moins un).

### 3.2 Rang de cointégration

Sous \(H_0 : \mathrm{rang}(\Pi) \le r_0\), les statistiques de Johansen sont :

\[
\mathrm{LR}_{\mathrm{trace}}(r_0) = -T \sum_{i=r_0+1}^{n} \ln(1-\hat\lambda_i)
\]

\[
\mathrm{LR}_{\max}(r_0) = -T \ln(1-\hat\lambda_{r_0+1})
\]

où \(\hat\lambda_1 \ge \cdots \ge \hat\lambda_n\) sont les valeurs propres du problème aux valeurs propres généralisées issu de la régression à rang réduit (*reduced-rank regression*).

### 3.3 Distributions asymptotiques

Les distributions limites ne sont **pas** des \(\chi^2\) ordinaires dès que des racines unitaires et des déterministes sont présents. Les quantiles et p-values utilisés ici sont ceux de **MacKinnon, Haug et Michelis (1999)** : surfaces de réponse estimées par simulation, puis évaluation d’une p-value lisse par régression GLS locale cubique (implémentation fidèle à `johrouts.f`).

---

## 4. Spécifications déterministes

Cinq cas classiques (Johansen / Osterwald-Lenum), plus trois variantes **JHJ** (termes déterministes présents à la fois dans la relation de cointégration et en dehors) :

| Code | Description | Dans la cointégration | Hors cointégration (VECM) |
|------|-------------|------------------------|---------------------------|
| `'1'` | Aucun déterministe | — | — |
| `'2'` | Constante restreinte | constante | — |
| `'3'` / `'3b'` | Constante non restreinte | — | constante |
| `'3a'` | Constante double (JHJ) | constante (orth.) | constante |
| `'4'` / `'4b'` | Tendance restreinte + const. libre | tendance | constante |
| `'4a'` | Tendance restreinte + const. double (JHJ) | tendance + const. (orth.) | constante |
| `'5'` / `'5b'` | Constante et tendance libres | — | constante + tendance |
| `'5a'` | Constante et tendance doubles (JHJ) | const. + tend. (orth.) | constante + tendance |

**Notes :**

- Pour le **test de rang** (`JohansenTest`, `select_coint_rank`), seuls les codes `'1'` … `'5'` s’appliquent via le paramètre **`test_case`** (la distinction JHJ porte sur la *répartition* des déterministes après choix du rang, pas sur le test lui-même).
- Pour l’**estimation VECM** (`VECMModel` / `vecm` / `VECM`), les huit codes sont disponibles via **`det_case`**. Défaut : `'3a'`.

Correspondance interne avec statsmodels (`deterministic`) :

| Code makystats | `deterministic` statsmodels |
|----------------|----------------------------|
| `'1'` | `'n'` |
| `'2'` | `'ci'` |
| `'3a'`, `'3b'` | `'co'` |
| `'4a'`, `'4b'` | `'coli'` |
| `'5a'`, `'5b'` | `'colo'` |

---

## 5. Test de Johansen

### 5.1 Classe `JohansenTest`

```python
from makystats.vecm import JohansenTest

j = JohansenTest(data, k_ar_diff=1, test_case='3')
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `data` | array-like \((T, n)\) | Variables en **niveau** |
| `k_ar_diff` | int | Nombre de retards en différences |
| `test_case` | `'1'`…`'5'` | Spécification déterministe |

Méthodes principales :

```python
j = JohansenTest(data, k_ar_diff=1, test_case='3').fit()
j.summary(signif=0.05)          # tableau trace + max-eigen
j.pvalues('trace')              # liste de p-values MHM
j.pvalues('max_eig')
```

Attributs après `fit()` :

| Attribut | Contenu |
|----------|---------|
| `eigenvalues_` | \(\hat\lambda_i\) (ordre décroissant) |
| `trace_stat_` | statistiques de trace pour \(r_0 = 0,\ldots,n-1\) |
| `max_eig_stat_` | statistiques max-eigenvalue |
| `trace_crit5_` / `max_eig_crit5_` | quantiles à 5 % |
| `trace_crit_` / `max_eig_crit_` | quantiles 10 %, 5 %, 1 % — shape `(n, 3)` |
| `T_` | nombre d’observations effectives |

### 5.2 `select_coint_rank` (enveloppe recommandée)

```python
from makystats import select_coint_rank

rank = select_coint_rank(
    data, k_ar_diff=1, test_case='3', method='trace', signif=0.05
)
print(rank.rank)       # rang sélectionné
rank.summary()         # délègue à JohansenTest.summary
rank.pvalues('trace')
```

Le rang est la **première non-rejet** de \(H_0\) au sens des **p-values** continues MHM (et non des seules valeurs critiques tabulées).

| Paramètre | Description |
|-----------|-------------|
| `method` | `'trace'` (défaut) ou `'max_eig'` |
| `signif` | Niveau (typiquement 0.10, 0.05, 0.01) |
| `test_case` | `'1'`…`'5'` |

### 5.3 Paramètre `signif` dans `summary`

```python
j.summary(signif=0.05)   # défaut
j.summary(signif=0.01)
j.summary(signif=0.10)
```

- La colonne **CV** et l’étoile de rejet `*` utilisent le niveau choisi.
- La colonne **Prob.** (p-values) est **toujours** la p-value continue MHM ; elle ne dépend pas de `signif`.

### 5.4 P-values vs valeurs critiques (rang affiché)

| Fonction | Critère de sélection du rang |
|----------|------------------------------|
| `select_coint_rank` | **P-values** continues MHM |
| `summarize_johansen` | **Valeurs critiques** tabulées MHM au niveau `signif` |

Les deux s’appuient sur MacKinnon–Haug–Michelis (1999). Près de la frontière (statistique ≈ quantile critique), les rangs peuvent différer d’une unité. Pour un usage reproductible, fixez une règle et reportez aussi les statistiques et les p-values / CV.

### 5.5 Fonctions MHM autonomes

```python
from makystats.vecm import mhm_pvalue, mhm_critical

mhm_pvalue(stat, npr, test='trace', case='3')
mhm_critical(level, npr, test='trace', case='3')
```

| Argument | Signification |
|----------|----------------|
| `stat` | Valeur de la statistique observée |
| `npr` | \(n - r_0\) (nombre de racines unitaires sous \(H_0\)), entier 1…12 |
| `test` | `'trace'` ou `'max_eig'` |
| `case` | `'1'` … `'5'` |
| `level` | Niveau nominal pour une valeur critique (ex. `0.05`) |

### 5.6 `summarize_johansen`

Compare les cinq cas déterministes :

```python
from makystats.vecm import summarize_johansen

summarize_johansen(data, k_ar_diff=1, columns=None, signif=0.05)
```

Affiche les statistiques de trace et de max-eigenvalue pour chaque cas, et le **rang sélectionné** (première non-rejet de \(H_0\) au niveau `signif`, **via les valeurs critiques**).

---

## 6. Estimation du VECM

### 6.1 Classe `VECMModel` et constructeurs

```python
from makystats.vecm import VECMModel, vecm
from makystats import VECM   # alias racine de vecm

model = VECMModel(
    data,
    k_ar_diff=1,
    coint_rank=1,
    det_case='3a',
    columns=None,
)
# équivalent :
model = vecm(data, k_ar_diff=1, coint_rank=1, det_case='3a').fit()
model = VECM(data, k_ar_diff=1, coint_rank=1, det_case='3a').fit()
```

| Paramètre | Description |
|-----------|-------------|
| `data` | Array ou DataFrame \((T_{\mathrm{total}}, n)\) en **niveau** |
| `k_ar_diff` | Retards en différences \(p\) |
| `coint_rank` | Rang \(r\) (entier ≥ 0) |
| `det_case` | Un des 8 codes (§4) ; défaut `'3a'` |
| `columns` | Noms des séries (affichage) |

```python
model = VECM(data, k_ar_diff=1, coint_rank=1, det_case='3a',
             columns=['PIB', 'M2']).fit()
model.summary()
```

### 6.2 Paramètres estimés (propriétés)

| Propriété | Shape | Description |
|-----------|-------|-------------|
| `alpha` | \((n, r)\) | Coefficients d’ajustement |
| `beta` | \((n, r)\) | Vecteurs de cointégration (variables endogènes) |
| `gamma` | \((n, n\cdot p)\) | \(\Gamma_1,\ldots,\Gamma_p\) concaténés |
| `const` | \((n,)\) ou `None` | Constante hors cointégration |
| `lin_trend` | \((n,)\) ou `None` | Tendance hors cointégration |
| `const_coint` | \((r,)\) ou `None` | Constante dans la relation de cointégration |
| `lin_trend_coint` | \((r,)\) ou `None` | Tendance dans la relation de cointégration |
| `det_coef` | \((n, k_d)\) | Déterministes non restreints |
| `det_coef_coint` | \((k_c, r)\) | Déterministes dans la cointégration |

Erreurs-types, \(t\)-stats, \(p\)-values et intervalles de confiance homologues :  
`stderr_*`, `tvalues_*`, `pvalues_*`, `conf_int_*` pour `alpha`, `beta`, `gamma`, `det_coef`, `det_coef_coint`.

### 6.3 Ajustement, résidus, vraisemblance

| Propriété / méthode | Description |
|---------------------|-------------|
| `resid` | Résidus : **DataFrame** (T × n), index d'estimation |
| `fittedvalues` | Δ̂_t : **DataFrame**, même index |
| `fitted_values(level=True\|False)` | Bascule **niveau** / **différences** (actual, fitted, resid) — pas un empilement |
| `sigma_u` | Covariance des résidus (normalisation \(T\)) |
| `llf` | Log-vraisemblance **joint** du système |
| `nobs`, `neqs`, `k_ar`, `dof_` | Dimensions et degrés de liberté |

### 6.4 Prévision (native makystats)

```python
fc = model.forecast(steps=8)                      # toutes les variables
fc = model.forecast(steps=8, variables='DTS')     # une variable
fc = model.forecast(steps=8, conf_int=True)       # IC analytiques
fc = model.forecast(steps=8, conf_int=True, method='bootstrap',
                    sim_reps=1000, seed=1, variables=['DTS', 'FMG'])
fc = model.forecast(steps=8, conf_int=True, method='gaussian', sim_reps=1000)
covs = model.forecast_cov(steps=8)
paths = model.simulate_forecast(steps=8, method='bootstrap', sim_reps=500)
fig = model.plot_forecast(steps=8, method='bootstrap', sim_reps=1000,
                          variables='DTS')
```

**Point central.** Récursion à **chocs nuls** (α, β, Γ, déterministes
makystats / JHJ). Indépendant de `method`.

**IC** (`conf_int=True`) — paramètres **fixés** (pas de re-estimation) :

| `method` | Principe | Bandes |
|----------|----------|--------|
| `'analytic'` (défaut) | MSE Lütkepohl \(\sum_j \Psi_j\Sigma_u\Psi_j'\) | quantiles normaux |
| `'bootstrap'` | rééchantillonnage par **lignes** des résidus centrés | percentiles empiriques |
| `'gaussian'` | \(N(0,\Sigma_u)\) | percentiles empiriques |

Sous normalité et `sim_reps` grand, simulation ≈ analytique. Résidus non
gaussiens : préférer `'bootstrap'` (proche de la simulation EViews pour les
SE de forecast). Alias : `'eviews'`/`'simulation'` → bootstrap ;
`'mc'`/`'normal'` → gaussian.

**`plot_forecast`** natif (matplotlib), mêmes options `method` / `sim_reps` / `variables`.

Le paramètre `variables` (str ou séquence) filtre le DataFrame renvoyé (défaut : **toutes** les séries). La prévision reste système multivarié en interne.

### 6.5 IRF, FEVD, Granger, normalité, whiteness (natifs)

Comme `makystats.VAR`, ces méthodes sont **natives** :

| Méthode | Contenu |
|---------|---------|
| `irf(periods)` | IRF réduites + orthogonales (Cholesky) ; `.to_frame()`, `.to_wide()`, `.plot()` |
| `fevd(periods)` | Décomposition de variance ; `.to_frame()`, `.to_wide()`, `.plot()` |
| `ma_rep` / `orth_ma_rep` / `var_rep` | Représentations MA et VAR en niveau |
| `test_normality()` | Jarque-Bera multivarié |
| `test_whiteness(nlags, adjusted=…)` | Portmanteau (Lütkepohl) |
| `test_granger_causality` / `test_causality` | F sur le **VAR en niveau** équivalent |
| `test_inst_causality` | Corrélations contemporaines des résidus |
| `plot_diagnostics(lags, equations=…)` | 1 ligne/équation + heatmap de corrélation en bas |
| `plot_data` | Séries en niveau |

Fondations : `_level_var_coefs`, `_ma_coefficients`, résidus reduced-rank.
Le profil moyen des IRF coïncide avec l'étape 1 statsmodels (mêmes α, β, Γ).
Granger makystats = F sur le VAR en niveau (peut différer du Wald statsmodels VECM).

```python
model.irf(10).plot()
model.fevd(10).to_wide()["DTS"]
model.test_normality()
model.test_whiteness(nlags=4)
model.test_causality("DTS", "FMG")
model.plot_diagnostics(lags=12)
```


## 7. Approche JHJ

Les cas `'3a'`, `'4a'`, `'5a'` implémentent la répartition **Johansen–Hendry–Juselius** des déterministes :

1. **Étape 1** — Estimation reduced-rank classique (identique au cas `'3b'` / `'4b'` / `'5b'` correspondant) : on obtient \(\hat\alpha\), \(\hat\beta\), \(\hat\Gamma\).
2. **Étape 2** — Pour chaque relation \(j\), régression OLS de \(\hat z_{j,t} = \hat\beta_j' Y_{t-1}\) (éventuellement enrichi des termes restreints) sur les régresseurs duaux \(W_t\) (constante et/ou tendance). Le coefficient, changé de signe, est affiché **dans** la relation de cointégration **sans écart-type** (ce n’est pas un paramètre de la vraisemblance, mais le résultat d’une orthogonalisation).
3. **Étape 3** — Ré-estimation de l’équation de court terme avec l’ECT orthogonalisé. \(\alpha\) et \(\Gamma\) restent numériquement inchangés ; les coefficients déterministes **hors** cointégration et **toutes** les erreurs-types sont recalculés sur la nouvelle matrice de régresseurs.

Cette procédure rend identifiable une constante (ou tendance) « dans » la cointégration et une autre « dehors », au prix de l’absence d’écart-type sur le terme dual affiché dans la relation de long terme.

---

## 8. Critères d’information et sélection de rang

```python
from makystats.vecm import information_criteria_table

information_criteria_table(data, k_ar_diff=1, columns=None, max_rank=None)
```

Produit, pour chaque rang \(r = 0,\ldots,n-1\) et chaque cas déterministe (y compris JHJ) :

- la log-vraisemblance du système ;
- l’AIC ;
- le critère de Schwarz (BIC).

**Pénalisation utilisée dans cette table** (spécifique à la sélection de rang) :

\[
N_{\mathrm{par}} = r\cdot n + r\cdot n_{\mathrm{restr}}
\]

où \(n_{\mathrm{restr}}\) compte les termes déterministes **restreints** pourvus d’écart-type (`ci` / `li`), **pas** les termes JHJ orthogonalisés, ni les \(\Gamma\), ni les composantes libres de \(\beta\).

Cette convention diffère de l’AIC/BIC affiché équation par équation dans `VECMModel.summary()`.

---

## 9. Référence de l’API

### 9.1 Fonctions de niveau module

| Fonction | Rôle |
|----------|------|
| `mhm_pvalue(stat, npr, test, case)` | P-value asymptotique MHM |
| `mhm_critical(level, npr, test, case)` | Quantile asymptotique MHM |
| `summarize_johansen(data, k_ar_diff, columns, signif)` | Tableau comparatif des 5 cas (rang via CV) |
| `information_criteria_table(data, k_ar_diff, columns, max_rank)` | AIC/BIC/LL par rang et cas |
| `select_coint_rank(data, k_ar_diff, test_case, method, signif, columns)` | Test + rang (p-values MHM) |
| `vecm(...)` / `VECM(...)` | Constructeur fluide → `VECMModel` |

### 9.2 `JohansenTest`

```text
__init__(data, k_ar_diff=1, test_case='3')
fit() -> self
pvalues(stat='trace'|'max_eig') -> list[float]
summary(signif=0.05) -> None  # impression
_crit_at(signif, stat) -> ndarray   # quantiles au niveau demandé
```

### 9.3 `VECMModel` — cycle de vie

```text
__init__(data, k_ar_diff=1, coint_rank=1, det_case='3a', columns=None)
fit() -> self
summary() -> None
fitted_values(level=True|False) -> DataFrame  # bascule niveau / différences
resid / fittedvalues -> DataFrame             # index d'estimation
forecast(..., variables=None) -> DataFrame    # défaut = toutes les variables
forecast_cov(steps=1, method='analytic', ...) -> list[ndarray]
simulate_forecast(...) -> ndarray
plot_forecast(..., variables=None) -> Figure
irf / fevd -> IRFResult / FEVDResult          # natifs
test_normality / test_whiteness / test_causality  # natifs
plot_diagnostics(lags=12, equations=None) -> Figure
predict(steps=1, conf_int=False, method='analytic', ...) -> ndarray
```

### 9.4 `VECMModel` — coefficients et inférence

Propriétés de coefficients : `alpha`, `beta`, `gamma`, `const`, `lin_trend`, `const_coint`, `lin_trend_coint`, `det_coef`, `det_coef_coint`, `deterministic`.

Pour chaque bloc concerné : `stderr_*`, `tvalues_*`, `pvalues_*`, `conf_int_*(alpha=0.05)`.

### 9.5 `VECMModel` — échantillon et qualité d’ajustement

`resid`, `fittedvalues`, `sigma_u`, `llf`, `nobs`, `neqs`, `k_ar`, `dof_`, `names`, `y_all`, `coint_rank_`.

---

## 10. Exemples d’utilisation

### 10.1 Test de rang puis estimation

```python
import numpy as np
from makystats import VECM, select_coint_rank
from makystats.vecm import JohansenTest, summarize_johansen

# data : array (T, n) ou DataFrame en niveau
data = ...  # à fournir
cols = ['y1', 'y2', 'y3']

# Vue d’ensemble des cas déterministes (rang via CV)
summarize_johansen(data, k_ar_diff=2, columns=cols, signif=0.05)

# Test détaillé + rang via p-values MHM
rank = select_coint_rank(data, k_ar_diff=2, test_case='3', method='trace')
rank.summary(signif=0.05)
print('Rang retenu:', rank.rank)
print('P-values trace:', rank.pvalues('trace'))

# Estimation au rang retenu
model = VECM(
    data, k_ar_diff=2, coint_rank=rank.rank, det_case='3a', columns=cols
).fit()
model.summary()

# Prévision
print(model.forecast(steps=4))
```

### 10.2 Lecture des coefficients de long terme

```python
print('beta:\n', model.beta)
print('alpha:\n', model.alpha)
print('constante hors CI:', model.const)
print('constante dans CI (JHJ):', model.const_coint)
```

### 10.3 Valeurs critiques personnalisées

```python
from makystats.vecm import mhm_critical, mhm_pvalue

# Quantile 5 %, trace, cas 2, n-r = 2
cv = mhm_critical(0.05, npr=2, test='trace', case='2')
pv = mhm_pvalue(18.5, npr=2, test='trace', case='2')
```

---

## 11. Conventions numériques

| Élément | Convention retenue |
|---------|-------------------|
| Échantillon effectif | \(T = T_{\mathrm{total}} - p - 1\) |
| SE de court terme (α, Γ, déterministes hors CI) | OLS équation par équation, \(\widehat{\mathrm{Var}} = \hat\sigma^2 (X'X)^{-1}\), \(\hat\sigma^2 = \mathrm{RSS}/(T-k)\) |
| SE de β et déterministes restreints | Formules asymptotiques Johansen, multipliées par \(\sqrt{T/(T-k)}\) pour cohérence petit-échantillon |
| Tests \(t\) court terme | Loi de Student à \(T-k\) degrés de liberté |
| Tests sur β | Loi normale asymptotique |
| Covariance des résidus `sigma_u` | Division par \(T\) (MLE) |
| Log-vraisemblance `llf` | Forme gaussienne **système** (déterminant de \(\hat\Sigma\)) |
| Tendance | Indexation alignée sur statsmodels : \(t = p+1,\ldots,T_{\mathrm{total}}\) sur l’échantillon d’estimation |
| Normalisation de β | Première variable normalisée à 1 (convention reduced rank) |
| P-values de rang | MacKinnon–Haug–Michelis (1999), \(n-r \in \{1,\ldots,12\}\) |

---

## 12. Limitations et périmètre

1. **Variables exogènes** : non prises en charge (`exog`, `exog_coint` restent `None`).
2. **Saisonnalité déterministe** : non implémentée (`seasonal`, `seasons` non actifs).
3. **P-values MHM** : asymptotiques ; pas de correction de taille finie au-delà de l’usage des quantiles MHM. Disponibles pour \(n-r \le 12\).
4. **Rang \(r = 0\)** : estimation VECM possible (pas de terme ECT) ; interpréter avec prudence.
5. **IRF / FEVD / Granger / normalité / whiteness** : **natifs** (voir §6.5).
   Le point IRF coïncide avec l'étape 1 (mêmes α, β, Γ). Granger = F sur le
   VAR en niveau équivalent (peut différer du Wald statsmodels VECM).
6. **Prévision** : **native** makystats (récursion JHJ, IC analytiques ou
   par simulation `bootstrap` / `gaussian`). Les paramètres restent fixés
   dans les IC (incertitude d’estimation de α, β, Γ non rejouée — comme la
   MSE asymptotique classique et la simulation de forecast SE type EViews
   à coefficients fixés).
7. **Identification de β** : une normalisation (première composante = 1) ;
   les rotations dans l’espace de cointégration pour \(r > 1\) ne sont pas
   explorées automatiquement.
8. **Données manquantes** : non gérées dans le backend brut ;
   `select_coint_rank` / `vecm` acceptent un DataFrame et en retirent les
   NaN via conversion numérique.

---

## 13. Références

1. Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12, 231–254.
2. Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.
3. Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration — with applications to the demand for money. *Oxford Bulletin of Economics and Statistics*, 52, 169–210.
4. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
5. MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. *Journal of Applied Econometrics*, 14, 563–577.  
   Programme et tables : http://qed.econ.queensu.ca/jae/1999-v14.5/mackinnon-haug-michelis/
6. Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic distribution of the maximum likelihood cointegration rank test statistics. *Oxford Bulletin of Economics and Statistics*, 54, 461–472.
7. Pesaran, M. H., Shin, Y., & Smith, R. J. (2000). Structural analysis of vector error correction models with exogenous I(1) variables. *Journal of Econometrics*, 97, 293–343.

---

## Annexe A — Schéma de workflow recommandé

```text
1. Analyse univariée (ordre d’intégration) — hors package
2. summarize_johansen(...)            → orientation sur le cas déterministe
3. select_coint_rank(...)             → choix du rang r (p-values MHM)
   # ou JohansenTest(...).fit().summary()
4. information_criteria_table(...)    → confirmation rang / cas (AIC, BIC)
5. VECM(..., coint_rank=r, det_case=...).fit()
6. Diagnostics (whiteness, normality, Granger, IRF, FEVD) — **natifs** (voir §6.5)
7. forecast (natif ; method='analytic'|'bootstrap'|'gaussian') / IRF (délégation) selon les besoins
```

## Annexe B — Mapping des codes de cas (rappel)

| Test de rang (`test_case`) | Estimation (`det_case`) classique | Estimation JHJ |
|----------------------------|-----------------------------------|----------------|
| `'1'` | `'1'` | — |
| `'2'` | `'2'` | — |
| `'3'` | `'3b'` | `'3a'` |
| `'4'` | `'4b'` | `'4a'` |
| `'5'` | `'5b'` | `'5a'` |

---

*Documentation du sous-package `makystats.vecm`. Voir aussi `README.md` dans ce répertoire pour un démarrage rapide.*
