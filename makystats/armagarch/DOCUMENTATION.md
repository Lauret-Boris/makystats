# makystats.armagarch — Documentation technique

**Estimation par maximum de vraisemblance de modèles ARMA + variance conditionnelle
(GARCH, GJR/TARCH, EGARCH, APARCH, FIGARCH).**

Sous-package de **makystats** (`makystats.armagarch`).  
API fluide inspirée de [`arch`](https://github.com/bashtage/arch).  
Dépendances : `numpy`, `pandas`, `scipy`, `statsmodels` ; optionnel : `numba`, `matplotlib`.

---

## Table des matières

1. [Vue d’ensemble](#1-vue-densemble)
2. [Installation](#2-installation)
3. [Modèle statistique](#3-modèle-statistique)
4. [Vraisemblance et échantillon](#4-vraisemblance-et-échantillon)
5. [Optimisation](#5-optimisation)
6. [Post-estimation](#6-post-estimation)
7. [Référence de l’API](#7-référence-de-lapi)
8. [Exemples](#8-exemples)
9. [Notes numériques](#9-notes-numériques)
10. [Limitations](#10-limitations)
11. [Références](#11-références)
11 bis. [GARCH-M (ARCH-M)](#11-bis-garch-m-arch-m)
12. [Value-at-Risk et Expected Shortfall](#12-value-at-risk-et-expected-shortfall)

---

## 1. Vue d’ensemble

| Composant | Rôle |
|-----------|------|
| `arma_garch` | Constructeur fluide → instance `ARMAGarch` |
| `ARMAGarch` | Classe d’estimation MLE (moyenne + variance) |
| `select_order` | Grille AIC / BIC / HQIC sur les ordres |

Estimation **conjointe** d’une équation de moyenne linéaire (ARMA / ARX / HAR / LS, optionnellement **GARCH-M** via ``in_mean``)
et d’une équation de variance conditionnelle, sous innovations normale, Student-\(t\),
skew-\(t\) (Hansen) ou GED. Optimisation BFGS **non contrainte** dans un espace
reparamétré ; écarts-types par défaut via l’estimateur OPG (*outer product of gradients*).

**Public cible** : économistes appliqués, finance quantitative, chercheurs en séries temporelles.

---

## 2. Installation

```bash
pip install makystats
pip install "makystats[numba]"   # accélération JIT optionnelle du filtre
pip install "makystats[plot]"    # matplotlib pour plot_diagnostics
```

### Imports

```python
from makystats import arma_garch, select_order, ARMAGarch

# ou depuis le sous-package
from makystats.armagarch import arma_garch, ARMAGarch, select_order
```

---

## 3. Modèle statistique

### 3.1 Équation de moyenne

La moyenne combine **(i)** des termes *déterministes* contrôlés par ``trend``
et **(ii)** une structure dynamique contrôlée par ``mean`` / ``ar`` / ``ma`` / ``x``.

Chemin déterministe (index \(\tau_t = 1,\ldots,T\)) :

\[
\mu_t = c\cdot\mathbf{1}_{\mathrm{const}} + \delta\,\tau_t\cdot\mathbf{1}_{\mathrm{trend}}.
\]

Équation complète (avec GARCH-M optionnel) :

\[
(y_t - \mu_t)
  = \sum_{i\in\mathcal{I}_{\mathrm{AR}}} \phi_i (y_{t-i}-\mu_{t-i})
  + \sum_{j\in\mathcal{I}_{\mathrm{MA}}} \theta_j \varepsilon_{t-j}
  + \mathbf{x}_t'\boldsymbol{\beta}
  + \lambda\, f(\sigma_t)
  + \varepsilon_t,
\]

où le terme \(\lambda f(\sigma_t)\) n’est présent que si ``in_mean`` est actif
(voir §11 bis). Sinon \(\lambda=0\).

Quand ``trend='c'`` (défaut), \(\mu_t \equiv c\) et l’écriture se réduit à la
forme classique « moyenne inconditionnelle » utilisée historiquement dans le package.

#### Structure dynamique (`mean`)

| `mean` | Description |
|--------|-------------|
| `'Constant'` | Pas de dynamique AR/MA par défaut (sauf si `ar`/`ma` sont passés explicitement) |
| `'Zero'` | **Pas de constante** (≠ « pas de dynamique ») : `ar`/`ma` restent autorisés |
| `'AR'` | Termes AR (ordres via `ar` ou `lags`) |
| `'ARX'` | AR + exogènes \(\mathbf{x}_t\) (pas de MA) |
| `'ARMAX'` / `'ARMA'` | AR + MA (+ \(\mathbf{x}_t\) si fourni) |
| `'LS'` | Déterministe + \(\mathbf{x}_t\) uniquement (p=q=0) |
| `'HAR'` / `'HARX'` | Moyennes mobiles causales multi-horizons (Corsi) ; horizons via `lags` (défaut `[1, 5, 22]`) |

Les ensembles \(\mathcal{I}_{\mathrm{AR}}\) et \(\mathcal{I}_{\mathrm{MA}}\) peuvent être
consécutifs \(\{1,\ldots,p\}\) ou **creux** (ex. `ma=[5]` : un seul MA au retard 5).

#### Termes déterministes (`trend`)

Paramètre ``trend`` ∈ {``'n'``, ``'c'``, ``'t'``, ``'ct'``} (convention EViews / statsmodels) :

| Valeur | Constante \(c\) | Tendance \(\delta\,\tau_t\) | Labels dans `coef_table` |
|--------|-----------------|-------------------------------|---------------------------|
| ``'n'`` | non | non | — |
| ``'c'`` (défaut) | oui | non | ``const`` |
| ``'t'`` | non | oui | ``trend`` |
| ``'ct'`` | oui | oui | ``const``, ``trend`` |

Interactions :

- ``mean='Zero'`` + ``trend='c'`` → forcé en ``trend='n'`` (rétrocompatibilité).
- ``mean='Zero'`` + ``trend='ct'`` → **erreur** (constante demandée et refusée).
- ``mean='Zero'`` + ``trend='t'`` → tendance seule, sans constante (valide).
- Équivalence : ``mean='Zero', ar=1`` ≡ ``mean='AR', trend='n', ar=1``.

**Centrage.** Avec `center=True` (défaut) et une **constante sans tendance**,
\(y\) est déméanisée pendant l’optimisation ; la constante est restaurée à
l’échelle d’origine après convergence. Dès qu’une **tendance** est présente
(`t` ou `ct`), le centrage est **désactivé** automatiquement (sinon \(\delta\)
serait mal interprété).

### 3.2 Variance conditionnelle

On pose \(\varepsilon_t = \sigma_t z_t\) avec \(z_t\) i.i.d. sous la loi choisie.

| `vol` | Récursion (schéma) |
|-------|---------------------|
| **GARCH(\(r,s\))** | \(\sigma_t^2 = \omega + \sum_{k=1}^{r}\alpha_k\varepsilon_{t-k}^2 + \sum_{\ell=1}^{s}\beta_\ell\sigma_{t-\ell}^2\) |
| **ARCH** | GARCH avec \(s=0\) |
| **GJR / TARCH** | GARCH + \(\sum_k \gamma_k\mathbf{1}_{\{\varepsilon_{t-k}<0\}}\varepsilon_{t-k}^2\) |
| **EGARCH** (Nelson 1991) | \(\log\sigma_t^2 = \omega + \sum_k\alpha_k(\lvert z_{t-k}\rvert-\mathbb{E}\lvert z\rvert) + \sum_k\gamma_k z_{t-k} + \sum_\ell\beta_\ell\log\sigma_{t-\ell}^2\) |
| **APARCH** (Ding–Granger–Engle) | \(\sigma_t^\delta = \omega + \sum_k\alpha_k(\lvert\varepsilon_{t-k}\rvert-\gamma_k\varepsilon_{t-k})^\delta + \sum_\ell\beta_\ell\sigma_{t-\ell}^\delta\) |
| **FIGARCH(\(p,d,q\))** | \(\sigma_t^2 = \omega + \beta(L)\sigma_t^2 + \lambda(L)\varepsilon_t^2\) avec poids fractionnaires de \((1-L)^d\) (troncature par défaut 1000) |

**EGARCH — formes d’affichage.** Le filtre interne utilise la spécification *centrée*
de Nelson. Avec `egarch_form='raw'`, seul l’intercept reporté est décalé :

\[
\omega_{\mathrm{raw}} = \omega_{\mathrm{centred}} - \mathbb{E}\lvert z\rvert\sum_k\alpha_k,
\]

de sorte que l’équation affichée corresponde à l’écriture non centrée. La
vraisemblance et les autres coefficients restent inchangés.

**Restrictions de signe.** Pour un GARCH multi-lags, les \(\alpha_k,\beta_\ell\)
individuels ne sont **pas** forcés positifs (seul \(\omega>0\) via \(\exp\)) ;
des coefficients négatifs restent admissibles tant que \(\sigma_t^2\) reste
positif dans le filtre. Le \(\gamma\) de GJR est libre ; celui d’APARCH est
mappé dans \((-1,1)\) via \(\tanh\) ; le \(d\) de FIGARCH est soft-borné pour
la stabilité.

### 3.3 Distributions d’innovation

| `dist` | Forme standardisée utilisée |
|--------|------------------------------|
| **normal** | \(-\tfrac12\log(2\pi)-\tfrac12\log\sigma_t^2-\varepsilon_t^2/(2\sigma_t^2)\) |
| **Student-\(t\)** | \(\nu>2\), variance unitaire |
| **skew-\(t\)** (Hansen 1994) | \(\nu>2\), asymétrie \(\lambda\in(-1,1)\) |
| **GED** | Forme \(\nu>0\) ; \(\nu=2\) retrouve le gaussien |

Les paramètres de forme sont reparamétrés pour que BFGS reste dans un domaine
ouvert non contraint (ex. \(\nu=2+e^{\tilde\nu}\) pour la loi \(t\)).

---

## 4. Vraisemblance et échantillon

### 4.1 Log-vraisemblance

\[
\mathcal{L}(\boldsymbol{\theta}) = \sum_{t=b+1}^{n} \ell(y_t\mid\mathcal{F}_{t-1};\boldsymbol{\theta}),
\]

où \(b\) est le burn-in et \(\ell\) utilise \(\varepsilon_t(\boldsymbol{\theta})\)
et \(\sigma_t^2(\boldsymbol{\theta})\) du filtre conjoint.

### 4.2 Burn-in et `adjust_sample`

| Flag | Comportement |
|------|----------------|
| `adjust_sample=True` (défaut) | \(b=\max(\text{lags AR},\,\text{lags MA},\,\texttt{hold\_back})\). Les \(b\) premières observations initialisent les retards ; elles **n’entrent pas** dans \(\mathcal{L}\). |
| `adjust_sample=False` | \(b=0\). Les \(n\) termes entrent dans \(\mathcal{L}\) ; les états pré-échantillon sont remplis par backcast. |

HAR impose au moins \(b=\max(\text{horizons})\) lorsque `adjust_sample=True`.

### 4.3 Pré-échantillon MA

Lorsque des termes MA sont présents, les innovations pré-échantillon ne sont
**pas** mises à zéro. L’implémentation construit des résidus AR-only, une
récursion **arrière** (style Bjørn–Johansen), prolonge le chemin avant
l’échantillon, puis lance la récursion ARMA **avant** avec ces états MA.

### 4.4 Backcast de variance

Variance conditionnelle pré-échantillon par lissage exponentiel des résidus
au carré (paramètre \(\lambda=\) `backcast`, défaut \(0.7\)) :

\[
h_0 = \lambda^T \bar{e}^2 + (1-\lambda)\sum_{j=0}^{T-1}\lambda^j e_j^2,
\]

avec \(j=0\) le résidu le **plus ancien** de la fenêtre d’estimation.
\(\lambda=1\) se réduit à \(h_0=\bar{e}^2\). Pour les modèles de type GARCH,
\(h_0\) est aussi **recalculé à partir des résidus courants** à chaque évaluation
de la vraisemblance, afin de rester cohérent avec les paramètres de moyenne.

---

## 5. Optimisation

### 5.1 BFGS non contraint + reparamétrisation

Les paramètres économiques \(\boldsymbol{\psi}\) sont obtenus par une application
lisse \(\boldsymbol{\theta}\mapsto\boldsymbol{\psi}\) :

| Bloc | Transformation (illustrative) |
|------|-------------------------------|
| AR / MA | \(\phi=0.99\tanh(\tilde\phi)\) |
| \(\omega\) (famille GARCH) | \(\omega=\exp(\tilde\omega)\) |
| \(\alpha,\beta\) (GARCH multi-lags) | libres (signés) ; le filtre borne \(\sigma_t^2>0\) |
| EGARCH \(\beta\) | positif, persistance \(<1\) |
| APARCH \(\gamma\) | \(\tanh\) vers \((-1,1)\) |
| FIGARCH \(d\) | intervalle stable soft |
| ddl \(t\) | \(\nu=2+e^{\tilde\nu}\) |

Aucune contrainte d’inégalité n’est passée à SciPy. Un **polish** (second BFGS)
relance depuis le point courant si le premier run ne signale pas le succès.

### 5.2 Valeurs initiales

- ARMA dense : ARIMA préliminaire (statsmodels) lorsque les lags sont \(1\ldots p\), \(1\ldots q\).
- ARMA creux : défauts neutres.
- Variance : petits \(\alpha\), \(\beta\) modéré, \(\omega\) calibré sur la variance des résidus.

### 5.3 Écarts-types

Défaut `cov_type='opg'` (BHHH) :

\[
\widehat{\mathrm{Var}}(\hat{\boldsymbol{\theta}})
= \Bigl(\sum_{t} \mathbf{g}_t\mathbf{g}_t'\Bigr)^{-1}.
\]

Alternative : `cov_type='hessian'` (Hessienne numérique de l’objectif).

---

## 6. Post-estimation

### 6.1 Séries (longueur \(n\))

| Attribut | Contenu |
|----------|---------|
| `residuals` | \(\hat\varepsilon_t\) |
| `std_resid` | \(\hat\varepsilon_t/\hat\sigma_t\) |
| `conditional_vol` | \(\hat\sigma_t\) |
| `fitted_values` | \(y_t-\hat\varepsilon_t\) (prévision de moyenne à un pas) |
| `coef_table` | coefficients, SE, \(z\), \(p\)-values, IC 95 % |
| `loglik`, `aic`, `bic` | critères |

### 6.2 Diagnostics

- **Ljung–Box** sur résidus standardisés et sur leurs carrés.
- **Jarque–Bera** sur résidus standardisés.
- **`plot_diagnostics(lags=20)`** : figure \(2\times 3\) — trajectoire des résidus,
  histogramme + \(\mathrm{N}(0,1)\) + KDE, QQ-plot normal, ACF de \(z_t\),
  ACF de \(z_t^2\), volatilité conditionnelle.

### 6.3 Prévisions

```python
fc = res.forecast(y, h=5, threshold=0.0, x=None)
# colonnes : mean, variance, vol, p_up
```

- **Moyenne** : récursion ARMA ; chocs futurs nuls (MSE minimale sous perte symétrique).
- **Variance** : récursion de la famille choisie.
- **`p_up`** : \(P(y_{T+h}>\texttt{threshold})\) sous la loi d’innovation ajustée.
- Exogènes futurs : fournir `x` avec \(h\) lignes ; sinon contribution nulle.

### 6.4 Sélection d’ordres (`select_order`)

Grille cartésienne sur les ordres AR, MA, ARCH, GARCH (et éventuellement listes
de `dist` / `variance_model`). Les critères sont **recalculés sur un burn-in
commun** pour rendre comparables des modèles d’ordres différents.

```python
tab = select_order(
    y,
    p_range=range(0, 3),
    q_range=range(0, 3),
    r_range=[1], s_range=[1],
    criterion='bic',   # 'aic' | 'bic' | 'hqic'
    n_jobs=1,
)
print(tab.attrs['best'])
```

**Limitation :** la grille utilise des ordres denses \(0,1,\ldots\). Les
spécifications creuses (`ma=[5]`) s’estiment directement via `arma_garch`.

---

## 7. Référence de l’API

### 7.1 `arma_garch(...)`

Constructeur fluide. Retourne une instance `ARMAGarch` prête pour `.fit()`.

| Argument | Rôle |
|----------|------|
| `y` | Série (optionnel si passé plus tard à `fit`) |
| `x` | Exogènes (\(n \times k\)) ; `DataFrame` → noms de colonnes dans `coef_table` |
| `mean` | `'Constant'`, `'Zero'`, `'AR'`, `'ARX'`, `'ARMAX'`, `'LS'`, `'HAR'`, `'HARX'` |
| `trend` | `'n'`, `'c'` (défaut), `'t'`, `'ct'` — déterministes (aucun / constante / tendance / les deux) |
| `vol` | `'GARCH'`, `'ARCH'`, `'GJR'`/`'TARCH'`, `'EGARCH'`, `'APARCH'`, `'FIGARCH'` |
| `dist` | `'normal'`, `'t'`, `'skewt'`, `'ged'` |
| `ar`, `ma` | Ordres ou listes de lags (prioritaires sur `lags` / `p` / `q` pour la moyenne) |
| `r`, `s` / `p`, `q` | Ordres ARCH / GARCH |
| `o` | Ordre d’asymétrie (GJR / EGARCH / APARCH) |
| `power` | Exposant \(\delta\) APARCH (défaut 2.0) |
| `backcast` | \(\lambda\) du backcast de variance (défaut 0.7) |
| `adjust_sample` | Voir §4.2 (défaut `True`) |
| `hold_back` | Observations supplémentaires exclues en tête |
| `center` | Déméanisation pendant l’optimisation (défaut `True`) ; désactivé si `trend` ∈ {`t`,`ct`} |
| `cov_type` | `'opg'` (défaut) ou `'hessian'` |
| `egarch_form` | `'centered'` (défaut) ou `'raw'` |
| `in_mean` | `None` (défaut), `'sigma'`, `'var'`, `'log'` — **GARCH-M** ; coef `archm` (voir §11 bis) |
| `transform` | ex. `'logdiff'` si `y` sont des prix |
| `max_iter`, `tol` | Options BFGS |

### 7.1 bis Noms des régresseurs exogènes

Les coefficients associés à \(\mathbf{x}_t\) apparaissent dans `coef_table` sous :

| Entrée | Labels |
|--------|--------|
| `x` est un `pandas.DataFrame` | noms des colonnes |
| `x` est une `pandas.Series` | `Series.name` |
| `fit(..., x_names=['a','b',…])` | noms fournis (prioritaires) |
| `ndarray` sans `x_names` | `x1`, `x2`, … |

Valable pour `mean` ∈ {`LS`, `ARX`, `ARMAX`, `HARX`} et pour
`select_order(..., x=DataFrame)` / `walk_forward_calibration(..., x=DataFrame)`.

```python
x = df[["Visiteurs", "Ouverture", "Crise"]]
res = arma_garch(y, x=x, mean="LS", vol="GARCH", r=1, s=1).fit()
# coef_table : const, Visiteurs, Ouverture, Crise, omega, …
```

### 7.2 `ARMAGarch`

Constructeur bas niveau (préférer `arma_garch` en usage courant).

Méthodes principales après construction :

Paramètre constructeur supplémentaire : ``in_mean`` (GARCH-M, §11 bis).

```text
fit(y=None, x=None, verbose=True) -> self
forecast(h=10, ...) -> DataFrame          # y/x mémorisés si omis
results_frame() / results_                # fitted, résidus, vol
var / expected_shortfall / plot_var
plot_diagnostics(lags=20) -> Figure
summary() / coef_table
```

Attributs après `fit` : `coef_table`, `loglik`, `aic`, `bic`, `residuals`,
`std_resid`, `conditional_vol`, `fitted_values`, `results_`, `params`, `se`,
`pvalues`, `in_mean`, …

### 7.3 `select_order(y, **kwargs)`

| Argument | Rôle |
|----------|------|
| `p_range`, `q_range`, `r_range`, `s_range` | Grilles d’ordres |
| `dist`, `variance_model` | str ou listes |
| `criterion` | `'aic'`, `'bic'`, `'hqic'` |
| `n_jobs` | `1` = séquentiel (défaut) ; `-1` = tous les cœurs |
| `return_best` | Si `True`, retourne `(DataFrame, best_model)` |
| `mean`, `x`, `lags`, `o`, `power`, `backcast`, … | Passés à chaque fit |
| `in_mean` | GARCH-M pour **toute** la grille (`None`/`sigma`/`var`/`log`) |
| `trend` | Déterministes pour toute la grille (`n`/`c`/`t`/`ct`) |

---

### 7.4 Value-at-Risk / Expected Shortfall

| Méthode | Rôle |
|---------|------|
| `var(..., aggregation, n_sims, seed)` | VaR : `marginal` (y_{T+h}) ou `cumulative` (somme MC) |
| `expected_shortfall(...)` | ES / CVaR (mêmes options `aggregation`) |
| `var_series(alpha\|alphas)` | Fitted VaR **1-pas** → `var_fitted_` |
| `var_backtest(alpha)` | Kupiec sur VaR 1-pas in-sample |
| `plot_var(alphas, window, style)` | Graphique dépassements **1-pas** |

Voir **§12** pour formules, conventions (`as_loss`, horizon) et styles graphiques.
`y` est optionnel après `fit()`.


## 8. Exemples

```python
from makystats import arma_garch, select_order

# ARMA(1,1)–GARCH(1,1)–Normal
res = arma_garch(y, mean='ARMA', ar=1, ma=1, vol='GARCH', dist='normal',
                 backcast=0.7).fit()
print(res.coef_table)
print(res.loglik, res.aic, res.bic)

# MA(5) creux + GARCH, échantillon complet dans la LL
res = arma_garch(y, mean='Constant', ma=[5], vol='GARCH',
                 adjust_sample=False, backcast=0.7).fit()

# Asymétrie + queues épaisses
res = arma_garch(y, mean='AR', ar=1, vol='GJR', o=1, dist='t').fit()

# EGARCH, affichage omega brut
res = arma_garch(y, mean='ARMA', ar=1, ma=1, vol='EGARCH',
                 egarch_form='raw').fit()

# HAR–GARCH
res = arma_garch(y, mean='HAR', lags=[1, 5, 22], vol='GARCH').fit()

# GARCH-M (ARCH-M variance, style EViews)
res = arma_garch(y, mean='AR', ar=1, ma=1, vol='GARCH', in_mean='var').fit()
print(res.coef_table.loc['archm'])

# Prévisions et diagnostics (y mémorisé)
print(res.forecast(h=5))

# Value-at-Risk
print(res.var(alpha=0.05, h=1))
print(res.var_series(alphas=(0.05, 0.01)).head())
print(res.var_backtest(alpha=0.05))
# res.plot_var(alphas=(0.05, 0.01), window=300)
fig = res.plot_diagnostics(lags=20)

# Grille de critères d’information
tab = select_order(y, p_range=range(0, 3), q_range=range(0, 3),
                   criterion='bic', n_jobs=1)
print(tab.attrs['best'])
```

Documentation en ligne : `help(arma_garch)`, `help(ARMAGarch.fit)`, `help(select_order)`.

---

## 9. Notes numériques

1. **Filtre unique.** Résidus et \(\sigma_t^2\) sont produits par une seule
   récursion ARMA + variance par évaluation de vraisemblance (JIT numba optionnel).
2. **Clamping.** \(\sigma_t^2\) est maintenu dans une plage positive large pour
   éviter les NaN en domaine log ; garde numérique, pas contrainte économique.
3. **OPG vs Hessienne.** L’OPG est robuste lorsque la Hessienne est bruitée.
4. **Reproductibilité vs logiciels de référence.** Aligner (i) `backcast`
   (souvent 0.7), (ii) `adjust_sample` avec la fenêtre d’échantillon,
   (iii) listes de lags creux si le logiciel utilise `ma(5)` plutôt que MA(5)
   dense, (iv) EGARCH centré vs brut pour l’affichage de \(\omega\).
5. **Persistance reportée.** \(\sum\alpha+\sum\beta\) (GARCH),
   \(\sum\alpha+\tfrac12\sum\gamma+\sum\beta\) (GJR), ou \(\sum\beta\) (EGARCH).

---

## 10. Limitations

1. La grille `select_order` ne couvre pas les lags creux ; les estimer via `arma_garch`.
2. Les prévisions avec exogènes exigent de fournir le futur \(\mathbf{x}\) ;
   sinon contribution nulle.
3. FIGARCH : polynôme de lags tronqué (défaut 1000).
4. Parallelisme `n_jobs=-1` : pertinent surtout pour de **grandes** grilles ;
   sur petites grilles, `n_jobs=1` est plus rapide (pas de coût de spawn).

---

## 11. Références

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31, 307–327.
2. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48, 1779–1801.
3. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59, 347–370.
4. Ding, Z., Granger, C. W. J., & Engle, R. F. (1993). A long memory property of stock market returns and a new model. *Journal of Empirical Finance*, 1, 83–106.
5. Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996). Fractionally integrated generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 74, 3–30.
6. Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35, 705–730.
7. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7, 174–196.

---

*Documentation du sous-package `makystats.armagarch`. Voir aussi `README.md` dans ce répertoire pour un démarrage rapide.*


## 11 bis. GARCH-M (ARCH-M)

### Définition

Paramètre API : **``in_mean``** ∈ {`None`, `'sigma'`, `'var'`, `'log'`}.

\[
y_t = \mu_t + \mathrm{ARMA}(y,\varepsilon) + \mathbf{x}_t'\boldsymbol{\beta}
      + \lambda\, f(\sigma_t) + \varepsilon_t,
\qquad
f \in \{\sigma_t,\,\sigma_t^2,\,\ln\sigma_t^2\}.
\]

| `in_mean` | \(f(\sigma_t)\) | Équivalent EViews | Label coef |
|-----------|----------------|-------------------|------------|
| `None` (défaut) | — | pas d'ARCH-M | — |
| `'sigma'` | \(\sigma_t\) | Std. Dev. | `archm` |
| `'var'` | \(\sigma_t^2\) | **Variance** | `archm` |
| `'log'` | \(\ln\sigma_t^2\) | Log(Var) | `archm` |

### Implémentation

- \(\sigma_t\) est **causal** (fonction de \(\varepsilon_{t-1},\sigma_{t-1},\ldots\)
  uniquement) : calculé **avant** d’entrer dans la moyenne à la date \(t\)
  (filtre intercalé `_filter_archm_py`).
- Le coefficient s’appelle **`archm`** dans `coef_table` / `params`.
- `fitted_values`, `forecast`, VaR / ES incluent déjà \(\lambda f(\hat\sigma)\).

### Identification

Si \(\sigma_t^2\) varie peu, `const` et `archm` sont **quasi colinéaires**
(\(c + \lambda\,\mathbb{E}[\sigma^2]\) proche d’une constante). Interpréter
\(\lambda\) seulement s’il est significatif ; reporter plutôt le niveau
combiné si besoin.

### API

```python
from makystats import arma_garch, select_order

# Estimation directe
res = arma_garch(y, mean='AR', ar=1, ma=1, vol='GARCH', in_mean='var').fit()
print(res.coef_table.loc['archm'])

# Grille d'ordres : même in_mean pour tous les modèles
tab, best = select_order(
    y, p_range=range(0, 3), q_range=range(0, 2),
    in_mean='var', return_best=True, criterion='bic',
)
# labels du type ARMA(1,1)-garch(1,1)-normal-ARCHM(var)
```

Via la classe :

```python
from makystats import ARMAGarch
m = ARMAGarch(p=1, q=1, r=1, s=1, mean='AR', in_mean='var')
m.fit(y)
```

## 12. Value-at-Risk et Expected Shortfall

Deux modes via ``aggregation`` :

**Marginale** (défaut) — quantile de \(y_{T+h}\) :

\[
\mathrm{VaR}^{(\alpha)}_{T+h} = \hat\mu_{T+h} + \hat\sigma_{T+h}\, q_\alpha(Z),
\qquad
\mathrm{ES}^{(\alpha)}_{T+h} = \hat\mu_{T+h} + \hat\sigma_{T+h}\,\mathbb{E}[Z\mid Z\le q_\alpha].
\]

**Cumulée** (`aggregation='cumulative'`) — quantile de
\(S_h = \sum_{k=1}^{h} y_{T+k}\) obtenu par **simulation Monte Carlo**
de trajectoires sous le modèle (GARCH + moyenne ARMA / x / GARCH-M) :

\[
\mathrm{VaR}^{(\alpha)}_{\mathrm{cum},h}
  = q_\alpha\!\left(S_h^{(1)},\ldots,S_h^{(n_{\mathrm{sims}})}\right).
\]

Paramètres : ``n_sims`` (défaut 10 000), ``seed`` (reproductibilité).
Pour ``h=1``, cumulée ≡ marginale. FIGARCH en cumulé : approximation type GARCH.

### Méthodes

| Méthode | Rôle |
|---------|------|
| `var(..., aggregation, n_sims, seed)` | VaR `marginal` (y_{T+h}) ou `cumulative` (somme MC) |
| `expected_shortfall(...)` | ES / CVaR (mêmes modes) |
| `var_series(alpha\|alphas, ...)` | **Fitted VaR** 1-pas in-sample (DataFrame) |
| `var_backtest(...)` | Taux de violations + LR de Kupiec (**1-pas**) |
| `plot_var(...)` | Graphique dépassements (**1-pas**, styles) |

- **parametric** : \(q_\alpha\) selon `dist` (normal, t, skewt, ged).
- **fhs** : quantile empirique des résidus standardisés (hors burn).
- **Convention** : par défaut quantile de `y` ; `as_loss=True` → perte \(-\mathrm{VaR}_{\mathrm{raw}}\).
- **Horizon / aggregation** :
  - `aggregation='marginal'` (défaut) : quantile de `y_{T+h}` ;
  - `aggregation='cumulative'` : quantile de `S_h` via Monte Carlo.
- **In-sample** : `var_series` / `var_backtest` / `plot_var` restent **1-pas** (pas de cumul).

### Fitted VaR (`var_series`) — **toujours 1-pas in-sample**

``var_series``, ``var_backtest`` et ``plot_var`` portent sur la VaR
**marginale 1-pas** (filtre causal). La VaR **cumulée** multi-périodes
n'existe que via ``var(..., aggregation='cumulative')`` / ``expected_shortfall``.

```python
# Un niveau
df = res.var_series(y, alpha=0.05)
# colonnes : mean, vol, actual, var, var_raw, hit, q_z

# Plusieurs niveaux
df = res.var_series(y, alphas=(0.05, 0.01), index=dates)
# colonnes : var_5, var_raw_5, hit_5, q_z_5, var_1, ...

# Dernier calcul mémorisé
df2 = res.var_fitted_
```

`hit` = 1 si `actual < var_raw` (violation de la queue basse). Les `burn`
premières lignes sont NaN.

### Graphique (`plot_var`) et styles

Les défauts (courbes orange/bleu, carré violet 5 %, croix rouge 1 %) sont dans
`ARMAGarch._default_plot_var_style()`. Surcharge partielle via `style` :

```python
res.plot_var(y, alphas=(0.05, 0.01), style={
    'var_lines': {0.05: {'color': 'darkorange', 'lw': 2}},
    'exceedance': {0.01: {'marker': 'x', 'c': 'crimson', 's': 30}},
    'no_exceedance': {'s': 8, 'alpha': 0.4},
    'mean': {'color': 'gray'},
    'burn': {'alpha': 0.1},
})
ax, data = res.plot_var(y, alphas=(0.05, 0.01), return_data=True)
```

Clés de `style` : `var_lines`, `exceedance`, `no_exceedance`, `mean`, `burn`.

Fenêtre d'affichage : `window=300` (dernières 300), `window=(i0, i1)`, `window=slice(-200, None)`, ou bornes de dates si `index=` est daté. `max_points=N` est un alias de `window=N`.

