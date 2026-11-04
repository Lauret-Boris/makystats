<p align="center">
  <img src="https://raw.githubusercontent.com/Lauret-Boris/makystats/main/docs/logo_makystats.png" alt="makystats logo" width="280"/>
</p>

# makystats

**Boîte à outils d’économétrie** (ARMA, ARMA–GARCH, VAR, VECM, ARDL, tests de racine unitaire / stationnarité) — la famille ARMA–GARCH est estimée par maximum de vraisemblance, avec une API fluide inspirée de [`arch`](https://github.com/bashtage/arch). Voir aussi [`makystats/armagarch/README.md`](makystats/armagarch/README.md) et le guide technique [`makystats/armagarch/DOCUMENTATION.md`](makystats/armagarch/DOCUMENTATION.md).

Le package réalise l’estimation **conjointe** d’une équation de moyenne linéaire (ARMA / ARX / HAR) et d’une équation de variance conditionnelle (GARCH, GJR-TARCH, EGARCH, APARCH, FIGARCH), sous innovations normale, Student-\(t\), skew-\(t\) de Hansen ou GED. L’optimisation est un BFGS **non contraint** dans un espace reparamétré ; les écarts-types utilisent par défaut l’estimateur OPG (*outer product of gradients*).

---


## GARCH-M (ARCH-M)

Paramètre ``in_mean`` : `None` (défaut), `'sigma'`, `'var'` (forme en variance), `'log'`.

```python
from makystats import arma_garch, select_order

res = arma_garch(y, mean='AR', ar=1, ma=1, vol='GARCH', in_mean='var').fit()
# coefficient : archm (λ) dans res.coef_table

tab = select_order(y, p_range=range(0, 3), q_range=range(0, 2), in_mean='var')
```

- \(\sigma_t\) est causal (passé uniquement).
- Si la variance conditionnelle varie peu, `const` et `archm` peuvent être quasi colinéaires : interpréter \(\lambda\) avec prudence.



## Value-at-Risk et Expected Shortfall

Sous le modèle ajusté :

```python
res = arma_garch(y, mean='AR', ar=1, ma=1, vol='GARCH').fit()

res.var(alpha=0.05, h=1)   # marginale (défaut) ; method='parametric'|'fhs'
res.var(alpha=0.05, h=10, aggregation='cumulative', n_sims=10000, seed=1)
res.expected_shortfall(alpha=0.05, h=10, aggregation='cumulative')
df = res.var_series(alphas=(0.05, 0.01))        # fitted VaR → aussi res.var_fitted_
res.var_backtest(alpha=0.05)                    # taux de violations + Kupiec
res.plot_var(alphas=(0.05, 0.01), window=300)   # dernières 300 obs.
```

- `aggregation='marginal'` (défaut) : quantile de **y_{T+h}**.
- `aggregation='cumulative'` : quantile de **sum y** sur h pas (Monte Carlo).
- `var_series` / `plot_var` / `var_backtest` : **toujours 1-pas** in-sample.
- `as_loss=True` : convention perte ; `y`/`x` optionnels après `fit()`.


## Documentation en ligne

Pages utilisateur (FR/EN) : [https://Lauret-Boris.github.io/makystats/](https://Lauret-Boris.github.io/makystats/)  
Générées depuis le dossier `website/` (GitHub Pages). Le dossier `docs/` du dépôt contient les images de formules du README et **n’est pas** la racine Pages.

Guides techniques complets : [VECM](https://github.com/Lauret-Boris/makystats/blob/main/makystats/vecm/DOCUMENTATION.md), [VAR](https://github.com/Lauret-Boris/makystats/blob/main/makystats/var_model/README.md), [ARMA-GARCH](https://github.com/Lauret-Boris/makystats/blob/main/makystats/armagarch/DOCUMENTATION.md), [ARMA](https://github.com/Lauret-Boris/makystats/blob/main/makystats/arma/README.md), [ARDL](https://github.com/Lauret-Boris/makystats/blob/main/makystats/ardl/DOCUMENTATION.md), [stationnarité / racines unitaires](https://github.com/Lauret-Boris/makystats/blob/main/makystats/stationarity_test/README.md) (aussi livrés dans le paquet installé).


## Version 0.1.5

- **Stationnarité** : API enrichie pour KPSS, Phillips–Perron, DF-GLS, ZA (OLS, comparaisons multi-specs, graphiques) ; ADF séquentiel (tests Φ, arbre) ; MAIC et politique de retards Perron–Qu pour le DF-GLS.
- **Documentation** : README `stationarity_test` étendu ; badges et guides alignés sur la 0.1.5.
- (0.1.4) ARDL : intercept de court terme de l'ECM pour les cas PSS 3–5 ; UECM exact pour `order=0` ; notebooks `exemples/ARDL/`.


## Installation

```bash
pip install makystats
pip install "makystats[numba]"   # accélération JIT optionnelle du filtre
pip install "makystats[plot]"    # matplotlib pour plot_diagnostics
```

Installation en mode développement (clone) :

```bash
pip install -e ".[numba,plot,test]"
```

```python
from makystats import arma_garch, select_order, ARMAGarch
```

---


## VAR

Estimation OLS native en niveau (`det='n'|'c'|'t'|'ct'|'ctt'`). Voir [`makystats/var_model/README.md`](makystats/var_model/README.md).

```python
from makystats import VAR, select_var_order

order = select_var_order(data, maxlags=4, criterion='bic')
res = VAR(data, k_ar=order.best, det='c').fit()
res.summary()
res.forecast(5, variables='DTS')
res.irf(10).plot()
res.plot_diagnostics()
```

## VECM (cointégration)

Implémentation VECM / Johansen (régression reduced-rank, cas déterministes JHJ, p-values et valeurs critiques MacKinnon–Haug–Michelis 1999). IRF, FEVD, Granger, JB, whiteness et diagnostics sont **natifs**. Les noms de paramètres suivent l’API makystats (`test_case`, `det_case`), pas statsmodels. Voir aussi [`makystats/vecm/README.md`](makystats/vecm/README.md).

```python
from makystats import VECM, select_coint_rank, select_var_order

p = select_var_order(data, maxlags=4, criterion='bic').best
k = p - 1  # k_ar_diff
rank = select_coint_rank(data, k_ar_diff=k, test_case='3')
model = VECM(data, k_ar_diff=k, coint_rank=max(rank.rank, 1), det_case='3a').fit()
model.summary()
print(model.forecast(steps=5, variables='DTS'))
model.fitted_values(level=True)
model.irf(10).plot()
model.plot_diagnostics()
```

| Paramètre | Valeurs | Rôle |
|-----------|---------|------|
| `test_case` | `'1'`…`'5'` | Spécification déterministe du **test de rang** |
| `det_case` | `'1'`,`'2'`,`'3a'`,`'3b'`,`'4a'`,`'4b'`,`'5a'`,`'5b'` | Spécification déterministe de l’**estimation VECM** (défaut `'3a'`) |


## ARMA / ARIMA / SARIMA (variance constante)

```python
from makystats import arma, arima, select_arma_order

res = arma(y, mean='ARMA', ar=1, ma=1, trend='c').fit()
print(res.params['AR(1)'], res.bse, res.cov_params())
res.test_normality(); res.test_serial_correlation(10)
print(res.arroots)  # |r| > 1 souhaité

res = arma(ylog, mean='ARMA', ar=0, ma=1, d=1,
           seasonal_order=(0, 1, 1, 12), trend='n').fit()
tab = select_arma_order(y, p_range=range(0, 4), q_range=range(0, 3), criterion='bic')
```

Voir [`makystats/arma/README.md`](makystats/arma/README.md). Pour la volatilité conditionnelle : `arma_garch`.

## ARDL (bounds test, ECM, prévisions)

Moteur **OLS natif** : ARDL en niveaux, UECM, multiplicateurs de long terme, test aux bornes
Pesaran / Narayan, ECM, CUSUM, prévisions conditionnelles. Voir
[`makystats/ardl/README.md`](makystats/ardl/README.md) et
[`makystats/ardl/DOCUMENTATION.md`](makystats/ardl/DOCUMENTATION.md).

```python
from makystats import ardl, select_ardl_order

res = ardl(y, x=x, lags=2, order=2, case=3).fit(verbose=False)
res.summary()          # ARDL + diagnostics
res.summary_uecm()     # UECM + long run
res.summary_ecm()
res.summary_bounds()   # Pesaran / Narayan
print(res.params, res.bse, res.diagnostic_summary())

tab = select_ardl_order(y, x=x, maxlag=4, maxorder=2, case=3, ic="bic", verbose=False)
# maxorder={"R": 2, "oil": 1}  # retards max par variable

res.long_run_fitted()
res.restrict_ecm(pvalue=0.10, keep_levels=True)
res.forecast(steps=8, exog_future=exog_f, conf_int=True)
res.forecast_ecm(steps=8, exog_future=exog_f, restricted=True, level=True)
```

`case=1…5` suit la spécification PSS (None / Rest. Constant / Constant /
Rest. Trend / Trend). Les régresseurs `fixed=` n’entrent pas dans la relation de
long terme ni dans le F aux bornes.


## 1. Modèle statistique

### 1.1 Équation de moyenne

Pour les observations \(y_t\), \(t = 1,\ldots,n\) :

\[
y_t = \mu + \sum_{i\in\mathcal{I}_{\mathrm{AR}}} \phi_i (y_{t-i}-\mu)
    + \sum_{j\in\mathcal{I}_{\mathrm{MA}}} \theta_j \varepsilon_{t-j}
    + \mathbf{x}_t'\boldsymbol{\beta}
    + \varepsilon_t.
\]

- \(\mathcal{I}_{\mathrm{AR}}\) et \(\mathcal{I}_{\mathrm{MA}}\) peuvent être consécutifs \(\{1,\ldots,p\}\) ou **creux** (ex. \(\{5\}\) seulement).
- **ARX** : termes AR + exogènes \(\mathbf{x}_t\), sans MA.
- **ARMAX** : AR + MA + \(\mathbf{x}_t\).
- **LS** : constante + \(\mathbf{x}_t\) uniquement.
- **HAR / HARX** (style Corsi) : régresseurs = moyennes mobiles causales de \(y\) aux horizons fournis (défaut \(1,5,22\)) ; l’estimation se fait comme un LS + GARCH sur l’échelle des résidus.

**Centrage.** Avec `center=True` (défaut), \(y\) est déméanisée pendant l’optimisation pour stabiliser le conditionnement numérique ; l’intercept est restauré à l’échelle d’origine après convergence. `mean='Zero'` impose \(\mu=0\).

### 1.2 Variance conditionnelle

On pose \(\varepsilon_t = \sigma_t z_t\) avec \(z_t\) i.i.d. sous la loi choisie (espérance nulle, variance unitaire après standardisation le cas échéant).

| `vol` | Récursion (schéma) |
|-------|---------------------|
| **GARCH(\(r,s\))** | \(\sigma_t^2 = \omega + \sum_{k=1}^{r}\alpha_k\varepsilon_{t-k}^2 + \sum_{\ell=1}^{s}\beta_\ell\sigma_{t-\ell}^2\) |
| **GJR / TARCH** | GARCH + \(\sum_k \gamma_k\mathbf{1}_{\{\varepsilon_{t-k}<0\}}\varepsilon_{t-k}^2\) |
| **EGARCH** (Nelson 1991) | \(\log\sigma_t^2 = \omega + \sum_k\alpha_k(\lvert z_{t-k}\rvert-\mathbb{E}\lvert z\rvert) + \sum_k\gamma_k z_{t-k} + \sum_\ell\beta_\ell\log\sigma_{t-\ell}^2\) |
| **APARCH** (Ding–Granger–Engle) | \(\sigma_t^\delta = \omega + \sum_k\alpha_k(\lvert\varepsilon_{t-k}\rvert-\gamma_k\varepsilon_{t-k})^\delta + \sum_\ell\beta_\ell\sigma_{t-\ell}^\delta\) |
| **FIGARCH(\(p,d,q\))** (Baillie–Bollerslev–Mikkelsen) | \(\sigma_t^2 = \omega + \beta(L)\sigma_t^2 + \lambda(L)\varepsilon_t^2\) avec poids fractionnaires issus de \((1-L)^d\) (polynôme tronqué, troncature par défaut 1000) |

**Formes d’affichage EGARCH.** En interne, le filtre utilise la spécification **centrée** de Nelson (terme de magnitude \(\lvert z\rvert-\mathbb{E}\lvert z\rvert\)). Avec `egarch_form='raw'`, seul l’intercept **affiché** est décalé :

\[
\omega_{\mathrm{raw}} = \omega_{\mathrm{centré}} - \mathbb{E}\lvert z\rvert\sum_k\alpha_k,
\]

de sorte que l’équation imprimée corresponde à l’écriture non centrée \(\log\sigma_t^2=\omega+\sum\alpha_k\lvert z_{t-k}\rvert+\cdots\). La vraisemblance et les autres coefficients restent identiques.

**Restrictions de signe.** Pour un GARCH multi-lags, les \(\alpha_k,\beta_\ell\) **individuels** ne sont pas forcés positifs (seul \(\omega>0\) via \(\exp\)) ; des coefficients négatifs restent admissibles tant que \(\sigma_t^2\) reste positif dans le filtre (clamp purement numérique). Cela rejoint la pratique du MV non contraint lorsque le polynôme de retards est d’ordre élevé. Le \(\gamma\) du GJR est libre ; le \(\gamma\) APARCH est ramené dans \((-1,1)\) par \(\tanh\) ; le \(d\) FIGARCH est borné de façon souple pour la stabilité.

### 1.3 Lois d’innovation

Les résidus standardisés \(z_t=\varepsilon_t/\sigma_t\) entrent dans la log-vraisemblance comme suit (à des constantes indépendantes des paramètres près) :

| `dist` | Densité (forme standardisée utilisée) |
|--------|----------------------------------------|
| **normale** | \(-\tfrac12\log(2\pi)-\tfrac12\log\sigma_t^2-\varepsilon_t^2/(2\sigma_t^2)\) |
| **Student-\(t\)** | forme \(t\) avec \(\nu>2\), mise à l’échelle pour \(\mathrm{Var}(z)=1\) |
| **skew-\(t\)** (Hansen 1994) | \(\nu>2\), asymétrie \(\lambda\in(-1,1)\) |
| **GED** | forme \(\nu>0\) ; \(\nu=2\) retrouve la gaussienne |

Les paramètres de forme sont reparamétrés pour que le BFGS reste dans un ouvert non contraint (ex. \(\nu=2+e^{\tilde\nu}\) pour la loi \(t\)).

---

## 2. Construction de la vraisemblance et traitement de l’échantillon

### 2.1 Log-vraisemblance

L’objectif maximisé est la log-vraisemblance conditionnelle (normale / \(t\) / skew-\(t\) / GED)

\[
\mathcal{L}(\boldsymbol{\theta}) = \sum_{t=b+1}^{n} \ell(y_t\mid\mathcal{F}_{t-1};\boldsymbol{\theta}),
\]

où \(b\) est le burn-in (ci-dessous) et \(\ell\) utilise \(\varepsilon_t(\boldsymbol{\theta})\) et \(\sigma_t^2(\boldsymbol{\theta})\) issus du filtre conjoint.

### 2.2 Burn-in et `adjust_sample`

| Option | Comportement |
|--------|----------------|
| `adjust_sample=True` (défaut) | \(b=\max(\text{lags AR},\,\text{lags MA},\,\texttt{hold\_back})\). Les \(b\) premières observations initialisent les retards ; elles **n’entrent pas** dans \(\mathcal{L}\). Analogie avec l’« échantillon ajusté » de la pratique ARMA classique. |
| `adjust_sample=False` | \(b=0\). Les \(n\) termes entrent dans \(\mathcal{L}\) ; les états pré-échantillon AR/MA/variance sont fournis par backcast. Adapté lorsque l’échantillon de référence est \(1\ldots n\) sans ajustement. |

Pour le HAR, si `adjust_sample=True`, on impose au moins \(b=\max(\text{horizons})\), car les moyennes glissantes sont incomplètes en début de série.

### 2.3 Pré-échantillon MA (récursion inverse)

En présence de termes MA, les innovations pré-échantillon **ne sont pas** mises à zéro. L’algorithme :

1. construit les résidus AR-seulement \(u_t\) sur la fenêtre d’estimation ;
2. effectue une récursion **arrière** pour obtenir une trajectoire de backcast \(e^{\mathrm{bc}}_t\) ;
3. prolonge cette trajectoire **avant** l’échantillon (bruit moteur nul) pour former les états MA pré-échantillon \(e^{\mathrm{ps}}\) ;
4. lance la récursion ARMA **avant** en utilisant \(e^{\mathrm{ps}}\) pour les retards qui tombent avant l’indice de burn.

Cela évite le biais d’un démarrage à innovations nulles pour les modèles MA ou ARMA purs, et pèse sur le niveau de vraisemblance lorsque l’échantillon effectif est court.

### 2.4 Backcast de variance

La variance conditionnelle pré-échantillon utilise un lissage exponentiel des carrés de résidus (paramètre \(\lambda=\) `backcast`, défaut \(0.7\)) :

\[
h_0 = \lambda^T \bar{e}^2 + (1-\lambda)\sum_{j=0}^{T-1}\lambda^j e_j^2,
\]

avec \(j=0\) le résidu **le plus ancien** de la fenêtre d’estimation (poids décroissants vers le présent). \(\lambda=1\) se réduit à \(h_0=\bar{e}^2\). Dans le filtre, les \(\varepsilon^2\) et \(\sigma^2\) retardés qui tombent dans la zone de burn sont remplacés par \(h_0\).

Pour les modèles type GARCH, \(h_0\) est en outre **recalculé à partir des résidus courants** à chaque évaluation de vraisemblance, afin que le backcast reste cohérent avec les paramètres de moyenne en cours.

---

## 3. Stratégie d’optimisation

### 3.1 BFGS non contraint + reparamétrisation

Les paramètres \(\boldsymbol{\theta}\) vivent dans un espace non contraint ; une application régulière \(\boldsymbol{\theta}\mapsto\boldsymbol{\psi}\) produit les coefficients économiques \(\boldsymbol{\psi}\) (AR, MA, variance, forme) :

| Bloc | Application (indicative) |
|------|---------------------------|
| Coefficients AR / MA | \(\phi=0.99\tanh(\tilde\phi)\) (boîte favorable à la stationnarité) |
| \(\omega\) (famille GARCH) | \(\omega=\exp(\tilde\omega)\) |
| \(\alpha,\beta\) (GARCH multi-lags) | libres (signés) ; le filtre maintient \(\sigma_t^2>0\) |
| EGARCH \(\beta\) | positifs, calés pour une persistance \(<1\) |
| APARCH \(\gamma\) | \(\tanh\) vers \((-1,1)\) |
| FIGARCH \(d\) | application souple vers un intervalle stable |
| degrés de liberté \(t\) | \(\nu=2+e^{\tilde\nu}\) |

Aucune contrainte d’inégalité n’est passée à SciPy ; le BFGS (gradient par différences finies) opère uniquement sur \(\boldsymbol{\theta}\). Les arguments de `exp` sont bornés pour éviter les débordements pendant les pas exploratoires.

**Polish.** Si le premier BFGS ne signale pas le succès (fréquent avec gradients numériques : « precision loss »), un second BFGS repart du point atteint.

### 3.2 Valeurs initiales

- ARMA dense : ARIMA préliminaire (statsmodels) lorsque les lags sont \(1\ldots p\), \(1\ldots q\).
- ARMA creux : défauts neutres (AR/MA petits, variance issue de la variance résiduelle).
- Variance : \(\alpha\) petit, \(\beta\) modéré, \(\omega\) calé sur la variance résiduelle (APARCH : \(\omega\propto\hat\sigma^\delta\)).

### 3.3 Écarts-types

Par défaut `cov_type='opg'` : BHHH / produit extérieur des gradients,

\[
\widehat{\mathrm{Var}}(\hat{\boldsymbol{\theta}})
= \Bigl(\sum_{t} \mathbf{g}_t\mathbf{g}_t'\Bigr)^{-1},
\]

avec \(\mathbf{g}_t=\partial\ell_t/\partial\boldsymbol{\theta}\) évalué numériquement au MV (rapporté à l’échelle des paramètres économiques). Alternative : `cov_type='hessian'` (Hessienne numérique de l’objectif).

---

## 4. Après estimation

### 4.1 Séries de sortie (longueur \(n\))

| Attribut | Contenu |
|----------|---------|
| `residuals` | \(\hat\varepsilon_t\) |
| `std_resid` | \(\hat\varepsilon_t/\hat\sigma_t\) |
| `conditional_vol` | \(\hat\sigma_t\) |
| `fitted_values` | \(y_t-\hat\varepsilon_t\) (prévision de moyenne à un pas) |
| `coef_table` | coefficients, ET, \(z\), \(p\)-values, IC 95 % |

### 4.2 Diagnostics

- **Ljung–Box** sur les résidus standardisés et sur leurs carrés (lag 10 mémorisé ; tables complètes dans `ljungbox_resid`, `ljungbox_resid2`).
- **Jarque–Bera** sur les résidus standardisés.
- **`plot_diagnostics()`** : figure \(2\times 3\) — trajectoire des résidus, histogramme + \(\mathrm{N}(0,1)\) + KDE, QQ-plot normal, ACF de \(z_t\), ACF de \(z_t^2\), volatilité conditionnelle.

### 4.3 Prévisions

`forecast(y, h, threshold=0.0, x=None)` renvoie un DataFrame aux colonnes `mean`, `variance`, `vol`, `p_up` pour les horizons \(1\ldots h\) :

- **Moyenne** : récursion ARMA aux coefficients estimés ; chocs futurs nuls (MSE minimale sous perte symétrique).
- **Variance** : récursion de l’équation de variance choisie, avec \(\varepsilon_{T+k}^2\) remplacé par \(\sigma_{T+k}^2\) pour \(k\ge 1\) le cas échéant.
- **`p_up`** = \(P(y_{T+h}>\texttt{threshold})\) sous la loi d’innovation ajustée (CDF normale / \(t\) ; GED et skew-\(t\) : approximations cohérentes).
- Les prévisions avec exogènes exigent le \(\mathbf{x}\) futur (`x` à \(h\) lignes) ; à défaut, la contribution exogène est mise à 0.

### 4.4 Sélection d’ordres (`select_order`)

Grille cartésienne sur les ordres AR, MA, ARCH, GARCH (et éventuellement des listes `dist` / `variance_model`). Chaque candidat est estimé ; les critères d’information sont **recalculés sur un burn-in commun** afin de comparer des modèles de longueurs de lags différentes. Critères : AIC, BIC, HQIC. Exécution par défaut **séquentielle** (`n_jobs=1`) ; le parallélisme multi-processus n’est activé que pour de *grandes* grilles sur machines multi-cœurs (les petites grilles restent séquentielles pour éviter le coût de spawn de processus).

**Limite :** la grille porte sur des ordres **denses** \(0,1,\ldots\). Les spécifications creuses du type `ma=[5]` s’estiment directement via `arma_garch`.

---

## 5. Notes numériques et d’implémentation

1. **Filtre unique.** Résidus et \(\sigma_t^2\) sont produits par une seule récursion ARMA + variance par appel de vraisemblance (JIT **numba** optionnel).
2. **Clamp.** \(\sigma_t^2\) est maintenu dans une plage positive large pour éviter les NaN en log ; garde numérique, non contrainte économique.
3. **OPG vs Hessienne.** L’OPG est robuste lorsque la Hessienne est bruitée ; asymptotiquement les deux coïncident sous régularité.
4. **Comparabilité avec d’autres logiciels.** Pour coller à un écran de référence, aligner en général (i) `backcast` (souvent 0,7), (ii) `adjust_sample` avec la fenêtre d’échantillon, (iii) les listes de lags creux lorsqu’un seul `ma(5)` est spécifié plutôt qu’un MA(5) dense, (iv) l’affichage EGARCH centré vs brut.
5. **Persistance.** Rapportée comme \(\sum\alpha+\sum\beta\) (GARCH), \(\sum\alpha+\tfrac12\sum\gamma+\sum\beta\) (GJR), ou \(\sum\beta\) (EGARCH).

---

## 6. Aperçu de l’API

```python
from makystats import arma_garch, select_order

# ARMA(1,1)–GARCH(1,1)–Normale
res = arma_garch(y, mean='ARMA', ar=1, ma=1, vol='GARCH', dist='normal',
                 backcast=0.7).fit()
print(res.coef_table)
print(res.loglik, res.aic, res.bic)

# MA(5) creux + GARCH, échantillon complet dans la vraisemblance
res = arma_garch(y, mean='Constant', ma=[5], vol='GARCH',
                 adjust_sample=False, backcast=0.7).fit()

# Asymétrie + queues épaisses
res = arma_garch(y, mean='AR', ar=1, vol='GJR', o=1, dist='t').fit()

# EGARCH, affichage omega brut
res = arma_garch(y, mean='ARMA', ar=1, ma=1, vol='EGARCH',
                 egarch_form='raw').fit()

# HAR–GARCH
res = arma_garch(y, mean='HAR', lags=[1, 5, 22], vol='GARCH').fit()

# Prévisions et diagnostics
print(res.forecast(y, h=5))
fig = res.plot_diagnostics(lags=20)

# Grille de critères d’information
tab = select_order(y, p_range=range(0, 3), q_range=range(0, 3),
                   criterion='bic', n_jobs=1)
print(tab.attrs['best'])
```

Documentation intégrée : `help(arma_garch)`, `help(ARMAGarch.fit)`, `help(select_order)`.

---

## 7. Organisation du package et tests

```text
makystats/
  __init__.py
  armagarch/            # ARMA–GARCH (MLE)
    __init__.py
    core.py
  vecm/                 # VECM / Johansen
    __init__.py
    core.py
tests/
examples/
pyproject.toml
README.md
README.fr.md
```

```bash
python -m pytest tests/ -q
python tests/test_vecm.py
python examples/recipes.py
```

## 8. Références (méthodes implémentées)

- Bollerslev, T. (1986), Generalized autoregressive conditional heteroskedasticity, *Journal of Econometrics*.
- Glosten, L., Jagannathan, R., Runkle, D. (1993), On the relation between the expected value and the volatility of the nominal excess return on stocks, *Journal of Finance* (GJR).
- Nelson, D. (1991), Conditional heteroskedasticity in asset returns: a new approach, *Econometrica* (EGARCH).
- Ding, Z., Granger, C., Engle, R. (1993), A long memory property of stock market returns and a new model, *Journal of Empirical Finance* (APARCH).
- Baillie, R., Bollerslev, T., Mikkelsen, H. (1996), Fractionally integrated generalized autoregressive conditional heteroskedasticity, *Journal of Econometrics* (FIGARCH).
- Hansen, B. (1994), Autoregressive conditional density estimation, *International Economic Review* (skew-\(t\)).
- Corsi, F. (2009), A simple approximate long-memory model of realized volatility, *Journal of Financial Econometrics* (HAR).
- Berndt, E., Hall, B., Hall, R., Hausman, J. (1974), Estimation and inference in nonlinear structural models, *Annals of Economic and Social Measurement* (BHHH / OPG).

---

## Licence

MIT (voir `pyproject.toml`). Version **0.1.5**.
