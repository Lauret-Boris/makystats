# ARDL — documentation technique (makystats)

Moteur **natif OLS** : ARDL en niveaux, reparamétrisation UECM, test aux bornes
Pesaran–Shin–Smith (PSS) et valeurs critiques Narayan (petits échantillons),
ECM, prévisions conditionnelles.

## 1. Modèle ARDL en niveaux

\[
y_t = c + \delta t + \sum_{i=1}^{p} \phi_i y_{t-i}
+ \sum_{j=1}^{k} \sum_{\ell=0}^{q_j} \beta_{j,\ell} x_{j,t-\ell}
+ \gamma' z_t + \varepsilon_t
\]

- \(z_t\) : régresseurs **fixed** (hors cointégration), contemporains uniquement.
- `case` (1…5) détermine la présence de \(c\) et \(t\) (spécification PSS).

## 2. UECM (reparamétrisation)

\[
\Delta y_t = c' + \delta' t + \alpha y_{t-1} + \sum_{j=1}^{k} \pi_j x_{j,t-1}
+ \text{dynamique de court terme} + \varepsilon_t
\]

avec \(\alpha = \sum_i \phi_i - 1\) (vitesse d’ajustement).

Les `fixed` restent en niveau contemporain dans l’UECM ; ils **ne** font **pas**
partie des restrictions du F aux bornes.

## 3. Multiplicateurs de long terme

\[
\theta_j = \frac{\sum_{\ell=0}^{q_j} \beta_{j,\ell}}{1 - \sum_{i=1}^{p} \phi_i}
\]

Écart-types par **delta method** (matrice de covariance OLS).  
Équilibre et ECT :

\[
y^*_t = \theta_0 + \theta_t t + \sum_j \theta_j x_{j,t}, \qquad
\mathrm{ECT}_t = y_t - y^*_t
\]

API : `long_run`, `long_run_table`, `long_run_fitted()`, `plot_long_run_fitted()`.

## 4. Test aux bornes (PSS)

Statistique F sur \(H_0 : \alpha = \pi_1 = \cdots = \pi_k = 0\)
(et restrictions supplémentaires selon le case 2 / 4).

- Valeurs critiques **Pesaran et al.** (asymptotiques)
- Tables **Narayan (2005)** (petits \(n\))

Interprétation usuelle à 5 % : F > I(1) → cointégration ;
F < I(0) → pas de relation de long terme ; zone grise sinon.

Le F est calculé sur le modèle **estimé** (y compris `fixed` dans les deux
régressions, restreinte et non restreinte). `k` = nombre de variables dans `x`
uniquement.

## 5. ECM restreint

`restrict_ecm(drop=..., pvalue=..., keep_levels=True)` ré-estime l’UECM en
retirant des colonnes.

- Par défaut, \(\alpha\) (`y.L1`) et les niveaux `x.L1` sont **protégés**.
- Le F aux bornes du `fit()` **n’est pas** recalculé (reste le modèle général).
- Prudence : pré-tests, tailles distordues.

## 6. Prévisions

### ARDL niveaux — `forecast`

Récurrence exacte de l’équation en niveaux.  
**Conditionnelle** aux trajectoires `exog_future` (et `fixed_future` si besoin).

IC gaussiens : variance multi-pas via poids \(\psi\) de la partie AR sur \(y\)
(les x futurs sont traités comme connus).

### UECM / ECM — `forecast_ecm`

\[
\widehat{\Delta y}_{T+h} \rightarrow
\hat y_{T+h} = y_T + \sum_{m=1}^{h} \widehat{\Delta y}_{T+m}
\]

- `level=True` (défaut) : `forecast` = **niveau**
- `level=False` : `forecast` = **Δy**
- Colonnes toujours présentes : `forecast_level`, `forecast_diff`
- `restricted=True` : coefficients de `restrict_ecm()` si disponible

Sans `x` futurs alors que le modèle en contient → `ValueError`.

## 7. Diagnostics

| Méthode | Contenu |
|---------|---------|
| `diagnostic_summary(lags_lb=10, nlags_arch=5)` | Tableau unique : JB + Ljung-Box + ARCH-LM |
| `test_normality()` | Jarque–Bera, skew, kurtosis |
| `test_serial_correlation(lags)` | Ljung–Box (+ ACF) |
| `test_heteroskedasticity(nlags)` | ARCH-LM (Engle) |
| `plot_diagnostics()` | Graphiques résiduels |
| `cusum` / `plot_cusum(alpha=0.05)` | CUSUM et CUSUM of Squares (bandes 5 % ou 10 %) |

Les p-values de `diagnostic_summary` coïncident avec celles des tests
individuels (dernier lag pour LB et ARCH-LM).

## 8. Récupération des coefficients

Après `fit()` :

```python
res.params["y.L1"]
res.bse; res.pvalues; res.tvalues
res.cov_params()
res.conf_int(alpha=0.05)
res.rsquared; res.rsquared_adj; res.sigma2
```

## 9. Références

- Pesaran, M.H., Shin, Y., Smith, R.J. (2001), Bounds testing approaches to the
  analysis of level relationships, *Journal of Applied Econometrics*.
- Narayan, P.K. (2005), The saving and investment nexus for China…, *Applied Economics*.
- Pesaran, M.H., Shin, Y. (1999), An autoregressive distributed lag modelling
  approach to cointegration analysis.
