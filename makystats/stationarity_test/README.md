# makystats.stationarity_test — unit-root and stationarity tests

Native implementations (no runtime dependency on third-party unit-root
engines for the core statistics). The subpackage is named
`stationarity_test` on purpose so it is not confused with estimation
modules such as `arma`, `ardl`, `var_model`, or `vecm`.

```python
from makystats import (
    adf, adf_sequential,
    kpss, kpss_compare,
    pp, pp_compare,
    dfgls, dfgls_compare,
    za, za_compare,
    engle_granger, unit_root_summary,
)
# aliases: phillips_perron, zivot_andrews_test, df_gls, ers, kpss_test
```

---

## Overview

| Test | H₀ | Main outputs |
|------|----|----------------|
| **ADF** | unit root | τ, MacKinnon p-values, OLS via `.ols` / `.regression` |
| **ADF sequential** | unit root (strategy) | models 3→2→1, Φ₁/Φ₂/Φ₃, path plot |
| **KPSS** | *stationarity* | η, Newey–West lags, OLS demean/detrend, plot |
| **Phillips–Perron** | unit root | Z(τ), Z(ρ), Newey–West, OLS |
| **DF-GLS (ERS)** | unit root | GLS-detrended ADF, MAIC, OLS on \(y^d\) |
| **Zivot–Andrews** | unit root *without* break | endogenous break, OLS, plot |
| **Engle–Granger** | no cointegration | ADF on residuals, EG critical values |

Helpers: `kpss_compare`, `pp_compare`, `dfgls_compare`, `za_compare`,
`unit_root_summary`, `adf_joint_test`, `adf_dx_regression`.

All scalar tests return a `StationarityResult` with `.statistic`, `.pvalue`,
`.critical_values`, `.lags`, `.nobs`, `.reject_null(alpha)`, `.summary()`,
and (where applicable) `.ols` / `.regression.summary()`, `.plot()`,
`.breakpoint` (ZA).

---

## ADF

```python
res = adf(y, regression="ct", maxlag=None, autolag="AIC")
print(res.summary())
print(res.ols.summary())          # ADF regression table
print(res.statistic, res.pvalue)
```

| `regression` | Deterministics |
|--------------|----------------|
| `"n"` | none |
| `"c"` | constant |
| `"ct"` | constant + linear trend |
| `"ctt"` | constant + trend + quadratic trend |

`autolag`: `"AIC"`, `"BIC"`, or `None` / `"NONE"` (fixed `maxlag`).
The time index of `y` is stored for plotting when available.

---

## Sequential ADF strategy

Decision tree over models with trend (3), constant (2), and none (1),
including Dickey–Fuller Φ joint tests:

```python
seq = adf_sequential(y, maxlag=4, alpha=0.05)
print(seq.summary())
print(seq.conclusion, seq.model)   # e.g. I(0) around trend / ct
seq.plot()                         # coloured path on the decision tree

# Inspect a model on the path
print(seq.adf_model(3).summary())
print(seq.adf_model(3).regression.summary())

# Phi joint tests used on the path (when computed)
print(seq.joint("phi3").summary())
# or compute manually:
from makystats import adf_joint_test
print(adf_joint_test(y, hypothesis="phi3", lags=2).summary())
```

---

## KPSS (null = stationarity)

```python
res = kpss(y, regression="c", nlags="auto")
print(res.summary())
print(res.ols.summary())   # demeaning or detrending regression
res.plot()

print(kpss_compare(y).summary())   # c vs ct side by side
```

| `regression` | H₀ |
|--------------|-----|
| `"c"` | level stationarity |
| `"ct"` | trend stationarity |

| `nlags` | Rule |
|---------|------|
| `"auto"` | Hobijn data-dependent |
| `"legacy"` | Schwert-style |
| `int` | fixed Newey–West lag |

---

## Phillips–Perron

Non-parametric (Newey–West / Bartlett) correction of the DF regression.
Reports **Z(τ)** (main statistic) and **Z(ρ)**.

```python
res = pp(y, regression="ct", lags="auto")
print(res.summary())
print(res.extra["z_rho"], res.extra["t_rho_ols"])
print(res.ols.summary())
print(pp_compare(y, regressions=("n", "c", "ct")).summary())
```

Specs: `"n"`, `"c"`, `"ct"`, `"ctt"`.  
`lags`: `"auto"` (Newey–West bandwidth), `"legacy"`, or `int`.

---

## DF-GLS (Elliott–Rothenberg–Stock)

GLS detrending, then ADF **without** constant on the detrended series.

```python
res = dfgls(y, regression="ct", maxlag=None, autolag="MAIC")
print(res.summary())
print(res.ols.summary())           # ADF on y^d
print(res.extra["rho_bar"], res.extra["beta_gls"])
print(dfgls_compare(y).summary())  # c vs ct
```

| `autolag` | Meaning |
|-----------|---------|
| `"AIC"` / `"BIC"` | information criteria |
| `"MAIC"` | Ng–Perron modified AIC (recommended for DF-GLS) |
| `None` | fixed `maxlag` |

When lag length is selected automatically, the search uses the
**OLS-detrended** series (Perron–Qu), while the reported statistic is
always computed on the **GLS-detrended** series (`extra["lag_policy"]`).

---

## Zivot–Andrews

Unit-root test allowing **one** endogenous structural break.

```python
res = za(y, regression="ct", maxlag=None, autolag="t-stat")
print(res.summary())
print(res.breakpoint, res.ols.summary())
res.plot()
print(za_compare(y).summary())   # models c / t / ct
```

---

## Engle–Granger (two series)

```python
res = engle_granger(y, x, trend="c", maxlag=None, autolag="AIC")
print(res.summary())   # H0: no cointegration
```

`trend`: `"n"` / `"c"` / `"ct"`. Critical values are EG-specific
(not plain ADF MacKinnon).

---

## Battery table

```python
print(unit_root_summary(y, regression="c"))
# columns: test, stat, pvalue, lags, ...
```

---

## Design notes

- **Native OLS** for ADF / PP / DF-GLS / ZA / KPSS residual constructions.
- Critical values and p-values follow standard references (MacKinnon for
  ADF/PP; Kwiatkowski et al. for KPSS; ERS for DF-GLS; published ZA tables).
- Results are intended to be **inspectable**: regression tables, residual
  processes, and multi-specification comparisons where useful.
- Optional plotting requires `matplotlib` (`pip install "makystats[plot]"`).

Further examples: notebooks under `exemples/` and the online guide
[https://Lauret-Boris.github.io/makystats/](https://Lauret-Boris.github.io/makystats/).
