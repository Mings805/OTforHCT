"""
OTfunc.py

Core utilities for:
  - Optimal transport (OT) weights and OT-based borrowing
  - Balance diagnostics (mean differences, SMD, ESS)
  - Data-generating processes (original paper setup + variants)
  - Monte Carlo experiments such as MSE vs lambda

This module is designed to be consistent with the Python code
used for the original OT paper simulations (Section 6 of arXiv:2505.00217).
"""

import numpy as np
import pandas as pd


# =========================================================
# Optional imports (POT, cvxpy) guarded by try/except
# =========================================================

def _try_import_pot() -> bool:
    """Return True if POT (Python Optimal Transport) is available."""
    try:
        import ot  # noqa: F401
        import ot.unbalanced  # noqa: F401
        return True
    except Exception:
        return False


def _try_import_cvxpy() -> bool:
    """Return True if cvxpy is available."""
    try:
        import cvxpy as cp  # noqa: F401
        return True
    except Exception:
        return False


# =========================================================
# Basic utilities
# =========================================================

def inv_logit(z):
    """Stable logistic inverse: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_scalar(fun,
                     low: float = -15.0,
                     high: float = 15.0,
                     tol: float = 1e-10,
                     max_iter: int = 200) -> float:
    """
    One-dimensional root finder using bisection with automatic bracket expansion.

    Finds a scalar a in [low, high] such that fun(a) ≈ 0, expanding the
    bracket if the signs at the endpoints do not differ.
    """
    low = float(low)
    high = float(high)
    fl = fun(low)
    fh = fun(high)

    # Expand bracket if needed
    b = 16.0
    it = 0
    while fl * fh > 0 and it < 25:
        low, high = -b, b
        fl, fh = fun(low), fun(high)
        b += 2.0
        it += 1

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        fm = fun(mid)
        if abs(fm) < tol:
            return mid
        if fl * fm <= 0:
            high, fh = mid, fm
        else:
            low, fl = mid, fm
    return 0.5 * (low + high)


def calibrate_intercept(X,
                        beta,
                        target: float,
                        low: float = -15.0,
                        high: float = 15.0,
                        tol: float = 1e-10,
                        max_iter: int = 200) -> float:
    """
    Calibrate a logistic intercept a such that

        mean(inv_logit(a + X @ beta)) ≈ target.
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    xb = X @ beta

    def f(a):
        return inv_logit(a + xb).mean() - target

    return calibrate_scalar(f, low=low, high=high, tol=tol, max_iter=max_iter)


def build_poly2(X, interactions: bool = True):
    """
    Build degree-2 polynomial features with optional pairwise interactions.

    This is used only for OT cost geometry; balance diagnostics use the
    original covariates.
    """
    X = np.asarray(X, dtype=float)
    Fe = [X, X ** 2]
    if interactions and X.shape[1] >= 2:
        inter = []
        p = X.shape[1]
        for i in range(p):
            for j in range(i + 1, p):
                inter.append((X[:, i] * X[:, j])[:, None])
        if inter:
            Fe.append(np.hstack(inter))
    return np.hstack(Fe)


def poly2_features(X, interactions: bool = True):
    """
    Alias for backward compatibility with earlier code that used poly2_features.
    """
    return build_poly2(X, interactions=interactions)


# =========================================================
# Cost geometry
# =========================================================

def build_cost_matrix(X_ec,
                      X_tgt,
                      metric: str = "mahalanobis",
                      augment: bool = True,
                      scale_cost: bool = True):
    """
    Build a non-negative cost matrix C between EC and target covariates.

    Parameters
    ----------
    X_ec : array-like, shape (nE, p)
        External control covariates.
    X_tgt : array-like, shape (nR, p)
        Target covariates (typically the full RCT sample).
    metric : {"mahalanobis", "euclidean"}
        Distance used in the feature space.
        - "mahalanobis": squared Mahalanobis distance in polynomial features.
        - "euclidean"  : squared Euclidean distance on standardized features.
    augment : bool
        If True, use quadratic + interaction features via build_poly2.
    scale_cost : bool
        If True, divide C by median positive entry to stabilize the
        scale of the entropic parameter eps.

    Returns
    -------
    C : ndarray, shape (nE, nR)
        Non-negative cost matrix.
    """
    Xe = np.asarray(X_ec, dtype=float)
    Xt = np.asarray(X_tgt, dtype=float)

    XeF = build_poly2(Xe, interactions=True) if augment else Xe
    XtF = build_poly2(Xt, interactions=True) if augment else Xt

    metric_lower = str(metric).lower()
    if metric_lower == "mahalanobis":
        XY = np.vstack([XeF, XtF])
        S = np.cov(XY.T)
        S = S + 1e-6 * np.eye(S.shape[0])
        Sinv = np.linalg.inv(S)

        qe = np.sum((XeF @ Sinv) * XeF, axis=1)[:, None]
        qt = np.sum((XtF @ Sinv) * XtF, axis=1)[None, :]
        C = qe + qt - 2.0 * (XeF @ Sinv @ XtF.T)
        C = np.maximum(C, 0.0)
    else:
        XY = np.vstack([XeF, XtF])
        std = XY.std(axis=0, ddof=1)
        std[std == 0.0] = 1.0
        mean = XY.mean(axis=0)

        XeS = (XeF - mean) / std
        XtS = (XtF - mean) / std

        Ae = np.sum(XeS * XeS, axis=1)[:, None]
        Rt = np.sum(XtS * XtS, axis=1)[None, :]
        C = Ae + Rt - 2.0 * (XeS @ XtS.T)
        C = np.maximum(C, 0.0)

    if scale_cost:
        pos = C[C > 0]
        if pos.size > 0:
            med = np.median(pos)
            if med > 0:
                C = C / med

    return C


# =========================================================
# Weighting schemes
# =========================================================

def weights_uot_symmetric(X_ec,
                          X_tgt,
                          eps: float,
                          reg_m: float = 1.0,
                          metric: str = "mahalanobis",
                          augment: bool = True,
                          clip_ratio: float = 10.0,
                          scale_cost: bool = True):
    """
    Symmetric unbalanced Sinkhorn (requires POT).

    The unbalanced Sinkhorn coupling gamma solves a regularized OT problem
    between the EC and target covariates. The row sums of gamma define
    weights on the EC sample.

    Parameters
    ----------
    X_ec, X_tgt : array-like
        EC and target covariates.
    eps : float
        Entropic regularization parameter.
    reg_m : float
        Unbalanced marginal penalty for both sides (POT's reg_m).
    metric, augment, scale_cost : see build_cost_matrix.
    clip_ratio : float or None
        If not None, cap weights at clip_ratio / nE and renormalize.

    Returns
    -------
    dict with keys:
        "w"     : weights on EC (array of length nE, sum to 1).
        "gamma" : transport plan (nE x nR).
        "C"     : cost matrix (nE x nR).
    """
    if not _try_import_pot():
        raise ImportError(
            "POT is required for ot_kind='uot'. Install via `pip install POT`."
        )

    import ot  # type: ignore  # noqa: F401
    import ot.unbalanced  # type: ignore  # noqa: F401

    C = build_cost_matrix(
        X_ec,
        X_tgt,
        metric=metric,
        augment=augment,
        scale_cost=scale_cost,
    )
    nE, nR = X_ec.shape[0], X_tgt.shape[0]
    a = np.full(nE, 1.0 / nE)
    b = np.full(nR, 1.0 / nR)

    gamma = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C, reg=eps, reg_m=reg_m, verbose=False
    )
    w = np.maximum(gamma.sum(axis=1), 0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(nE, 1.0 / nE)

    if clip_ratio is not None and np.isfinite(clip_ratio) and clip_ratio > 0:
        cap = clip_ratio / nE
        w = np.minimum(w, cap)
        w = w / w.sum()

    return {"w": w, "gamma": gamma, "C": C}


def weights_semi_relaxed_closed_form(X_ec,
                                     X_tgt,
                                     eps: float,
                                     metric: str = "mahalanobis",
                                     augment: bool = True,
                                     clip_ratio: float = 10.0,
                                     scale_cost: bool = True):
    """
    Semi-relaxed entropic OT via closed-form kernel scaling.

    Column marginal (target) is constrained to be uniform; the row sums of
    the kernel-scaled plan define weights on EC.

    Parameters
    ----------
    X_ec, X_tgt : array-like
        EC and target covariates.
    eps : float
        Entropic regularization parameter.
    metric, augment, scale_cost : see build_cost_matrix.
    clip_ratio : float or None
        If not None, cap weights at clip_ratio / nE and renormalize.

    Returns
    -------
    dict with keys:
        "w"          : weights on EC (array, length nE, sum to 1).
        "gamma"      : scaled kernel (not mass-conserving in general).
        "C"          : cost matrix.
        "col_L1_dev" : L1 deviation from the target column marginal.
    """
    C = build_cost_matrix(
        X_ec,
        X_tgt,
        metric=metric,
        augment=augment,
        scale_cost=scale_cost,
    )
    nE, nR = X_ec.shape[0], X_tgt.shape[0]

    K = np.exp(-C / eps)
    b = np.full(nR, 1.0 / nR)
    colsum = K.sum(axis=0) + 1e-12
    v = b / colsum
    w = K @ v
    w = np.maximum(w, 0.0)
    w = w / w.sum()

    if clip_ratio is not None and np.isfinite(clip_ratio) and clip_ratio > 0:
        cap = clip_ratio / nE
        w = np.minimum(w, cap)
        w = w / w.sum()

    gamma = K * v
    col_dev = np.linalg.norm(gamma.sum(axis=0) - b, ord=1)

    return {"w": w, "gamma": gamma, "C": C, "col_L1_dev": float(col_dev)}


def weights_entropy_balancing(X_ec,
                              X_tgt,
                              degree: int = 1,
                              interactions: bool = False,
                              delta: float = 1e-3,
                              ridge: float = 1e-6,
                              clip_ratio: float = 10.0):
    """
    Entropy balancing weights with approximate moment constraints.

    Solve:
        minimize_{w}  sum_i w_i log w_i + ridge * ||w - 1/n||_2^2
        subject to    w >= 0, sum w = 1,
                      || Phi(X_ec)^T w - mean(Phi(X_tgt)) ||_inf <= delta
    where Phi includes powers and interactions up to the specified degree.
    """
    if not _try_import_cvxpy():
        raise ImportError(
            "cvxpy is required for ot_kind='eb'. Install via `pip install cvxpy`."
        )

    import cvxpy as cp  # type: ignore  # noqa: F401

    Xe = np.asarray(X_ec, dtype=float)
    Xt = np.asarray(X_tgt, dtype=float)
    nE = Xe.shape[0]
    u = np.full(nE, 1.0 / nE)

    def _phi(X):
        X = np.asarray(X, dtype=float)
        Z = [X]
        if degree >= 2:
            Z.append(X ** 2)
            if interactions and X.shape[1] >= 2:
                inter = []
                p = X.shape[1]
                for i in range(p):
                    for j in range(i + 1, p):
                        inter.append((X[:, i] * X[:, j])[:, None])
                if inter:
                    Z.append(np.hstack(inter))
        return np.hstack(Z)

    Phi_e = _phi(Xe)
    Phi_t = _phi(Xt)
    m_t = Phi_t.mean(axis=0)

    w = cp.Variable(nE, nonneg=True)
    constraints = [
        cp.sum(w) == 1.0,
        cp.norm_inf(Phi_e.T @ w - m_t) <= delta,
    ]
    obj = cp.Minimize(cp.sum(cp.rel_entr(w, 1.0)) + ridge * cp.sum_squares(w - u))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError(
            "Entropy balancing infeasible; try larger `delta`, smaller `degree`, or set `interactions=False`."
        )

    w_hat = np.maximum(w.value, 0.0)
    w_hat = w_hat / w_hat.sum()

    if clip_ratio is not None and np.isfinite(clip_ratio) and clip_ratio > 0:
        cap = clip_ratio / nE
        w_hat = np.minimum(w_hat, cap)
        w_hat = w_hat / w_hat.sum()

    return {"w": w_hat}


def get_weights(X_ec,
                X_tgt,
                ot_kind: str = "semi_relaxed",
                eps: float = 0.1,
                reg_m: float = 1.0,
                metric: str = "mahalanobis",
                augment: bool = True,
                scale_cost: bool = True,
                clip_ratio: float = 10.0,
                eb_degree: int = 1,
                eb_interactions: bool = False,
                eb_delta: float = 1e-3,
                eb_ridge: float = 1e-6):
    """
    Unified entry point to obtain weights for external controls.

    ot_kind:
        - "uot"          : symmetric unbalanced OT (requires POT)
        - "semi_relaxed" : one-sided entropic OT (closed form)
        - "eb"           : entropy balancing via cvxpy
    """
    if ot_kind == "uot":
        return weights_uot_symmetric(
            X_ec,
            X_tgt,
            eps=eps,
            reg_m=reg_m,
            metric=metric,
            augment=augment,
            clip_ratio=clip_ratio,
            scale_cost=scale_cost,
        )
    elif ot_kind == "semi_relaxed":
        return weights_semi_relaxed_closed_form(
            X_ec,
            X_tgt,
            eps=eps,
            metric=metric,
            augment=augment,
            clip_ratio=clip_ratio,
            scale_cost=scale_cost,
        )
    elif ot_kind == "eb":
        return weights_entropy_balancing(
            X_ec,
            X_tgt,
            degree=eb_degree,
            interactions=eb_interactions,
            delta=eb_delta,
            ridge=eb_ridge,
            clip_ratio=clip_ratio,
        )
    else:
        raise ValueError("Unknown `ot_kind`. Use 'uot', 'semi_relaxed', or 'eb'.")


def ot_weights_semi_relaxed(X_ec,
                            X_tgt,
                            eps: float = 0.1,
                            metric: str = "mahalanobis",
                            augment: bool = True,
                            clip_ratio: float = 10.0,
                            scale_cost: bool = True):
    """
    Backward-compatible wrapper that returns only the EC weights for
    semi-relaxed entropic OT.
    """
    res = weights_semi_relaxed_closed_form(
        X_ec,
        X_tgt,
        eps=eps,
        metric=metric,
        augment=augment,
        clip_ratio=clip_ratio,
        scale_cost=scale_cost,
    )
    return res["w"]


# =========================================================
# Diagnostics: MD, SMD, ESS
# =========================================================

def md_smd_ess(rct_df,
               ec_df,
               w,
               vars=("X1", "X2", "X3")):
    """
    Compute mean differences (raw), standardized mean differences (SMD),
    and effective sample size (ESS).

    Parameters
    ----------
    rct_df : DataFrame
        RCT dataset with covariates vars.
    ec_df : DataFrame
        External control dataset with covariates vars.
    w : array-like
        Weights for EC (will be normalized to sum to 1).
    vars : sequence of str
        Column names to use as covariates.

    Returns
    -------
    dict with keys:
        "per_var"   : DataFrame with per-variable MD and SMD.
        "aggregate" : DataFrame with aggregate MD norms.
        "balance"   : dict with "max_SMD", "SMD" vector, and "ESS".
    """
    X_tgt = rct_df.loc[:, list(vars)].to_numpy(float)
    X_ec = ec_df.loc[:, list(vars)].to_numpy(float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()

    mu_t = X_tgt.mean(axis=0)
    mu_e = X_ec.mean(axis=0)
    mu_w = w @ X_ec

    md_pre = mu_e - mu_t
    md_post = mu_w - mu_t

    sd_t = X_tgt.std(axis=0, ddof=1)
    sd_t[sd_t == 0.0] = 1.0
    smd = np.abs((mu_w - mu_t) / sd_t)

    ess = 1.0 / np.sum(w ** 2)

    per_var = pd.DataFrame(
        {
            "var": list(vars),
            "md_pre": md_pre,
            "md_post": md_post,
            "smd_post": smd,
        }
    )

    agg = pd.DataFrame(
        {
            "which": ["Unweighted EC", "Weighted EC"],
            "max_abs_MD": [np.max(np.abs(md_pre)), np.max(np.abs(md_post))],
            "mean_abs_MD": [np.mean(np.abs(md_pre)), np.mean(np.abs(md_post))],
            "l2_norm": [np.linalg.norm(md_pre), np.linalg.norm(md_post)],
        }
    )

    return {
        "per_var": per_var,
        "aggregate": agg,
        "balance": {"max_SMD": float(np.max(smd)), "SMD": smd, "ESS": float(ess)},
    }


def ot_balance_diagnostics(rct,
                           ec,
                           w,
                           covariate_cols=("X1", "X2", "X3")):
    """
    Convenience wrapper that exposes a simple balance summary for OT weights.
    """
    res = md_smd_ess(rct, ec, w, vars=covariate_cols)
    bal = res["balance"]
    agg = res["aggregate"]

    # Use the weighted row ("Weighted EC") for MD summaries
    idx_w = 1
    return {
        "max_SMD": bal["max_SMD"],
        "SMD": bal["SMD"],
        "ESS": bal["ESS"],
        "max_abs_MD": float(agg.loc[idx_w, "max_abs_MD"]),
        "mean_abs_MD": float(agg.loc[idx_w, "mean_abs_MD"]),
        "l2_MD": float(agg.loc[idx_w, "l2_norm"]),
    }


# =========================================================
# Data generation (original paper setup)
# =========================================================

def simulate_once_paper(n1: int = 500,
                        n0: int = 250,
                        nE: int = 2000,
                        eta=np.array([2.0, 2.0, 2.0]),
                        beta0=np.array([1.0, 1.0, 1.0]),
                        beta1=np.array([2.0, 2.0, 2.0]),
                        target_y0: float = 0.3,
                        target_y1: float = 0.4,
                        pool_factor: int = 50,
                        truth: str = "sample",
                        seed: int = 1):
    """
    Original paper DGP (Section 6 of the OT paper).

      - Large pool X ~ Uniform[-2, 2]^p.
      - Calibrate eta0 so that E[pi(X)] = nR / (nR + nE),
          with pi(x) = inv_logit(eta0 + x^T eta).
      - Sample RCT with probability proportional to pi(X),
        EC with probability proportional to 1 - pi(X).
      - Calibrate outcome intercepts a0, a1 on the reference
        distribution (full RCT or realized RCT).
      - Generate binary outcomes and return rct, ec, tau_true.
    """
    rng = np.random.default_rng(seed)
    p = len(beta0)
    nR = n1 + n0
    Npool = pool_factor * (nR + nE)

    Xpool = rng.uniform(-2.0, 2.0, size=(Npool, p))
    target = nR / (nR + nE)

    def f_eta0(e0):
        return (1.0 / (1.0 + np.exp(e0 + Xpool @ eta))).mean() - target

    eta0 = calibrate_scalar(f_eta0, low=-10, high=10)
    pS = 1.0 / (1.0 + np.exp(eta0 + Xpool @ eta))


    idx_all = np.arange(Npool)
    idx_rct = rng.choice(idx_all, size=nR, replace=False, p=pS / pS.sum())
    remain = np.setdiff1d(idx_all, idx_rct, assume_unique=False)
    p_rem = 1.0 - pS[remain]
    p_rem = p_rem / p_rem.sum()
    idx_ec = rng.choice(remain, size=nE, replace=False, p=p_rem)

    X_rct = Xpool[idx_rct, :]
    X_ec = Xpool[idx_ec, :]

    if truth == "fullRCT":
        idx_rct2 = rng.choice(
            idx_all,
            size=min(Npool, pool_factor * nR),
            replace=False,
            p=pS / pS.sum(),
        )
        X_ref = Xpool[idx_rct2, :]
    else:
        X_ref = X_rct

    a0 = calibrate_scalar(
        lambda a: inv_logit(a + X_ref @ beta0).mean() - target_y0
    )
    a1 = calibrate_scalar(
        lambda a: inv_logit(a + X_ref @ beta1).mean() - target_y1
    )

    A = np.zeros(nR, dtype=int)
    A[rng.choice(np.arange(nR), size=n1, replace=False)] = 1

    mu0_rct = inv_logit(a0 + X_rct @ beta0)
    mu1_rct = inv_logit(a1 + X_rct @ beta1)
    mu0_ec = inv_logit(a0 + X_ec @ beta0)

    Y_rct_alt = rng.binomial(1, np.where(A == 1, mu1_rct, mu0_rct))
    Y_rct_null = rng.binomial(1, mu0_rct)
    Y_ec = rng.binomial(1, mu0_ec)

    tau_true = float(
        inv_logit(a1 + X_ref @ beta1).mean()
        - inv_logit(a0 + X_ref @ beta0).mean()
    )

    rct = pd.DataFrame(
        {
            "X1": X_rct[:, 0],
            "X2": X_rct[:, 1],
            "X3": X_rct[:, 2],
            "A": A,
            "Y_alt": Y_rct_alt,
            "Y_null": Y_rct_null,
        }
    )
    ec = pd.DataFrame(
        {
            "X1": X_ec[:, 0],
            "X2": X_ec[:, 1],
            "X3": X_ec[:, 2],
            "Y": Y_ec,
        }
    )
    return {"rct": rct, "ec": ec, "tau_true": tau_true}


# Optional alias matching earlier naming in standalone scripts
simulate_once_py = simulate_once_paper


# =========================================================
# Additional DGPs (overlap scenarios, SM wrong / OM correct)
# =========================================================

def _mix_gaussians_2d(n, means, covs, probs, rng):
    means = [np.asarray(m, float) for m in means]
    covs = [np.asarray(c, float) for c in covs]
    probs = np.asarray(probs, float)
    probs = probs / probs.sum()
    comp = rng.choice(len(means), size=n, p=probs)
    X = np.zeros((n, 2))
    for k in range(len(means)):
        idx = comp == k
        nk = idx.sum()
        if nk > 0:
            X[idx, :] = rng.multivariate_normal(means[k], covs[k], size=nk)
    return X

def overlap_metrics(rct, ec, vars=("X1","X2","X3")):
    """
    Unweighted overlap diagnostics between RCT and EC:

    Parameters
    ----------
    rct : pandas.DataFrame
        Must contain columns listed in `vars`.
    ec  : pandas.DataFrame
        Must contain columns listed in `vars`.
    vars : tuple/list of str
        Names of covariate columns, e.g. ("X1","X2","X3").

    Returns
    -------
    dict with keys:
      - max_abs_MD : max |mean_RCT - mean_EC|
      - mean_abs_MD: mean |mean_RCT - mean_EC|
      - l2_MD      : Euclidean norm of mean differences
      - maha_d2    : Mahalanobis squared distance
    """
    import numpy as np

    Xr = rct.loc[:, list(vars)].to_numpy(float)
    Xe = ec.loc[:,  list(vars)].to_numpy(float)

    mu_r = Xr.mean(axis=0)
    mu_e = Xe.mean(axis=0)
    md   = mu_r - mu_e

    max_abs_MD  = float(np.max(np.abs(md)))
    mean_abs_MD = float(np.mean(np.abs(md)))
    l2_MD       = float(np.linalg.norm(md))

    # pooled covariance for Mahalanobis distance
    X_all = np.vstack([Xr, Xe])
    Sigma = np.cov(X_all.T) + 1e-6 * np.eye(X_all.shape[1])
    invS  = np.linalg.inv(Sigma)
    maha_d2 = float(md @ invS @ md)

    return {
        "max_abs_MD":  max_abs_MD,
        "mean_abs_MD": mean_abs_MD,
        "l2_MD":       l2_MD,
        "maha_d2":     maha_d2
    }

def sample_X_overlap(nR, nE, scenario, rng):
    """
    Helper for overlap-based DGP: generates X_rct, X_ec with varying overlap.
    """
    if scenario == "moderate_mixture":
        means_r = [(-1.0, -0.5), (0.8, 0.9)]
        means_e = [(-0.2, -0.3), (1.3, 0.4)]
        covs_r = [0.4 ** 2 * np.eye(2)] * 2
        covs_e = [0.4 ** 2 * np.eye(2)] * 2
        probs_r = [0.5, 0.5]
        probs_e = [0.5, 0.5]
        Xr2 = _mix_gaussians_2d(nR, means_r, covs_r, probs_r, rng)
        Xe2 = _mix_gaussians_2d(nE, means_e, covs_e, probs_e, rng)
    elif scenario == "bad_checkerboard":
        means_r = [(-1.3, -1.3), (1.3, 1.3)]
        means_e = [(-1.3, 1.3), (1.3, -1.3)]
        covs_r = [0.3 ** 2 * np.eye(2)] * 2
        covs_e = [0.3 ** 2 * np.eye(2)] * 2
        probs_r = [0.5, 0.5]
        probs_e = [0.5, 0.5]
        Xr2 = _mix_gaussians_2d(nR, means_r, covs_r, probs_r, rng)
        Xe2 = _mix_gaussians_2d(nE, means_e, covs_e, probs_e, rng)
    elif scenario == "wavy_tilted":
        x1_r = rng.uniform(-2.0, 2.0, size=nR)
        x2_r = np.sin(2.0 * np.pi * x1_r / 2.0) + 0.3 * rng.normal(size=nR)
        Xr2 = np.column_stack([x1_r, x2_r])
        x1_e = rng.uniform(-2.0, 2.0, size=nE)
        x2_e = -np.sin(2.0 * np.pi * x1_e / 2.0) + 0.3 * rng.normal(size=nE) + 0.6
        Xe2 = np.column_stack([x1_e, x2_e])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    x3_r = rng.normal(0.0, 1.0, size=nR)
    x3_e = rng.normal(0.0, 1.0, size=nE)
    X_rct = np.column_stack([Xr2, x3_r])
    X_ec = np.column_stack([Xe2, x3_e])
    return X_rct, X_ec


def simulate_once_overlap(n1: int = 500,
                          n0: int = 250,
                          nE: int = 2000,
                          scenario: str = "moderate_mixture",
                          beta0=np.array([1.0, 1.0, 1.0]),
                          beta1=np.array([2.0, 2.0, 2.0]),
                          target_y0: float = 0.3,
                          target_y1: float = 0.4,
                          seed: int = 1):
    """
    Overlap-based DGP:
      - X_rct and X_ec from different overlapping distributions.
      - a0, a1 calibrated on RCT covariate distribution.
      - Outcome model logistic linear in X, same for RCT and EC.
    """
    rng = np.random.default_rng(seed)
    nR = n1 + n0

    X_rct, X_ec = sample_X_overlap(nR, nE, scenario, rng)

    a0 = calibrate_intercept(X_rct, beta0, target_y0)
    a1 = calibrate_intercept(X_rct, beta1, target_y1)

    A = np.zeros(nR, dtype=int)
    A[rng.choice(nR, size=n1, replace=False)] = 1

    mu0_rct = inv_logit(a0 + X_rct @ beta0)
    mu1_rct = inv_logit(a1 + X_rct @ beta1)
    mu0_ec = inv_logit(a0 + X_ec @ beta0)

    Y_rct_alt = rng.binomial(1, np.where(A == 1, mu1_rct, mu0_rct))
    Y_rct_null = rng.binomial(1, mu0_rct)
    Y_ec = rng.binomial(1, mu0_ec)

    tau_true = float(mu1_rct.mean() - mu0_rct.mean())

    rct = pd.DataFrame(
        {
            "X1": X_rct[:, 0],
            "X2": X_rct[:, 1],
            "X3": X_rct[:, 2],
            "A": A,
            "Y_alt": Y_rct_alt,
            "Y_null": Y_rct_null,
        }
    )
    ec = pd.DataFrame(
        {
            "X1": X_ec[:, 0],
            "X2": X_ec[:, 1],
            "X3": X_ec[:, 2],
            "Y": Y_ec,
        }
    )
    return {"rct": rct, "ec": ec, "tau_true": tau_true}


def simulate_once_smWrong_omCorrect(n1: int = 500,
                                    n0: int = 250,
                                    nE: int = 2000,
                                    eta=np.array([2.0, 2.0, 2.0]),
                                    beta0=np.array([1.0, 1.0, 1.0]),
                                    beta1=np.array([2.0, 2.0, 2.0]),
                                    target_y0: float = 0.3,
                                    target_y1: float = 0.4,
                                    pool_factor: int = 50,
                                    seed: int = 1):
    """
    DGP where:
      - Sampling model is truly nonlinear in transformed covariates X*,
        so a logistic model in X is misspecified.
      - Outcome model is logistic linear in X (correct),
        same for RCT and EC (no hidden bias).
    """
    rng = np.random.default_rng(seed)
    p = len(beta0)
    nR = n1 + n0
    Npool = pool_factor * (nR + nE)

    Xpool = rng.uniform(-2.0, 2.0, size=(Npool, p))
    Xstar = np.exp(Xpool) + 10.0 * np.sin(Xpool) * np.cos(Xpool)

    target = nR / (nR + nE)

    def f_eta0(e0):
        return inv_logit(e0 + Xstar @ eta).mean() - target

    eta0 = calibrate_scalar(f_eta0)

    pS = inv_logit(eta0 + Xstar @ eta)

    idx_all = np.arange(Npool)
    idx_rct = rng.choice(idx_all, size=nR, replace=False, p=pS / pS.sum())
    remain = np.setdiff1d(idx_all, idx_rct, assume_unique=False)
    p_rem = 1.0 - pS[remain]
    p_rem = p_rem / p_rem.sum()
    idx_ec = rng.choice(remain, size=nE, replace=False, p=p_rem)

    X_rct = Xpool[idx_rct, :]
    X_ec = Xpool[idx_ec, :]

    X_ref = X_rct

    a0 = calibrate_scalar(
        lambda a: inv_logit(a + X_ref @ beta0).mean() - target_y0
    )
    a1 = calibrate_scalar(
        lambda a: inv_logit(a + X_ref @ beta1).mean() - target_y1
    )

    A = np.zeros(nR, dtype=int)
    A[rng.choice(np.arange(nR), size=n1, replace=False)] = 1

    mu0_rct = inv_logit(a0 + X_rct @ beta0)
    mu1_rct = inv_logit(a1 + X_rct @ beta1)
    mu0_ec = inv_logit(a0 + X_ec @ beta0)

    Y_rct_alt = rng.binomial(1, np.where(A == 1, mu1_rct, mu0_rct))
    Y_rct_null = rng.binomial(1, mu0_rct)
    Y_ec = rng.binomial(1, mu0_ec)

    mu0_ref = inv_logit(a0 + X_ref @ beta0)
    mu1_ref = inv_logit(a1 + X_ref @ beta1)
    tau_true = float(mu1_ref.mean() - mu0_ref.mean())

    rct = pd.DataFrame(
        {
            "X1": X_rct[:, 0],
            "X2": X_rct[:, 1],
            "X3": X_rct[:, 2],
            "A": A,
            "Y_alt": Y_rct_alt,
            "Y_null": Y_rct_null,
        }
    )
    ec = pd.DataFrame(
        {
            "X1": X_ec[:, 0],
            "X2": X_ec[:, 1],
            "X3": X_ec[:, 2],
            "Y": Y_ec,
        }
    )
    return {"rct": rct, "ec": ec, "tau_true": tau_true}


# =========================================================
# MSE vs lambda experiment (original DGP)
# =========================================================

def tau_hat_mixed(rct_df,
                  muE: float,
                  lam: float = 0.0,
                  outcome_col: str = "Y_alt") -> float:
    """
    Mixed estimator:

        tau = mean(Y | A=1) - [(1 - lam) * mean(Y | A=0) + lam * muE]
    """
    y = rct_df[outcome_col].to_numpy(float)
    a = rct_df["A"].to_numpy(int)
    muT = y[a == 1].mean()
    muC = y[a == 0].mean()
    return float(muT - ((1.0 - lam) * muC + lam * muE))


def mse_vs_lambda(R: int = 100,
                  lambda_grid=None,
                  n1: int = 500,
                  n0: int = 250,
                  nE: int = 2000,
                  ot_kind: str = "semi_relaxed",
                  eps: float = 0.1,
                  reg_m: float = 1.0,
                  metric: str = "mahalanobis",
                  augment: bool = True,
                  scale_cost: bool = True,
                  clip_ratio: float = 10.0,
                  eb_degree: int = 1,
                  eb_interactions: bool = False,
                  eb_delta: float = 1e-3,
                  eb_ridge: float = 1e-6,
                  truth: str = "sample",
                  seed: int = 123,
                  vars=("X1", "X2", "X3")):
    """
    Monte Carlo experiment to produce MSE(lambda) under the original DGP.

    For each replicate:
      - simulate one dataset (RCT + EC) under simulate_once_paper,
      - compute OT/EB weights once against the full RCT covariate distribution,
      - form the mixed estimator for each lambda in lambda_grid,
      - record squared error relative to tau_true.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 0.2, 11)

    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(R):
        dat = simulate_once_paper(
            n1=n1,
            n0=n0,
            nE=nE,
            truth=truth,
            seed=int(rng.integers(1, 10**9)),
        )
        rct = dat["rct"]
        ec = dat["ec"]
        tau_true = dat["tau_true"]

        X_tgt = rct.loc[:, list(vars)].to_numpy(float)
        X_ec = ec.loc[:, list(vars)].to_numpy(float)

        wobj = get_weights(
            X_ec,
            X_tgt,
            ot_kind=ot_kind,
            eps=eps,
            reg_m=reg_m,
            metric=metric,
            augment=augment,
            scale_cost=scale_cost,
            clip_ratio=clip_ratio,
            eb_degree=eb_degree,
            eb_interactions=eb_interactions,
            eb_delta=eb_delta,
            eb_ridge=eb_ridge,
        )
        w = wobj["w"]

        muE = float(np.sum(w * ec["Y"].to_numpy(float)))

        for lam in lambda_grid:
            t_hat = tau_hat_mixed(rct, muE, lam=lam, outcome_col="Y_alt")
            rows.append({"lambda": lam, "sq_err": (t_hat - tau_true) ** 2})

    df = pd.DataFrame(rows)
    out = df.groupby("lambda", as_index=False).agg(
        MSE=("sq_err", "mean"),
        MSE_sd=("sq_err", "std"),
    )
    return out


# =========================================================
# OT-based RD with plug-in lambda*
# =========================================================

def ot_lambda_star_rd(rct,
                      ec,
                      rct_outcome_col: str = "Y_alt",
                      ec_outcome_col: str = "Y",
                      treat_col: str = "A",
                      covariate_cols=("X1", "X2", "X3"),
                      eps: float = 0.1,
                      clip_ratio: float = 10.0,
                      lambda_grid=None,
                      lambda_penalty: float = 0.3):
    """
    OT-based RD estimator with plug-in lambda*, where lambda* is selected
    by grid search over a proxy MSE that accounts for both variance and
    the observed disagreement between RCT and OT-weighted EC controls.

    The proxy control MSE minimized over lambda is:

        MSE_ctrl(lam) ~= (1-lam)^2 Var_C + lam^2 Var_E + lam^2 (muE - muC)^2

    Parameters
    ----------
    rct : DataFrame
        Must contain covariate_cols, treat_col, rct_outcome_col.
    ec : DataFrame
        Must contain covariate_cols, ec_outcome_col.
    eps : float
        Entropic parameter for semi-relaxed OT.
    clip_ratio : float
        Weight clipping ratio passed to OT.
    lambda_grid : array-like or None
        Candidate values for lambda in [0,1]. If None, uses np.linspace(0,1,51).
    lambda_penalty : float
        Optional multiplier on the squared bias term (muE - muC)^2.

    Returns
    -------
    tau_hat : float
        RD point estimate using the plug-in lambda*.
    info : dict
        Contains lambda_star, varC, varE, delta, max_SMD, ESS, and proxy_MSE.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 51)

    Xr = rct.loc[:, covariate_cols].to_numpy(float)
    Xe = ec.loc[:, covariate_cols].to_numpy(float)
    Y_ec = ec[ec_outcome_col].to_numpy(float)

    w = ot_weights_semi_relaxed(Xe, Xr, eps=eps, clip_ratio=clip_ratio)
    muE = float(np.sum(w * Y_ec))

    y_r = rct[rct_outcome_col].to_numpy(float)
    a = rct[treat_col].to_numpy(int)
    muT = float(y_r[a == 1].mean())
    muC = float(y_r[a == 0].mean())
    n0 = int((a == 0).sum())

    pC_hat = muC
    varC_hat = pC_hat * (1.0 - pC_hat) / max(n0, 1)

    pE_hat = muE
    varE_hat = pE_hat * (1.0 - pE_hat) * float(np.sum(w ** 2))

    delta = muE - muC

    lam_arr = np.asarray(lambda_grid, dtype=float)
    lam_arr = np.clip(lam_arr, 0.0, 1.0)

    var_ctrl = (1.0 - lam_arr) ** 2 * varC_hat + lam_arr ** 2 * varE_hat
    bias2 = (lam_arr * delta) ** 2 * float(lambda_penalty)
    proxy_mse = var_ctrl + bias2

    idx_min = int(np.argmin(proxy_mse))
    lambda_star = float(lam_arr[idx_min])

    tau_hat = muT - ((1.0 - lambda_star) * muC + lambda_star * muE)

    diag = ot_balance_diagnostics(rct, ec, w, covariate_cols=covariate_cols)

    info = {
        "lambda_star": lambda_star,
        "varC": float(varC_hat),
        "varE": float(varE_hat),
        "delta": float(delta),
        "proxy_MSE": float(proxy_mse[idx_min]),
        "max_SMD": diag["max_SMD"],
        "ESS": diag["ESS"],
    }
    return tau_hat, info


# --- New helper: lambda selection via outcome-model calibrated grid search ---

def select_lambda_grid_om(
    muC,
    muE,
    rct,
    x_cols=("X1", "X2", "X3"),
    outcome_col="Y_alt",
    treat_col="A",
    lambda_grid=None,
    degree2=True,
    l2=1e-4,
):
    """
    Select lambda by matching the mixed control mean to an outcome-model-based
    estimate of the target control mean.

    The outcome model is a logistic regression for Y | A=0, X fitted on RCT
    controls only. lambda_grid is searched to minimize
        ( (1-lambda)*muC + lambda*muE - mu0_model )^2.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 0.5, 26)  # 0.00, 0.02, ..., 0.50

    rct = rct.copy()
    a = rct[treat_col].to_numpy(int)
    idx_ctrl = a == 0

    X_ctrl = rct.loc[idx_ctrl, x_cols].to_numpy(float)
    Y_ctrl = rct.loc[idx_ctrl, outcome_col].to_numpy(float)

    if X_ctrl.shape[0] == 0:
        # Fallback: no controls in RCT, revert to lambda = 0.0
        return 0.0, float(muC), np.zeros_like(lambda_grid, dtype=float)

    XF_ctrl = build_poly2(X_ctrl, interactions=True) if degree2 else X_ctrl
    coef0, intercept0 = fit_logistic_l2(XF_ctrl, Y_ctrl, l2=l2)
    p0_ctrl = predict_proba(XF_ctrl, coef0, intercept0)
    mu0_model = float(p0_ctrl.mean())

    losses = []
    for lam in lambda_grid:
        mu_mix = (1.0 - lam) * muC + lam * muE
        losses.append((mu_mix - mu0_model) ** 2)

    losses = np.asarray(losses, dtype=float)
    best_idx = int(np.argmin(losses))
    lambda_star = float(lambda_grid[best_idx])

    return lambda_star, mu0_model, losses


def ot_lambda_grid_rd(
    rct,
    ec,
    rct_outcome_col: str = "Y_alt",
    ec_outcome_col: str = "Y",
    treat_col: str = "A",
    covariate_cols=("X1", "X2", "X3"),
    eps: float = 0.1,
    clip_ratio: float = 10.0,
    lambda_grid=None,
    om_degree2: bool = True,
    om_l2: float = 1e-4,
):
    """
    OT-based RD estimator with lambda chosen by outcome-model calibrated
    grid search.

    Steps:
      1) Compute semi-relaxed OT weights w on EC vs full RCT covariates.
      2) Form EC control mean muE = sum w_i Y_i^EC.
      3) Compute RCT treated mean muT and RCT control mean muC.
      4) Fit a logistic outcome model for Y | A=0, X on RCT controls only,
         get mu0_model = mean predicted Y(0) in the RCT population.
      5) Choose lambda in lambda_grid minimizing
            ((1-lambda)*muC + lambda*muE - mu0_model)^2.
      6) Plug-in lambda into the hybrid estimator:
            tau = muT - [(1-lambda)*muC + lambda*muE].
    """
    Xr = rct.loc[:, covariate_cols].to_numpy(float)
    Xe = ec.loc[:, covariate_cols].to_numpy(float)
    Y_ec = ec[ec_outcome_col].to_numpy(float)

    # OT weights and EC mean
    w = ot_weights_semi_relaxed(
        Xe,
        Xr,
        eps=eps,
        metric="mahalanobis",
        augment=True,
        clip_ratio=clip_ratio,
        scale_cost=True,
    )
    muE = float(np.sum(w * Y_ec))

    y_r = rct[rct_outcome_col].to_numpy(float)
    a = rct[treat_col].to_numpy(int)
    muT = float(y_r[a == 1].mean())
    muC = float(y_r[a == 0].mean())

    lambda_star, mu0_model, losses = select_lambda_grid_om(
        muC=muC,
        muE=muE,
        rct=rct,
        x_cols=covariate_cols,
        outcome_col=rct_outcome_col,
        treat_col=treat_col,
        lambda_grid=lambda_grid,
        degree2=om_degree2,
        l2=om_l2,
    )

    tau_hat = muT - ((1.0 - lambda_star) * muC + lambda_star * muE)

    # Balance diagnostics
    diag = ot_balance_diagnostics(
        rct,
        ec,
        w,
        covariate_cols=covariate_cols,
    )

    info = {
        "lambda_star": float(lambda_star),
        "mu0_model": float(mu0_model),
        "muC": float(muC),
        "muE": float(muE),
        "lambda_grid": np.array(lambda_grid if lambda_grid is not None else []),
        "loss_curve": losses,
        "max_SMD": diag["max_SMD"],
        "ESS": diag["ESS"],
    }
    return tau_hat, info




# =========================================================
# Propensity-score/IPW utilities (kept for completeness)
# =========================================================

def fit_logistic_l2(X,
                    y,
                    l2: float = 1e-4,
                    max_iter: int = 200,
                    tol: float = 1e-8):
    """
    Logistic regression with L2 penalty on coefficients (not intercept),
    implemented via IRLS / Newton-Raphson without external dependencies.

    Parameters
    ----------
    X : array-like, shape (n, p)
    y : array-like, shape (n,)
    l2 : float
        L2 penalty strength on coefficients.
    max_iter : int
    tol : float

    Returns
    -------
    coef : ndarray, shape (p,)
    intercept : float
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape

    Z = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    R = np.eye(d + 1)
    R[0, 0] = 0.0  # do not penalize intercept

    for _ in range(max_iter):
        eta = Z @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w
        WZ = Z * w[:, None]
        A = Z.T @ WZ + l2 * R
        b = Z.T @ (w * z)
        beta_new = np.linalg.solve(A, b)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    coef = beta[1:]
    intercept = beta[0]
    return coef, intercept


def predict_proba(X, coef, intercept):
    """
    Predict probabilities for a logistic regression model.
    """
    X = np.asarray(X, float)
    return inv_logit(intercept + X @ coef)


def ipw_rct_estimate(rct,
                     x_cols=("X1", "X2", "X3"),
                     outcome_col: str = "Y_alt",
                     treat_col: str = "A",
                     degree2: bool = True,
                     l2: float = 1e-4,
                     ps_clip: float = 1e-3):
    """
    IPW ATE estimator using estimated treatment propensity scores within the RCT:

        tau_hat = mean(A * Y / e_hat - (1 - A) * Y / (1 - e_hat))
    """
    X = rct.loc[:, x_cols].to_numpy(float)
    A = rct[treat_col].to_numpy(int)
    Y = rct[outcome_col].to_numpy(float)

    XF = build_poly2(X, interactions=True) if degree2 else X
    coef, intercept = fit_logistic_l2(XF, A, l2=l2)
    e_hat = predict_proba(XF, coef, intercept)
    e_hat = np.clip(e_hat, ps_clip, 1.0 - ps_clip)

    tau_hat = float(np.mean(A * Y / e_hat - (1 - A) * Y / (1.0 - e_hat)))
    return tau_hat


def borrow_ipw_rd(rct,
                  ec,
                  x_cols=("X1", "X2", "X3"),
                  y_rct_col: str = "Y_alt",
                  y_ec_col: str = "Y",
                  treat_col: str = "A",
                  l2_sm: float = 1e-4,
                  l2_ps: float = 1e-4,
                  ps_clip: float = 1e-3,
                  pi_clip: float = 1e-3):
    """
    Simplified Borrow-IPW-style estimator for RD, using:
      - sampling model pi(X) = P(S=1 | X),
      - treatment PS e(X) within the RCT sample,
      - r(X) ≡ 1 (no hidden bias) so that both RCT and EC controls contribute.
    """
    rct2 = rct.copy()
    ec2 = ec.copy()
    rct2["S"] = 1
    ec2["S"] = 0
    ec2[treat_col] = 0  # EC are controls

    dat = pd.concat([rct2, ec2], ignore_index=True)

    X = dat.loc[:, x_cols].to_numpy(float)
    S = dat["S"].to_numpy(int)
    A = dat[treat_col].to_numpy(int)

    Y = np.where(
        dat["S"].to_numpy(int) == 1,
        dat[y_rct_col].to_numpy(float),
        dat[y_ec_col].to_numpy(float),
    )

    XF_sm = build_poly2(X, interactions=True)
    coef_pi, int_pi = fit_logistic_l2(XF_sm, S, l2=l2_sm)
    pi_hat = inv_logit(int_pi + XF_sm @ coef_pi)
    pi_hat = np.clip(pi_hat, pi_clip, 1.0 - pi_clip)

    idx_rct = S == 1
    X_e = X[idx_rct, :]
    A_e = A[idx_rct]
    XF_e = build_poly2(X_e, interactions=True)
    coef_e, int_e = fit_logistic_l2(XF_e, A_e, l2=l2_ps)

    XF_all = build_poly2(X, interactions=True)
    e_hat = inv_logit(int_e + XF_all @ coef_e)
    e_hat = np.clip(e_hat, ps_clip, 1.0 - ps_clip)

    numer = S * (1 - A) + (1 - S)
    denom = pi_hat * (1.0 - e_hat) + (1.0 - pi_hat)
    k = numer / (denom + 1e-12)

    nR = int((S == 1).sum())
    theta0_hat = float(np.sum(pi_hat * k * Y) / max(nR, 1))

    Y_rct = rct[y_rct_col].to_numpy(float)
    A_rct = rct[treat_col].to_numpy(int)
    mu1_hat = float(Y_rct[A_rct == 1].mean())

    tau_hat = mu1_hat - theta0_hat
    return tau_hat
