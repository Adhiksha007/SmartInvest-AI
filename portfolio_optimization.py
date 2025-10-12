import warnings
# Suppresses the repetitive Scikit-learn feature name warning
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')
# Suppresses the Pandas 'M' deprecation warning (or rely on the code fix below)
warnings.filterwarnings('ignore', category=FutureWarning)


# portfolio_module.py
"""
Portfolio module with:
 - MPT (mean-variance)
 - Black-Litterman
 - AI forecaster using LightGBM + time-series CV + cross-asset features
 - Annualization utilities
 - Plotting / backtest helpers

Dependencies: numpy, pandas, scipy, scikit-learn, lightgbm, yfinance (for notebook), matplotlib

Save this file and import the classes in a notebook or script.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings


# ---------------------------
# Portfolio math helpers
# ---------------------------
def portfolio_return(weights, mu):
    return float(np.dot(weights, mu))

def portfolio_volatility(weights, cov):
    return float(np.sqrt(weights.T @ cov @ weights))

def sharpe_ratio(weights, mu, cov, risk_free):
    r = portfolio_return(weights, mu)
    v = portfolio_volatility(weights, cov)
    return (r - risk_free) / v if v > 0 else -1e6

# ---------------------------
# MPT Optimizer
# ---------------------------

class MPTOptimizer:
    def __init__(self, expected_returns: np.ndarray, cov: np.ndarray,
                 risk_free: float = 0.02,
                 max_weight: float = 0.4,
                 allow_short: bool = False,
                 transaction_cost_bps: float = 5.0,
                 slippage_bps: float = 2.0,
                 weight_sum: float = 1.0):
        """
        Modern Portfolio Theory Optimizer.

        Parameters
        ----------
        expected_returns : array
            Vector of expected returns.
        cov : array
            Covariance matrix of returns.
        risk_free : float
            Risk-free rate (annualized).
        max_weight : float
            Maximum allocation per asset.
        allow_short : bool
            Allow negative weights (shorting).
        transaction_cost_bps : float
            Transaction cost per trade (basis points).
        slippage_bps : float
            Slippage per trade (basis points).
        weight_sum : float
            Sum of portfolio weights (default 1.0).
        """
        self.mu = np.asarray(expected_returns).reshape(-1)
        self.cov = np.asarray(cov)
        self.n = len(self.mu)
        self.risk_free = risk_free
        self.max_weight = max_weight
        self.allow_short = allow_short
        self.transaction_cost = transaction_cost_bps / 10000
        self.slippage = slippage_bps / 10000
        self.weight_sum = weight_sum

    def _constraints(self):
        return ({'type': 'eq', 'fun': lambda w: np.sum(w) - self.weight_sum},)

    def _init_guess(self):
        return np.ones(self.n) / self.n

    def _bounds(self):
        if self.allow_short:
            return tuple((-1.0, self.max_weight) for _ in range(self.n))
        else:
            return tuple((0.0, self.max_weight) for _ in range(self.n))

    def _penalty(self, weights):
        """Transaction + slippage costs (L1 penalty)."""
        return np.sum(np.abs(weights)) * (self.transaction_cost + self.slippage)

    def min_variance(self):
        def obj(w):
            return float(w.T @ self.cov @ w) + self._penalty(w)
        res = minimize(obj, self._init_guess(), method='SLSQP',
                       bounds=self._bounds(), constraints=self._constraints())
        if not res.success:
            raise RuntimeError('Min variance optimization failed: ' + res.message)
        return res.x

    def max_sharpe(self):
        def neg_sharpe(w):
            return -(sharpe_ratio(w, self.mu, self.cov, self.risk_free) - self._penalty(w))
        res = minimize(neg_sharpe, self._init_guess(), method='SLSQP',
                       bounds=self._bounds(), constraints=self._constraints())
        if not res.success:
            raise RuntimeError('Max Sharpe optimization failed: ' + res.message)
        return res.x

    def efficient_frontier(self, returns_grid=None):
        if returns_grid is None:
            min_ret = float(np.min(self.mu))
            max_ret = float(np.max(self.mu))
            returns_grid = np.linspace(min_ret, max_ret, 50)
        results = []
        for targ in returns_grid:
            cons = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - self.weight_sum},
                {'type': 'eq', 'fun': lambda w, targ=targ: float(np.dot(w, self.mu)) - targ}
            )
            def obj(w):
                return float(w.T @ self.cov @ w) + self._penalty(w)
            try:
                res = minimize(obj, self._init_guess(), method='SLSQP',
                               bounds=self._bounds(), constraints=cons)
                if res.success:
                    w = res.x
                    results.append({
                        'target_return': targ,
                        'weights': w,
                        'volatility': portfolio_volatility(w, self.cov),
                        'sharpe': sharpe_ratio(w, self.mu, self.cov, self.risk_free)
                    })
            except Exception:
                continue
        return results

    def report(self, weights, as_dict=False):
        """Generate a summary report for portfolio weights."""
        ret = portfolio_return(weights, self.mu)
        vol = portfolio_volatility(weights, self.cov)
        sr = sharpe_ratio(weights, self.mu, self.cov, self.risk_free)
        cost = self._penalty(weights)
        max_w = np.max(weights)

        report_data = {
            "Expected Return": round(ret, 6),
            "Volatility": round(vol, 6),
            "Sharpe Ratio": round(sr, 4),
            "Transaction+Slippage Cost": round(cost, 6),
            "Max Weight Used": round(max_w, 3)
        }

        if as_dict:
            return report_data
        else:
            print("📊 Portfolio Report")
            for k, v in report_data.items():
                print(f"{k:25}: {v}")
            return report_data

# ---------------------------
# Black-Litterman
# ---------------------------

class BlackLitterman:
    def __init__(self, cov: np.ndarray, tau: float = 0.05, delta: float = 2.5):
        self.Sigma = np.asarray(cov)
        self.tau = tau
        self.delta = delta
        self.n = self.Sigma.shape[0]

    def equilibrium_returns(self, market_weights: np.ndarray):
        w = np.asarray(market_weights).reshape(-1)
        return self.delta * (self.Sigma @ w)

    def posterior(self, pi: np.ndarray, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray = None):
        pi = np.asarray(pi).reshape(-1)
        P = np.asarray(P)
        Q = np.asarray(Q).reshape(-1)
        k = P.shape[0]
        if Omega is None:
            Omega = np.diag(np.diag(P @ (self.tau * self.Sigma) @ P.T))
        tauSigma = self.tau * self.Sigma
        inv_tauSigma = np.linalg.inv(tauSigma)
        middle = np.linalg.inv(P @ tauSigma @ P.T + Omega)
        mu_bl = np.linalg.inv(inv_tauSigma + P.T @ middle @ P) @ (inv_tauSigma @ pi + P.T @ middle @ Q)
        Sigma_bl = self.Sigma + np.linalg.inv(inv_tauSigma + P.T @ middle @ P)
        return mu_bl.reshape(-1), Sigma_bl


# ---------------------------
# Backtest helpers & plotting
# ---------------------------
def backtest_weights(price_df: pd.DataFrame, weights: np.ndarray):
    """Simple buy-and-hold backtest: returns cumulative returns series for portfolio weights.
    weights: 1D array aligning to columns order in price_df
    """
    prices = price_df.sort_index()
    returns = prices.pct_change().fillna(0)
    port_ret = returns.dot(weights)
    cum = (1 + port_ret).cumprod()
    return cum, port_ret
