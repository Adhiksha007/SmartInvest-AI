import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import warnings
warnings.filterwarnings("ignore")
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class AIForecasterLGB:
    """
    Hybrid LGB Forecaster:
    - Supports macroeconomic features (optional).
    - Builds richer features: lagged returns, rolling vol, cross-sectional momentum.
    - Optional GridSearchCV for hyperparameter tuning.
    - Forecasts both returns and volatility.
    """

    def __init__(self, lags=5, n_jobs=1, lgb_params=None):
        if lgb is None:
            raise ImportError("lightgbm not available. Please install lightgbm to use AIForecasterLGB.")
        self.lags = lags
        self.n_jobs = n_jobs
        self.lgb_params = lgb_params or {
            "objective": "regression",
            "n_estimators": 200,
            "random_state": 42,
            "verbose": -1,
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 31
        }
        self.trained = False

    def _make_features_all(self, returns_df: pd.DataFrame, macro_df: pd.DataFrame = None):
        """Create feature set for each asset."""
        r = returns_df.copy().sort_index()
        features = {}

        # Cross-asset features
        cross_mom = r.rolling(window=self.lags).mean().shift(1)
        cross_rank = r.rolling(window=self.lags).mean().rank(axis=1, pct=True).shift(1)

        for col in r.columns:
            X = pd.DataFrame(index=r.index)

            # Own lags
            for l in range(1, self.lags + 1):
                X[f"lag_{l}"] = r[col].shift(l)

            # Own rolling stats
            X[f"own_ma_{self.lags}"] = r[col].rolling(window=self.lags).mean().shift(1)
            X[f"own_vol_{self.lags}"] = r[col].rolling(window=self.lags).std().shift(1)

            # Cross-sectional features
            X["cross_mom_mean"] = cross_mom.mean(axis=1)
            X["cross_mom_median"] = cross_mom.median(axis=1)
            X["cross_rank_mean"] = cross_rank.mean(axis=1)

            # Macro features if available
            if macro_df is not None:
                mf = macro_df.reindex(X.index).fillna(method="ffill").fillna(method="bfill")
                for c in mf.columns:
                    X[f"macro_{c}"] = mf[c]

            # Drop NaNs
            X = X.dropna()
            features[col] = X

        return features

    def fit(self, price_df: pd.DataFrame, horizon=1, target_vol_window=20,
            macro_df: pd.DataFrame = None, tune=False, param_grid=None):
        """Fit LightGBM models for each asset."""
        price_df = price_df.sort_index()
        returns = price_df.pct_change().dropna()
        self.returns = returns
        self.tickers = returns.columns.tolist()

        feat_map = self._make_features_all(returns, macro_df=macro_df)
        self.models = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for col in self.tickers:
            X = feat_map[col]
            y_ret = returns[col].shift(-horizon).loc[X.index]
            y_vol = returns[col].rolling(window=target_vol_window).std().shift(-horizon).loc[X.index]

            valid = y_ret.notna() & y_vol.notna()
            Xv = X.loc[valid]; yretv = y_ret.loc[valid]; yvolv = y_vol.loc[valid]

            # If too few samples → fallback
            if len(Xv) < 80:
                self.models[col] = None
                continue

            base_est = lgb.LGBMRegressor(**self.lgb_params)
            pipe_ret = Pipeline([("scaler", StandardScaler()), ("lgb", base_est)])
            pipe_vol = Pipeline([("scaler", StandardScaler()), ("lgb", base_est)])

            if tune and param_grid is not None:
                # CV grid search for best hyperparameters
                gscv_ret = GridSearchCV(pipe_ret, param_grid, cv=tscv,
                                        scoring="neg_mean_squared_error", n_jobs=self.n_jobs)
                gscv_vol = GridSearchCV(pipe_vol, param_grid, cv=tscv,
                                        scoring="neg_mean_squared_error", n_jobs=self.n_jobs)
                gscv_ret.fit(Xv, yretv)
                gscv_vol.fit(Xv, yvolv)
                self.models[col] = {
                    "ret_model": gscv_ret.best_estimator_,
                    "vol_model": gscv_vol.best_estimator_,
                    "tuned": True
                }
            else:
                pipe_ret.fit(Xv, yretv)
                pipe_vol.fit(Xv, yvolv)
                self.models[col] = {"ret_model": pipe_ret, "vol_model": pipe_vol, "tuned": False}

        self.trained = True
        return self.models

    def predict(self, macro_df: pd.DataFrame = None):
        """Predict expected returns and covariance matrix."""
        if not self.trained:
            raise RuntimeError("Call fit() before predict()")

        returns = self.returns
        tickers = self.tickers
        exp_rets, exp_vols = [], []

        feat_map = self._make_features_all(returns, macro_df=macro_df)
        for col in tickers:
            X = feat_map[col]
            if X.empty or self.models.get(col) is None:
                r = returns[col]
                exp_rets.append(float(r.mean()))
                exp_vols.append(float(r.std()))
                continue

            latest = X.iloc[-1:]
            mdl = self.models[col]
            mu = float(mdl["ret_model"].predict(latest)[0])
            vol = float(abs(mdl["vol_model"].predict(latest)[0]))
            if vol <= 1e-10:
                vol = float(returns[col].std())
            exp_rets.append(mu)
            exp_vols.append(vol)

        exp_rets = np.array(exp_rets)
        exp_vols = np.array(exp_vols)

        # Build covariance matrix
        hist_corr = self.returns.corr().values
        cov = np.outer(exp_vols, exp_vols) * hist_corr
        cov = (cov + cov.T) / 2.0 + 1e-10 * np.eye(len(tickers))

        return exp_rets, cov, exp_vols

"""#2. Sentiment Analyzer (FinBERT)"""
class SentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT.
    - Fetches Google News headlines per ticker.
    - Computes sentiment = P(pos) - P(neg).
    - Can return raw per-text scores or aggregated daily scores.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def _score_texts(self, texts: list) -> np.ndarray:
        """Return sentiment scores (P(pos) - P(neg)) for a list of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        probs = softmax(outputs.logits.detach().numpy(), axis=1)
        pos = probs[:, 2]   # positive class
        neg = probs[:, 0]   # negative class
        return pos - neg    # same as pipeline version



    def _score_single(self, text: str) -> float:
        """Return sentiment score for a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        logits = self.model(**inputs).logits[0].detach().numpy()
        probs = softmax(logits)
        return float(probs[2] - probs[0])

    def fetch_headlines(self, ticker: str):
        """Fetch news headlines for a ticker via Google RSS."""
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}")
        return [e.title for e in feed.entries], feed

    def aggregate_sentiment(self, tickers: list):
        """Return mean sentiment per ticker and feeds dict."""
        scores, feeds = {}, {}
        for t in tickers:
            titles, feed = self.fetch_headlines(t)
            feeds[t] = feed
            if titles:
                vals = self._score_texts(titles)
                scores[t] = float(np.mean(vals))
            else:
                scores[t] = 0.0
        return scores, feeds

    def build_dataframe(self, tickers: list) -> pd.DataFrame:
        """
        Build a dataframe of headlines with per-title sentiment,
        plus daily aggregated sentiment per ticker.
        """
        scores, feeds = self.aggregate_sentiment(tickers)

        items = []
        for ticker in tickers:
            for e in feeds[ticker].entries:
                items.append({
                    "Ticker": ticker,
                    "Title": e.title,
                    "Link": e.link,
                    "Published": e.published,
                    "Sentiment": self._score_single(e.title)
                })
        df = pd.DataFrame(items)

        if df.empty:
            return df, None

        df['Published'] = pd.to_datetime(df['Published'])
        df["Date"] = df["Published"].dt.date

        df_daily = (
            df.groupby(["Ticker", "Date"])["Sentiment"]
              .mean()
              .reset_index()
              .rename(columns={"Sentiment": "DailySentiment"})
        )

        sentiment_wide = df_daily.pivot(index='Date', columns='Ticker', values='DailySentiment')

        return df, sentiment_wide

"""#3. Regime Detector"""

# ---------- Regime Detector ----------
class RegimeDetector:
    """
    Detects market regimes using HMM (Gaussian emissions) or GMM clustering.
    Fit on feature matrix (e.g. returns, vol, macro features).
    """
    def __init__(self, n_regimes: int = 3, method: str = 'hmm', random_state: int = 42):
        assert method in ('hmm', 'gmm')
        self.n_regimes = n_regimes
        self.method = method
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray):
        if self.method == 'hmm':
            try:
                from hmmlearn.hmm import GaussianHMM
            except Exception as e:
                raise ImportError("Install hmmlearn: pip install hmmlearn") from e
            self.model = GaussianHMM(n_components=self.n_regimes, covariance_type='full', random_state=self.random_state, n_iter=200)
            self.model.fit(X)
        else:
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(n_components=self.n_regimes, random_state=self.random_state, covariance_type='full')
            self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if self.method == 'hmm':
            return self.model.predict(X)
        else:
            return self.model.predict(X)

"""#4. Risk Forecaster"""

# ---------- Risk Forecaster ----------
class RiskForecaster:
    """
    Risk forecasting with GARCH (per-asset vol) and rolling covariance.
    For GARCH, uses arch package.
    """
    def __init__(self, garch_order=(1,1), rolling_window=60):
        self.garch_order = garch_order
        self.rolling_window = rolling_window
        self.garch_models = {}  # store fitted models per asset

    def fit_garch_for_series(self, returns: pd.Series, asset_name: str):
        try:
            from arch import arch_model
        except Exception as e:
            raise ImportError("Install arch: pip install arch") from e
        am = arch_model(returns.dropna()*100, vol='Garch', p=self.garch_order[0], q=self.garch_order[1], dist='normal')
        res = am.fit(disp='off')
        self.garch_models[asset_name] = res
        return res

    def forecast_volatility(self, asset_name: str, horizon=1) -> float:
        res = self.garch_models.get(asset_name, None)
        if res is None:
            raise RuntimeError("No GARCH model for asset. Call fit_garch_for_series first.")
        f = res.forecast(horizon=horizon, reindex=False)
        # annualize? depends on returns frequency. We'll return daily vol estimate
        var = f.variance.iloc[-1].mean()
        vol = np.sqrt(var)/100.0
        return vol

    def rolling_covariance(self, price_df: pd.DataFrame) -> pd.DataFrame:
        # price_df: columns are tickers, index are dates. returns are simple returns.
        returns = price_df.pct_change().dropna()
        cov = returns.rolling(self.rolling_window).cov(pairwise=True)
        # For simple use: return last covariance matrix
        last_idx = cov.index.get_level_values(0).unique()[-1]
        last_cov = cov.xs(last_idx)
        return last_cov

"""#5. Reinforcement Learning Agent"""

# ---------- Reinforcement Learning Agent (optional) ----------
class RLAgent:
    """
    Optional PPO agent to learn dynamic rebalancing. This is a skeleton to show how you'd attach SB3.
    Requires stable-baselines3 and gym; user must implement a custom environment.
    """
    def __init__(self, policy="MlpPolicy", **ppo_kwargs):
        self.policy = policy
        self.ppo_kwargs = ppo_kwargs
        self.model = None

    def build(self, env):
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            raise ImportError("Install stable-baselines3 and gym: pip install stable-baselines3 gym") from e
        self.model = PPO(self.policy, env, verbose=0, **self.ppo_kwargs)
        return self

    def train(self, total_timesteps=100_000):
        if self.model is None:
            raise RuntimeError("Call build(env) with an environment first.")
        self.model.learn(total_timesteps=total_timesteps)
        return self

    def act(self, obs):
        if self.model is None:
            raise RuntimeError("Model not built/trained.")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

"""# 6. Pipeline Coordinator"""

class PipelineCoordinator:
    """
    Unified pipeline integrating:
     - ReturnPredictor (LightGBM/XGBoost)
     - SentimentAnalyzer (FinBERT)
     - RegimeDetector (HMM/GMM)
     - RiskForecaster (GARCH + rolling covariance)
     - RLAgent (PPO, optional)
    """

    def __init__(self, tickers: list, use_rl: bool = False,
                 return_model_cfg: dict = None,
                 regime_cfg: dict = None,
                 risk_cfg: dict = None,
                 rl_cfg: dict = None):
        self.tickers = tickers
        self.use_rl = use_rl

        # Initialize modules
        self.return_predictor = AIForecasterLGB(**(return_model_cfg or {}))
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = RegimeDetector(**(regime_cfg or {}))
        self.risk_forecaster = RiskForecaster(**(risk_cfg or {}))
        self.rl_agent = RLAgent(**(rl_cfg or {})) if use_rl else None

        self.trained = False

    # ------------------ Price/News Feature Creation ------------------
    @staticmethod
    def _make_lag_features(price_df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
        returns = price_df.pct_change().dropna()
        feat_list = []
        for lag in range(1, lags + 1):
            feat_list.append(returns.shift(lag).add_prefix(f"r_lag{lag}_"))
        vol = returns.rolling(20).std().add_prefix("vol_")
        feats = pd.concat(feat_list + [vol], axis=1).dropna()
        return feats

    def build_features(self, price_df: pd.DataFrame, news_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Builds feature matrix from wide-format price and sentiment data.

        Parameters
        ----------
        price_df : pd.DataFrame
            DataFrame indexed by Date, columns are tickers with price values.
        news_df : pd.DataFrame, optional
            DataFrame indexed by Date, columns are tickers with sentiment scores.

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with [Date, Ticker] and features per ticker.
        """
        # Ensure proper datetime index
        price_df.index = pd.to_datetime(price_df.index)
        feats = self._make_lag_features(price_df)

        # Make lag-based features per ticker
        feats_list = []
        for ticker in price_df.columns:
            # Add sentiment if available
            if news_df is not None and ticker in news_df.columns:
                news_df.index = pd.to_datetime(news_df.index)
                feats[f'{ticker}_sentiment'] = news_df[ticker].reindex(feats.index).fillna(0.0)
            else:
                feats[f'{ticker}_sentiment'] = 0.0

        return feats


    # ------------------ Fit / Train Modules ------------------
    def fit_return_model(self, price_df: pd.DataFrame, macro_df: pd.DataFrame = None, horizon: int = 1):
        return self.return_predictor.fit(price_df, horizon=horizon, macro_df=macro_df)

    def fit_sentiment(self):
        # SentimentAnalyzer does not require fitting
        return self.sentiment_analyzer

    def fit_regime(self, feature_matrix: np.ndarray):
        return self.regime_detector.fit(feature_matrix)

    def fit_risk(self, price_df: pd.DataFrame):
        for t in price_df.columns:
            returns = price_df[t].pct_change().dropna()
            self.risk_forecaster.fit_garch_for_series(returns, t)
        return self.risk_forecaster

    def train_rl(self, env, total_timesteps: int = 100_000):
        if not self.use_rl or env is None:
            return None
        self.rl_agent.build(env)
        self.rl_agent.train(total_timesteps=total_timesteps)
        return self.rl_agent

    # ------------------ Run Full Pipeline ------------------
    def run_pipeline(self,
                     price_df: pd.DataFrame,
                     macro_df: pd.DataFrame = None,
                     feature_matrix: np.ndarray = None,
                     news_texts_by_date: dict = None,
                     env=None,
                     rl_timesteps: int = 100_000) -> dict:

        results = {}

        # 1. Forecast returns & covariance
        exp_rets, cov, exp_vols = self.return_predictor.predict(macro_df=macro_df)
        results['forecaster'] = {'expected_returns': exp_rets, 'covariance': cov, "expected_volatility":exp_vols}

        # 2. Sentiment
        df_sentiment, sentiment_wide = self.sentiment_analyzer.build_dataframe(self.tickers)
        results['sentiment'] = {'df': df_sentiment, 'daily_pivot': sentiment_wide}

        # 3. Regime detection
        if feature_matrix is not None:
            regimes = self.regime_detector.predict(feature_matrix)
            results['regime'] = regimes
        else:
            results['regime'] = None

        # 4. Risk forecasting
        rolling_cov = self.risk_forecaster.rolling_covariance(price_df)
        per_asset_vol = {t: self.risk_forecaster.forecast_volatility(t) for t in self.tickers}
        results['risk'] = {'rolling_cov': rolling_cov, 'per_asset_vol': per_asset_vol}

        # 5. Optional RL agent
        if self.use_rl and env is not None:
            self.train_rl(env, total_timesteps=rl_timesteps)
            results['rl_agent'] = self.rl_agent
        else:
            results['rl_agent'] = None

        self.trained = True
        return results

    # ------------------ Generate Trade Signals ------------------
    def generate_signals(self, features: pd.DataFrame, price_df: pd.DataFrame, macro_df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """
        Simple strategy:
         - predict aggregated future return from features
         - rank by predicted score
         - allocate weights to top_n long, bottom_n short (equal weight), scaled by risk (vol)
        """
        pred_rets, _, _ = self.return_predictor.predict(macro_df)
        pred_series = pd.Series(pred_rets, index=self.tickers, name='pred')
        last_date = price_df.index[-1]

        recent_returns = price_df.pct_change().iloc[-5:].mean().sort_values(ascending=False)
        longs = recent_returns.index[:top_n]
        shorts = recent_returns.index[-top_n:]
        weights = pd.Series(0.0, index=self.tickers)

        if len(longs) > 0:
            weights.loc[longs] = 0.5 / len(longs)
        if len(shorts) > 0:
            weights.loc[shorts] = -0.5 / len(shorts)

        # scale by inverse vol
        vols = {}
        for t in weights.index:
            try:
                vols[t] = self.risk_forecaster.forecast_volatility(t)
            except Exception:
                vols[t] = np.nan
        vols_series = pd.Series(vols)
        inv_vol = 1.0 / (vols_series.replace(0, np.nan) + 1e-8)
        inv_vol = inv_vol.fillna(inv_vol.mean())
        weights = weights * (inv_vol / inv_vol.abs().sum())  # normalize

        return weights
