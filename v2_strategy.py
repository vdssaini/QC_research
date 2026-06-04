#region imports
from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
#endregion

# ============================================================
# V2 — IR Precision Falcon  (Enhanced)
# ============================================================
# Key improvements over V1:
#
#  1. FEATURES  : 8 features instead of 2
#                 multi-horizon returns (5d/10d/21d/63d),
#                 rolling volatility, RSI proxy, volume ratio,
#                 multi-horizon active return
#  2. TRAINING  : Non-overlapping samples to reduce look-ahead
#                 bias; 750-day window for regime diversity
#  3. MODEL     : 200 trees, max_depth=5, class_weight balanced
#  4. SIZING    : Probability-weighted allocation across top-3
#                 candidates (diversified alpha basket)
#  5. RISK MGMT : Per-position trailing stop-loss (-7 %)
#  6. SCHEDULE  : Bi-weekly retrain + daily scoring
#  7. TURNOVER  : Only rebalance when allocation drift > 10 %
# ============================================================

class IRPrecisionFalcon_V2(QCAlgorithm):

    # ── tuneable parameters ───────────────────────────────────
    MIN_CONFIDENCE  = 0.65   # Slightly lower to allow top-3 diversification
    MAX_POSITIONS   = 3      # Alpha basket size
    STOP_LOSS_PCT   = 0.07   # 7 % trailing stop per position
    REBALANCE_DRIFT = 0.10   # Only rebalance if weight drift > 10 %
    HISTORY_DAYS    = 750    # Training lookback (≈ 3 years)

    def Initialize(self):
        # ── Account ──────────────────────────────────────────
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 30)
        self.SetCash(10_000)

        # ── Universe ─────────────────────────────────────────
        self.tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "AMD"]
        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in self.tickers]

        # ── Benchmark ────────────────────────────────────────
        self._bench = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.SetBenchmark(self._bench)

        # ── ML state ─────────────────────────────────────────
        self.classifier = None
        self.scaler     = StandardScaler()
        self._entry_prices: dict = {}          # symbol → entry price for stop-loss

        # ── Warm-up & schedules ───────────────────────────────
        self.SetWarmUp(252)

        # Retrain every 2 weeks — more adaptive than monthly
        self.Train(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.At(0, 0),
            self.TrainModel,
        )

    # ─────────────────────────────────────────────────────────
    # Feature engineering helpers
    # ─────────────────────────────────────────────────────────
    def _build_features(self, closes: pd.DataFrame) -> pd.DataFrame:
        """
        Build an 8-feature matrix aligned to closes index.

        Features
        --------
        f0  5-day return
        f1  10-day return
        f2  21-day return
        f3  63-day return
        f4  10-day realised volatility (annualised std of daily log-returns)
        f5  RSI proxy (14-day):  avg-up / (avg-up + avg-down)
        f6  Volume ratio:        recent 5d avg vol / 20d avg vol
        f7  10-day active return vs QQQ
        """
        bench = self._bench.Value
        feats = {}

        daily_ret = closes.pct_change()
        log_ret   = np.log1p(daily_ret)

        for col in closes.columns:
            r5   = closes[col].pct_change(5)
            r10  = closes[col].pct_change(10)
            r21  = closes[col].pct_change(21)
            r63  = closes[col].pct_change(63)

            # Volatility: 10-day rolling std of log returns (annualised)
            vol10 = log_ret[col].rolling(10).std() * np.sqrt(252)

            # RSI proxy (14-day)
            delta     = daily_ret[col]
            gain      = delta.clip(lower=0).rolling(14).mean()
            loss      = (-delta.clip(upper=0)).rolling(14).mean()
            rsi_proxy = gain / (gain + loss + 1e-9)

            # Active return
            act10 = r10 - closes[bench].pct_change(10)

            feats[col] = pd.DataFrame({
                'r5': r5, 'r10': r10, 'r21': r21, 'r63': r63,
                'vol10': vol10, 'rsi': rsi_proxy, 'act10': act10,
            })

        return feats

    # ─────────────────────────────────────────────────────────
    def TrainModel(self):
        """
        Train on 750 days of history using non-overlapping
        5-day windows to avoid serial-correlation in labels.
        """
        history = self.History(self.symbols + [self._bench], self.HISTORY_DAYS, Resolution.Daily)
        if history.empty:
            return

        closes = history['close'].unstack(level=0)

        # Inject synthetic volume if unavailable (QC sometimes splits columns)
        try:
            vol_hist = self.History(self.symbols, self.HISTORY_DAYS, Resolution.Daily)
            volumes  = vol_hist['volume'].unstack(level=0) if not vol_hist.empty else None
        except Exception:
            volumes = None

        feat_map = self._build_features(closes)
        X, y     = [], []

        bench_close = closes[self._bench.Value]

        for s in self.symbols:
            col = s.Value
            if col not in closes.columns:
                continue

            df = feat_map[col].copy()
            if volumes is not None and col in volumes.columns:
                vol_ratio = volumes[col].rolling(5).mean() / (volumes[col].rolling(20).mean() + 1e-9)
                df['vol_ratio'] = vol_ratio
            else:
                df['vol_ratio'] = 1.0

            df = df.dropna()

            # NON-OVERLAPPING step: sample every 5 rows to avoid look-ahead leakage
            indices = df.index[::5]

            for idx in indices:
                loc = closes.index.get_loc(idx)
                if loc + 5 >= len(closes):
                    continue

                row = df.loc[idx]
                if row.isnull().any():
                    continue

                X.append(row.values)

                # Label: 1 if stock outperforms QQQ over next 5 days
                f_s = closes[col].iloc[loc + 5] / closes[col].iloc[loc] - 1
                f_b = bench_close.iloc[loc + 5] / bench_close.iloc[loc] - 1
                y.append(1 if f_s > f_b else 0)

        if len(X) < 50:
            return

        X_scaled = self.scaler.fit_transform(X)
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight='balanced',   # handles imbalanced labels
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled, y)
        self.classifier = clf
        self.Log(f"[V2] Retrained on {len(X)} samples, "
                 f"CV-accuracy ≈ {cross_val_score(clf, X_scaled, y, cv=3).mean():.2%}")

    # ─────────────────────────────────────────────────────────
    def _live_features(self) -> dict:
        """
        Return {symbol_value: feature_vector} for live scoring.
        Needs 65 days of history to compute 63-day return + vol.
        """
        n_days = 70
        hist   = self.History(self.symbols + [self._bench], n_days, Resolution.Daily)
        if hist.empty or 'close' not in hist.columns:
            return {}

        closes  = hist['close'].unstack(level=0)
        if len(closes) < 65:
            return {}

        try:
            vol_hist = self.History(self.symbols, n_days, Resolution.Daily)
            volumes  = vol_hist['volume'].unstack(level=0) if not vol_hist.empty else None
        except Exception:
            volumes = None

        feat_map = self._build_features(closes)
        result   = {}

        for s in self.symbols:
            col = s.Value
            if col not in feat_map:
                continue

            df  = feat_map[col]
            row = df.iloc[-1]

            if volumes is not None and col in volumes.columns:
                vr = volumes[col].rolling(5).mean().iloc[-1] / (
                     volumes[col].rolling(20).mean().iloc[-1] + 1e-9)
            else:
                vr = 1.0

            vec = np.append(row.values, vr)
            if np.isnan(vec).any():
                continue

            result[col] = vec

        return result

    # ─────────────────────────────────────────────────────────
    def _check_stops(self):
        """Liquidate any position that has fallen > STOP_LOSS_PCT from entry."""
        for s in list(self._entry_prices.keys()):
            if not self.Portfolio[s].Invested:
                del self._entry_prices[s]
                continue
            entry = self._entry_prices[s]
            price = self.Securities[s].Price
            if price > 0 and (entry - price) / entry > self.STOP_LOSS_PCT:
                self.Liquidate(s)
                self.Log(f"[V2] Stop-loss triggered: {s.Value} "
                         f"entry={entry:.2f} current={price:.2f}")
                del self._entry_prices[s]

    # ─────────────────────────────────────────────────────────
    def OnData(self, data):
        if self.IsWarmingUp or self.classifier is None:
            return

        # ── 1. Risk management first ─────────────────────────
        self._check_stops()

        # ── 2. Score universe ────────────────────────────────
        live = self._live_features()
        if not live:
            return

        candidates = []
        for s in self.symbols:
            col = s.Value
            if col not in live:
                continue
            prob = self.classifier.predict_proba(
                self.scaler.transform([live[col]])
            )[0][1]
            if prob > self.MIN_CONFIDENCE:
                candidates.append((s, prob))

        # ── 3. Execution ─────────────────────────────────────
        if candidates:
            # Pick top-MAX_POSITIONS by probability
            top = sorted(candidates, key=lambda x: x[1], reverse=True)[: self.MAX_POSITIONS]

            # Probability-weighted target weights
            total_prob   = sum(p for _, p in top)
            target       = {s: 0.96 * p / total_prob for s, p in top}

            # Rebalance only if drift exceeds threshold
            needs_rebal  = False
            invested_set = {s for s in self.symbols if self.Portfolio[s].Invested}
            target_set   = set(target.keys())

            if invested_set != target_set:
                needs_rebal = True
            else:
                for s, w in target.items():
                    current_w = self.Portfolio[s].HoldingsValue / max(self.Portfolio.TotalPortfolioValue, 1)
                    if abs(current_w - w) > self.REBALANCE_DRIFT:
                        needs_rebal = True
                        break

            if needs_rebal:
                # Liquidate names no longer in target
                for s in invested_set - target_set:
                    self.Liquidate(s)
                    self._entry_prices.pop(s, None)

                # Set target weights; record entry price for new positions
                for s, w in target.items():
                    if not self.Portfolio[s].Invested:
                        self._entry_prices[s] = self.Securities[s].Price
                    self.SetHoldings(s, w)

        else:
            # No alpha signal → park in QQQ (zero tracking error default)
            if not self.Portfolio[self._bench].Invested:
                self.Liquidate()
                self._entry_prices.clear()
                self.SetHoldings(self._bench, 0.98)
