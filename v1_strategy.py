#region imports
from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#endregion

# ============================================================
# V1 — Original IR Precision Falcon Strategy
# ============================================================
# Focus  : Maximize Information Ratio vs QQQ
# Model  : Random Forest (2 features, 100 trees, depth=4)
# Sizing : 100% concentration in highest-confidence name
# Regime : Default to QQQ when no signal (zero tracking error)
# Retrain: Monthly
# ============================================================

class IRPrecisionFalcon_V1(QCAlgorithm):

    def Initialize(self):
        # ── Account ──────────────────────────────────────────
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 30)
        self.SetCash(10_000)

        # ── Universe ─────────────────────────────────────────
        self.tickers  = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "AMD"]
        self.symbols  = [self.AddEquity(t, Resolution.Daily).Symbol for t in self.tickers]

        # ── Benchmark ────────────────────────────────────────
        self._bench = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.SetBenchmark(self._bench)

        # ── ML ───────────────────────────────────────────────
        self.classifier     = None
        self.scaler         = StandardScaler()
        self.min_confidence = 0.70          # High bar → fewer trades, lower TE

        # ── Warm-up & training schedule ──────────────────────
        self.SetWarmUp(252)
        self.Train(self.DateRules.MonthStart(), self.TimeRules.At(0, 0), self.TrainModel)

    # ─────────────────────────────────────────────────────────
    def TrainModel(self):
        """
        Train a binary classifier to predict whether each equity
        will outperform QQQ over the next 5 trading days.

        Features : [10-day return, 10-day active return vs QQQ]
        Label    : 1 if stock beats QQQ over the next 5 days
        """
        history = self.History(self.symbols + [self._bench], 500, Resolution.Daily)
        if history.empty:
            return

        all_closes = history['close'].unstack(level=0)
        ret_10d    = all_closes.pct_change(10).dropna()

        features, labels = [], []
        bench_ret = ret_10d[self._bench.Value]

        for s in self.symbols:
            if s.Value not in ret_10d.columns:
                continue

            s_ret      = ret_10d[s.Value]
            active_ret = s_ret - bench_ret

            for i in range(len(active_ret) - 5):
                features.append([s_ret.iloc[i], active_ret.iloc[i]])

                # Forward 5-day return comparison
                f_s = (all_closes[s.Value].iloc[i + 5] / all_closes[s.Value].iloc[i]) - 1
                f_b = (all_closes[self._bench.Value].iloc[i + 5] / all_closes[self._bench.Value].iloc[i]) - 1
                labels.append(1 if f_s > f_b else 0)

        if features:
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42
            )
            self.classifier.fit(self.scaler.fit_transform(features), labels)

    # ─────────────────────────────────────────────────────────
    def OnData(self, data):
        if self.IsWarmingUp or self.classifier is None:
            return

        # ── Live feature calculation ──────────────────────────
        hist = self.History(self.symbols + [self._bench], 11, Resolution.Daily)
        if hist.empty or 'close' not in hist.columns:
            return
        hist = hist['close'].unstack(level=0)
        if len(hist) < 11:
            return

        cur_rets = hist.pct_change(10).iloc[-1]
        b_ret    = cur_rets.get(self._bench.Value)
        if b_ret is None:
            return

        # ── Score each candidate ─────────────────────────────
        candidates = []
        for s in self.symbols:
            if s.Value not in cur_rets:
                continue
            s_ret = cur_rets[s.Value]
            feat  = self.scaler.transform([[s_ret, s_ret - b_ret]])
            prob  = self.classifier.predict_proba(feat)[0][1]
            if prob > self.min_confidence:
                candidates.append((s, prob))

        # ── Execution ────────────────────────────────────────
        if candidates:
            # Scenario A: concentrate 98% in the top signal
            best_s = max(candidates, key=lambda x: x[1])[0]
            if not self.Portfolio[best_s].Invested:
                self.Liquidate()
                self.SetHoldings(best_s, 0.98)
        elif not self.Portfolio[self._bench].Invested:
            # Scenario B: park in QQQ → zero tracking error baseline
            self.Liquidate()
            self.SetHoldings(self._bench, 0.98)
