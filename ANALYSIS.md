# IR Precision Falcon — Strategy Analysis & Comparison

## Overview

Both strategies target **Information Ratio (IR)** maximisation vs QQQ using a Random Forest classifier on a tech-heavy universe (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA, AMD). When no alpha signal exists, both park 98% in QQQ to hold tracking error near zero.

---

## V1 — Pros & Cons

### Pros
| # | Point |
|---|-------|
| 1 | **Simple, debuggable logic** — easy to reason about; few moving parts |
| 2 | **High confidence threshold (0.70)** reduces whipsaw trades and keeps turnover low |
| 3 | **QQQ default** prevents IR degradation during no-signal periods |
| 4 | **Monthly retraining** captures broad regime shifts |
| 5 | **No margin** — liquidate-before-buy pattern eliminates leverage risk |

### Cons
| # | Point | Impact |
|---|-------|--------|
| 1 | **Only 2 features** (10d return, 10d active return). These are nearly collinear (active_ret = s_ret − b_ret). The model has very limited information to distinguish regimes. | High — underfitting likely |
| 2 | **100% single-name concentration**. If the top pick gaps down, the entire portfolio is hurt. | High — extreme idiosyncratic risk |
| 3 | **Overlapping training windows** (every row of a 10-day rolling return used sequentially) produces autocorrelated samples, inflating apparent training accuracy. | Medium — over-optimism in training metrics |
| 4 | **No stop-loss or risk control** beyond confidence threshold. An adverse move compounds without a floor. | High — drawdown unbounded |
| 5 | **Monthly retrain** too infrequent for high-volatility tech names; regime can shift in days (e.g., NVDA earnings, Fed announcements). | Medium |
| 6 | **`all_closes.iloc[i+5]` misalignment risk** — `all_closes` and `ret_10d` have different row counts after `dropna()`, so index `i` in `ret_10d` does not necessarily correspond to index `i` in `all_closes`. Can cause subtle look-ahead bias or index errors. | High — potential data-integrity issue |
| 7 | **Switching cost underestimated** — daily `OnData` with Liquidate → SetHoldings creates high effective turnover; slippage and commissions erode alpha. | Medium |

---

## V2 — Improvements

| Dimension | V1 | V2 |
|-----------|----|----|
| **Features** | 2 (10d return, 10d active return) | 8: 5d/10d/21d/63d returns, 10d volatility, RSI proxy (14d), volume ratio, 10d active return |
| **Training window** | 500 days | 750 days (≈ 3 full years, more regime diversity) |
| **Training sampling** | All overlapping rows | Non-overlapping every-5th-row sample — breaks autocorrelation |
| **Model config** | 100 trees, depth=4 | 200 trees, depth=5, `class_weight='balanced'` |
| **Confidence bar** | 0.70 | 0.65 (lower bar enabled because diversification absorbs individual errors) |
| **Position sizing** | 100% in #1 name | Probability-weighted across top-3 names (sum = 96%) |
| **Rebalance logic** | Every day unconditionally | Only if allocation drift > 10% — reduces unnecessary turnover |
| **Retrain schedule** | Monthly | Every Monday — bi-weekly captures faster regime changes |
| **Stop-loss** | None | 7% trailing stop-loss per position |
| **Data alignment** | Potential misalignment bug | Explicit `get_loc` indexing against `closes` DataFrame |
| **CV monitoring** | None | 3-fold cross-validation accuracy logged at each retrain |

---

## Expected Backtest Characteristics (2020 – 2025)

The period spans several distinct market regimes:

| Period | Regime | Key challenge |
|--------|--------|---------------|
| Jan–Mar 2020 | COVID crash | Sudden drawdown; no model trained on it |
| Apr 2020–Dec 2021 | FAANG/NVDA mega-rally | Momentum strongly rewarded |
| Jan–Dec 2022 | Tech bear market | Active return systematically negative for universe |
| Jan–Dec 2023 | AI-driven recovery (NVDA ×3) | Concentrated alpha in a single name |
| 2024–2025 | Rate-cut cycle, mixed sentiment | Rotation between names |

### V1 Expected Behaviour
- **Strong in 2021, 2023** — momentum features capture tech rallies well.
- **Weak in 2022** — single-name concentration amplifies losses; no stop-loss.
- **High Sharpe periods followed by sharp drawdowns** — typical of undiversified momentum.
- **IR likely unstable**: positive in trending years, near-zero or negative in mean-reverting years.

### V2 Expected Behaviour
- **More consistent IR across regimes** — multi-horizon features help in both trending and mean-reverting markets.
- **Lower max drawdown** — stop-loss cap and 3-name diversification limit catastrophic single-stock events.
- **Slightly lower peak return in best years** — cost of diversification; but better Sharpe ratio.
- **Turnover reduction** — drift-threshold rebalancing saves 20–40 bps/year in transaction costs.

---

## Quantitative Comparison — Simulated Metrics

> The numbers below are **illustrative estimates** derived from strategy design analysis, not live QuantConnect backtest output. Run both `v1_strategy.py` and `v2_strategy.py` on QuantConnect to get precise figures.

| Metric | V1 (estimate) | V2 (estimate) | Δ |
|--------|--------------|--------------|---|
| CAGR | 18 – 24% | 16 – 22% | V2 slightly lower in bull years |
| Sharpe Ratio | 0.85 – 1.10 | 1.05 – 1.35 | **V2 better** |
| Max Drawdown | -35 – -45% | -20 – -28% | **V2 better** |
| Information Ratio vs QQQ | 0.40 – 0.65 | 0.55 – 0.80 | **V2 better** |
| Annual Turnover | 400 – 600% | 150 – 250% | **V2 better** |
| Avg # positions | 1 | 2 – 3 | V2 diversified |
| Confidence threshold hits | ~15% of days | ~25% of days | V2 trades more often |

---

## Running the Strategies on QuantConnect

1. Open [QuantConnect](https://www.quantconnect.com/terminal).
2. Create a new algorithm, paste the contents of `v1_strategy.py` (class `IRPrecisionFalcon_V1`).
3. Run backtest from **2020-01-01** to **2025-12-30** with **$10,000** starting capital.
4. Note the following from the results panel:
   - **Information Ratio** (vs QQQ)
   - **Sharpe Ratio**
   - **Max Drawdown**
   - **Compounding Annual Return**
5. Repeat steps 2–4 for `v2_strategy.py` (class `IRPrecisionFalcon_V2`).
6. Use the **Strategy Equity** chart overlay to compare cumulative returns.

---

## Key Takeaways

1. **V1 is a solid proof-of-concept** but its 2-feature model and 100% concentration make it a high-variance bet on continued tech momentum.
2. **V2 addresses the three most critical flaws** (feature richness, concentration risk, missing stop-loss) while keeping the same core logic and QQQ fallback.
3. **The QQQ-default regime is the most important design choice in both versions** — it guarantees that IR never catastrophically degrades during no-signal periods.
4. **ML in trading lives and dies by feature quality** — adding multi-horizon returns and volatility gives the forest enough orthogonal signal to generalise beyond the training period.
5. **For live trading**, consider adding: macro regime filters (VIX level, yield-curve slope), earnings-blackout windows (avoid holding 2 days into earnings), and realistic slippage models.
