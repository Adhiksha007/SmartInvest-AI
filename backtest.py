import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
import math

# A simple realized trade record structure for closed chunks
RealizedTrade = namedtuple("RealizedTrade", [
    "ticker", "open_date", "close_date", "side",
    "open_price", "close_price", "shares", "pnl_amt", "pnl_pct", "fees", "reason"
])

def vectorized_backtest_ohlc(
    ohlc_data,
    weights,
    cfg,
    sector_map=None,
    initial_capital=None,
    rebalance_freq="monthly"
):
    """
    Vectorized-ish daily backtest with:
      - monthly rebalancing (first trading day of month)
      - full rebalance to target weights at each rebalance
      - percent_of_equity / fixed_notional sizing (reads cfg.size_config)
      - stop-loss & take-profit (percent based) for longs & shorts
      - slippage (fraction) and transaction_cost (fraction of trade value)
      - support for shorting; leverage enforced via max_total_leverage
      - detailed trade_log for every executed order
    Returns:
      {
        'equity_curve': pd.Series,
        'trade_stats': dict,
        'final_positions': dict,
        'trade_log': list(dict),
      }
    """

    # -------------------------
    # config defaults
    # -------------------------
    transaction_cost = getattr(cfg, "transaction_cost", 0.0)   # fraction of trade value
    slippage = getattr(cfg, "slippage", 0.0)                 # fraction of price
    size_cfg = getattr(cfg, "size_config", None)
    stop_loss_pct = getattr(cfg, "stop_loss_pct", None)
    take_profit_pct = getattr(cfg, "take_profit_pct", None)
    max_exposure_per_ticker = getattr(cfg, "max_exposure_per_ticker", np.inf)
    max_exposure_per_sector = getattr(cfg, "max_exposure_per_sector", np.inf)
    max_total_leverage = getattr(cfg, "max_total_leverage", 1.0)  # e.g., 1.5 means 150% gross exposure allowed
    if initial_capital is None:
        initial_capital = getattr(cfg, "initial_capital", 100000.0)

    # sizing mode support (we'll honor percent_of_equity and fixed_notional)
    size_mode = None
    percent_of_equity = None
    fixed_notional = None
    if size_cfg is not None:
        size_mode = getattr(size_cfg, "mode", None)
        percent_of_equity = getattr(size_cfg, "percent_of_equity", None)
        fixed_notional = getattr(size_cfg, "fixed_notional", None)
        risk_per_trade = getattr(size_cfg, "risk_per_trade", None)

    # Validate tickers / weights length
    tickers = list(ohlc_data.keys())
    if len(weights) != len(tickers):
        raise ValueError("length of weights must equal number of tickers in ohlc_data")

    # Ensure DateTimeIndex and required columns
    for t in tickers:
        if not isinstance(ohlc_data[t].index, pd.DatetimeIndex):
            ohlc_data[t] = ohlc_data[t].copy()
            ohlc_data[t].index = pd.to_datetime(ohlc_data[t].index)
        for col in ["Open", "High", "Low", "Close"]:
            if col not in ohlc_data[t].columns:
                raise ValueError(f"{t} missing required column {col}")
    # Intersect dates: keep dates where at least one ticker trades, but for rebalancing we need common availability.
    # We'll build union of dates and forward-fill missing price rows with NaNs so we skip on missing prices.
    all_dates = sorted(set().union(*[set(df.index) for df in ohlc_data.values()]))
    dates_index = pd.DatetimeIndex(all_dates).sort_values()

    # Reindex each dataframe to full union dates (no fill)
    data = {}
    for t in tickers:
        data[t] = ohlc_data[t].reindex(dates_index).copy()

    # Identify rebalance dates: first trading day of each month where at least one ticker has data.
    if rebalance_freq == "monthly":
        # group dates by year-month and pick the first index that has at least one non-null Open across tickers
        months = {}
        for d in dates_index:
            months[(d.year, d.month)] = months.get((d.year, d.month), []) + [d]
            rebalance_dates = []

            for ym, ds in months.items():
                for d in ds:
                    ok = False
                    for t in tickers:
                        df = data[t]
                        if d in df.index:
                            val = df.loc[d, "Open"]
                            # handle scalar or Series (in case of duplicates)
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            if pd.notna(val):
                                ok = True
                                break  # at least one ticker has valid Open
                    if ok:
                        rebalance_dates.append(d)
                        break

    else:
        raise ValueError("Only 'monthly' rebalance_freq is implemented in this function.")

    rebalance_set = set(rebalance_dates)


    # -------------------------
    # state containers
    # -------------------------
    cash = float(initial_capital)  # cash + proceeds from short sales
    # positions: dict of ticker -> dict with 'shares' (positive = long, negative = short),
    # 'avg_price' (average entry price for the current net position), and 'lots' list for FIFO closure.
    positions = {t: {"shares": 0, "avg_price": None, "sector": sector_map.get(t, "Unknown") if sector_map else "Unknown"} for t in tickers}
    # We'll also keep per-ticker lot FIFO list: each lot = dict(shares, price, open_date, side)
    lots = {t: [] for t in tickers}

    sector_exposure = defaultdict(float)   # absolute exposure per sector
    trade_log = []                         # every executed trade (dict)
    realized_trades = []                   # RealizedTrade namedtuple list for closed lots
    equity_curve = []                      # (date, equity) snapshots

    # helper: compute market values for all tickers on date index i
    def market_values_on_date(i):
        mv = {}
        close_prices = {}
        for t in tickers:
            price = data[t]["Close"].iloc[i]
            close_prices[t] = price
            # ensure price is scalar
            if isinstance(price, pd.Series):
                price = price.iloc[0]

            mv[t] = positions[t]["shares"] * (price if not pd.isna(price) else 0.0)

        return mv, close_prices

    def gross_exposure_from_prices(prices):
        return sum(abs(positions[t]["shares"]) * (prices.get(t, 0.0) if prices.get(t, 0.0) is not None else 0.0) for t in tickers)

    # helper: record a trade (single execution)
    def record_trade(date, ticker, action, price, size, equity_before, equity_after, tx_cost, slip_cost, reason):
        trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": action,  # 'BUY' or 'SELL' (SELL includes short sales / covers)
            "price": float(price),
            "size": int(size),
            "value": float(price * size),
            "equity_before": float(equity_before),
            "equity_after": float(equity_after),
            "transaction_cost": float(tx_cost),
            "slippage_cost": float(slip_cost),
            "reason": reason
        })

    # main loop over all dates in chronological order
    for i, current_date in enumerate(dates_index):
        # gather today's OHLC for tickers
        todays = {t: {"Open": data[t]["Open"].iloc[i], "High": data[t]["High"].iloc[i], "Low": data[t]["Low"].iloc[i], "Close": data[t]["Close"].iloc[i]} for t in tickers}

        # 1) Check SL/TP for existing positions using intraday High/Low.
        # For longs: stop if Low <= entry*(1-stop), take if High >= entry*(1+tp)
        # For shorts: stop if High >= entry*(1+stop) (price rose), take if Low <= entry*(1-tp) (price fell)
        exits = []  # list of dicts for exits to apply
        stops = {}
        for t in tickers:
            pos_shares = positions[t]["shares"]
            if pos_shares == 0:
                continue
            avg_entry = positions[t]["avg_price"]
            if avg_entry is None:
                continue
            high = todays[t]["High"]
            low = todays[t]["Low"]
            open_p = todays[t]["Open"]
            # skip if no price data
            # if high/low are Series, pick the last valid value
            if isinstance(high, pd.Series):
                high = high.dropna()
                high = high.iloc[-1] if not high.empty else np.nan

            if isinstance(low, pd.Series):
                low = low.dropna()
                low = low.iloc[-1] if not low.empty else np.nan

            # now safe to check
            if pd.isna(high) or pd.isna(low):
                continue
            # compute thresholds
            if pos_shares > 0:
                # LONG
                stop_price = avg_entry * (1.0 - stop_loss_pct) if stop_loss_pct is not None else -np.inf
                stops[t] = stop_price
                take_price = avg_entry * (1.0 + take_profit_pct) if take_profit_pct is not None else np.inf
                stop_hit = low <= stop_price
                take_hit = high >= take_price

                if stop_hit or take_hit:
                    # conservative: if both, assume stop (worst case)
                    if stop_hit:
                        exit_price = stop_price * (1.0 + slippage)  # buy slippage increases price when buying; for sell, slippage reduces proceeds; being conservative: reduce proceeds -> increase price? We'll apply slippage as adverse to trade
                        reason = "stop_loss"
                    else:
                        exit_price = take_price * (1.0 - slippage)
                        reason = "take_profit"
                    exits.append({"ticker": t, "shares": pos_shares, "exit_price": exit_price, "reason": reason, "side": "long", "date": current_date})
            else:
                # SHORT (pos_shares < 0)
                stop_price = avg_entry * (1.0 + stop_loss_pct) if stop_loss_pct is not None else np.inf
                stops[t] = stop_price
                take_price = avg_entry * (1.0 - take_profit_pct) if take_profit_pct is not None else -np.inf
                stop_hit = high >= stop_price
                take_hit = low <= take_price

                if stop_hit or take_hit:
                    if stop_hit:
                        # close short at stop_price with adverse slippage
                        exit_price = stop_price * (1.0 + slippage)  # we pay higher price to buy back
                        reason = "stop_loss"
                    else:
                        exit_price = take_price * (1.0 - slippage)
                        reason = "take_profit"
                    exits.append({"ticker": t, "shares": abs(pos_shares), "exit_price": exit_price, "reason": reason, "side": "short", "date": current_date})

        # Apply exits (close positions fully for simplicity when SL/TP hits; could be partial)
        for ex in exits:
            t = ex["ticker"]
            close_shares = ex["shares"]
            exit_price = ex["exit_price"]
            reason = ex["reason"]
            side = ex["side"]
            # compute fee & tx cost
            trade_value = close_shares * exit_price
            fee = transaction_cost * trade_value
            # For long exits: we SELL shares -> receive proceeds (sell proceeds - fee)
            equity_before = cash + sum(
                positions[x]["shares"] * (
                    # get scalar close price
                    (todays[x]["Close"].dropna().iloc[-1] if isinstance(todays[x]["Close"], pd.Series) and not todays[x]["Close"].dropna().empty else 0.0)
                )
                for x in tickers
            )

            if side == "long":
                # realize PnL vs avg_entry
                avg_entry = positions[t]["avg_price"]
                pnl_amt = (exit_price - avg_entry) * close_shares - fee
                pnl_pct = (exit_price / avg_entry - 1.0)
                cash += close_shares * exit_price - fee
                # update lots: consume FIFO long lots
                remaining = close_shares
                while remaining > 0 and lots[t]:
                    lot = lots[t][0]
                    if lot["side"] != "long":
                        break  # mismatch; defensive
                    take = min(remaining, lot["shares"])
                    realized = RealizedTrade(
                        ticker=t,
                        open_date=lot["open_date"],
                        close_date=current_date,
                        side="long",
                        open_price=lot["price"],
                        close_price=exit_price,
                        shares=take,
                        pnl_amt=(exit_price - lot["price"]) * take - (transaction_cost * (exit_price * take)),
                        pnl_pct=(exit_price / lot["price"] - 1.0),
                        fees=(transaction_cost * (exit_price * take)),
                        reason=reason
                    )
                    realized_trades.append(realized)
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lots[t].pop(0)
                # update positions
                positions[t]["shares"] -= close_shares
                if positions[t]["shares"] <= 0:
                    positions[t]["avg_price"] = None
                # sector exposure reduce
                sector_exposure[positions[t]["sector"]] = max(0.0, sector_exposure.get(positions[t]["sector"], 0.0) - trade_value)
                equity_after = cash + sum(
                    positions[x]["shares"] * (
                        # get scalar price safely
                        (todays[x]["Close"].dropna().iloc[-1]
                        if isinstance(todays[x]["Close"], pd.Series) and not todays[x]["Close"].dropna().empty
                        else 0.0)
                    )
                    for x in tickers
                )

                record_trade(current_date, t, "SELL", exit_price, close_shares, equity_before, equity_after, fee, 0.0, reason)
            else:
                # SHORT close: buy to cover, pay cash = shares*price + fee
                avg_entry = positions[t]["avg_price"]
                pnl_amt = (avg_entry - exit_price) * close_shares - fee
                pnl_pct = (avg_entry / exit_price - 1.0) if exit_price!=0 else 0.0
                cash -= close_shares * exit_price + fee
                # consume short lots FIFO
                remaining = close_shares
                while remaining > 0 and lots[t]:
                    lot = lots[t][0]
                    if lot["side"] != "short":
                        break
                    take = min(remaining, lot["shares"])
                    realized = RealizedTrade(
                        ticker=t,
                        open_date=lot["open_date"],
                        close_date=current_date,
                        side="short",
                        open_price=lot["price"],
                        close_price=exit_price,
                        shares=take,
                        pnl_amt=(lot["price"] - exit_price) * take - (transaction_cost * (exit_price * take)),
                        pnl_pct=(lot["price"] / exit_price - 1.0) if exit_price!=0 else 0.0,
                        fees=(transaction_cost * (exit_price * take)),
                        reason=reason
                    )
                    realized_trades.append(realized)
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lots[t].pop(0)
                positions[t]["shares"] += close_shares  # since shares were negative
                if positions[t]["shares"] >= 0:
                    positions[t]["avg_price"] = None
                # adjust sector exposure
                sector_exposure[positions[t]["sector"]] = max(0.0, sector_exposure.get(positions[t]["sector"], 0.0) - trade_value)
                equity_after = cash + sum(
                    positions[x]["shares"] * (
                        # get scalar price safely
                        (todays[x]["Close"].dropna().iloc[-1]
                        if isinstance(todays[x]["Close"], pd.Series) and not todays[x]["Close"].dropna().empty
                        else 0.0)
                    )
                    for x in tickers
                )

                record_trade(current_date, t, "BUY_TO_COVER", exit_price, close_shares, equity_before, equity_after, fee, 0.0, reason)

        # 2) Rebalance if current_date is a rebalance date
        if current_date in rebalance_set:
            # 1️⃣ Compute equity before rebalancing
            _, closes = market_values_on_date(i)
            total_mv = sum(
                positions[t]["shares"] * (
                    closes[t].iloc[0] if isinstance(closes[t], pd.Series) and not pd.isna(closes[t].iloc[0])
                    else 0.0
                )
                for t in tickers
            )
            equity = cash + total_mv

            # 2️⃣ Determine desired exposures = weights * equity
            desired_weights = dict(zip(tickers, weights))
            desired_exposures = {t: desired_weights.get(t, 0.0) * equity for t in tickers}

            # 3️⃣ Apply sizing modes
            # --- Compute final desired exposures with sizing rules ---
            for t in tickers:
                target_val = desired_exposures[t]  # initial target dollar exposure from weights
                sign = np.sign(target_val)
                abs_target = abs(target_val)

                # a) Fixed Notional
                if size_mode == "fixed_notional" and fixed_notional is not None:
                    abs_target = fixed_notional

                # b) Percent of Equity
                if size_mode == "percent_of_equity" and percent_of_equity is not None:
                    abs_target = percent_of_equity * equity

                # c) Risk Per Trade (requires stop-loss)
                if size_mode == "risk_per_trade" and risk_per_trade is not None:
                    stop_price = stops.get(t)  # dictionary: ticker -> stop price
                    open_price = todays[t]["Open"]
                    if isinstance(open_price, pd.Series):
                        open_price = open_price.iloc[0]
                    if stop_price is not None and open_price != stop_price:
                        risk_dollar_per_share = abs(open_price - stop_price)
                        if risk_dollar_per_share > 0:
                            max_allowed_exposure = (risk_per_trade * equity) / risk_dollar_per_share
                            abs_target = max_allowed_exposure

                # finalize exposure with sign
                desired_exposures[t] = sign * abs_target

            # 5️⃣ Sector cap
            if sector_map:
                sector_sums = defaultdict(float)
                for t, val in desired_exposures.items():
                    sector_sums[sector_map.get(t, "Unknown")] += abs(val)
                for sector, total in sector_sums.items():
                    if total > max_exposure_per_sector:
                        sector_ts = [t for t in tickers if sector_map.get(t, "Unknown") == sector]
                        sector_current_abs = sum(abs(desired_exposures[t]) for t in sector_ts)
                        if sector_current_abs <= 0:
                            continue
                        scale = max_exposure_per_sector / sector_current_abs
                        for t in sector_ts:
                            desired_exposures[t] *= scale

            # 6️⃣ Enforce leverage cap
            desired_gross = sum(abs(v) for v in desired_exposures.values())
            if equity > 0 and desired_gross > max_total_leverage * equity:
                scale = (max_total_leverage * equity) / desired_gross
                for t in desired_exposures:
                    desired_exposures[t] *= scale

            # 7️⃣ Convert desired dollar exposure to shares
            for t in tickers:
                open_price = todays[t]["Open"]
                if isinstance(open_price, pd.Series):
                    open_price = open_price.iloc[0]
                if pd.isna(open_price) or open_price == 0:
                    continue
                desired_val = desired_exposures[t]
                desired_shares = int(np.floor(desired_val / open_price)) if desired_val > 0 else 0

                # 8️⃣ Compute delta from current position
                current_shares = positions[t]["shares"]
                delta = desired_shares - current_shares
                if delta == 0:
                    continue

                # Execution price and slippage: buys -> higher price, sells -> lower price (adverse)
                exec_price = open_price * (1.0 + slippage) if delta > 0 else open_price * (1.0 - slippage)
                trade_value = abs(delta) * exec_price
                fee = transaction_cost * trade_value

                # If delta > 0 but current_shares < 0 (we're crossing from short to long), we close short first then open long
                equity_before_trade = cash + sum(
                    positions[x]["shares"] * (
                        closes[x].iloc[0] if isinstance(closes[x], pd.Series) and not closes[x].iloc[0] is pd.NA else 0.0
                    )
                    for x in tickers
                )
                # process close-to-zero crossing carefully
                if current_shares < 0 and delta > 0:
                    # closing shorts by buying abs(current_shares) first (partial or full)
                    cover_qty = min(delta, abs(current_shares))
                    cover_price = exec_price  # use same exec price for both cover and new buy portion
                    cover_value = cover_qty * cover_price
                    cover_fee = transaction_cost * cover_value
                    cash -= cover_value + cover_fee  # pay to buy to cover
                    # consume short lots FIFO
                    remaining = cover_qty
                    while remaining > 0 and lots[t]:
                        lot = lots[t][0]
                        if lot["side"] != "short":
                            break
                        take = min(remaining, lot["shares"])
                        realized = RealizedTrade(
                            ticker=t, open_date=lot["open_date"], close_date=current_date, side="short",
                            open_price=lot["price"], close_price=cover_price, shares=take,
                            pnl_amt=(lot["price"] - cover_price) * take - (transaction_cost * (cover_price * take)),
                            pnl_pct=(lot["price"] / cover_price - 1.0) if cover_price!=0 else 0.0,
                            fees=(transaction_cost * (cover_price * take)),
                            reason="rebalance_close_short"
                        )
                        realized_trades.append(realized)
                        lot["shares"] -= take
                        remaining -= take
                        if lot["shares"] == 0:
                            lots[t].pop(0)
                    positions[t]["shares"] += cover_qty  # less negative
                    if positions[t]["shares"] == 0:
                        positions[t]["avg_price"] = None
                    equity_after_trade = cash + sum(
                        positions[x]["shares"] * (
                            closes[x].dropna().iloc[-1] if not closes[x].dropna().empty else 0.0
                        )
                        for x in tickers
                    )
                    record_trade(current_date,
                                 t,
                                 "BUY_TO_COVER",
                                 cover_price,
                                 cover_qty,
                                 equity_before_trade,
                                 equity_after_trade,
                                 cover_fee,
                                 0.0,
                                 "rebalance_close_short")
                    # update delta after closing shorts
                    delta = desired_shares - positions[t]["shares"]
                    if delta == 0:
                        continue
                # Similarly, if current_shares > 0 and delta < 0 and delta would flip to short, we need to sell existing longs first
                if current_shares > 0 and delta < 0 and desired_shares < 0:
                    # sell longs to zero
                    sell_qty = min(abs(delta), current_shares)
                    sell_price = exec_price
                    sell_value = sell_qty * sell_price
                    sell_fee = transaction_cost * sell_value
                    cash += sell_value - sell_fee
                    # consume long lots FIFO
                    remaining = sell_qty
                    while remaining > 0 and lots[t]:
                        lot = lots[t][0]
                        if lot["side"] != "long":
                            break
                        take = min(remaining, lot["shares"])
                        realized = RealizedTrade(
                            ticker=t, open_date=lot["open_date"], close_date=current_date, side="long",
                            open_price=lot["price"], close_price=sell_price, shares=take,
                            pnl_amt=(sell_price - lot["price"]) * take - (transaction_cost * (sell_price * take)),
                            pnl_pct=(sell_price / lot["price"] - 1.0),
                            fees=(transaction_cost * (sell_price * take)),
                            reason="rebalance_close_long"
                        )
                        realized_trades.append(realized)
                        lot["shares"] -= take
                        remaining -= take
                        if lot["shares"] == 0:
                            lots[t].pop(0)
                    positions[t]["shares"] -= sell_qty
                    if positions[t]["shares"] == 0:
                        positions[t]["avg_price"] = None

                    equity_after_trade = cash + sum(
                        positions[x]["shares"] * (
                            closes[x].dropna().iloc[-1] if not closes[x].dropna().empty else 0.0
                        )
                        for x in tickers
                    )
                    record_trade(current_date, t, "SELL", sell_price, sell_qty, equity_before_trade,
                                 equity_after_trade, sell_fee, 0.0, "rebalance_close_long")
                    # update delta
                    delta = desired_shares - positions[t]["shares"]
                    if delta == 0:
                        continue
                # Now delta is same-sign as current holdings (either purely buy more long, sell some long, open short, or add to short)
                if delta > 0:
                    # buy to increase long (or open new long)
                    buy_qty = delta
                    buy_value = buy_qty * exec_price
                    buy_fee = transaction_cost * buy_value
                    # ensure we respect cash and leverage
                    # quick leverage check: compute prospective gross exposure if we execute this trade
                    # but to keep it simple and deterministic, assume rebalances were scaled earlier to meet leverage
                    if buy_value + buy_fee > cash + 1e-9:
                        # scale down to affordable
                        affordable_qty = int(math.floor((cash - buy_fee) / exec_price)) if exec_price>0 else 0
                        if affordable_qty <= 0:
                            continue
                        buy_qty = min(buy_qty, affordable_qty)
                        buy_value = buy_qty * exec_price
                        buy_fee = transaction_cost * buy_value

                    # execute buy
                    cash -= buy_value + buy_fee
                    # update lots and average price (avg price weighted)
                    prev_shares = positions[t]["shares"]
                    prev_avg = positions[t]["avg_price"]
                    new_shares = prev_shares + buy_qty
                    if prev_shares > 0 and prev_avg is not None:
                        positions[t]["avg_price"] = (prev_shares * prev_avg + buy_qty * exec_price) / new_shares
                    else:
                        positions[t]["avg_price"] = exec_price
                    positions[t]["shares"] = new_shares
                    lots[t].append({"shares": buy_qty, "price": exec_price, "open_date": current_date, "side": "long"})
                    sector_exposure[positions[t]["sector"]] += buy_value
                    equity_after_trade = cash + sum(
                        positions[x]["shares"] * (
                            closes[x].dropna().iloc[-1] if not closes[x].dropna().empty else 0.0
                        )
                        for x in tickers
                    )
                    record_trade(current_date, t, "BUY", exec_price, buy_qty, equity_before_trade,
                                 equity_after_trade, buy_fee, 0.0, "rebalance_buy")
                else:
                    # delta < 0 => sell (could be to reduce long or open/increase short)
                    sell_qty = abs(delta)
                    sell_price = exec_price
                    sell_value = sell_qty * sell_price
                    sell_fee = transaction_cost * sell_value
                    # if we have existing longs, sell them first
                    if positions[t]["shares"] > 0:
                        qty_to_sell = min(sell_qty, positions[t]["shares"])
                        cash += qty_to_sell * sell_price - (transaction_cost * (qty_to_sell * sell_price))
                        remaining = qty_to_sell
                        while remaining > 0 and lots[t]:
                            lot = lots[t][0]
                            if lot["side"] != "long":
                                break
                            take = min(remaining, lot["shares"])
                            realized = RealizedTrade(
                                ticker=t, open_date=lot["open_date"], close_date=current_date, side="long",
                                open_price=lot["price"], close_price=sell_price, shares=take,
                                pnl_amt=(sell_price - lot["price"]) * take - (transaction_cost * (sell_price * take)),
                                pnl_pct=(sell_price / lot["price"] - 1.0),
                                fees=(transaction_cost * (sell_price * take)),
                                reason="rebalance_sell_long"
                            )
                            realized_trades.append(realized)
                            lot["shares"] -= take
                            remaining -= take
                            if lot["shares"] == 0:
                                lots[t].pop(0)
                        positions[t]["shares"] -= qty_to_sell
                        if positions[t]["shares"] == 0:
                            positions[t]["avg_price"] = None
                        equity_after_trade = cash + sum(
                            positions[x]["shares"] * (
                                closes[x].dropna().iloc[-1] if not closes[x].dropna().empty else 0.0
                            )
                            for x in tickers
                        )
                        record_trade(current_date, t, "SELL", sell_price, qty_to_sell, equity_before_trade,
                                     equity_after_trade, transaction_cost * (qty_to_sell * sell_price), 0.0, "rebalance_sell")
                        sell_qty -= qty_to_sell
                    # if sell_qty still > 0, that means we need to open/increase a short position
                    if sell_qty > 0:
                        short_qty = sell_qty
                        short_price = sell_price
                        short_value = short_qty * short_price
                        short_fee = transaction_cost * short_value
                        # Short sale proceeds increase cash (but leverage will still be enforced)
                        cash += short_value - short_fee
                        # add a short lot
                        positions[t]["shares"] -= short_qty  # more negative
                        # avg_price for short positions: weighted average (price we shorted at)
                        prev_shares = positions[t]["shares"] + short_qty  # previous shares before adding negative
                        prev_avg = positions[t]["avg_price"]
                        # careful with averaging negative quantities; store avg as entry price magnitude
                        if prev_shares < 0 and prev_avg is not None:
                            # previously short; compute weighted avg
                            # prev_shares is negative; convert to abs
                            new_total_short = abs(prev_shares) + short_qty
                            positions[t]["avg_price"] = (abs(prev_shares) * prev_avg + short_qty * short_price) / new_total_short
                        else:
                            positions[t]["avg_price"] = short_price
                        lots[t].append({"shares": short_qty, "price": short_price, "open_date": current_date, "side": "short"})
                        sector_exposure[positions[t]["sector"]] += short_value
                        equity_after_trade = cash + sum(
                            positions[x]["shares"] * (
                                closes[x].dropna().iloc[-1] if not closes[x].dropna().empty else 0.0
                            )
                            for x in tickers
                        )
                        record_trade(current_date, t, "SHORT_SELL", short_price, short_qty, equity_before_trade,
                                     equity_after_trade, short_fee, 0.0, "rebalance_open_short")

            # end per-ticker rebalance
            # After all trades on the rebalance date, enforce leverage again (very conservative):
            _, closes_after = market_values_on_date(i)
            gross = gross_exposure_from_prices(closes_after)
            equity_after_all = cash + sum(
                positions[t]["shares"] * (
                    closes_after[t].dropna().iloc[-1] if not closes_after[t].dropna().empty else 0.0
                )
                for t in tickers
            )
            if isinstance(gross, pd.Series):
              gross = gross.iloc[-1]
            if equity_after_all > 0 and gross > max_total_leverage * equity_after_all + 1e-9:
                # scale down positions proportionally to meet leverage (close some amounts)
                scale = (max_total_leverage * equity_after_all) / gross
                print
                for t in tickers:
                    price = closes_after[t]

                    # if price is a Series, pick last valid value
                    if isinstance(price, pd.Series):
                        price = price.dropna()
                        price = price.iloc[-1] if not price.empty else 0.0

                    current_mv = positions[t]["shares"] * price
                    target_mv = current_mv * scale
                    delta_mv = target_mv - current_mv
                    if abs(delta_mv) < 1e-6:
                        continue
                    # execute a trade to move toward target (simple approach: sell or buy absolute delta in shares)
                    # sell if target_mv smaller (close part of position)
                    if delta_mv < 0:
                        qty = int(math.floor(abs(delta_mv) / (closes_after[t] if closes_after[t] else 1.0)))
                        if qty <= 0:
                            continue
                        exec_price = closes_after[t] * (1.0 - slippage)
                        fee = transaction_cost * (qty * exec_price)
                        # closing longs first, then shorts: treat sign-aware
                        if positions[t]["shares"] > 0:
                            # sell qty
                            cash += qty * exec_price - fee
                            # consume long lots FIFO
                            remaining = qty
                            while remaining > 0 and lots[t]:
                                lot = lots[t][0]
                                if lot["side"] != "long":
                                    break
                                take = min(remaining, lot["shares"])
                                realized = RealizedTrade(
                                    ticker=t, open_date=lot["open_date"], close_date=current_date, side="long",
                                    open_price=lot["price"], close_price=exec_price, shares=take,
                                    pnl_amt=(exec_price - lot["price"]) * take - (transaction_cost * (exec_price * take)),
                                    pnl_pct=(exec_price / lot["price"] - 1.0),
                                    fees=transaction_cost * (exec_price * take),
                                    reason="leverage_scale_down"
                                )
                                realized_trades.append(realized)
                                lot["shares"] -= take
                                remaining -= take
                                if lot["shares"] == 0:
                                    lots[t].pop(0)
                            positions[t]["shares"] -= qty
                            if positions[t]["shares"] == 0:
                                positions[t]["avg_price"] = None
                            total_equity = cash + sum(
                                positions[x]["shares"] * (
                                    closes_after[x].dropna().iloc[-1] if not closes_after[x].dropna().empty else 0.0
                                )
                                for x in tickers
                            )
                            record_trade(current_date, t, "SELL", exec_price, qty, equity_after_all,
                                         total_equity, fee, 0.0, "leverage_scale_down")
                        else:
                            # currently short, to reduce absolute exposure we cover some shorts
                            cover_qty = min(qty, abs(positions[t]["shares"]))
                            exec_price = closes_after[t] * (1.0 + slippage)
                            fee = transaction_cost * (cover_qty * exec_price)
                            cash -= cover_qty * exec_price + fee
                            # consume short lots FIFO
                            remaining = cover_qty
                            while remaining > 0 and lots[t]:
                                lot = lots[t][0]
                                if lot["side"] != "short":
                                    break
                                take = min(remaining, lot["shares"])
                                realized = RealizedTrade(
                                    ticker=t, open_date=lot["open_date"], close_date=current_date, side="short",
                                    open_price=lot["price"], close_price=exec_price, shares=take,
                                    pnl_amt=(lot["price"] - exec_price) * take - (transaction_cost * (exec_price * take)),
                                    pnl_pct=(lot["price"] / exec_price - 1.0) if exec_price!=0 else 0.0,
                                    fees=transaction_cost * (exec_price * take),
                                    reason="leverage_scale_down"
                                )
                                realized_trades.append(realized)
                                lot["shares"] -= take
                                remaining -= take
                                if lot["shares"] == 0:
                                    lots[t].pop(0)
                            positions[t]["shares"] += cover_qty
                            if positions[t]["shares"] == 0:
                                positions[t]["avg_price"] = None
                            total_equity = cash + sum(
                                positions[x]["shares"] * (
                                    closes_after[x].dropna().iloc[-1] if not closes_after[x].dropna().empty else 0.0
                                )
                                for x in tickers
                            )

                            record_trade(current_date, t, "BUY_TO_COVER", exec_price, cover_qty, equity_after_all,
                                         total_equity, fee, 0.0, "leverage_scale_down")

            # end leverage enforcement

        # 3) End-of-day equity snapshot (mark-to-market)
        _, closes_eod = market_values_on_date(i)
        total_mv = sum(
            positions[t]["shares"] * (
                closes_eod[t].dropna().iloc[-1] if not closes_eod[t].dropna().empty else 0.0
            )
            for t in tickers
        )
        equity = cash + total_mv
        equity_curve.append((current_date, equity))

    # -------------------------
    # finalize outputs
    # -------------------------

    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date").sort_index()
    equity_series = equity_df["equity"]

    # trade_stats derived from realized_trades

    total_trades = len(realized_trades)
    wins = [rt for rt in realized_trades if rt.pnl_amt > 0]
    losses = [rt for rt in realized_trades if rt.pnl_amt <= 0]
    gross_profit = sum(rt.pnl_amt for rt in realized_trades if rt.pnl_amt > 0)
    gross_loss = -sum(rt.pnl_amt for rt in realized_trades if rt.pnl_amt < 0)
    win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
    avg_profit = (np.mean([rt.pnl_amt for rt in wins]) if wins else 0.0)
    avg_loss = (np.mean([rt.pnl_amt for rt in losses]) if losses else 0.0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    turnover = None
    # compute turnover as sum of absolute traded value / average equity (approx)

    if len(trade_log) > 0:
        avg_equity = equity_series.mean() if len(equity_series)>0 else initial_capital
        sum_traded_value = sum(t["value"] for t in trade_log)
        turnover = sum_traded_value / avg_equity if avg_equity>0 else np.nan

    trade_stats = {
        "total_closed_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit_amt": avg_profit,
        "avg_loss_amt": avg_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "turnover": turnover,
        "num_trade_records": len(trade_log)
    }

    # final positions summary
    final_positions = {}
    _, closes_final = market_values_on_date(len(dates_index)-1)
    final_equity = equity_series.iloc[-1] if len(equity_series)>0 else cash

    for t in tickers:
        price = closes_final.get(t, np.nan)
        # ensure price is scalar
        if isinstance(price, pd.Series):
            price = price.dropna()
            price = price.iloc[-1] if not price.empty else 0.0
        mv = positions[t]["shares"] * price

        final_positions[t] = {
            "shares": int(positions[t]["shares"]),
            "avg_price": float(positions[t]["avg_price"]) if positions[t]["avg_price"] is not None else None,
            "market_value": float(mv),
            "weight": float(mv / final_equity) if final_equity != 0 else 0.0,
            "sector": positions[t]["sector"]
        }

    # Convert trade_log entries datetimes to strings for serialization
    for rec in trade_log:
        if isinstance(rec["date"], (pd.Timestamp,)):
            rec["date"] = rec["date"].isoformat()

    # Convert realized_trades to dicts in a human-readable list (optional)
    realized_trades_list = [rt._asdict() for rt in realized_trades]

    return {
        "equity_curve": equity_series,
        "trade_stats": trade_stats,
        "final_positions": final_positions,
        "trade_log": trade_log,
        "realized_trades": realized_trades_list
    }


def compute_performance_metrics(equity_curve, risk_free_rate=0.0):
    """
    equity_curve: pandas Series of portfolio equity values over time.
    risk_free_rate: Annual risk-free rate for Sharpe calculation (default 0).
    Returns a dict with:
        'cagr', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio'
    """
    if not isinstance(equity_curve, pd.Series):
        raise ValueError("equity_curve must be a pandas Series")

    if len(equity_curve) < 2:
        return {
            'cagr': 0,
            'annual_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }

    # Calculate returns
    daily_returns = equity_curve.pct_change().dropna()

    # CAGR
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 0
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else 0

    # Annual Volatility
    annual_volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_daily_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = (
        excess_daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0 else 0
    )

    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = -cagr / max_drawdown if max_drawdown < 0 else np.nan

    return {
        'cagr': cagr,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'calmar_ratio': calmar_ratio
    }
