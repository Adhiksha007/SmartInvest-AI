from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

# Position sizing config (same as before)
@dataclass
class SizeConfig:
    mode: str = "fixed_notional"  # "fixed_notional", "percent_of_equity", "risk_per_trade"
    fixed_notional: float = 1000.0
    percent_of_equity: float = 0.1  # 10% of equity
    risk_per_trade: float = 0.01  # 1% of equity
    min_qty: int = 1

# Execution config with SL/TP percents
@dataclass
class ExecConfig:
    transaction_cost: float = 0.0  # fraction e.g. 0.0005
    slippage: float = 0.0
    size_config: SizeConfig = field(default_factory=SizeConfig)
    stop_loss_pct: Optional[float] = None  # e.g., 0.03 (3%)
    take_profit_pct: Optional[float] = None  # e.g., 0.06 (6%)
    max_exposure_per_ticker: Optional[float] = None  # absolute currency exposure e.g., 20000
    max_exposure_per_sector: Optional[float] = None  # absolute currency exposure per sector
    max_total_leverage: Optional[float] = None  # e.g., 1.0 = no leverage, >1 allowed

class PaperTrader:
    def __init__(self, cash: float = 10000.0):
        self.cash = cash
        self.positions = {}  # ticker -> qty
        self.trade_history = []  # list of dicts

    def buy(self, ticker: str, price: float, qty: int):
        cost = price * qty
        if cost <= self.cash:
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + qty
            self.trade_history.append({"side": "BUY", "ticker": ticker, "price": price, "qty": qty})
            return True
        return False

    def sell(self, ticker: str, price: float, qty: int):
        if self.positions.get(ticker, 0) >= qty:
            self.positions[ticker] -= qty
            self.cash += price * qty
            self.trade_history.append({"side": "SELL", "ticker": ticker, "price": price, "qty": qty})
            return True
        return False

    def value(self, prices: Dict[str, float]) -> float:
        pv = sum(prices.get(t, 0) * q for t, q in self.positions.items())
        return self.cash + pv


# -----------------------------
# Paper adapter (lightweight)
# -----------------------------
class PaperBrokerAdapter:
    def __init__(self, paper_trader):
        self.trader = paper_trader

    def place_order(self, ticker, side, qty, price, order_type="market", **kwargs):
        if side.upper() == "BUY":
            return self.trader.buy(ticker, price, qty)
        else:
            return self.trader.sell(ticker, price, qty)

    def get_cash(self):
        return self.trader.cash

    def get_positions(self):
        return dict(self.trader.positions)

    def close(self):
        return

# -----------------------------
# Zerodha Kite Adapter (sandbox pattern)
# -----------------------------
# Requires: pip install kiteconnect
# Notes: Use Kite Connect test credentials. For live trading the flow includes user login and request tokens.
try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

class ZerodhaAdapter:
    """
    Example Zerodha/Kite adapter.
    IMPORTANT: For real trading you must implement the login/refresh-token flow.
    This example shows usage patterns and sandbox placeholders.
    """
    def __init__(self, api_key: str, api_secret: str, access_token: str = None, is_sandbox: bool = True):
        if KiteConnect is None:
            raise ImportError("kiteconnect package not installed. pip install kiteconnect")
        self.is_sandbox = is_sandbox
        self.kite = KiteConnect(api_key=api_key)
        # If you have a stored access_token, set it
        if access_token:
            self.kite.set_access_token(access_token)
        # For sandbox behavior, we won't call live order endpoints unless you change a flag
        self._positions = {}
        self._cash = 100000.0  # sandbox default paper cash

    def place_order(self, ticker, side, qty, price, order_type="market", **kwargs):
        # Example pattern: market/limit order wrapper
        if self.is_sandbox:
            # emulate order fill
            if side.upper() == "BUY":
                cost = qty * price
                if cost <= self._cash:
                    self._cash -= cost
                    self._positions[ticker] = self._positions.get(ticker, 0) + qty
                    return True
                return False
            else:
                if self._positions.get(ticker, 0) >= qty:
                    self._positions[ticker] -= qty
                    self._cash += qty * price
                    return True
                return False
        else:
            # Example real API call — caution: must handle request tokens and exceptions
            mp = self.kite.place_order(
                tradingsymbol=ticker,
                exchange="NSE",  # or BSE depending on instrument
                transaction_type="BUY" if side.upper()=="BUY" else "SELL",
                quantity=qty,
                order_type="MARKET" if order_type == "market" else "LIMIT",
                product="MIS",  # or CNC for delivery
                price=price if order_type != "market" else None
            )
            return bool(mp)

    def get_cash(self):
        if self.is_sandbox:
            return self._cash
        # In real mode, use kite.margins() or /portfolio endpoints
        return self._cash

    def get_positions(self):
        if self.is_sandbox:
            return dict(self._positions)
        # Real: kite.positions() parsing required
        return {}

    def close(self):
        return

# -----------------------------
# Alpaca Adapter (paper)
# -----------------------------
# Requires: pip install alpaca-trade-api
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

class AlpacaAdapter:
    """
    Alpaca adapter using alpaca-trade-api. Use paper API endpoint for sandbox.
    """
    def __init__(self, key_id: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        if tradeapi is None:
            raise ImportError("alpaca-trade-api not installed. pip install alpaca-trade-api")
        self.api = tradeapi.REST(key_id, secret_key, base_url=base_url)
        # On init fetch account to set cash
        try:
            acct = self.api.get_account()
            self._cash = float(acct.cash)
        except Exception:
            # fallback to default if offline
            self._cash = 100000.0
        self._positions_cache = {}

    def place_order(self, ticker, side, qty, price, order_type="market", **kwargs):
        try:
            if order_type == "market":
                order = self.api.submit_order(symbol=ticker, qty=qty, side=side.lower(), type='market', time_in_force='day')
            else:
                order = self.api.submit_order(symbol=ticker, qty=qty, side=side.lower(), type='limit', limit_price=price, time_in_force='day')
            # optimistic: assume filled for paper demo
            # update simple local caches
            if side.upper() == "BUY":
                self._positions_cache[ticker] = self._positions_cache.get(ticker, 0) + qty
                self._cash -= qty * (price or 0)
            else:
                self._positions_cache[ticker] = max(0, self._positions_cache.get(ticker, 0) - qty)
                self._cash += qty * (price or 0)
            return True
        except Exception as e:
            print("Alpaca order error:", e)
            return False

    def get_cash(self):
        try:
            acct = self.api.get_account()
            return float(acct.cash)
        except Exception:
            return self._cash

    def get_positions(self):
        try:
            pos = self.api.list_positions()
            return {p.symbol: int(float(p.qty)) for p in pos}
        except Exception:
            return dict(self._positions_cache)

    def close(self):
        return

# -----------------------------
# Interactive Brokers (ib_insync) adapter
# -----------------------------
# Requires: pip install ib_insync
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
except Exception:
    IB = None

class IBAdapter:
    """
    IB adapter using ib_insync. Use IB Gateway/TWS paper account.
    """
    def __init__(self, host='127.0.0.1', port=7497, clientId=1, paper: bool = True):
        if IB is None:
            raise ImportError("ib_insync not installed. pip install ib_insync")
        self.ib = IB()
        self.ib.connect(host, port, clientId=clientId)
        # basic caches
        self._positions = {}
        self._cash = 100000.0
        try:
            # update cache with account values
            for pos in self.ib.positions():
                sym = pos.contract.symbol
                self._positions[sym] = self._positions.get(sym, 0) + pos.position
            # account summary may be used to get cash
            # summar = self.ib.accountSummary()
        except Exception:
            pass

    def place_order(self, ticker, side, qty, price, order_type="market", **kwargs):
        try:
            contract = Stock(ticker, 'SMART', 'USD')
            if order_type == "market":
                order = MarketOrder(side, qty)
            else:
                order = LimitOrder(side, qty, price)
            trade = self.ib.placeOrder(contract, order)
            # For demo, assume immediate fill in paper mode after small wait
            self.ib.sleep(0.5)
            # update local caches (optimistic)
            if side.upper() == "BUY":
                self._positions[ticker] = self._positions.get(ticker, 0) + qty
                self._cash -= qty * (price or 0)
            else:
                self._positions[ticker] = max(0, self._positions.get(ticker, 0) - qty)
                self._cash += qty * (price or 0)
            return True
        except Exception as e:
            print("IB order error:", e)
            return False

    def get_cash(self):
        return self._cash

    def get_positions(self):
        return dict(self._positions)

    def close(self):
        try:
            self.ib.disconnect()
        except Exception:
            pass

