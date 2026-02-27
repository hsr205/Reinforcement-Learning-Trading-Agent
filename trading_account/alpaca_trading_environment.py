import asyncio
import queue
import random
from asyncio import Task
from collections import deque
from datetime import datetime, time
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import requests
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest
from requests.models import Response
from torch import Tensor

from config.config import settings
from logger.logger import AppLogger
from trading_account.historical_stock_data_obj import HistoricalStockDataObj
from utils.constants import Constants


class AlpacaTradingEnvironment:
    ObsType: TypeVar = TypeVar("ObsType")

    def __init__(self) -> None:
        self._api_key: str = settings.api_key
        self._bar_queue: queue.Queue[dict] = queue.Queue()
        self._bar_history: deque[dict] = deque(maxlen=5000)
        self._latest_bar_dict: dict[str, Any] | None = None
        self._api_secret_key: str = settings.api_secret_key
        self._action_space: list[str] = Constants.ACTIONS_LIST
        self._first_bar_event: asyncio.Event = asyncio.Event()
        self._close_of_market_time: time = time(16, 0)
        self._trading_client: TradingClient = TradingClient(self._api_key, self._api_secret_key, paper=True)

        self.logger = AppLogger.get_logger(self.__class__.__name__)

    async def _handle_bar(self, data) -> None:
        data_bar: Bar = data
        bar_dict: dict = data.model_dump()

        # store latest + history if you want
        self._latest_bar_dict = bar_dict
        self._bar_history.append(bar_dict)

        # thread-safe handoff to your RL consumer loop
        self._bar_queue.put(bar_dict)

    # TODO: Implement this method
    def step(self, action_str: str) -> tuple[ObsType, Tensor, bool, bool, dict[str, Any]]:
        pass

        # return observation, reward_tensor, terminated, truncated, _

    async def initialize_trading_environment(self) -> None:

        data_stream: StockDataStream = StockDataStream(api_key=self._api_key, secret_key=self._api_secret_key)
        self.logger.info("=" * 100)
        self.logger.info("Initializing Trading Environment")

        try:

            # TODO: In order to test, minimize the amount of equities being traded
            data_stream.subscribe_bars(self._handle_bar, "AAPL", "TSLA", "META", "AMZN", "MSFT", "NVDA", "GOOGL")

            stream_task: Task = asyncio.create_task(asyncio.to_thread(data_stream.run))

            current_time_step: int = 1

            while True:

                portfolio_list: list = self._trading_client.get_all_positions()

                self._populate_portfolio(portfolio_list=portfolio_list)

                state_data_dict: dict = await asyncio.to_thread(self._bar_queue.get) | self.get_market_features_dict(
                    portfolio_list=portfolio_list)

                self._execute_action_to_balance_portfolio(state_data_dict=state_data_dict)

                random_action_order_side_action: OrderSide | str = self._get_random_order_side_action()

                if random_action_order_side_action != "HOLD":

                    self.logger.info(f"Timestep Num: {current_time_step}")

                    portfolio_cash:float = state_data_dict.get("cash")
                    portfolio_equity:float = state_data_dict.get("equity")

                    self.logger.info(f"Portfolio Equity: {portfolio_equity:,2f} -> Cash On Hand: ${portfolio_cash:,.2f}")
                    self.logger.info("=" * 150)


                else:
                    self.logger.info(f"Action Selected -> {random_action_order_side_action}")
                    continue

                random_quantity_dict: dict[
                    str, tuple[int, float, OrderSide]] = self._get_random_quantity_per_symbol_dict(
                    state_data_dict=state_data_dict,
                    portfolio_list=portfolio_list)

                self.execute_random_action(random_quantity_dict=random_quantity_dict)

                timestamp_utc: datetime = state_data_dict.get("timestamp")
                current_time_est: time = timestamp_utc.astimezone(ZoneInfo("America/New_York")).time()

                if current_time_est >= self._close_of_market_time:
                    self.logger.info(f"Broken at timestep: {current_time_step}")
                    self.logger.info("=" * 200)
                    break

                # <-- THIS is where you build your (s, a, r, s') transition
                # e.g.:
                # s  = build_state(latest_bar_dict, market_features_dict, ...)
                # a  = agent.act(s)
                # r  = reward(...)
                # s2 = next_state(...)
                current_time_step += 1

            await stream_task  # (won't reach in infinite loop unless you break)

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

    def _get_random_order_side_action(self) -> OrderSide | str:
        return random.choice(self._action_space)

    def _get_random_quantity_per_symbol_dict(self, state_data_dict: dict[str, Any], portfolio_list: list, ) -> dict[
        str, tuple[int, float, OrderSide]]:

        current_cash_t: float = float(state_data_dict.get("cash", 0.0))

        random_quantity_dict: dict[str, tuple[int, float, OrderSide]] = {}

        for stock_data_collection in portfolio_list:

            ticker_symbol_str: str = stock_data_collection.symbol
            individual_stock_data_dict: dict[str, int | float] = state_data_dict.get(stock_data_collection.symbol, "")

            current_stock_quantity: int = int(individual_stock_data_dict.get("qty_available", 0))
            current_stock_price: float = float(individual_stock_data_dict.get("current_price", 0.0))

            order_side: OrderSide = random.choice(Constants.ORDER_SIDE_ACTIONS_LIST)

            is_buy_side: bool = order_side == OrderSide.BUY
            is_sell_side: bool = order_side == OrderSide.SELL

            if is_sell_side:
                max_valid_quantity: int = current_stock_quantity

                if max_valid_quantity <= 0:
                    random_quantity_dict[ticker_symbol_str] = (0, current_stock_price, order_side)
                    continue

                random_quantity: int = random.randint(1, max_valid_quantity)
                random_quantity_dict[ticker_symbol_str] = (random_quantity, current_stock_price, order_side)
                continue

            if is_buy_side:
                max_valid_quantity = int(current_cash_t // current_stock_price)

                if max_valid_quantity <= 0:
                    random_quantity_dict[ticker_symbol_str] = (0, current_stock_price, order_side)
                    transaction_cost: float = current_stock_price * max_valid_quantity
                    self.logger.warning(
                        f"Invalid {order_side.name} of {current_stock_quantity:,} share(s) of {ticker_symbol_str}:"
                    )
                    self.logger.warning(
                        f"Cash On Hand -> ${current_cash_t:,.2f}, Current Stock Price -> ${current_stock_price:,.2f}, Transaction Cost -> ${transaction_cost:,.2f}")
                    continue

                random_quantity = random.randint(1, max_valid_quantity)

                # (Optional) sanity affordability check using the *actual* random_quantity
                transaction_cost: float = current_stock_price * random_quantity
                if transaction_cost > current_cash_t:
                    random_quantity_dict[ticker_symbol_str] = (0, current_stock_price, order_side)
                    self.logger.warning(
                        f"Invalid {order_side.name} of {random_quantity:,} share(s) 0f {ticker_symbol_str}:"
                    )
                    self.logger.warning(
                        f"Quantity -> {random_quantity}, Transaction Cost ->${transaction_cost:,.2f} exceeds Cash On Hand ->${current_cash_t:,.2f}")
                    continue

                random_quantity_dict[ticker_symbol_str] = (random_quantity, current_stock_price, order_side)

        return random_quantity_dict

    def execute_random_action(self, random_quantity_dict: dict[str, tuple[int, float, OrderSide]]) -> None:

        try:

            for ticker_symbol_str, ticker_symbol_tuple in random_quantity_dict.items():

                stock_quantity: int = ticker_symbol_tuple[0]
                stock_purchase_price: float = float(ticker_symbol_tuple[1])
                stock_action: OrderSide = ticker_symbol_tuple[2]

                if stock_quantity <= 0:
                    continue

                market_order_request: MarketOrderRequest = MarketOrderRequest(
                    symbol=ticker_symbol_str,
                    qty=stock_quantity,
                    side=stock_action,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )

                market_order: Order = self._trading_client.submit_order(
                    order_data=market_order_request
                )

                market_order_qty_int: int = int(market_order.qty)

                self.logger.info(
                    f"Successfully {market_order.side.name} {market_order_qty_int} share(s) of {market_order.symbol} @ ${stock_purchase_price:,.2f} for ${stock_purchase_price * market_order_qty_int:,.2f}")

            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _populate_portfolio(self, portfolio_list: list) -> None:

        try:
            if not portfolio_list:

                self.logger.info("Populating Empty Portfolio")
                self.logger.info("=" * 100)

                for ticker_symbol_str in Constants.TICKER_SYMBOL_LIST:
                    market_order_request: MarketOrderRequest = MarketOrderRequest(
                        symbol=ticker_symbol_str,
                        qty=1,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    market_order: Order = self._trading_client.submit_order(
                        order_data=market_order_request
                    )

                    self.logger.info(
                        f"Successful {market_order.side.name} {market_order.qty} share(s) of {market_order.symbol}")

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _execute_action_to_balance_portfolio(self, state_data_dict: dict) -> None:

        try:

            for key, value in state_data_dict.items():

                if isinstance(value, dict):

                    ticker_quantity: int = int(value.get("quantity"))
                    ticker_qty_available: int = int(value.get("qty_available"))

                    if ticker_quantity < 0 or ticker_qty_available < 0:
                        quantity_to_buy = abs(ticker_quantity)

                        market_order_request: MarketOrderRequest = MarketOrderRequest(
                            symbol=key,
                            qty=quantity_to_buy,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            time_in_force=TimeInForce.DAY
                        )

                        market_order: Order = self._trading_client.submit_order(
                            order_data=market_order_request
                        )

                        self.logger.info(
                            f"Successful {market_order.side.name} {market_order.qty} share(s) of {market_order.symbol}")

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _get_ticker_data_dict(self, portfolio_list: list) -> dict[str, dict[str, Any]]:

        ticker_data_dict: dict[str, dict[str, Any]] = {}

        for index, position in enumerate(portfolio_list):
            data_dict: dict[str, float] = {
                "quantity": position.qty,
                "qty_available": position.qty_available,
                "cost_basis": position.cost_basis,
                "current_price": position.current_price,
                "change_today": position.change_today,
            }

            ticker_data_dict[position.symbol] = data_dict

        return ticker_data_dict

    def _get_account_data_dict(self) -> dict[str, float]:
        try:

            headers_dict: dict[str, str] = {
                "accept": "application/json",
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._api_secret_key
            }

            response: Response = requests.get(Constants.ALPACA_ACCOUNT_URL, headers=headers_dict)
            response_dict: dict[str, Any] = response.json()

            cash_float: float = float(response_dict["cash"])
            equity_float: float = float(response_dict["equity"])

            account_data_dict: dict[str, float] = {
                "cash": cash_float,
                "equity": equity_float,
            }

            return account_data_dict


        except Exception as e:
            self.logger.error(f"Exception thrown: {e}")



    def get_market_features_dict(self, portfolio_list: list) -> dict[str, Any]:

        account_data_dict: dict[str, Any] = self._get_account_data_dict()
        ticker_data_dict: dict[str, dict[str, Any]] = self._get_ticker_data_dict(portfolio_list=portfolio_list)
        market_features_dict: dict[str, Any] = account_data_dict | ticker_data_dict

        return market_features_dict

    def get_individual_stock_data_list(self) -> list[HistoricalStockDataObj]:

        file_path: Path = self._get_data_file_path()

        historical_stock_data_list: list[HistoricalStockDataObj] = []

        with open(file=file_path, mode="r") as data_file:

            next(data_file)

            for observation_str in data_file:

                observation_str = observation_str.strip()

                if not observation_str:
                    continue

                values: list[str] = observation_str.split(",")

                historical_stock_obj: HistoricalStockDataObj = HistoricalStockDataObj(
                    observation_num=int(values[0]),
                    ticker_symbol_id=int(values[2]),
                    timestamp=datetime.fromisoformat(values[3]),
                    open=float(values[4]),
                    close=float(values[5]),
                    high=float(values[6]),
                    low=float(values[7]),
                    volume=float(values[8])
                )

                historical_stock_data_list.append(historical_stock_obj)

        return historical_stock_data_list

    def _get_data_file_path(self) -> Path:

        data_directory_path: Path = Path("historical_stock_data/")

        file_path: Path = Path()

        for element in data_directory_path.iterdir():
            if element.is_file() and self._ticker_symbol_str in str(element):
                file_path = element
                break

        return file_path
