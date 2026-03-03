import asyncio
import math
import queue
import random
from asyncio import Task
from collections import deque
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from alpaca.trading import Position
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest

from config.config import settings
from logger.logger import AppLogger
from trading_account.alpaca_trading_account import AlpacaTradingAccount
from utils.constants import Constants
from utils.trading_activity_csv_writer import TradingActivityCsvWriter


# TODO: Consider moving some of the methods in this class to a helper class
class AlpacaTradingEnvironmentRandomPolicy:
    alpaca_trading_account: AlpacaTradingAccount = AlpacaTradingAccount()

    def __init__(self) -> None:
        self._base_directory: Path = Path.cwd()
        self._api_key_random: str = settings.api_key_random
        self._bar_queue: queue.Queue[dict] = queue.Queue()
        self._bar_history: deque[dict] = deque(maxlen=5000)
        self._latest_bar_dict: dict[str, Any] | None = None
        self._action_space: list[str] = Constants.ACTIONS_LIST
        self._first_bar_event: asyncio.Event = asyncio.Event()
        self._close_of_market_time: time = time(16, 0)
        self._logs_directory_path: Path = self._get_logs_directory_path()
        self._api_secret_key_random: str = settings.api_secret_key_random
        self._trading_csv_writer: TradingActivityCsvWriter = TradingActivityCsvWriter(_base_dir=self._base_directory)
        self._trading_client: TradingClient = TradingClient(api_key=self._api_key_random,
                                                            secret_key=self._api_secret_key_random, paper=True)
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    # TODO: Move the following to a helper class
    async def _handle_bar(self, data) -> None:
        data_bar: Bar = data
        bar_dict: dict = data.model_dump()

        # store latest + history if you want
        self._latest_bar_dict = bar_dict
        self._bar_history.append(bar_dict)

        # thread-safe handoff to your RL consumer loop
        self._bar_queue.put(bar_dict)

    async def initialize_trading_environment_random_policy(self) -> None:

        data_stream: StockDataStream = StockDataStream(api_key=self._api_key_random,
                                                       secret_key=self._api_secret_key_random)
        self.logger.info("=" * 100)
        self.logger.info("Initializing Trading Environment")

        try:

            data_stream.subscribe_bars(self._handle_bar, *Constants.TICKER_SYMBOL_LIST)
            stream_task: Task = asyncio.create_task(asyncio.to_thread(data_stream.run))

            current_time_step: int = 1

            while True:

                account_dict: dict[str, Any] = self._get_account_dict()
                all_positions_list: list[Position] = self._trading_client.get_all_positions()

                self._populate_portfolio(all_positions_list=all_positions_list)

                all_positions_list = self._trading_client.get_all_positions()

                state_data_dict: dict = await asyncio.to_thread(self._bar_queue.get)

                random_action: OrderSide | str = self._get_random_order_side_action()

                if random_action != "HOLD":
                    portfolio_cash: float = account_dict.get("cash", 0.0)
                    portfolio_equity: float = account_dict.get("equity", 0.0)
                    current_datetime: datetime = datetime.now().astimezone(ZoneInfo("America/New_York"))

                    self.logger.info(
                        f"Timestep: {current_time_step} -> Timestamp: {current_datetime.time()} -> Portfolio Equity: {portfolio_equity:,.2f} -> Portfolio Cash Available: ${portfolio_cash:,.2f}")
                    self.logger.info("=" * 150)

                    self._trading_csv_writer.append_row_to_csv(
                        logs_directory_path=self._logs_directory_path,
                        timestep=current_time_step,
                        current_datetime=current_datetime,
                        portfolio_equity=portfolio_equity,
                        portfolio_cash_available=portfolio_cash,
                        all_positions_list=all_positions_list
                    )

                else:
                    self.logger.info(f"Action Selected -> {random_action}")
                    continue

                random_quantity_dict: dict[
                    str, tuple[int, float, OrderSide]] = self._get_random_quantity_per_symbol_dict(
                    account_dict=account_dict,
                    all_positions_list=all_positions_list)

                self.execute_random_action(random_quantity_dict=random_quantity_dict)

                current_time_est: time = datetime.now().astimezone(ZoneInfo("America/New_York")).time()

                if current_time_est >= self._close_of_market_time:
                    self.logger.info(f"Broken at timestep: {current_time_step}")
                    self.logger.info("=" * 200)
                    break

                current_time_step += 1

            await stream_task

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

    def _get_random_quantity_per_symbol_dict(self, account_dict: dict[str, Any], all_positions_list: list, ) -> dict[
        str, tuple[int, float, OrderSide]]:

        current_cash_t: float = float(account_dict.get("cash", 0.0))
        random_quantity_dict: dict[str, tuple[int, float, OrderSide]] = {}

        for stock_position in all_positions_list:
            order_side: OrderSide = random.choice(Constants.ORDER_SIDE_ACTIONS_LIST)
            is_buy_side: bool = order_side == OrderSide.BUY
            is_sell_side: bool = order_side == OrderSide.SELL

            if is_sell_side:
                self._is_sell_side_order(order_side=order_side, random_quantity_dict=random_quantity_dict,
                                         stock_position=stock_position)
            elif is_buy_side:
                self._is_buy_side_order(order_side=order_side, random_quantity_dict=random_quantity_dict,
                                        stock_position=stock_position, current_cash_t=current_cash_t)

        return random_quantity_dict

    def _is_sell_side_order(self, order_side: OrderSide, random_quantity_dict: dict[str, tuple[int, float, OrderSide]],
                            stock_position: Position) -> None:

        ticker_symbol_str: str = stock_position.symbol
        stock_quantity: int = int(stock_position.qty_available)
        stock_price: float = float(stock_position.current_price)

        max_valid_quantity: int = stock_quantity

        if max_valid_quantity <= 0:
            random_quantity_dict[ticker_symbol_str] = (0, stock_price, order_side)
            return

        random_quantity: int = math.ceil(random.randint(1, max_valid_quantity) / 2)
        random_quantity_dict[ticker_symbol_str] = (random_quantity, stock_price, order_side)
        return

    def _is_buy_side_order(self, order_side: OrderSide, random_quantity_dict: dict[str, tuple[int, float, OrderSide]],
                           stock_position: Position, current_cash_t: float) -> None:

        ticker_symbol_str: str = stock_position.symbol
        stock_price: float = float(stock_position.current_price)

        max_valid_quantity = int(current_cash_t // stock_price)

        if self._is_max_quantity_less_or_equal_to_zero(order_side=order_side,
                                                       random_quantity_dict=random_quantity_dict,
                                                       stock_position=stock_position,
                                                       current_cash_t=current_cash_t,
                                                       max_valid_quantity=max_valid_quantity):
            return

        random_quantity = math.ceil(random.randint(1, max_valid_quantity) / 2)

        if self._is_transaction_cost_greater_than_cash_available(order_side=order_side,
                                                                 random_quantity_dict=random_quantity_dict,
                                                                 stock_position=stock_position,
                                                                 current_cash_t=current_cash_t,
                                                                 random_quantity=random_quantity):
            return

        random_quantity_dict[ticker_symbol_str] = (random_quantity, stock_price, order_side)

    def _is_transaction_cost_greater_than_cash_available(self, order_side: OrderSide, random_quantity_dict: dict[
        str, tuple[int, float, OrderSide]], stock_position: Position, current_cash_t: float,
                                                         random_quantity: int) -> bool:

        ticker_symbol_str: str = stock_position.symbol
        stock_price: float = float(stock_position.current_price)

        transaction_cost: float = stock_price * random_quantity
        if transaction_cost > current_cash_t:
            random_quantity_dict[ticker_symbol_str] = (0, stock_price, order_side)
            self.logger.warning(
                f"Invalid {order_side.name} of {random_quantity:,} share(s) of {ticker_symbol_str}:"
            )
            self.logger.warning(
                f"Quantity -> {random_quantity}, Transaction Cost ->${transaction_cost:,.2f} exceeds Cash On Hand ->${current_cash_t:,.2f}")
            return True

        return False

    # TODO: Review the logic see if this can be removed
    def _is_max_quantity_less_or_equal_to_zero(self, order_side: OrderSide, random_quantity_dict: dict[
        str, tuple[int, float, OrderSide]], stock_position: Position, current_cash_t: float,
                                               max_valid_quantity: int) -> bool:

        ticker_symbol_str: str = stock_position.symbol
        stock_quantity: int = int(stock_position.qty_available)
        stock_price: float = float(stock_position.current_price)

        if max_valid_quantity <= 0:
            random_quantity_dict[ticker_symbol_str] = (0, stock_price, order_side)
            transaction_cost: float = stock_price * max_valid_quantity
            self.logger.warning(
                f"Invalid {order_side.name} of {stock_quantity:,} share(s) of {ticker_symbol_str}:"
            )
            self.logger.warning(
                f"Cash On Hand -> ${current_cash_t:,.2f}, Current Stock Price -> ${stock_price:,.2f}, Transaction Cost -> ${transaction_cost:,.2f}")
            return True

        return False

    def execute_random_action(self, random_quantity_dict: dict[str, tuple[int, float, OrderSide]]) -> None:

        try:

            for ticker_symbol_str, ticker_symbol_tuple in random_quantity_dict.items():

                stock_quantity: int = ticker_symbol_tuple[0]
                stock_price: float = float(ticker_symbol_tuple[1])
                stock_action: OrderSide = ticker_symbol_tuple[2]

                if stock_quantity <= 0 and stock_action == OrderSide.SELL:
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
                    f"Successfully {market_order.side.name} {market_order_qty_int} share(s) of {market_order.symbol} @ ${stock_price:,.2f} for ${stock_price * market_order_qty_int:,.2f}")

            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _populate_portfolio(self, all_positions_list: list[Position]) -> None:

        positions_str_list: list[str] = self._get_positions_str_list(all_positions_list=all_positions_list)

        try:

            for ticker_symbol_str in Constants.TICKER_SYMBOL_LIST:

                if ticker_symbol_str not in positions_str_list:
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
                        f"Successfully {market_order.side.name} {market_order.qty} share(s) of {market_order.symbol}")

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _get_account_dict(self) -> dict[str, Any]:

        account_dict: dict[str, Any] = self._trading_client.get_account().model_dump()

        result_dict: dict[str, Any] = {

            "cash": float(account_dict.get("cash", 0.0)),
            "equity": float(account_dict.get("equity", 0.0)),
            "buying_power": float(account_dict.get("buying_power", 0.0)),
            "portfolio_value": float(account_dict.get("portfolio_value", 0.0)),
            "daytrading_buying_power": float(account_dict.get("daytrading_buying_power", 0.0))

        }

        return result_dict

    def _get_logs_directory_path(self) -> Path:
        current_datetime: datetime = datetime.now()
        date_directory_name: str = current_datetime.strftime("%Y-%m-%d")
        logs_directory_path: Path = self._base_directory / "logs" / "random_trading_activity" / date_directory_name
        return logs_directory_path

    def _get_positions_str_list(self, all_positions_list: list[Position]) -> list[str]:
        positions_str_list: list[str] = [x.symbol for x in all_positions_list]
        return positions_str_list

    def _get_random_order_side_action(self) -> OrderSide | str:
        return random.choice(self._action_space)
