import asyncio
import queue
from asyncio import Task
from collections import deque
from datetime import datetime, time
from pathlib import Path
from typing import Any
from typing import TypeVar
from typing import Union
from zoneinfo import ZoneInfo

import torch
from alpaca.common import RawData
from alpaca.data import StockLatestQuoteRequest, Quote
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading import Position
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest
from torch import multiprocessing, device

from config.config import settings
from logger.logger import AppLogger
from trading_account.alpaca_trading_account import AlpacaTradingAccount
from utils.constants import Constants
from utils.trading_activity_csv_writer import TradingActivityCsvWriter


# TODO: Use unsloth
class AlpacaTradingEnvironmentPPO:
    ObsType: TypeVar = TypeVar("ObsType")
    alpaca_trading_account: AlpacaTradingAccount = AlpacaTradingAccount()

    def __init__(self) -> None:
        self._base_directory: Path = Path.cwd()
        self._api_key_ppo: str = settings.api_key_ppo
        self._bar_queue: queue.Queue[dict] = queue.Queue()
        self._bar_history: deque[dict] = deque(maxlen=5000)
        self._latest_bar_dict: dict[str, Any] | None = None
        self._device: device = self._get_processing_device()
        self._action_space: list[str] = Constants.ACTIONS_LIST
        self._first_bar_event: asyncio.Event = asyncio.Event()
        self._close_of_market_time: time = time(16, 0)
        self._api_secret_key_ppo: str = settings.api_secret_key_ppo
        self._logs_directory_path: Path = self._get_logs_directory_path()
        self._trading_csv_writer: TradingActivityCsvWriter = TradingActivityCsvWriter(_base_dir=self._base_directory)
        self._trading_client: TradingClient = TradingClient(api_key=self._api_key_ppo,
                                                            secret_key=self._api_secret_key_ppo, paper=True)

        self._historical_trading_client: StockHistoricalDataClient = StockHistoricalDataClient(
            api_key=self._api_key_ppo,
            secret_key=self._api_secret_key_ppo)
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    async def _handle_bar(self, data) -> None:
        bar_dict: dict = data.model_dump()

        # store latest + history if you want
        self._latest_bar_dict = bar_dict
        self._bar_history.append(bar_dict)

        # thread-safe handoff to your RL consumer loop
        self._bar_queue.put(bar_dict)

    async def initialize_trading_environment_ppo(self) -> None:
        pass

    # TODO: Implement this method
    # async def step(self, action_str: str) -> tuple[ObsType, Tensor, bool, bool, dict[str, Any]]:
    async def step(self) -> None:

        data_stream: StockDataStream = StockDataStream(api_key=self._api_key_ppo,
                                                       secret_key=self._api_secret_key_ppo)
        self.logger.info("=" * 100)
        self.logger.info("Initializing Trading Environment")

        try:

            data_stream.subscribe_bars(self._handle_bar, *Constants.TICKER_SYMBOL_LIST)
            stream_task: Task = asyncio.create_task(asyncio.to_thread(data_stream.run))

            current_time_step: int = 1

            current_time_est: time = datetime.now().astimezone(ZoneInfo("America/New_York")).time()
            is_terminal: bool = current_time_est >= self._close_of_market_time

            while True:

                account_dict: dict[str, float] = self._get_account_dict()
                all_positions_list: list[Position] = self._trading_client.get_all_positions()

                self.logger.info("=" * 100)

                self._balance_empty_portfolio(all_positions_list=all_positions_list, account_dict=account_dict)

                account_dict = self._get_account_dict()
                all_positions_list = self._trading_client.get_all_positions()

                positions_list_of_dicts: list[dict[str, float | int]] = self._get_positions_list_of_dicts(
                    all_positions_list=all_positions_list, account_dict=account_dict)

                if is_terminal:
                    self.logger.info(f"Broken at timestep: {current_time_step}")
                    self.logger.info("=" * 200)
                    break

                current_time_step += 1

                exit()

            await stream_task

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

        # return observation, reward_tensor, is_terminal, truncated, _

    def _balance_empty_portfolio(self, all_positions_list: list[Position], account_dict: dict[str, Any]) -> None:

        try:

            if not all_positions_list:

                stock_quotes_request: StockLatestQuoteRequest = StockLatestQuoteRequest(
                    symbol_or_symbols=Constants.TICKER_SYMBOL_LIST
                )

                latest_quotes: Union[
                    dict[str, Quote], RawData] = self._historical_trading_client.get_stock_latest_quote(
                    request_params=stock_quotes_request)

                portfolio_value: float = account_dict.get("portfolio_value", 0.0)
                investment_per_ticker_symbol: float = portfolio_value * 0.01

                for ticker_symbol_str in Constants.TICKER_SYMBOL_LIST:
                    stock_bid_price: float = float(latest_quotes[ticker_symbol_str].bid_price)

                    num_shares: float = investment_per_ticker_symbol / stock_bid_price

                    market_order_request: MarketOrderRequest = MarketOrderRequest(
                        symbol=ticker_symbol_str,
                        qty=num_shares,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    market_order: Order = self._trading_client.submit_order(
                        order_data=market_order_request
                    )

                    self.logger.info(
                        f"Successfully {market_order.side.name} {float(market_order.qty):,.2f} share(s) of {market_order.symbol}")

        except Exception as e:
            self.logger.warning(f"Exception Thrown: {e}")

    def _get_positions_list_of_dicts(self, all_positions_list: list[Position], account_dict: dict[str, float]) -> list[
        dict[str, Any]]:

        positions_list_of_dicts: list[dict[str, float]] = []

        self._populate_missing_ticker_entries(all_positions_list=all_positions_list, account_dict=account_dict,
                                              positions_list_of_dicts=positions_list_of_dicts)

        for position_obj in all_positions_list:
            ticker_symbol_str: str = position_obj.symbol

            ticker_symbol_id: int = Constants.TICKER_SYMBOL_TO_ID.get(ticker_symbol_str, 99_999)

            cash: float = account_dict.get("cash", 0.0)
            buying_power: float = account_dict.get("buying_power", 0.0)
            portfolio_value: float = account_dict.get("portfolio_value", 0.0)

            market_value: float = float(position_obj.market_value)
            qty_available: float = float(position_obj.qty_available)

            cash_to_portfolio_value: float = cash / portfolio_value
            portfolio_weight: float = market_value / portfolio_value
            buying_power_to_portfolio_value: float = buying_power / portfolio_value
            market_value_to_portfolio_value: float = market_value / portfolio_value
            cost_basis_to_portfolio_value: float = float(position_obj.cost_basis) / portfolio_value
            unrealized_pl_to_portfolio_value: float = float(position_obj.unrealized_pl) / portfolio_value

            ticker_dict: dict[str, float | int] = {

                "ticker_symbol_id": ticker_symbol_id,
                "qty_available": qty_available,
                "portfolio_value": portfolio_value,
                "portfolio_weight": portfolio_weight,
                "cash_to_portfolio_value": cash_to_portfolio_value,
                "cost_basis_to_portfolio_value": cost_basis_to_portfolio_value,
                "market_value_to_portfolio_value": market_value_to_portfolio_value,
                "buying_power_to_portfolio_value": buying_power_to_portfolio_value,
                "unrealized_pl_to_portfolio_value": unrealized_pl_to_portfolio_value,

                "change_today": float(position_obj.change_today),
            }

            positions_list_of_dicts.append(ticker_dict)

        for position_obj in positions_list_of_dicts:
            self.logger.info(f"position_obj = {position_obj}")
            self.logger.info("=" * 100)

        return positions_list_of_dicts

    def _populate_missing_ticker_entries(self, all_positions_list: list[Position], account_dict: dict[str, float],
                                         positions_list_of_dicts: list[dict[str, float]]) -> None:

        full_ticker_symbol_list: list[str] = Constants.TICKER_SYMBOL_LIST
        positions_str_list: list[str] = self._get_positions_str_list(all_positions_list=all_positions_list)

        for ticker_symbol_str in full_ticker_symbol_list:

            if ticker_symbol_str not in positions_str_list:
                ticker_symbol_id: int = Constants.TICKER_SYMBOL_TO_ID.get(ticker_symbol_str, 99_999)

                cash: float = account_dict.get("cash", 0.0)
                buying_power: float = account_dict.get("buying_power", 0.0)
                portfolio_value: float = account_dict.get("portfolio_value", 0.0)

                cash_to_portfolio_value: float = cash / portfolio_value
                buying_power_to_portfolio_value: float = buying_power / portfolio_value

                ticker_dict = {
                    "ticker_symbol_id": ticker_symbol_id,
                    "qty_available": 0.0,
                    "position_value": 0.0,
                    "portfolio_weight": 0.0,
                    "portfolio_value": portfolio_value,
                    "cost_basis_to_portfolio_value": 0.0,
                    "market_value_to_portfolio_value": 0.0,
                    "unrealized_pl_to_portfolio_value": 0.0,
                    "cash_to_portfolio_value": cash_to_portfolio_value,
                    "buying_power_to_portfolio_value": buying_power_to_portfolio_value,
                    "change_today": 0.0,
                }

                positions_list_of_dicts.append(ticker_dict)

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

    def _get_processing_device(self) -> device:

        is_fork: bool = multiprocessing.get_start_method() == "fork"

        if torch.cuda.is_available() and not is_fork:
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_logs_directory_path(self) -> Path:
        current_datetime: datetime = datetime.now()
        date_directory_name: str = current_datetime.strftime("%Y-%m-%d")
        logs_directory_path: Path = self._base_directory / "logs" / "ppo_trading_activity" / date_directory_name
        return logs_directory_path

    def _get_positions_str_list(self, all_positions_list: list[Position]) -> list[str]:
        positions_str_list: list[str] = [x.symbol for x in all_positions_list]
        return positions_str_list
