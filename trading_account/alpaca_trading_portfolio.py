import asyncio
import queue
from collections import deque
from datetime import time, datetime
from pathlib import Path
from typing import Any
from typing import Union

import numpy as np
import torch
from alpaca.common import RawData
from alpaca.data import StockLatestQuoteRequest, Quote, StockBarsRequest, TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading import Position
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest
from torch import Tensor

from config.config import settings
from logger.logger import AppLogger
from utils.constants import Constants
from utils.trading_activity_csv_writer import TradingActivityCsvWriter


class AlpacaTradingPortfolio:

    def __init__(self, device, trading_client: TradingClient) -> None:
        self._device = device
        self._cost_coefficient: float = 0.001
        self._base_directory: Path = Path.cwd()
        self._api_key_ppo: str = settings.api_key_ppo
        self._bar_queue: queue.Queue[dict] = queue.Queue()
        self._bar_history: deque[dict] = deque(maxlen=5000)
        self._latest_bar_dict: dict[str, Any] | None = None
        self._trading_client: TradingClient = trading_client
        self._action_space: list[str] = Constants.ACTIONS_LIST
        self._first_bar_event: asyncio.Event = asyncio.Event()
        self._close_of_market_time: time = time(16, 0)
        self._api_secret_key_ppo: str = settings.api_secret_key_ppo
        self._trading_csv_writer: TradingActivityCsvWriter = TradingActivityCsvWriter(_base_dir=self._base_directory)

        self._historical_trading_client: StockHistoricalDataClient = StockHistoricalDataClient(
            api_key=self._api_key_ppo,
            secret_key=self._api_secret_key_ppo)
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    def get_portfolio_weights_tensor(self, per_ticker_array: np.ndarray) -> Tensor:

        portfolio_weights_list: list[float] = [x[0] for x in per_ticker_array]

        portfolio_weights_tensor: Tensor = torch.tensor(data=portfolio_weights_list, dtype=torch.float32)

        return portfolio_weights_tensor

    def get_observation_tensor(self, all_positions_list: list[Position],
                               account_dict: dict[str, float]) -> Tensor:

        matrix_list: list[list[float]] = self._get_matrix_list(all_positions_list_t=all_positions_list,
                                                               account_dict_t=account_dict)

        matrix_tensor: Tensor = torch.tensor(matrix_list)

        observation_tensor: Tensor = matrix_tensor.view(-1)

        return observation_tensor

    # def get_observation_tensor(self, all_positions_list: list[Position],
    #                                    account_dict: dict[str, float]) -> tuple[Tensor, np.ndarray]:
    #
    #     matrix_list: list[list[float]] = self._get_matrix_list(all_positions_list_t=all_positions_list,
    #                                                            account_dict_t=account_dict)
    #
    #     per_ticker_array: np.ndarray = np.array(matrix_list, dtype=np.float32)
    #
    #     matrix_tensor: Tensor = torch.tensor(matrix_list)
    #
    #     flattened_tensor = matrix_tensor.view(-1)
    #
    #     observation_tensor = torch.flatten(flattened_tensor)
    #
    #     return observation_tensor, per_ticker_array

    def get_account_dict(self) -> dict[str, Any]:

        account_dict: dict[str, Any] = self._trading_client.get_account().model_dump()

        result_dict: dict[str, Any] = {

            "cash": float(account_dict.get("cash", 0.0)),
            "equity": float(account_dict.get("equity", 0.0)),
            "buying_power": float(account_dict.get("buying_power", 0.0)),
            "portfolio_value": float(account_dict.get("portfolio_value", 0.0)),
            "daytrading_buying_power": float(account_dict.get("daytrading_buying_power", 0.0))

        }

        return result_dict

    def balance_empty_portfolio(self) -> None:

        account_dict: dict[str, float] = self.get_account_dict()
        all_positions_list: list[Position] = self._trading_client.get_all_positions()

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

    def _get_matrix_list(self, all_positions_list_t: list[Position], account_dict_t: dict[str, float]) -> list[
        list[float]]:

        matrix_list: list[list[float]] = []
        ticker_features_list: list[str] = Constants.TICKER_FEATURES_LIST
        positions_dict: dict[int, dict[str, float]] = self._get_positions_dict(
            all_positions_list=all_positions_list_t, account_dict=account_dict_t)

        for ticker_id_num in Constants.TICKER_SYMBOL_TO_ID.values():
            ticker_dict: dict[str, float] = positions_dict.get(ticker_id_num, {})

            matrix_row_list: list[float] = []

            for feature_str in ticker_features_list:
                feature_value: float | Any = ticker_dict.get(feature_str, 0.0)
                matrix_row_list.append(float(feature_value))

            matrix_list.append(matrix_row_list)

        return matrix_list

    def _get_positions_dict(self, all_positions_list: list[Position], account_dict: dict[str, float]) -> dict[
        int, dict[str, float]]:

        positions_dict: dict[int, dict[str, float]] = {}

        self._populate_missing_ticker_entries(all_positions_list=all_positions_list, account_dict=account_dict,
                                              positions_dict=positions_dict)

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
            cost_basis_to_portfolio_value: float = float(position_obj.cost_basis) / portfolio_value
            unrealized_pl_to_portfolio_value: float = float(position_obj.unrealized_pl) / portfolio_value

            ticker_dict: dict[str, float] = {
                "qty_available": qty_available,
                "portfolio_value": portfolio_value,
                "portfolio_weight": portfolio_weight,
                "cash_to_portfolio_value": cash_to_portfolio_value,
                "cost_basis_to_portfolio_value": cost_basis_to_portfolio_value,
                "buying_power_to_portfolio_value": buying_power_to_portfolio_value,
                "unrealized_pl_to_portfolio_value": unrealized_pl_to_portfolio_value,
                "change_today": float(position_obj.change_today),
            }

            positions_dict[ticker_symbol_id] = ticker_dict

        return positions_dict

    def _populate_missing_ticker_entries(self, all_positions_list: list[Position], account_dict: dict[str, float],
                                         positions_dict: dict[int, dict[str, float]]) -> None:

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

                ticker_dict: dict[str, float] = {
                    "qty_available": 0.0,
                    "position_value": 0.0,
                    "portfolio_weight": 0.0,
                    "portfolio_value": portfolio_value,
                    "cost_basis_to_portfolio_value": 0.0,
                    "unrealized_pl_to_portfolio_value": 0.0,
                    "cash_to_portfolio_value": cash_to_portfolio_value,
                    "buying_power_to_portfolio_value": buying_power_to_portfolio_value,
                    "change_today": 0.0,
                }

                positions_dict[ticker_symbol_id] = ticker_dict

    def _get_positions_str_list(self, all_positions_list: list[Position]) -> list[str]:
        positions_str_list: list[str] = [x.symbol for x in all_positions_list]
        return positions_str_list
