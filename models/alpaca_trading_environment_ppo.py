import asyncio
import queue
from asyncio import Task
from collections import deque
from datetime import datetime, time
from pathlib import Path
from typing import Any
from typing import TypeVar
from zoneinfo import ZoneInfo

import numpy as np
import torch
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from torch import multiprocessing, device, Tensor

from config.config import settings
from logger.logger import AppLogger
from trading_account.alpaca_trading_portfolio import AlpacaTradingPortfolio
from utils.constants import Constants
from utils.trading_activity_csv_writer import TradingActivityCsvWriter


# TODO: Use unsloth
class AlpacaTradingEnvironmentPPO:

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
        self._cost_coefficient_tensor: torch.Tensor = torch.tensor(data=0.001,
                                                                   device=self._device,
                                                                   dtype=torch.float32)
        self._trading_client: TradingClient = TradingClient(api_key=self._api_key_ppo,
                                                            secret_key=self._api_secret_key_ppo, paper=True)

        self._alpaca_trading_account: AlpacaTradingPortfolio = AlpacaTradingPortfolio(device=self._device,
                                                                                      trading_client=self._trading_client)

        self.logger = AppLogger.get_logger(self.__class__.__name__)

    async def _handle_bar(self, data) -> None:
        bar_dict: dict = data.model_dump()

        self._latest_bar_dict = bar_dict
        self._bar_history.append(bar_dict)

        self._bar_queue.put(bar_dict)

    async def step(self, action) -> tuple[Tensor, float, bool]:

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

            self._alpaca_trading_account.balance_empty_portfolio()

            account_dict_t: dict[str, float] = self._alpaca_trading_account.get_account_dict()
            all_positions_list_t = self._trading_client.get_all_positions()

            observation_tensor_t, per_ticker_array_t = self._alpaca_trading_account.get_ticker_feature_collections(
                all_positions_list=all_positions_list_t, account_dict=account_dict_t)

            self._execute_trades(action=action)

            account_dict_t_1 = self._alpaca_trading_account.get_account_dict()
            all_positions_list_t_1 = self._trading_client.get_all_positions()

            observation_tensor_t_1, per_ticker_array_t_1 = self._alpaca_trading_account.get_ticker_feature_collections(
                all_positions_list=all_positions_list_t_1, account_dict=account_dict_t_1)

            reward: float = self._get_reward(account_dict_t=account_dict_t,
                                             account_dict_t_1=account_dict_t_1,
                                             per_ticker_array_t=per_ticker_array_t,
                                             per_ticker_array_t_1=per_ticker_array_t_1)

            self.logger.info(f"reward = {reward}")

            await stream_task

            current_time_step += 1

            return observation_tensor_t_1, reward, is_terminal

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

    def _execute_trades(self, action) -> None:
        pass

    def _get_reward(self, account_dict_t: dict[str, float], account_dict_t_1: dict[str, float],
                    per_ticker_array_t: np.ndarray, per_ticker_array_t_1: np.ndarray) -> float:

        current_portfolio_value_tensor: torch.Tensor = torch.tensor(
            data=account_dict_t.get("portfolio_value", 0.0),
            device=self._device,
            dtype=torch.float32
        )

        new_portfolio_value_tensor: torch.Tensor = torch.tensor(
            data=account_dict_t_1.get("portfolio_value", 0.0),
            device=self._device,
            dtype=torch.float32
        )

        portfolio_weights_tensor_t: Tensor = self._alpaca_trading_account.get_portfolio_weights_tensor(
            per_ticker_array=per_ticker_array_t)

        portfolio_weights_tensor_t_1: Tensor = self._alpaca_trading_account.get_portfolio_weights_tensor(
            per_ticker_array=per_ticker_array_t_1)

        turnover_value: torch.Tensor = torch.sum(
            torch.abs(portfolio_weights_tensor_t_1 - portfolio_weights_tensor_t),
            dtype=torch.float32
        )

        safe_denominator: torch.Tensor = torch.clamp(current_portfolio_value_tensor, min=1e-12)
        portfolio_delta_log_return: torch.Tensor = torch.log(new_portfolio_value_tensor / safe_denominator)
        reward_tensor: torch.Tensor = portfolio_delta_log_return - self._cost_coefficient_tensor * turnover_value

        reward: float = float(reward_tensor.item())

        return reward

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
