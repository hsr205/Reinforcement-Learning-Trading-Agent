import asyncio
import queue
from asyncio import Task
from collections import deque
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from tensordict import TensorDict, TensorDictBase
from torch import multiprocessing, Tensor
from torchrl.data import Composite, UnboundedContinuous, Bounded
from torchrl.envs import EnvBase

from config.config import settings
from logger.logger import AppLogger
from models.ppo_config import PPOConfig
from trading_account.alpaca_trading_portfolio import AlpacaTradingPortfolio
from utils.constants import Constants
from utils.trading_activity_csv_writer import TradingActivityCsvWriter


# TODO: Use unsloth
class AlpacaTradingEnvironmentPPO(EnvBase):

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()

        self._step_count: int = 0

        self._config: PPOConfig = config
        self._dtype = torch.float32
        self._device: torch.device = self._get_processing_device()

        self._observation_dim: int = config.observation_dimension
        self._action_dimension: int = config.action_dimension

        self._current_weights_tensor: Tensor = torch.zeros(self._action_dimension, dtype=self._dtype,
                                                           device=self._device)
        self._current_observation_tensor: Tensor = torch.zeros(self._observation_dim, dtype=self._dtype,
                                                               device=self._device)
        # Composite - observation is raw tensor wih a dict like structure
        self.observation_spec = Composite(
            observation=UnboundedContinuous(
                shape=torch.Size([self._observation_dim]),
                dtype=self._dtype,
                device=self._device,
            ),
            shape=torch.Size([]),
        )

        # UnboundedContinuous - a continues value between (-∞, ∞)
        self.action_spec = UnboundedContinuous(
            shape=torch.Size([self._action_dimension]),
            dtype=self._dtype,
            device=self._device,
        )

        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([1]),
            dtype=self._dtype,
            device=self._device,
        )

        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self._device,
            ),
            terminated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self._device,
            ),
            truncated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self._device,
            ),
            shape=torch.Size([]),
        )

        self._base_directory: Path = Path.cwd()
        self._api_key_ppo: str = settings.api_key_ppo
        self._bar_queue: queue.Queue[dict] = queue.Queue()
        self._bar_history: deque[dict] = deque(maxlen=5000)
        self._latest_bar_dict: dict[str, Any] | None = None

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

    def _project_action_to_target_weights(self, action_tensor: Tensor) -> Tensor:
        weights_tensor: Tensor = torch.softmax(action_tensor, dim=-1)
        return weights_tensor

    async def _handle_bar(self, data) -> None:
        bar_dict: dict = data.model_dump()

        self._latest_bar_dict = bar_dict
        self._bar_history.append(bar_dict)

        self._bar_queue.put(bar_dict)

    def _reset(
            self,
            tensordict: TensorDictBase | None = None,
            **kwargs
    ) -> TensorDictBase:

        return TensorDict()

    def _set_seed(self, seed: int | None = None) -> None:
        if seed is None:
            return

        torch.manual_seed(seed)
        np.random.seed(seed)

    async def _step(self, tensordict) -> TensorDict:
        action_tensor: Tensor = tensordict["action"].to(self._device)

        data_stream: StockDataStream = StockDataStream(api_key=self._api_key_ppo,
                                                       secret_key=self._api_secret_key_ppo)

        try:

            data_stream.subscribe_bars(self._handle_bar, *Constants.TICKER_SYMBOL_LIST)
            stream_task: Task = asyncio.create_task(asyncio.to_thread(data_stream.run))

            # self._alpaca_trading_account.balance_empty_portfolio()

            account_dict: dict[str, float] = self._alpaca_trading_account.get_account_dict()
            all_positions_list = self._trading_client.get_all_positions()

            current_portfolio_value_tensor = torch.tensor(
                data=account_dict.get("portfolio_value", 0.0),
                dtype=self._dtype,
                device=self._device,
            )

            current_weights_tensor = self._current_weights_tensor.clone()

            target_weights_tensor = self._project_action_to_target_weights(action_tensor=action_tensor)

            new_portfolio_value_tensor = self._simulate_portfolio_value_transition(
                current_portfolio_value_tensor=current_portfolio_value_tensor,
                current_weights_tensor=current_weights_tensor,
                target_weights_tensor=target_weights_tensor,
            )

            self._current_weights_tensor = target_weights_tensor.detach()
            self._step_count += 1

            self._current_observation_tensor = self._alpaca_trading_account.get_observation_tensor(
                all_positions_list=all_positions_list, account_dict=account_dict)

            reward_tensor: Tensor = self._get_reward_tensor(
                current_portfolio_value_tensor=current_portfolio_value_tensor,
                new_portfolio_value_tensor=new_portfolio_value_tensor,
                portfolio_weights_tensor_t=current_weights_tensor,
                portfolio_weights_tensor_t_1=target_weights_tensor,
            )

            current_time_est: time = datetime.now().astimezone(ZoneInfo("America/New_York")).time()
            is_terminal: bool = current_time_est >= self._close_of_market_time
            is_terminal_tensor: Tensor = torch.tensor([is_terminal], dtype=torch.bool, device=self._device)

            await stream_task

            return TensorDict(
                {
                    "observation": self._current_observation_tensor,
                    "reward": reward_tensor.reshape(1),
                    "done": is_terminal_tensor,
                    "terminated": is_terminal_tensor,
                    "truncated": torch.zeros(1, dtype=torch.bool, device=self._device),
                },
                batch_size=[],
                device=self._device,
            )

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

    def _execute_trades(self, action) -> None:
        pass

    def _simulate_portfolio_value_transition(self, current_portfolio_value_tensor: Tensor,
                                             current_weights_tensor: Tensor, target_weights_tensor: Tensor) -> Tensor:
        # TODO
        """
        Replace this with your real trading mechanics:
        - execute trades
        - apply fills / slippage / commissions if desired
        - update holdings
        - advance market by one step
        - compute new portfolio value
        """

        pass

    def _get_reward_tensor(self, current_portfolio_value_tensor: Tensor, new_portfolio_value_tensor: Tensor,
                           portfolio_weights_tensor_t: Tensor, portfolio_weights_tensor_t_1: Tensor) -> Tensor:

        turnover_value: torch.Tensor = torch.sum(
            torch.abs(portfolio_weights_tensor_t_1 - portfolio_weights_tensor_t),
            dtype=torch.float32
        )

        safe_denominator: torch.Tensor = torch.clamp(current_portfolio_value_tensor, min=1e-12)
        portfolio_delta_log_return: torch.Tensor = torch.log(new_portfolio_value_tensor / safe_denominator)
        reward_tensor: torch.Tensor = portfolio_delta_log_return - self._cost_coefficient_tensor * turnover_value

        return reward_tensor

    # def _get_reward_tensor(self, account_dict_t: dict[str, float], account_dict_t_1: dict[str, float],
    #                        per_ticker_array_t: np.ndarray, per_ticker_array_t_1: np.ndarray) -> Tensor:
    #
    #     current_portfolio_value_tensor: torch.Tensor = torch.tensor(
    #         data=account_dict_t.get("portfolio_value", 0.0),
    #         device=self._device,
    #         dtype=torch.float32
    #     )
    #
    #     new_portfolio_value_tensor: torch.Tensor = torch.tensor(
    #         data=account_dict_t_1.get("portfolio_value", 0.0),
    #         device=self._device,
    #         dtype=torch.float32
    #     )
    #
    #     portfolio_weights_tensor_t: Tensor = self._alpaca_trading_account.get_portfolio_weights_tensor(
    #         per_ticker_array=per_ticker_array_t)
    #
    #     portfolio_weights_tensor_t_1: Tensor = self._alpaca_trading_account.get_portfolio_weights_tensor(
    #         per_ticker_array=per_ticker_array_t_1)
    #
    #     turnover_value: torch.Tensor = torch.sum(
    #         torch.abs(portfolio_weights_tensor_t_1 - portfolio_weights_tensor_t),
    #         dtype=torch.float32
    #     )
    #
    #     safe_denominator: torch.Tensor = torch.clamp(current_portfolio_value_tensor, min=1e-12)
    #     portfolio_delta_log_return: torch.Tensor = torch.log(new_portfolio_value_tensor / safe_denominator)
    #     reward_tensor: torch.Tensor = portfolio_delta_log_return - self._cost_coefficient_tensor * turnover_value
    #
    #     return reward_tensor

    def _get_processing_device(self) -> torch.device:

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
