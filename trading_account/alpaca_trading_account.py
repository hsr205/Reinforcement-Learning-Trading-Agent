from typing import Any

import requests
from requests.models import Response

from config.config import settings
from logger.logger import AppLogger
from utils.constants import Constants


class AlpacaTradingAccount:

    def __init__(self) -> None:
        self._api_key_random: str = settings.api_key_random
        self._api_secret_key_random: str = settings.api_secret_key_random
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    def get_market_features_dict(self, portfolio_list: list) -> dict[str, Any]:

        account_data_dict: dict[str, Any] = self._get_account_data_dict()
        ticker_data_dict: dict[str, dict[str, Any]] = self._get_ticker_data_dict(portfolio_list=portfolio_list)
        market_features_dict: dict[str, Any] = account_data_dict | ticker_data_dict

        return market_features_dict

    def _get_account_data_dict(self) -> dict[str, float]:
        try:

            headers_dict: dict[str, str] = {
                "accept": "application/json",
                "APCA-API-KEY-ID": self._api_key_random,
                "APCA-API-SECRET-KEY": self._api_secret_key_random
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
