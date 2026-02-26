from typing import Any

from alpaca.trading.enums import OrderSide


class Constants:
    LOGGER_COLOR_RESET: str = "\033[0m"
    LOGGER_COLOR_WHITE: str = "\033[60m"
    LOGGER_COLOR_ORANGE: str = "\033[33m"
    LOGGER_COLOR_DARK_RED: str = "\033[31m"

    ALPACA_ACCOUNT_URL: str = "https://paper-api.alpaca.markets/v2/account"

    ACTIONS_LIST: list[OrderSide | str] = [OrderSide.BUY, OrderSide.SELL, "HOLD"]

    TICKER_SYMBOL_LIST: list[str] = ["TSLA", "AAPL", "META", "AMZN", "MSFT", "NVDA", "GOOGL"]
