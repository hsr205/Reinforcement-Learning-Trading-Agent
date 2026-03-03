import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from alpaca.trading import Position

from logger.logger import AppLogger
from utils.constants import Constants


@dataclass
class TradingActivityCsvWriter:
    _base_dir: Path
    _csv_path: Path | None = None
    _logger = AppLogger.get_logger(__name__)

    def _ensure_directory_creation(self, logs_directory_path: Path) -> Path:
        if self._csv_path is not None:
            return self._csv_path

        current_datetime: datetime = datetime.now()

        file_name: str = current_datetime.strftime("trading_activity_%Y_%m_%d_%H_%M_%S.csv")

        logs_directory_path.mkdir(parents=True, exist_ok=True)

        self._csv_path = logs_directory_path / file_name

        with self._csv_path.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            columns_list: list[str] = Constants.CSV_OUTPUT_COLUMNS_LIST
            writer.writerow(columns_list)

        return self._csv_path

    def _get_positions_dict(self, all_positions_list: list[Position]) -> dict[str, float]:

        positions_dict: dict[str, float] = {}

        for position_obj in all_positions_list:

            ticker_symbol_str: str = position_obj.symbol

            if ticker_symbol_str not in positions_dict:
                positions_dict[ticker_symbol_str] = float(position_obj.qty)

            else:

                self._logger.error(f"Duplicate instance of {ticker_symbol_str} ticker in positions list")

        return positions_dict

    def append_row_to_csv(self, *, logs_directory_path: Path, timestep: int, current_datetime: datetime,
                          portfolio_equity: float,
                          portfolio_cash_available: float, all_positions_list: list[Position]) -> None:
        csv_path: Path = self._ensure_directory_creation(logs_directory_path=logs_directory_path)

        positions_dict: dict[str, float] = self._get_positions_dict(all_positions_list=all_positions_list)

        apple_stock_quantity: float = positions_dict.get("AAPL", {})
        amazon_stock_quantity: float = positions_dict.get("AMZN", {})
        google_stock_quantity: float = positions_dict.get("GOOGL", {})
        meta_stock_quantity: float = positions_dict.get("META", {})
        nvidia_stock_quantity: float = positions_dict.get("NVDA", {})
        microsoft_stock_quantity: float = positions_dict.get("MSFT", {})
        tesla_stock_quantity: float = positions_dict.get("TSLA", {})

        with csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    current_datetime.isoformat(),
                    f"{portfolio_equity:.2f}",
                    f"{portfolio_cash_available:.2f}",
                    f"{apple_stock_quantity}",
                    f"{amazon_stock_quantity}",
                    f"{google_stock_quantity}",
                    f"{meta_stock_quantity}",
                    f"{nvidia_stock_quantity}",
                    f"{microsoft_stock_quantity}",
                    f"{tesla_stock_quantity}",
                ]
            )
