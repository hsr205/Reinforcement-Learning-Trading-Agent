import csv
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path


@dataclass
class TradingActivityCsvWriter:

    _base_dir: Path
    _csv_path: Path | None = None

    def _ensure_directory_creation(self, logs_directory_path:Path) -> Path:
        if self._csv_path is not None:
            return self._csv_path

        current_datetime: datetime = datetime.now()

        file_name: str = current_datetime.strftime("trading_activity_%Y_%m_%d_%H_%M_%S.csv")

        logs_directory_path.mkdir(parents=True, exist_ok=True)

        self._csv_path = logs_directory_path / file_name

        with self._csv_path.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestep", "Timestamp", "Portfolio Equity", "Portfolio Cash Available"])

        return self._csv_path

    def append_row_to_csv(self, *, logs_directory_path:Path, timestep: int, current_time: time, portfolio_equity: float,
                          portfolio_cash_available: float) -> None:
        csv_path: Path = self._ensure_directory_creation(logs_directory_path=logs_directory_path)

        with csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    current_time.isoformat(),
                    f"{portfolio_equity:.2f}",
                    f"{portfolio_cash_available:.2f}",
                ]
            )
