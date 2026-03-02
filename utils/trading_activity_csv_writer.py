import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TradingActivityCsvWriter:
    """
    Creates a per-run CSV file and appends rows to it inside your trading loop.

    Directory layout:
      logs/
        trading_activity/
          YY-MM-DD/
            trading_activity_YY_MM_DD_HH_MM_SS.csv
    """
    _base_dir: Path
    _csv_path: Path | None = None

    def _ensure_directory_creation(self) -> Path:
        if self._csv_path is not None:
            return self._csv_path

        current_datetime: datetime = datetime.now()

        date_directory_name: str = current_datetime.strftime("%Y-%m-%d")
        file_name: str = current_datetime.strftime("trading_activity_%Y_%m_%d_%H_%M_%S.csv")

        logs_directory: Path = self._base_dir / "logs" / "trading_activity" / date_directory_name
        logs_directory.mkdir(parents=True, exist_ok=True)

        self._csv_path = logs_directory / file_name

        with self._csv_path.open(mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestep", "Timestamp", "Portfolio Equity", "Portfolio Cash Available"])

        return self._csv_path

    def append_row_to_csv(self, *, timestep: int, timestamp: datetime, portfolio_equity: float,
                          portfolio_cash_available: float) -> None:
        csv_path: Path = self._ensure_directory_creation()

        with csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    timestamp.isoformat(),
                    f"{portfolio_equity:.2f}",
                    f"{portfolio_cash_available:.2f}",
                ]
            )
