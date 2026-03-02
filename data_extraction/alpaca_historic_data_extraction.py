import calendar
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from tqdm import tqdm

from config.config import settings
from logger.logger import AppLogger
from utils.constants import Constants


class AlpacaHistoricDataExtraction:

    def __init__(self) -> None:
        self._api_key_random: str = settings.api_key_random
        self._api_secret_key_random: str = settings.api_secret_key_random
        self._eastern_timezone: ZoneInfo = ZoneInfo("America/New_York")
        self._export_director_path: Path = Path("historical_stock_data/")
        self._stock_historical_data_client: StockHistoricalDataClient = StockHistoricalDataClient(self._api_key_random,
                                                                                                  self._api_secret_key_random)
        self.logger = AppLogger.get_logger(self.__class__.__name__)

    def export_historical_stock_data(self, year_of_data_to_collect: int = 2025) -> None:
        try:

            for ticker_symbol in tqdm(Constants.TICKER_SYMBOL_LIST, desc="Extracting Historical Stock Data"):

                file_name_str: str = ""
                stock_dataframe_list: list[pd.DataFrame] = []

                for month_num in tqdm(range(1, 13), desc="Extracting Specific Stock Data"):
                    days_in_month_num: int = self._get_days_in_month(year=year_of_data_to_collect, month=month_num)

                    start_datetime: datetime = datetime(
                        year=year_of_data_to_collect,
                        month=month_num,
                        day=1,
                        hour=0,
                        minute=0,
                        second=0,
                        tzinfo=self._eastern_timezone
                    )

                    end_datetime: datetime = datetime(
                        year=year_of_data_to_collect,
                        month=month_num,
                        day=days_in_month_num,
                        hour=23,
                        minute=59,
                        second=59,
                        tzinfo=self._eastern_timezone
                    )

                    stock_bars_request: StockBarsRequest = StockBarsRequest(
                        timeframe=TimeFrame.Minute,
                        symbol_or_symbols=ticker_symbol,
                        start=start_datetime,
                        end=end_datetime
                    )

                    bars_set: BarSet = self._stock_historical_data_client.get_stock_bars(stock_bars_request)

                    self._clean_stock_dataframe(bars_set=bars_set, stock_dataframe_list=stock_dataframe_list,
                                                ticker_symbol=ticker_symbol)

                    file_name_str = self._get_file_name_str(ticker_symbol=ticker_symbol)

                result_dataframe: pd.DataFrame = pd.concat(stock_dataframe_list)

                export_file_path: Path = self._get_export_file_path(file_name_str=file_name_str)

                self.logger.info(f"Exporting data to: {export_file_path}")

                result_dataframe.to_csv(path_or_buf=export_file_path)

                self.logger.info(f"Successfully exported data to: {export_file_path}")

        except Exception as e:
            self.logger.error(f"Exception Thrown: {e}")

    def _get_export_file_path(self, file_name_str: str) -> Path:

        director_path: Path = self._export_director_path

        if not director_path.exists():
            director_path.mkdir(parents=True)
            self.logger.info(f"Creating directory: {director_path}")

        export_file_path: Path = Path(director_path / file_name_str)

        return export_file_path

    def _clean_stock_dataframe(self, bars_set: BarSet, stock_dataframe_list: list[pd.DataFrame],
                               ticker_symbol: str) -> None:

        ticker_symbol_unique_label_dict: dict[str, int] = self._get_ticker_symbol_unique_label_dict()

        stock_dataframe: pd.DataFrame = bars_set.df.reset_index()[
            ["symbol", "timestamp", "open", "close", "high", "low", "volume"]]

        unique_ticker_symbol_id: int = ticker_symbol_unique_label_dict[ticker_symbol]

        stock_dataframe['symbol_id'] = np.where(stock_dataframe['symbol'].str.contains(ticker_symbol),
                                                unique_ticker_symbol_id, 9999999)

        stock_dataframe["timestamp"] = pd.to_datetime(stock_dataframe["timestamp"], utc=True)
        stock_dataframe["timestamp"] = stock_dataframe["timestamp"].dt.tz_convert("America/New_York")
        stock_dataframe = stock_dataframe.set_index("timestamp")

        stock_dataframe = stock_dataframe.between_time(start_time="09:30", end_time="16:00")
        stock_dataframe = stock_dataframe.reset_index()[
            ["symbol", "symbol_id", "timestamp", "open", "close", "high", "low", "volume"]]

        stock_dataframe_list.append(stock_dataframe)

    def _get_ticker_symbol_unique_label_dict(self) -> dict[str, int]:
        ticker_symbol_dict: dict[str, int] = {}
        ticker_symbol_list: list[str] = Constants.TICKER_SYMBOL_LIST

        for ticker_str in ticker_symbol_list:
            ticker_hash_value: int = zlib.adler32(ticker_str.encode('utf-8'))

            ticker_symbol_dict[ticker_str] = ticker_hash_value

        return ticker_symbol_dict

    @staticmethod
    def _get_file_name_str(ticker_symbol: str) -> str:
        start_datetime_str: str = "2025" + "_" + "01" + "_" + "01"
        end_datetime_str: str = "2025" + "_" + "12" + "_" + "31"

        return ticker_symbol + "_" + start_datetime_str + "_" + end_datetime_str + ".csv"

    @staticmethod
    def _get_days_in_month(year, month):
        num_days_int: int = calendar.monthrange(year, month)[1]
        return num_days_int
