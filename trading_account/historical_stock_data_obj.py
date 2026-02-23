from dataclasses import dataclass
from datetime import datetime


@dataclass
class HistoricalStockDataObj:
    observation_num: int
    ticker_symbol_id: int
    timestamp: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float
