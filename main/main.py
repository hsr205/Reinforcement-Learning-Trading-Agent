import asyncio
from logging import Logger

from logger.logger import AppLogger
from trading_account.alpaca_trading_environment import AlpacaTradingEnvironment


async def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    try:

        alpaca_trading_env: AlpacaTradingEnvironment = AlpacaTradingEnvironment()

        await alpaca_trading_env.initialize_trading_environment()


    except Exception as e:
        logger.info(f"Exception Thrown: {e}")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
