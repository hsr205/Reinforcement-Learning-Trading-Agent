import asyncio
from logging import Logger

from logger.logger import AppLogger
from trading_account.alpaca_trading_environment_random_policy import AlpacaTradingEnvironmentRandomPolicy


async def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    try:

        alpaca_trading_env_random_policy: AlpacaTradingEnvironmentRandomPolicy = AlpacaTradingEnvironmentRandomPolicy()

        await alpaca_trading_env_random_policy.initialize_trading_environment_random_policy()


    except Exception as e:
        logger.info(f"Exception Thrown: {e}")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
