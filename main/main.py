import asyncio
from logging import Logger

from logger.logger import AppLogger
from models.alpaca_trading_environment_ppo import AlpacaTradingEnvironmentPPO
from models.alpaca_trading_environment_random_policy import AlpacaTradingEnvironmentRandomPolicy
from models.alpaca_trading_ppo_neural_network import AlpacaTradingPPONeuralNetwork
from models.ppo_config import PPOConfig


async def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    try:

        alpaca_trading_env_random_policy: AlpacaTradingEnvironmentRandomPolicy = AlpacaTradingEnvironmentRandomPolicy()

        await alpaca_trading_env_random_policy.initialize_trading_environment_random_policy()

        ppo_config: PPOConfig = PPOConfig()
        environment = AlpacaTradingEnvironmentPPO(config=ppo_config)
        alpaca_trading_ppo_neural_network: AlpacaTradingPPONeuralNetwork = AlpacaTradingPPONeuralNetwork(
            env=environment, config=ppo_config)

        alpaca_trading_ppo_neural_network.train_model()


    except Exception as e:
        logger.info(f"Exception Thrown: {e}")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
