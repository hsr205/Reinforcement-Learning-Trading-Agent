from logging import Logger

from logger.logger import AppLogger
from trading_account.alpaca_trading_environment import AlpacaTradingEnvironment


def main() -> int:
    logger: Logger = AppLogger().get_logger(__name__)

    try:

        # # TODO: Change the value of `num_observations`
        # deep_q_neural_network: DQN = DQN(num_observations=0, num_actions=2)
        # deep_q_neural_network.get_target_network_state_dict()

        # alpaca_historic_data:AlpacaHistoricDataExtraction = AlpacaHistoricDataExtraction()
        # alpaca_historic_data.export_historical_stock_data(year_of_data_to_collect=2025)

        alpaca_trading_env: AlpacaTradingEnvironment = AlpacaTradingEnvironment(ticker_symbol_str="AAPL")
        # alpaca_trading_env.execute_action(ticker_str="AMZN",quantity=1,action_type=OrderSide.BUY)
        # alpaca_trading_env.get_market_features_dict()
        # logger.info("=" * 100)
        # alpaca_trading_env.get_individual_stock_data_list()

        alpaca_trading_env.get_state_dict()


    except Exception as e:
        logger.info(f"Exception Thrown: {e}")

    return 0


if __name__ == "__main__":
    main()
