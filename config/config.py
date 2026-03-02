from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key_random: str = Field(..., description="API key for connection to Random_Trading_Account on Alpaca Markets")
    api_secret_key_random: str = Field(..., description="API secret key for connection to Random_Trading_Account on Alpaca Markets")

    api_key_ppo: str = Field(..., description="API key for connection to PPO_Trading_Account on Alpaca Markets")
    api_secret_key_ppo: str = Field(..., description="API secret key for connection to PPO_Trading_Account on Alpaca Markets")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
