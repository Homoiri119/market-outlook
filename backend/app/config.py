from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8")

    # J-Quants authentication
    jquants_mail: str = ""
    jquants_password: str = ""
    jquants_refresh_token: str = ""
    jquants_api_key: str = ""  # paid API key (for sector/stock-level backtest)

    # EDINET
    edinet_api_key: str = ""

    # Discord
    # Preferred: ANALYZE_DISCORD_WEBHOOK_URL; falls back to DISCORD_WEBHOOK_URL.
    analyze_discord_webhook_url: str = ""
    discord_webhook_url: str = ""

    @property
    def effective_discord_webhook(self) -> str:
        return self.analyze_discord_webhook_url or self.discord_webhook_url

    # App
    database_url: str = f"sqlite:///{BASE_DIR / 'app.db'}"
    target_stocks_file: str = str(BASE_DIR / "data" / "target_stocks.json")

    # Analysis
    history_days: int = 365
    macro_signal_threshold: float = 0.001  # 0.1% predicted return threshold for BUY/SELL
    stock_signal_threshold: float = 0.001

    # Morning outlook (pre-open, ~08:00 JST): thresholds on the expected Nikkei open move.
    # Blends the overnight Nikkei-futures gap (forward-looking, primary) with a
    # US-market -> next-day Nikkei regression estimate.
    outlook_strong_threshold: float = 0.008  # >= +0.8% => STRONG_UP (<= -0.8% => STRONG_DOWN)
    outlook_flat_threshold: float = 0.002  # within +/-0.2% => FLAT
    futures_gap_weight: float = 0.7  # weight of the futures gap vs. the regression estimate

    # Scheduler
    daily_run_hour: int = 8
    daily_run_minute: int = 0
    scheduler_timezone: str = "Asia/Tokyo"


settings = Settings()
