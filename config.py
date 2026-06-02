import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DATABASE_PATH = os.getenv("DATABASE_PATH", "trading_intel.db")
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "10"))
MORNING_BRIEF_HOUR = int(os.getenv("MORNING_BRIEF_HOUR", "6"))
MORNING_BRIEF_MINUTE = int(os.getenv("MORNING_BRIEF_MINUTE", "45"))
MORNING_BRIEF_LOOKBACK_HOURS = int(os.getenv("MORNING_BRIEF_LOOKBACK_HOURS", "12"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Interactive Brokers — TWS or IB Gateway must be running locally
IB_HOST      = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT      = int(os.getenv("IB_PORT", "7497"))   # 7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "99"))  # unique per connection; 99 avoids clash with manual TWS sessions
