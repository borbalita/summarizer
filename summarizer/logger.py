import logging
import logging.handlers
import os

from dotenv import load_dotenv

load_dotenv()

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logger = logging.getLogger("summarizer")
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level = "DEBUG"
logger.setLevel(log_level)

file_handler = logging.handlers.RotatingFileHandler(
    "summarizer.log", maxBytes=5_000_000, backupCount=5
)
file_handler.setFormatter(logging.Formatter(FORMAT))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(FORMAT))
logger.addHandler(console_handler)
