import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_setup_logging():
    from src.utils.logging_utils import setup_logging
    import logging

    logger = setup_logging(level="DEBUG")

    assert logger.level == logging.DEBUG


def test_get_logger():
    from src.utils.logging_utils import get_logger

    logger = get_logger("test_module")

    assert logger.name == "arbitrage_bot.test_module"
