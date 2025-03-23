import logging

from summarizer.logger import logger


def test_logger_basic_functionality(caplog):
    """Test that the logger writes messages at the correct log level."""
    test_message = "This is a test log message."

    with caplog.at_level(logging.DEBUG, logger="summarizer"):
        logger.debug(test_message)

    assert test_message in caplog.text
    assert "DEBUG" in caplog.text
