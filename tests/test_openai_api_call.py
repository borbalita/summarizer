from unittest.mock import Mock, patch

import pytest
from openai import APIConnectionError, AuthenticationError, OpenAIError, RateLimitError

from summarizer.openai_api_call import sync_openai_call
from summarizer.summary_parameters import SummaryParameters


@pytest.fixture
def summary_params():
    """Fixture for SummaryParameters."""
    return SummaryParameters(model="gpt-3.5-turbo", max_summary_tokens=100)


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_success(mock_create, summary_params):
    """Test a successful OpenAI API call."""
    sync_openai_call("Summarize this text.", summary_params)


@pytest.fixture
def mock_response():
    """Fixture for a mock RateLimitError response."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.body = b'{"error": {"message": "You have hit the rate limit"}}'
    mock_response.http_version = "1.1"
    mock_response.request = {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
    }
    return mock_response


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_authentication_error(
    mock_create, summary_params, mock_response
):
    """Test handling of AuthenticationError."""
    mock_create.side_effect = AuthenticationError(
        "Invalid API key.", body={}, response=mock_response
    )
    with pytest.raises(AuthenticationError, match="Invalid API key."):
        sync_openai_call("Summarize this text.", summary_params)


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_rate_limit_error(mock_create, summary_params, mock_response):
    """Test handling of RateLimitError with retries."""
    mock_create.side_effect = RateLimitError(
        "Rate limit exceeded.", body={}, response=mock_response
    )
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(
            Exception, match="OpenAI API call failed after all retry attempts."
        ):
            sync_openai_call(
                "Summarize this text.", summary_params, retries=2, backoff=1.0
            )
        assert mock_sleep.call_count == 2
        mock_create.assert_called()


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_api_connection_error(mock_create, summary_params):
    """Test handling of APIConnectionError with retries."""
    mock_create.side_effect = APIConnectionError(
        message="Connection error.", request={}
    )
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(
            Exception, match="OpenAI API call failed after all retry attempts."
        ):
            sync_openai_call(
                "Summarize this text.", summary_params, retries=3, backoff=1.0
            )
        assert mock_sleep.call_count == 3
        mock_create.assert_called()


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_generic_openai_error(mock_create, summary_params):
    """Test handling of a generic OpenAIError."""
    mock_create.side_effect = OpenAIError("Generic OpenAI error.")
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(
            Exception, match="OpenAI API call failed after all retry attempts."
        ):
            sync_openai_call(
                "Summarize this text.", summary_params, retries=2, backoff=1.0
            )
        assert mock_sleep.call_count == 2  # Retries twice
        mock_create.assert_called()


@patch("summarizer.openai_api_call.client.responses.create")
def test_sync_openai_call_unexpected_error(mock_create, summary_params):
    """Test handling of an unexpected exception."""
    mock_create.side_effect = Exception("Unexpected error.")
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(
            Exception, match="OpenAI API call failed after all retry attempts."
        ):
            sync_openai_call(
                "Summarize this text.", summary_params, retries=2, backoff=1.0
            )
        assert mock_sleep.call_count == 2
        mock_create.assert_called()
