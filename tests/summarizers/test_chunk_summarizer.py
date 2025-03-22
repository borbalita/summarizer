from unittest.mock import patch

import pytest

from summarizer.summarizers.chunk_summarizer import (
    summarize_text_chunk,
    summarize_text_chunks,
)
from summarizer.summary_parameters import SummaryParameters


@pytest.mark.asyncio
async def test_summarize_text_chunk_success():
    """Test that summarize_chunk successfully summarizes the given text."""
    text = "This is a test passage."
    mock_summary = "This is a summary."
    summary_params = SummaryParameters(max_summary_tokens=50)

    with patch(
        "summarizer.summarizers.chunk_summarizer.sync_openai_call",
        return_value=mock_summary,
    ):
        result = await summarize_text_chunk(text, summary_params)
        assert result == mock_summary


@pytest.mark.asyncio
async def test_summarize_text_chunk_empty_text(caplog):
    """Test that summarize_chunk returns an empty string for empty input and logs a warning."""
    text = ""
    summary_params = SummaryParameters(max_summary_tokens=50)
    with caplog.at_level("WARNING"):
        result = await summarize_text_chunk(text, summary_params)
        assert result == ""
        assert (
            "Empty text chunk passed to the chunk summarizer. Returning an empty string."
            in caplog.text
        )


@pytest.mark.asyncio
async def test_summarize_text_chunks_success():
    """Test that summarize_chunk successfully summarizes the given text."""
    text = "This is a test passage."
    mock_summary = "This is a summary."
    summary_params = SummaryParameters(max_summary_tokens=50)

    with patch(
        "summarizer.summarizers.chunk_summarizer.sync_openai_call",
        return_value=mock_summary,
    ):
        result = await summarize_text_chunks([text] * 3, summary_params)
        assert result == [mock_summary] * 3
