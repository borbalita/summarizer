from unittest.mock import patch

import pytest

from summarizer.summarizers.text_summarizer import summarize_text
from summarizer.summary_parameters import SummaryParameters


@pytest.fixture
def summary_params():
    """Fixture for SummaryParameters."""
    return SummaryParameters(
        model="gpt-3.5-turbo", max_summary_tokens=50, chunk_size=500, chunk_overlap=0
    )


@pytest.mark.asyncio
async def test_summarize_text_integration(
    summary_params,
):
    """Integration test for summarization without mocking."""
    base_text = "This is a long text that will be split into multiple chunks."
    text = base_text * 100

    mock_summary = "This is a summary."

    with patch(
        "summarizer.summarizers.text_summarizer.sync_openai_call",
        return_value=mock_summary,
    ):
        result = await summarize_text(text, summary_params)

    assert result == mock_summary
