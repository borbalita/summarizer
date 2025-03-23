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
async def test_summarize_text_logging(summary_params, caplog):
    """Test logging output during summarization."""
    base_text = "This is a long text that will be split into multiple chunks."
    text = base_text * 100

    mock_summary = "This is a summary."

    with patch(
        "summarizer.summarizers.text_summarizer.sync_openai_call",
        return_value=mock_summary,
    ):
        with caplog.at_level("INFO", logger="summarizer"):
            result = await summarize_text(text, summary_params)

    assert result == mock_summary

    assert "Split the text into 13 chunks." in caplog.text
    assert "Completed 13/13 chunks." in caplog.text
    assert "Beginning the combination of 13 summaries." in caplog.text
    assert """Split summaries into 4 summary groups of size 4.""" in caplog.text
    assert "Completed 4/4 summary groups." in caplog.text
    assert """Split summaries into 1 summary groups of size 4.""" in caplog.text
    assert "Completed 1/1 summary groups." in caplog.text
    assert "Finished summarizing text." in caplog.text
