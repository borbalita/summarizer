import logging

from summarizer.summary_parameters import SummaryParameters
from summarizer.text_splitter import split_text


def test_split_text_single_chunk():
    """Test short text returned as a single chunk."""
    text = "This is a tiny text."
    summary_parameters = SummaryParameters(
        chunk_size=20, chunk_overlap=0, max_summary_tokens=2
    )
    result = split_text(text, summary_parameters)
    assert result == ["This is a tiny text."]


def test_split_text_multiple_chunks():
    """Test longer text split into multiple chunks without overlap."""
    text = "This is a longer text that needs to be split into multiple chunks because of the chunk size."
    summary_parameters = SummaryParameters(
        chunk_size=20, chunk_overlap=0, max_summary_tokens=2
    )
    result = split_text(text, summary_parameters)
    assert result == [
        "This is a longer",
        "text that needs to",
        "be split into",
        "multiple chunks",
        "because of the",
        "chunk size.",
    ]


def test_split_text_recursive_splitting():
    """Test recursive splitting prioritizes \n\n over \n or word boundaries."""
    text = "This\nis a\n\nlonger text\n that needs to be\n\nsplit\ninto multiple chunks because of the chunk size."
    summary_parameters = SummaryParameters(
        chunk_size=20, chunk_overlap=0, max_summary_tokens=2
    )
    result = split_text(text, summary_parameters)
    assert result == [
        "This\nis a",
        "longer text",
        "that needs to be",
        "split",
        "into multiple",
        "chunks because of",
        "the chunk size.",
    ]


def test_split_text_with_overlap():
    """Test overlapping chunks created correctly."""
    text = "This is a test for overlapping chunks."
    summary_parameters = SummaryParameters(
        chunk_size=10, chunk_overlap=5, max_summary_tokens=1
    )
    result = split_text(text, summary_parameters)
    assert result == [
        "This is a",
        "is a test",
        "test for",
        "overlappi",
        "lapping",
        "chunks.",
    ]


def test_split_text_empty_input():
    """Test empty input returns an empty list."""
    text = ""
    summary_parameters = SummaryParameters(
        chunk_size=10, chunk_overlap=0, max_summary_tokens=1
    )
    result = split_text(text, summary_parameters)
    assert result == []


def test_split_text_logging(caplog):
    """Test that split_text writes the expected logs."""
    text = "This is a short text."
    summary_parameters = SummaryParameters(
        chunk_size=10, chunk_overlap=0, max_summary_tokens=1
    )

    with caplog.at_level(logging.DEBUG, logger="summarizer"):
        result = split_text(text, summary_parameters)

    assert (
        "Splitting text of length 21 into chunks with chunk size 10 and overlap 0."
        in caplog.text
    )

    assert len(result) == 3
    assert "Split text into 3 chunks." in caplog.text
