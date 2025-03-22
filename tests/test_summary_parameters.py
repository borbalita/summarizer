import pytest

from summarizer.summary_parameters import SummaryParameters


@pytest.mark.parametrize(
    "model, max_summary_tokens, chunk_size, chunk_overlap",
    [
        ("gpt-3.5-turbo", 50, 1000, 100),
        ("gpt-4-turbo", 200, 5000, 500),
        ("gpt-3.5-turbo", 100, 2000, 0),
        ("gpt-4-turbo", 10, 200, 20),
    ],
)
def test_summary_parameters_valid_values(
    model, max_summary_tokens, chunk_size, chunk_overlap
):
    """Test that valid values are accepted."""
    params = SummaryParameters(
        model=model,
        max_summary_tokens=max_summary_tokens,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    assert params.model == model
    assert params.max_summary_tokens == max_summary_tokens
    assert params.chunk_size == chunk_size
    assert params.chunk_overlap == chunk_overlap


def test_summary_parameters_default_values():
    """Test that default values are correctly set."""
    params = SummaryParameters()
    assert params.model == "gpt-4-turbo"
    assert params.max_summary_tokens == 100
    assert params.chunk_size == 2000
    assert params.chunk_overlap == 200


@pytest.mark.parametrize(
    "invalid_model",
    ["invalid-model", "", None],
)
def test_summary_parameters_invalid_model(invalid_model):
    """Test that an invalid model raises a validation error."""
    with pytest.raises(
        ValueError, match="Input should be 'gpt-3.5-turbo' or 'gpt-4-turbo'"
    ):
        SummaryParameters(model=invalid_model)


@pytest.mark.parametrize(
    "max_summary_tokens",
    [-1, 0],
)
def test_summary_parameters_invalid_max_summary_tokens(max_summary_tokens):
    """Test that invalid max_summary_tokens raises a validation error."""
    with pytest.raises(ValueError, match="Input should be greater than 0"):
        SummaryParameters(max_summary_tokens=max_summary_tokens)


@pytest.mark.parametrize(
    "chunk_size",
    [-1, 0],
)
def test_summary_parameters_invalid_chunk_size_too_big(chunk_size):
    """Test that invalid chunk_size raises a validation error."""
    with pytest.raises(ValueError, match="Input should be greater than 0"):
        SummaryParameters(chunk_size=chunk_size)


@pytest.mark.parametrize(
    "chunk_overlap",
    [-1],
)
def test_summary_parameters_invalid_chunk_overlap(chunk_overlap):
    """Test that invalid chunk_overlap raises a validation error."""
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        SummaryParameters(chunk_overlap=chunk_overlap)


@pytest.mark.parametrize(
    "max_summary_tokens, chunk_size",
    [
        (200, 1500),
        (100, 500),
    ],
)
def test_summary_parameters_chunk_size_vs_max_summary_tokens(
    max_summary_tokens, chunk_size
):
    """Test that invalid chunk size and may summary token cominations are rejected."""
    with pytest.raises(
        ValueError, match="`chunk_size` must be at least 10× `max_summary_tokens`"
    ):
        SummaryParameters(max_summary_tokens=max_summary_tokens, chunk_size=chunk_size)
