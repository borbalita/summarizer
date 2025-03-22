from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SummaryParameters(BaseModel):
    """Pydantic model for text summarization parameters."""

    model: Literal["gpt-3.5-turbo", "gpt-4-turbo"] = Field(
        default="gpt-4-turbo", description="Model to use for summarization."
    )
    max_summary_tokens: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum number of tokens for summarization.",
    )
    chunk_size: int = Field(
        default=2000,
        gt=0,
        le=20000,
        description=(
            "Chunk size in **characters** used to divide the input text for summarization. "
            "Note: 200 characters roughly equals 50 tokens. "
            "Must be at least 10× larger than `max_summary_tokens` to allow for effective summarization "
            "and ensure that recursive summarization will come to an end."
        ),
    )
    chunk_overlap: int = Field(
        default=200, ge=0, description="Chunk overlap for processing large texts."
    )

    @model_validator(mode="after")
    def check_chunk_size_vs_summary_tokens(self) -> "SummaryParameters":
        if self.chunk_size < 10 * self.max_summary_tokens:
            raise ValueError(
                f"`chunk_size` must be at least 10× `max_summary_tokens` "
                f"(got chunk_size={self.chunk_size}, max_summary_tokens={self.max_summary_tokens})."
            )
        return self
