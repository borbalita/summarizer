import asyncio

from summarizer.logger import logger
from summarizer.openai_api_call import sync_openai_call
from summarizer.summary_parameters import SummaryParameters

CHUNK_SUMMARY_PROMPT = """Given the following passage, generate a concise and coherent summary that captures the main ideas while eliminating redundancy. Keep the summary within {max_tokens} words and ensure it retains key details. If the passage is too short, you can return the original text as the summary.

Passage:
{text}

Summary:
"""

# ToDo: add LLM as a judge to check output


async def summarize_text_chunk(text: str, summary_params: SummaryParameters) -> str:
    """Summarize a text chunk."""
    if len(text) == 0:
        logger.warning(
            "Empty text chunk passed to the chunk summarizer. Returning an empty string."
        )
        return ""

    logger.debug(
        f"Summarizing a text chunk of length {len(text)} with max summary tokens: {summary_params.max_summary_tokens}."
    )

    prompt = CHUNK_SUMMARY_PROMPT.format(
        text=text, max_tokens=summary_params.max_summary_tokens
    )
    summary = await asyncio.to_thread(sync_openai_call, prompt, summary_params)

    # ToDo: define a custom detailed debug level for logging the generated summary
    logger.debug(f"Generated a summary of length {len(summary)}")

    return summary


async def summarize_text_chunks(
    chunks: list[str], summary_params: SummaryParameters
) -> list[str]:
    """Summarize chunks in parallel."""
    tasks = [summarize_text_chunk(chunk, summary_params) for chunk in chunks]
    summaries = await asyncio.gather(*tasks)
    return summaries
