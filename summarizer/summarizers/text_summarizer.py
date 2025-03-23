import asyncio
import math

from summarizer.logger import logger
from summarizer.openai_api_call import sync_openai_call
from summarizer.summarizers.chunk_summarizer import summarize_text_chunks
from summarizer.summary_parameters import SummaryParameters
from summarizer.text_splitter import split_text

COMBINE_SUMMARY_PROMPT = """Below are chunk-level summaries from different sections of a long document. Each chunk summary represents the content of a specific portion of the text and may contain overlapping or repeated information.

Your task is to synthesize these into one coherent and concise final summary. Do not copy from the chunks directly. Instead, distill the key ideas, eliminate redundancies, and present the overall meaning in a clear, structured form.

Use a neutral, informative tone and write in natural, fluent language.

Also, it's very important that you keep the summary within {max_tokens} words and ensure it retains key details.

### Chunk Summaries:

{chunk_summaries}

### Final Summary:
"""


async def _summarize_summary_group(
    summaries: list[str], summary_params: SummaryParameters
) -> str:
    """Summarize a group of summaries."""
    # todo: check max summary length not exceeded
    formatted_summaries = [f"Summary {i}: {s}" for i, s in enumerate(summaries)]
    prompt = COMBINE_SUMMARY_PROMPT.format(
        chunk_summaries="\n\n".join(formatted_summaries),
        max_tokens=summary_params.max_summary_tokens,
    )
    return await asyncio.to_thread(sync_openai_call, prompt, summary_params)


# ToDo: group_size could be automatically set based on chunk size / summary size ratio.
async def _summarize_summary_groups(
    summaries: list[str], summary_params: SummaryParameters, group_size=4
) -> list[str]:
    """Summarize subsets of summaries in parallel."""
    n_groups = math.ceil(len(summaries) / group_size)

    logger.info(f"Split summaries into {n_groups} summary groups of size {group_size}.")

    groups = [summaries[i * group_size : (i + 1) * group_size] for i in range(n_groups)]
    tasks = [_summarize_summary_group(group, summary_params) for group in groups]

    combined_summaries: list[str] = []
    completed = 0

    for task in asyncio.as_completed(tasks):
        summary = await task
        combined_summaries.append(summary)
        completed += 1
        if completed % 10 == 0 or completed == n_groups:
            logger.info(f"Completed {completed}/{n_groups} summary groups.")

    return combined_summaries


async def summarize_text(text: str, summary_params: SummaryParameters) -> str:
    """Summarize a text."""
    logger.info("Beginning to summarize text")

    chunks = split_text(text, summary_params)
    logger.info(f"Split the text into {len(chunks)} chunks.")

    summaries = await summarize_text_chunks(chunks, summary_params)

    while len(summaries) > 1:
        logger.info(f"Beginning the combination of {len(summaries)} summaries.")
        summaries = await _summarize_summary_groups(summaries, summary_params)

    logger.info("Finished summarizing text.")

    return summaries[0]
