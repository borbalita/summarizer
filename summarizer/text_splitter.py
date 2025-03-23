from langchain_text_splitters import RecursiveCharacterTextSplitter

from summarizer.logger import logger
from summarizer.summary_parameters import SummaryParameters


def split_text(text: str, summary_parameters: SummaryParameters) -> list[str]:
    """Split text into chunks of the requested size."""
    logger.debug(
        f"Splitting text of length {len(text)} into chunks with chunk size {summary_parameters.chunk_size} and overlap {summary_parameters.chunk_overlap}."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=summary_parameters.chunk_size,
        chunk_overlap=summary_parameters.chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    logger.debug(f"Split text into {len(chunks)} chunks.")

    return chunks
