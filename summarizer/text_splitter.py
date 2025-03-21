from langchain_text_splitters import RecursiveCharacterTextSplitter

from summarizer.summary_parameters import SummaryParameters


def split_text(text: str, summary_parameters: SummaryParameters) -> list[str]:
    """Split text into chunks of the requested size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=summary_parameters.chunk_size,
        chunk_overlap=summary_parameters.chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_text(text)
