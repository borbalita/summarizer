import asyncio

import streamlit as st

from summarizer.summarizers.text_summarizer import summarize_text
from summarizer.summary_parameters import SummaryParameters

st.title("Text Summarization App")

st.sidebar.header("Summary Parameters")

model = st.sidebar.selectbox(
    "Model",
    options=["gpt-3.5-turbo", "gpt-4-turbo"],
    index=0,
    help="Select the model to use for summarization.",
)

max_summary_tokens = st.sidebar.number_input(
    "Max Summary Tokens",
    min_value=10,
    value=200,
    step=10,
    help="Maximum number of tokens for the summary.",
)
chunk_size = st.sidebar.number_input(
    "Chunk Size",
    min_value=200,
    value=10000,
    step=100,
    help="Chunk size in characters for splitting the "
    "input text. Must be at least 10x bigger than "
    "`max_summary_tokens`. Keep in mind the maximum "
    "token limit for the model.",
)
chunk_overlap = st.sidebar.number_input(
    "Chunk Overlap",
    min_value=0,
    value=1000,
    step=50,
    help="Number of overlapping characters between "
    "chunks. Must be significantly smaller than "
    "`chunk_size`.",
)

uploaded_file = st.file_uploader(
    "Upload a text file (txt or md):",
    type=["txt", "md"],
    help="Upload a .txt or .md file containing the text you want to summarize.",
)

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
else:
    text = ""


if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        summary_params = SummaryParameters(
            model=model,  # type: ignore
            max_summary_tokens=max_summary_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        with st.spinner("Summarizing..."):
            try:
                summary = asyncio.run(summarize_text(text, summary_params))
                st.success("Summarization complete!")
                st.text_area("Summary:", value=summary, height=200)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
