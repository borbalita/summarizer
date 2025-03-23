from pathlib import Path

from summarizer.summarizers.text_summarizer import summarize_text
from summarizer.summary_parameters import SummaryParameters


# TODO: test quality of summary with an LLM judge
def test_summarize_fantasy_story():
    file_path = Path(__file__).parent / "data" / "fantasy_story.md"

    assert file_path.exists(), f"File {file_path} does not exist."

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    summary_params = SummaryParameters()
    summary = summarize_text(content, summary_params)

    assert summary != ""
