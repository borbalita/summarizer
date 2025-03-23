# Summary Task Report

## Design Choices

For this task, I implemented a simple text summarizer focused primarily on preserving and condensing content. Note that a single summarization approach would not universally apply to all use cases. For example, analyzing a meeting transcript might benefit from a structured, bullet-point format to highlight key decisions and action items. In contrast, summarizing a book requires capturing broader narratives, themes, and a coherent arc — something best achieved with a continuous, prose-style summary. I decided to focus on the latter considering the time limit for solving this challenge.

### Recursive chunking

To ensure flexibility and robustness across different model backends, I employed a character-based recursive chunking strategy. This approach performs consistently well regardless of the language model used. In a real-world application, it would be worthwhile to experiment with multiple chunking strategies tailored to different content types — what works well for summarizing a novel might not be optimal for a technical whitepaper or an academic transcript.

### Hierarchical Summarization

Summaries are generated through an iterative aggregation process. Initially, the input text is split into manageable chunks. Each chunk is summarized individually, and the resulting summaries are grouped together and summarized again. This recursive summarization continues until a final summary is produced. This strategy helps preserve coherence and structure across very large texts, where a single-pass summarization might fail to retain important context.

### Asynchronous Processing

To make the processing efficient, the summarizer is built using Python’s asyncio, which enables asynchronous handling of tasks like chunking, summarizing, and recombining summaries. This parallelism optimizes runtime performance, especially when dealing with very large inputs.

## Challenges Encountered

### Open AI's Rate Limit
One of the main limitations faced during implementation was OpenAI’s rate limiting, which significantly slowed down the summarization process. To mitigate this, I implemented robust error handling with retries and exponential backoff, allowing the process to continue even after hitting rate limits. However, the waiting time caused by the rate limit slows down the application significantly for large documents. For instance, summarizing War and Peace — using a chunk size of 10,000 characters and a summary size of 1,000 characters — took approximately 5 minutes.

### Model Output Limit

Another issue arose with how the final sentence of some summaries gets cut off due to the output token limit. This could be addressed either by refining the prompt to ask the model to ensure completeness at the end of the output or by implementing post-processing logic to detect and complete truncated sentences, possibly by feeding the tail back into the model for continuation.

## Potential Improvements

There are several areas where the summarizer could be enhanced further:

### Quality Evaluation

Integrating a large language model (LLM) as a judge could help assess the quality of generated summaries. While LLM-based evaluation is inherently flaky — since many different summaries might be valid — it can still offer useful signals. However, to ensure more reliable assessment, the summarizer's outputs should also be reviewed by human evaluators. Human judgment remains significantly more dependable for evaluating coherence, fidelity to the original text, and overall usefulness, especially in nuanced or complex summaries.

### Semantic Similarity Testing

Another approach would be to compare generated summaries against high-quality, human-written ones using embedding similarity. However, this too has its limitations, as multiple valid summaries can differ significantly in form and wording.

### Improved Testing

Testing was somewhat underdeveloped in this prototype, largely due to time constraints. More structured and varied tests would help validate the summarizer's behavior under different scenarios and improve its robustness.

### Support for Multiple Summarization Styles

Depending on the content, users might prefer different summary formats — bullet points, timelines, thematic overviews, etc. Introducing modular summarizers optimized for different tasks (books, business reports, transcripts) would make the tool much more versatile.
