# Text Summarizer

The **Text Summarizer** is a Python-based application designed to summarize large pieces of text using OpenAI's language models. It provides a Streamlit-based user interface for easy interaction and allows users to tweak summarization parameters such as maximum summary tokens, chunk size, and chunk overlap.


## Running the Summarizer App with Docker

### Prerequisites
- Ensure you have Docker installed on your system. You can download it from [Docker's official website](https://www.docker.com/).

### Steps to Run the App

1. **Build the Docker Image:**
   Navigate to the project directory and build the Docker image using the following command:
   ```bash
   docker build -t text-summarizer .
   ```

2. **Run the Docker Container:**
    Start the container and map port 8501 to your local machine:

    ```bash
    docker run -p 8501:8501 text-summarizer
    ```
3. **Access the App:**
    Open your web browser and go to `http://localhost:8501` to access the Text Summarizer app.