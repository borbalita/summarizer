import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from summarizer.logger import logger
from summarizer.summary_parameters import SummaryParameters

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def sync_openai_call(
    prompt: str,
    summary_params: SummaryParameters,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Calls the OpenAI API synchronously with retry and error handling."""
    for attempt in range(retries):
        sleep_time = backoff * (2**attempt)
        try:
            response = client.responses.create(
                model=summary_params.model,
                instructions="You are a helpful assistant that summarizes text concisely.",
                input=prompt,
                temperature=0.5,
                max_output_tokens=summary_params.max_summary_tokens,
            )
            # ToDo: look up expected response type
            return response.output[0].content[0].text  # type: ignore

        except AuthenticationError as e:
            logger.error("Authentication failed. Please check your OpenAI API key.")
            raise e

        except RateLimitError:
            logger.error(
                f"Rate limit exceeded. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{retries})"
            )

        except APIConnectionError:
            logger.error(
                f"API connection error. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{retries})"
            )

        except OpenAIError as e:
            logger.error(
                f"OpenAI API error: {e}. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{retries})"
            )

        except Exception as e:
            logger.error(
                f"Unexpected error: {e}. Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{retries})"
            )

        time.sleep(backoff * (2**attempt))

    logger.critical("OpenAI API call failed after all retry attempts.")
    raise Exception("OpenAI API call failed after all retry attempts.")
