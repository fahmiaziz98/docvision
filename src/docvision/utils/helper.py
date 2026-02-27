import re
from typing import Optional


def extract_transcription(text: str) -> Optional[str]:
    """
    Extract the content enclosed between <transcription> tags.

    Args:
        text: The raw text containing the transcription tags.

    Returns:
        The extracted content if tags are found, otherwise the original text stripped.
    """
    pattern = r"<transcription>(.*?)</transcription>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return text.strip()


def extract_response_text(response) -> str:
    """Safely extract text content from a VLM API response object."""
    try:
        return response.choices[0].message.content.strip()
    except (AttributeError, IndexError):
        return ""
