import re
from typing import Optional


def detect_retention_loop(
    text: str, min_pattern_length: int = 10, threshold: int = 50
) -> Optional[str]:
    """
    Detect if text ends with a repeated pattern (repetition loop).

    This is useful for VLM responses that sometimes get stuck repeating the same text.

    Args:
        text: The text to check for repetition.
        min_pattern_length: Minimum length of the pattern to detect.
        threshold: Number of consecutive repetitions required to trigger detection.

    Returns:
        The text before the repetition loop started, or None if no loop is detected.
    """
    if len(text) < min_pattern_length:
        return None

    # Check the last 5000 characters for loops
    sample_size = 5000
    tail = text[-sample_size:]

    for pattern_len in range(min_pattern_length, 200):
        if pattern_len > len(tail):
             break
             
        pattern = tail[-pattern_len:]

        if not pattern.strip():
            continue

        count = 1  # Start with 1 for the pattern at the end
        pos = len(tail) - pattern_len * 2  # Start checking immediately before the last pattern

        while pos >= 0: 
             chunk = tail[pos : pos + pattern_len]
             if chunk == pattern:
                count += 1
                pos -= pattern_len
             else:
                break
        
        if count >= threshold:
            # loop_start is relative to the FULL text
            loop_len = count * pattern_len
            loop_start = len(text) - loop_len
            return text[:loop_start].rstrip()

    return None


def extract_transcription(text: str) -> Optional[str]:
    """
    Extract the content enclosed between <transcription> tags.

    Args:
        text: The raw text containing the transcription tags.

    Returns:
        The extracted content if tags are found, otherwise None.
    """
    pattern = r"<transcription>(.*?)</transcription>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def has_complete_transcription(text: str) -> bool:
    """
    Check if the text contains a closing </transcription> tag.

    Args:
        text: The text to check.

    Returns:
        True if the closing tag is present, False otherwise.
    """
    return "</transcription>" in text


def check_max_tokens_hit(response) -> bool:
    """
    Verify if the OpenAI API response was truncated because it reached the max_tokens limit.

    Args:
        response: The OpenAI API response object.

    Returns:
        True if the finish_reason is 'length', False otherwise.
    """
    if hasattr(response, "choices") and response.choices:
        finish_reason = response.choices[0].finish_reason
        return finish_reason == "length"

    return False
