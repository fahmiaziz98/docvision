import pytest

from docvision.utils.helper import extract_transcription


@pytest.mark.unit
class TestTranscriptionExtraction:
    """Test XML transcription tag extraction."""

    def test_extract_complete(self):
        """Should extract content between tags."""
        text = "preamble <transcription>extracted content</transcription> postamble"
        assert extract_transcription(text) == "extracted content"

    def test_extract_multiline(self):
        """Should handle multiline content."""
        text = """<transcription>
Line 1
Line 2
Line 3
</transcription>"""
        result = extract_transcription(text)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_no_tags(self):
        """Should return original text when no tags present."""
        text = "No tags here, just plain text"
        assert extract_transcription(text) == text

    def test_incomplete_opening_tag(self):
        """Should return original text for incomplete opening tag."""
        text = "<transcriptioncontent but no closing tag"
        assert extract_transcription(text) == text

    def test_incomplete_closing_tag(self):
        """Should return original text for incomplete closing tag."""
        text = "<transcription>content but no closing tag"
        assert extract_transcription(text) == text

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        text = "<transcription>   content with spaces   </transcription>"
        assert extract_transcription(text) == "content with spaces"
