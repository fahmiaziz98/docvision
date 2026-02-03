import pytest
from docvision.utils.helper import (
    detect_retention_loop,
    extract_transcription,
    has_complete_transcription,
    check_max_tokens_hit
)


@pytest.mark.unit
class TestRepetitionDetection:
    """Test repetition loop detection."""
    
    def test_no_repetition(self):
        """Should return None for normal text."""
        text = "This is normal text without any loops or repetition patterns."
        assert detect_retention_loop(text) is None
    
    def test_simple_repetition(self):
        """Should detect simple repetition pattern."""
        base = "Some content before the loop starts. "
        pattern = "This text repeats. "
        text = base + (pattern * 60)  # 60 repetitions (threshold is 50)
        
        result = detect_retention_loop(text)
        assert result == base.rstrip()
    
    def test_edge_case_short_text(self):
        """Should return None for text too short to have loops."""
        text = "Too short"
        assert detect_retention_loop(text) is None
    
    def test_below_threshold(self):
        """Should not detect if below threshold."""
        base = "Content. "
        pattern = "Repeat. "
        # Only 10 repetitions (below threshold of 50)
        text = base + (pattern * 10)
        
        assert detect_retention_loop(text) is None
    
    def test_custom_threshold(self):
        """Should respect custom threshold."""
        base = "Content. "
        pattern = "Rep. "
        text = base + (pattern * 15)
        
        # With threshold=10, should detect
        # IMPORTANT: min_pattern_length must be smaller than "Rep. " (5 chars)
        result = detect_retention_loop(text, min_pattern_length=4, threshold=10)
        assert result == base.rstrip()
        
        # With threshold=20, should not detect
        result = detect_retention_loop(text, threshold=20)
        assert result is None


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
        """Should return None when no tags present."""
        text = "No tags here, just plain text"
        assert extract_transcription(text) is None
    
    def test_incomplete_opening_tag(self):
        """Should return None for incomplete opening tag."""
        text = "<transcriptioncontent but no closing tag"
        assert extract_transcription(text) is None
    
    def test_incomplete_closing_tag(self):
        """Should return None for incomplete closing tag."""
        text = "<transcription>content but no closing tag"
        assert extract_transcription(text) is None
    
    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        text = "<transcription>   content with spaces   </transcription>"
        assert extract_transcription(text) == "content with spaces"


@pytest.mark.unit
class TestCompleteTranscriptionCheck:
    """Test checking for complete transcription."""
    
    def test_has_closing_tag(self):
        """Should return True when closing tag present."""
        text = "<transcription>content</transcription>"
        assert has_complete_transcription(text) is True
    
    def test_no_closing_tag(self):
        """Should return False when no closing tag."""
        text = "<transcription>content without closing"
        assert has_complete_transcription(text) is False
    
    def test_only_closing_tag(self):
        """Should return True even with only closing tag."""
        text = "some text </transcription>"
        assert has_complete_transcription(text) is True


@pytest.mark.unit
class TestMaxTokensDetection:
    """Test max_tokens detection from API response."""
    
    def test_finish_reason_length(self, mock_openai_response):
        """Should detect when finish_reason is 'length'."""
        response = mock_openai_response(finish_reason="length")
        assert check_max_tokens_hit(response) is True
    
    def test_finish_reason_stop(self, mock_openai_response):
        """Should not detect when finish_reason is 'stop'."""
        response = mock_openai_response(finish_reason="stop")
        assert check_max_tokens_hit(response) is False
    
    def test_no_choices(self):
        """Should handle response with no choices."""
        class EmptyResponse:
            choices = []
        
        assert check_max_tokens_hit(EmptyResponse()) is False
    
    def test_no_choices_attribute(self):
        """Should handle response without choices attribute."""
        class NoChoicesResponse:
            pass
        
        assert check_max_tokens_hit(NoChoicesResponse()) is False
