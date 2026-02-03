"""
Standardized prompts for document parsing workflows using Vision Language Models.
"""

DEFAULT_SYSTEM_PROMPT = """
You are a document parser. Convert this to clean Markdown.

- Preserve text, numbers, tables, order, language
- Use "|" tables only. "-" bullets only. "#" headers
- DO NOT omit details in Catatan/Notes columns
- NO summaries or inference

IMPORTANT: Wrap ONLY the content in <transcription></transcription> tags.

<image>
Parse this page into Markdown.
<transcription>
"""

TRANSCRIPTION = "IMPORTANT: Wrap ONLY the content in <transcription></transcription> tags."

DEFAULT_USER_PROMPT = """<image>
Parse this page into Markdown only.
"""

CONTINUE_PROMPT = """
continue

(You were transcribing a document. Here's where you left off:)
...{context}

Continue from exactly where you stopped. Close with </transcription> when done."""

FIX_PROMPT = """
You got stuck in a repetition loop. Continue transcription from before the loop.

Content before repetition started:
...{restart_from}

Continue from exactly this point. Don't repeat what's already written.
Close with </transcription> when finished.
"""
