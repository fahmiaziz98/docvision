# 📄 DocVision Parser

> Production-ready document parsing framework powered by Vision Language Models (VLMs) and Native PDF extraction.

[![Tests](https://github.com/fahmiaziz98/doc-vision-parser/workflows/Tests/badge.svg)](https://github.com/fahmiaziz98/doc-vision-parser/actions)
[![PyPI version](https://badge.fury.io/py/doc-vision-parser.svg)](https://badge.fury.io/py/doc-vision-parser)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

DocVision Parser is a robust Python library designed to extract high-quality structured text and markdown from documents (images and PDFs). It combines the speed of **native PDF extraction** with the reasoning power of **Vision Language Models** (like GPT-4o, Claude, or Llama 3.2).

The framework provides three powerful parsing modes:
1.  **PDF (Native)**: Ultra-fast extraction of text and tables using deterministic rules.
2.  **VLM Mode**: High-fidelity single-shot parsing using Vision models to understand layout and context.
3.  **Agentic Mode**: A self-correcting, iterative workflow that handles long documents and complex layouts by automatically detecting truncation or repetition.

## Features

-   **Hybrid PDF Parsing**: Extract native text/tables and optionally use VLM to describe charts and images in-situ.
-   **Agentic/Iterative Workflow**: Self-correcting loop that handles model token limits and ensures complete transcription for long pages.
-   **Intelligent Vision Pipeline**: Automatic image rotation correction, DPI management, and dynamic optimization for the best VLM input.
-   **Async-First**: High-throughput processing with built-in concurrency control (Semaphores).
-   **Structured Output**: Native Pydantic support for extracting structured JSON data from any document.
-   **Production-Ready**: Automatic retries, error handling, and direct export to Markdown or JSON files.

## Installation

Install using `pip`:

```bash
pip install doc-vision-parser
```

Or using `uv` (recommended):

```bash
uv add doc-vision-parser
```

---

## Quick Start

### Basic Usage

Initialize the `DocumentParser` and parse an image into Markdown.

```python
import asyncio
from docvision import DocumentParser

async def main():
    # Initialize the parser
    parser = DocumentParser(
        vlm_base_url="https://api.openai.com/v1",
        vlm_model="gpt-4o-mini",
        vlm_api_key="your_api_key"
    )

    # Parse an image
    result = await parser.parse_image("document.jpg")
    
    print(result.content)
    print(f"ID: {result.id}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Parsing PDFs

The parser can handle PDFs using different strategies.

```python
from docvision import DocumentParser, ParsingMode

async def parse_doc():
    parser = DocumentParser(vlm_base_url=..., vlm_model=..., vlm_api_key=...)

    # Mode 1: Native PDF (Fastest, no Vision costs)
    results = await parser.parse_pdf("report.pdf", parsing_mode=ParsingMode.PDF)

    # Mode 2: VLM (Best for complex layouts/handwriting)
    results = await parser.parse_pdf("scanned.pdf", parsing_mode=ParsingMode.VLM)

    # Mode 3: AGENTIC (Self-correcting for long tables/text)
    results = await parser.parse_pdf("dense.pdf", parsing_mode=ParsingMode.AGENTIC)

    # Save results directly to file
    await parser.parse_pdf("input.pdf", save_path="./output/results.md")
```

---

## Advanced Features

### Structured Output (JSON)

Extract data directly into Pydantic models.

```python
from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    description: str
    price: float

class Invoice(BaseModel):
    invoice_no: str
    items: List[Item]

# Note: system_prompt is required when using structured output
parser = DocumentParser(
    vlm_api_key="...", 
    system_prompt="Extract invoice details correctly."
)

result = await parser.parse_image("invoice.png", output_schema=Invoice)
print(result.content.invoice_no) # Content is now a Pydantic object
```

### Hybrid Parsing (Native + VLM)

Use native extraction for text but let the VLM describe the charts.

```python
parser = DocumentParser(
    vlm_api_key="...", 
    chart_description=True # This enables VLM hybrid for Native Mode
)

# Text and Tables are extracted natively, but <chart> tags 
# will contain VLM-generated descriptions.
results = await parser.parse_pdf("chart_heavy.pdf", parsing_mode=ParsingMode.PDF)
```

---

## Configuration

The `DocumentParser` is configured during initialization.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vlm_base_url` | `str` | `None` | OpenAI-compatible API base URL. |
| `vlm_model` | `str` | `None` | Model name (e.g., `gpt-4o`). |
| `vlm_api_key` | `str` | `None` | Your API key. |
| `temperature` | `float` | `0.7` | Model sampling temperature. |
| `max_tokens` | `int` | `4096` | Max tokens per VLM call. |
| `max_iterations` | `int` | `3` | Max retries/loops in Agentic mode. |
| `max_concurrency`| `int` | `5` | Max concurrent pages being processed. |
| `enable_rotate` | `bool` | `True` | Auto-fix image orientation. |
| `chart_description`| `bool` | `False`| Use VLM to describe charts in Native mode. |
| `render_zoom` | `float` | `2.0` | DPI multiplier for PDF rendering. |
| `debug_dir` | `str` | `None` | Directory to save debug images. |

---

## Architecture

DocVision Parser is built for reliability and scale:

1.  **VLMClient**: Handles asynchronous communication with OpenAI/Groq/OpenRouter with built-in retries and timeout management.
2.  **NativePDFParser**: Uses `pdfplumber` to extract structured text and complex tables while maintaining reading order.
3.  **ImageProcessor**: A high-performance pipeline for converting PDFs and optimizing images (resizing, padding, rotating).
4.  **AgenticWorkflow**: A state-machine that manages long-running generation tasks, ensuring complete document transcription.

## Development

```bash
# Setup
uv sync --dev

# Run Tests
make test

# Lint & Format
make lint
make format
```

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Author

**Fahmi Aziz Fadhil**
- GitHub: [@fahmiaziz98](https://github.com/fahmiaziz98)
- Email: fahmiazizfadhil09@gmail.com
