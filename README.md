# 📄 DocVision Parser

> Framework document parsing powered by Vision Language Models (VLMs) and OCR.

[![Tests](https://github.com/fahmiaziz98/doc-vision-parser/workflows/Tests/badge.svg)](https://github.com/fahmiaziz98/doc-vision-parser/actions)
[![PyPI version](https://badge.fury.io/py/docvision.svg)](https://badge.fury.io/py/docvision)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

---
> [!WARNING]
> **This project is still under active development and is not ready for production environments.**
> The API, code structure, and behavior may change at any time without prior notice.
> Use only in development or experimental environments.
---

## Overview

DocVision Parser is a Python library for extracting high-quality structured text and markdown from documents (images and PDFs). It combines **PaddleOCR ONNX** for fast, offline text extraction with the reasoning power of **Vision Language Models** (GPT-4o, Claude, Llama, etc.).

Three parsing modes:

| Mode | Best For | Requires |
| :--- | :--- | :--- |
| **BASIC_OCR** | Fast offline extraction, no GPU needed | — |
| **VLM** | Complex layouts, handwriting, mixed content | VLM API key |
| **AGENTIC** | Long documents, dense tables, self-correcting | VLM API key |

---

## What's New in v0.3.0

- **`BASIC_OCR` mode** — PaddleOCR ONNX via RapidOCR, models auto-downloaded from HuggingFace on first use. No PyTorch, no GPU required.
- **Dual preprocessing pipeline** — `preprocess_for_ocr` (CLAHE, deskew, DPI normalization) and `preprocess_for_vlm` (adaptive resize, rotation, crop) are now separate optimized pipelines.
- **Agentic reflect pattern** — Critic/refiner replace the old repetition-detection loop. Critic uses Pydantic structured output for reliable evaluation.
- **Multi-language OCR** — English, Latin (ID/FR/DE/ES), Chinese, Korean, Arabic, Hindi, Tamil, Telugu.
- **Breaking**: `ParsingMode.PDF` renamed to `ParsingMode.BASIC_OCR`.
- **Breaking**: `process_image()` replaced by `preprocess_for_ocr()` / `preprocess_for_vlm()`.

---

## Installation

```bash
pip install docvision
```

Or using `uv` (recommended):

```bash
uv add docvision
```

> **Note:** OCR models (~100MB) are downloaded automatically to `~/.cache/docvision/models/` on first use.

---

## Quick Start

### BASIC_OCR — No API key needed

```python
import asyncio
from docvision import DocumentParser, ParsingMode

async def main():
    parser = DocumentParser(
        ocr_language="english",  # or "latin" for Indonesian/European
    )

    # Parse a single image
    result = await parser.parse_image("document.jpg", parsing_mode=ParsingMode.BASIC_OCR)
    print(result.content)

    # Parse a PDF
    results = await parser.parse_pdf("report.pdf", parsing_mode=ParsingMode.BASIC_OCR)
    for page in results:
        print(f"Page {page.metadata['page_number']}:\n{page.content}")

asyncio.run(main())
```

### VLM — High-fidelity parsing

```python
from docvision import DocumentParser, ParsingMode

async def main():
    parser = DocumentParser(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        api_key="your_api_key",
    )

    result = await parser.parse_image("scanned.jpg", parsing_mode=ParsingMode.VLM)
    print(result.content)
```

### AGENTIC — Self-correcting for complex documents

```python
async def main():
    parser = DocumentParser(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key="your_api_key",
        max_reflect_cycles=2,  # critic→refine cycles per page (default: 2, max recommended: 2)
    )

    results = await parser.parse_pdf(
        "dense_report.pdf",
        parsing_mode=ParsingMode.AGENTIC,
        start_page=1,
        end_page=10,
    )

    for page in results:
        print(f"Page {page.metadata['page_number']} "
              f"(critic score: {page.metadata['final_critic_score']}):\n"
              f"{page.content}")
```

---

## Advanced Features

### Structured Output (JSON)

Extract data directly into Pydantic models using VLM mode.

```python
from pydantic import BaseModel
from typing import List

class LineItem(BaseModel):
    description: str
    quantity: int
    price: float

class Invoice(BaseModel):
    invoice_no: str
    total: float
    items: List[LineItem]

parser = DocumentParser(
    base_url="...",
    model_name="gpt-4o",
    api_key="...",
    system_prompt="Extract all invoice fields accurately.",
)

result = await parser.parse_image("invoice.png", output_schema=Invoice)
# result.content is a JSON string of the validated Invoice
print(result.content)
```

### Multi-language OCR

```python
# Indonesian, French, German, Spanish, etc. → use "latin"
parser = DocumentParser(ocr_language="latin")

# Chinese, Korean, Arabic, Hindi, Tamil, Telugu
parser = DocumentParser(ocr_language="chinese")

# Custom model directory (skip auto-download)
parser = DocumentParser(
    ocr_language="english",
    ocr_model_dir="/path/to/models",
)
```

### Save Results

```python
# Save as Markdown
await parser.parse_pdf("input.pdf", save_path="output/result.md")

# Save as JSON
await parser.parse_pdf("input.pdf", save_path="output/result.json")

# Save to directory (auto-creates output.json inside)
await parser.parse_pdf("input.pdf", save_path="output/")
```

---

## Configuration

```python
parser = DocumentParser(
    # VLM config (required for VLM and AGENTIC modes)
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o",
    api_key="your_key",
    temperature=0.7,
    max_tokens=4096,
    system_prompt=None,

    # Agentic config
    max_reflect_cycles=2,       # values > 2 emit UserWarning

    # OCR config (for BASIC_OCR mode)
    ocr_language="english",     # see supported languages below
    ocr_model_dir=None,         # None = auto-download to ~/.cache/docvision/

    # Image processing
    enable_crop=True,           # crop image to content
    enable_rotate=True,         # auto-correct orientation
    enable_deskew=True,         # correct small skew angles (OCR mode)
    dpi=300,                    # PDF render DPI multiplier
    post_crop_max_size=1024,    # max image dimension for VLM input
    max_concurrency=5,          # max concurrent pages
    debug_dir=None,             # save debug images here
)
```

### Supported OCR Languages

| Value | Covers |
| :--- | :--- |
| `"english"` | English |
| `"latin"` | Indonesian, French, German, Spanish, Portuguese, and other Latin-script languages |
| `"chinese"` | Simplified + Traditional Chinese |
| `"korean"` | Korean |
| `"arabic"` | Arabic |
| `"hindi"` | Hindi (Devanagari) |
| `"tamil"` | Tamil |
| `"telugu"` | Telugu |

---

## Architecture

```
DocumentParser
├── VLMClient          — async OpenAI-compatible API
├── OCREngine          — PaddleOCR ONNX via RapidOCR, HuggingFace 
├── ImageProcessor
│   ├── preprocess_for_ocr()   — deskew, DPI normalization, CLAHE contrast
│   └── preprocess_for_vlm()   — adaptive resize
└── AgenticWorkflow (LangGraph)
    ├── generate   — initial VLM parse
    ├── critic     — structural evaluation via Pydantic structured output
    ├── refine     — targeted fix based on critic issues
    └── complete   — terminal node
```

**Agentic reflect loop:**
```
generate → critic ──(score ≥ 8 or max cycles)──→ complete → END
               └──(score < 9)──→ refine → critic (loop)
```

---

## Development

```bash
# Setup
uv sync --dev

# Run tests
make test

# Lint & format
make lint
make format
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Author

**Fahmi Aziz Fadhil**
- GitHub: [@fahmiaziz98](https://github.com/fahmiaziz98)
- Email: fahmiazizfadhil09@gmail.com