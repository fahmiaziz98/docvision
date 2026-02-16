# 📄 DocVision Parser

> Production-ready document parsing framework powered by Vision Language Models (VLMs).

[![Tests](https://github.com/fahmiaziz98/doc-vision-parser/workflows/Tests/badge.svg)](https://github.com/fahmiaziz98/doc-vision-parser/actions)
[![PyPI version](https://badge.fury.io/py/doc-vision-parser.svg)](https://badge.fury.io/py/doc-vision-parser)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

DocVision Parser is a robust Python library designed to extract high-quality structured text and markdown from documents (images and PDFs) using state-of-the-art Vision Language Models like GPT-4o. It goes beyond simple OCR by leveraging the reasoning capabilities of VLMs to understand layout, context, and complex formatting.

The framework supports two primary modes:
1.  **VLM Mode**: Fast, single-shot parsing for standard documents.
2.  **Agentic Mode**: A self-correcting, iterative workflow using sophisticated graph logic to handle token limits, repetition loops, and incomplete outputs, ensuring the highest possible accuracy for complex documents.

## Features

-   **Agentic Workflow**: Self-correcting parsing loop that automatically detects and fixes issues like token truncation and repetitive generation.
-   **Async Support**: Built-in high-throughput asynchronous methods for processing large batches of documents efficiently.
-   **Smart Preprocessing**: Intelligent content-aware cropping, DPI management, and dynamic image optimization to ensure the VLM receives the best possible input.
-   **OpenAI-Compatible**: Designed to work with any OpenAI-compatible API, including standard OpenAI endpoints, Azure OpenAI, and self-hosted models via vLLM or SGLang.
-   **Production-Ready**: Includes robust error handling, automatic retries with exponential backoff, and strict output validation.
-   **Structured Output**: Native support for Pydantic models to extract structured data from documents.

## Installation

Install using `pip`:

```bash
pip install doc-vision-parser
```

Or using `uv` (recommended):

```bash
uv add doc-vision-parser
```

### Requirements

- Python 3.10 or higher
- An OpenAI API key (or compatible endpoint)

## Quick Start

### Basic Usage

The simplest way to parse an image is using the `DocumentParsingAgent` in VLM mode.

```python
import os
from docvision import DocumentParsingAgent, ParserConfig

# Initialize configuration
config = ParserConfig(
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize the agent
agent = DocumentParsingAgent(config=config)

# Parse an image synchronously
result = agent.parse_image("path/to/document.jpg")

print(result.content)
print(f"Processing time: {result.processing_time:.2f}s")
```

## Advanced Usage

### Asynchronous & Batch Processing

For processing PDFs or multiple files efficiently, use the asynchronous efficiency of the agent.

```python
import asyncio
from docvision import DocumentParsingAgent, ParserConfig, ParsingMode

async def main():
    config = ParserConfig(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    agent = DocumentParsingAgent(config=config)

    # Parse a PDF asynchronously
    result = await agent.aparse_pdf(
        "path/to/document.pdf",
        mode=ParsingMode.VLM,
        max_concurrent=3,
        start_page=1,
        end_page=3
    )

    print(f"Processed {result.total_pages} pages in {result.total_time:.2f}s")
    for i in range(len(result.results)):
        md_content = result.results[i].content
        page_num = result.results[i].page_number
        print(f"Page: {page_num}: Content\n{md_content}")
        print("====="*50)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agentic Mode (Self-Correcting)

Use `ParsingMode.AGENTIC` for critical documents where accuracy is paramount. This mode enables the self-correcting workflow that validates output and continues generation if cut off.

```python
# Agentic mode is always async
result = await agent.aparse_image(
    "path/to/complex_document.jpg",
    mode=ParsingMode.AGENTIC
)

# Access metadata about the generation process
print(result.metadata["generation_history"])
print(f"Iterations needed: {result.metadata['iterations']}")
```

### Structured Output with Pydantic

You can force the model to return structured JSON data by providing a Pydantic model.

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    date: str

system_prompt = """
You are a financial analyst. 
Extract the following information from the document: 
    - invoice number, 
    - total amount, 
    - and date.
"""

    config = ParserConfig(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt=system_prompt,  # for structured_output must explicitly define system_prompt
    )
    
    agent = DocumentParsingAgent(config=config)

result = agent.parse_image(
    "invoice.jpg",
    mode=ParsingMode.VLM,
    output_schema=Invoice
)

invoice_data = result.content  # This will be an instance of Invoice
print(invoice_data.total_amount)
```

## Configuration

The `DocumentParsingAgent` is configured via the `ParserConfig` class.

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `base_url` | `str` | `"https://api.openai.com/v1"` | Base URL for the VLM API. |
| `api_key` | `Optional[str]` | `None` | API key for authentication. |
| `model_name` | `str` | `"gpt-4o-mini"` | The VLM model to use. |
| `timeout` | `float` | `300.0` | Request timeout in seconds. |
| `temperature` | `float` | `0.1` | Sampling temperature. |
| `max_tokens` | `int` | `2048` | Maximum tokens for response. |
| `render_zoom` | `float` | `2.0` | PDF render zoom factor (2.0 ≈ 300 DPI). |
| `post_crop_max_size` | `int` | `2048` | Max dimension after cropping. |
| `enable_auto_rotate` | `bool` | `True` | Automatically correct image orientation. |
| `enable_crop` | `bool` | `True` | Enable intelligent content cropping. |
| `crop_padding` | `int` | `10` | Padding around cropped content. |
| `crop_ignore_bottom_percent` | `float` | `12.0` | Percentage of bottom margin to ignore during crop detection. |
| `debug_save_path` | `Optional[str]` | `None` | Path to save debug images (crops, rotations). |

### Comprehensive Configuration Example

For fine-grained control over image processing and model parameters, utilize `ParserConfig` fully:

```python
import os
from docvision import DocumentParsingAgent, ParserConfig, ParsingMode

# Define custom prompts
system_prompt = """You are a document extraction assistant. Extract all text from the document preserving the original structure and formatting. Use markdown format for the output."""

user_prompt = """Please extract all text from this image, maintaining the original structure and formatting."""

# Create advanced configuration
config = ParserConfig(
    # VLM Settings
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.2-90b-vision-preview",
    timeout=300.0,
    temperature=0.1,
    max_tokens=4096,
    
    # Custom Prompts
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    
    # Image Configuration
    render_zoom=3.0,          # High quality rendering (~450 DPI)
    enable_crop=True,         # Auto-crop to content
    enable_auto_rotate=True,  # Fix orientation
    crop_padding=20,          # Add breathing room
    crop_ignore_bottom_percent=10.0, # Ignore footer noise
    post_crop_max_size=3072,  # Allow larger images for the model
    
    # Debugging
    debug_save_path="./debug_output", # Save intermediate images
)

# Initialize agent with custom config
agent = DocumentParsingAgent(config=config)
```

## Architecture

The framework is built on three core components:

1.  **VLMClient**: A reliable wrapper around the OpenAI API that handles connection pooling, retries, and error mapping.
2.  **ImageProcessor**: Handles the visual pipeline, including PDF rendering, smart cropping, and optimization to maximize model potential.
3.  **AgenticWorkflow**: A LangGraph-based state machine that orchestrates the cognitive process of parsing, verifying, and correcting output.

## Development

To set up the development environment:

1.  Install dependencies:
    ```bash
    uv sync --dev
    ```

2.  Run linting and formatting:
    ```bash
    make lint
    make format
    ```

3.  Run tests:
    ```bash
    make test
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Author

**Fahmi Aziz Fadhil**
- GitHub: [@fahmiaziz98](https://github.com/fahmiaziz98)
- Email: fahmiazizfadhil09@gmail.com

---

⭐ If you find this project helpful, please consider giving it a star on GitHub!
