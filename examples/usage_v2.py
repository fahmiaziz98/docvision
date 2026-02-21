import asyncio
import os
from typing import List

from pydantic import BaseModel
from docvision import DocumentParser, ParsingMode

# Configuration
system_prompt = """
You are a financial analyst.
Extract structured information from the document.
"""

class Items(BaseModel):
    name: str
    quantity: int
    price: float
    amount: float

class Invoice(BaseModel):
    number: str
    date: str
    items: List[Items]
    tax: float
    total: float

# Initialize Parser
parser = DocumentParser(
    vlm_base_url="https://api.openai.com/v1", # Or Groq, OpenRouter, etc.
    vlm_model="gpt-4o-mini",
    vlm_api_key=os.getenv("OPENAI_API_KEY", "your_api_key"),
    max_tokens=4096,
    temperature=0.1,
    chart_description=False,
    enable_rotate=True,
    debug_dir="./debug_image",
)

metadata = {
    "project": "Document Analysis",
    "batch": "2024-Q1",
}

async def run_examples():
    # Example 1: Parse PDF in Agentic Mode (Self-correcting)
    print("--- Example 1: Agentic PDF Parsing ---")
    results = await parser.parse_pdf(
        "example.pdf", # Replace with your PDF path
        start_page=1,
        end_page=2,
        save_path="./output/results.md",
        metadata=metadata,
        parsing_mode=ParsingMode.AGENTIC,
    )
    
    for page in results:
        print(f"Page {page.metadata.get('page_number')}: {page.content[:200]}...")

    # Example 2: Structured Output from Image
    print("\n--- Example 2: Structured Output ---")
    try:
        result = await parser.parse_image(
            "invoice.png", # Replace with your image path
            output_schema=Invoice,
            save_path="./output/invoice.json"
        )
        print(f"Extracted Invoice #: {result.content.number}")
    except Exception as e:
        print(f"Image parsing skipped or failed: {e}")

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./debug_image", exist_ok=True)
    
    asyncio.run(run_examples())
