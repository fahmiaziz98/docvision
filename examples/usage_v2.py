import asyncio
from typing import List

from pydantic import BaseModel

from docvision import DocumentParser, ParsingMode

system_prompt = """
You are a financial analyst.
You are given a invoice image and you need to extract the items from the invoice.
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


parser = DocumentParser(
    vlm_base_url="https://api.groq.com/openai/v1",
    vlm_model="meta-llama/llama-4-maverick-17b-128e-instruct",
    vlm_api_key="you_api_key",
    max_tokens=4096,
    temperature=1.5,
    chart_description=False,
    enable_rotate=True,
    # system_prompt=system_prompt,
    debug_dir="./debug_image",
)

metadata = {
    "company_name": "PT Alamtri resources Indonesia",
    "year": 2024,
    "keyword": ["financial statement", "annual report", "ADRO"],
}


async def run():
    results = await parser.parse_pdf(
        "ADRO.pdf",
        start_page=407,
        end_page=408,
        save_path="./output/ADRO_407_408.md",
        metadata=metadata,
        parsing_mode=ParsingMode.AGENTIC,
    )
    for i in range(len(results)):
        print(f"Page {i + 1}")
        print(results[i].content)
        print("=" * 100)

    # results = await parser.parse_image(
    #     "complex_invoice.png",
    #     save_path="./output/complex_invoice.json",
    #     output_schema=Invoice
    # )
    # print(results)


if __name__ == "__main__":
    asyncio.run(run())
