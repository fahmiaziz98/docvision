import os
import asyncio
from pathlib import Path

from docvision import DocumentParsingAgent, ParserConfig, ParsingMode

system_prompt = """You are a document extraction assistant. Extract all text from the document preserving the original structure and formatting. Use markdown format for the output."""

user_prompt = """Please extract all text from this image, maintaining the original structure and formatting."""


async def main():
    
    config = ParserConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        timeout=300.0,
        temperature=1.0,
        max_tokens=2048,
        # Prompts
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        # Image Processing - Optimized combo
        render_zoom=2.0,  # DPI = 300 → zoom ~2.0
        enable_crop=True,
        enable_auto_rotate=True,
        crop_padding=5,
        crop_ignore_bottom_percent=10.0,
        post_crop_max_size=2048,
        # Debug
        debug_save_path="./test_4",
    )
    
    agent = DocumentParsingAgent(config=config)

    # Parse PDF
    # pdf_path = Path("ANTM_801_821.pdf") 
    pdf_path = Path("ADRO_390_410.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"file {pdf_path} doesnt exist")

    start_pg = 5
    end_pg = 8
    print(f"Parsing PDF: {pdf_path}")
    print(f"Mode: AGENTIC (self-correcting)")
    print(f"Processing pages {start_pg}-{end_pg} ({end_pg - start_pg + 1} pages)...\n")

    result = await agent.aparse_pdf(
        pdf_path,
        mode=ParsingMode.AGENTIC,
        start_page=start_pg,
        end_page=end_pg,
    )

    print(f"Successfully parsed {result.total_pages} pages")
    print(f"Total processing time: {result.total_time:.2f}s\n")

    # Save to file
    output_file = Path("./output_custom.md")
    with output_file.open("w", encoding="utf-8") as f:
        for page_result in result.results:
            f.write(f"# Page {page_result.page_number}\n\n")
            f.write(page_result.content)
            f.write("\n\n---\n\n")

    print(f"Results saved to: {output_file}")
    print(f"Debug images saved to: ./test_4/")


if __name__ == "__main__":
    asyncio.run(main())
