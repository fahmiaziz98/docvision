# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - Unreleased

### ⚠️ Breaking Changes

- `ParsingMode.PDF` has been **removed**. Use `ParsingMode.BASIC_OCR` instead.
  ```python
  # Before (v0.2.x):
  results = await parser.parse_pdf("doc.pdf", parsing_mode=ParsingMode.PDF)

  # After (v0.3.0):
  results = await parser.parse_pdf("doc.pdf", parsing_mode=ParsingMode.BASIC_OCR)
  ```

- `NativePDFParser` has been removed from the public API.
  All PDF parsing now goes through the image pipeline (render → OCR or VLM).

- `ContentCropper` has been removed.

- `NativeParseResult` and `PageBlock` types have been removed.

- `CONTINUE_PROMPT`, `FIX_PROMPT` constants have been removed.
  These were internal to the old agentic loop and are replaced by the reflect pattern.

- `detect_retention_loop`, `has_complete_transcription`, `check_max_tokens_hit`
  utility functions have been removed. Handled internally by the critic agent.

### Added

- **`ParsingMode.BASIC_OCR`**: New OCR-based parsing mode powered by PaddleOCR ONNX.
  No PyTorch required. Models are auto-downloaded on first use (~92MB).
  ```python
  parser = DocumentParser(ocr_language="english")
  results = await parser.parse_pdf("doc.pdf", parsing_mode=ParsingMode.BASIC_OCR)
  ```

- **`ocr_language` parameter** on `DocumentParser`: Select OCR language.
  Supported: `english`, `latin` (covers Indonesian, French, German, Spanish, etc.),
  `chinese`, `korean`, `arabic`, `hindi`, `tamil`, `telugu`. Default: `english`.

- **`ocr_model_dir` parameter** on `DocumentParser`: Override auto-download with
  a custom local model directory (useful for offline/production environments).

- **`enable_deskew` parameter** on `DocumentParser`: Correct small skew angles
  before OCR. Default: `True`.

- **`max_reflect_cycles` parameter** on `DocumentParser`: Control critic/refine
  cycles in AGENTIC mode. Default: `2`. Values above `2` emit a `UserWarning`.

- **Reflect pattern in AGENTIC mode**: Generator → Critic → Refiner workflow
  replaces the old repetition-detection loop. The critic evaluates structural
  completeness (not content accuracy) and returns a structured score + issues list.

- **`reflect_iterations` and `final_critic_score`** added to `ParseResult.metadata`
  when using `ParsingMode.AGENTIC`.

- **`CHANGELOG.md`**: This file.

### Changed

- PDF page counting now uses `fitz` (PyMuPDF) instead of `pdfplumber`,
  removing the `pdfplumber` dependency entirely.

- `ImageProcessor.process_image()` replaced by two intentional methods:
  - `preprocess_for_ocr(image, enable_deskew=True)`
  - `preprocess_for_vlm(image, page_num=None)`

- `AgenticParseState` has three new fields: `critic_score`, `critic_issues`,
  `reflect_iteration`.

### Removed

- `pdfplumber` dependency.
- `NativePDFParser`, `ContentCropper` classes.
- `NativeParseResult`, `PageBlock` dataclasses.
- `detect_retention_loop`, `has_complete_transcription`, `check_max_tokens_hit` utils.
- `CONTINUE_PROMPT`, `FIX_PROMPT` constants.

---

## [0.2.0] - Previous release

- Initial public release with PDF, VLM, and AGENTIC parsing modes.