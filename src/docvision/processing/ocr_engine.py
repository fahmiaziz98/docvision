import asyncio
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

LANGUAGE_MAP = {
    "english": "english",
    "latin": "latin",  # covers Indonesian, Spanish, French, Portuguese, etc.
    "chinese": "chinese",
    "korean": "korean",
    "arabic": "arabic",
    "hindi": "hindi",
    "tamil": "tamil",
    "telugu": "telugu",
}

SUPPORTED_LANGUAGES = sorted(LANGUAGE_MAP.keys())

_HF_REPO = "monkt/paddleocr-onnx"
_DET_PATH = "detection/v5/det.onnx"
_REC_TEMPLATE = "languages/{lang}/rec.onnx"
_DICT_TEMPLATE = "languages/{lang}/dict.txt"


class OCREngine:
    """
    PaddleOCR ONNX inference engine.

    Wraps text detection + recognition into a single `recognize()` call.
    Models are downloaded from HuggingFace on first use and cached locally.

    Usage:
        engine = OCREngine(language="english")
        text = await engine.recognize(image)   # image: BGR numpy array
    """

    def __init__(
        self,
        language: str = "english",
        ocr_model_dir: Optional[Union[str, Path]] = None,
        enable_deskew: bool = True,
    ):
        """
        Initialize the OCR engine.

        Models are NOT downloaded here — lazy init on first recognize() call.

        Args:
            language: OCR language. One of: english, latin, chinese, korean,
                      arabic, hindi, tamil, telugu. Default 'english'.
                      Use 'latin' for Indonesian, Spanish, French, etc.
            ocr_model_dir: Custom directory for model files.
                           Defaults to ~/.cache/docvision/models/
            enable_deskew: Passed through for reference; actual deskew is
                           handled by ImageProcessor before this engine runs.
        """
        if language not in LANGUAGE_MAP:
            raise ValueError(
                f"Unsupported language: '{language}'. Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        self.language = language
        self.enable_deskew = enable_deskew
        self._model_dir = (
            Path(ocr_model_dir)
            if ocr_model_dir
            else Path.home() / ".cache" / "docvision" / "models"
        )

        self._ocr = None

    async def recognize(self, image: np.ndarray) -> str:
        """
        Run OCR on a preprocessed BGR image.

        Downloads models on first call if not cached.

        Args:
            image: Preprocessed BGR numpy array (output of ImageProcessor.preprocess_for_ocr).

        Returns:
            Extracted text as a single string, lines joined by newlines.
            Returns empty string if no text detected.
        """
        ocr = await asyncio.to_thread(self._get_or_init_ocr)
        result = await asyncio.to_thread(self._run_inference, ocr, image)
        return result

    def _get_or_init_ocr(self):
        """Initialize RapidOCR with ONNX models, downloading if necessary."""
        if self._ocr is not None:
            return self._ocr

        det_path, rec_path, dict_path = self._ensure_models()
        self._ocr = self._build_ocr(det_path, rec_path, dict_path)
        return self._ocr

    def _ensure_models(self):
        """
        Check if models are cached locally. Download from HuggingFace if not.

        Returns:
            Tuple of (det_model_path, rec_model_path, dict_path).
        """
        lang_key = LANGUAGE_MAP[self.language]

        det_path = self._model_dir / "v5" / "det.onnx"
        rec_path = self._model_dir / "languages" / lang_key / "rec.onnx"
        dict_path = self._model_dir / "languages" / lang_key / "dict.txt"

        if not det_path.exists():
            print("[docvision] Downloading detection model... (~84 MB)")
            self._hf_download(_DET_PATH, det_path)

        if not rec_path.exists():
            print(f"[docvision] Downloading '{self.language}' recognition model... (~7–10 MB)")
            self._hf_download(_REC_TEMPLATE.format(lang=lang_key), rec_path)

        if not dict_path.exists():
            print(f"[docvision] Downloading '{self.language}' dictionary... (~1 MB)")
            self._hf_download(_DICT_TEMPLATE.format(lang=lang_key), dict_path)

        return det_path, rec_path, dict_path

    def _hf_download(self, hf_file_path: str, local_path: Path) -> None:
        """
        Download a single file from HuggingFace Hub to local_path.

        Args:
            hf_file_path: Path within the HF repo (e.g. 'detection/v5/det.onnx').
            local_path: Where to save the file locally.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for model auto-download. "
                "Install with: pip install huggingface-hub"
            )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = hf_hub_download(
            repo_id=_HF_REPO,
            filename=hf_file_path,
            local_dir=self._model_dir,
            local_dir_use_symlinks=False,
        )

        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != local_path.resolve():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.replace(local_path)

    def _build_ocr(self, det_path: Path, rec_path: Path, dict_path: Path):
        """
        Build a RapidOCR instance with the given ONNX model paths.

        Args:
            det_path: Path to detection model (.onnx).
            rec_path: Path to recognition model (.onnx).
            dict_path: Path to character dictionary (.txt).

        Returns:
            Configured RapidOCR instance.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "rapidocr-onnxruntime is required for BASIC_OCR mode. "
                "Install with: pip install rapidocr-onnxruntime"
            )

        return RapidOCR(
            det_model_path=str(det_path),
            rec_model_path=str(rec_path),
            rec_keys_path=str(dict_path),
            use_angle_cls=False,
        )

    def _run_inference(self, ocr, image: np.ndarray) -> str:
        """
        Run detection + recognition and collect text lines.

        RapidOCR returns: List[Tuple[box_coords, text, confidence]] | None

        Args:
            ocr: Initialized RapidOCR instance.
            image: Preprocessed BGR numpy array.

        Returns:
            Extracted text with lines joined by newlines.
        """
        result, _ = ocr(image)

        if not result:
            return ""

        # Sort by vertical position (top-to-bottom reading order).
        # Each box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — use y of top-left corner.
        def _top_y(item):
            try:
                return float(item[0][0][1])
            except (IndexError, TypeError):
                return 0.0

        lines: List[str] = []
        for item in sorted(result, key=_top_y):
            text = item[1] if len(item) > 1 else ""
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """
        Remove all downloaded model files from the local cache.

        Useful when switching languages or updating models.
        Does not affect models in a custom ocr_model_dir.
        """
        import shutil

        default_cache = Path.home() / ".cache" / "docvision" / "models"
        if self._model_dir == default_cache and default_cache.exists():
            shutil.rmtree(default_cache)
            print(f"[docvision] Model cache cleared: {default_cache}")
        else:
            print(
                "[docvision] clear_cache() only removes the default cache. "
                f"Remove manually: {self._model_dir}"
            )

    @property
    def model_dir(self) -> Path:
        """Path where models are stored."""
        return self._model_dir

    @property
    def is_initialized(self) -> bool:
        """Whether the OCR engine has been initialized (models loaded into memory)."""
        return self._ocr is not None

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "lazy (not yet loaded)"
        return (
            f"OCREngine(language='{self.language}', model_dir='{self._model_dir}', status={status})"
        )
