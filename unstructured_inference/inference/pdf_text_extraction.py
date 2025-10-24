from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTPage, LTTextContainer, LTTextLine
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFInterpreterError

from unstructured_inference.constants import ElementType, Source
from unstructured_inference.inference.elements import Rectangle
from unstructured_inference.inference.layoutelement import LayoutElement, LayoutElements

logger = logging.getLogger(__name__)


@dataclass
class PDFPageText:
    """Container for text extracted from a PDF page."""

    layout: LayoutElements

    @property
    def has_text(self) -> bool:
        return len(self.layout.element_coords) > 0


def extract_pdf_text_layouts(
    filename: str,
    password: Optional[str] = None,
    *,
    laparams: Optional[LAParams] = None,
) -> List[PDFPageText]:
    """Extract text regions from a PDF using pdfminer and return LayoutElements per page."""

    laparams = laparams or LAParams()
    try:
        pages = extract_pages(filename, password=password, laparams=laparams)
    except (PDFTextExtractionNotAllowed, PDFInterpreterError) as exc:
        logger.debug("Unable to extract embedded PDF text: %s", exc)
        return []

    page_text_layouts: List[PDFPageText] = []
    for page_index, page_layout in enumerate(pages):
        if not isinstance(page_layout, LTPage):
            logger.debug(
                "Skipping page %s because pdfminer did not return LTPage (got %s).",
                page_index,
                type(page_layout),
            )
            page_text_layouts.append(_empty_page())
            continue

        text_lines: Sequence[LTTextLine] = tuple(_iter_text_lines(page_layout))
        if not text_lines:
            page_text_layouts.append(_empty_page())
            continue

        layout_elements: List[LayoutElement] = []
        for line in text_lines:
            text = _normalize_text(line.get_text())
            if not text:
                continue
            x0, y0, x1, y1 = line.bbox
            bbox = _pdf_bbox_to_rectangle(page_layout, x0, y0, x1, y1)
            element = LayoutElement(
                text=text,
                type=ElementType.TEXT,
                source=Source.PDF_TEXT,
                bbox=bbox,
            )
            layout_elements.append(element)

        if layout_elements:
            layout = LayoutElements.from_list(layout_elements)
        else:
            layout = _empty_layout()

        page_text_layouts.append(PDFPageText(layout=layout))

    return page_text_layouts


def _iter_text_lines(root: LTTextContainer) -> Iterable[LTTextLine]:
    """Yield LTTextLine instances recursively from the pdfminer layout tree."""
    if isinstance(root, LTTextLine):
        yield root
        return

    if isinstance(root, LTTextContainer):
        for child in root:
            yield from _iter_text_lines(child)  # type: ignore[arg-type]
    else:
        if hasattr(root, "__iter__"):
            for child in root:  # type: ignore[attr-defined]
                yield from _iter_text_lines(child)  # type: ignore[arg-type]


def _normalize_text(raw_text: str) -> str:
    """Strip pdfminer artefacts from extracted text."""
    cleaned = raw_text.replace("\u00a0", " ")
    cleaned = cleaned.replace("\r", "")
    cleaned = cleaned.rstrip("\n")
    cleaned = cleaned.strip()
    return cleaned


def _pdf_bbox_to_rectangle(page: LTPage, x0: float, y0: float, x1: float, y1: float) -> Rectangle:
    """Convert pdfminer coordinates (origin bottom-left) to our Rectangle (origin top-left)."""
    height = page.height
    top = height - y1
    bottom = height - y0
    return Rectangle(x1=x0, y1=top, x2=x1, y2=bottom)


def _empty_page() -> PDFPageText:
    return PDFPageText(layout=_empty_layout())


def _empty_layout() -> LayoutElements:
    return LayoutElements(
        element_coords=np.empty((0, 4), dtype=float),
        texts=np.array([], dtype=object),
        element_probs=np.array([], dtype=float),
        element_class_ids=np.array([], dtype=int),
        element_class_id_map={},
        sources=np.array([], dtype=object),
        text_as_html=np.array([], dtype=object),
        table_as_cells=np.array([], dtype=object),
    )
