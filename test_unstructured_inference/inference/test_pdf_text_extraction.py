import os

from unstructured_inference.inference.pdf_text_extraction import (
    extract_pdf_text_layouts,
)


def test_extract_pdf_text_layouts_returns_text():
    sample_pdf = os.path.join("sample-docs", "loremipsum.pdf")
    layouts = extract_pdf_text_layouts(sample_pdf)
    assert layouts, "expected at least one page of text"
    first_page = layouts[0]
    assert first_page.layout is not None
    assert first_page.has_text
