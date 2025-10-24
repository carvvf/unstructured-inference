from __future__ import annotations

import math
import os
import tempfile
from functools import cached_property
from pathlib import PurePath
from typing import Any, BinaryIO, Collection, List, Optional, Tuple, Union, cast

import numpy as np
import pdf2image
from PIL import Image, ImageSequence

from unstructured_inference.inference.elements import (
    TextRegion,
)
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    LayoutElements,
    merge_inferred_layout_with_extracted_layout,
)
from unstructured_inference.inference.pdf_text_extraction import (
    PDFPageText,
    extract_pdf_text_layouts,
)
from unstructured_inference.logger import logger
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import (
    UnstructuredElementExtractionModel,
    UnstructuredObjectDetectionModel,
)
from unstructured_inference.visualize import draw_bbox


_PAGE_ROTATION_ENV_VAR = "UNSTRUCTURED_ENABLE_LAYOUT_PAGE_ROTATION_DETECTION"
_PDF_TEXT_ENV_VAR = "UNSTRUCTURED_USE_PDF_TEXT_EXTRACTION"


def _env_var_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_ENABLE_PAGE_ROTATION_DETECTION = _env_var_to_bool(
    os.environ.get(_PAGE_ROTATION_ENV_VAR),
    default=False,
)

DEFAULT_ENABLE_PDF_TEXT_EXTRACTION = _env_var_to_bool(
    os.environ.get(_PDF_TEXT_ENV_VAR),
    default=False,
)


def _clone_layout_elements(layout: LayoutElements) -> LayoutElements:
    return LayoutElements(
        element_coords=layout.element_coords.copy(),
        texts=layout.texts.copy(),
        element_probs=layout.element_probs.copy(),
        element_class_ids=layout.element_class_ids.copy(),
        element_class_id_map=dict(layout.element_class_id_map),
        sources=layout.sources.copy(),
        text_as_html=layout.text_as_html.copy(),
        table_as_cells=layout.table_as_cells.copy(),
    )


def _rotate_coords_to_original(
    coords: np.ndarray,
    original_size: Tuple[int, int],
    rotated_size: Tuple[int, int],
    angle: float,
) -> np.ndarray:
    if coords.size == 0 or angle % 360 == 0:
        return coords

    theta = math.radians(-angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    orig_w, orig_h = original_size
    rot_w, rot_h = rotated_size
    orig_cx, orig_cy = orig_w / 2.0, orig_h / 2.0
    rot_cx, rot_cy = rot_w / 2.0, rot_h / 2.0

    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    corners = np.stack(
        (
            np.stack((x1, y1), axis=-1),
            np.stack((x1, y2), axis=-1),
            np.stack((x2, y1), axis=-1),
            np.stack((x2, y2), axis=-1),
        ),
        axis=1,
    )

    corners[..., 0] -= rot_cx
    corners[..., 1] -= rot_cy
    transformed = corners @ rotation_matrix.T
    transformed[..., 0] += orig_cx
    transformed[..., 1] += orig_cy

    mins = transformed.min(axis=1)
    maxs = transformed.max(axis=1)

    new_coords = np.stack((mins[:, 0], mins[:, 1], maxs[:, 0], maxs[:, 1]), axis=1)
    new_coords[:, [0, 2]] = np.clip(new_coords[:, [0, 2]], 0, orig_w)
    new_coords[:, [1, 3]] = np.clip(new_coords[:, [1, 3]], 0, orig_h)
    return new_coords


def _rotate_image(image: Image.Image, angle: float) -> Image.Image:
    if angle == 0:
        return image
    return image.rotate(angle, expand=True)


def _detect_layout_with_orientations(
    image: Image.Image,
    detection_model: UnstructuredObjectDetectionModel,
    page_number: Optional[int] = None,
) -> LayoutElements:
    from unstructured_inference.inference.layoutelement import LayoutElements

    def _score(layout: LayoutElements) -> Tuple[int, float]:
        probs = np.nan_to_num(layout.element_probs, nan=0.0)
        return (len(layout), float(probs.sum()))

    def _evaluate(
        angle: float,
        oriented_image: Image.Image,
        direction: str,
    ) -> Tuple[LayoutElements, Tuple[int, float], float, str]:
        layout_raw = detection_model(oriented_image)
        if isinstance(layout_raw, list):
            layout_raw = LayoutElements.from_list(layout_raw)
        layout = _clone_layout_elements(layout_raw)
        if angle != 0:
            layout.element_coords = _rotate_coords_to_original(
                layout.element_coords,
                original_size,
                oriented_image.size,
                angle,
            )
        layout = detection_model.deduplicate_detected_elements(layout)
        return layout, _score(layout), angle, direction

    original_size = image.size

    rotation_candidates: List[Tuple[float, str]] = [
        (0, "original"),
        (-90.0, "clockwise"),
        (90.0, "counterclockwise"),
    ]

    best_layout: LayoutElements | None = None
    best_score: Tuple[int, float] = (-1, float("-inf"))
    best_angle: float = 0.0
    best_direction: str = "original"

    for angle, direction in rotation_candidates:
        oriented = _rotate_image(image, angle)
        layout, score, used_angle, used_direction = _evaluate(angle, oriented, direction)
        if best_layout is None or score > best_score:
            best_layout = layout
            best_score = score
            best_angle = used_angle
            best_direction = used_direction

    if best_layout is None:
        best_layout = LayoutElements(
            element_coords=np.empty((0, 4), dtype=float),
            texts=np.array([], dtype=object),
            element_probs=np.array([], dtype=float),
            element_class_ids=np.array([], dtype=int),
            element_class_id_map={},
            sources=np.array([], dtype=object),
            text_as_html=np.array([], dtype=object),
            table_as_cells=np.array([], dtype=object),
        )

    if best_score[0] == 0 and best_score[1] == 0:
        oriented = _rotate_image(image, 180)
        layout, score, used_angle, used_direction = _evaluate(180, oriented, "upside-down")
        if score > best_score:
            best_layout = layout
            best_score = score
            best_angle = used_angle
            best_direction = used_direction

    if best_direction != "original":
        if best_direction == "clockwise":
            display_angle = 90
        elif best_direction == "counterclockwise":
            display_angle = 90
        elif best_direction == "upside-down":
            display_angle = 180
        else:
            display_angle = int((best_angle + 360) % 360)
        if page_number is not None:
            logger.debug(
                "Layout detection rotated page %s by %s degrees (%s).",
                page_number,
                int(display_angle),
                best_direction,
            )
        else:
            logger.debug(
                "Layout detection rotated page by %s degrees (%s).",
                int(display_angle),
                best_direction,
            )

    return best_layout


class DocumentLayout:
    """Class for handling documents that are saved as .pdf files. For .pdf files, a
    document image analysis (DIA) model detects the layout of the page prior to extracting
    element."""

    def __init__(self, pages=None):
        self._pages = pages

    def __str__(self) -> str:
        return "\n\n".join([str(page) for page in self.pages])

    @property
    def pages(self) -> List[PageLayout]:
        """Gets all elements from pages in sequential order."""
        return self._pages

    @classmethod
    def from_pages(cls, pages: List[PageLayout]) -> DocumentLayout:
        """Generates a new instance of the class from a list of `PageLayouts`s"""
        doc_layout = cls()
        doc_layout._pages = pages
        return doc_layout

    @classmethod
    def from_file(
        cls,
        filename: str,
        fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
        pdf_image_dpi: int = 200,
        password: Optional[str] = None,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")

        enable_page_rotation_detection = kwargs.pop("enable_page_rotation_detection", None)
        enable_pdf_text_extraction = kwargs.pop("enable_pdf_text_extraction", None)
        if enable_pdf_text_extraction is None:
            enable_pdf_text_extraction = DEFAULT_ENABLE_PDF_TEXT_EXTRACTION

        pdf_text_layouts: list[PDFPageText] = []
        if enable_pdf_text_extraction:
            pdf_text_layouts = extract_pdf_text_layouts(filename, password=password)

        with tempfile.TemporaryDirectory() as temp_dir:
            _image_paths = convert_pdf_to_image(
                filename,
                pdf_image_dpi,
                output_folder=temp_dir,
                path_only=True,
                password=password,
            )
            image_paths = cast(List[str], _image_paths)
            number_of_pages = len(image_paths)
            pages: List[PageLayout] = []
            if fixed_layouts is None:
                fixed_layouts = [None for _ in range(0, number_of_pages)]
            for i, (image_path, fixed_layout) in enumerate(zip(image_paths, fixed_layouts)):
                pdf_text_layout = (
                    pdf_text_layouts[i].layout
                    if enable_pdf_text_extraction and i < len(pdf_text_layouts)
                    and pdf_text_layouts[i].has_text
                    else None
                )
                # NOTE(robinson) - In the future, maybe we detect the page number and default
                # to the index if it is not detected
                with Image.open(image_path) as image:
                    page = PageLayout.from_image(
                        image,
                        number=i + 1,
                        document_filename=filename,
                        fixed_layout=fixed_layout,
                        pdf_text_layout=pdf_text_layout,
                        enable_page_rotation_detection=enable_page_rotation_detection,
                        **kwargs,
                    )
                    pages.append(page)
            return cls.from_pages(pages)

    @classmethod
    def from_image_file(
        cls,
        filename: str,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        fixed_layout: Optional[List[TextRegion]] = None,
        **kwargs,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")

        enable_page_rotation_detection = kwargs.pop("enable_page_rotation_detection", None)
        # This option only applies to PDFs but may be passed through process_file_with_model.
        kwargs.pop("enable_pdf_text_extraction", None)

        try:
            image = Image.open(filename)
            format = image.format
            images: list[Image.Image] = []
            for i, im in enumerate(ImageSequence.Iterator(image)):
                im = im.convert("RGB")
                im.format = format
                images.append(im)
        except Exception as e:
            if os.path.isdir(filename) or os.path.isfile(filename):
                raise e
            else:
                raise FileNotFoundError(f'File "{filename}" not found!') from e
        pages = []
        for i, image in enumerate(images):  # type: ignore
            page = PageLayout.from_image(
                image,
                image_path=filename,
                number=i,
                detection_model=detection_model,
                element_extraction_model=element_extraction_model,
                fixed_layout=fixed_layout,
                enable_page_rotation_detection=enable_page_rotation_detection,
                **kwargs,
            )
            pages.append(page)
        return cls.from_pages(pages)


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image.Image,
        image_metadata: Optional[dict] = None,
        image_path: Optional[Union[str, PurePath]] = None,  # TODO: Deprecate
        document_filename: Optional[Union[str, PurePath]] = None,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        password: Optional[str] = None,
        enable_page_rotation_detection: Optional[bool] = None,
        fixed_layout: Optional[List[TextRegion]] = None,
        pdf_text_layout: Optional[LayoutElements] = None,
    ):
        if detection_model is not None and element_extraction_model is not None:
            raise ValueError("Only one of detection_model and extraction_model should be passed.")
        self.image: Optional[Image.Image] = image
        if image_metadata is None:
            image_metadata = {}
        self.image_metadata = image_metadata
        self.image_path = image_path
        self.image_array: Union[np.ndarray[Any, Any], None] = None
        self.document_filename = document_filename
        self.number = number
        self.detection_model = detection_model
        self.element_extraction_model = element_extraction_model
        self.elements_array: LayoutElements | None = None
        self.password = password
        if enable_page_rotation_detection is None:
            enable_page_rotation_detection = DEFAULT_ENABLE_PAGE_ROTATION_DETECTION
        self.enable_page_rotation_detection = enable_page_rotation_detection
        self.fixed_layout = fixed_layout
        self.pdf_text_layout = pdf_text_layout
        # NOTE(alan): Dropped LocationlessLayoutElement that was created for chipper - chipper has
        # locations now and if we need to support LayoutElements without bounding boxes we can make
        # the bbox property optional

    def __str__(self) -> str:
        return "\n\n".join([str(element) for element in self.elements])

    @cached_property
    def elements(self) -> Collection[LayoutElement]:
        """return a list of layout elements from the array data structure; intended for backward
        compatibility"""
        if self.elements_array is None:
            return []
        return self.elements_array.as_list()

    def get_elements_using_image_extraction(
        self,
        inplace=True,
    ) -> Optional[list[LayoutElement]]:
        """Uses end-to-end text element extraction model to extract the elements on the page."""
        if self.element_extraction_model is None:
            raise ValueError(
                "Cannot get elements using image extraction, no image extraction model defined",
            )
        assert self.image is not None
        elements = self.element_extraction_model(self.image)
        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_with_detection_model(
        self,
        inplace: bool = True,
    ) -> Optional[LayoutElements]:
        """Uses specified model to detect the elements on the page."""
        if self.detection_model is None:
            model = get_model()
            if isinstance(model, UnstructuredObjectDetectionModel):
                self.detection_model = model
            else:
                raise NotImplementedError("Default model should be a detection model")

        # NOTE(mrobinson) - We'll want make this model inference step some kind of
        # remote call in the future.
        assert self.image is not None
        if self.enable_page_rotation_detection:
            inferred_layout = _detect_layout_with_orientations(
                self.image,
                self.detection_model,
                page_number=self.number,
            )
        else:
            layout_raw = self.detection_model(self.image)
            if isinstance(layout_raw, list):
                layout_raw = LayoutElements.from_list(layout_raw)
            inferred_layout = _clone_layout_elements(layout_raw)
            inferred_layout = self.detection_model.deduplicate_detected_elements(
                inferred_layout,
            )

        extracted_regions: List[TextRegion] = []
        if self.fixed_layout:
            extracted_regions.extend(self.fixed_layout)
        if self.pdf_text_layout is not None and len(self.pdf_text_layout.element_coords):
            extracted_regions.extend(self.pdf_text_layout.as_list())

        if extracted_regions:
            page_width, page_height = self.image.size  # type: ignore[union-attr]
            merged_layout = merge_inferred_layout_with_extracted_layout(
                inferred_layout.as_list(),
                extracted_regions,
                (page_width, page_height),
            )
            inferred_layout = LayoutElements.from_list(merged_layout)

        if inplace:
            self.elements_array = inferred_layout
            return None

        return inferred_layout

    def _get_image_array(self) -> Union[np.ndarray[Any, Any], None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            if self.image:
                self.image_array = np.array(self.image)
            else:
                image = Image.open(self.image_path)  # type: ignore
                self.image_array = np.array(image)
        return self.image_array

    def annotate(
        self,
        colors: Optional[Union[List[str], str]] = None,
        image_dpi: int = 200,
        annotation_data: Optional[dict[str, dict]] = None,
        add_details: bool = False,
        sources: Optional[List[str]] = None,
    ) -> Image.Image:
        """Annotates the elements on the page image.
        if add_details is True, and the elements contain type and source attributes, then
        the type and source will be added to the image.
        sources is a list of sources to annotate. If sources is ["all"], then all sources will be
        annotated. Current sources allowed are "yolox","detectron2_onnx" and "detectron2_lp" """
        if colors is None:
            colors = ["red" for _ in self.elements]
        if isinstance(colors, str):
            colors = [colors]
        # If there aren't enough colors, just cycle through the colors a few times
        if len(colors) < len(self.elements):
            n_copies = (len(self.elements) // len(colors)) + 1
            colors = colors * n_copies

        # Hotload image if it hasn't been loaded yet
        if self.image:
            img = self.image.copy()
        elif self.image_path:
            img = Image.open(self.image_path)
        else:
            img = self._get_image(self.document_filename, self.number, image_dpi)

        if annotation_data is None:
            for el, color in zip(self.elements, colors):
                if sources is None or el.source in sources:
                    img = draw_bbox(img, el, color=color, details=add_details)
        else:
            for attribute, style in annotation_data.items():
                if hasattr(self, attribute) and getattr(self, attribute):
                    color = style["color"]
                    width = style["width"]
                    for region in getattr(self, attribute):
                        required_source = getattr(region, "source", None)
                        if (sources is None) or (required_source in sources):
                            img = draw_bbox(
                                img,
                                region,
                                color=color,
                                width=width,
                                details=add_details,
                            )

        return img

    def _get_image(self, filename, page_number, pdf_image_dpi: int = 200) -> Image.Image:
        """Hotloads a page image from a pdf file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            _image_paths = pdf2image.convert_from_path(
                filename,
                dpi=pdf_image_dpi,
                output_folder=temp_dir,
                paths_only=True,
            )
            image_paths = cast(List[str], _image_paths)
            if page_number > len(image_paths):
                raise ValueError(
                    f"Page number {page_number} is greater than the number of pages in the PDF.",
                )

            with Image.open(image_paths[page_number - 1]) as image:
                return image.copy()

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        image_path: Optional[Union[str, PurePath]] = None,
        document_filename: Optional[Union[str, PurePath]] = None,
        number: int = 1,
        detection_model: Optional[UnstructuredObjectDetectionModel] = None,
        element_extraction_model: Optional[UnstructuredElementExtractionModel] = None,
        fixed_layout: Optional[List[TextRegion]] = None,
        enable_page_rotation_detection: Optional[bool] = None,
        pdf_text_layout: Optional[LayoutElements] = None,
        **kwargs,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""

        page = cls(
            number=number,
            image=image,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            enable_page_rotation_detection=enable_page_rotation_detection,
            fixed_layout=fixed_layout,
            pdf_text_layout=pdf_text_layout,
        )
        # FIXME (yao): refactor the other methods so they all return elements like the third route
        if page.element_extraction_model is not None:
            page.get_elements_using_image_extraction()
        else:
            page.get_elements_with_detection_model()

        if (
            page.elements_array is None
            and page.pdf_text_layout is not None
            and len(page.pdf_text_layout.texts)
        ):
            page.elements_array = page.pdf_text_layout

        page.image_metadata = {
            "format": page.image.format if page.image else None,
            "width": page.image.width if page.image else None,
            "height": page.image.height if page.image else None,
        }
        page.image_path = os.path.abspath(image_path) if image_path else None
        page.document_filename = os.path.abspath(document_filename) if document_filename else None

        # Clear the image to save memory
        page.image = None

        return page


def process_data_with_model(
    data: BinaryIO,
    model_name: Optional[str],
    password: Optional[str] = None,
    **kwargs: Any,
) -> DocumentLayout:
    """Process PDF or image as file-like object `data` into a `DocumentLayout`.

    Uses the model identified by `model_name`.
    """
    # Note: We use a temp dir, not a temp file,
    # because the latter fails on Windows
    # https://github.com/Unstructured-IO/unstructured-inference/pull/376
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        file_path = os.path.join(tmp_dir_path, "document")
        with open(file_path, "wb") as f:
            f.write(data.read())
            f.flush()
        layout = process_file_with_model(
            file_path,
            model_name,
            password=password,
            **kwargs,
        )

    return layout


def process_file_with_model(
    filename: str,
    model_name: Optional[str],
    is_image: bool = False,
    fixed_layouts: Optional[List[Optional[List[TextRegion]]]] = None,
    pdf_image_dpi: int = 200,
    password: Optional[str] = None,
    **kwargs: Any,
) -> DocumentLayout:
    """Processes pdf or image file with name filename into a DocumentLayout by using
    a model identified by model_name."""

    enable_page_rotation_detection = kwargs.pop("enable_page_rotation_detection", None)
    enable_pdf_text_extraction = kwargs.pop("enable_pdf_text_extraction", None)
    model = get_model(model_name, **kwargs)
    if isinstance(model, UnstructuredObjectDetectionModel):
        detection_model = model
        element_extraction_model = None
    elif isinstance(model, UnstructuredElementExtractionModel):
        detection_model = None
        element_extraction_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    layout_kwargs = dict(kwargs)
    if enable_page_rotation_detection is not None:
        layout_kwargs["enable_page_rotation_detection"] = enable_page_rotation_detection
    if enable_pdf_text_extraction is not None:
        layout_kwargs["enable_pdf_text_extraction"] = enable_pdf_text_extraction

    layout = (
        DocumentLayout.from_image_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            **layout_kwargs,
        )
        if is_image
        else DocumentLayout.from_file(
            filename,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            fixed_layouts=fixed_layouts,
            pdf_image_dpi=pdf_image_dpi,
            password=password,
            **layout_kwargs,
        )
    )
    return layout


def convert_pdf_to_image(
    filename: str,
    dpi: int = 200,
    output_folder: Optional[Union[str, PurePath]] = None,
    path_only: bool = False,
    password: Optional[str] = None,
) -> Union[List[Image.Image], List[str]]:
    """Get the image renderings of the pdf pages using pdf2image"""

    if path_only and not output_folder:
        raise ValueError("output_folder must be specified if path_only is true")

    if output_folder is not None:
        images = pdf2image.convert_from_path(
            filename,
            dpi=dpi,
            output_folder=output_folder,
            paths_only=path_only,
            userpw=password or "",
        )
    else:
        images = pdf2image.convert_from_path(
            filename,
            dpi=dpi,
            paths_only=path_only,
            userpw=password or "",
        )

    return images
