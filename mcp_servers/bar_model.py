"""Bar model visualization library for Singapore Math problems.

This module provides classes and functions to create bar model diagrams,
a visual representation technique commonly used in Singapore primary school
mathematics to solve word problems.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# --- Drawing Constants ---
RECT_COLOR: Tuple[int, int, int] = (0, 0, 0)
LINE_COLOR: Tuple[int, int, int] = (0, 0, 0)
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)

# --- Brace Constants ---
BRACE_ARM_LENGTH: int = 10
BRACE_ARROW_SIZE: int = 5
MAX_BRACE_LEVELS: int = 5


def start_canvas(width: int, height: int, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Create a new canvas with specified dimensions and background color.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        bg_color: Background color as RGB tuple (default: white).

    Returns:
        A numpy array representing the canvas.
    """
    return np.full((height, width, 3), bg_color, dtype=np.uint8)


def save_image(image: np.ndarray, file_path: str) -> None:
    """Save the image to specified file path.

    Args:
        image: The image array to save.
        file_path: Destination file path.
    """
    cv2.imwrite(file_path, image)


def calculate_font_thickness(font_size: float, base_thickness: int = 2) -> int:
    """Calculate font thickness based on font size with a constant multiplier.

    Args:
        font_size: The font scale/size.
        base_thickness: Base thickness multiplier (default: 2).

    Returns:
        Calculated thickness value, minimum 1.
    """
    return max(1, int(base_thickness * (font_size / 0.7)))


def get_text_dimensions(text: str, font_scale: float, thickness: int) -> Tuple[int, int]:
    """Get text width and height for given font parameters.

    Args:
        text: The text string to measure.
        font_scale: Font scale factor.
        thickness: Font thickness.

    Returns:
        Tuple of (width, height) in pixels.
    """
    if not text:
        return 0, 0
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    return text_w, text_h


class LayoutConfig:
    """Configuration class for all layout parameters and gaps."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize layout configuration.

        Args:
            config: Optional dictionary of custom configuration values.
        """
        # Default configuration values
        self.defaults: Dict[str, Any] = {
            # Step 5: Margin configurations
            'brace_to_bar_gap': 8,           # Small distance from level 1 brace to bar
            'v_gap_between_bars': 20,        # Vertical gap between bars
            'bar_label_to_bar_gap': 15,      # Gap between bar label and bar
            'brace_to_label_gap': 30,        # Gap between brace and brace label
            'brace_level_gap': 5,            # Gap between sets of [brace, brace label]

            # Canvas margins
            'left_margin_ratio': 0.1,
            'right_margin_ratio': 0.1,
            'top_margin_ratio': 0.15,
            'bottom_margin_ratio': 0.15,

            # Bar sizing
            'max_bar_height_ratio': 0.2,     # Maximum bar height as ratio of canvas height
            'font_size_ratio': 0.35,         # Step 2: Font size as percentage of bar height

            # Layout ratios
            'bar_label_width_ratio': 0.12
        }

        # Update with provided config
        if config:
            self.defaults.update(config)

    def get(self, key: str) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key name.

        Returns:
            Configuration value, or 0 if key not found.
        """
        return self.defaults.get(key, 0)


class BraceLevelManager:
    """Manages brace level assignment to prevent overlapping (Step 4)."""

    def __init__(self, max_levels: int = MAX_BRACE_LEVELS) -> None:
        """Initialize brace level manager.

        Args:
            max_levels: Maximum number of brace levels to support.
        """
        self.max_levels: int = max_levels
        self.h_brace_levels: Dict[str, List[List[Tuple[int, int]]]] = {'top': [], 'bottom': []}
        self.v_brace_levels: List[List[Tuple[int, int]]] = []

    def find_h_brace_level(self, start_x: int, end_x: int, location: str) -> int:
        """Find available level for horizontal brace to prevent overlap.

        Args:
            start_x: Starting x-coordinate.
            end_x: Ending x-coordinate.
            location: Brace location ('top' or 'bottom').

        Returns:
            Available level index (0-indexed).
        """
        if location not in self.h_brace_levels:
            self.h_brace_levels[location] = []

        levels = self.h_brace_levels[location]

        # Ensure we have enough level lists
        while len(levels) < self.max_levels:
            levels.append([])

        # Find first available level
        for level in range(self.max_levels):
            if not self._ranges_overlap(start_x, end_x, levels[level]):
                levels[level].append((start_x, end_x))
                return level

        # If no level available, use the last one
        levels[-1].append((start_x, end_x))
        return self.max_levels - 1

    def find_v_brace_level(self, start_y: int, end_y: int) -> int:
        """Find available level for vertical brace to prevent overlap.

        Args:
            start_y: Starting y-coordinate.
            end_y: Ending y-coordinate.

        Returns:
            Available level index (0-indexed).
        """
        # Ensure we have enough level lists
        while len(self.v_brace_levels) < self.max_levels:
            self.v_brace_levels.append([])

        # Find first available level
        for level in range(self.max_levels):
            if not self._ranges_overlap(start_y, end_y, self.v_brace_levels[level]):
                self.v_brace_levels[level].append((start_y, end_y))
                return level

        # If no level available, use the last one
        self.v_brace_levels[-1].append((start_y, end_y))
        return self.max_levels - 1

    def _ranges_overlap(self, start: int, end: int, existing_ranges: List[Tuple[int, int]]) -> bool:
        """Check if new range overlaps with any existing ranges.

        Args:
            start: Start of new range.
            end: End of new range.
            existing_ranges: List of existing ranges.

        Returns:
            True if overlap exists, False otherwise.
        """
        for ex_start, ex_end in existing_ranges:
            if max(start, ex_start) < min(end, ex_end):
                return True
        return False


class Bar:
    """Represents a single bar with segments and horizontal braces.

    Follows the drawing steps 10-13 from instructions.
    """

    def __init__(
        self,
        canvas: np.ndarray,
        bar_data: Dict[str, Any],
        layout: Dict[str, Any],
        config: LayoutConfig
    ) -> None:
        """Initialize a Bar object.

        Args:
            canvas: The canvas to draw on.
            bar_data: Dictionary containing bar data:
                'segments': List of tuples (length, label).
                'h_braces': List of tuples (start_idx, end_idx, label, location).
                'label': The label for the entire bar.
            layout: Dictionary with layout parameters.
            config: LayoutConfig object.
        """
        self.canvas: np.ndarray = canvas
        self.segments_data: List[Tuple[float, str]] = bar_data.get('segments', [])
        self.h_braces_data: List[Tuple[int, int, str, str]] = bar_data.get('h_braces', [])
        self.bar_label: str = bar_data.get('label', '')
        self.layout: Dict[str, Any] = layout
        self.config: LayoutConfig = config

        # Bar positioning and dimensions
        self.rect_x: int = layout['rect_x']
        self.rect_y: int = layout['rect_y']
        self.rect_w: int = layout['rect_w']
        self.rect_h: int = layout['rect_h']

        # Font settings
        self.base_font_scale: float = layout['base_font_scale']
        self.font_thickness: int = calculate_font_thickness(self.base_font_scale)

        # Segment boundaries for brace calculations
        self.segment_boundaries: List[int] = []

        # Brace level manager
        self.brace_manager: BraceLevelManager = BraceLevelManager()

        # Execute drawing steps
        self._draw_bar_elements()

    def _draw_bar_elements(self) -> None:
        """Execute drawing steps 10-13: bar label, bar, segments, h-braces."""
        # Step 10: Draw bar labels (leftmost elements)
        self._draw_bar_label()

        # Step 11: Draw bars with proper positioning
        self._draw_main_bar()

        # Step 12: Draw segments and segment labels with font resizing
        self._draw_segments()

        # Step 13: Draw horizontal braces with labels and font resizing
        self._draw_h_braces()

    def _draw_bar_label(self) -> None:
        """Step 10: Draw bar label as leftmost element."""
        if not self.bar_label:
            return

        label_x = self.layout['bar_label_x']
        label_y = self.rect_y + self.rect_h // 2

        # Use base font scale for consistency with other text elements
        # Only scale down if absolutely necessary to fit
        max_width = self.layout['bar_label_width'] - self.config.get('bar_label_to_bar_gap')
        fitted_scale = self._fit_font_to_width(self.bar_label, max_width, self.base_font_scale)
        # Ensure font doesn't go below base scale unless absolutely necessary
        fitted_scale = max(fitted_scale, self.base_font_scale * 0.9)
        thickness = calculate_font_thickness(fitted_scale)

        self._draw_text(self.bar_label, label_x, label_y, fitted_scale, thickness)

    def _draw_main_bar(self) -> None:
        """Step 11: Draw main bar rectangle."""
        cv2.rectangle(
            self.canvas,
            (self.rect_x, self.rect_y),
            (self.rect_x + self.rect_w, self.rect_y + self.rect_h),
            RECT_COLOR,
            max(1, int(self.rect_h / 40))
        )

    def _draw_segments(self) -> None:
        """Step 12: Draw segments and segment labels with font resizing."""
        if not self.segments_data:
            self.segment_boundaries = [self.rect_x, self.rect_x + self.rect_w]
            return

        total_length = sum(seg[0] for seg in self.segments_data if seg[0] > 0)
        if total_length == 0:
            self.segment_boundaries = [self.rect_x, self.rect_x + self.rect_w]
            return

        unit_width = self.rect_w / total_length
        current_x = self.rect_x
        self.segment_boundaries = [current_x]

        # Calculate line thickness same as bar rectangle
        line_thickness = max(1, int(self.rect_h / 40))

        for i, (length, label) in enumerate(self.segments_data):
            if length <= 0:
                continue

            seg_width = length * unit_width
            seg_center_x = current_x + seg_width / 2
            seg_center_y = self.rect_y + self.rect_h / 2

            # Step 12: Resize font to fit within segment
            if label:
                max_width = max(10, seg_width * 0.9)  # 90% of segment width
                fitted_scale = self._fit_font_to_width(label, max_width, self.base_font_scale)
                thickness = calculate_font_thickness(fitted_scale)
                self._draw_text(label, int(seg_center_x), int(seg_center_y), fitted_scale, thickness)

            current_x += seg_width
            self.segment_boundaries.append(int(current_x))

            # Draw segment divider with same thickness as bar rectangle
            if i < len(self.segments_data) - 1:
                cv2.line(
                    self.canvas,
                    (int(current_x), self.rect_y),
                    (int(current_x), self.rect_y + self.rect_h),
                    LINE_COLOR,
                    line_thickness
                )

        # Ensure final boundary aligns
        if self.segment_boundaries:
            self.segment_boundaries[-1] = self.rect_x + self.rect_w

    def _draw_h_braces(self) -> None:
        """Step 13: Draw horizontal braces with labels and font resizing."""
        if not self.h_braces_data or not self.segment_boundaries:
            return

        # Prepare brace data with positions
        brace_positions = []
        for start_idx, end_idx, label, location in self.h_braces_data:
            if not self._valid_brace_indices(start_idx, end_idx):
                continue

            start_x = self.segment_boundaries[start_idx]
            end_x = self.segment_boundaries[end_idx + 1]
            brace_positions.append(((start_idx, end_idx, label, location), start_x, end_x))

        # Sort by width (shortest first for better level assignment)
        brace_positions.sort(key=lambda x: abs(x[2] - x[1]))

        # Calculate brace level height
        level_height = self._calculate_brace_level_height()

        # Draw each brace
        for brace_info, start_x, end_x in brace_positions:
            _, _, label, location = brace_info

            # Find appropriate level
            level = self.brace_manager.find_h_brace_level(start_x, end_x, location)

            # Draw brace
            self._draw_single_h_brace(start_x, end_x, label, location, level, level_height)

    def _draw_single_h_brace(
        self,
        start_x: int,
        end_x: int,
        label: str,
        location: str,
        level: int,
        level_height: int
    ) -> None:
        """Draw a single horizontal brace with proper positioning.

        Args:
            start_x: Starting x-coordinate.
            end_x: Ending x-coordinate.
            label: Brace label text.
            location: Brace location ('top' or 'bottom').
            level: Brace level (for stacking).
            level_height: Height of each level.
        """
        y_offset = level * level_height
        brace_gap = self.config.get('brace_to_bar_gap')

        if location == 'top':
            y_base = self.rect_y - brace_gap - y_offset
            y_line = y_base - BRACE_ARM_LENGTH
            text_y = y_line - self.config.get('brace_to_label_gap')
        else:  # bottom
            y_base = self.rect_y + self.rect_h + brace_gap + y_offset
            y_line = y_base + BRACE_ARM_LENGTH
            text_y = y_line + self.config.get('brace_to_label_gap')

        # Draw brace graphics
        self._draw_h_brace_graphics(start_x, end_x, y_line, y_base, location)

        # Step 13: Draw label with font resizing to fit brace width
        if label:
            brace_width = abs(end_x - start_x)
            fitted_scale = self._fit_font_to_width(label, brace_width, self.base_font_scale)
            thickness = calculate_font_thickness(fitted_scale)
            center_x = (start_x + end_x) // 2
            self._draw_text(label, center_x, int(text_y), fitted_scale, thickness)

    def _draw_h_brace_graphics(
        self,
        start_x: int,
        end_x: int,
        y_line: int,
        y_base: int,
        location: str
    ) -> None:
        """Draw the graphical elements of a horizontal brace (Step 6).

        Args:
            start_x: Starting x-coordinate.
            end_x: Ending x-coordinate.
            y_line: Y-coordinate of the horizontal line.
            y_base: Y-coordinate of the base.
            location: Brace location ('top' or 'bottom').
        """
        mid_x = (start_x + end_x) // 2
        arrow_gap = BRACE_ARROW_SIZE
        radius = min(8, abs(end_x - start_x) // 8, BRACE_ARM_LENGTH // 2)

        # Use same thickness as bar rectangle
        line_thickness = max(1, int(self.rect_h / 40))

        # Draw corner arcs
        if location == 'top':
            # Top-left corner
            cv2.ellipse(
                self.canvas,
                (start_x + radius, y_line + radius),
                (radius, radius),
                0, 180, 270,
                LINE_COLOR,
                line_thickness
            )
            # Top-right corner
            cv2.ellipse(
                self.canvas,
                (end_x - radius, y_line + radius),
                (radius, radius),
                0, 270, 360,
                LINE_COLOR,
                line_thickness
            )
            arm_tip_y = y_line + radius
            arrow_tip_y = y_line - arrow_gap
        else:  # bottom
            # Bottom-left corner
            cv2.ellipse(
                self.canvas,
                (start_x + radius, y_line - radius),
                (radius, radius),
                0, 90, 180,
                LINE_COLOR,
                line_thickness
            )
            # Bottom-right corner
            cv2.ellipse(
                self.canvas,
                (end_x - radius, y_line - radius),
                (radius, radius),
                0, 0, 90,
                LINE_COLOR,
                line_thickness
            )
            arm_tip_y = y_line - radius
            arrow_tip_y = y_line + arrow_gap

        # Draw horizontal lines (with gap for arrow)
        cv2.line(
            self.canvas,
            (start_x + radius, y_line),
            (mid_x - arrow_gap, y_line),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (mid_x + arrow_gap, y_line),
            (end_x - radius, y_line),
            LINE_COLOR,
            line_thickness
        )

        # Draw vertical arms
        cv2.line(
            self.canvas,
            (start_x, arm_tip_y),
            (start_x, y_base),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (end_x, arm_tip_y),
            (end_x, y_base),
            LINE_COLOR,
            line_thickness
        )

        # Draw arrow
        cv2.line(
            self.canvas,
            (mid_x - arrow_gap, y_line),
            (mid_x, arrow_tip_y),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (mid_x, arrow_tip_y),
            (mid_x + arrow_gap, y_line),
            LINE_COLOR,
            line_thickness
        )

    def _calculate_brace_level_height(self) -> int:
        """Step 6: Calculate height required for each brace level.

        Returns:
            Height in pixels for each brace level.
        """
        _, text_h = get_text_dimensions("M", self.base_font_scale, self.font_thickness)
        return (
            BRACE_ARM_LENGTH +
            self.config.get('brace_to_label_gap') +
            text_h +
            self.config.get('brace_level_gap')
        )

    def _fit_font_to_width(self, text: str, max_width: int, initial_scale: float) -> float:
        """Fit font scale to available width (Steps 2, 12, 13).

        Args:
            text: Text to fit.
            max_width: Maximum width available.
            initial_scale: Initial font scale.

        Returns:
            Fitted font scale.
        """
        if not text or max_width <= 0:
            return initial_scale

        thickness = calculate_font_thickness(initial_scale)
        text_w, _ = get_text_dimensions(text, initial_scale, thickness)

        if text_w <= max_width or text_w == 0:
            return initial_scale

        # Scale down to fit - protect against division by zero
        fitted_scale = (max_width * 0.9) / max(text_w, 1) * initial_scale
        return max(0.1, fitted_scale)

    def _draw_text(
        self,
        text: str,
        center_x: int,
        center_y: int,
        font_scale: float,
        thickness: int
    ) -> None:
        """Helper function to draw centered text.

        Args:
            text: Text to draw.
            center_x: X-coordinate of text center.
            center_y: Y-coordinate of text center.
            font_scale: Font scale factor.
            thickness: Font thickness.
        """
        if not text:
            return

        text_w, text_h = get_text_dimensions(text, font_scale, thickness)
        org_x = int(center_x - text_w / 2)
        org_y = int(center_y + text_h / 2)
        cv2.putText(
            self.canvas,
            text,
            (org_x, org_y),
            FONT,
            font_scale,
            FONT_COLOR,
            thickness,
            cv2.LINE_AA
        )

    def _valid_brace_indices(self, start_idx: int, end_idx: int) -> bool:
        """Validate brace indices against segment boundaries.

        Args:
            start_idx: Starting segment index.
            end_idx: Ending segment index.

        Returns:
            True if indices are valid, False otherwise.
        """
        num_segments = len(self.segments_data)
        return (
            0 <= start_idx <= end_idx < num_segments and
            start_idx < len(self.segment_boundaries) - 1 and
            end_idx < len(self.segment_boundaries) - 1
        )


class BarModel:
    """Manages multiple bars and vertical braces.

    Implements the complete 14-step process from instructions for creating
    bar model visualizations.
    """

    def __init__(
        self,
        canvas: np.ndarray,
        bars_data: List[Dict[str, Any]],
        vertical_braces_data: Optional[List[Tuple[int, int, str]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the BarModel.

        Args:
            canvas: The canvas to draw on.
            bars_data: A list of dictionaries, where each dictionary represents a bar.
            vertical_braces_data: A list of tuples for vertical braces.
                Each tuple is (start_bar_idx, end_bar_idx, label).
                Indices are 0-indexed and refer to the `bars_data` list.
            config: Optional dictionary for layout configuration.
        """
        self.canvas: np.ndarray = canvas
        self.canvas_h: int
        self.canvas_w: int
        self.canvas_h, self.canvas_w = canvas.shape[:2]
        self.bars_data: List[Dict[str, Any]] = bars_data
        self.v_braces_data: List[Tuple[int, int, str]] = vertical_braces_data or []
        self.config: LayoutConfig = LayoutConfig(config)

        # Layout calculations (Steps 1-9)
        self.layout_params: Dict[str, Any] = {}
        self.bars: List[Bar] = []

        # Initialize calculated values
        self.available_w: float = 0
        self.available_h: float = 0
        self.margins: Dict[str, float] = {}
        self.bar_h: float = 0
        self.base_font_scale: float = 0
        self.base_font_thickness: int = 0
        self.bar_label_w: int = 0
        self.v_brace_w: int = 0
        self.max_bar_w: int = 0
        self.total_diagram_h: float = 0
        self.diagram_start_x: float = 0
        self.diagram_start_y: float = 0
        self.bar_total_lengths: List[float] = []
        self.max_total_length: float = 1
        self.bar_widths: List[int] = []

        # Execute the 14-step process
        self._calculate_layout()  # Steps 1-9
        self._draw_all_elements()  # Steps 10-14

    def _calculate_layout(self) -> None:
        """Steps 1-9: Calculate all layout parameters before drawing."""
        # Step 1: Calculate initial dimensions and margins
        self._calculate_initial_dimensions()

        # Calculate bar total lengths for proportional width calculations
        self._calculate_bar_total_lengths()

        # Step 2: Use iterative approach to resolve bar height and brace space dependencies
        self._calculate_optimal_bar_height()

        # Step 2: Calculate font settings based on final bar height
        self._calculate_font_settings()

        # Steps 7-8: Calculate suitable widths and heights
        self._calculate_diagram_dimensions()

        # Step 9: Center diagram vertically and horizontally
        self._center_diagram()

    def _calculate_initial_dimensions(self) -> None:
        """Step 1: Calculate available space and margins."""
        # Calculate available space
        left_margin = self.canvas_w * self.config.get('left_margin_ratio')
        right_margin = self.canvas_w * self.config.get('right_margin_ratio')
        top_margin = self.canvas_h * self.config.get('top_margin_ratio')
        bottom_margin = self.canvas_h * self.config.get('bottom_margin_ratio')

        self.available_w = self.canvas_w - left_margin - right_margin
        self.available_h = self.canvas_h - top_margin - bottom_margin

        # Store margin values
        self.margins = {
            'left': left_margin,
            'right': right_margin,
            'top': top_margin,
            'bottom': bottom_margin
        }

    def _calculate_optimal_bar_height(self) -> None:
        """Calculate optimal bar height based on remaining space after accounting for all other elements."""
        num_bars = len(self.bars_data)
        if num_bars == 0:
            self.bar_h = 50  # Default fallback
            return

        # Use an initial estimate for bar height to calculate font and brace spaces
        max_bar_height = self.canvas_h * self.config.get('max_bar_height_ratio')
        estimated_bar_height = min(max_bar_height, self.available_h / (num_bars * 1.5))  # Conservative estimate

        # Calculate initial font settings based on estimated height
        initial_font_scale = estimated_bar_height * self.config.get('font_size_ratio') / 50.0
        initial_font_scale = max(0.3, min(initial_font_scale, 2.0))
        initial_font_thickness = calculate_font_thickness(initial_font_scale)

        # Store these for brace calculations
        self.base_font_scale = initial_font_scale
        self.base_font_thickness = initial_font_thickness

        # Calculate space needed for horizontal braces for all bars
        total_h_brace_space = 0
        for bar_data in self.bars_data:
            total_h_brace_space += self._calculate_required_h_brace_space(bar_data, 'top')
            total_h_brace_space += self._calculate_required_h_brace_space(bar_data, 'bottom')

        # Calculate space needed for vertical gaps between bars
        total_v_gaps = (num_bars - 1) * self.config.get('v_gap_between_bars') if num_bars > 1 else 0

        # Calculate remaining space for actual bars
        remaining_space = self.available_h - total_h_brace_space - total_v_gaps

        # Divide remaining space by number of bars
        calculated_bar_height = remaining_space / num_bars if num_bars > 0 else 50

        # Apply the cap at max_bar_height_ratio of canvas height
        self.bar_h = max(20, min(calculated_bar_height, max_bar_height))  # Minimum 20px

    def _calculate_font_settings(self) -> None:
        """Step 2-3: Calculate font size and thickness based on final bar height."""
        # Step 2: Font size as percentage of bar height
        self.base_font_scale = self.bar_h * self.config.get('font_size_ratio') / 50.0
        self.base_font_scale = max(0.3, min(self.base_font_scale, 2.0))

        # Step 3: Adjust font thickness based on font size
        self.base_font_thickness = calculate_font_thickness(self.base_font_scale)

    def _calculate_diagram_dimensions(self) -> None:
        """Steps 7-8: Calculate suitable widths and full diagram height."""
        # Calculate bar label area width
        max_label_w = 0
        for bar_data in self.bars_data:
            label = bar_data.get('label', '')
            if label:
                text_w, _ = get_text_dimensions(label, self.base_font_scale, self.base_font_thickness)
                max_label_w = max(max_label_w, text_w)

        self.bar_label_w = max_label_w

        # Calculate vertical brace area width using proper calculation
        self.v_brace_w = self._calculate_required_v_brace_space()

        # Step 7: Calculate maximum available bar width for the longest bar
        max_available_width = (
            self.available_w -
            self.bar_label_w -
            self.config.get('bar_label_to_bar_gap') -
            self.v_brace_w
        )
        max_available_width = max(50, max_available_width)  # Minimum bar width

        # Calculate individual bar widths based on proportional lengths
        self.bar_widths = []
        for total_length in self.bar_total_lengths:
            if self.max_total_length > 0:
                proportional_width = (total_length / self.max_total_length) * max_available_width
            else:
                proportional_width = max_available_width

            # Ensure minimum width for visibility
            proportional_width = max(20, proportional_width)
            self.bar_widths.append(int(proportional_width))

        # Store the maximum bar width for layout calculations (longest bar width)
        self.max_bar_w = max(self.bar_widths) if self.bar_widths else int(max_available_width)

        # Step 8: Calculate full diagram height by summing up actual required space
        num_bars = len(self.bars_data)
        if num_bars > 0:
            total_bars_h = num_bars * self.bar_h
            total_gaps_h = (num_bars - 1) * self.config.get('v_gap_between_bars') if num_bars > 1 else 0

            # Calculate total brace space by iterating through each bar
            total_braces_h = 0
            for bar_data in self.bars_data:
                total_braces_h += self._calculate_required_h_brace_space(bar_data, 'top')
                total_braces_h += self._calculate_required_h_brace_space(bar_data, 'bottom')

            self.total_diagram_h = total_bars_h + total_gaps_h + total_braces_h
        else:
            self.total_diagram_h = 0

    def _center_diagram(self) -> None:
        """Step 9: Center diagram vertically, horizontal centering via margins."""
        # Horizontal positioning (left margin defines left edge)
        self.diagram_start_x = self.margins['left']

        # Step 9: Vertical centering
        if self.total_diagram_h < self.available_h:
            remaining_space = self.available_h - self.total_diagram_h
            self.diagram_start_y = self.margins['top'] + remaining_space / 2
        else:
            self.diagram_start_y = self.margins['top']

    def _draw_all_elements(self) -> None:
        """Steps 10-14: Draw all diagram elements in proper order."""
        if not self.bars_data:
            return

        # Calculate positions for each bar
        current_y = self.diagram_start_y
        bar_layouts: List[Dict[str, Any]] = []

        for i, bar_data in enumerate(self.bars_data):
            # Accurately calculate space needed for top and bottom braces
            top_brace_space = self._calculate_required_h_brace_space(bar_data, 'top')
            bottom_brace_space = self._calculate_required_h_brace_space(bar_data, 'bottom')

            # Use individual bar width for this bar
            bar_width = self.bar_widths[i] if i < len(self.bar_widths) else 50

            bar_layout = {
                'rect_x': int(self.diagram_start_x + self.bar_label_w + self.config.get('bar_label_to_bar_gap')),
                'rect_y': int(current_y + top_brace_space),
                'rect_w': int(bar_width),
                'rect_h': int(self.bar_h),
                'bar_label_x': int(self.diagram_start_x + self.bar_label_w / 2),
                'bar_label_width': int(self.bar_label_w),
                'base_font_scale': self.base_font_scale
            }

            bar_layouts.append(bar_layout)

            # Steps 10-13: Create and draw bar
            bar = Bar(self.canvas, bar_data, bar_layout, self.config)
            self.bars.append(bar)

            # Move to next bar position
            current_y += top_brace_space + self.bar_h + bottom_brace_space + self.config.get('v_gap_between_bars')

        # Step 14: Draw vertical braces
        self._draw_v_braces(bar_layouts)

    def _draw_v_braces(self, bar_layouts: List[Dict[str, Any]]) -> None:
        """Step 14: Draw vertical braces.

        Args:
            bar_layouts: List of bar layout dictionaries.
        """
        if not self.v_braces_data or not bar_layouts:
            return

        # V-brace positioning - start right after the longest bar with proper gap
        v_brace_start_x = (
            self.diagram_start_x + self.bar_label_w +
            self.config.get('bar_label_to_bar_gap') + self.max_bar_w +
            self.config.get('brace_to_bar_gap')
        )

        # Prepare v-brace data with positions
        v_brace_positions = []
        for start_bar_idx, end_bar_idx, label in self.v_braces_data:
            if not (0 <= start_bar_idx <= end_bar_idx < len(bar_layouts)):
                continue

            start_y = bar_layouts[start_bar_idx]['rect_y']
            end_y = bar_layouts[end_bar_idx]['rect_y'] + bar_layouts[end_bar_idx]['rect_h']
            v_brace_positions.append(((start_bar_idx, end_bar_idx, label), start_y, end_y))

        # Sort by height (shortest first)
        v_brace_positions.sort(key=lambda x: abs(x[2] - x[1]))

        # Level manager for vertical braces
        v_brace_manager = BraceLevelManager()
        level_width = self._calculate_v_brace_level_width()

        # Draw each vertical brace
        for brace_info, start_y, end_y in v_brace_positions:
            _, _, label = brace_info

            # Find appropriate level
            level = v_brace_manager.find_v_brace_level(start_y, end_y)

            # Draw v-brace
            self._draw_single_v_brace(v_brace_start_x, start_y, end_y, label, level, level_width)

    def _draw_single_v_brace(
        self,
        base_x: int,
        start_y: int,
        end_y: int,
        label: str,
        level: int,
        level_width: int
    ) -> None:
        """Draw a single vertical brace.

        Args:
            base_x: Base x-coordinate.
            start_y: Starting y-coordinate.
            end_y: Ending y-coordinate.
            label: Brace label text.
            level: Brace level (for stacking).
            level_width: Width of each level.
        """
        x_offset = level * level_width
        x_arm_end = int(base_x + x_offset)
        x_main_line = int(x_arm_end + BRACE_ARM_LENGTH)

        mid_y = (start_y + end_y) // 2
        arrow_gap = BRACE_ARROW_SIZE
        radius = min(8, abs(end_y - start_y) // 8, BRACE_ARM_LENGTH // 2)

        # Use same thickness as bar rectangle (estimate from typical bar height)
        line_thickness = max(1, int(self.bar_h / 40))

        # Draw corner arcs with correct positioning (matching draw.py)
        # Top corner: 270-360 degrees
        cv2.ellipse(
            self.canvas,
            (int(x_main_line - radius), int(start_y + radius)),
            (radius, radius),
            0, 270, 360,
            LINE_COLOR,
            line_thickness
        )
        # Bottom corner: 0-90 degrees
        cv2.ellipse(
            self.canvas,
            (int(x_main_line - radius), int(end_y - radius)),
            (radius, radius),
            0, 0, 90,
            LINE_COLOR,
            line_thickness
        )

        # Draw vertical lines (with gap for arrow) along the main line
        cv2.line(
            self.canvas,
            (x_main_line, int(start_y + radius)),
            (x_main_line, int(mid_y - arrow_gap)),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (x_main_line, int(mid_y + arrow_gap)),
            (x_main_line, int(end_y - radius)),
            LINE_COLOR,
            line_thickness
        )

        # Draw horizontal arms from ellipse connection points to arm endpoints
        cv2.line(
            self.canvas,
            (int(x_main_line - radius), start_y),
            (x_arm_end, start_y),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (int(x_main_line - radius), end_y),
            (x_arm_end, end_y),
            LINE_COLOR,
            line_thickness
        )

        # Draw arrow
        arrow_tip_x = int(x_main_line + arrow_gap)
        cv2.line(
            self.canvas,
            (x_main_line, int(mid_y - arrow_gap)),
            (arrow_tip_x, mid_y),
            LINE_COLOR,
            line_thickness
        )
        cv2.line(
            self.canvas,
            (arrow_tip_x, mid_y),
            (x_main_line, int(mid_y + arrow_gap)),
            LINE_COLOR,
            line_thickness
        )

        # Draw label with correct positioning
        if label:
            # Position text starting from main line + text padding
            text_left_x = x_main_line + self.config.get('brace_to_label_gap')
            brace_height = abs(end_y - start_y)
            fitted_scale = self._fit_font_to_height(label, brace_height, self.base_font_scale)
            thickness = calculate_font_thickness(fitted_scale)

            # Get text width to calculate center position
            text_w, _ = get_text_dimensions(label, fitted_scale, thickness)
            text_center_x = text_left_x + text_w // 2

            self._draw_text(label, int(text_center_x), mid_y, fitted_scale, thickness)

    def _fit_font_to_height(self, text: str, max_height: int, initial_scale: float) -> float:
        """Fit font scale to available height for vertical brace labels.

        Args:
            text: Text to fit.
            max_height: Maximum height available.
            initial_scale: Initial font scale.

        Returns:
            Fitted font scale.
        """
        if not text or max_height <= 0:
            return initial_scale

        thickness = calculate_font_thickness(initial_scale)
        _, text_h = get_text_dimensions(text, initial_scale, thickness)

        if text_h <= max_height or text_h == 0:
            return initial_scale

        # Scale down to fit - protect against division by zero
        fitted_scale = (max_height * 0.9) / max(text_h, 1) * initial_scale
        return max(0.1, fitted_scale)

    def _draw_text(
        self,
        text: str,
        center_x: int,
        center_y: int,
        font_scale: float,
        thickness: int
    ) -> None:
        """Helper function to draw centered text.

        Args:
            text: Text to draw.
            center_x: X-coordinate of text center.
            center_y: Y-coordinate of text center.
            font_scale: Font scale factor.
            thickness: Font thickness.
        """
        if not text:
            return

        text_w, text_h = get_text_dimensions(text, font_scale, thickness)
        org_x = int(center_x - text_w / 2)
        org_y = int(center_y + text_h / 2)
        cv2.putText(
            self.canvas,
            text,
            (org_x, org_y),
            FONT,
            font_scale,
            FONT_COLOR,
            thickness,
            cv2.LINE_AA
        )

    def _estimate_brace_level_height(self) -> int:
        """Estimate height needed for each brace level.

        Returns:
            Estimated height in pixels.
        """
        _, text_h = get_text_dimensions("M", self.base_font_scale, self.base_font_thickness)
        return (
            BRACE_ARM_LENGTH +
            self.config.get('brace_to_label_gap') +
            text_h +
            self.config.get('brace_level_gap')
        )

    def _calculate_v_brace_level_width(self) -> int:
        """Calculate width needed for each vertical brace level.

        Returns:
            Width in pixels for each vertical brace level.
        """
        _, text_h = get_text_dimensions("M", self.base_font_scale, self.base_font_thickness)
        max_label_w = 0
        for _, _, label in self.v_braces_data:
            if label:
                text_w, _ = get_text_dimensions(label, self.base_font_scale, self.base_font_thickness)
                max_label_w = max(max_label_w, text_w)

        return (
            BRACE_ARM_LENGTH +
            self.config.get('brace_to_label_gap') +
            max_label_w +
            self.config.get('brace_level_gap')
        )

    def _calculate_required_h_brace_space(self, bar_data: Dict[str, Any], location: str) -> int:
        """Calculate the vertical space required for horizontal braces for a single bar at a given location.

        Args:
            bar_data: Bar data dictionary.
            location: Brace location ('top' or 'bottom').

        Returns:
            Required vertical space in pixels.
        """
        h_braces = bar_data.get('h_braces', [])
        location_braces = [b for b in h_braces if len(b) > 3 and b[3] == location]

        if not location_braces:
            return self.config.get('brace_to_bar_gap')

        segments_data = bar_data.get('segments', [])
        if not segments_data:
            return self.config.get('brace_to_bar_gap')

        # --- Dry run of brace level assignment ---
        # 1. Create hypothetical segment boundaries
        total_length = sum(seg[0] for seg in segments_data if seg[0] > 0)
        if total_length == 0:
            return self.config.get('brace_to_bar_gap')

        hypothetical_width = 1000  # A dummy width for proportional calculation
        unit_width = hypothetical_width / total_length
        current_x = 0
        segment_boundaries = [0]
        for length, _ in segments_data:
            if length > 0:
                current_x += length * unit_width
                segment_boundaries.append(int(current_x))
        segment_boundaries[-1] = hypothetical_width

        # 2. Prepare brace data with hypothetical positions
        brace_positions = []
        for start_idx, end_idx, _, _ in location_braces:
            # Basic validation
            if not (0 <= start_idx <= end_idx < len(segments_data) and end_idx + 1 < len(segment_boundaries)):
                continue
            start_x = segment_boundaries[start_idx]
            end_x = segment_boundaries[end_idx + 1]
            brace_positions.append((start_x, end_x))

        # 3. Use BraceLevelManager to find max levels
        if not brace_positions:
            return self.config.get('brace_to_bar_gap')

        brace_manager = BraceLevelManager()
        # Sort by width (shortest first) for better packing
        brace_positions.sort(key=lambda x: abs(x[1] - x[0]))

        max_level = -1
        for start_x, end_x in brace_positions:
            level = brace_manager.find_h_brace_level(start_x, end_x, location)
            max_level = max(max_level, level)

        num_levels = max_level + 1

        if num_levels == 0:
            return 0

        # 4. Calculate space based on number of levels
        # Level height includes the gap, but we only need N-1 gaps between N levels
        level_height = self._estimate_brace_level_height()
        total_space = num_levels * level_height
        # Subtract one gap since we don't need a gap after the last level
        if num_levels > 0:
            total_space -= self.config.get('brace_level_gap')
        return self.config.get('brace_to_bar_gap') + total_space

    def _calculate_required_v_brace_space(self) -> int:
        """Calculate the maximum number of vertical brace levels needed.

        Returns:
            Required width in pixels for vertical braces.
        """
        if not self.v_braces_data or not self.bars_data:
            return 0

        # Simulate the level assignment to find maximum levels needed
        v_brace_manager = BraceLevelManager()

        # Create hypothetical bar positions for level calculation
        bar_positions = []
        current_y = 0
        for i, bar_data in enumerate(self.bars_data):
            # Estimate space for top braces (rough estimate for this calculation)
            top_space = 100  # Rough estimate
            bar_start_y = current_y + top_space
            bar_end_y = bar_start_y + self.bar_h
            bar_positions.append((bar_start_y, bar_end_y))

            # Move to next position
            bottom_space = 100  # Rough estimate
            current_y = bar_end_y + bottom_space + self.config.get('v_gap_between_bars')

        # Sort v-braces by height (shortest first) for better level assignment
        v_brace_positions = []
        for start_bar_idx, end_bar_idx, label in self.v_braces_data:
            if not (0 <= start_bar_idx <= end_bar_idx < len(bar_positions)):
                continue
            start_y = bar_positions[start_bar_idx][0]
            end_y = bar_positions[end_bar_idx][1]
            v_brace_positions.append((start_y, end_y))

        v_brace_positions.sort(key=lambda x: abs(x[1] - x[0]))

        # Find maximum level assigned
        max_level = -1
        for start_y, end_y in v_brace_positions:
            level = v_brace_manager.find_v_brace_level(start_y, end_y)
            max_level = max(max_level, level)

        num_levels = max_level + 1

        if num_levels == 0:
            return 0

        # Calculate space needed for vertical braces
        # Level width includes the gap, but we only need N-1 gaps between N levels
        level_width = self._calculate_v_brace_level_width()
        total_space = num_levels * level_width
        # Subtract one gap since we don't need a gap after the last level
        if num_levels > 0:
            total_space -= self.config.get('brace_level_gap')
        return total_space + self.config.get('brace_to_bar_gap')

    def _calculate_bar_total_lengths(self) -> None:
        """Calculate total segment lengths for all bars and find maximum."""
        self.bar_total_lengths = []

        for bar_data in self.bars_data:
            segments_data = bar_data.get('segments', [])
            total_length = sum(seg[0] for seg in segments_data if seg[0] > 0)
            self.bar_total_lengths.append(total_length)

        # Find maximum total length for scaling reference
        self.max_total_length = max(self.bar_total_lengths) if self.bar_total_lengths else 1

        # Prevent division by zero
        if self.max_total_length == 0:
            self.max_total_length = 1

    def get_canvas(self) -> np.ndarray:
        """Return the canvas with the drawn model.

        Returns:
            The canvas with all drawn elements.
        """
        return self.canvas
