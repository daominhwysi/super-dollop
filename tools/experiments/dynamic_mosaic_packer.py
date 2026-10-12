"""
Dynamic Mosaic Shelf Packer (OpenCV & NumPy native).

Packs variable-sized formula crops into adaptive 2D canvas mosaics using shelf packing.
Each item is rendered with an unambiguous, high-contrast label badge (e.g. [F1], [F2]).
"""

from __future__ import annotations

from typing import List, Tuple, Sequence
import cv2
import numpy as np


class DynamicMosaicPacker:
    """
    Packs variable-sized formula crops onto an adaptive 2D canvas using shelf packing.
    """

    def __init__(
        self,
        max_canvas_width: int = 1024,
        max_canvas_height: int = 1024,
        padding: int = 10,
        badge_height: int = 24,
        border_color: Tuple[int, int, int] = (200, 200, 200),
        badge_bg: Tuple[int, int, int] = (240, 240, 240),
        badge_fg: Tuple[int, int, int] = (0, 0, 180),  # BGR format for cv2 (Dark Red)
    ):
        self.max_canvas_width = max_canvas_width
        self.max_canvas_height = max_canvas_height
        self.padding = padding
        self.badge_height = badge_height
        self.border_color = border_color
        self.badge_bg = badge_bg
        self.badge_fg = badge_fg

    def _render_framed_item(self, label: str, crop: np.ndarray) -> np.ndarray:
        """Frames a single formula crop with a top badge and border."""
        # Ensure 3-channel BGR
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        h, w = crop.shape[:2]

        cell_w = max(w + 2 * self.padding, 90)
        cell_h = h + self.badge_height + 2 * self.padding

        cell = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255

        # Draw cell outer border
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), self.border_color, 1)

        # Draw badge pill at the top
        cv2.rectangle(cell, (2, 2), (cell_w - 3, self.badge_height), self.badge_bg, -1)
        cv2.rectangle(cell, (2, 2), (cell_w - 3, self.badge_height), self.border_color, 1)

        # Draw badge text centered in badge area
        badge_text = f"[{label}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
        text_x = max(4, (cell_w - text_size[0]) // 2)
        text_y = (self.badge_height + text_size[1]) // 2
        cv2.putText(cell, badge_text, (text_x, text_y), font, font_scale, self.badge_fg, thickness, cv2.LINE_AA)

        # Paste crop centered below the badge
        crop_x = (cell_w - w) // 2
        crop_y = self.badge_height + self.padding
        cell[crop_y : crop_y + h, crop_x : crop_x + w] = crop

        return cell

    def pack(
        self,
        formula_items: Sequence[Tuple[str, np.ndarray]],
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Packs a list of (formula_id, crop_image) pairs into one or more mosaic canvases.
        Returns a list of tuples: (canvas_image_bgr, list_of_included_formula_ids).
        """
        if not formula_items:
            return []

        # Prepare framed cells
        cells: List[Tuple[str, np.ndarray]] = []
        for label, crop in formula_items:
            framed = self._render_framed_item(label, crop)
            cells.append((label, framed))

        canvases: List[Tuple[np.ndarray, List[str]]] = []
        current_rows: List[List[Tuple[str, np.ndarray]]] = [[]]
        current_row_widths: List[int] = [0]
        current_row_heights: List[int] = [0]
        total_canvas_height = self.padding

        def finalize_canvas():
            nonlocal current_rows, current_row_widths, current_row_heights, total_canvas_height
            if not any(current_rows):
                return

            included_ids: List[str] = []
            canvas_w = min(
                self.max_canvas_width,
                max(w for w in current_row_widths if w > 0) + 2 * self.padding,
            )
            canvas_h = total_canvas_height + self.padding
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

            y_offset = self.padding
            for r_idx, row in enumerate(current_rows):
                if not row:
                    continue
                x_offset = self.padding
                r_height = current_row_heights[r_idx]
                for fid, cell_img in row:
                    included_ids.append(fid)
                    ch, cw = cell_img.shape[:2]
                    canvas[y_offset : y_offset + ch, x_offset : x_offset + cw] = cell_img
                    x_offset += cw + self.padding
                y_offset += r_height + self.padding

            canvases.append((canvas, included_ids))
            current_rows = [[]]
            current_row_widths = [0]
            current_row_heights = [0]
            total_canvas_height = self.padding

        for fid, cell_img in cells:
            ch, cw = cell_img.shape[:2]

            # Check if cell fits on the current row
            if current_row_widths[-1] + cw + self.padding <= self.max_canvas_width:
                current_rows[-1].append((fid, cell_img))
                current_row_widths[-1] += cw + self.padding
                current_row_heights[-1] = max(current_row_heights[-1], ch)
            else:
                # Need to start a new row. Check if new row fits on current canvas
                row_h = current_row_heights[-1]
                if total_canvas_height + row_h + ch + self.padding > self.max_canvas_height:
                    # Canvas full: finalize and start new canvas
                    total_canvas_height += row_h
                    finalize_canvas()
                    current_rows[-1].append((fid, cell_img))
                    current_row_widths[-1] = cw + self.padding
                    current_row_heights[-1] = ch
                else:
                    # Start new row on same canvas
                    total_canvas_height += row_h + self.padding
                    current_rows.append([(fid, cell_img)])
                    current_row_widths.append(cw + self.padding)
                    current_row_heights.append(ch)

        # Finalize the last pending canvas
        if any(current_rows):
            total_canvas_height += current_row_heights[-1]
            finalize_canvas()

        return canvases
