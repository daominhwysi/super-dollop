import unittest
import numpy as np
from tools.experiments.dynamic_mosaic_packer import DynamicMosaicPacker


class TestDynamicMosaicPacker(unittest.TestCase):
    def test_pack_single_item(self):
        packer = DynamicMosaicPacker(max_canvas_width=800, max_canvas_height=800)
        crop = np.ones((40, 100, 3), dtype=np.uint8) * 255
        items = [("F1", crop)]

        canvases = packer.pack(items)
        self.assertEqual(len(canvases), 1)
        canvas, included_ids = canvases[0]
        self.assertEqual(included_ids, ["F1"])
        h, w = canvas.shape[:2]
        self.assertGreaterEqual(w, 100)
        self.assertGreaterEqual(h, 40)

    def test_pack_multiple_items_with_overflow(self):
        packer = DynamicMosaicPacker(max_canvas_width=300, max_canvas_height=150, padding=5)
        items = [
            (f"F{i}", np.ones((40, 120, 3), dtype=np.uint8) * 255)
            for i in range(1, 10)
        ]

        canvases = packer.pack(items)
        self.assertGreater(len(canvases), 1)
        total_packed = sum(len(ids) for _, ids in canvases)
        self.assertEqual(total_packed, 9)


if __name__ == "__main__":
    unittest.main()
