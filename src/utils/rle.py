"""RLE encode/decode and submission CSV writer.

Submission format: one row per predicted filament instance, columns
`filament_id` (e.g. "20150125172714Mh_1") and `segmentation_rle` (pycocotools
RLE `counts` only -- size is fixed at 2048x2048 for every image and is not stored).
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

SIZE = (2048, 2048)


def mask_to_rle_counts(binary_mask):
    """Encode a (H, W) binary mask (0/1 or bool) into a pycocotools RLE counts string."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def rle_counts_to_mask(counts, height=SIZE[0], width=SIZE[1]):
    """Decode a submission-format RLE counts string back into a (H, W) uint8 mask."""
    rle = {"size": [height, width], "counts": counts.encode("utf-8")}
    return mask_utils.decode(rle)


class SubmissionWriter:
    """Accumulates per-image predicted instance masks and writes the final
    submission CSV.

    Usage:
        writer = SubmissionWriter()
        writer.add_image("20150125172714Mh", [mask1, mask2, mask3])
        writer.write("submission.csv")
    """

    def __init__(self):
        self._rows = []

    def add_image(self, file_stem, masks):
        for i, mask in enumerate(masks, start=1):
            self._rows.append(
                {
                    "filament_id": f"{file_stem}_{i}",
                    "segmentation_rle": mask_to_rle_counts(mask),
                }
            )

    def write(self, out_path):
        df = pd.DataFrame(self._rows, columns=["filament_id", "segmentation_rle"])
        df.to_csv(out_path, index=False)
        return df
