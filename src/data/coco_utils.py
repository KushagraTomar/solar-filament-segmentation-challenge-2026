"""COCO-style MAGFiLO annotation loading and polygon/spine conversion.

Schema reference (see competition Data tab):
  image["id"]      : "<annotator_batch>-<file_name w/o ext>", e.g. "040301-20140609195854Bh"
  image["file_name"]: underlying capture, e.g. "20140609195854Bh.jpeg"
  annotation["segmentation"]: [[x0,y0,x1,y1,...]]  -- single closed polygon per filament
  annotation["spine"]       : [x0,y0,x1,y1,...]    -- single open polyline per filament
  annotation["iscrowd"]     : always 0

Multiple annotator batches may annotate the same underlying file_name independently
(different image["id"], same file_name) -- these must be kept together on the same
side of any train/val split (see split.py) to avoid near-duplicate leakage.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

_INSTRUMENT_RE = re.compile(r"(\d{14})([A-Za-z]{2})\.jpe?g$")


def instrument_code(file_name):
    """Extract the 2-letter GONG site code from a file_name, e.g. 'Bh' from
    '20140609195854Bh.jpeg'. Returns None if the name doesn't match the expected pattern."""
    m = _INSTRUMENT_RE.search(file_name)
    return m.group(2) if m else None


def capture_datetime(file_name):
    """Extract the YYYYMMDDHHMMSS capture timestamp string from a file_name."""
    m = _INSTRUMENT_RE.search(file_name)
    return m.group(1) if m else None


def polygon_to_mask(segmentation, height, width):
    """Convert a COCO-style polygon segmentation (list of one or more flat
    [x0,y0,x1,y1,...] point lists) into a binary uint8 mask of shape (height, width)."""
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles) if len(rles) > 1 else rles[0]
    return mask_utils.decode(rle)


def spine_to_points(spine):
    """Convert a flat [x0,y0,x1,y1,...] spine list into an (N, 2) float array."""
    arr = np.asarray(spine, dtype=np.float32)
    return arr.reshape(-1, 2)


def rasterize_spine(spine_points, height, width, thickness=1):
    """Rasterize a spine polyline (N, 2) into a thin binary uint8 mask.

    Used as the ground-truth skeleton for clDice, in place of a morphologically
    approximated skeleton -- the annotated spine is a more accurate centerline
    than anything soft-skeletonization can recover from the mask alone.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(spine_points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=thickness)
    return canvas


class MagfiloAnnotations:
    """Indexes a MAGFiLO COCO-style JSON for fast lookup by image_id and file_name."""

    def __init__(self, json_path):
        with open(json_path) as f:
            self.raw = json.load(f)

        self.categories = {c["id"]: c["name"] for c in self.raw["categories"]}
        self.images_by_id = {im["id"]: im for im in self.raw["images"]}

        self.anns_by_image_id = defaultdict(list)
        for ann in self.raw["annotations"]:
            self.anns_by_image_id[ann["image_id"]].append(ann)

        self.image_ids_by_file_name = defaultdict(list)
        for im in self.raw["images"]:
            self.image_ids_by_file_name[im["file_name"]].append(im["id"])

    @property
    def file_names(self):
        return sorted(self.image_ids_by_file_name.keys())

    def annotations_for(self, image_id):
        return self.anns_by_image_id[image_id]

    def masks_for(self, image_id):
        """Return a list of (binary_mask, category_name, spine_points) for one image_id."""
        im = self.images_by_id[image_id]
        h, w = im["height"], im["width"]
        out = []
        for ann in self.anns_by_image_id[image_id]:
            mask = polygon_to_mask(ann["segmentation"], h, w)
            cat = self.categories[ann["category_id"]]
            spine = spine_to_points(ann["spine"]) if ann.get("spine") else None
            out.append((mask, cat, spine))
        return out

    def summary(self):
        return {
            "num_image_ids": len(self.images_by_id),
            "num_unique_file_names": len(self.image_ids_by_file_name),
            "num_annotations": len(self.raw["annotations"]),
            "categories": self.categories,
        }
