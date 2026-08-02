"""Train/val split for MAGFiLO.

Splits by underlying file_name (NOT by image_id), because the same file_name can
carry multiple independent annotator batches (different image_id, same pixels).
Splitting by image_id would leak near-duplicate frames across train/val and
inflate validation Dice. Stratifies by instrument code so both splits see a
representative mix of GONG sites.
"""
import json
from collections import defaultdict

from sklearn.model_selection import train_test_split

from .coco_utils import MagfiloAnnotations, instrument_code


def make_split(json_path, val_fraction=0.15, seed=42):
    """Returns (train_file_names, val_file_names), split by file_name and
    stratified by instrument code."""
    anns = MagfiloAnnotations(json_path)
    file_names = anns.file_names
    strata = [instrument_code(fn) for fn in file_names]

    train_fns, val_fns = train_test_split(
        file_names,
        test_size=val_fraction,
        random_state=seed,
        stratify=strata,
    )
    return sorted(train_fns), sorted(val_fns)


def image_ids_for_file_names(json_path, file_names):
    """Expand a list of file_names back into all image_ids (annotator batches) for them."""
    anns = MagfiloAnnotations(json_path)
    file_name_set = set(file_names)
    return [
        im_id
        for fn in file_name_set
        for im_id in anns.image_ids_by_file_name[fn]
    ]


def write_split(json_path, out_path, val_fraction=0.15, seed=42):
    train_fns, val_fns = make_split(json_path, val_fraction=val_fraction, seed=seed)
    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_file_names": train_fns,
        "val_file_names": val_fns,
        "train_image_ids": image_ids_for_file_names(json_path, train_fns),
        "val_image_ids": image_ids_for_file_names(json_path, val_fns),
    }
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return split
