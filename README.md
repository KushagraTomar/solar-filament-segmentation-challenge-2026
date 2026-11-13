# Solar Filament Segmentation (MAGFiLO / Kaggle 2026)

Instance segmentation of solar filaments in GONG H-alpha images, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup). Scored with Panoptic Quality (PQ).

## Approach

A semantic U-Net followed by morphological instance extraction:

1. **U-Net** (`segmentation_models_pytorch`) with an ImageNet-pretrained
   EfficientNet-B3 encoder and a 1-channel (grayscale) input predicts a
   per-pixel filament probability.
   - Trained on all 1154 annotator batches (707 images) at full resolution,
     using random 1024×1024 crops of the 2048×2048 images.
   - Loss: BCE + Dice. Optimiser: AdamW with a one-cycle schedule. 30 epochs
     with mixed precision.
   - Augmentation: flips, 90° rotations, mild affine, brightness/contrast,
     gamma and blur.
2. **Inference** on the whole 2048×2048 image, averaging 4 flipped predictions
   (test-time augmentation).
3. **Post-processing** turns the probability map into individual filaments:
   - threshold (0.4);
   - morphological closing and hole filling, because the annotations are
     smooth polygons without holes;
   - connected components of a dilated mask, so nearby fragments of one
     filament share an id while each filament keeps its original undilated
     shape;
   - removal of components that are small (< 400 px) or low-confidence
     (mean probability < 0.85).

The post-processing parameters were chosen by grid search on validation PQ in
earlier experiments, and are fixed in the final pipeline.

## Repository

| Path | Content |
|---|---|
| `filament_segmentation.ipynb` | the entire pipeline: data → training → inference → post-processing → `submission.csv` |
| `weights/unet_effb3.pth` | trained model weights, loaded by the notebook |
| `requirements.txt` | exact package versions used |
| `submission.csv` | the submitted predictions |
| `kaggle_run/training_run.ipynb` | the executed Kaggle notebook (single T4) that trained `weights/unet_effb3.pth` and produced `submission.csv`, with its training log |

## Reproducing the predictions

1. Download the competition data and unzip it so that
   `data/MAGFiLO_1.0_Kaggle_2026/` exists. On Kaggle, the data is found
   automatically under `/kaggle/input`.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `filament_segmentation.ipynb` from the repository root with
   `TRAIN = False` (the default). The notebook loads `weights/unet_effb3.pth`
   and regenerates `submission.csv` without training. A GPU is recommended;
   on a CPU, inference takes several seconds per image.

We checked this on Kaggle (T4 GPU): the regenerated file has the same 1041
filaments as the committed `submission.csv`. A few masks differ by a handful of
boundary pixels (104 of 2.38M filament pixels; every mask IoU ≥ 0.994) because
mixed-precision GPU arithmetic is not bit-exact between runs.

To retrain from scratch, set `TRAIN = True`. This takes about 2 hours on a
single T4 GPU and overwrites the weights.
