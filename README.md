# Deforestation Segmentation in the Amazon

Semantic segmentation of deforestation from Sentinel-2 satellite imagery, with a U-Net++ model optimized for fast CPU inference using OpenVINO.

## Overview

This project builds a pixel-level deforestation detector for a region of the Brazilian Amazon:

1. **Data pipeline:** Sentinel-2 imagery and Hansen Global Forest Change labels are pulled from Google Earth Engine and cut into 256×256 tiles.
2. **Model:** U-Net and U-Net++ (ResNet34 encoder, ImageNet-pretrained) trained with a combined Dice + BCE loss.
3. **Deployment:** the best model is exported to ONNX, converted to OpenVINO (FP32 / FP16 / INT8), and benchmarked on CPU and GPU.

The repository description also targets cross-region generalization via domain-adversarial learning. <!-- TODO: describe that work here if it's in deforestation.ipynb -->

## Data

| Item | Detail |
|------|--------|
| Imagery | Sentinel-2 SR Harmonized (`COPERNICUS/S2_SR_HARMONIZED`), 2023 median composite, scenes with <20% cloud cover, RGB bands (B4, B3, B2) |
| Labels | Hansen Global Forest Change v1.13 (`UMD/hansen/global_forest_change_2025_v1_13`): pixels with >30% tree cover in 2000 **and** recorded forest loss |
| Study area | Bounding box lon −63.0 to −62.5, lat −8.5 to −8.0 |
| Tiles | 256×256 px at 10 m resolution, a 21×21 grid = **441 tiles** |
| Class balance | ~24% deforested pixels overall; 65 tiles (14.7%) contain no deforestation |

Note that the labels are cumulative forest loss since 2000, while the imagery is a 2023 composite.

### Splits

Tiles are split by **row** (not randomly) so that neighbouring tiles don't leak between sets. Rows are ranked by deforestation rate into low, mid, and high thirds. Each third contributes 5 rows to train and 1 each to val and test (seed 1), so every split covers a mix of low and high deforestation areas.

| Split | Tiles | % deforested |
|-------|-------|--------------|
| Train | 315 | 22.9% |
| Val | 63 | 23.8% |
| Test | 63 | 29.4% |

## Model & Training

- **Architectures:** U-Net (baseline) and U-Net++ (v2), both with a ResNet34 encoder, via [`segmentation-models-pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
- **Loss:** Dice + BCE with logits
- **Optimizer:** AdamW (weight decay 1e-4), mixed precision
  - Baseline: lr 1e-4, ReduceLROnPlateau, 30 epochs
  - v2: encoder lr 1e-4 and decoder lr 5e-4, cosine annealing, 60 epochs
- **Augmentation:** random flips and 90° rotations; v2 adds per-sample brightness/contrast jitter
- **Preprocessing:** per-channel standardization using train-set mean/std, tiles resized to 256×256
- **Inference:** decision threshold tuned on the validation set (0.4), optional 4-way flip test-time augmentation (TTA)

## Results

Test set, 63 tiles:

| Model | IoU | F1 | Precision | Recall |
|-------|-----|----|-----------|--------|
| U-Net (ResNet34) | 0.772 | 0.872 | 0.853 | 0.891 |
| U-Net++ (ResNet34) | 0.785 | 0.880 | 0.872 | 0.888 |
| U-Net++ + TTA | **0.811** | **0.895** | 0.903 | 0.888 |

### Quantization & Speed

Single-tile (1×3×256×256) inference. PyTorch FP32 latency is on CPU; OpenVINO results below are CPU only.

| Runtime | Precision | Latency (ms) | Throughput (tiles/s) | Speedup | Test IoU |
|---------|-----------|--------------|----------------------|---------|----------|
| PyTorch | FP32 | 254.6 | 3.9 | 1.00× | 0.785 |
| OpenVINO | FP32 | 157.2 | 6.7 | 1.62× | 0.785 |
| OpenVINO | FP16 | 160.5 | 6.7 | 1.59× | 0.785 |
| OpenVINO | INT8 | 105.0 | 9.9 | **2.42×** | 0.778 |

INT8 quantization (NNCF, 300 calibration tiles) costs only ~0.7 IoU points. INT8 with TTA reaches 0.804 IoU. The full benchmark, including GPU devices, is written to `benchmark_results.csv` by `quantized.ipynb`.

## Repository Structure

| File | Description |
|------|-------------|
| `data_exploration.ipynb` | Pulls imagery and labels from Earth Engine, tiles the study area, builds the train/val/test splits |
| `deforestation.ipynb` | <!-- TODO: one-line description --> |
| `quantized.ipynb` | Trains the baseline and U-Net++ models, evaluates with threshold tuning and TTA, exports to ONNX/OpenVINO, quantizes to INT8, and benchmarks |
| `requirements.txt` | Python dependencies |

## Getting Started

```bash
git clone https://github.com/gurliv21/deforestation-amazon.git
cd deforestation-amazon
pip install -r requirements.txt
```

**Run order:**

1. `data_exploration.ipynb` requires a [Google Earth Engine](https://earthengine.google.com/) account and project. It was run on Google Colab with outputs saved to Google Drive.
2. `quantized.ipynb` was run on Kaggle with a GPU. It expects the tiles, `splits/`, and `norm_stats.json` produced in step 1. Update `base_dir` at the top of the notebook to point to your data.

Additional packages used for deployment: `segmentation-models-pytorch`, `openvino`, `nncf`, `onnx`.


## Acknowledgments

- Hansen et al., *High-Resolution Global Maps of 21st-Century Forest Cover Change*, Science (2013)
- Copernicus Sentinel-2 data, via Google Earth Engine

## License

<!-- TODO: add a license -->
