# CULane-Based Real-Time Lane Detection

This project implements a lightweight ENet-21 inspired lane detection system in PyTorch, trained on the `CULane` dataset and designed to be fine-tuned later on your own Indian-road frames.

The project now uses:

- `CULane` as the base training dataset
- `Albumentations` for weather and image-quality augmentation
- `ENet-21 inspired` binary segmentation model
- `Custom fine-tuning` support for your own labeled Indian-road frames
- `Evaluation` support for comparing baseline, augmented, and fine-tuned models

## Why CULane Instead of TuSimple

`CULane` is a better base dataset for your final project because it is larger and much more challenging than TuSimple. It contains:

- urban roads
- shadows
- night scenes
- heavy traffic
- occlusion
- curved lanes
- more difficult lane visibility

This makes it a much better starting point before adapting to Indian roads.

## Project Files

```text
lane_detection/
├── dataset/
│   ├── __init__.py
│   ├── culane.py
│   └── custom.py
├── models/
│   ├── __init__.py
│   └── enet21.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
├── augmentations.py
├── config.py
├── evaluate.py
├── finetune.py
├── README.md
├── CULANE_FINETUNING_GUIDE.md
├── ML_TERMS_AND_FORMULAS.md
├── CODE_VIVA_GUIDE.md
├── requirements.txt
├── train.py
└── results/
```

## Expected CULane Structure

This code expects the standard CULane folder layout with official list files:

```text
CULane/
├── driver_23_30frame/
├── driver_37_30frame/
├── laneseg_label_w16/
├── list/
│   ├── train_gt.txt
│   ├── val_gt.txt
│   └── test.txt or test_gt.txt
```

Important note:

- `train_gt.txt` and `val_gt.txt` contain image path, mask path, and lane-existence flags.
- For binary segmentation, this project uses the segmentation mask path and converts every non-zero lane pixel into `1`.

## Installation

```bash
pip install -r lane_detection/requirements.txt
```

## Train on CULane

Train both baseline and augmented versions:

```powershell
python lane_detection/train.py --data-root D:\datasets\CULane --experiment both
```

Train only the weather-augmented version:

```powershell
python lane_detection/train.py --data-root D:\datasets\CULane --experiment augmented
```

Train with custom list files:

```powershell
python lane_detection/train.py --data-root D:\datasets\CULane --train-list list\train_gt.txt --val-list list\val_gt.txt --test-list list\val_gt.txt --epochs 40 --batch-size 8
```

If `test_gt.txt` is not available, the code automatically falls back to `val_gt.txt` for test-time reporting.

## Evaluate

```powershell
python lane_detection/evaluate.py --data-root D:\datasets\CULane --checkpoint lane_detection\results\weather_augmented\checkpoints\best_model.pth --split val
```

## Fine-Tune on Your Indian Frames

After you extract and label your 800 Indian-road frames, place them like this:

```text
custom_data/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── masks/
    ├── frame_0001.png
    ├── frame_0002.png
    └── ...
```

The mask file name must match the image stem.

Fine-tune from the best CULane checkpoint:

```powershell
python lane_detection/finetune.py --images-dir D:\custom_data\images --masks-dir D:\custom_data\masks --checkpoint lane_detection\results\weather_augmented\checkpoints\best_model.pth --epochs 20 --learning-rate 1e-4 --use-weather-augmentation
```

## Best Practical Workflow

1. Train on CULane first.
2. Keep the best augmented checkpoint.
3. Extract 800 diverse Indian-road frames.
4. Label them as binary masks.
5. Fine-tune using `finetune.py`.
6. Compare:
   CULane-only model vs fine-tuned Indian-road model.

## Study Material Added

To help with understanding and viva:

- [CULANE_FINETUNING_GUIDE.md](/D:/lane/lane_detection/CULANE_FINETUNING_GUIDE.md)
- [ML_TERMS_AND_FORMULAS.md](/D:/lane/lane_detection/ML_TERMS_AND_FORMULAS.md)
- [CODE_VIVA_GUIDE.md](/D:/lane/lane_detection/CODE_VIVA_GUIDE.md)
