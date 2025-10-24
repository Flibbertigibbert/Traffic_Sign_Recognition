# Traffic Sign Recognition with YOLOv8 (GTSRB • GTSDB)

This repository contains a reproducible pipeline for training and evaluating a traffic sign classifier/detector using [Ultralytics YOLOv8]. The project is based on the uploaded notebook `notebooks/TSR.ipynb` and the Kaggle dataset **traffic-sign-dataset-gtsrb-gtsdb**.

## Highlights
- Train a compact YOLOv8 model on GTSRB and GTSDB
- Ready-to-use data config and augmentation settings
- Export to ONNX for fast CPU inference
- Colab-friendly and local training instructions
- Clean project layout for collaboration

## Dataset
Source: Kaggle `adebolarabiu/traffic-sign-dataset-gtsrb-gtsdb`  
It bundles GTSRB images and labels, with splits for training and testing. Please review and comply with the dataset license on Kaggle.

## Project Structure
```
.
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── TSR.ipynb
├── configs/
│   └── gtsrb.yaml
├── scripts/
│   ├── train.sh
│   └── infer.py
└── .github/
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   └── feature_request.md
    └── PULL_REQUEST_TEMPLATE.md
```

## Quickstart

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Data
Download the dataset from Kaggle and arrange as:
```
data/
  gtsrb/
    images/
      Train/
      Test/
    labels/   # if using detection with YOLO-format labels
```
You can also use the notebook cell that downloads and unzips the dataset directly in Colab.

### 3) Configuration
Update paths in `configs/gtsrb.yaml` if your data directories differ.

### 4) Train
Using the Ultralytics CLI for **detection** example
```bash
yolo detect train model=yolov8n.pt data=configs/gtsrb.yaml imgsz=416 epochs=15 batch=16 patience=10
```
For **classification** use:
```bash
yolo classify train model=yolov8n-cls.pt data=configs/gtsrb.yaml imgsz=224 epochs=15 batch=32 patience=10
```

### 5) Evaluate
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=configs/gtsrb.yaml
# or classification:
yolo classify val model=runs/classify/train/weights/best.pt data=configs/gtsrb.yaml
```

### 6) Export to ONNX
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=416  # detection
# or classification
yolo export model=runs/classify/train/weights/best.pt format=onnx imgsz=224
```

## Inference

### Python
```python
from ultralytics import YOLO

# detection example
model = YOLO("runs/detect/train/weights/best.pt")
r = model.predict(source="path/to/image_or_folder", conf=0.25, imgsz=416)
for res in r:
    res.show()  # or res.save()

# classification example
model = YOLO("runs/classify/train/weights/best.pt")
r = model.predict(source="path/to/image_or_folder", imgsz=224)
print(r[0].probs)  # top probabilities
```

### Script
```bash
python scripts/infer.py --weights runs/detect/train/weights/best.pt --source path/to/images --conf 0.25 --imgsz 416
```

## Augmentation
The notebook sets a rich set of augmentation options inside the YAML file. You can tune these in `configs/gtsrb.yaml`.

## Classes
There are 43 classes in this dataset. Examples include:
- Speed Limit 20 kmph
- Speed Limit 30 kmph
- Speed Limit 50 kmph
- Speed Limit 60 kmph
- Speed Limit 70 kmph
- Speed Limit 80 kmph
- End of Speed Limit 80 kmph
- Speed Limit 100 kmph
- Speed Limit 120 kmph
- No Passing
- No Passing vehicle over 3.5 ton
- Right-of-way at intersection
- Priority road
- Yield
- Stop
- No vehicles
- Veh > 3.5 tons prohibited
- No entry
- General caution
- Dangerous curve left
... and more. See the YAML file for the complete list.

## Results
After training you will see artifacts in `runs/detect/train` or `runs/classify/train`. Typical files include:
- `results.csv` and `results.png`
- validation batch previews
- confusion matrix images
- `weights/best.pt` and optionally `best.onnx`

Record your key metrics here once you train on your hardware.

## Reproducing the Notebook
Open in Colab with this badge after pushing to your GitHub repo:


Replace `USER` and `REPO` with your GitHub handle and repository name.

## Citation
- Ultralytics YOLOv8
- Kaggle: traffic-sign-dataset-gtsrb-gtsdb by adebolarabiu

Please cite original GTSRB and GTSDB papers when publishing results.

## License
This template is provided under the MIT License. Review third-party licenses for datasets and models you use.

## Maintainers
- Adebola Rabiu
