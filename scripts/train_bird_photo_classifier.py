import os, shutil
from pathlib import Path
from ultralytics import YOLO

INPUT_DATASET_DIR = Path("../data/iNaturalistPrunedCropped")
TEMP_DATASET_DIR = Path("../data/temp_iNaturalistPrunedCropped")

# Load a model
model = YOLO("yolo26m-cls.pt")

# Make temporary dataset from pruned + cropped w/ YOLO
if os.path.exists(TEMP_DATASET_DIR):
    shutil.rmtree(TEMP_DATASET_DIR)
os.makedirs(TEMP_DATASET_DIR)
shutil.copytree(INPUT_DATASET_DIR / "train", TEMP_DATASET_DIR / "train")
shutil.copytree(INPUT_DATASET_DIR / "val", TEMP_DATASET_DIR / "test")

# Train the model
results = model.train(
    data=TEMP_DATASET_DIR,
    epochs=100,
    imgsz=128,
    device="cuda:1",
    batch=1024,
    cache="ram",
    save_dir="../models/yolo26_bird_classifier",
)

# Remove temporary dataset
shutil.rmtree(TEMP_DATASET_DIR)
