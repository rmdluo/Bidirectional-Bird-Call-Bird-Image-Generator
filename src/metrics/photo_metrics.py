from pathlib import Path
from ultralytics import YOLO
from cleanfid import fid
from datasets.bird_photos import BirdPhotos
import torch


def calculate_fid(dir1, dir2):
    return fid.compute_fid(dir1, dir2)


def calculate_accuracy(
    photo_dir, split="train", yolo_path=".", batch_size=1024, device="cuda"
):
    model = YOLO(yolo_path)
    model = model.to(device)

    score = 0
    total = 0

    photo_dataset = BirdPhotos(photo_dir, split=split)

    dataloader = torch.utils.data.DataLoader(
        photo_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    for images, species in dataloader:
        images = images.to(device)
        predictions = model(images.to(torch.float32) / 255.0)
        prediction_names = [pred.names[pred.probs.top1] for pred in predictions]

        for pred, gt in zip(prediction_names, species):
            score += 1 if pred == gt else 0
        total += len(images)

    return score / total
