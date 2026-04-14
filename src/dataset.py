import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from torchvision import transforms


def get_transform(image_size=224, split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class MIMICCXRDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=224):
        self.data_dir = Path(data_dir)
        self.transform = get_transform(image_size, split)
        meta_path = self.data_dir / f"{split}_metadata.json"
        with open(meta_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.data_dir / s["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "report": s["findings"],
            "indication": s.get("indication", "Chest X-ray evaluation"),
            "study_id": s["study_id"],
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "report": [b["report"] for b in batch],
        "indication": [b["indication"] for b in batch],
        "study_id": [b["study_id"] for b in batch],
    }