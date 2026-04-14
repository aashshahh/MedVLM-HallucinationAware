"""Creates dummy MIMIC-style data so you can run the full pipeline immediately."""
import json
import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_dataset(output_dir="data/mimic-cxr", n_samples=100):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(exist_ok=True)

    samples = []
    findings_pool = [
        "The lungs are clear without focal consolidation. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal.",
        "There is a small right pleural effusion. The lungs demonstrate mild pulmonary vascular congestion. Cardiomegaly is present.",
        "No acute cardiopulmonary abnormality. The heart size is normal. Pulmonary vasculature is unremarkable.",
        "Bilateral lower lobe opacities consistent with pneumonia or aspiration. No pneumothorax identified.",
        "Mild cardiomegaly without pulmonary edema. No pleural effusion. Bony structures are intact.",
    ]

    for i in range(n_samples):
        # Create dummy grayscale X-ray image
        arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        # Add some structure to make it look like a chest
        arr[60:200, 80:176] = np.random.randint(100, 180, (140, 96), dtype=np.uint8)
        arr[80:180, 100:156] = np.random.randint(30, 80, (100, 56), dtype=np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        img_path = f"images/dummy_{i:04d}.jpg"
        img.save(Path(output_dir) / img_path)

        samples.append({
            "study_id": f"s{i:08d}",
            "image_path": img_path,
            "findings": findings_pool[i % len(findings_pool)],
            "impression": findings_pool[i % len(findings_pool)].split(".")[0] + ".",
            "indication": ["Cough and fever", "Shortness of breath", "Chest pain", "Routine follow-up"][i % 4],
            "split": "train" if i < 70 else ("val" if i < 85 else "test")
        })

    for split in ["train", "val", "test"]:
        split_samples = [s for s in samples if s["split"] == split]
        with open(Path(output_dir) / f"{split}_metadata.json", "w") as f:
            json.dump(split_samples, f, indent=2)

    print(f"Created {n_samples} dummy samples in {output_dir}")
    print("train: 70 | val: 15 | test: 15")


def create_knowledge_corpus(output_path="data/knowledge_corpus.json", n_facts=200):
    """Build a small knowledge corpus of radiology finding sentences."""
    facts = [
        {"text": "The lungs are clear without focal consolidation."},
        {"text": "No pleural effusion or pneumothorax is identified."},
        {"text": "The cardiomediastinal silhouette is within normal limits."},
        {"text": "There is cardiomegaly with pulmonary vascular congestion."},
        {"text": "Bilateral lower lobe atelectasis is present."},
        {"text": "No acute osseous abnormalities are identified."},
        {"text": "There is a small right pleural effusion."},
        {"text": "The aortic knob is calcified."},
        {"text": "Pulmonary vasculature is unremarkable."},
        {"text": "There is patchy opacity in the right lower lobe."},
        {"text": "The trachea is midline."},
        {"text": "No pneumothorax is seen."},
        {"text": "The diaphragms are well-defined."},
        {"text": "Heart size is mildly enlarged."},
        {"text": "No free air below the right hemidiaphragm."},
        {"text": "The mediastinal and hilar contours are unremarkable."},
        {"text": "Mild pulmonary edema is present."},
        {"text": "No focal airspace disease is identified."},
        {"text": "There is a right-sided pleural effusion."},
        {"text": "Bony structures are intact."},
    ]
    # Repeat to reach n_facts
    extended = (facts * (n_facts // len(facts) + 1))[:n_facts]
    with open(output_path, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"Created knowledge corpus with {len(extended)} facts at {output_path}")


if __name__ == "__main__":
    create_dummy_dataset()
    create_knowledge_corpus()