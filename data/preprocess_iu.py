"""
Preprocesses IU-Xray into the same metadata format used by the pipeline.
Run after downloading:
    python data/preprocess_iu.py
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import random

random.seed(42)

REPORTS_DIR = Path("data/iu-xray/reports/ecgen-radiology")
IMAGES_DIR  = Path("data/iu-xray/images")
OUT_DIR     = Path("data/iu-xray")


def parse_report(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        findings  = ""
        impression = ""
        indication = ""

        for tag in root.iter("AbstractText"):
            label = tag.attrib.get("Label", "").upper()
            text  = (tag.text or "").strip()
            if not text:
                continue
            if label == "FINDINGS":
                findings = text
            elif label == "IMPRESSION":
                impression = text
            elif label == "INDICATION":
                indication = text

        # Find associated image
        images = []
        for fig in root.iter("parentImage"):
            img_id = fig.attrib.get("id", "")
            if img_id:
                images.append(img_id)

        return {
            "findings":   findings,
            "impression": impression,
            "indication": indication,
            "images":     images,
        }
    except Exception:
        return None


def build_metadata():
    xml_files = list(REPORTS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} XML report files")

    samples = []
    skipped = 0

    for xml_path in xml_files:
        parsed = parse_report(xml_path)
        if not parsed:
            skipped += 1
            continue
        if not parsed["findings"] and not parsed["impression"]:
            skipped += 1
            continue

        # Find a valid image for this report
        img_path = None
        for img_id in parsed["images"]:
            candidate = IMAGES_DIR / f"{img_id}.png"
            if candidate.exists():
                img_path = f"images/{img_id}.png"
                break

        if not img_path:
            # try finding any image with matching prefix
            stem = xml_path.stem
            for ext in [".png", ".jpg"]:
                candidate = IMAGES_DIR / f"{stem}{ext}"
                if candidate.exists():
                    img_path = f"images/{stem}{ext}"
                    break

        if not img_path:
            skipped += 1
            continue

        findings = parsed["findings"] or parsed["impression"]

        samples.append({
            "study_id":   xml_path.stem,
            "image_path": img_path,
            "findings":   findings,
            "impression": parsed["impression"] or findings[:100],
            "indication": parsed["indication"] or "Chest X-ray evaluation",
            "split":      None,
        })

    print(f"Valid samples: {len(samples)} | Skipped: {skipped}")

    # Shuffle and split 70/15/15
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    for i, s in enumerate(samples):
        if i < n_train:
            s["split"] = "train"
        elif i < n_train + n_val:
            s["split"] = "val"
        else:
            s["split"] = "test"

    for split in ["train", "val", "test"]:
        split_data = [s for s in samples if s["split"] == split]
        out_path = OUT_DIR / f"{split}_metadata.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  {split}: {len(split_data)} samples -> {out_path}")

    return samples


if __name__ == "__main__":
    print("=== Preprocessing IU-Xray ===\n")
    samples = build_metadata()
    print(f"\nDone. Total samples: {len(samples)}")
    print("Now update configs/default.yaml:")
    print('  data.mimic_cxr_path: "data/iu-xray"')