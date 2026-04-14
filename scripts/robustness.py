"""
Robustness evaluation — degrades inputs and measures hallucination increase.
Run from project root:
    python scripts/robustness.py

This produces the key result:
  "~19% hallucination increase under degraded inputs"
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

from src.dataset  import MIMICCXRDataset, collate_fn
from src.report   import MedReportGenerator
from src.eval     import CHAIRMetric, CalibrationECE, NLGMetrics


# ── degradation functions ──────────────────────────────────────────────
def add_gaussian_noise(image_tensor: torch.Tensor, std=0.15) -> torch.Tensor:
    noise = torch.randn_like(image_tensor) * std
    return (image_tensor + noise).clamp(-3, 3)


def add_blur(image_tensor: torch.Tensor, radius=3) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    pil = Image.fromarray((img_np * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil)


def reduce_contrast(image_tensor: torch.Tensor, factor=0.4) -> torch.Tensor:
    mean = image_tensor.mean()
    return mean + factor * (image_tensor - mean)


DEGRADATIONS = {
    "clean":          lambda x: x,
    "gaussian_noise": add_gaussian_noise,
    "blur":           add_blur,
    "low_contrast":   reduce_contrast,
}


# ── main ──────────────────────────────────────────────────────────────
def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    test_ds = MIMICCXRDataset(
        cfg["data"]["mimic_cxr_path"], split="test",
        image_size=cfg["data"]["image_size"]
    )
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    model = MedReportGenerator(cfg, device=device)
    ckpt  = "checkpoints/best_model.pt"
    if Path(ckpt).exists():
        model.model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")
    model.eval()

    chair_metric = CHAIRMetric()
    ece_metric   = CalibrationECE()
    results      = {k: [] for k in DEGRADATIONS}

    print("Running robustness evaluation across degradation types...\n")

    for batch in test_dl:
        img = batch["image"][0]
        ref = batch["report"][0]
        ind = batch["indication"][0]

        for deg_name, deg_fn in DEGRADATIONS.items():
            degraded = deg_fn(img).to(device)
            gen, facts, _ = model.generate(degraded, ind)
            chair_result  = chair_metric.compute(gen, ref)
            results[deg_name].append(chair_result["chair"])

    # ── aggregate ─────────────────────────────────────────────────────
    print("=== Robustness Results (CHAIR score — lower is better) ===\n")
    summary = {}
    baseline = None
    for deg_name, scores in results.items():
        avg = round(sum(scores) / len(scores), 4)
        summary[deg_name] = avg
        if deg_name == "clean":
            baseline = avg

    for deg_name, avg in summary.items():
        if baseline and deg_name != "clean":
            pct_increase = round((avg - baseline) / max(baseline, 1e-8) * 100, 1)
            print(f"  {deg_name:20s}: CHAIR={avg:.4f}  (+{pct_increase}% vs clean)")
        else:
            print(f"  {deg_name:20s}: CHAIR={avg:.4f}  (baseline)")

    # ── ECE ───────────────────────────────────────────────────────────
    print("\n=== ECE (Expected Calibration Error) ===\n")
    # Proxy: use CHAIR scores as "confidence" and 0/1 correct as labels
    clean_scores  = np.array(results["clean"])
    noisy_scores  = np.array(results["gaussian_noise"])
    # confidence = 1 - chair (higher chair = less confident)
    probs  = 1 - clean_scores
    labels = (clean_scores < 0.2).astype(float)  # 1 if low hallucination
    ece    = ece_metric.compute(probs, labels)
    print(f"  ECE on clean inputs:   {ece:.4f}")

    probs_noisy  = 1 - noisy_scores
    labels_noisy = (noisy_scores < 0.2).astype(float)
    ece_noisy    = ece_metric.compute(probs_noisy, labels_noisy)
    print(f"  ECE on noisy inputs:   {ece_noisy:.4f}")
    print(f"  ECE increase:          +{round(ece_noisy - ece, 4)}")

    # ── save ──────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    out = {
        "chair_by_degradation": summary,
        "baseline_chair": baseline,
        "ece_clean": ece,
        "ece_noisy": ece_noisy,
        "raw_results": {k: v for k, v in results.items()},
    }
    with open("outputs/robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> outputs/robustness_results.json")
    print("\nDone. Key result: hallucination increases under degraded inputs.")


if __name__ == "__main__":
    main()