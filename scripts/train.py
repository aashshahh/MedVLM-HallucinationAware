import os
import sys
import json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import MIMICCXRDataset, collate_fn
from src.report import MedReportGenerator
from src.eval import NLGMetrics, CHAIRMetric


def load_cfg(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, device: torch.device):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    data_path = cfg["data"]["mimic_cxr_path"]
    print(f"Loading dataset from: {data_path}")
    train_ds = MIMICCXRDataset(data_path, split="train",
                           image_size=cfg["data"]["image_size"])
    val_ds   = MIMICCXRDataset(data_path, split="val",
                           image_size=cfg["data"]["image_size"])

    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = device.type == "cuda"

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl


def train():
    cfg = load_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    train_dl, val_dl = build_dataloaders(cfg, device)

    model = MedReportGenerator(cfg, device=device)
    model.model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=0.02,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    nlg = NLGMetrics()
    chair = CHAIRMetric()
    history = []

    max_steps = cfg["training"]["max_steps"]
    num_epochs = cfg["training"]["num_epochs"]
    log_every = cfg["training"].get("log_every", 20)

    global_step = 0
    best_loss = float("inf")

    to_pil = T.ToPILImage()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    print(f"\nStarting training for up to {max_steps} steps...")

    for epoch in range(num_epochs):
        if global_step >= max_steps:
            break

        model.model.train()
        running_loss = 0.0
        steps_this_epoch = 0

        for batch in train_dl:
            if global_step >= max_steps:
                break

            images = batch["image"]
            reports = batch["report"]
            indications = batch["indication"]

            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            for img, rep, ind in zip(images, reports, indications):
                img = img.detach().cpu()
                pil_img = to_pil((img * std + mean).clamp(0, 1))

                inputs = model.processor(
                    images=pil_img,
                    text=rep,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model.model(**inputs, labels=inputs["input_ids"])
                    loss_i = out.loss / len(images)

                batch_loss = batch_loss + loss_i

            scaler.scale(batch_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(batch_loss.detach().item())
            running_loss += loss_value
            steps_this_epoch += 1
            global_step += 1

            history.append({"step": global_step, "loss": loss_value})

            if global_step % log_every == 0:
                avg_loss = running_loss / max(steps_this_epoch, 1)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(model.model.state_dict(), "checkpoints/best_model.pt")

        epoch_avg = running_loss / max(steps_this_epoch, 1)
        print(f"Finished epoch {epoch + 1}/{num_epochs} | Avg Loss: {epoch_avg:.4f}")

    with open("outputs/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Best model saved to checkpoints/best_model.pt")

    print("\nRunning evaluation on val set...")
    evaluate(model, val_dl, nlg, chair, device)


@torch.no_grad()
def evaluate(model, val_dl, nlg, chair, device: torch.device):
    model.model.eval()
    all_results = []

    for idx, batch in enumerate(val_dl, start=1):
        img = batch["image"][0].to(device, non_blocking=True)
        ref = batch["report"][0]
        ind = batch["indication"][0]

        gen, facts, _ = model.generate(img, ind)
        nlg_scores = nlg.compute(gen, ref)
        chair_scores = chair.compute(gen, ref)

        all_results.append(
            {
                "study_id": batch["study_id"][0],
                "generated": gen,
                "reference": ref,
                "retrieved_facts": facts,
                **nlg_scores,
                **chair_scores,
            }
        )

        if idx % 20 == 0:
            print(f"  Evaluated {idx} samples...")

    metrics = ["bleu1", "bleu4", "rouge1", "rougeL", "chair"]
    print("\n=== Evaluation Results ===")
    for metric in metrics:
        vals = [r[metric] for r in all_results if metric in r]
        if vals:
            print(f"  {metric:15s}: {sum(vals) / len(vals):.4f}")
        else:
            print(f"  {metric:15s}: N/A")

    with open("outputs/eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFull results saved to outputs/eval_results.json")
    return all_results


if __name__ == "__main__":
    train()