"""Run evaluation only — loads saved checkpoint."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, json, torch
from torch.utils.data import DataLoader
from src.dataset import MIMICCXRDataset, collate_fn
from src.report import MedReportGenerator
from src.eval import NLGMetrics, CHAIRMetric

def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = MIMICCXRDataset(cfg["data"]["mimic_cxr_path"], split="test",
                              image_size=cfg["data"]["image_size"])
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    model = MedReportGenerator(cfg, device=device)
    ckpt = "checkpoints/best_model.pt"
    if os.path.exists(ckpt):
        model.model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint from {ckpt}")
    model.eval()

    nlg = NLGMetrics()
    chair = CHAIRMetric()
    results = []

    for i, batch in enumerate(test_dl):
        img = batch["image"][0]
        ref = batch["report"][0]
        ind = batch["indication"][0]
        gen, facts, scores = model.generate(img.to(device), ind)
        results.append({
            "id": batch["study_id"][0],
            "generated": gen,
            "reference": ref,
            "top_facts": facts[:3],
            **nlg.compute(gen, ref),
            **chair.compute(gen, ref),
        })
        print(f"[{i+1}/{len(test_dl)}] CHAIR: {results[-1]['chair']:.3f} | BLEU-4: {results[-1]['bleu4']:.3f}")

    avg = lambda k: round(sum(r[k] for r in results) / len(results), 4)
    print("\n=== Test Set Results ===")
    for m in ["bleu1","bleu4","rouge1","rougeL","chair"]:
        print(f"  {m:15s}: {avg(m)}")

    with open("outputs/test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()