import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from pathlib import Path

def test_dataset_loads():
    from src.dataset import MIMICCXRDataset
    if not Path("data/mimic-cxr/train_metadata.json").exists():
        import pytest; pytest.skip("Dummy data not generated yet")
    ds = MIMICCXRDataset("data/mimic-cxr", split="train")
    assert len(ds) > 0
    sample = ds[0]
    assert "image" in sample and "report" in sample
    assert sample["image"].shape[0] == 3
def test_chair_metric():
    from src.eval import CHAIRMetric
    m   = CHAIRMetric()
    gen = "The lungs are clear. There is cardiomegaly."
    ref = "The lungs are clear. No acute cardiopulmonary abnormality."
    res = m.compute(gen, ref)
    assert "chair" in res
    assert 0.0 <= res["chair"] <= 1.0
    assert "cardiomegaly" in res["false_findings"]
    print(f"PASS test_chair_metric  (CHAIR={res['chair']})")
def test_knowledge_corpus_exists():
    path = Path("data/knowledge_corpus.json")
    if not path.exists():
        import pytest; pytest.skip("Run create_dummy_data.py first")
    with open(path) as f:
        corpus = json.load(f)
    assert len(corpus) > 0
    assert "text" in corpus[0]

def test_nlg_metrics():
    from src.eval import NLGMetrics
    m = NLGMetrics()
    result = m.compute(
        "The lungs are clear without focal consolidation.",
        "The lungs are clear without focal consolidation or pneumothorax."
    )
    assert "bleu1" in result and "rougeL" in result
    assert 0 <= result["bleu1"] <= 1

if __name__ == "__main__":
    test_chair_metric()
    test_nlg_metrics()
    print("All tests passed.")