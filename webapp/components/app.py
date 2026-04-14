"""
Gradio demo webapp.
Run from the project root:
    python webapp/components/app.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import yaml
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

from src.report import MedReportGenerator
from src.eval   import CHAIRMetric, GradCAMGrounding, NLGMetrics, HallucinationCorrector

CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading model on {DEVICE}...")
model     = MedReportGenerator(cfg, device=DEVICE)
model.eval()

chair_metric = CHAIRMetric()
nlg_metric   = NLGMetrics()
gradcam      = GradCAMGrounding(model.model)
corrector    = HallucinationCorrector()

_TFM = transforms.Compose([
    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
print("Model ready.\n")


def run_inference(image, indication, reference_report, show_gradcam):
    if image is None:
        return "Please upload a chest X-ray image.", None, "", ""

    pil_img    = Image.fromarray(image).convert("RGB")
    img_tensor = _TFM(pil_img)

    generated, retrieved_facts, retrieval_scores = model.generate(
        img_tensor.to(DEVICE),
        indication or "Chest X-ray evaluation",
    )

    # Post-hoc hallucination correction
    correction = corrector.correct(generated, retrieved_facts)
    corrected  = correction["corrected_report"]

    # Grad-CAM overlay
    overlay = None
    if show_gradcam:
        overlay = gradcam.overlay_on_image(img_tensor, alpha=0.45)

    # Metrics
    metrics_md = ""
    if reference_report and reference_report.strip():
        nlg_res   = nlg_metric.compute(generated, reference_report)
        chair_res = chair_metric.compute(generated, reference_report)
        chair_corrected = chair_metric.compute(corrected, reference_report)

        hallucinated_str = (
            ", ".join(chair_res["false_findings"])
            if chair_res["false_findings"] else "None detected"
        )
        missed_str = (
            ", ".join(chair_res["missed_detections"])
            if chair_res["missed_detections"] else "None"
        )
        reduction = correction["hallucination_reduction_pct"]

        metrics_md = (
            "### Evaluation metrics\n\n"
            "| Metric | Score |\n|--------|-------|\n"
            f"| BLEU-1  | {nlg_res['bleu1']} |\n"
            f"| BLEU-4  | {nlg_res['bleu4']} |\n"
            f"| ROUGE-1 | {nlg_res['rouge1']} |\n"
            f"| ROUGE-L | {nlg_res['rougeL']} |\n"
            f"| CHAIR (before correction) | {chair_res['chair']} |\n"
            f"| CHAIR (after correction)  | {chair_corrected['chair']} |\n"
            f"| Hallucination reduction   | {reduction}% |\n\n"
            f"**False findings (hallucinated):** {hallucinated_str}\n\n"
            f"**Missed detections:** {missed_str}\n"
        )
    else:
        metrics_md = (
            "### Evaluation metrics\n\n"
            "*Paste a reference report to compute BLEU / ROUGE / CHAIR scores.*\n\n"
            f"**Hallucination reduction after correction:** "
            f"{correction['hallucination_reduction_pct']}%"
        )

    # Retrieved knowledge display
    facts_md = "### Retrieved clinical knowledge\n\n"
    for i, (fact, score) in enumerate(
        zip(retrieved_facts[:5], retrieval_scores[:5])
    ):
        facts_md += f"**{i+1}.** {fact} &nbsp; `sim={score:.3f}`\n\n"

    # Show corrected report
    display_report = (
        f"**Original:**\n{generated}\n\n"
        f"**After hallucination correction:**\n{corrected}"
    )

    return display_report, overlay, metrics_md, facts_md


with gr.Blocks(title="MedVLM — Hallucination-Aware Report Generator") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px;">
      <h1 style="font-size:1.7rem; margin-bottom:4px;">
        Hallucination-Aware Medical Report Generator
      </h1>
      <p style="color:#64748b;">
        Chest X-ray → radiology report + Grad-CAM + hallucination detection + correction
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_in   = gr.Image(label="Chest X-ray", type="numpy", height=280)
            indication = gr.Textbox(label="Clinical indication",placeholder="e.g. Shortness of breath, rule out pneumonia",
            )
            reference  = gr.Textbox(label="Reference report (optional — enables CHAIR scoring)",lines=4,placeholder="Paste ground-truth findings here...",
            )
            show_cam   = gr.Checkbox(label="Show Grad-CAM overlay", value=True)
            submit_btn = gr.Button("Generate report", variant="primary", size="lg")

        with gr.Column(scale=1):
            report_out = gr.Textbox(label="Generated + corrected report",lines=12,
                )
            cam_out = gr.Image(label="Grad-CAM attention map", height=260)

    with gr.Row():
        metrics_out = gr.Markdown()
        facts_out   = gr.Markdown()

    submit_btn.click(
        fn=run_inference,inputs=[image_in, indication, reference, show_cam],outputs=[report_out, cam_out, metrics_out, facts_out],)

    gr.Markdown("""
    ---
    **Model:** BLIP-large + LoRA &nbsp;|&nbsp;
    **Knowledge:** CLIP retrieval + purification &nbsp;|&nbsp;
    **Eval:** CHAIR + BLEU + ROUGE + Grad-CAM + ECE  
    **Papers:** KERM (Zhao et al. 2026) · MedVH (Gu et al. 2025)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860,share=True,theme=gr.themes.Soft(),)