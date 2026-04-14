import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BlipProcessor, BlipForConditionalGeneration
)
from peft import get_peft_model, LoraConfig, TaskType
from .get_knowledge import KnowledgeRetriever


class MedReportGenerator(nn.Module):
    """
    Production-ready: swap BLIP-2 for LLaVA-1.5 if you have 40GB+ VRAM.
    For local dev and GitHub demo: BLIP large works on 8GB.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        print("Loading BLIP for report generation...")
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float32
        ).to(device)

        # LoRA on text decoder
        lora_config = LoraConfig(
            r=cfg["model"]["lora_r"],
            lora_alpha=cfg["model"]["lora_alpha"],
            lora_dropout=cfg["model"]["lora_dropout"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        try:
            self.model.text_decoder = get_peft_model(self.model.text_decoder, lora_config)
            print("LoRA applied to text decoder")
        except Exception as e:
            print(f"LoRA skipped: {e}")

        self.retriever = KnowledgeRetriever(cfg["knowledge"]["corpus_path"], device)

    def generate(self, image_tensor: torch.Tensor, indication: str = "", max_new_tokens=150):
        # Retrieve and purify knowledge
        facts, scores = self.retriever.retrieve_from_image(
            image_tensor, top_k=self.cfg["knowledge"]["top_k_retrieval"]
        )
        purified = self.retriever.purify(
            facts, indication, top_k=self.cfg["knowledge"]["top_k_purified"]
        )
        knowledge_prefix = " ".join(purified[:3])  # top 3 facts as context

        prompt = f"a chest xray showing {knowledge_prefix}"

        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        pil_img = to_pil((image_tensor.cpu() * std + mean).clamp(0, 1))

        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                repetition_penalty=1.3,
                length_penalty=1.0,
            )
        report = self.processor.decode(out[0], skip_special_tokens=True)
        return report, purified, scores

    def forward(self, images, indications):
        reports = []
        retrieved_facts = []
        for img, ind in zip(images, indications):
            report, facts, _ = self.generate(img, ind)
            reports.append(report)
            retrieved_facts.append(facts)
        return reports, retrieved_facts