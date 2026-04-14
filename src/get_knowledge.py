"""
Knowledge retrieval module.
Filename: src/get_knowledge.py
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor


class KnowledgeRetriever:
    def __init__(self, corpus_path: str, device="cpu"):
        self.device = device

        if not Path(corpus_path).exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}.\n"
                "Run this first:  python data/dummy.py"
            )

        print("  Loading CLIP for knowledge retrieval...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.corpus_texts = [item["text"] for item in self.corpus]

        print(f"  Encoding {len(self.corpus_texts)} knowledge facts...")
        self.corpus_embeddings = self._encode_corpus()

    def _get_text_emb(self, inputs):
        """Safely extract text embedding tensor from CLIP output."""
        out = self.model.get_text_features(**inputs)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        raise ValueError(f"Unexpected CLIP output type: {type(out)}")

    def _get_image_emb(self, inputs):
        """Safely extract image embedding tensor from CLIP output."""
        out = self.model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        raise ValueError(f"Unexpected CLIP output type: {type(out)}")

    def _encode_corpus(self) -> torch.Tensor:
        batch_size = 64
        all_embs = []
        for i in range(0, len(self.corpus_texts), batch_size):
            batch = self.corpus_texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                embs = self._get_text_emb(inputs)
            all_embs.append(F.normalize(embs, dim=-1))
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def retrieve_from_image(self, image_tensor: torch.Tensor, top_k: int = 10):
        inputs = self.processor(
            images=image_tensor,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        img_emb = self._get_image_emb(inputs)
        img_emb = F.normalize(img_emb, dim=-1)

        sims = (img_emb @ self.corpus_embeddings.T).squeeze(0)
        k = min(top_k, len(self.corpus_texts))
        top_indices = sims.topk(k).indices.tolist()
        facts  = [self.corpus_texts[i] for i in top_indices]
        scores = sims[top_indices].tolist()
        return facts, scores

    @torch.no_grad()
    def purify(self, retrieved_facts: list, indication: str, top_k: int = 5) -> list:
        if not indication or not retrieved_facts:
            return retrieved_facts[:top_k]

        all_texts = [indication] + retrieved_facts
        inputs = self.processor(
            text=all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        embs = self._get_text_emb(inputs)
        embs = F.normalize(embs, dim=-1)

        ctx       = embs[0]
        fact_embs = embs[1:]
        scores    = ctx @ fact_embs.T
        top_idx   = scores.topk(min(top_k, len(retrieved_facts))).indices.tolist()
        return [retrieved_facts[i] for i in top_idx]