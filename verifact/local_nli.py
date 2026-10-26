"""
Local (trained) NLI backend for VeriFact. Drop-in alternative to the Groq zero-shot NLI.

    nli = LocalNLI("models/verifact-nli")
    nli.predict(premise, claim) -> {"verdict": "ENTAILED|CONTRADICTED|UNSUPPORTED", "reasoning": str, "probs": {...}}
"""
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]


class LocalNLI:
    def __init__(self, model_dir: str = "models/verifact-nli", max_len: int = 256, device: str = None):
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"Trained NLI model not found at '{model_dir}'. Run: python train/train_nli.py")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_len = max_len
        # trust the label order saved in the checkpoint config
        self.id2label = {int(i): l for i, l in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict_batch(self, pairs: List[tuple]) -> List[Dict]:
        enc = self.tok([p for p, _ in pairs], [c for _, c in pairs], truncation=True,
                       max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
        out = []
        for row in probs:
            idx = int(row.argmax())
            dist = {self.id2label[i]: round(float(row[i]), 4) for i in range(len(row))}
            out.append({
                "verdict": self.id2label[idx],
                "reasoning": f"Trained DeBERTa NLI: {self.id2label[idx]} "
                             f"(p={float(row[idx]):.2f}; " + ", ".join(f"{k}={v:.2f}" for k, v in dist.items()) + ")",
                "probs": dist,
            })
        return out

    def predict(self, premise: str, claim: str) -> Dict:
        return self.predict_batch([(premise, claim)])[0]
