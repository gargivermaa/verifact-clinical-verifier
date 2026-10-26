"""
Fine-tune a DeBERTa NLI classifier for VeriFact (Milestone 4: the "trained model").

3-way labels (same as the rest of the pipeline): ENTAILED / CONTRADICTED / UNSUPPORTED.

Plain PyTorch loop (no Trainer/accelerate dependency).

Examples
    # Laptop CPU, small model (slow but works):
    python train/train_nli.py --model microsoft/deberta-v3-small --epochs 3 --batch_size 16

    # Colab/Kaggle GPU (recommended), starts from an MNLI-pretrained NLI checkpoint:
    python train/train_nli.py --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli --epochs 3 --batch_size 16

Outputs:  models/verifact-nli/  (model + tokenizer + train_log.json)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# order = (entailment, neutral, contradiction) so an MNLI-pretrained head starts aligned
LABELS = ["ENTAILED", "UNSUPPORTED", "CONTRADICTED"]
L2I = {l: i for i, l in enumerate(LABELS)}


def read_jsonl(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def make_loader(rows, tok, max_len, batch_size, shuffle):
    def collate(batch):
        enc = tok([b["premise"] for b in batch], [b["hypothesis"] for b in batch],
                  truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([L2I[b["label"]] for b in batch])
        return enc
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        ps += model(**batch).logits.argmax(-1).cpu().tolist()
        ys += y.cpu().tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-small",
                    help="HF model id or local dir")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="models/verifact-nli")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=None, help="cap #train pairs (quick CPU runs)")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    print(f"device={device}  model={a.model}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I,
        ignore_mismatched_sizes=True)  # tolerate a classifier head with a different #labels
    model.to(device)

    train_rows = read_jsonl(a.train, a.max_train)
    val_rows = read_jsonl(a.val)
    train_dl = make_loader(train_rows, tok, a.max_len, a.batch_size, True)
    val_dl = make_loader(val_rows, tok, a.max_len, a.batch_size * 2, False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(train_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log, best_f1, t0 = [], -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
            if step % 50 == 0 or step == len(train_dl):
                print(f"  epoch {ep} step {step}/{len(train_dl)} loss {run / step:.4f}  ({time.time() - t0:.0f}s)")

        ys, ps = evaluate(model, val_dl, device)
        acc, mf1 = accuracy_score(ys, ps), f1_score(ys, ps, average="macro")
        print(f"[epoch {ep}] val acc={acc:.4f} macro-F1={mf1:.4f}")
        log.append({"epoch": ep, "train_loss": run / len(train_dl), "val_acc": acc, "val_macro_f1": mf1})
        if mf1 > best_f1:
            best_f1 = mf1
            model.save_pretrained(out); tok.save_pretrained(out)
            print(f"  saved best -> {out}")

    ys, ps = evaluate(AutoModelForSequenceClassification.from_pretrained(out).to(device), val_dl, device)
    print("\nValidation report (best checkpoint, synthetic held-out notes):")
    print(classification_report(ys, ps, target_names=LABELS, digits=4))
    json.dump({"args": vars(a), "log": log, "best_val_macro_f1": best_f1,
               "train_pairs": len(train_rows), "val_pairs": len(val_rows)},
              open(out / "train_log.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
