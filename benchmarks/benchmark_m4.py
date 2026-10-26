"""
Milestone 4 benchmark: TRAINED NLI model vs. zero-shot LLM, each with and without the rule layer,
on the same 40 hand-written claims used in Milestone 3 (held-out: the model never trains on them,
and its training drugs are disjoint from the benchmark drugs).

    python benchmarks/benchmark_m4.py                       # trained model (+rules)
    python benchmarks/benchmark_m4.py --with-groq           # also re-run Groq zero-shot (needs GROQ_API_KEY)
    python benchmarks/benchmark_m4.py --no-retrieval        # skip Qdrant/MiniLM (notes are single-chunk anyway)

Writes benchmarks/benchmark_m4_results.json
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifact.rules import DeterministicConsistencyChecker  # noqa: E402

LABELS = ["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]


def load_benchmark():
    """Read MIMIC_EVAL_DATASET out of benchmark_m3.py without importing it (no groq/qdrant needed)."""
    src = (ROOT / "benchmarks" / "benchmark_m3.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MIMIC_EVAL_DATASET" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("MIMIC_EVAL_DATASET not found in benchmark_m3.py")


def arbitrate(nli_verdict, rule_verdict):
    """Same priority rule as main.py: a rule CONTRADICTED always wins."""
    return "CONTRADICTED" if rule_verdict == "CONTRADICTED" else nli_verdict


def groq_nli(client, model, premise, claim):
    prompt = f"""Premise:
{premise}

Hypothesis:
{claim}

Determine the logical relationship between the Premise and Hypothesis.
Select exactly one label:
- ENTAILED: The hypothesis is directly supported by the premise.
- CONTRADICTED: The hypothesis contradicts the premise.
- UNSUPPORTED: The premise does not contain sufficient information to verify the hypothesis.

Return a JSON object:
{{"verdict": "ENTAILED" | "CONTRADICTED" | "UNSUPPORTED", "reasoning": "brief explanation"}}"""
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=800, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You are a Natural Language Inference classifier."},
                      {"role": "user", "content": prompt}])
        v = json.loads(r.choices[0].message.content).get("verdict", "UNSUPPORTED").upper()
        return v if v in LABELS else "UNSUPPORTED"
    except Exception as e:  # keep going, but make failures visible
        print(f"   [groq error] {e}")
        return "UNSUPPORTED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "models" / "verifact-nli"))
    ap.add_argument("--with-groq", action="store_true")
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "benchmark_m4_results.json"))
    a = ap.parse_args()

    from verifact.local_nli import LocalNLI
    nli = LocalNLI(a.model_dir)
    rules = DeterministicConsistencyChecker()
    data = load_benchmark()

    retriever = None
    if not a.no_retrieval:
        from verifact.retriever import PremiseRetriever
        retriever = PremiseRetriever()
        for item in data:
            retriever.index_note(item["note_id"], item["note_text"])
    else:
        print("[info] --no-retrieval: using the full note as premise")

    groq_client = groq_model = None
    if a.with_groq:
        from dotenv import load_dotenv
        from groq import Groq
        load_dotenv(ROOT / ".env")
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    rows = []
    for item in data:
        for c in item["claims"]:
            premise = (retriever.retrieve_premise_for_claim(c["claim"], item["note_id"], top_k=2)["premise_text"]
                       if retriever else item["note_text"])
            t = nli.predict(premise, c["claim"])
            rv, rr = rules.check_claim(c["claim"], premise)
            row = {"note_id": item["note_id"], "claim": c["claim"], "ground_truth": c["ground_truth"],
                   "trained_pred": t["verdict"], "trained_probs": t["probs"], "rule_pred": rv, "rule_reason": rr,
                   "trained_plus_rules_pred": arbitrate(t["verdict"], rv)}
            if groq_client:
                g = groq_nli(groq_client, groq_model, premise, c["claim"])
                row["groq_pred"] = g
                row["groq_plus_rules_pred"] = arbitrate(g, rv)
            rows.append(row)
            print(f"GT={c['ground_truth']:<12} trained={t['verdict']:<12} +rules={row['trained_plus_rules_pred']:<12} {c['claim'][:60]}")

    y = [r["ground_truth"] for r in rows]
    configs = {"Trained DeBERTa (NLI only)": "trained_pred",
               "Trained DeBERTa + Rules": "trained_plus_rules_pred"}
    if a.with_groq:
        configs.update({"Groq zero-shot (NLI only)": "groq_pred", "Groq zero-shot + Rules": "groq_plus_rules_pred"})

    print("\n" + "=" * 96)
    print(f"{'Config':<30}{'Acc':>8}{'MacroF1':>10}{'F1 ENT':>9}{'F1 CON':>9}{'F1 UNS':>9}{'#correct':>10}")
    print("-" * 96)
    summary = {}
    for name, key in configs.items():
        p = [r[key] for r in rows]
        f1s = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
        summary[name] = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
                         "per_class_f1": dict(zip(LABELS, map(float, f1s))), "n_correct": sum(a_ == b for a_, b in zip(y, p)),
                         "report": classification_report(y, p, labels=LABELS, output_dict=True, zero_division=0),
                         "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, p, labels=LABELS).tolist()}
        s = summary[name]
        print(f"{name:<30}{s['accuracy']*100:>7.1f}%{s['macro_f1']*100:>9.1f}%{f1s[0]*100:>8.1f}%{f1s[1]*100:>8.1f}%{f1s[2]*100:>8.1f}%{s['n_correct']:>7}/{len(y)}")
    print("=" * 96)

    for name, key in list(configs.items())[:2]:
        print(f"\nConfusion matrix - {name}   (rows = truth, cols = pred; order {LABELS})")
        for lab, line in zip(LABELS, summary[name]["confusion_matrix_rows_true_cols_pred"]):
            print(f"  {lab:<13}{line}")

    ov = [r for r in rows if r["trained_pred"] != r["trained_plus_rules_pred"]]
    print(f"\nRule layer changed the trained model's verdict on {len(ov)} claim(s):")
    for r in ov:
        ok = "FIXED" if r["trained_plus_rules_pred"] == r["ground_truth"] else "WRONG"
        print(f"  [{ok}] {r['claim'][:70]}  ({r['trained_pred']} -> {r['trained_plus_rules_pred']})  {r['rule_reason']}")
    wrong = [r for r in rows if r["trained_plus_rules_pred"] != r["ground_truth"]]
    print(f"\nStill wrong after Trained+Rules: {len(wrong)}")
    for r in wrong:
        print(f"  GT={r['ground_truth']} pred={r['trained_plus_rules_pred']}  {r['claim'][:80]}")

    Path(a.out).write_text(json.dumps({"n_claims": len(rows), "retrieval": not a.no_retrieval,
                                       "summary": summary, "detailed_results": rows}, indent=2))
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
