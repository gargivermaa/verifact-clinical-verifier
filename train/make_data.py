"""
Synthetic clinical NLI data generator for VeriFact Milestone 4.

Produces (premise, hypothesis, label) pairs with the SAME 3-way label space the
pipeline already uses: ENTAILED / CONTRADICTED / UNSUPPORTED.

Design choices (so the evaluation is honest):
  * Real MIMIC-IV needs PhysioNet credentialing, so training data is template-generated.
  * Drugs here are DISJOINT from every drug in the 40-claim M3 benchmark. The benchmark is
    therefore a held-out test set on unseen drug names, written by hand (not by these templates).
  * Contradictions include "near-miss" doses and cross-talk (drug A given drug B's dose),
    which are the failure cases the project is about.

Usage:
    python train/make_data.py --n_notes 1500 --seed 13 --out data
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Vocabulary. NOTE: none of these drugs appear in benchmarks/benchmark_m3.py
# ----------------------------------------------------------------------------
BENCH_DRUGS = {
    "metoprolol", "aspirin", "atorvastatin", "prednisone", "ceftriaxone", "ciprofloxacin",
    "insulin", "glargine", "lispro", "clopidogrel", "alteplase", "warfarin", "metformin",
    "vancomycin", "semaglutide", "amiodarone", "solu-medrol", "duoneb",
}

# name -> (plausible doses, unit, route)
DRUGS = {
    "Lisinopril": ([2.5, 5, 10, 20, 40], "mg", "PO"),
    "Amlodipine": ([2.5, 5, 10], "mg", "PO"),
    "Losartan": ([25, 50, 100], "mg", "PO"),
    "Carvedilol": ([3.125, 6.25, 12.5, 25], "mg", "PO"),
    "Furosemide": ([20, 40, 80], "mg", "PO"),
    "Spironolactone": ([12.5, 25, 50], "mg", "PO"),
    "Levothyroxine": ([25, 50, 75, 100, 125], "mcg", "PO"),
    "Omeprazole": ([20, 40], "mg", "PO"),
    "Pantoprazole": ([20, 40], "mg", "PO"),
    "Simvastatin": ([10, 20, 40, 80], "mg", "PO"),
    "Rosuvastatin": ([5, 10, 20, 40], "mg", "PO"),
    "Apixaban": ([2.5, 5], "mg", "PO"),
    "Enoxaparin": ([30, 40, 60, 80], "mg", "SC"),
    "Gabapentin": ([100, 300, 600, 800], "mg", "PO"),
    "Sertraline": ([25, 50, 100, 150], "mg", "PO"),
    "Levofloxacin": ([250, 500, 750], "mg", "PO"),
    "Azithromycin": ([250, 500], "mg", "PO"),
    "Amoxicillin": ([250, 500, 875], "mg", "PO"),
    "Doxycycline": ([50, 100], "mg", "PO"),
    "Hydrochlorothiazide": ([12.5, 25, 50], "mg", "PO"),
    "Glipizide": ([2.5, 5, 10], "mg", "PO"),
    "Sitagliptin": ([25, 50, 100], "mg", "PO"),
    "Montelukast": ([5, 10], "mg", "PO"),
    "Tamsulosin": ([0.4, 0.8], "mg", "PO"),
    "Allopurinol": ([100, 200, 300], "mg", "PO"),
    "Digoxin": ([62.5, 125, 250], "mcg", "PO"),
    "Diltiazem": ([30, 60, 120, 180], "mg", "PO"),
    "Hydralazine": ([10, 25, 50], "mg", "PO"),
    "Labetalol": ([100, 200, 300], "mg", "PO"),
    "Cefazolin": ([1, 2], "g", "IV"),
    "Piperacillin-tazobactam": ([3.375, 4.5], "g", "IV"),
    "Methylprednisolone": ([40, 60, 125], "mg", "IV"),
}
assert not ({d.lower() for d in DRUGS} & BENCH_DRUGS), "training drugs must be disjoint from benchmark"

# canonical frequency -> surface forms
FREQ = {
    "qd": ["daily", "once daily", "QD"],
    "bid": ["BID", "twice daily"],
    "tid": ["TID", "three times daily"],
    "qhs": ["at bedtime", "QHS"],
    "q8": ["every 8 hours", "Q8H"],
}
INDICATIONS = ["blood pressure control", "rate control", "volume management", "symptom control",
               "secondary prevention", "infection", "glycemic control", "pain management", "reflux"]

LABS = {  # name: (unit, low, high, decimals)
    "creatinine": ("mg/dL", 0.6, 3.8, 1),
    "potassium": ("mEq/L", 3.0, 5.8, 1),
    "sodium": ("mEq/L", 124, 146, 0),
    "hemoglobin": ("g/dL", 6.5, 14.5, 1),
    "white blood cell count": ("K/uL", 3.0, 22.0, 1),
    "blood glucose": ("mg/dL", 80, 480, 0),
    "troponin": ("ng/mL", 0.01, 6.5, 2),
    "INR": ("", 0.9, 4.5, 1),
}

DISPO = {
    "home": ["discharged home", "discharged to home in stable condition"],
    "snf": ["discharged to a skilled nursing facility", "transferred to a skilled nursing facility for rehabilitation"],
    "rehab": ["discharged to an acute inpatient rehabilitation facility", "transferred to inpatient rehabilitation"],
    "hospice": ["discharged to hospice care", "transferred to hospice"],
    "homehealth": ["discharged home with home health nursing", "discharged home with visiting nurse services"],
}
DISPO_CLAIM = {
    "home": ["The patient was discharged home.", "Patient went home at discharge."],
    "snf": ["The patient was discharged to a skilled nursing facility.", "Patient was transferred to a SNF."],
    "rehab": ["The patient was discharged to inpatient rehabilitation.", "Patient was sent to an acute rehab facility."],
    "hospice": ["The patient was discharged to hospice.", "Patient transitioned to hospice care."],
    "homehealth": ["The patient was discharged home with home health services.", "Patient went home with visiting nurse support."],
}

SPECIALTIES = ["cardiology", "nephrology", "pulmonology", "neurology", "endocrinology", "infectious disease"]
FOLLOWUP_WEEKS = [1, 2, 3, 4, 6, 8]

# (positive finding, negative finding) pairs -> contradiction by flipping
FINDINGS = [
    ("chest x-ray showed a right lower lobe infiltrate", "chest x-ray showed no acute infiltrate"),
    ("CT head showed no intracranial hemorrhage", "CT head showed an acute intracranial hemorrhage"),
    ("blood cultures showed no growth at 48 hours", "blood cultures grew gram-negative rods at 48 hours"),
    ("renal ultrasound showed no hydronephrosis", "renal ultrasound showed moderate hydronephrosis"),
    ("telemetry showed normal sinus rhythm", "telemetry showed atrial fibrillation with rapid ventricular response"),
    ("abdominal CT showed no free fluid", "abdominal CT showed a small amount of free fluid"),
]
# facts that can be asserted as UNSUPPORTED when absent from the note
ABSENT_FACT_CLAIMS = [
    "The patient underwent an echocardiogram showing an ejection fraction of {n}%.",
    "The patient received a blood transfusion of {k} units of packed red cells.",
    "The patient was discharged on {k} L/min of home oxygen.",
    "A urine culture grew Escherichia coli.",
    "The patient was started on a heparin infusion.",
    "Physical therapy recommended a rolling walker at discharge.",
    "A cardiology consult was obtained during the admission.",
    "The patient had a repeat CT scan on hospital day {k}.",
    "The patient was counseled on smoking cessation.",
    "A nasogastric tube was placed on hospital day {k}.",
]

DIAGNOSES = ["community-acquired pneumonia", "acute decompensated heart failure", "cellulitis",
             "acute kidney injury", "hypertensive urgency", "COPD exacerbation", "syncope",
             "gastrointestinal bleeding", "urinary tract infection", "chest pain"]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def other_dose(doses, cur, rng):
    """Return a different dose; half the time an adjacent (near-miss) one, else any other."""
    others = [d for d in doses if d != cur]
    if rng.random() < 0.5:
        i = doses.index(cur)
        near = [doses[j] for j in (i - 1, i + 1) if 0 <= j < len(doses)]
        if near:
            return rng.choice(near)
    if others:
        return rng.choice(others)
    return cur * 2  # only one dose in list


def med_claim(drug, dose, unit, route, freq_form, rng):
    t = rng.choice([
        "The patient was prescribed {d} {dose} {u} {f}.",
        "{d} was initiated at {dose}{u} {r} {f}.",
        "Discharge medications included {d} {dose} {u} {f}.",
        "The patient takes {d} {dose}{u} {f}.",
    ])
    return t.format(d=drug, dose=fmt(dose), u=unit, r=route, f=freq_form)


def build_example(rng, note_idx):
    # ---- choose note content ---------------------------------------------------
    n_meds = rng.choice([2, 2, 3, 3, 4])
    drug_names = rng.sample(list(DRUGS), n_meds)
    meds = []
    for name in drug_names:
        doses, unit, route = DRUGS[name]
        fk = rng.choice(list(FREQ))
        meds.append(dict(name=name, dose=rng.choice(doses), doses=doses, unit=unit, route=route,
                         fkey=fk, fform=rng.choice(FREQ[fk]),
                         discontinued=(rng.random() < 0.12)))
    labs = []
    for name in rng.sample(list(LABS), rng.choice([1, 2])):
        unit, lo, hi, dec = LABS[name]
        val = round(rng.uniform(lo, hi), dec)
        labs.append(dict(name=name, unit=unit, val=val, dec=dec, lo=lo, hi=hi))
    findings = rng.sample(FINDINGS, rng.choice([0, 1, 1, 2]))
    findings = [(f, rng.random() < 0.5) for f in findings]  # (pair, use_positive_first)
    dispo = rng.choice(list(DISPO))
    has_followup = rng.random() < 0.6
    follow = (rng.choice(SPECIALTIES), rng.choice(FOLLOWUP_WEEKS)) if has_followup else None

    # ---- render note -------------------------------------------------------------
    s = [f"{rng.randint(34, 89)}-year-old {rng.choice(['male', 'female'])} admitted with {rng.choice(DIAGNOSES)}."]
    for m in meds:
        if m["discontinued"]:
            s.append(f"{m['name']} was discontinued during the hospital course.")
        else:
            s.append(f"{m['name']} {fmt(m['dose'])} {m['unit']} {m['route']} {m['fform']} was "
                     f"{rng.choice(['started', 'initiated', 'continued'])} for {rng.choice(INDICATIONS)}.")
    for l in labs:
        u = f" {l['unit']}" if l["unit"] else ""
        s.append(f"Admission {l['name']} was {l['val']}{u}.")
    for (pos, neg), use_pos in findings:
        s.append(f"The {pos if use_pos else neg}.")
    s.append(f"The patient was {rng.choice(DISPO[dispo])}.")
    if follow:
        s.append(f"Follow-up with {follow[0]} was arranged in {follow[1]} weeks.")
    rest = s[1:]
    rng.shuffle(rest)  # keep intro first; order of the other facts is irrelevant to the label
    note = " ".join([s[0]] + rest)

    # ---- claims ----------------------------------------------------------------
    active = [m for m in meds if not m["discontinued"]]
    ex = []

    def ENT():
        kind = rng.choice(["med", "med", "lab", "dispo", "followup", "finding"])
        if kind == "med" and active:
            m = rng.choice(active)
            alt = rng.choice(FREQ[m["fkey"]])
            return med_claim(m["name"], m["dose"], m["unit"], m["route"], alt, rng)
        if kind == "lab" and labs:
            l = rng.choice(labs)
            u = f" {l['unit']}" if l["unit"] else ""
            return f"The admission {l['name']} was {l['val']}{u}."
        if kind == "finding" and findings:
            (pos, neg), use_pos = rng.choice(findings)
            return f"The {pos if use_pos else neg}."
        if kind == "followup" and follow:
            return f"Follow-up with {follow[0]} was scheduled within {follow[1]} weeks."
        # fallback (kind == "dispo", or the chosen kind had no matching fact in this note)
        if rng.random() < 0.5:
            return rng.choice(DISPO_CLAIM[dispo])
        m = rng.choice(meds)
        if m["discontinued"]:
            return f"{m['name']} was stopped during the admission."
        return med_claim(m["name"], m["dose"], m["unit"], m["route"], m["fform"], rng)

    def CON():
        kinds = ["dose", "dose", "freq", "swap", "dispo", "lab", "finding", "status", "followup"]
        rng.shuffle(kinds)
        for kind in kinds:
            if kind == "dose" and active:
                m = rng.choice(active)
                nd = other_dose(m["doses"], m["dose"], rng)
                return med_claim(m["name"], nd, m["unit"], m["route"], m["fform"], rng)
            if kind == "freq" and active:
                m = rng.choice(active)
                nk = rng.choice([k for k in FREQ if k != m["fkey"]])
                return med_claim(m["name"], m["dose"], m["unit"], m["route"], rng.choice(FREQ[nk]), rng)
            if kind == "swap":
                pool = [m for m in active]
                if len(pool) >= 2:
                    a, b = rng.sample(pool, 2)
                    if (a["dose"], a["unit"]) != (b["dose"], b["unit"]):
                        return med_claim(a["name"], b["dose"], b["unit"], a["route"], a["fform"], rng)
            if kind == "dispo":
                nd = rng.choice([d for d in DISPO if d != dispo])
                return rng.choice(DISPO_CLAIM[nd])
            if kind == "lab" and labs:
                l = rng.choice(labs)
                factor = rng.choice([0.5, 0.7, 0.85, 1.15, 1.3, 1.6, 2.0])
                nv = round(l["val"] * factor, l["dec"])
                if nv == l["val"]:
                    nv = round(l["val"] + 10 ** (-l["dec"]) * 3, l["dec"])
                u = f" {l['unit']}" if l["unit"] else ""
                return f"The admission {l['name']} was {nv}{u}."
            if kind == "finding" and findings:
                (pos, neg), use_pos = rng.choice(findings)
                return f"The {neg if use_pos else pos}."
            if kind == "status":
                if any(m["discontinued"] for m in meds):
                    m = rng.choice([m for m in meds if m["discontinued"]])
                    return med_claim(m["name"], rng.choice(m["doses"]), m["unit"], m["route"], rng.choice(FREQ[m["fkey"]]), rng)
                if active:
                    m = rng.choice(active)
                    return f"{m['name']} was discontinued during the admission."
            if kind == "followup" and follow:
                wk = rng.choice([w for w in FOLLOWUP_WEEKS if w != follow[1]])
                return f"Follow-up with {follow[0]} was scheduled within {wk} weeks."
        m = active[0] if active else meds[0]
        return med_claim(m["name"], other_dose(m["doses"], m["dose"], rng), m["unit"], m["route"], m["fform"], rng)

    def UNS():
        kind = rng.choice(["drug", "drug", "fact", "followup"])
        if kind == "drug":
            pool = [d for d in DRUGS if d not in drug_names]
            name = rng.choice(pool)
            doses, unit, route = DRUGS[name]
            fk = rng.choice(list(FREQ))
            return med_claim(name, rng.choice(doses), unit, route, rng.choice(FREQ[fk]), rng)
        if kind == "followup" and not follow:
            return f"Follow-up with {rng.choice(SPECIALTIES)} was scheduled within {rng.choice(FOLLOWUP_WEEKS)} weeks."
        t = rng.choice(ABSENT_FACT_CLAIMS)
        return t.format(n=rng.choice([20, 25, 30, 35, 40, 55]), k=rng.randint(1, 4))

    rows = []
    for label, fn, count in (("ENTAILED", ENT, 2), ("CONTRADICTED", CON, 2), ("UNSUPPORTED", UNS, 2)):
        seen = set()
        for _ in range(count):
            c = fn()
            if c in seen:
                continue
            seen.add(c)
            rows.append({"note_id": f"syn_{note_idx}", "premise": note, "hypothesis": c, "label": label})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_notes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    notes = [build_example(rng, i) for i in range(a.n_notes)]
    n_val = max(1, int(a.n_notes * a.val_frac))
    val_rows = [r for grp in notes[:n_val] for r in grp]       # split BY NOTE -> no premise leakage
    train_rows = [r for grp in notes[n_val:] for r in grp]
    rng.shuffle(train_rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} pairs  {dict(Counter(r['label'] for r in rows))}")

    print("\nSample pairs:")
    for r in rng.sample(train_rows, 5):
        print(f"  [{r['label']}] {r['hypothesis']}")
    print(f"\nWrote {out}/train.jsonl and {out}/val.jsonl")


if __name__ == "__main__":
    main()
