import os
import json
import threading
import time
from typing import List, Literal, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field

from verifact.indexer import ClinicalVectorIndexer
from verifact.retriever import PremiseRetriever
from verifact.rules import DeterministicConsistencyChecker
from verifact.local_nli import LocalNLI
# ---------------------------------------------------------
# Load Environment & Services
# ---------------------------------------------------------
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("[Warning] GROQ_API_KEY not found in environment or .env file.")

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Initialize modular indexing, retrieval, and rule checking pipelines
indexer = ClinicalVectorIndexer(collection_name="verifact_active_notes")
retriever = PremiseRetriever(indexer=indexer)
rule_checker = DeterministicConsistencyChecker()
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "verifact-nli"
)

local_nli = LocalNLI(MODEL_DIR)
print(f"[VeriFact] Local NLI model loaded from: {MODEL_DIR}")

# ---------------------------------------------------------
# Application Setup
# ---------------------------------------------------------
app = FastAPI(
    title="VeriFact: Discharge Summary Factuality Verifier",
    description="Automated two-stage clinical factuality audit engine with Qdrant vector retrieval, semantic NLI, and deterministic rule arbitration.",
    version="1.1.0"
)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class VerifyRequest(BaseModel):
    source_note: str = Field(..., description="Raw EHR progress note or admission record")
    discharge_summary: str = Field(..., description="Draft discharge summary or care plan to verify")
    note_id: Optional[str] = Field("active_note", description="Identifier for indexing into Qdrant")
    top_k_premises: Optional[int] = Field(2, description="Number of evidence premise chunks to retrieve per claim")

class AtomicClaim(BaseModel):
    id: str = Field(..., description="Unique claim identifier, e.g., 'c1'")
    claim_text: str = Field(..., description="Atomic clinical statement")
    entity_type: str = Field(..., description="Type of entity, e.g., 'medication', 'dosage', 'frequency', 'diagnosis', 'disposition'")
    value: Optional[str] = Field(None, description="Extracted numerical dosage or entity value, e.g., '80mg'")

class ClaimDecomposition(BaseModel):
    claims: List[AtomicClaim]

class FinalClaimVerdict(BaseModel):
    claim: AtomicClaim
    retrieved_premise: str
    nli_verdict: Literal["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]
    rule_verdict: Optional[Literal["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]] = None
    final_verdict: Literal["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]
    override_applied: bool = False
    reasoning: str

class VerifyResponse(BaseModel):
    note_id: str
    factuality_score: float
    total_claims: int
    entailed_claims: int
    contradicted_claims: int
    unsupported_claims: int
    overrides_count: int
    verdicts: List[FinalClaimVerdict]


# ---------------------------------------------------------
# Pipeline Methods
# ---------------------------------------------------------

def decompose_claims_groq(discharge_summary: str) -> List[AtomicClaim]:
    """
    Decomposes draft summary into atomic factual claims using Groq LLM with JSON structure.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API client is not configured.")

    system_prompt = (
        "You are a specialized clinical data extraction assistant.\n\n"
        "CRITICAL: When encountering a list of multiple medications or interventions (e.g., following a colon or separated by commas), "
        "you MUST extract every single medication, its dosage, and its frequency as separate, distinct atomic claims. "
        "Never truncate lists or ignore items following a colon.\n\n"
        "CRITICAL: Always bind medication attributes (dosage, frequency) to their specific temporal context "
        "(e.g., admission, hospital course, or discharge). If a summary states a medication is prescribed at discharge, "
        "the extracted claim MUST specify this context (e.g., 'Discharge Amlodipine dosage is 10mg' instead of just "
        "'Amlodipine dosage is 10mg'). This prevents attributes of discontinued admission drugs from being falsely validated."
    )

    prompt = f"""Decompose the following clinical discharge summary into atomic, verifiable factual claims.
Pay special attention to medications, exact dosages, frequencies, lab values, and discharge dispositions.

Extraction Rules:
1. CRITICAL: When encountering a list of multiple medications or interventions (e.g., following a colon or separated by commas), you MUST extract every single medication, its dosage, and its frequency as separate, distinct atomic claims. Never truncate lists or ignore items following a colon.
2. CRITICAL: Always bind medication attributes (dosage, frequency) to their specific temporal context (e.g., admission, hospital course, or discharge). If a summary states a medication is prescribed at discharge, the extracted claim MUST specify this context (e.g., 'Discharge Amlodipine dosage is 10mg' instead of just 'Amlodipine dosage is 10mg'). This prevents attributes of discontinued admission drugs from being falsely validated.

Discharge Summary:
{discharge_summary}

Respond ONLY with a valid JSON object matching this schema:
{{
  "claims": [
    {{
      "id": "c1",
      "claim_text": "...",
      "entity_type": "medication | dosage | frequency | diagnosis | procedure | disposition",
      "value": "extracted value like '80mg' or null"
    }}
  ]
}}"""

    try:
        resp = groq_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1200
        )
        data = json.loads(resp.choices[0].message.content.strip())
        claims_raw = data.get("claims", [])
        return [AtomicClaim(**c) for c in claims_raw]
    except Exception as e:
        print(f"[Error in Claim Decomposition]: {e}")
        # Fallback to simple sentence-split claims if decomposition encounters an issue
        sentences = [s.strip() for s in discharge_summary.split(".") if len(s.strip()) > 5]
        return [
            AtomicClaim(id=f"c{i+1}", claim_text=s, entity_type="clinical_statement", value=None)
            for i, s in enumerate(sentences)
        ]


def check_nli_groq(premise_text: str, claim_text: str) -> dict:
    """
    Executes semantic NLI comparing an atomic claim against retrieved Qdrant premise context.
    """
    if not groq_client:
        return {"verdict": "UNSUPPORTED", "reasoning": "Groq client not initialized"}

    prompt = f"""Premise:
{premise_text}

Hypothesis:
{claim_text}

You are a strict clinical factuality and Natural Language Inference (NLI) auditor.
Classify the relationship between Premise and Hypothesis into exactly one of three categories:
- ENTAILED: The Hypothesis is directly and completely supported by the Premise.
- CONTRADICTED: The Premise and Hypothesis discuss the SAME medication, test, or clinical topic, but state conflicting, contradictory, or mutually exclusive values (such as dosage differences, frequency mismatches, or opposite outcomes).
- UNSUPPORTED: The Hypothesis mentions medications, treatments, or claims that are NOT mentioned anywhere in the Premise. If an entity is absent or unmentioned in the Premise, classify as UNSUPPORTED, not Contradicted.

Return a JSON object:
{{"verdict": "ENTAILED" | "CONTRADICTED" | "UNSUPPORTED", "reasoning": "brief clinical rationale"}}"""

    try:
        resp = groq_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical NLI expert. Provide strict, evidence-grounded classifications."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        verdict = parsed.get("verdict", "UNSUPPORTED").upper()
        if verdict not in ["ENTAILED", "CONTRADICTED", "UNSUPPORTED"]:
            verdict = "UNSUPPORTED"
        return {
            "verdict": verdict,
            "reasoning": parsed.get("reasoning", "")
        }
    except Exception as e:
        return {"verdict": "UNSUPPORTED", "reasoning": f"Groq NLI error: {str(e)}"}


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "VeriFact Factuality Verifier",
        "model": DEFAULT_MODEL,
        "vector_store": "Qdrant In-Memory",
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.post("/verify", response_model=VerifyResponse)
def verify_discharge_summary(req: VerifyRequest):
    """
    Main verification pipeline endpoint:
      1. Vector Ingestion: Indexes source note into Qdrant vector store.
      2. Claim Decomposition: Extracts atomic clinical propositions from draft summary.
      3. Premise Retrieval: Retrieves top-k evidence chunks from Qdrant for each claim.
      4. Hybrid Evaluation:
         - Semantic NLI via Groq
         - Deterministic rule checking (dosage, unit, frequency, disposition)
         - Strict Priority Arbitration (rule contradiction overrides NLI entailment)
      5. Factuality Scoring.
    """
    note_id = req.note_id or f"note_{int(time.time())}"

    # Step 1: Ingest source note into Qdrant
    retriever.index_note(note_id=note_id, note_text=req.source_note)

    # Step 2: Atomic Claim Decomposition
    claims = decompose_claims_groq(req.discharge_summary)
    if not claims:
        return VerifyResponse(
            note_id=note_id,
            factuality_score=100.0,
            total_claims=0,
            entailed_claims=0,
            contradicted_claims=0,
            unsupported_claims=0,
            overrides_count=0,
            verdicts=[]
        )

    verdicts: List[FinalClaimVerdict] = []
    entailed_count = 0
    contradicted_count = 0
    unsupported_count = 0
    overrides_count = 0

    for claim in claims:
        # Step 3: Targeted premise retrieval from Qdrant
        retrieval_res = retriever.retrieve_premise_for_claim(
            claim_text=claim.claim_text,
            note_id=note_id,
            top_k=req.top_k_premises or 2
        )
        premise_text = retrieval_res["premise_text"]

        # Step 4a: Semantic NLI
        nli_out=local_nli.predict(premise_text,claim.claim_text)
        nli_verdict = nli_out["verdict"]
        reasoning = (
            "Trained local DeBERTa NLI prediction. "
            f"Class probabilities: {nli_out['probs']}"
        )


        # Step 4b: Deterministic Rule Check
        rule_verdict, rule_reason = rule_checker.check_claim(
            claim_text=claim.claim_text,
            premise_text=premise_text,
            entity_value=claim.value
        )

        # Step 4c: Deterministic Priority Arbitration
        final_verdict = nli_verdict
        override_applied = False

        if final_verdict == "ENTAILED" and rule_verdict == "CONTRADICTED":
            final_verdict = "CONTRADICTED"
            override_applied = True
            overrides_count += 1
            reasoning = f"{reasoning} | [DETERMINISTIC OVERRIDE]: {rule_reason}"
        elif rule_verdict == "CONTRADICTED" and final_verdict != "CONTRADICTED":
            final_verdict = "CONTRADICTED"
            override_applied = True
            overrides_count += 1
            reasoning = f"{reasoning} | [DETERMINISTIC OVERRIDE]: {rule_reason}"

        if final_verdict == "ENTAILED":
            entailed_count += 1
        elif final_verdict == "CONTRADICTED":
            contradicted_count += 1
        else:
            unsupported_count += 1

        verdicts.append(
            FinalClaimVerdict(
                claim=claim,
                retrieved_premise=premise_text,
                nli_verdict=nli_verdict,
                rule_verdict=rule_verdict,
                final_verdict=final_verdict,
                override_applied=override_applied,
                reasoning=reasoning
            )
        )

    # Step 5: Score Computation
    factuality_score = (entailed_count / len(claims)) * 100.0 if claims else 0.0

    return VerifyResponse(
        note_id=note_id,
        factuality_score=round(factuality_score, 2),
        total_claims=len(claims),
        entailed_claims=entailed_count,
        contradicted_claims=contradicted_count,
        unsupported_claims=unsupported_count,
        overrides_count=overrides_count,
        verdicts=verdicts
    )


# ---------------------------------------------------------
# Test Runner Block
# ---------------------------------------------------------
if __name__ == "__main__":
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(2)

    print("=" * 80)
    print("VERIFACT: DISCHARGE SUMMARY FACTUALITY VERIFIER (TEST RUNNER)")
    print("=" * 80)

    test_payload = {
        "note_id": "mimic_demo_001",
        "source_note": (
            "Patient presented with mild chest pain. Prescribed Aspirin 81mg once daily. "
            "Also taking 40mg Lisinopril for hypertension. Discharged to home in stable condition."
        ),
        "discharge_summary": (
            "Patient was prescribed Aspirin 81mg daily. "
            "Patient was prescribed Lisinopril 80mg daily. "
            "Patient was given Amoxicillin 500mg."
        ),
        "top_k_premises": 2
    }

    print("Sending test payload to http://127.0.0.1:8000/verify...")
    print(f"Source Note: {test_payload['source_note']}")
    print(f"Discharge Summary: {test_payload['discharge_summary']}\n")

    try:
        res = httpx.post("http://127.0.0.1:8000/verify", json=test_payload, timeout=40.0)
        print("RESPONSE STATUS:", res.status_code)
        print("RESPONSE BODY:")
        print(json.dumps(res.json(), indent=2))
    except Exception as exc:
        print(f"Error calling /verify: {exc}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    time.sleep(1)
