"""
SMART STRIVE - FastAPI inference service
Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs

NOTE: Output is an INDICATOR of possible stress concerns, not a diagnosis.
"""
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# load artifacts relative to this file, so it works from any folder
ART = Path(__file__).resolve().parent / "artifacts"
model = joblib.load(ART / "stress_model.joblib")
meta = json.loads((ART / "model_meta.json").read_text())
FEATURES: List[str] = meta["features"]
CLASSES: List[str] = meta["classes"]
RANGES: Dict[str, List[float]] = meta["feature_ranges"]
IMPORTANCE: Dict[str, float] = meta["feature_importance"]

# ---- thresholds (document these in Chapter 3) ----
CONCERN_STRONG = 0.6     # a factor this far into its "bad" range is a clear concern
CONCERN_WEAK = 0.4       # used only when no clear concern exists but stress is not Low
HIGH_RISK_CONF = 0.7     # minimum confidence for the high-risk flag
BORDERLINE_CONF = 0.6    # below this, a non-Low result is described as borderline

app = FastAPI(title="SMART STRIVE Stress API", version="1.1")

# Allow the React dev server to call this API from the browser.
# Add your frontend's address here if it runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Recommendation knowledge base (rule layer on top of ML prediction) ----
# 'direction': "high" -> concern when value is in upper part of range,
#              "low"  -> concern when value is in lower part of range.
KB = {
    "study_load": ("high", "Academic workload / deadlines",
                   ["Break tasks into daily targets (Pomodoro 25/5).",
                    "Rank tasks by deadline and weight; use a weekly planner.",
                    "Talk to your professor about deadline extensions if overloaded."]),
    "academic_performance": ("low", "Academic performance / exams",
                   ["Use spaced-repetition and active recall for reviewing.",
                    "Form a study group or ask for peer tutoring.",
                    "Visit your adviser to plan a recovery strategy."]),
    "future_career_concerns": ("high", "Future/career worries",
                   ["Schedule a career-guidance session with the Guidance Office.",
                    "Write down 3 small, concrete next steps for your goals."]),
    "teacher_student_relationship": ("low", "Teacher-student relationship",
                   ["Attend consultation hours to build rapport.",
                    "Prepare specific questions before approaching your professor."]),
    "basic_needs": ("low", "Financial / basic-needs concerns",
                   ["Ask about scholarships, financial assistance, or student jobs.",
                    "Create a simple weekly budget and track expenses."]),
    "social_support": ("low", "Limited social support",
                   ["Reach out to a trusted friend or family member this week.",
                    "Join a campus organization or study circle."]),
    "peer_pressure": ("high", "Peer pressure",
                   ["Practice assertive communication ('I prefer not to').",
                    "Spend time with supportive peers; consider talking to Guidance."]),
    "bullying": ("high", "Bullying / harassment",
                   ["Document incidents and report to the Guidance Office.",
                    "You do not have to handle this alone - book a counseling appointment."]),
    "extracurricular_activities": ("high", "Time management / overcommitment",
                   ["Review commitments and drop or delay the lowest-priority ones.",
                    "Block time for rest in your weekly schedule."]),
    "sleep_quality": ("low", "Poor sleep",
                   ["Keep a consistent sleep/wake schedule.",
                    "Avoid screens 30-60 min before bed; avoid all-nighters before exams."]),
    "living_conditions": ("low", "Living conditions",
                   ["Find a quiet study space (library, study room).",
                    "Ask the Guidance Office about campus support resources."]),
    "safety": ("low", "Feeling unsafe",
                   ["Talk to a trusted person or campus security.",
                    "Book a Guidance appointment to discuss safety concerns."]),
    "noise_level": ("high", "Noisy environment",
                   ["Use noise-cancelling headphones or study off-site/in the library."]),
}
GENERAL = {
    "Low": ["Keep up your current routines and check in with your mood regularly."],
    "Moderate": ["Consider monitoring your mood weekly and using the coping tips below."],
    "High": ["Your responses indicate a high level of possible stress. "
             "We strongly encourage booking an appointment with the Guidance Office."],
}
DISCLAIMER = ("This result indicates possible areas of concern only and is not a "
              "medical or psychological diagnosis.")


class StudentInput(BaseModel):
    student_id: str = Field(..., description="Pseudonymous ID, not the real name")
    features: Dict[str, float]


class Prediction(BaseModel):
    student_id: str
    stress_level: str
    confidence: float
    probabilities: Dict[str, float]
    stress_score: float
    top_concerns: List[str]
    recommendations: List[str]
    high_risk_flag: bool
    disclaimer: str


def is_concern(feat: str, value: float, direction: str) -> float:
    """Return a 0-1 'concern score' based on position within the feature's range."""
    lo, hi = RANGES[feat]
    if hi == lo:
        return 0.0
    pos = (value - lo) / (hi - lo)
    return pos if direction == "high" else 1 - pos


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model"], "features": FEATURES}


@app.post("/predict", response_model=Prediction)
def predict(inp: StudentInput):
    missing = [f for f in FEATURES if f not in inp.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    for f in FEATURES:
        lo, hi = RANGES[f]
        if not (lo <= inp.features[f] <= hi):
            raise HTTPException(422, f"'{f}' must be between {lo} and {hi}")

    X = pd.DataFrame([[inp.features[f] for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    level = CLASSES[idx]
    conf = float(proba[idx])

    # 0-100 stress score: probability-weighted level (Low 0, Moderate 0.5, High 1)
    score = round(float((proba[1] * 0.5 + proba[2] * 1.0) * 100), 1)

    # Rank concerns by how bad the student's value is. Model importance is only a
    # tie-breaker (clipped at 0 because permutation importance can be negative).
    # Each entry: (concern 0-1, importance, label, tips)
    scored = []
    for f, (direction, label, tips) in KB.items():
        if f in FEATURES:
            c = is_concern(f, inp.features[f], direction)
            if c >= CONCERN_WEAK:
                scored.append((round(c, 2), max(IMPORTANCE.get(f, 0), 0), label, tips))
    scored.sort(reverse=True)
    strong = [s for s in scored if s[0] >= CONCERN_STRONG]
    # if nothing is clearly problematic but stress is not Low, still show the top 2
    top = strong[:3] if strong else (scored[:2] if level != "Low" else [])

    recs = list(GENERAL[level])
    if level != "Low" and conf < BORDERLINE_CONF:
        recs.append("This result is borderline. Please repeat the check-in next week, "
                    "and consider talking to the Guidance Office if you are worried.")
    for _, _, _, tips in top:
        recs.extend(tips[:2])

    return Prediction(
        student_id=inp.student_id,
        stress_level=level,
        confidence=round(conf, 4),
        probabilities={c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
        stress_score=score,
        top_concerns=[t[2] for t in top],
        recommendations=recs,
        high_risk_flag=bool(level == "High" and conf >= HIGH_RISK_CONF),
        disclaimer=DISCLAIMER,
    )
