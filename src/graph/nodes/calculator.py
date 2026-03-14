from __future__ import annotations

import json
import re

from llama_index.core import Settings

from src.graph.state import AgentState


_AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[- ]?year(?:s)?[- ]?old", re.IGNORECASE)
_WEIGHT_RE = re.compile(r"(?:weight\s*(?:=|:|of)?\s*)?(\d+(?:\.\d+)?)\s*kg\b", re.IGNORECASE)
_SCR_RE = re.compile(
    r"(?:serum\s*creatinine|creatinine|scr|s\.cr|\bcr\b(?!\s*cl\b))\D*(\d+(?:\.\d+)?)\s*(mg\s*/\s*dL|mg/dL|umol\s*/\s*L|µmol\s*/\s*L|umol/L|µmol/L)?",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    text = query.replace("\u00a0", " ").replace("\u200b", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_age(query: str) -> float | None:
    match = _AGE_RE.search(query)
    return float(match.group(1)) if match else None


def _extract_weight_kg(query: str) -> float | None:
    match = _WEIGHT_RE.search(query)
    return float(match.group(1)) if match else None


def _extract_sex(query: str) -> str | None:
    lowered = query.lower()
    if "female" in lowered:
        return "female"
    if "male" in lowered:
        return "male"
    return None


def _extract_scr_mg_dl(query: str) -> tuple[float | None, str | None]:
    match = _SCR_RE.search(query)
    if not match:
        return None, None

    value = float(match.group(1))
    unit = (match.group(2) or "mg/dL").replace(" ", "").lower()
    if unit in {"umol/l", "µmol/l"}:
        return value / 88.4, "umol/L"
    return value, "mg/dL"


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, (int, float, str)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_sex(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"female", "male"}:
        return lowered
    return None


def _extract_with_llm(query: str) -> dict[str, object] | None:
    # Do not touch Settings.llm property here: it may auto-resolve to OpenAI
    # when no local model has been registered yet.
    llm = getattr(Settings, "_llm", None)
    if llm is None:
        return None

    prompt = (
        "Extract patient fields for Cockcroft-Gault as strict JSON only. "
        "Return exactly one JSON object with keys: age, weight, creatinine, sex, unit. "
        "Sex must be 'female' or 'male'. Unit must be 'mg/dL' or 'umol/L'. "
        "If unknown, set value to null. No markdown, no extra text.\n\n"
        f"Question: {query}"
    )
    try:
        response = llm.complete(prompt)
    except Exception:
        return None

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return None

    match = _JSON_BLOCK_RE.search(text)
    payload = match.group(0) if match else text
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_patient_fields(query: str) -> tuple[float | None, float | None, str | None, float | None, str | None]:
    age = _extract_age(query)
    weight_kg = _extract_weight_kg(query)
    sex = _extract_sex(query)
    scr_mg_dl, original_unit = _extract_scr_mg_dl(query)

    # Fallback extractor: use LLM as a pre-processor only for missing fields.
    if all(value is not None for value in (age, weight_kg, sex, scr_mg_dl)):
        return age, weight_kg, sex, scr_mg_dl, original_unit

    llm_fields = _extract_with_llm(query)
    if not llm_fields:
        return age, weight_kg, sex, scr_mg_dl, original_unit

    if age is None:
        age = _coerce_float(llm_fields.get("age"))
    if weight_kg is None:
        weight_kg = _coerce_float(llm_fields.get("weight"))
    if sex is None:
        sex = _coerce_sex(llm_fields.get("sex"))

    if scr_mg_dl is None:
        creatinine = _coerce_float(llm_fields.get("creatinine"))
        unit = str(llm_fields.get("unit") or "mg/dL").replace(" ", "").lower()
        if creatinine is not None:
            if unit in {"umol/l", "µmol/l"}:
                scr_mg_dl = creatinine / 88.4
                original_unit = "umol/L"
            else:
                scr_mg_dl = creatinine
                original_unit = "mg/dL"

    return age, weight_kg, sex, scr_mg_dl, original_unit


def _missing_fields(age: float | None, weight_kg: float | None, sex: str | None, scr_mg_dl: float | None) -> list[str]:
    missing: list[str] = []
    if age is None:
        missing.append("age")
    if weight_kg is None:
        missing.append("weight (kg)")
    if sex is None:
        missing.append("sex")
    if scr_mg_dl is None:
        missing.append("serum creatinine")
    return missing


def _fondaparinux_recommendation(query: str, crcl: float) -> str:
    if "fondaparinux" not in query.lower():
        return ""
    if crcl < 30.0:
        return "Based on CrCl < 30 mL/min, Fondaparinux is contraindicated for initial anticoagulation."
    if crcl < 50.0:
        return "Based on CrCl 30-50 mL/min, Fondaparinux is not contraindicated but should be used with caution due to increased exposure and bleeding risk."
    return "Based on CrCl >= 50 mL/min, Fondaparinux is not contraindicated on renal-function grounds."


def calculator(state: AgentState) -> AgentState:
    query = _normalize_query(state.get("query", ""))
    age, weight_kg, sex, scr_mg_dl, original_unit = _extract_patient_fields(query)

    missing = _missing_fields(age, weight_kg, sex, scr_mg_dl)
    if missing:
        draft = (
            "Cannot compute Cockcroft-Gault creatinine clearance because required inputs are missing: "
            + ", ".join(missing)
            + "."
        )
        if "fondaparinux" in query.lower():
            draft += " Fondaparinux initial-anticoagulation suitability cannot be determined without a calculable creatinine clearance."
        return {
            **state,
            "draft_answer": draft,
            "faithfulness_score": 1.0,
            "critic_feedback": "",
            "retrieved_docs": state.get("retrieved_docs", []),
        }

    assert age is not None and weight_kg is not None and sex is not None and scr_mg_dl is not None
    base = ((140.0 - age) * weight_kg) / (72.0 * scr_mg_dl)
    sex_factor = 0.85 if sex == "female" else 1.0
    crcl = base * sex_factor

    unit_note = ""
    if original_unit == "umol/L":
        unit_note = " Serum creatinine was converted from umol/L to mg/dL for the formula."

    draft = (
        "Cockcroft-Gault calculation:\n"
        f"- Age: {age:.0f} years\n"
        f"- Weight: {weight_kg:.1f} kg\n"
        f"- Sex: {sex}\n"
        f"- Serum creatinine: {scr_mg_dl:.3f} mg/dL\n"
        f"- Formula: CrCl = ((140 - age) * weight) / (72 * SCr) * sex_factor\n"
        f"- Substitution: ((140 - {age:.0f}) * {weight_kg:.1f}) / (72 * {scr_mg_dl:.3f}) * {sex_factor:.2f}\n"
        f"- Estimated creatinine clearance: {crcl:.1f} mL/min"
        f"{unit_note}"
    )

    fondaparinux_line = _fondaparinux_recommendation(query, crcl)
    if fondaparinux_line:
        draft = draft + "\n- " + fondaparinux_line

    return {
        **state,
        "draft_answer": draft,
        "faithfulness_score": 1.0,
        "critic_feedback": "",
        "retrieved_docs": state.get("retrieved_docs", []),
    }
