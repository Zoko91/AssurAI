"""
Tests unitaires et fonctionnels du moteur d'analyse des sinistres.
Vérifie les 8 cas du Golden Dataset :
- Cas de référence 1 : Vélo volé dans local commun non fermé à clé -> NON_COUVERT
- Cas de référence 2 : Bougie fondue sur table de jardin -> NON_COUVERT
- Cas de contrôle : Couvert, Non couvert, Ambigu
"""

import json
from pathlib import Path
import pytest

from src.schemas import ClaimInput, DecisionVerdict
from src.contract_store import ContractStore
from src.engine import ClaimEvaluationEngine


@pytest.fixture
def engine():
    store = ContractStore("data/processed/clauses_db.json")
    return ClaimEvaluationEngine(store)


@pytest.fixture
def benchmark_cases():
    data = json.loads(Path("data/eval/benchmark_sinistres.json").read_text(encoding="utf-8"))
    return [ClaimInput(**c) for c in data]


def test_reference_case_1_bike_unlocked_room(engine, benchmark_cases):
    """Cas 1 de référence : Vélo volé dans local commun non fermé à clé."""
    case = next(c for c in benchmark_cases if c.claim_id == "CAS-HAB-01")
    result = engine.evaluate_claim(case)

    assert result.verdict == DecisionVerdict.NON_COUVERT
    assert "Article 3.3" in result.applicable_clauses
    assert len(result.triggered_exclusions) > 0 or len(result.unfulfilled_conditions) > 0
    assert result.confidence_score >= 0.90
    assert any("Article 3.3" in ext for ext in result.cited_extracts)


def test_reference_case_2_candle_melting_table(engine, benchmark_cases):
    """Cas 2 de référence : Bougie fondue sur table de jardin."""
    case = next(c for c in benchmark_cases if c.claim_id == "CAS-HAB-02")
    result = engine.evaluate_claim(case)

    assert result.verdict == DecisionVerdict.NON_COUVERT
    assert "Article 2.2" in result.applicable_clauses or "Article 2.1" in result.applicable_clauses
    assert len(result.triggered_exclusions) > 0
    assert result.confidence_score >= 0.90
    assert any("bougie" in ext.lower() or "chaleur" in ext.lower() for ext in result.cited_extracts)


def test_control_case_burglary_covered(engine, benchmark_cases):
    """Cas de contrôle : Cambriolage avec effraction dans logement verrouillé -> COUVERT."""
    case = next(c for c in benchmark_cases if c.claim_id == "CTRL-CAS-03")
    result = engine.evaluate_claim(case)

    assert result.verdict == DecisionVerdict.COUVERT
    assert "Article 3.1" in result.applicable_clauses
    assert len(result.triggered_exclusions) == 0
    assert result.confidence_score >= 0.85


def test_control_case_ambiguous_status(engine, benchmark_cases):
    """Cas de contrôle : Intempéries sur salon de jardin avec options non vérifiées -> AMBIGU_A_VERIFIER."""
    case = next(c for c in benchmark_cases if c.claim_id == "CTRL-CAS-08")
    result = engine.evaluate_claim(case)

    assert result.verdict == DecisionVerdict.AMBIGU_A_VERIFIER
    assert result.confidence_score < 0.80
    assert "Dispositions Particulières" in result.summary_reasoning


def test_all_benchmark_cases_run_without_error(engine, benchmark_cases):
    """Évalue l'ensemble des cas du golden dataset et s'assure de la cohérence globale."""
    for case in benchmark_cases:
        res = engine.evaluate_claim(case)
        assert res.claim_id == case.claim_id
        assert res.verdict in [DecisionVerdict.COUVERT, DecisionVerdict.NON_COUVERT, DecisionVerdict.AMBIGU_A_VERIFIER]
        assert len(res.applicable_clauses) > 0
        assert len(res.summary_reasoning) > 20
        assert len(res.recommendation_for_agent) > 10
