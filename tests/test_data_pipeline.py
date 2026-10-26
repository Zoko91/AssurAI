"""
Tests de validation : Pipeline de données & Structuration relationnelle.
Valide la conformité Pydantic, la présence des clauses critiques
et la structure du golden dataset.
"""

import json
from pathlib import Path
import pytest

from src.schemas import ContractClause, GuaranteeType, ClauseType, DecisionVerdict
from src.parser import parse_raw_contract_to_clauses


RAW_CONTRACT_PATH = "data/raw/conditions_generales_habitation.txt"
PROCESSED_DB_PATH = "data/processed/clauses_db.json"
BENCHMARK_PATH = "data/eval/benchmark_sinistres.json"


def test_raw_contract_exists_and_not_empty():
    path = Path(RAW_CONTRACT_PATH)
    assert path.exists(), f"Le fichier brut {RAW_CONTRACT_PATH} doit exister."
    text = path.read_text(encoding="utf-8")
    assert len(text) > 1000, "Le contrat brut doit être substantiel."
    assert "Article 3.3" in text
    assert "Article 2.2" in text


def test_parser_produces_valid_pydantic_clauses():
    clauses = parse_raw_contract_to_clauses(RAW_CONTRACT_PATH)
    assert len(clauses) >= 12, f"Attendu >= 12 clauses, obtenu {len(clauses)}"
    for c in clauses:
        assert isinstance(c, ContractClause)
        assert c.clause_id.startswith("CG-HAB-2026-LU-")
        assert len(c.raw_text) > 50


def test_case_1_bike_clause_integrity():
    """Vérifie que la clause du local commun (Article 3.3) est parfaitement modélisée."""
    path = Path(PROCESSED_DB_PATH)
    assert path.exists(), f"La base {PROCESSED_DB_PATH} doit exister."
    data = json.loads(path.read_text(encoding="utf-8"))
    
    art_3_3 = next((c for c in data if c["article_num"] == "Article 3.3"), None)
    assert art_3_3 is not None, "Article 3.3 manquant dans clauses_db.json"
    assert art_3_3["guarantee"] == GuaranteeType.VOL_ET_VANDALISME.value
    assert "parties_communes" in art_3_3["locations"]
    assert "local_non_verrouille_ou_ouvert" in art_3_3["explicit_exclusions"]
    assert "verrouillage_a_cle" in art_3_3["mandatory_conditions"]
    assert "pas fermée à clé" in art_3_3["raw_text"]


def test_case_2_candle_heat_exclusion_integrity():
    """Vérifie que la clause d'exclusion chaleur/bougie (Article 2.2) est présente."""
    path = Path(PROCESSED_DB_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    
    art_2_2 = next((c for c in data if c["article_num"] == "Article 2.2"), None)
    assert art_2_2 is not None, "Article 2.2 manquant dans clauses_db.json"
    assert art_2_2["guarantee"] == GuaranteeType.INCENDIE_ET_ASSIMILES.value
    assert art_2_2["clause_type"] == ClauseType.EXCLUSION.value
    assert "action_seule_chaleur_sans_flamme" in art_2_2["explicit_exclusions"]
    assert "bougie" in art_2_2["raw_text"]


def test_benchmark_dataset_integrity():
    """Vérifie que le golden dataset contient les cas de référence et de contrôle."""
    path = Path(BENCHMARK_PATH)
    assert path.exists(), f"Le benchmark {BENCHMARK_PATH} doit exister."
    cases = json.loads(path.read_text(encoding="utf-8"))
    
    assert len(cases) >= 8, f"Le benchmark doit comporter au moins 8 cas, obtenu {len(cases)}"
    
    ids = [c["claim_id"] for c in cases]
    assert "CAS-HAB-01" in ids, "Cas 1 de référence (vélo) manquant"
    assert "CAS-HAB-02" in ids, "Cas 2 de référence (bougie) manquant"
    
    for c in cases:
        assert c["expected_verdict"] in [v.value for v in DecisionVerdict]
        assert len(c["expected_clauses"]) >= 1
        assert len(c["ground_truth_reasoning"]) > 30
