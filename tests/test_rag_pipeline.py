"""
Tests de validation de la pipeline RAG temps réel (Retrieval + RRF + Guardrail + LLM MLX).
"""

import pytest
from src.rag_pipeline import RealtimeRAGPipeline


@pytest.fixture(scope="module")
def rag_pipeline():
    return RealtimeRAGPipeline()


def test_user_case_unlocked_door_intrusion(rag_pipeline):
    """Vérifie que l'intrusion par porte non fermée est formellement exclue (Article 3.2)."""
    q = "Si la porte n'est PAS fermée. Qu'une personne s'introduit chez moi juste en rentrant dedans et vole quelque chose, est-ce couvert ?"
    res = rag_pipeline.query(q)
    assert res["verdict"] == "NON_COUVERT"
    assert "Article 3.2" in res["answer"]
    assert res["retrieved_clauses"][0]["article"] == "Article 3.2"


def test_user_case_unlocked_door_general_definition(rag_pipeline):
    """Vérifie qu'un vol avec porte non fermée à clé est exclu de garantie."""
    q = "Est ce qu'un vol est considéré comme un vol pour les assurances si la porte n'était pas fermée à clé ?"
    res = rag_pipeline.query(q)
    assert res["verdict"] == "NON_COUVERT"
    assert "Article 3.2" in res["answer"]
    assert res["retrieved_clauses"][0]["article"] == "Article 3.2"


def test_reference_q1_bike_unlocked(rag_pipeline):
    """Cas 1 : Vélo volé dans local commun non fermé."""
    q = "Je me fais voler mon vélo dans le local commun de l'immeuble, qui n'était pas fermé à clé. Suis-je couvert ?"
    res = rag_pipeline.query(q)
    assert res["verdict"] == "NON_COUVERT"
    assert "Article 3.3" in res["answer"]
    assert res["retrieved_clauses"][0]["article"] == "Article 3.3"


def test_reference_q2_candle_melted(rag_pipeline):
    """Cas 2 : Bougie fondue sur table de jardin."""
    q = "Un assuré laisse une bougie sur la table de son jardin pendant ses vacances. Avec la chaleur, la bougie fond et abîme la table. Est-ce couvert ?"
    res = rag_pipeline.query(q)
    assert res["verdict"] == "NON_COUVERT"
    assert "Article 2.2" in res["answer"]
    assert res["retrieved_clauses"][0]["article"] == "Article 2.2"


def test_anti_hallucination_guardrail(rag_pipeline):
    """Cas 3 : Question hors domaine pour tester le guardrail anti-hallucination."""
    q = "Donne-moi une recette de cuisine pour préparer des crêpes au Nutella."
    res = rag_pipeline.query(q)
    assert res["guardrail_triggered"] is True
    assert res["verdict"] == "NON_COUVERT_OU_HORS_CHAMP"
    assert res["confidence_score"] == 0.0
