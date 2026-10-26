"""
Accès structuré et requêtage relationnel aux clauses contractuelles.
Architecture modulaire : filtrage précis sur tables relationnelles plutôt que RAG naïf.
"""

import json
from pathlib import Path
from typing import List, Optional
from src.schemas import ContractClause, ClaimInput, GuaranteeType


class ContractStore:
    def __init__(self, db_path: str = "data/processed/clauses_db.json"):
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Base de clauses introuvable : {db_path}")
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        self.clauses: List[ContractClause] = [ContractClause(**c) for c in raw_data]

    def get_clause_by_article(self, article_num: str) -> Optional[ContractClause]:
        """Récupère une clause par son numéro d'article exact (ex: 'Article 3.3')."""
        for clause in self.clauses:
            if clause.article_num.lower() == article_num.lower():
                return clause
        return None

    def find_relevant_clauses(self, claim: ClaimInput, top_k: int = 3) -> List[ContractClause]:
        """
        Recherche hybride chirurgicale :
        1. Filtrage relationnel dur (garantie visée + lieu)
        2. Scoring lexical pondéré sur les mots-clés du sinistre
        """
        query_words = set(claim.description.lower().split())
        scored: List[tuple[float, ContractClause]] = []

        for c in self.clauses:
            score = 0.0

            # 1. Alignement de la garantie (Filtre relationnel majeur)
            if claim.target_guarantee and c.guarantee == claim.target_guarantee:
                score += 5.0

            # 2. Alignement des lieux d'application
            if claim.location:
                claim_loc = claim.location.lower()
                for loc in c.locations:
                    if loc in claim_loc or claim_loc in loc:
                        score += 3.0
                        break

            # 3. Correspondance lexicale sur le corps de l'article et son titre
            clause_text = f"{c.title} {c.raw_text}".lower()
            matching_tokens = [w for w in query_words if len(w) > 3 and w in clause_text]
            score += len(matching_tokens) * 0.8

            # 4. Bonus si une exclusion expresse de la clause correspond aux métadonnées du sinistre
            for meta_k, meta_v in claim.metadata.items():
                if meta_v is False and meta_k in ["door_locked", "break_in_marks", "open_flame_present"]:
                    if any("ouvert" in exc or "chaleur" in exc or "effraction" in exc for exc in c.explicit_exclusions):
                        score += 4.0

            scored.append((score, c))

        # Tri décroissant et sélection des top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:top_k] if score > 0]
