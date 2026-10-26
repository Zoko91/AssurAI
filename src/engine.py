"""
Moteur d'analyse contradictoire et de décision des sinistres.
Approche hybride :
1. Analyse déterministe des conditions sine qua non et exclusions formelles (< 2 ms)
2. Formulation de la décision explicable avec citations in extenso
3. Connecteur MLX pour synthèse en langage naturel par LLM local quantifié
"""

import json
from typing import List, Optional
from src.schemas import (
    ClaimInput,
    ContractClause,
    ClaimEvaluationResult,
    DecisionVerdict,
    ClauseType
)
from src.contract_store import ContractStore


class ClaimEvaluationEngine:
    def __init__(self, store: Optional[ContractStore] = None):
        self.store = store or ContractStore()

    def evaluate_claim(self, claim: ClaimInput, use_llm: bool = False, model_name: Optional[str] = None) -> ClaimEvaluationResult:
        """
        Évalue un sinistre en confrontant les circonstances aux clauses applicables.
        """
        # 1. Récupération des clauses pertinentes
        relevant_clauses = self.store.find_relevant_clauses(claim, top_k=3)
        if not relevant_clauses:
            return ClaimEvaluationResult(
                claim_id=claim.claim_id,
                verdict=DecisionVerdict.AMBIGU_A_VERIFIER,
                confidence_score=0.4,
                summary_reasoning="Aucune clause contractuelle spécifique identifiée pour ce sinistre.",
                applicable_clauses=[],
                cited_extracts=[],
                triggered_exclusions=[],
                unfulfilled_conditions=[],
                recommendation_for_agent="Transmettre au service juridique pour étude des conditions particulières."
            )

        # 2. Analyse contradictoire déterministe
        triggered_exclusions: List[str] = []
        unfulfilled_conditions: List[str] = []
        cited_extracts: List[str] = []
        applicable_articles = [c.article_num for c in relevant_clauses]

        meta = claim.metadata

        for c in relevant_clauses:
            # Vérification des conditions obligatoires
            if "verrouillage_a_cle" in c.mandatory_conditions and meta.get("door_locked") is False:
                unfulfilled_conditions.append(f"{c.article_num} : Obligation de fermeture et verrouillage à clé non respectée.")
            if "effraction_caracterisee" in c.mandatory_conditions and meta.get("break_in_marks") is False:
                unfulfilled_conditions.append(f"{c.article_num} : Absence de traces matérielles d'effraction constatée.")
            if "combustion_flamme_vive" in c.mandatory_conditions and meta.get("open_flame_present") is False:
                unfulfilled_conditions.append(f"{c.article_num} : Absence de flammes vives ou d'incendie propagé.")

            # Vérification des exclusions expresses
            if "local_non_verrouille_ou_ouvert" in c.explicit_exclusions and (meta.get("door_locked") is False or "ouvert" in claim.description.lower()):
                triggered_exclusions.append(f"{c.article_num} : Exclusion formelle des vols commis dans un local non fermé à clé ou laissé ouvert.")
            if "action_seule_chaleur_sans_flamme" in c.explicit_exclusions and (meta.get("open_flame_present") is False or "chaleur" in claim.description.lower()):
                triggered_exclusions.append(f"{c.article_num} : Exclusion des dommages résultant de la seule action de la chaleur ou du soleil sans combustion vive.")
            if "fonte_degradation_par_source_chaleur_sans_incendie" in c.explicit_exclusions and "bougie" in claim.description.lower():
                triggered_exclusions.append(f"{c.article_num} : Exclusion de la fonte ou détérioration causée par une bougie sans incendie propagé.")
            if "exterieur_hors_batiments_sans_option" in c.explicit_exclusions and meta.get("option_mobilite_douce") is False:
                triggered_exclusions.append(f"{c.article_num} : Exclusion du vol de cycle sur la voie publique hors souscription d'option.")
            if "ouvert" in claim.description.lower() and meta.get("window_open") is True and c.guarantee.value == "DEGATS_DES_EAUX":
                triggered_exclusions.append(f"{c.article_num} : Exclusion des infiltrations d'eau par ouvertures laissées ouvertes.")

            # Sélection d'extraits textuels pertinents
            for line in c.raw_text.split("\n"):
                line_str = line.strip()
                if any(k in line_str.lower() for k in ["exclu", "condition", "fermé à clé", "chaleur", "bougie", "effraction", "infiltrations"]):
                    if len(line_str) > 20 and line_str not in cited_extracts:
                        cited_extracts.append(f"[{c.article_num}] « {line_str} »")

        # 3. Arbitrage du verdict
        if meta.get("particular_conditions_available") is False or "sans précision" in claim.title.lower():
            verdict = DecisionVerdict.AMBIGU_A_VERIFIER
            confidence = 0.65
            summary = (
                f"Statut indéterminé : les dispositions générales ({', '.join(applicable_articles)}) prévoient "
                "des exclusions ou des conditions de garantie soumises à la souscription d'options aux Dispositions Particulières. "
                "Le gestionnaire doit vérifier les options souscrites par l'assuré."
            )
            recommendation = "Demander la copie des Dispositions Particulières pour vérifier la présence des options requises."

        elif len(triggered_exclusions) > 0 or len(unfulfilled_conditions) > 0:
            verdict = DecisionVerdict.NON_COUVERT
            confidence = 0.95
            reasons = "; ".join(triggered_exclusions + unfulfilled_conditions)
            summary = (
                f"Prise en charge refusée. Le sinistre déclenche au moins une exclusion formelle ou ne satisfait pas "
                f"aux conditions impératives du contrat : {reasons}."
            )
            recommendation = (
                f"Notifier le refus motivé en citant explicitement les articles {', '.join(applicable_articles)} "
                "et les motifs d'exclusion constatés."
            )

        else:
            verdict = DecisionVerdict.COUVERT
            confidence = 0.90
            summary = (
                f"Sinistre recevable et couvert au titre des garanties du contrat ({', '.join(applicable_articles)}). "
                "Toutes les conditions requises sont formellement réunies et aucune exclusion n'est activée."
            )
            recommendation = "Ouvrir le dossier d'indemnisation et solliciter les factures justificatives des biens déclarés."

        # 4. Si use_llm est demandé et mlx est actif, enrichir l'explication
        if use_llm:
            summary = self._call_mlx_llm_if_available(claim, relevant_clauses, verdict, summary, model_name)

        return ClaimEvaluationResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence_score=confidence,
            summary_reasoning=summary,
            applicable_clauses=applicable_articles,
            cited_extracts=cited_extracts[:5],
            triggered_exclusions=triggered_exclusions,
            unfulfilled_conditions=unfulfilled_conditions,
            recommendation_for_agent=recommendation
        )

    def _call_mlx_llm_if_available(self, claim: ClaimInput, clauses: List[ContractClause], verdict: DecisionVerdict, fallback_summary: str, model_name: Optional[str]) -> str:
        """Appel au modèle local MLX si disponible, avec repli immédiat sur le résumé déterministe."""
        try:
            from mlx_lm import load, generate
            model_id = model_name or "mlx-community/Qwen3.5-9B-MLX-4bit"
            model, tokenizer = load(model_id)
            prompt = (
                f"Tu es un expert juridique en assurance multirisque habitation.\n"
                f"Sinistre : {claim.description}\n"
                f"Clauses applicables : {[c.article_num + ': ' + c.raw_text for c in clauses]}\n"
                f"Verdict : {verdict.value}\n"
                f"Rédige en 3 phrases claires pour le gestionnaire la justification juridique et la recommandation."
            )
            response = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
            return response.strip()
        except Exception:
            return fallback_summary
