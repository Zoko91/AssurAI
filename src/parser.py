"""
Module de parsing et de structuration relationnelle des Conditions Générales d'Assurance.
Parsing structuré au niveau clause : tables relationnelles plutôt que chunks de texte bruts.
Transforme le texte juridique brut en unités atomiques de clauses avec métadonnées d'application et exclusions strictes.
"""

import json
import re
from pathlib import Path
from typing import List

from src.schemas import ContractClause, GuaranteeType, ClauseType


def parse_raw_contract_to_clauses(raw_text_path: str) -> List[ContractClause]:
    """
    Lit le fichier brut de conditions générales et extrait les clauses sous forme structurée Pydantic.
    """
    path = Path(raw_text_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de contrat introuvable: {raw_text_path}")

    text = path.read_text(encoding="utf-8")
    
    # Repérage de la référence du contrat
    contract_ref = "CG-HAB-2026-LU"
    ref_match = re.search(r"Réf\.\s*([A-Z0-9\-]+)", text)
    if ref_match:
        contract_ref = ref_match.group(1)

    clauses: List[ContractClause] = []

    # Découpage par Article
    article_pattern = re.compile(
        r"(Article\s+(\d+\.\d+)\s*-\s*([^\n]+))\n(.*?)(?=(?:Article\s+\d+\.\d+|\Z))",
        re.DOTALL
    )

    matches = article_pattern.findall(text)

    for full_title_line, art_num, title_clean, body_text in matches:
        # Nettoyage des bannières de chapitres ou lignes de séparation dans le corps
        cleaned_body = re.sub(r"=+\s*\nCHAPITRE\s+\d+.*?(?=\n\n|\Z)", "", body_text, flags=re.DOTALL)
        cleaned_body = re.sub(r"=+", "", cleaned_body).strip()
        art_num_clean = f"Article {art_num}"
        clause_id = f"{contract_ref}-{art_num}"

        # Détermination de la garantie principale
        guarantee = GuaranteeType.DISPOSITIONS_GENERALES
        if art_num.startswith("1."):
            guarantee = GuaranteeType.DISPOSITIONS_GENERALES
        elif art_num.startswith("2."):
            guarantee = GuaranteeType.INCENDIE_ET_ASSIMILES
        elif art_num.startswith("3."):
            guarantee = GuaranteeType.VOL_ET_VANDALISME
        elif art_num.startswith("4."):
            guarantee = GuaranteeType.DEGATS_DES_EAUX
        elif art_num.startswith("5."):
            guarantee = GuaranteeType.RESPONSABILITE_CIVILE
        elif "extérieur" in title_clean.lower() or "jardin" in title_clean.lower():
            guarantee = GuaranteeType.AMENAGEMENTS_EXTERIEURS

        # Détermination du type de clause
        clause_type = ClauseType.INCLUSION
        title_lower = title_clean.lower()
        if "exclusion" in title_lower:
            clause_type = ClauseType.EXCLUSION
        elif "définition" in title_lower:
            clause_type = ClauseType.DEFINITION
        elif "exigence" in title_lower or "condition" in title_lower or "circonstance" in title_lower:
            clause_type = ClauseType.CONDITION_APPLICATION

        # Extraction des lieux
        locations = []
        body_lower = cleaned_body.lower()
        if "commune" in body_lower or "collectif" in body_lower or "local commun" in body_lower or "local à vélo" in body_lower:
            locations.extend(["parties_communes", "local_collectif", "local_velo"])
        if "privatif" in body_lower or "pièces d'habitation" in body_lower or "logement" in body_lower:
            locations.append("logement_privatif")
        if "extérieur" in body_lower or "jardin" in body_lower or "terrasse" in body_lower:
            locations.extend(["jardin", "terrasse", "exterieur"])
        if "dépendance" in body_lower or "cave" in body_lower or "garage" in body_lower:
            locations.extend(["cave", "garage", "dependance"])

        # Extraction des conditions sine qua non
        mandatory_conditions = []
        if "effraction" in body_lower:
            mandatory_conditions.append("effraction_caracterisee")
        if re.search(r"ferm[ée]e?s?\s*à\s*cl[ée]|verrouill[ée]e?s?", body_lower):
            mandatory_conditions.append("verrouillage_a_cle")
        if "point d'ancrage fixe" in body_lower or "antivol" in body_lower or "point fixe" in body_lower:
            mandatory_conditions.append("antivol_point_fixe")
        if "combustion avec flammes" in body_lower:
            mandatory_conditions.append("combustion_flamme_vive")
        if "déclaration" in body_lower and "délai" in body_lower:
            mandatory_conditions.append("respect_delai_declaration")

        # Extraction des exclusions formelles
        explicit_exclusions = []
        if re.search(r"non\s*ferm[ée]e?s?\s*à\s*cl[ée]|non\s*verrouill[ée]e?s?|ouvert", body_lower):
            explicit_exclusions.append("local_non_verrouille_ou_ouvert")
        if "seule action de la chaleur" in body_lower or "rayonnement thermique" in body_lower or "soleil" in body_lower:
            explicit_exclusions.append("action_seule_chaleur_sans_flamme")
        if "bougie" in body_lower or "fonte" in body_lower or "détérioration" in body_lower:
            explicit_exclusions.append("fonte_degradation_par_source_chaleur_sans_incendie")
        if "voie publique" in body_lower or "hors des bâtiments" in body_lower:
            explicit_exclusions.append("exterieur_hors_batiments_sans_option")
        if "intentionnellement" in body_lower:
            explicit_exclusions.append("acte_intentionnel")

        clause = ContractClause(
            clause_id=clause_id,
            contract_ref=contract_ref,
            article_num=art_num_clean,
            title=title_clean.strip(),
            guarantee=guarantee,
            clause_type=clause_type,
            locations=list(set(locations)),
            mandatory_conditions=list(set(mandatory_conditions)),
            explicit_exclusions=list(set(explicit_exclusions)),
            raw_text=f"{full_title_line}\n{cleaned_body}"
        )
        clauses.append(clause)

    return clauses


def save_clauses_to_json(clauses: List[ContractClause], output_path: str) -> None:
    """Enregistre la base relationnelle de clauses sous forme JSON indenté et typé."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = [c.model_dump() for c in clauses]
    out.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raw_file = "data/raw/conditions_generales_habitation.txt"
    out_file = "data/processed/clauses_db.json"
    parsed_clauses = parse_raw_contract_to_clauses(raw_file)
    save_clauses_to_json(parsed_clauses, out_file)
    print(f"✅ {len(parsed_clauses)} clauses extraites et structurées dans {out_file}")
