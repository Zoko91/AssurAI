"""
Modèles de données stricts pour l'analyse de sinistres (Assurance Multirisque Habitation).
Architecture relationnelle et interprétable pour l'évaluation des clauses contractuelles.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    DEFINITION = "DEFINITION"
    INCLUSION = "INCLUSION"
    EXCLUSION = "EXCLUSION"
    CONDITION_APPLICATION = "CONDITION_APPLICATION"


class GuaranteeType(str, Enum):
    VOL_ET_VANDALISME = "VOL_ET_VANDALISME"
    INCENDIE_ET_ASSIMILES = "INCENDIE_ET_ASSIMILES"
    AMENAGEMENTS_EXTERIEURS = "AMENAGEMENTS_EXTERIEURS"
    DEGATS_DES_EAUX = "DEGATS_DES_EAUX"
    RESPONSABILITE_CIVILE = "RESPONSABILITE_CIVILE"
    DISPOSITIONS_GENERALES = "DISPOSITIONS_GENERALES"


class DecisionVerdict(str, Enum):
    COUVERT = "COUVERT"
    NON_COUVERT = "NON_COUVERT"
    AMBIGU_A_VERIFIER = "AMBIGU_A_VERIFIER"


class ContractClause(BaseModel):
    """
    Clause contractuelle relationnelle atomique.
    Découpage structuré : tables relationnelles plutôt que chunks de texte bruts.
    """
    clause_id: str = Field(..., description="Identifiant unique de la clause (ex: CG-HAB-3.3)")
    contract_ref: str = Field(..., description="Référence du contrat parent")
    article_num: str = Field(..., description="Numéro d'article (ex: Article 3.3)")
    title: str = Field(..., description="Titre de la clause")
    guarantee: GuaranteeType = Field(..., description="Famille de garantie rattachée")
    clause_type: ClauseType = Field(..., description="Nature juridique de la clause")
    locations: List[str] = Field(default_factory=list, description="Lieux d'application")
    mandatory_conditions: List[str] = Field(
        default_factory=list, 
        description="Conditions sine qua non à réunir pour ouvrir droit à indemnisation"
    )
    explicit_exclusions: List[str] = Field(
        default_factory=list, 
        description="Faits ou circonstances entraînant le rejet immédiat de la prise en charge"
    )
    raw_text: str = Field(..., description="Texte juridique in extenso pour citation rigoureuse")


class ClaimInput(BaseModel):
    """
    Déclaration du sinistre soumise par l'assuré ou le conseiller.
    """
    claim_id: str = Field(..., description="Identifiant de la déclaration")
    title: str = Field(..., description="Intitulé concis du sinistre")
    description: str = Field(..., description="Récit complet des faits")
    target_guarantee: Optional[GuaranteeType] = Field(None, description="Garantie visée si connue")
    location: Optional[str] = Field(None, description="Lieu précis du sinistre")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Détails factuels (ex: verrouillé, effraction, flamme, extérieur)"
    )


class ClaimEvaluationResult(BaseModel):
    """
    Restitution métier formalisée pour le gestionnaire de sinistres.
    Chaque décision est impérativement motivée et sourcée.
    """
    claim_id: str = Field(..., description="Identifiant du sinistre évalué")
    verdict: DecisionVerdict = Field(..., description="Statut de la prise en charge")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Niveau de certitude (0 à 1)")
    summary_reasoning: str = Field(..., description="Synthèse pédagogique rédigée dans le langage métier")
    applicable_clauses: List[str] = Field(..., description="Numéros des articles contractuels mobilisés")
    cited_extracts: List[str] = Field(..., description="Extraits textuels in extenso prouvant la décision")
    triggered_exclusions: List[str] = Field(default_factory=list, description="Exclusions formelles constatées")
    unfulfilled_conditions: List[str] = Field(default_factory=list, description="Conditions requises non respectées")
    recommendation_for_agent: str = Field(..., description="Action recommandée pour le gestionnaire")
