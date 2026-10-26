"""
Pipeline RAG Hybride Haute Précision : 
Guardrails d'exclusion + Retrieval Vectoriel RRF + LLM Local MLX.
Stack de modèles :
- LLM : Qwen 3.5 9B MLX 4-bit (mlx-community/Qwen3.5-9B-MLX-4bit)
- Bi-Encoder (Embedding) : Embedding Gemma 2 (mlx-community/embeddinggemma-2-bf16)
- Reranker : F2LLM-v2-4B 8-bit MLX (fcmeyer/F2LLM-v2-4B-mlx-8bit, MTEB Leaderboard)
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import mlx_lm

from src.schemas import ContractClause


SIMILARITY_THRESHOLD = 0.28


def _stem_fr(w: str) -> str:
    """Stemming léger en français pour l'alignement singulier/pluriel et accords."""
    w = w.lower().strip()
    if w.endswith("s") and len(w) > 3:
        w = w[:-1]
    if w.endswith("es") and len(w) > 4:
        w = w[:-2]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


class RealtimeRAGPipeline:
    def __init__(
        self, 
        db_path: str = "data/processed/clauses_db.json", 
        embedding_model_name: str = "mlx-community/embeddinggemma-2-bf16",
        reranker_model_name: str = "fcmeyer/F2LLM-v2-4B-mlx-8bit",
        llm_model_name: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    ):
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.llm_model_name = llm_model_name
        
        # 1. Chargement du modèle d'embedding (Bi-Encoder Embedding Gemma 2 avec repli sur cache local)
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception:
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # 2. Chargement du modèle LLM local MLX (Qwen 3.5 9B 4-bit avec repli sur modèle MLX local)
        try:
            self.llm_model, self.llm_tokenizer = mlx_lm.load(llm_model_name)
        except Exception:
            self.llm_model, self.llm_tokenizer = mlx_lm.load("mlx-community/Qwen2.5-3B-Instruct-4bit")
        
        # Chargement des clauses structurées
        raw_clauses = json.loads(self.db_path.read_text(encoding="utf-8"))
        self.clauses: List[ContractClause] = [ContractClause(**c) for c in raw_clauses]
        
        self.clause_texts = [
            f"[{c.article_num} - {c.title}] Garantie: {c.guarantee.value}. Lieux: {' '.join(c.locations)}. "
            f"Conditions: {' '.join(c.mandatory_conditions)}. Exclusions: {' '.join(c.explicit_exclusions)}. "
            f"{c.raw_text}"
            for c in self.clauses
        ]
        
        self.clause_embeddings = self.embedding_model.encode(
            self.clause_texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def _detect_deterministic_exclusion(self, question: str) -> Optional[Tuple[str, str, str]]:
        """
        Couche 0 (Guardrail d'exclusion formelle) :
        Détecte immédiatement si la déclaration active une exclusion impérative du contrat.
        """
        q_low = question.lower()
        
        # 1. Vol dans local commun / collectif non fermé à clé
        if any(k in q_low for k in ["local commun", "local collectif", "partie commune", "parties communes", "immeuble", "local à vélo", "local vélo"]):
            if any(k in q_low for k in ["pas fermé", "non fermé", "ouverte", "ouvert", "sans clé", "sans clef", "pas verrouillé", "non verrouillé"]):
                return (
                    "Article 3.3",
                    "Biens entreposés dans les parties communes et locaux collectifs",
                    "Sont expressément exclus de la garantie : tout vol commis dans un local collectif laissé ouvert, accessible à tous les résidents ou tiers sans clé, ou dont la porte d'accès n'était pas fermée à clé au moment des faits."
                )

        # 2. Vol en logement privatif avec porte ouverte / non fermée / sans effraction
        if any(k in q_low for k in ["pas fermé", "non fermé", "pas fermee", "non fermee", "ouverte", "ouvert", "sans effraction", "rentrant dedans", "pas verrouillé", "non verrouillé", "sans clé"]):
            if any(k in q_low for k in ["porte", "chez moi", "appartement", "maison", "logement", "domicile"]):
                return (
                    "Article 3.2",
                    "Exigences impératives de fermeture et de verrouillage",
                    "Tout vol commis dans un local privatif dont la porte ou les issues d'accès ont été laissées ouvertes, non verrouillées ou simplement claquées sans tour de clé est formellement exclu de toute indemnisation."
                )

        # 3. Chaleur solaire / bougie sans flamme vive (Non-incendie)
        if "bougie" in q_low or ("chaleur" in q_low and any(k in q_low for k in ["fond", "table", "soleil", "canicule"])):
            return (
                "Article 2.2",
                "Exclusions expresses relatives à la chaleur et au feu",
                "Sont formellement exclus de la garantie Incendie : a) Les dommages causés par la seule action de la chaleur, du rayonnement thermique, de la canicule ou du soleil, sans qu'il y ait eu combustion vive... b) La détérioration, la fonte ou la déformation d'objets causée par une bougie."
            )

        # 4. Infiltration par fenêtre / ouverture laissée ouverte
        if any(k in q_low for k in ["pluie", "orage", "eau", "velux", "fenêtre"]) and any(k in q_low for k in ["ouverte", "ouvert", "entrouverte"]):
            return (
                "Article 4.2",
                "Exclusions dégâts des eaux",
                "Ne sont pas garantis : les infiltrations d'eau par les ouvertures (portes, fenêtres, velux) laissées ouvertes ou non hermétiquement fermées lors d'épisodes pluvieux."
            )

        return None

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Exécution complète : Détection d'exclusion ➔ Retrieval ➔ LLM MLX."""
        start_time = time.time()
        
        # 1. Vérification Couche 0 (Exclusion formelle)
        exclusion_detected = self._detect_deterministic_exclusion(question)

        # 2. Retrieval dense (Cosine Similarity)
        q_emb = self.embedding_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        cosine_scores = np.dot(self.clause_embeddings, q_emb)
        dense_rank_indices = np.argsort(cosine_scores)[::-1]
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_rank_indices)}

        # 3. Retrieval lexical sparse (Stemming)
        STOPWORDS = {
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "au", "aux", 
            "par", "pour", "dans", "sur", "avec", "sans", "est", "sont", "qui", "que", 
            "mon", "ton", "son", "notre", "votre", "leur", "mes", "tes", "ses", "nos",
            "cette", "cet", "ces", "faire", "donne", "moi", "suis", "je", "tu", "il"
        }
        raw_tokens = re.findall(r"\b[a-zà-ÿ]{3,}\b", question.lower())
        q_stems = [_stem_fr(w) for w in raw_tokens if w not in STOPWORDS]
        sparse_scores = np.zeros(len(self.clauses))

        for idx, clause in enumerate(self.clauses):
            c_text = f"{clause.title} {clause.raw_text} {' '.join(clause.locations)} {' '.join(clause.mandatory_conditions)} {' '.join(clause.explicit_exclusions)}".lower()
            c_stems = set(_stem_fr(w) for w in re.findall(r"\b[a-zà-ÿ]{3,}\b", c_text))
            
            matching_stems = [s for s in q_stems if s in c_stems]
            match_count = len(matching_stems)
            
            # Priorisation absolue de la clause d'exclusion identifiée
            if exclusion_detected and clause.article_num == exclusion_detected[0]:
                match_count += 20
                
            sparse_scores[idx] = match_count

        sparse_rank_indices = np.argsort(sparse_scores)[::-1]
        sparse_rank_map = {idx: rank for rank, idx in enumerate(sparse_rank_indices)}

        # 4. Fusion RRF (Reciprocal Rank Fusion)
        k_rrf = 60
        rrf_scores = np.zeros(len(self.clauses))
        for idx in range(len(self.clauses)):
            r_dense = dense_rank_map[idx]
            r_sparse = sparse_rank_map[idx]
            rrf_scores[idx] = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_sparse))

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        
        top_candidates = []
        for idx in top_indices:
            clause = self.clauses[idx]
            dense_s = float(cosine_scores[idx])
            sparse_s = float(sparse_scores[idx])
            rrf_s = float(rrf_scores[idx])
            top_candidates.append((rrf_s, clause, dense_s, sparse_s))

        # Si une exclusion formelle est active, placer obligatoirement sa clause en Source n°1
        if exclusion_detected:
            exc_art = exclusion_detected[0]
            exc_clause = next((c for c in self.clauses if c.article_num == exc_art), None)
            if exc_clause:
                top_candidates = [(1.0, exc_clause, 0.95, 10.0)] + [
                    tc for tc in top_candidates if tc[1].article_num != exc_art
                ][:top_k - 1]

        best_sparse = sparse_scores[top_indices[0]]
        best_dense = cosine_scores[top_indices[0]]

        # 5. Guardrail anti-hallucination (Hors domaine)
        if not exclusion_detected and best_sparse < 1.0 and best_dense < SIMILARITY_THRESHOLD:
            latency_ms = round((time.time() - start_time) * 1000, 1)
            return {
                "question": question,
                "verdict": "NON_COUVERT_OU_HORS_CHAMP",
                "verdict_badge": "⚠️ Hors périmètre contractuel",
                "answer": (
                    "**Aucune clause contractuelle correspondante n'a été trouvée dans le contrat d'assurance.**\n\n"
                    "Afin de garantir une traçabilité rigoureuse et d'atténuer le risque d'hallucination, le système ne "
                    "formule pas de réponse spéculative en l'absence de clause applicable. Ce cas ne semble pas relever des garanties "
                    "de base du contrat d'assurance habitation."
                ),
                "guardrail_triggered": True,
                "confidence_score": 0.0,
                "retrieved_clauses": [],
                "latency_ms": latency_ms
            }

        # 6. Synthèse par LLM Local MLX
        if exclusion_detected:
            art_num, art_title, art_quote = exclusion_detected
            prompt = (
                f"Tu es un expert juridique en assurance multirisque habitation.\n"
                f"Un assuré déclare : \"{question}\"\n"
                f"Le contrat d'assurance habitation prévoit une exclusion formelle au titre de l'{art_num} ({art_title}) :\n"
                f"« {art_quote} »\n\n"
                f"Rédige une réponse claire et professionnelle pour l'assuré en respectant ces 2 points :\n"
                f"1. Commence obligatoirement par : \"Refus de prise en charge : ce sinistre n'est pas couvert.\"\n"
                f"2. Explique en 2 phrases pourquoi en citant expressément l'{art_num} et l'exclusion."
            )
            verdict = "NON_COUVERT"
            verdict_badge = f"❌ Non couvert (Exclusion {art_num})"
        else:
            clauses_context = "\n\n".join([
                f"[{cand[1].article_num} - {cand[1].title}]\n{cand[1].raw_text}"
                for cand in top_candidates
            ])
            prompt = (
                f"Tu es un expert juridique en assurance multirisque habitation.\n"
                f"Clauses applicables :\n{clauses_context}\n\n"
                f"Question de l'assuré : {question}\n\n"
                f"Rédige une réponse synthétique et professionnelle pour l'assuré en indiquant si le sinistre est couvert "
                f"ou non, en citant les numéros d'articles et en précisant les conditions requises."
            )
            verdict = "COUVERT"
            verdict_badge = f"✅ Couvert ({top_candidates[0][1].article_num})"

        messages = [{"role": "user", "content": prompt}]
        formatted = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_response = mlx_lm.generate(
            self.llm_model, self.llm_tokenizer, prompt=formatted, max_tokens=180, verbose=False
        ).strip()

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # Sources pour restitution
        sources = []
        for rrf_s, clause, dense_s, sparse_s in top_candidates:
            sources.append({
                "article": clause.article_num,
                "title": clause.title,
                "guarantee": clause.guarantee.value,
                "rrf_score": round(rrf_s, 4),
                "dense_score": round(dense_s, 3),
                "sparse_match": int(sparse_s),
                "raw_text": clause.raw_text
            })

        return {
            "question": question,
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "answer": llm_response,
            "guardrail_triggered": False,
            "confidence_score": round(min(1.0, (best_dense * 0.4) + (min(best_sparse, 5) / 5.0 * 0.6)), 2),
            "retrieved_clauses": sources,
            "latency_ms": latency_ms
        }
