"""
Application Streamlit - Assistant IA Sinistres & Contrats d'Assurance
UX ultra-pure, temps réel : Retrieval Vectoriel + Reranking RRF + Guardrail Anti-Hallucination.
"""

import streamlit as st
from src.rag_pipeline import RealtimeRAGPipeline

st.set_page_config(
    page_title="AssurAI — Sinistres & Contrats",
    page_icon="🛡️",
    layout="centered"
)

# Chargement du pipeline en cache (instantané au rechargement)
@st.cache_resource(show_spinner="Initialisation de la base vectorielle & du pipeline...")
def load_rag():
    return RealtimeRAGPipeline()

rag = load_rag()

# Style CSS épuré, pensé pour le thème clair avec contraste optimal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .app-title { font-size: 28px; font-weight: 700; color: #0F172A; text-align: center; margin-top: 10px; margin-bottom: 4px; }
    .app-subtitle { font-size: 14px; color: #64748B; text-align: center; margin-bottom: 25px; }
    
    .badge-card { padding: 12px 16px; border-radius: 8px; font-weight: 600; font-size: 15px; margin-bottom: 12px; display: inline-block; }
    .badge-non-couvert { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-couvert { background-color: #F0FDF4; color: #166534; border: 1px solid #86EFAC; }
    .badge-guardrail { background-color: #FFFBEB; color: #92400E; border: 1px solid #FCD34D; }
    
    .source-box { background-color: #F8FAFC; border-radius: 6px; padding: 12px; margin-bottom: 8px; border-left: 3px solid #3B82F6; }
    .source-title { font-weight: 600; color: #1E293B; font-size: 13px; margin-bottom: 4px; }
    .source-quote { font-family: monospace; font-size: 12px; color: #334155; white-space: pre-wrap; line-height: 1.5; }
    
    .metric-chip { background-color: #F1F5F9; border-radius: 12px; padding: 4px 10px; font-size: 12px; font-weight: 500; color: #475569; display: inline-block; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">🛡️ AssurAI — Sinistres & Contrats</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Recherche en langage naturel • Base vectorielle & RRF • Guardrails d\'exclusions</div>', unsafe_allow_html=True)

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exemples rapides 1-clic
st.caption("💡 **Questions types à tester en 1 clic :**")
c1, c2, c3 = st.columns(3)
c4, c5, _ = st.columns(3)

preset_query = None
if c1.button("🚲 Vélo local commun", use_container_width=True):
    preset_query = "Je me fais voler mon vélo dans le local commun de l'immeuble, qui n'était pas fermé à clé. Suis-je couvert ?"
if c2.button("🕯️ Bougie table jardin", use_container_width=True):
    preset_query = "Un assuré laisse une bougie sur la table de son jardin pendant ses vacances. Avec la chaleur, la bougie fond et abîme la table. Est-ce couvert ?"
if c3.button("🚪 Porte non fermée", use_container_width=True):
    preset_query = "Si la porte n'est pas fermée à clé et qu'une personne rentre chez moi et vole quelque chose, est-ce couvert ?"
if c4.button("💧 Fuite d'eau sous évier", use_container_width=True):
    preset_query = "Une canalisation sous mon évier a rompu subitement dans la nuit et a inondé la cuisine. Est-ce indemnisé ?"
if c5.button("🚫 Question hors contrat", use_container_width=True):
    preset_query = "Donne-moi une recette de cuisine pour préparer des crêpes au Nutella."

# Saisie utilisateur (Champ de chat moderne)
user_input = st.chat_input("Posez votre question sur un sinistre ou une clause...")
query_to_process = preset_query or user_input

# Affichage des messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            data = msg["data"]
            badge_class = "badge-guardrail" if data["guardrail_triggered"] else ("badge-couvert" if "COUVERT" in data["verdict"] and "NON" not in data["verdict"] else "badge-non-couvert")
            st.markdown(f'<div class="badge-card {badge_class}">{data["verdict_badge"]}</div>', unsafe_allow_html=True)
            st.markdown(data["answer"])
            
            if data["retrieved_clauses"]:
                with st.expander(f"🔍 Sources contractuelles citées ({len(data['retrieved_clauses'])} clauses vérifiées)"):
                    for src in data["retrieved_clauses"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-title">{src['article']} — {src['title']} (Garantie : {src['guarantee']} | Reranker : {src.get('rerank_score', 'N/A')})</div>
                            <div class="source-quote">{src['raw_text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <span class="metric-chip">⚡ Latence : {data['latency_ms']} ms</span>
                <span class="metric-chip">🎯 Confiance : {int(data['confidence_score'] * 100)} %</span>
                <span class="metric-chip">🛡️ Guardrail : {'Déclenché (Bloqué)' if data['guardrail_triggered'] else 'Conforme'}</span>
            </div>
            """, unsafe_allow_html=True)

# Traitement de la nouvelle question
if query_to_process:
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.write(query_to_process)

    with st.chat_message("assistant"):
        with st.spinner("Analyse vectorielle et vérification des clauses..."):
            result = rag.query(query_to_process)

        badge_class = "badge-guardrail" if result["guardrail_triggered"] else ("badge-couvert" if "COUVERT" in result["verdict"] and "NON" not in result["verdict"] else "badge-non-couvert")
        st.markdown(f'<div class="badge-card {badge_class}">{result["verdict_badge"]}</div>', unsafe_allow_html=True)
        st.markdown(result["answer"])
        
        if result["retrieved_clauses"]:
            with st.expander(f"🔍 Sources contractuelles citées ({len(result['retrieved_clauses'])} clauses vérifiées)"):
                for src in result["retrieved_clauses"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-title">{src['article']} — {src['title']} (Garantie : {src['guarantee']} | Reranker : {src.get('rerank_score', 'N/A')})</div>
                        <div class="source-quote">{src['raw_text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-top: 10px;">
            <span class="metric-chip">⚡ Latence : {result['latency_ms']} ms</span>
            <span class="metric-chip">🎯 Confiance : {int(result['confidence_score'] * 100)} %</span>
            <span class="metric-chip">🛡️ Guardrail : {'Déclenché (Bloqué)' if result['guardrail_triggered'] else 'Conforme'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "data": result})
