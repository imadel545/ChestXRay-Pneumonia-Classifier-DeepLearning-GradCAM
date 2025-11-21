import os
import time
from typing import Optional

import keras
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ---------------------------------------------------------
# Import Groq de manière OPTIONNELLE (pas de crash si pb)
# ---------------------------------------------------------
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None  # type: ignore[misc]

from util import (
    classify,
    set_background,
    generate_gradcam,
    analyze_image_quality,
)


# =========================================================
# CONFIG GLOBALE & CONSTANTES
# =========================================================
APP_TITLE = "AI Pneumonia Detector"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_RELATIVE_PATH = os.path.join("model", "pneumonia_classifier.h5")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Image de fond globale
set_background("./bgs/images.jpg")


# =========================================================
# CSS – UI PRO / GLASSMORPHISM
# =========================================================
st.markdown(
    """
<style>
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(10, 14, 39, 0.85);
        z-index: 0;
        pointer-events: none;
    }

    .main, .block-container {
        position: relative;
        z-index: 1;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
        margin: 20px 0;
    }

    .neon-title {
        font-size: 3.2em;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #00f5ff, #00ff88, #0099ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.75);
        margin-bottom: 4px;
        letter-spacing: 0.03em;
    }

    .product-subtitle {
        text-align: center;
        color: #a0f2ff;
        font-size: 1.05em;
        margin-bottom: 25px;
        font-weight: 400;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .result-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12), rgba(0, 0, 0, 0.45));
        border-radius: 16px;
        padding: 24px 26px;
        margin: 15px 0;
        border: 1px solid rgba(0, 212, 255, 0.45);
        box-shadow: 0 0 32px rgba(0, 212, 255, 0.25);
    }

    .diagnosis-badge {
        display: inline-block;
        padding: 12px 30px;
        border-radius: 999px;
        font-size: 1.5em;
        font-weight: 700;
        text-align: center;
        margin: 10px 0 18px 0;
        animation: fadeInScale 0.4s ease-out;
    }

    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.7);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    .badge-normal {
        background: radial-gradient(circle at 0% 0%, #00ffb2, #00c97a);
        color: #022b18;
        box-shadow: 0 0 40px rgba(0, 255, 180, 0.6);
    }

    .badge-pneumonia {
        background: radial-gradient(circle at 0% 0%, #ff4b5c, #b80030);
        color: #fff;
        box-shadow: 0 0 40px rgba(255, 75, 92, 0.7);
    }

    .confidence-score {
        font-size: 2.8em;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00ff88, #00f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .confidence-label {
        text-align: center;
        color: #a0f2ff;
        font-size: 1em;
    }

    .upload-zone {
        border: 2px dashed rgba(0, 212, 255, 0.7);
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        background: rgba(0, 212, 255, 0.06);
        transition: all 0.28s ease;
    }

    .upload-zone:hover {
        border-color: #00f5ff;
        background: rgba(0, 212, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.35);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0099ff);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 10px 30px;
        font-size: 1em;
        font-weight: 600;
        transition: all 0.23s ease;
        box-shadow: 0 8px 22px rgba(0, 212, 255, 0.35);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0, 212, 255, 0.5);
    }

    .info-card {
        background: rgba(8, 16, 40, 0.9);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0;
        border-left: 4px solid #00d4ff;
    }

    .glow-text {
        color: #00f5ff;
        text-shadow: 0 0 14px rgba(0, 212, 255, 0.7);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .image-container {
        border: 2px solid rgba(0, 212, 255, 0.4);
        border-radius: 16px;
        padding: 8px;
        background: rgba(0, 0, 0, 0.55);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.25);
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# CLIENT GROQ – SÉCURISATION API KEY
# =========================================================
def get_groq_client() -> Optional["Groq"]:
    if Groq is None:
        return None

    api_key: Optional[str] = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


groq_client = get_groq_client()


# =========================================================
# FONCTION – JAUGE PLOTLY
# =========================================================
def create_gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20, "color": "#a0f2ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#00d4ff"},
                "bar": {"color": "#00d4ff"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "#00d4ff",
                "steps": [
                    {"range": [0, 50], "color": "rgba(0, 255, 136, 0.25)"},
                    {"range": [50, 80], "color": "rgba(255, 204, 0, 0.25)"},
                    {"range": [80, 100], "color": "rgba(255, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.7,
                    "value": value,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0f7ff", "family": "Arial"},
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# =========================================================
# FONCTION – CHATBOT MÉDICAL (GROQ)
# =========================================================
def get_medical_advice(
    diagnosis: str, confidence: float, user_question: Optional[str] = None
) -> str:
    if groq_client is None:
        return (
            "Mode assistant IA désactivé (bibliothèque Groq ou clé API indisponible).\n\n"
            "Configurez `GROQ_API_KEY` dans `.streamlit/secrets.toml` ou les variables d'environnement "
            "et assurez-vous que le package `groq` est installé pour activer cette fonctionnalité."
        )

    if user_question:
        prompt = f"""
Tu es un expert médical spécialisé en radiologie pulmonaire.

Contexte du diagnostic IA :
- Résultat : {diagnosis}
- Confiance du modèle IA : {confidence:.1f} %

Question du patient :
{user_question}

Réponds en français, de manière professionnelle, claire et empathique.
Rappelle toujours que cette analyse ne remplace jamais une consultation médicale.
"""
    else:
        prompt = f"""
Tu es un expert médical spécialisé en radiologie pulmonaire.

Un modèle d'IA a analysé une radiographie thoracique.

Résultat :
- Diagnostic IA : {diagnosis}
- Niveau de confiance : {confidence:.1f} %

Explique en français, en moins de 200 mots :
1. Ce que signifie ce résultat
2. Les symptômes éventuels associés
3. Les prochaines étapes recommandées
4. Le rappel que seule une consultation médicale permet un vrai diagnostic.
"""

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant médical spécialisé en radiologie pulmonaire. "
                        "Tu fournis des informations pédagogiques, claires et empathiques, en français. "
                        "Tu ne poses jamais de diagnostic définitif et tu renvoies toujours vers un médecin."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.6,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de la consultation de l'assistant médical IA : {e}"


# =========================================================
# CHARGEMENT MODÈLE
# =========================================================
@st.cache_resource
def load_pneumonia_model():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if not os.path.exists(MODEL_RELATIVE_PATH):
        st.error(f"❌ Modèle introuvable à l'emplacement : {MODEL_RELATIVE_PATH}")
        return None

    try:
        model = keras.saving.load_model(MODEL_RELATIVE_PATH, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        st.success("✅ Modèle DenseNet121 chargé avec succès.")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None


with st.spinner("🔄 Initialisation du moteur IA..."):
    model = load_pneumonia_model()


# =========================================================
# HEADER
# =========================================================
col_logo, col_title, col_space = st.columns([1, 3, 1])

with col_title:
    st.markdown(f'<h1 class="neon-title">{APP_TITLE.upper()}</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p class='product-subtitle'>Radiologie assistée par IA · Outil d'aide à la décision</p>",
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🩺 Profil du moteur")
    st.metric("Type de modèle", "DenseNet121", "Transfer Learning + Focal Loss")
    st.metric("Mode", "Binaire", "NORMAL / PNEUMONIA")

    if model is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Infos runtime")
        st.write(f"Backend Keras : `{keras.backend.backend()}`")
        st.write(f"Fichier modèle : `{MODEL_RELATIVE_PATH}`")

    st.markdown("---")
    st.markdown("### ⚠️ Avertissement médical")
    st.info(
        "Cet outil est **strictement éducatif**.\n\n"
        "- Ne remplace **jamais** un avis médical.\n"
        "- Ne pas utiliser pour des décisions cliniques.\n"
        "- En cas de doute ou de symptôme, consultez un médecin."
    )

    st.markdown("---")
    st.markdown("### 🤖 Assistant IA")
    if groq_client is None:
        st.warning("Assistant médical IA désactivé (Groq non opérationnel).")
    else:
        st.success(f"Assistant médical IA actif (Groq · {GROQ_MODEL_NAME})")


# =========================================================
# CONTENU PRINCIPAL – TABS
# =========================================================
tab_diag, tab_about = st.tabs(["🔎 Diagnostic", "ℹ️ À propos du modèle"])

with tab_diag:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown(
        """
<div style='text-align: center; color: #a0f2ff; font-size: 1.05em; margin-bottom: 24px;'>
    Importez une radiographie thoracique (face) pour une analyse assistée par IA.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Glissez-déposez une radiographie (JPEG / PNG)",
            type=["jpeg", "jpg", "png"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        if model is None:
            st.error("❌ Modèle non disponible. Impossible d'analyser l'image.")
        else:
            with st.spinner("🔬 Analyse de l'image en cours..."):
                progress_bar = st.progress(0)

                image = Image.open(uploaded_file).convert("RGB")

                # Analyse qualité image
                quality = analyze_image_quality(image)

                # Inférence + latence
                t0 = time.perf_counter()
                class_name, conf_score = classify(image, model, CLASS_NAMES)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                confidence_percentage = float(conf_score * 100.0)

                for i in range(100):
                    time.sleep(0.003)
                    progress_bar.progress(i + 1)

            st.markdown("---")
            st.markdown(
                f"**🔎 Proba brute PNEUMONIA (sortie modèle)** : `{conf_score:.4f}`  "
                f"&nbsp;&nbsp;·&nbsp;&nbsp; ⏱️ Latence d'inférence : `{latency_ms:.1f} ms`"
            )

            # Warnings qualité si besoin
            if quality["warnings"]:
                st.warning(
                    "⚠️ Qualité d'image à interpréter avec prudence :\n\n"
                    + "\n".join(f"- {w}" for w in quality["warnings"])
                )

            col_img, col_res = st.columns([1, 1])

            # ---------- VISUELS (image + Grad-CAM) ----------
            with col_img:
                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.markdown("##### 📷 Image originale")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, width="stretch")
                    st.markdown("</div>", unsafe_allow_html=True)

                with sub_col2:
                    st.markdown("##### 🧠 Grad-CAM (zone de focus)")

                    alpha = st.slider(
                        "Transparence de la Grad-CAM",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.45,
                        step=0.05,
                        key="gradcam_alpha",
                    )

                    try:
                        gradcam_img = generate_gradcam(image, model, alpha=alpha)
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(gradcam_img, width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Impossible de générer la Grad-CAM : {e}")

            # ---------- RÉSULTAT & SCORE ----------
            with col_res:
                st.markdown("#### 🎯 Résultat du modèle")

                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                badge_class = "badge-normal" if class_name == "NORMAL" else "badge-pneumonia"
                icon = "✅" if class_name == "NORMAL" else "⚠️"

                st.markdown(
                    f"""
                    <div class="diagnosis-badge {badge_class}">
                        {icon} {class_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="confidence-score">
                        {confidence_percentage:.1f}%
                    </div>
                    <div class="confidence-label">
                        Niveau de confiance du modèle
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)

                st.plotly_chart(
                    create_gauge_chart(confidence_percentage, "Score de confiance"),
                    width="stretch",
                )

            # ---------- ANALYSE DÉTAILLÉE ----------
            st.markdown("---")
            st.markdown("### 📊 Analyse détaillée")

            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("**Diagnostic IA**")
                st.markdown(f"<span class='glow-text'>{class_name}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_d2:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("**Confiance**")
                st.markdown(
                    f"<span class='glow-text'>{confidence_percentage:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_d3:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("**Type de sortie**")
                st.markdown(
                    "<span class='glow-text'>Classification binaire (sigmoïde)</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # ---------- RECOMMANDATIONS ----------
            st.markdown("---")
            st.markdown("### 💡 Recommandations générales (non médicales)")

            if class_name == "PNEUMONIA":
                st.error(
                    "⚠️ **Suspicion de pneumonie par le modèle IA.**\n\n"
                    "- Conservez cette image et ce rapport.\n"
                    "- Ne prenez **aucune décision médicale** sur cette base.\n"
                    "- Consultez un médecin si vous avez des symptômes."
                )
            else:
                st.success(
                    "✅ **Aucun signe de pneumonie détecté par le modèle.**\n\n"
                    "- Ce résultat ne remplace jamais une lecture par un radiologue.\n"
                    "- En cas de symptômes, consultez un professionnel de santé."
                )

            # ---------- ASSISTANT IA ----------
            st.markdown("---")
            st.markdown("### 🤖 Assistant médical IA")

            with st.expander("📋 Explication du résultat (IA médicale)", expanded=True):
                with st.spinner("🧠 L'assistant IA prépare une explication..."):
                    explanation = get_medical_advice(class_name, confidence_percentage)
                    st.markdown(
                        f"""
                        <div class="info-card" style="background: rgba(0, 212, 255, 0.04);">
                            {explanation}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("### 💬 Poser une question à l'assistant")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_question = st.text_input(
                "Votre question (symptômes, compréhension du résultat, etc.)",
                placeholder="Ex : Quels sont les signes cliniques typiques d'une pneumonie ?",
            )

            col_q1, col_q2 = st.columns([3, 1])

            with col_q1:
                if st.button("🔍 Demander à l'assistant"):
                    if user_question:
                        with st.spinner("🤔 L'assistant réfléchit..."):
                            answer = get_medical_advice(
                                class_name, confidence_percentage, user_question
                            )
                            st.session_state.chat_history.append(
                                {"question": user_question, "answer": answer}
                            )

            with col_q2:
                if st.button("🗑️ Effacer l'historique"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()

            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("#### 📜 Historique des échanges avec l'assistant")

                for chat in reversed(st.session_state.chat_history):
                    st.markdown(
                        f"""
                        <div class="info-card" style="margin: 10px 0;">
                            <p style="color: #00f5ff; font-weight: 600;">❓ Question :</p>
                            <p style="color: #ffffff; margin-left: 16px;">{chat['question']}</p>
                            <p style="color: #00ff88; font-weight: 600; margin-top: 6px;">💡 Réponse IA :</p>
                            <p style="color: rgba(255,255,255,0.9); margin-left: 16px;">{chat['answer']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with tab_about:
    st.markdown("### ℹ️ À propos du modèle et du dataset")
    st.markdown(
        """
- **Backbone** : DenseNet121 pré-entraîné sur ImageNet, affiné sur le dataset *Chest X-Ray Pneumonia* (Kaggle).
- **Tête de classification** : GlobalAveragePooling2D + Dropout + Dense(1, activation sigmoïde).
- **Perte** : Focal Loss (déséquilibre NORMAL / PNEUMONIA).
- **Optimisation** : AdamW, fine-tuning en deux phases.
- **Métriques test** : ≈ 90 % accuracy, AUC ≈ 0.95.

Preuve de concept d'IA médicale, à usage **pédagogique uniquement**.
"""
    )

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: rgba(255, 255, 255, 0.55); padding: 18px;'>
    <p>🔬 Développé par <b>Imadeddine EL JEDDAOUI</b></p>
    <p style='font-size: 0.8em;'>
        Cet outil est à but éducatif uniquement et ne remplace en aucun cas un avis médical professionnel.
    </p>
</div>
""",
    unsafe_allow_html=True,
)