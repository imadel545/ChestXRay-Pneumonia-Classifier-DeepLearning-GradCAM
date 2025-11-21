import base64
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

import cv2
import tensorflow as tf


# ============================
# FOND D'ÉCRAN
# ============================
def set_background(image_file: str) -> None:
    """
    Définit l'image de fond de l'app Streamlit.
    """
    img_path = Path(image_file)
    if not img_path.exists():
        return

    with open(img_path, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()

    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# ============================
# PRÉ-TRAITEMENT IMAGE
# ============================
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Pré-traitement identique au notebook DenseNet :
    - resize 224x224
    - RGB
    - normalisation [-1, 1] (rescale=1./127.5 puis -1)
    """
    img = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img = img.convert("RGB")

    arr = np.asarray(img).astype("float32")
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


# ============================
# ANALYSE QUALITÉ IMAGE
# ============================
def analyze_image_quality(image: Image.Image) -> Dict[str, Any]:
    """
    Analyse simple de la qualité :
    - luminosité moyenne
    - contraste (écart-type)
    - ratio largeur/hauteur
    """
    img_gray = image.convert("L")
    arr = np.asarray(img_gray).astype("float32") / 255.0

    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))
    h, w = arr.shape
    aspect_ratio = float(w / h) if h > 0 else 0.0

    warnings = []

    if brightness < 0.15:
        warnings.append("Image très sombre (sous-exposée).")
    elif brightness > 0.85:
        warnings.append("Image très claire (sur-exposée).")

    if contrast < 0.05:
        warnings.append("Contraste très faible (image possiblement délavée).")

    if aspect_ratio < 0.6 or aspect_ratio > 1.8:
        warnings.append(
            "Rapport largeur/hauteur inhabituel pour une radiographie thoracique standard."
        )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "aspect_ratio": aspect_ratio,
        "warnings": warnings,
    }


# ============================
# INFÉRENCE KERAS
# ============================
def classify(image: Image.Image, model, class_names) -> Tuple[str, float]:
    """
    Retourne (class_name, confidence_score)
    - class_name: "NORMAL" ou "PNEUMONIA"
    - confidence_score: probabilité de la classe prédite (entre 0 et 1)
    """
    x = preprocess_image(image)

    # Prédiction brute
    y = model.predict(x, verbose=0)

    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"Forme de sortie inattendue: {y.shape}")

    pneumonia_prob = float(y[0, 0])  # P(y=1) = PNEUMONIA

    try:
        idx_normal = class_names.index("NORMAL")
        idx_pneumo = class_names.index("PNEUMONIA")
    except ValueError:
        class_names = ["NORMAL", "PNEUMONIA"]
        idx_normal, idx_pneumo = 0, 1

    if pneumonia_prob >= 0.5:
        pred_index = idx_pneumo
        confidence_score = pneumonia_prob
    else:
        pred_index = idx_normal
        confidence_score = 1.0 - pneumonia_prob

    class_name = class_names[pred_index]

    # Debug optionnel (tu peux commenter ces lignes si tu ne veux plus de logs Streamlit)
    st.write(f"🔍 Proba brute PNEUMONIA : {pneumonia_prob:.4f}")
    st.write(f"🔍 Classe prédite : {class_name} — confiance: {confidence_score:.4f}")

    return class_name, confidence_score


# ============================
# HEATMAP TYPE GRADIENT (SALiency MAP)
# ============================
def generate_gradcam(
    image: Image.Image,
    model,
    alpha: float = 0.45,
    target_size=(224, 224),
) -> np.ndarray:
    """
    Génère une image RGB (np.ndarray) avec une carte de chaleur basée sur
    le gradient de la sortie PNEUMONIA par rapport à l'image d'entrée.

    Avantage : on ne dépend pas des couches internes -> aucun problème
    de Functional.call / structure d'inputs.
    """
    # 1) Pré-traitement identique à la prédiction
    img_array = preprocess_image(image, target_size=target_size)  # (1, H, W, 3)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # 2) Gradients de la sortie w.r.t. l'entrée
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        # on prend la sortie pour la classe positive (PNEUMONIA)
        loss = preds[:, 0]

    grads = tape.gradient(loss, img_tensor)  # (1, H, W, 3)
    if grads is None:
        raise RuntimeError("Impossible de calculer les gradients pour la heatmap.")

    # 3) Importance par pixel = moyenne des |gradients| sur les canaux RGB
    grads_abs = tf.math.abs(grads)
    heatmap = tf.reduce_mean(grads_abs, axis=-1)[0]  # (H, W)

    # 4) Normalisation [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_np: np.ndarray = heatmap.numpy()

    # 5) Resize + overlay sur l'image originale
    original_size = image.size  # (W, H)
    heatmap_resized: np.ndarray = cv2.resize(heatmap_np, original_size)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = np.asarray(heatmap_resized, dtype=np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(
        img_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img