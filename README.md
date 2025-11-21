# 🫁 AI Pneumonia Detector – Radiologie Assistée par IA (DenseNet121 + Grad-CAM)

Système avancé d'analyse de radiographies thoraciques basé sur **DenseNet121**, utilisant un vrai dataset médical (_Chest X-Ray Pneumonia – Kaggle_) et une interface d'inférence **Streamlit**.  
Le projet inclut également une visualisation **Grad-CAM** pour interpréter les décisions du modèle.

📌 Développé dans un contexte académique en **Deep Learning / Vision Médicale**.  
❗ **Usage strictement pédagogique — ne remplace en aucun cas un diagnostic médical.**

---

# 1. 🗂 Structure du projet

```bash
lab_pneumonia/
├── README.md
├── django/                        # Squelettes API Django (optionnel)
├── fastAPI/                       # Squelettes API FastAPI (optionnel)
├── flask/                         # Squelettes API Flask (optionnel)
├── notebooks/
│   └── pneumonia_ultra_pro.ipynb  # Notebook d'entraînement (Google Colab, GPU T4)
└── streamlit/
    ├── app.py                     # Interface Streamlit (diagnostic + Grad-CAM)
    ├── bgs/images.jpg             # Image de fond
    ├── model/
    │   ├── labels.txt             # Labels NORMAL / PNEUMONIA
    │   └── pneumonia_classifier.h5# Modèle DenseNet121 entraîné
    ├── util.py                    # Prétraitement, classification, Grad-CAM
    └── requirements.txt           # Dépendances exactes


⸻

2. 🔬 Détails techniques

✔ Architecture modèle
	•	DenseNet121 (pretrained ImageNet)
	•	Fine-tuning sur dataset médical Kaggle
	•	Head personnalisée :
	•	GlobalAveragePooling2D
	•	Dropout
	•	Dense(1, activation="sigmoid")

✔ Tâche

Classification binaire :
	•	NORMAL (0)
	•	PNEUMONIA (1)

Sortie du modèle → probabilité ( p \in [0,1] ).

⸻

##3. 📦 Dataset utilisé (Kaggle Chest X-Ray Pneumonia)

Dataset réel composé de radiographies thoraciques pédiatriques annotées par des professionnels.

📎 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Structure :
	•	train/
	•	val/
	•	test/

Le notebook pneumonia_ultra_pro.ipynb détaille :
	•	Chargement & nettoyage des données
	•	Augmentations
	•	Entraînement GPU T4 (Google Colab)
	•	Évaluation
	•	Export du modèle final .h5

⸻

4. ⚙️ Installation & exécution

4.1. Cloner le projet

git clone <URL_DU_REPO>
cd lab_pneumonia

4.2. Créer l’environnement virtuel

python3 -m venv venv
source venv/bin/activate

4.3. Installer les dépendances de l’interface

cd streamlit
pip install -r requirements.txt

4.4. Lancer l’application

streamlit run app.py

👉 L’application s’ouvre automatiquement sur :

http://localhost:8501


⸻

5. 🧠 Comment fonctionne la prédiction ?

Lors d’un upload d’image :
	1.	L’image est convertie en RGB
	2.	Redimensionnée en 224×224
	3.	Normalisée dans [-1, 1]
	4.	Passée au modèle → sortie ( p ):
	•	p ≥ 0.5 → PNEUMONIA
	•	p < 0.5 → NORMAL

L’application affiche :
	•	Proba brute
	•	Classe prédite
	•	Jauge de confiance
	•	Heatmap Grad-CAM

⸻

## 6. 🔥 Visualisation Grad-CAM (Interprétation)

Le Grad-CAM met en évidence les régions de l’image utilisées par le modèle.

🎨 Légende des couleurs

## Couleur	Signification
🔴 Rouge	Zone très importante pour la décision
🟡 Jaune	Importance modérée
🔵 Bleu	    Zone ignorée

⚠ Important
Grad-CAM ≠ zone malade
C’est une explication du raisonnement du modèle, pas un outil clinique.

⸻

7. 🩺 Interprétation des résultats (à l’attention du médecin)

✔ Score élevé (≥ 0.80)

Probabilité forte selon le modèle

✔ Score intermédiaire (0.55 – 0.75)

Zone grise → modèle incertain
Fréquent même pour les radiologues (qualité image, bruit, subjectivité).

✔ Score faible (< 0.50)

Modèle penche pour NORMAL
Toujours nécessiter un avis spécialisé.

⸻

8. ❗ Limitations & avertissements
	•	Qualité image fortement impactante
	•	Risque de faux positifs / faux négatifs
	•	Ne doit JAMAIS être utilisé pour décider un traitement

⸻

9. 📘 Comment utiliser Grad-CAM efficacement ?
	1.	Regarder si l’activation se concentre dans la zone pulmonaire
	2.	Si l’attention est dispersée :
	•	Image bruitée
	•	Mauvais centrage
	•	Radiographie atypique
	3.	Pour un diagnostic correct, Grad-CAM doit montrer :
	•	Des points chauds dans les zones d’opacités ou infiltrations
	•	Peu ou pas d’activité sur les bords, côtes, diaphragme

⸻

10. 🛠 Technologies utilisées
	•	TensorFlow 2.19
	•	Keras 3.12
	•	Streamlit
	•	NumPy / Pillow
	•	OpenCV (Grad-CAM)
	•	Google Colab (GPU T4)
	•	DenseNet121 (ImageNet)

⸻

11. 📜 Licence

Usage académique uniquement.

⸻

12. 👤 Auteur & contact

Projet développé par Imad Eljeddaoui
imadeljeddaoui545@gmail.com
Étudiant ingénieur informatique EMSI — option MIAGE&UNICA — M2/MBDS, Développement & DL/IA.
Passionné par la vision médicale, le deep learning et l’innovation IA.

⸻


---

```
