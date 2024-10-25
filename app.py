import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib import patches
import time

# Charger le modèle sauvegardé
@st.cache_resource
def load_trained_model():
    model = load_model("ship_detection.keras")
    return model

model = load_trained_model()

# Fonction de prédiction pour détecter les navires avec une barre de progression
def detect_ships(image, model, stride=20, threshold=0.95):
    height, width, _ = image.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Initialiser la barre de progression
    progress_bar = st.progress(0)
    total_steps = (height - 80) // stride * (width - 80) // stride
    current_step = 0

    for h in range(0, height-80, stride):
        for w in range(0, width-80, stride):
            img_box = image[h:h+80, w:w+80]
            img_box = cv2.resize(img_box, (80, 80))  # S'assurer que la taille est compatible
            img_box = np.expand_dims(img_box, axis=0)

            # Faire la prédiction
            prediction = model.predict(img_box, verbose=False)
            prediction_probability = np.max(prediction)
            prediction_class = np.argmax(prediction)

            # Afficher le rectangle rouge si un navire est détecté avec une forte probabilité
            if prediction_class == 1 and prediction_probability > threshold:
                ax.add_patch(patches.Rectangle((w, h), 80, 80, edgecolor='r', facecolor='none'))

            # Mettre à jour la barre de progression
            current_step += 1
            progress_bar.progress(min(current_step / total_steps, 1.0))

    st.pyplot(fig)

# Interface utilisateur Streamlit
st.title("Détection de Navires sur Images Satellites")
st.write("Téléversez une image satellite pour détecter des navires.")

# Charger une image
uploaded_image = st.file_uploader("Charger une image satellite", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Lire et afficher l'image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir l'image en RGB
    st.image(image, caption="Image satellite téléversée", use_column_width=True)

    # Bouton pour lancer la détection
    if st.button("Détecter les navires"):
        with st.spinner("Détection en cours, cela peut prendre quelques instants..."):
            detect_ships(image, model)
        st.success("Détection terminée !")
