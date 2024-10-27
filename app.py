import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from matplotlib import patches
import time
import os

# Vérifier si le GPU est disponible
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    st.write("GPU détecté. Utilisation du GPU pour l'inférence.")
else:
    st.write("GPU non détecté. L'inférence se fera sur le CPU.")

# Charger le modèle sauvegardé
# @st.cache_resource
# def load_trained_model():
#     model = load_model("ship_detection_model")
#     return model

# model = load_trained_model()

# Fonction de détection des navires avec optimisations
def detect_ships(image, model, stride=40, threshold=0.90, batch_size=64, scale_percent=60):
    start_time = time.time()

    # Redimensionner l'image pour accélérer le traitement
    original_height, original_width = image.shape[:2]
    width = int(original_width * scale_percent / 100)
    height = int(original_height * scale_percent / 100)
    image_resized = cv2.resize(image, (width, height))

    # Préparer l'affichage de l'image avec matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_resized)

    # Initialiser la barre de progression
    progress_bar = st.progress(0)
    total_steps = ((height - 80) // stride + 1) * ((width - 80) // stride + 1)
    current_step = 0

    batch = []
    batch_positions = []

    # Parcourir les sous-images avec un stride optimisé
    for h in range(0, height - 80 + 1, stride):
        for w in range(0, width - 80 + 1, stride):
            img_box = image_resized[h:h+80, w:w+80]
            batch.append(img_box)
            batch_positions.append((h, w))

            # Traiter les prédictions par lots
            if len(batch) == batch_size:
                predictions = model.predict(np.array(batch), verbose=False)
                for i, prediction in enumerate(predictions):
                    if np.argmax(prediction) == 1 and np.max(prediction) > threshold:
                        h_pos, w_pos = batch_positions[i]
                        rect = patches.Rectangle((w_pos, h_pos), 80, 80, edgecolor='r', facecolor='none', linewidth=2)
                        ax.add_patch(rect)

                # Réinitialiser le batch et positions
                batch = []
                batch_positions = []

            # Mettre à jour la barre de progression
            current_step += 1
            if current_step % 10 == 0 or current_step == total_steps:
                progress_bar.progress(min(current_step / total_steps, 1.0))

    # Traiter les derniers éléments restants dans le batch
    if batch:
        predictions = model.predict(np.array(batch), verbose=False)
        for i, prediction in enumerate(predictions):
            if np.argmax(prediction) == 1 and np.max(prediction) > threshold:
                h_pos, w_pos = batch_positions[i]
                rect = patches.Rectangle((w_pos, h_pos), 80, 80, edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(rect)

    # Affichage final du temps écoulé
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.pyplot(fig)
    st.write(f"Temps de traitement total : {elapsed_time:.2f} secondes")

# Interface utilisateur Streamlit
st.title("ShipFinder")
st.subheader("Détection de Navires sur Images Satellites")

# Option pour sélectionner le mode d'entrée
option = st.radio(
    "Choisissez une méthode pour fournir l'image :",
    ('Sélectionner une image existante', 'Téléverser votre propre image')
)

# Initialisation de la variable image
image = None
image_name = None

if option == 'Sélectionner une image existante':
    # Chemin vers le dossier contenant les images
    image_folder = "./shipsnet/scenes/"

    # Obtenir la liste des fichiers image dans le dossier
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Vérifier s'il y a des images dans le dossier
    if len(image_files) == 0:
        st.write("Aucune image trouvée dans le dossier spécifié.")
    else:
        # Ajouter un menu déroulant pour sélectionner une image
        selected_image_name = st.selectbox("Sélectionnez une image", image_files)

        # Charger et afficher l'image sélectionnée
        image_path = os.path.join(image_folder, selected_image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir l'image en RGB
        st.image(image, caption=f"Image sélectionnée : {selected_image_name}", use_column_width=True)
        image_name = selected_image_name

elif option == 'Téléverser votre propre image':
    # Charger une image
    uploaded_image = st.file_uploader("Charger une image satellite", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Lire et afficher l'image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir l'image en RGB
        st.image(image, caption="Image satellite téléversée", use_column_width=True)
        image_name = uploaded_image.name

# Bouton pour lancer la détection
if image is not None:
    if st.button("Détecter les navires"):
        with st.spinner("Détection en cours, cela peut prendre quelques instants..."):
            detect_ships(image, model)
        st.success("Détection terminée !")
