import numpy as np
import streamlit as st
from skimage.io import imread
from skimage.color import rgb2gray
from distance_metrics import euclidean_distance, manhattan_distance, chebyshev_distance, canberra_distance
from extract_glcm_features import extract_glcm_features
from extract_bit_features import extract_bit_features
import os

def load_features(feature_file):
    return np.load(feature_file, allow_pickle=True)

def get_folder_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

st.title("Recherche d'Images")

# Permet de téléverser un fichier image avec des types spécifiés
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

# Sélection du descripteur parmi les options GLCM et BiT
descriptor = st.selectbox("Choisissez le descripteur", ["GLCM", "BiT"])

# Sélection de la mesure de distance parmi les options Euclidean, Manhattan, Chebyshev et Canberra
distance_metric = st.selectbox("Choisissez la mesure de distance", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])

# Curseur pour choisir le nombre d'images similaires à afficher
number_of_images = st.slider("Nombre d'images similaires à afficher", 1, 12, 5)

if uploaded_file is not None:
    # Lecture de l'image téléversée
    image = imread(uploaded_file)
    st.image(image, caption='Image Téléversée', use_column_width=True)
    
    # Gérer les images avec un canal alpha
    if image.ndim == 3 and image.shape[2] == 4:  # Vérifie si l'image a un canal alpha
        image = image[..., :3]  # Supprime le canal alpha

    # Conversion de l'image en niveaux de gris
    gray_image = rgb2gray(image)
    
    # Extraction des caractéristiques selon le descripteur choisi
    if descriptor == "GLCM":
        features = extract_glcm_features(uploaded_file)
        stored_features = load_features('glcm_features.npy')
        stored_paths = load_features('glcm_features_paths.npy')
    elif descriptor == "BiT":
        features = extract_bit_features(uploaded_file)
        stored_features = load_features('bit_features.npy')
        stored_paths = load_features('bit_features_paths.npy')
    
    # Vérification si les caractéristiques ou les chemins ne sont pas chargés
    if stored_features is None or stored_paths is None:
        st.write("Erreur : Aucune caractéristique ou chemin chargé.")
    elif features is None:
        st.write("Erreur : Impossible d'extraire les caractéristiques de l'image téléversée.")
    else:
        # Dictionnaire des fonctions de distance
        distance_functions = {
            "Euclidean": euclidean_distance,
            "Manhattan": manhattan_distance,
            "Chebyshev": chebyshev_distance,
            "Canberra": canberra_distance
        }
        
        # Calcul des distances entre les caractéristiques extraites et les caractéristiques stockées
        distances = [distance_functions[distance_metric](features, f) for f in stored_features]
        sorted_indices = np.argsort(distances)

        # Affichage des images les plus similaires dans 4 colonnes
        for i, idx in enumerate(sorted_indices[:number_of_images]):
            image_path = stored_paths[idx]
            folder_name = get_folder_name(image_path)
            if i % 4 == 0:
                cols = st.columns(4)
            with cols[i % 4]:
                st.image(imread(image_path), caption=f"Image {i+1}\nDossier : {folder_name}")
