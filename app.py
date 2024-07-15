import streamlit as st
import numpy as np
import cv2
from descriptor import glcm, bitdesc
from distances import manhattan, euclidean, chebyshev, canberra, retrieve_similar_image
import os

# List of descriptors and distance functions
descriptors = {
    'GLCM': glcm,
    'BiT': bitdesc
}
distance_functions = {
    'Euclidean': euclidean,
    'Manhattan': manhattan,
    'Chebyshev': chebyshev,
    'Canberra': canberra
}

# Load precomputed features
signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
features_db = {
    'GLCM': signatures_glcm,
    'BiT': signatures_bitdesc
}

# Streamlit interface
st.title('Content-Based Image Retrieval (CBIR)')

uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
descriptor_choice = st.selectbox("Choisissez un descripteur", ['GLCM', 'BiT'])
distance_choice = st.selectbox("Choisissez une mesure de distance", ['Euclidean', 'Manhattan', 'Chebyshev', 'Canberra'])
num_results = st.slider("Nombre d'images similaires à afficher", 1, 20, 5)

def ensure_three_channels(img):
    if len(img.shape) == 2:  
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img = ensure_three_channels(img)

    st.image(img, channels="BGR", caption="Uploaded Image")

    if descriptor_choice and distance_choice and num_results:
        query_features = descriptors[descriptor_choice](cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        similar_images = retrieve_similar_image(features_db[descriptor_choice], query_features, distance_choice.lower(), num_results)

        st.write(f"Top {num_results} images similaires:")

        cols = st.columns(3)
        for i, (img_path, dist, label) in enumerate(similar_images):
            print(f"Attempting to load image: {img_path}")  
            if not os.path.isabs(img_path):
                img_path = os.path.join('./images', img_path)

            similar_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            similar_img = ensure_three_channels(similar_img)
            if similar_img is not None:
                cols[i % 3].image(similar_img, channels="BGR", caption=f"images similaires - {label}")
            else:
                st.write(f"Échec du chargement de l'image: {img_path}")
