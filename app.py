import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os

# ==========================================================
# Section 1: Feature Extraction and Recommendation Functions
# ==========================================================

# Load precomputed embeddings and dataset
df = pd.read_csv("styles_cleaned.csv", encoding="utf-8")
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filename = pickle.load(open("filename.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
complementary_mapping=pickle.load(open("complementary_mapping.pkl", "rb"))

#=====================================
# Define function to extract features
#=====================================

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    exp_img = np.expand_dims(img_array, axis=0)
    prepro_img = preprocess_input(exp_img)
    result = model.predict(prepro_img, verbose=0).flatten()
    return result / norm(result)

#===================================
# Function to find similar products
#===================================

def find_similar_products(uploaded_image_path, model):
    uploaded_features = extract_features(uploaded_image_path, model)
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances,indices = neighbors.kneighbors([uploaded_features])
    return [filename[idx] for idx in indices[0]]

#=============================================
# Function to detect article type & gender
#=============================================

def get_article_type_and_gender(uploaded_image_path, model):
    similar_img = find_similar_products(uploaded_image_path, model)[0]  # Get closest match
    img_id = os.path.splitext(os.path.basename(similar_img))[0]
    matching_row = df[df["id"] == int(img_id)]
    if not matching_row.empty:
        return matching_row.iloc[0]["articleType"], matching_row.iloc[0]["gender"]
    return None, None

#================================================
# Function to get occasion-based recommendations
#================================================

import random
def get_occasion_recommendations(article_type, gender):
    article_type, gender = article_type.strip().lower(), gender.strip().lower()
    df["articleType"] = df["articleType"].str.strip().str.lower()
    df["gender"] = df["gender"].str.strip().str.lower()

    # Check if article type has predefined complementary items
    if article_type in complementary_mapping:
        related_articles = complementary_mapping[article_type]
    else:
        # Fallback: Get other articles from the same occasion
        occasion = df[df["articleType"] == article_type]["usage"].values
        if len(occasion) == 0:
            return [("No occasion found", "Check articleType spelling")]
        occasion = occasion[0]  # Assume one occasion per item
        related_articles = df[df["usage"] == occasion]["articleType"].unique().tolist()

    # Filter for recommendations (same gender, related article types)
    recommendations = df[
        (df["articleType"].isin(related_articles)) & 
        (df["gender"] == gender)
    ]

    # Ensure uniqueness: Pick one random product per articleType
    recommendations = recommendations.groupby("articleType").apply(lambda x: x.sample(1, random_state=random.randint(1, 10000)))

    # Select top 5 unique recommendations
    return recommendations[["articleType", "productDisplayName", "id"]].head(5).values.tolist() or [("No complementary items found", "Try another product")]

# ==========================================================
# Section 2: StreamLit UI
# ==========================================================

st.title("Fashion Recommender System 👗👔")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_path = "uploaded_image.jpg"
    image_pil = Image.open(uploaded_file)
    image_pil.save(img_path)

    # Display uploaded image in small size
    st.image(image_pil, caption="Uploaded Image", width=150)

    # Find similar products
    similar_products = find_similar_products(img_path, model)

    # Detect article type & gender
    article_type, gender = get_article_type_and_gender(img_path, model)

    # Get occasion-based recommendations
    recommendations = get_occasion_recommendations(article_type, gender)

    st.subheader("🔍 Similar Products")
    col_sim = st.columns(5)  # 5 products in one row

    for i, product_path in enumerate(similar_products[:5]):
        img = Image.open(product_path)

        # Extract product ID from the image filename
        product_id = os.path.splitext(os.path.basename(product_path))[0]

        # Retrieve product details from the dataset
        product_info = df[df["id"] == int(product_id)]  # Match image ID with dataset
        if not product_info.empty:
            product_name = product_info.iloc[0]["productDisplayName"]
        else:
            product_name = "No details available"

        # Display image and details
        with col_sim[i]:
            st.image(img, width=120)
            st.caption(product_name)  # Smaller text for product name


    st.subheader("🎉 Occasion-Based Recommendations")
    col_rec = st.columns(5)  # 5 recommended products in one row
    for i, rec in enumerate(recommendations[:5]):
        product_img_path = f"images/{rec[2]}.jpg"  # Assuming images are named by ID
        if os.path.exists(product_img_path):
            img = Image.open(product_img_path)
            with col_rec[i]:
                st.image(img, caption=f"{rec[0]}: {rec[1]}", width=120)
