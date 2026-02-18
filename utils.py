import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

IMG_SIZE = 256

# Load model once
@tf.keras.utils.register_keras_serializable()
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_food(model, image, class_names):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    label = class_names[class_idx]
    return label, confidence

def calculate_nutrition(food_label, weight, nutrition_df):
    rows = nutrition_df[nutrition_df["label"] == food_label]
    if rows.empty:
        return None
    row = rows.iloc[0]

    # Schema A: values already stored as per-gram nutrients.
    per_gram_cols = {
        "calories_per_gram": "Calories (kcal)",
        "protein_per_gram": "Protein (g)",
        "carbs_per_gram": "Carbs (g)",
        "fat_per_gram": "Fats (g)",
        "fiber_per_gram": "Fiber (g)",
        "sugar_per_gram": "Sugar (g)",
        "sodium_per_gram": "Sodium (mg)",
    }
    if all(col in nutrition_df.columns for col in per_gram_cols):
        return {out_key: row[col] * weight for col, out_key in per_gram_cols.items()}

    # Schema B: values stored for a specific serving weight (e.g., 80g/100g/120g).
    absolute_cols = {
        "calories": "Calories (kcal)",
        "protein": "Protein (g)",
        "carbohydrates": "Carbs (g)",
        "fats": "Fats (g)",
        "fiber": "Fiber (g)",
        "sugars": "Sugar (g)",
        "sodium": "Sodium (mg)",
    }
    if "weight" in nutrition_df.columns and all(col in nutrition_df.columns for col in absolute_cols):
        # Use the nearest known serving row for this label, then scale linearly.
        nearest_idx = (rows["weight"] - weight).abs().idxmin()
        nearest_row = rows.loc[nearest_idx]
        base_weight = float(nearest_row["weight"])
        if base_weight <= 0:
            return None

        ratio = float(weight) / base_weight
        return {out_key: float(nearest_row[col]) * ratio for col, out_key in absolute_cols.items()}

    return None
