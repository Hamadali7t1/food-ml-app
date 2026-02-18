import gdown
import os

url =https://drive.google.com/drive/folders/1mTI34Qq3frhW90pGTB6zS15RdAQkoiio?usp=drive_link
output = "class_names.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

from db import (
    add_intake_entry,
    clear_today_entries,
    create_user,
    get_daily_totals,
    get_hourly_totals,
    get_recent_entries,
    init_db,
    verify_user,
)
from utils import calculate_nutrition, predict_food


st.set_page_config(
    page_title="AI Nutrition Assistant",
    layout="wide",
)


@st.cache_resource
def load_model():
    candidate_paths = [
        "model/food101_efficientnetb2_final.h5",
        "model/best_efficientnetb2_food101.keras",
        "model/food101_mobilenetv2_final.h5",
        "model/best_mobilenetv2_food101.keras",
    ]
    for model_path in candidate_paths:
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            continue
    raise FileNotFoundError("No compatible model file found in model/.")


@st.cache_data
def load_nutrition():
    return pd.read_csv("data/nutrition.csv")


@st.cache_data
def load_class_names():
    with open("model/class_names.pkl", "rb") as f:
        return pickle.load(f)


# Preload resources before rendering the UI.
model = load_model()
nutrition_df = load_nutrition()
class_names = load_class_names()

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "name" not in st.session_state:
    st.session_state.name = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"
if "auth_flash" not in st.session_state:
    st.session_state.auth_flash = None

st.title("AI Food Recognition and Nutrition Tracker")

st.sidebar.header("Account")
if not st.session_state.authenticated:
    auth_mode = st.sidebar.radio(
        "Choose action",
        ["Login", "Sign up"],
        key="auth_mode",
    )
    if st.session_state.auth_flash:
        st.sidebar.success(st.session_state.auth_flash)
        st.session_state.auth_flash = None

    if auth_mode == "Login":
        with st.sidebar.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")

        if login_submit:
            ok, user_id, name, message = verify_user(login_email, login_password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.session_state.name = name
                st.rerun()
            st.sidebar.error(message)

    else:
        with st.sidebar.form("signup_form"):
            signup_name = st.text_input("Name")
            signup_email = st.text_input("Email")
            signup_password = st.text_input("Password", type="password")
            signup_submit = st.form_submit_button("Create account")

        if signup_submit:
            ok, message = create_user(signup_name, signup_email, signup_password)
            if ok:
                st.session_state.auth_mode = "Login"
                st.session_state.auth_flash = message
                st.rerun()
            else:
                st.sidebar.error(message)

    st.info("Login required to save and monitor your intake data.")
    st.stop()
else:
    st.sidebar.success(f"Logged in as: {st.session_state.name}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.name = None
        st.rerun()

st.sidebar.header("Upload and Portion")
uploaded_file = st.sidebar.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])
weight = st.sidebar.number_input("Enter weight (grams)", min_value=1, step=1)

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, width="stretch")

    with col2:
        with st.spinner("Analyzing..."):
            label, confidence = predict_food(model, image, class_names)
        display_label = label.replace("_", " ").title()

        st.success(f"Prediction: {display_label}")
        st.write(f"Confidence: {confidence:.2%}")

        nutrients = calculate_nutrition(label, weight, nutrition_df)
        if nutrients is None:
            st.warning("Nutrition data not found for this food label.")
        else:
            st.markdown("### Nutritional Breakdown")
            row1 = st.columns(2)
            row2 = st.columns(2)
            row1[0].metric("Calories (kcal)", f"{nutrients['Calories (kcal)']:.0f}")
            row1[1].metric("Protein (g)", f"{nutrients['Protein (g)']:.1f}")
            row2[0].metric("Carbs (g)", f"{nutrients['Carbs (g)']:.1f}")
            row2[1].metric("Fats (g)", f"{nutrients['Fats (g)']:.1f}")

            macro_data = {
                "Macro": ["Protein", "Carbs", "Fats"],
                "Value": [
                    nutrients["Protein (g)"],
                    nutrients["Carbs (g)"],
                    nutrients["Fats (g)"],
                ],
            }
            fig = px.pie(macro_data, names="Macro", values="Value", title="Macronutrient Distribution")
            st.plotly_chart(fig, width="stretch")

            if st.button("Add to Intake Log"):
                add_intake_entry(
                    user_id=st.session_state.user_id,
                    food_label=label,
                    weight_grams=weight,
                    confidence=confidence,
                    nutrients=nutrients,
                )
                st.success("Entry saved to your account.")
                st.rerun()

st.markdown("---")
st.header("Daily Nutrition Summary")
daily = get_daily_totals(st.session_state.user_id)

row1 = st.columns(2)
row2 = st.columns(2)
row1[0].metric("Total Calories (kcal)", f"{daily['Calories']:.0f}")
row1[1].metric("Total Protein (g)", f"{daily['Protein']:.1f}")
row2[0].metric("Total Carbs (g)", f"{daily['Carbs']:.1f}")
row2[1].metric("Total Fats (g)", f"{daily['Fats']:.1f}")

st.header("Hourly Intake (Today)")
hourly_df = get_hourly_totals(st.session_state.user_id)
if hourly_df.empty:
    st.info("No entries logged today.")
else:
    hourly_long = hourly_df.melt(id_vars=["hour"], var_name="Nutrient", value_name="Amount")
    fig_hourly = px.bar(
        hourly_long,
        x="hour",
        y="Amount",
        color="Nutrient",
        barmode="group",
        title="Hourly Nutrient Intake",
    )
    st.plotly_chart(fig_hourly, width="stretch")

st.header("Recent Entries")
recent_df = get_recent_entries(st.session_state.user_id, limit=30)
if recent_df.empty:
    st.info("No intake history yet.")
else:
    recent_df = recent_df.rename(
        columns={
            "created_at": "Timestamp",
            "food_label": "Food",
            "weight_grams": "Weight (g)",
            "confidence": "Confidence",
            "calories": "Calories (kcal)",
            "protein": "Protein (g)",
            "carbs": "Carbs (g)",
            "fats": "Fats (g)",
        }
    )
    recent_df["Confidence"] = (recent_df["Confidence"] * 100).round(2).astype(str) + "%"
    st.dataframe(recent_df, hide_index=True)

if st.button("Reset Today's Intake"):
    clear_today_entries(st.session_state.user_id)
    st.success("Today's intake data has been reset.")
    st.rerun()


