# AI Food Recognition and Nutrition Tracker

## Introduction
AI Food Recognition and Nutrition Tracker is a Streamlit-based web application that predicts food from an uploaded image and calculates nutrient intake based on portion size.  
The app includes secure user authentication, database-backed intake logging, and per-user daily/hourly nutrition monitoring.

## Problem Statement
Many people want to track daily nutrition, but manual food logging is time-consuming and often inaccurate.  
Users need a faster way to:
- Identify food from images
- Estimate nutrition by portion size
- Store and monitor personal intake history over time

## Solution
This project combines computer vision and nutrition analytics in one workflow:
1. User uploads a food image.
2. A trained deep learning model predicts the food class.
3. User enters food weight (grams).
4. The app computes nutrients from a nutrition dataset.
5. Intake is saved to a user-specific SQLite database.
6. The app displays daily totals, hourly trends, and recent intake entries.

## Key Features
- Food image classification with TensorFlow/Keras model
- Nutrition estimation based on portion weight
- Secure signup/login with hashed passwords
- Unique email enforcement for user accounts
- Per-user persistent intake logs
- Daily nutrient summary (calories, protein, carbs, fats)
- Hourly nutrient monitoring chart
- Recent entries table
- Reset today’s intake data

## Tech Stack
- Frontend/App: Streamlit
- ML Inference: TensorFlow / Keras
- Data Handling: Pandas, NumPy
- Visualization: Plotly
- Image Processing: Pillow
- Database: SQLite

## Project Structure
```text
Nutrient Calculator/
├─ app.py              # Main Streamlit app (UI, auth flow, prediction, dashboards)
├─ utils.py            # Image preprocessing, model prediction, nutrition calculation
├─ db.py               # Database schema, auth, intake persistence, reporting queries
├─ requirements.txt    # Python dependencies
├─ model/              # Trained model files + class names
└─ data/               # Nutrition CSV + SQLite database
```

## How It Works
### 1. Authentication
- Signup requires `name`, `email`, and `password`.
- Login requires `email` and `password`.
- Passwords are salted and hashed using PBKDF2-SHA256.

### 2. Prediction Pipeline
- Uploaded image is resized to 256x256 RGB.
- EfficientNet-compatible preprocessing is applied.
- Model predicts a food label and confidence score.
- Label display is formatted for UI readability.

### 3. Nutrition Calculation
- User-provided weight is used to scale nutrient values.
- Supports nutrition datasets with either:
  - per-gram columns, or
  - absolute serving values with a `weight` column.

### 4. Tracking and Monitoring
- Intake entries are stored per user with timestamp.
- Daily totals are aggregated from database entries.
- Hourly chart groups today’s intake by hour.

## Database Design (SQLite)
### `users`
- `id` (PK)
- `name`
- `email` (UNIQUE)
- `password_hash`
- `salt`
- `created_at`

### `intake_entries`
- `id` (PK)
- `user_id` (FK -> users.id)
- `food_label`
- `weight_grams`
- `confidence`
- `calories`
- `protein`
- `carbs`
- `fats`
- `created_at`

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Usage
1. Create an account (name, email, password).
2. Log in with email and password.
3. Upload a food image and enter weight.
4. Review prediction + nutrient breakdown.
5. Save entry to intake log.
6. Monitor daily and hourly nutrient trends.

## Current Scope and Notes
- Nutrition values depend on the quality and coverage of `data/nutrition.csv`.
- Prediction quality depends on model training quality and input image quality.
- This app is intended for educational and productivity use, not medical diagnosis.

## Future Improvements
- Stronger email validation and password policy
- Password reset and profile management
- Weekly/monthly analytics dashboards
- Multi-food meal detection
- Export reports (CSV/PDF)

## Author
**Hamad Ali**  
Email: **hamadali7t1@gmail.com**

