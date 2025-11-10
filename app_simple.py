# -*- coding: utf-8 -*-
"""House Price Predictor Streamlit App

This app loads an XGBoost model and uses explicit feature engineering to predict house prices.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "xgboost_model.pkl"
META_PATH = APP_DIR / "model_metadata.pkl"

# Page configuration
st.set_page_config(
    page_title="House Price Predictor - Simple",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.markdown("### Quick Price Estimate")
st.markdown("---")

# ---------------------------------------------------------------------
# Load model + metadata
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(META_PATH)
        return model, metadata
    except FileNotFoundError as e:
        st.error(
            f"Missing required file: {e}. Ensure both 'xgboost_model.pkl' "
            f"and 'model_metadata.pkl' are in the repository."
        )
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, metadata = load_model()

# ---------------------------------------------------------------------
# Helper: prediction using SAME logic as Colab
# ---------------------------------------------------------------------
def predict_price_from_basic_features(
    model,
    metadata,
    lot_size_sqft: float,
    living_area: float,
    baths: float,
    beds: float,
    year_built: int = 2000,
    postal_code: str | None = None,
    current_year: int = 2025,
) -> float:
    """
    Predict property price using the same feature engineering pipeline
    used during training / Colab experiments.
    """

    feature_names = metadata["feature_names"]
    smear = float(metadata.get("smearing_factor", 1.0))

    # --- Feature engineering (same as notebook) ---
    property_age = current_year - year_built
    property_age_squared = property_age**2
    total_rooms = beds + baths
    bath_bed_ratio = baths / (beds + 0.1)
    sqft_per_bedroom = living_area / (beds + 0.1)
    is_luxury = int(
        (baths >= 4) and (beds >= 4) and (living_area >= 3000)
    )

    log_living_area = np.log1p(living_area)
    log_lot_size = np.log1p(lot_size_sqft)

    # --- Start with all-zero row for every expected feature ---
    input_data = {col: 0.0 for col in feature_names}

    def set_if_exists(name: str, value):
        if name in input_data:
            input_data[name] = value

    # Base numeric features
    set_if_exists("LivingArea", living_area)
    set_if_exists("LotSizeSquareFeet", lot_size_sqft)
    set_if_exists("BathroomsTotalInteger", baths)
    set_if_exists("BedroomsTotal", beds)
    set_if_exists("YearBuilt", year_built)

    # Engineered numeric features
    set_if_exists("PropertyAge", property_age)
    set_if_exists("PropertyAge_squared", property_age_squared)
    set_if_exists("TotalRooms", total_rooms)
    set_if_exists("Bath_Bed_Ratio", bath_bed_ratio)
    set_if_exists("SqFt_Per_Bedroom", sqft_per_bedroom)
    set_if_exists("IsLuxury", is_luxury)

    # Log features
    set_if_exists("log_LivingArea", log_living_area)
    set_if_exists("log_LotSizeSquareFeet", log_lot_size)

    # Simple amenity defaults
    set_if_exists("HasGarage", 1)
    set_if_exists("HasPool", 0)
    set_if_exists("GarageSpaces", 2)
    set_if_exists("Stories", 1)

    # Optional: one-hot ZIP if model has that column
    if postal_code is not None:
        zip_str = str(postal_code).strip()
        zip_col = f"PostalCode_{zip_str}"
        if zip_col in input_data:
            input_data[zip_col] = 1.0

    # --- To DataFrame in correct order ---
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_df = input_df.astype(np.float32)

    # --- Predict on log scale ---
    if isinstance(model, xgb.Booster) or model.__class__.__name__ == "Booster":
        dtest = xgb.DMatrix(input_df, feature_names=feature_names)
        log_pred = float(model.predict(dtest)[0])
    else:
        log_pred = float(model.predict(input_df)[0])

    # --- Back-transform: log -> dollars + smearing factor ---
    predicted_price = np.expm1(log_pred) * smear
    return float(predicted_price)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
if model is not None and metadata is not None:
    st.success("✅ Model and metadata loaded successfully!")

    st.markdown("### Enter Property Details")
    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=2000,
            step=50,
        )
        beds = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    with col2:
        baths = st.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
        )
        lot_size = st.number_input(
            "Lot Size (sq ft)",
            min_value=1,
            max_value=100000,
            value=7000,
            step=100,
        )

    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            predicted_price = predict_price_from_basic_features(
                model=model,
                metadata=metadata,
                lot_size_sqft=lot_size,
                living_area=living_area,
                baths=baths,
                beds=beds,
                year_built=2000,
                postal_code="92101",
            )

            st.markdown("### 💰 Estimated Price")
            st.markdown(f"## ${predicted_price:,.0f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

else:
    st.error(
        "❌ Model components failed to load. Please check that "
        "'xgboost_model.pkl' and 'model_metadata.pkl' are present."
    )
