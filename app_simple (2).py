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

# Load model and metadata only
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)      # use robust path
        metadata = joblib.load(META_PATH)    # use robust path
        return model, metadata
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}. Ensure both 'xgboost_model.pkl' and 'model_metadata.pkl' are in the repository.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, metadata = load_model()

if model is not None and metadata is not None:
    st.success("✅ Model and Metadata loaded successfully!")

    # Simple 4-input form
    st.markdown("### Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=2000, # Updated default to match common use-case
            step=50
        )

        beds = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

    with col2:
        baths = st.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.5, # Updated default to match common use-case
            step=0.5
        )

        lot_size = st.number_input(
            "Lot Size (sq ft)",
            min_value=1, # Changed from 0 to 1 as 0 lot size is unrealistic and breaks log1p(0)
            max_value=100000,
            value=7000, # Updated default to match common use-case
            step=100
        )

    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            # --- Hardcoded Features and Feature Engineering ---
            year_built = 2000  # Default year
            current_year = 2025
            property_age = current_year - year_built
            postal_code = '92101' # Default San Diego ZIP
            
            # Prepare the features dictionary
            features = {
                'LivingArea': living_area,
                'BedroomsTotal': beds,
                'BathroomsTotalInteger': baths,
                'LotSizeSquareFeet': lot_size,

                # Engineered features and hardcoded defaults:
                'YearBuilt': year_built,
                'Stories': 1,
                'GarageSpaces': 2,
                'PropertyAge': property_age,
                'PropertyAge_squared': property_age ** 2,
                'TotalRooms': beds + baths,
                'Bath_Bed_Ratio': baths / (beds + 0.1), 
                'SqFt_Per_Bedroom': living_area / (beds + 0.1),
                'IsLuxury': int((baths >= 4) and (beds >= 4) and (living_area >= 3000)),
                'HasPool': 0,
                'HasGarage': 1,
                'log_LivingArea': np.log1p(living_area),
                'log_LotSizeSquareFeet': np.log1p(lot_size),

                # OHE feature defaults that might exist as 'PostalCode_XXX' in the model
                'PoolPrivateYN': 0,
                'BasementYN': 0,
                'FireplaceYN': 0,
                'WaterfrontYN': 0,
                'ViewYN': 0,
                'NewConstructionYN': 0,
            }
            
            # --- Performance Fix: Initialize all columns at once ---
            
            # 1. Start with all features the model expects, setting them to 0 (default/OHE off)
            input_data = {col: 0 for col in metadata['feature_names']}
            
            # 2. Update the columns that have non-zero or specific values from the inputs/engineering
            input_data.update(features)
            
            # 3. Handle the specific PostalCode OHE column
            postal_col_name = f'PostalCode_{postal_code}'
            if postal_col_name in input_data:
                input_data[postal_col_name] = 1
            
            # 4. Create the DataFrame in the correct order, avoiding fragmentation
            input_df = pd.DataFrame([input_data])[metadata['feature_names']]
            input_df = input_df.astype(np.float32)

            # --- Prediction ---

            # Handle both Booster and sklearn-style models
            if model.__class__.__name__ == "Booster":
                # Use UN-SCALED data
                dtest = xgb.DMatrix(input_df.values, feature_names=list(input_df.columns))
                yhat = model.predict(dtest)
            else:
                # Use UN-SCALED data
                yhat = model.predict(input_df)

            #log_pred = float(yhat[0])
            # Reverse the log transformation
            #predicted_price = np.expm1(log_pred) * metadata.get("smearing_factor", 1.0)
            yhat_value = float(yhat[0])

            # Reverse normalization if metadata includes target scaling
            if "target_mean" in metadata and "target_std" in metadata:
                yhat_value = (yhat_value * metadata["target_std"]) + metadata["target_mean"]

            # Heuristic adjustment for normalized log outputs
            adjusted_log = (yhat_value + 1) * 12.0
            predicted_price = np.expm1(adjusted_log) * metadata.get("smearing_factor", 1.0)

            
            # --- Display ---
            st.markdown("### 💰 Estimated Price")
            st.markdown(f"## ${predicted_price:,.0f}")

            lower = predicted_price * 0.9
            upper = predicted_price * 1.1
            st.markdown(f"**Range:** ${lower:,.0f} - ${upper:,.0f}")

            with st.expander("📊 Input Summary and Model Assumptions"):
                st.write(f"Living Area: {living_area:,} sq ft")
                st.write(f"Bedrooms: {beds}")
                st.write(f"Bathrooms: {baths}")
                st.write(f"Lot Size: {lot_size:,} sq ft")
                st.caption("*The model assumes default values like Year Built: 2000, 1 Story, 2 Garage Spaces, and Postal Code: 92101.*")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e) # Display the full error trace for debugging

        st.markdown("---")
        st.info("💡 This simplified version uses default values for features not shown above.")

# Handle case where loading failed
else:
    st.error("❌ Model components failed to load. Please check your file paths and ensure all required .pkl files are uploaded to GitHub.")


    

