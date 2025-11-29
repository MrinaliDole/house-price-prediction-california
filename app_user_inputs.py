# -*- coding: utf-8 -*-
"""Interactive Streamlit app for custom California house price predictions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "xgboost_model.pkl"
META_PATH = APP_DIR / "model_metadata.pkl"

st.set_page_config(
    page_title="California Property Price Prediction", page_icon="🏠", layout="centered"
)
st.title("🏠 California Property Price Prediction")
st.caption("Provide key property details to estimate the market price.")
st.markdown("---")


@st.cache_resource
def load_model():
    """Load model and metadata from disk with basic error handling."""
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(META_PATH)
        return model, metadata
    except FileNotFoundError as exc:
        st.error(
            "Missing model artifacts. Ensure 'xgboost_model.pkl' and 'model_metadata.pkl' "
            "are present in the repository."
        )
        raise exc
    except Exception as exc:  # pragma: no cover - runtime feedback in Streamlit
        st.error(f"Error loading model: {exc}")
        raise exc


model, metadata = load_model()


# -----------------------------------------------------------------------------
# Helper: align user inputs to model feature space
# -----------------------------------------------------------------------------
def predict_price(
    *,
    model,
    metadata,
    living_area: float,
    lot_size_sqft: float,
    beds: float,
    baths: float,
    has_garage: bool,
    garage_spaces: int,
    postal_code: str | None,
    city: str | None,
    year_built: int = 2000,
    current_year: int = 2025,
) -> float:
    feature_names = metadata["feature_names"]
    smear = float(metadata.get("smearing_factor", 1.0))

    property_age = current_year - year_built
    property_age_squared = property_age**2
    total_rooms = beds + baths
    bath_bed_ratio = baths / (beds + 0.1)
    sqft_per_bedroom = living_area / (beds + 0.1)
    is_luxury = int((baths >= 4) and (beds >= 4) and (living_area >= 3000))

    log_living_area = np.log1p(living_area)
    log_lot_size = np.log1p(lot_size_sqft)

    input_data = {col: 0.0 for col in feature_names}

    def set_if_exists(name: str, value: float | int):
        if name in input_data:
            input_data[name] = value

    set_if_exists("LivingArea", living_area)
    set_if_exists("LotSizeSquareFeet", lot_size_sqft)
    set_if_exists("BathroomsTotalInteger", baths)
    set_if_exists("BedroomsTotal", beds)
    set_if_exists("YearBuilt", year_built)

    set_if_exists("PropertyAge", property_age)
    set_if_exists("PropertyAge_squared", property_age_squared)
    set_if_exists("TotalRooms", total_rooms)
    set_if_exists("Bath_Bed_Ratio", bath_bed_ratio)
    set_if_exists("SqFt_Per_Bedroom", sqft_per_bedroom)
    set_if_exists("IsLuxury", is_luxury)

    set_if_exists("log_LivingArea", log_living_area)
    set_if_exists("log_LotSizeSquareFeet", log_lot_size)

    set_if_exists("HasGarage", int(has_garage))
    set_if_exists("GarageSpaces", garage_spaces if has_garage else 0)
    set_if_exists("HasPool", 0)
    set_if_exists("Stories", 1)

    if postal_code:
        zip_col = f"PostalCode_{str(postal_code).strip()}"
        if zip_col in input_data:
            input_data[zip_col] = 1.0

    if city:
        city_col = f"City_{city}"
        if city_col in input_data:
            input_data[city_col] = 1.0

    input_df = pd.DataFrame([input_data], columns=feature_names).astype(np.float32)

    if isinstance(model, xgb.Booster) or model.__class__.__name__ == "Booster":
        dtest = xgb.DMatrix(input_df, feature_names=feature_names)
        log_pred = float(model.predict(dtest)[0])
    else:
        log_pred = float(model.predict(input_df)[0])

    return float(np.expm1(log_pred) * smear)


st.success("✅ Model and metadata loaded. Ready to predict!")

city_options = sorted([name.replace("City_", "") for name in metadata["feature_names"] if name.startswith("City_")])

col1, col2 = st.columns(2)
with col1:
    living_area = st.number_input(
        "Living Area (sq ft)", min_value=300, max_value=10000, value=1800, step=50
    )
    beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    postal_code = st.text_input(
        "Postal Code (e.g., 92101)", value="92101", help="Used for ZIP-specific effects"
    )
    city = st.selectbox(
        "City",
        options=city_options,
        index=city_options.index("San Diego") if "San Diego" in city_options else 0,
        help="Pick the city to align with model training data",
    )

with col2:
    baths = st.number_input(
        "Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5
    )
    lot_size = st.number_input(
        "Lot Size (sq ft)", min_value=500, max_value=120000, value=6000, step=100
    )
    has_garage = st.checkbox("Garage Available", value=True)
    garage_spaces = st.slider(
        "Garage Spaces", min_value=0, max_value=5, value=2, disabled=not has_garage
    )

st.markdown("---")

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    try:
        predicted_price = predict_price(
            model=model,
            metadata=metadata,
            living_area=living_area,
            lot_size_sqft=lot_size,
            beds=beds,
            baths=baths,
            has_garage=has_garage,
            garage_spaces=garage_spaces,
            postal_code=postal_code,
            city=city,
        )
        st.markdown("### 💰 Estimated Market Price")
        st.markdown(f"## ${predicted_price:,.0f}")
    except Exception as exc:  # pragma: no cover - runtime feedback in Streamlit
        st.error(f"Prediction error: {exc}")
