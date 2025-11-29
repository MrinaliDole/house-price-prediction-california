from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------------------------------------------------------------
# Path Setup (Streamlit Cloud–safe)
# -----------------------------------------------------------------------------
APP_DIR = Path.cwd()               # avoids __file__ issues
MODEL_PATH = APP_DIR / "xgboost_model.pkl"
META_PATH = APP_DIR / "model_metadata.pkl"

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="California Property Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Property Price Prediction")
st.caption("Provide key property details to estimate the market price.")
st.markdown("---")

# -----------------------------------------------------------------------------
# Load Model + Metadata
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(META_PATH)
        return model, metadata
    except FileNotFoundError as exc:
        st.error(
            "Missing model files. Ensure 'xgboost_model.pkl' and 'model_metadata.pkl' "
            "are in the same folder as this app."
        )
        raise exc
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        raise exc

model, metadata = load_model()

st.success("✅ Model and metadata loaded successfully!")

# -----------------------------------------------------------------------------
# Prediction Logic
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

    # engineered features
    property_age = current_year - year_built
    property_age_squared = property_age ** 2
    total_rooms = beds + baths
    bath_bed_ratio = baths / (beds + 0.1)
    sqft_per_bedroom = living_area / (beds + 0.1)
    is_luxury = int((baths >= 4) and (beds >= 4) and (living_area >= 3000))

    log_living_area = np.log1p(living_area)
    log_lot_size = np.log1p(lot_size_sqft)

    # create model-aligned input vector
    input_data = {col: 0.0 for col in feature_names}

    def set_if_exists(name, value):
        if name in input_data:
            input_data[name] = value

    # base features
    set_if_exists("LivingArea", living_area)
    set_if_exists("LotSizeSquareFeet", lot_size_sqft)
    set_if_exists("BathroomsTotalInteger", baths)
    set_if_exists("BedroomsTotal", beds)
    set_if_exists("YearBuilt", year_built)

    # engineered features
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

    # ZIP / City One-Hot Encoding
    if postal_code:
        postal_code = str(postal_code).strip().replace(" ", "")
        zip_col = f"PostalCode_{postal_code}"
        if zip_col in input_data:
            input_data[zip_col] = 1.0

    if city:
        city_col = f"City_{city}"
        if city_col in input_data:
            input_data[city_col] = 1.0

    # finalize DF
    input_df = pd.DataFrame([input_data], columns=feature_names).astype(np.float32)

    # inference
    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(input_df, feature_names=feature_names)
        log_pred = float(model.predict(dtest)[0])
    else:
        log_pred = float(model.predict(input_df)[0])

    return float(np.expm1(log_pred) * smear)

# -----------------------------------------------------------------------------
# City List (Searchable Dropdown)
# -----------------------------------------------------------------------------
city_options = sorted([
    name.replace("City_", "")
    for name in metadata["feature_names"]
    if name.startswith("City_")
])

default_city_index = 0 if city_options else None

# -----------------------------------------------------------------------------
# UI Inputs
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input("Living Area (sq ft)", 300, 10000, 1800, step=50)
    beds = st.number_input("Bedrooms", 1, 10, 3)
    postal_code = st.text_input("Postal Code (e.g., 92101)", value="92101")
    city = st.selectbox(
        "City (Searchable)",
        options=city_options,
        index=default_city_index,
        help="Search any California city used in the model."
    )

with col2:
    baths = st.number_input("Bathrooms", 1.0, 10.0, 2.0, step=0.5)
    lot_size = st.number_input("Lot Size (sq ft)", 500, 120000, 6000)
    has_garage = st.checkbox("Garage Available", value=True)
    garage_spaces = st.slider("Garage Spaces", 0, 5, 2, disabled=not has_garage)

st.markdown("---")

# -----------------------------------------------------------------------------
# Prediction Button
# -----------------------------------------------------------------------------
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
        st.markdown(f"## **${predicted_price:,.0f}**")

    except Exception as exc:
        st.error(f"❌ Prediction error: {exc}")
