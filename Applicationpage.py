from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------------------------------------------------------------
# Paths (Streamlit Cloud–friendly)
# -----------------------------------------------------------------------------
APP_DIR = Path.cwd()
MODEL_PATH = APP_DIR / "xgboost_model.pkl"
META_PATH = APP_DIR / "model_metadata.pkl"
GEO_PATH = APP_DIR / "city_zip_county_mapping.csv"

# Default value for lot size (since we removed it from UI)
DEFAULT_LOT_SIZE_SQFT = 6000.0

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="California Property Price Prediction",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 California Property Price Prediction")
st.caption("Provide key property details to estimate the market price.")
st.markdown("---")


# -----------------------------------------------------------------------------
# Load model & metadata
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(META_PATH)
        return model, metadata
    except FileNotFoundError as exc:
        st.error(
            "Missing model artifacts. Ensure 'xgboost_model.pkl' and "
            "'model_metadata.pkl' are present in the app directory."
        )
        raise exc
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        raise exc


# -----------------------------------------------------------------------------
# Load geography mapping (City → ZIP → County)
# -----------------------------------------------------------------------------
@st.cache_data
def load_geography():
    if not GEO_PATH.exists():
        st.error(
            "Missing geography mapping file 'city_zip_county_mapping.csv'. "
            "Make sure it is in the same folder as this app."
        )
        return None, {}, {}

    df = pd.read_csv(GEO_PATH, dtype=str, sep=None, engine="python")
    # Normalize
    df["city"] = df["city"].astype(str).str.strip()
    df["zip"] = df["zip"].astype(str).str.strip()
    df["county"] = df["county"].astype(str).str.strip()

    # Build mapping dicts
    city_to_zips = (
        df.groupby("city")["zip"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )
    zip_to_counties = (
        df.groupby("zip")["county"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    return df, city_to_zips, zip_to_counties


model, metadata = load_model()
geo_df, CITY_TO_ZIPS, ZIP_TO_COUNTY = load_geography()

if geo_df is None:
    st.stop()

st.success("✅ Model, metadata, and geography mapping loaded. Ready to predict!")

# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
def predict_price(
    *,
    model,
    metadata,
    living_area: float,
    beds: float,
    baths: float,
    has_garage: bool,
    has_pool: bool,
    zip_code: str | None,
    city: str | None,
    county: str | None,
    lot_size_sqft: float = DEFAULT_LOT_SIZE_SQFT,
    year_built: int = 2000,
    current_year: int = 2025,
) -> float:
    """
    Align user inputs to model feature space and return predicted price.
    """
    feature_names = metadata["feature_names"]
    smear = float(metadata.get("smearing_factor", 1.0))

    # Engineered features
    property_age = current_year - year_built
    property_age_squared = property_age**2
    total_rooms = beds + baths
    bath_bed_ratio = baths / (beds + 0.1)
    sqft_per_bedroom = living_area / (beds + 0.1)
    is_luxury = int((baths >= 4) and (beds >= 4) and (living_area >= 3000))

    log_living_area = np.log1p(living_area)
    log_lot_size = np.log1p(lot_size_sqft)

    # Start with all-zero feature vector
    input_data = {col: 0.0 for col in feature_names}

    def set_if_exists(name: str, value: float | int):
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

    set_if_exists("log_LivingArea", log_living_area)
    set_if_exists("log_LotSizeSquareFeet", log_lot_size)

    # Booleans / flags
    set_if_exists("HasGarage", int(has_garage))
    # Since we now only have Yes/No for garage, use a representative value
    set_if_exists("GarageSpaces", 2 if has_garage else 0)
    set_if_exists("HasPool", int(has_pool))
    set_if_exists("Stories", 1)

    # One-hot: ZIP code
    if zip_code:
        zip_code_clean = str(zip_code).strip()
        zip_col = f"PostalCode_{zip_code_clean}"
        if zip_col in input_data:
            input_data[zip_col] = 1.0

    # One-hot: City
    if city:
        city_col = f"City_{city}"
        if city_col in input_data:
            input_data[city_col] = 1.0

    # One-hot: County
    if county:
        county_col = f"CountyOrParish_{county}"
        if county_col in input_data:
            input_data[county_col] = 1.0

    # Build DataFrame in correct column order
    input_df = pd.DataFrame([input_data], columns=feature_names).astype(np.float32)

    # XGBoost native booster or sklearn wrapper
    if isinstance(model, xgb.Booster) or model.__class__.__name__ == "Booster":
        dtest = xgb.DMatrix(input_df, feature_names=feature_names)
        log_pred = float(model.predict(dtest)[0])
    else:
        log_pred = float(model.predict(input_df)[0])

    # Inverse of log1p with smearing
    return float(np.expm1(log_pred) * smear)


# -----------------------------------------------------------------------------
# UI: City → Zip Code → County + other property inputs
# -----------------------------------------------------------------------------
city_options = sorted(CITY_TO_ZIPS.keys())
default_city_index = 0 if city_options else None

col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1800,
        step=50,
    )
    beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)

    city = st.selectbox(
        "City",
        options=city_options,
        index=default_city_index,
        help="Select the city where the property is located.",
    )

    # Zip Code options depend on selected city
    zip_options = CITY_TO_ZIPS.get(city, [])
    if not zip_options:
        zip_code = None
        county = None
        st.warning("No ZIP codes found for this city in the mapping.")
    else:
        zip_code = st.selectbox(
            "Zip Code",
            options=zip_options,
            help="Zip codes available for the selected city.",
        )

        # County options depend on selected ZIP
        county_options = ZIP_TO_COUNTY.get(zip_code, [])
        if not county_options:
            county = None
            st.warning("No counties found for this ZIP code in the mapping.")
        else:
            county = st.selectbox(
                "County",
                options=county_options,
                help="County inferred from the selected ZIP code.",
            )

with col2:
    baths = st.number_input(
        "Bathrooms",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    has_garage_str = st.selectbox(
        "Garage Available",
        options=["Yes", "No"],
        index=0,
    )
    has_garage = has_garage_str == "Yes"

    has_pool_str = st.selectbox(
        "Swimming Pool",
        options=["No", "Yes"],
        index=0,
        help="Select 'Yes' if the property has a swimming pool.",
    )
    has_pool = has_pool_str == "Yes"

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
            beds=beds,
            baths=baths,
            has_garage=has_garage,
            has_pool=has_pool,
            zip_code=zip_code,
            city=city,
            county=county,
            lot_size_sqft=DEFAULT_LOT_SIZE_SQFT,
        )

        st.markdown("### 💰 Estimated Market Price")
        st.markdown(f"## **${predicted_price:,.0f}**")

    except Exception as exc:
        st.error(f"Prediction error: {exc}")
