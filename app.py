import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"
HIST = pd.read_csv(DATA_DIR / "historical_food.csv")
NGOS = pd.read_csv(DATA_DIR / "ngos.csv")

st.set_page_config(page_title="FoodBridge MVP", page_icon="🍱", layout="wide")

# ---------- Helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def predict_demand(donor, food_type, prepared_qty, target_day):
    df = HIST[(HIST.donor == donor) & (HIST.food_type == food_type)].copy()
    if df.empty:
        df = HIST[HIST.food_type == food_type].copy()

    if df.empty:
        return max(0, round(prepared_qty * 0.75))

    # Prototype model: XGBoost when available; otherwise a transparent rolling-mean fallback.
    df["day_num"] = pd.to_datetime(df["date"]).dt.dayofweek
    target_day_num = pd.Timestamp(target_day).dayofweek

    X = pd.DataFrame({
        "day_num": df["day_num"],
        "week_num": pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int),
        "trend": np.arange(len(df))
    })
    y = df["consumed_qty"]

    if XGB_AVAILABLE and len(df) >= 8:
        model = XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            objective="reg:squarederror", random_state=42, verbosity=0
        )
        model.fit(X, y)
        future = pd.DataFrame({
            "day_num": [target_day_num],
            "week_num": [pd.Timestamp(target_day).isocalendar().week],
            "trend": [len(df) + 1]
        })
        pred = float(model.predict(future)[0])
    else:
        same_day = df[df["day_num"] == target_day_num]
        pred = float(same_day["consumed_qty"].mean() if not same_day.empty else df["consumed_qty"].tail(5).mean())

    # Keep prototype output realistic and non-negative.
    return max(0, int(round(pred)))

def route_estimate(donor_lat, donor_lon, ngo_lat, ngo_lon):
    straight = haversine_km(donor_lat, donor_lon, ngo_lat, ngo_lon)
    # Approximate road factor for MVP. OSRM can replace this when internet is available.
    road_km = straight * 1.25
    minutes = road_km / 25 * 60
    return round(road_km, 2), round(minutes)

# ---------- Sidebar ----------
st.sidebar.title("🍱 FoodBridge")
st.sidebar.caption("AI-powered food surplus & smart redistribution — MVP")
page = st.sidebar.radio("Navigate", ["Donor Dashboard", "NGO Dashboard", "Analytics", "About"])


# ---------- Donor ----------
if page == "Donor Dashboard":
    st.title("🍱 Donor Dashboard")
    st.write("Predict demand, estimate surplus, and find suitable nearby NGOs.")

    c1, c2, c3 = st.columns(3)
    with c1:
        donor = st.selectbox("Food donor / restaurant", sorted(HIST.donor.unique()))
    with c2:
        food_type = st.selectbox("Food type", sorted(HIST[HIST.donor == donor].food_type.unique()))
    with c3:
        prepared_qty = st.number_input("Prepared food (servings)", min_value=0, value=150, step=5)

    target_date = st.date_input("Prediction date", value=date.today())
    donor_lat = st.number_input("Donor latitude", value=12.9710, format="%.4f")
    donor_lon = st.number_input("Donor longitude", value=77.5930, format="%.4f")

    if st.button("Predict demand & find NGOs", type="primary", use_container_width=True):
        demand = predict_demand(donor, food_type, prepared_qty, target_date)
        surplus = max(0, prepared_qty - demand)
        donation_qty = int(round(surplus * 0.9))

        st.session_state["result"] = {
            "donor": donor, "food_type": food_type, "prepared_qty": prepared_qty,
            "demand": demand, "surplus": surplus, "donation_qty": donation_qty,
            "lat": donor_lat, "lon": donor_lon
        }

    result = st.session_state.get("result")
    if result:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prepared", f'{result["prepared_qty"]} servings')
        m2.metric("Predicted demand", f'{result["demand"]} servings')
        m3.metric("Potential surplus", f'{result["surplus"]} servings')
        m4.metric("Recommended donation", f'{result["donation_qty"]} servings')

        if result["surplus"] <= 0:
            st.warning("No potential surplus is predicted for this scenario.")
        else:
            st.success(f'Potential surplus detected: {result["surplus"]} servings of {result["food_type"]}.')

            matches = NGOS[NGOS.food_type == result["food_type"]].copy()
            if matches.empty:
                st.info("No matching NGO requirement is available in the demo dataset.")
            else:
                matches["distance_km"], matches["travel_min"] = zip(*matches.apply(
                    lambda r: route_estimate(result["lat"], result["lon"], r.lat, r.lon), axis=1
                ))
                matches["match_score"] = (
                    (matches["required_qty"].clip(upper=result["donation_qty"]) / max(result["donation_qty"], 1)) * 60
                    + (1 / (1 + matches["distance_km"])) * 40
                )
                matches = matches.sort_values("match_score", ascending=False)

                st.subheader("🤝 Recommended NGO matches")
                display = matches[["ngo","required_qty","distance_km","travel_min","match_score"]].copy()
                display.columns = ["NGO","Required","Road distance (km)","Est. travel (min)","Match score"]
                st.dataframe(display, use_container_width=True, hide_index=True)

                best = matches.iloc[0]
                st.info(
                    f'Best match: **{best["ngo"]}** — needs {int(best["required_qty"])} servings, '
                    f'about {best["distance_km"]:.1f} km away ({int(best["travel_min"])} min estimated travel).'
                )

# ---------- NGO ----------
elif page == "NGO Dashboard":
    st.title("🏢 NGO Dashboard")
    st.write("Register/view requirements and discover available matching surplus.")

    ngo_name = st.selectbox("NGO", NGOS.ngo)
    ngo = NGOS[NGOS.ngo == ngo_name].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Food needed", f'{int(ngo.required_qty)} servings')
    b.metric("Food type", ngo.food_type)
    c.metric("Location", f'{ngo.lat:.3f}, {ngo.lon:.3f}')

    st.subheader("Available donor opportunities")
    opportunities = []
    for donor in HIST.donor.unique():
        food_rows = HIST[HIST.donor == donor]
        if ngo.food_type in food_rows.food_type.unique():
            pred = predict_demand(donor, ngo.food_type, 150, date.today())
            surplus = max(0, 150 - pred)
            if surplus > 0:
                opportunities.append([donor, ngo.food_type, pred, surplus])

    if opportunities:
        odf = pd.DataFrame(opportunities, columns=["Donor","Food type","Predicted demand","Potential surplus"])
        st.dataframe(odf, use_container_width=True, hide_index=True)
    else:
        st.info("No matching surplus opportunities found in the demo data.")

# ---------- Analytics ----------
elif page == "Analytics":
    st.title("📊 FoodBridge Analytics")
    st.write("Prototype EDA from the preloaded historical dataset.")

    x1, x2 = st.columns(2)
    with x1:
        st.subheader("Average consumption by food type")
        avg = HIST.groupby("food_type", as_index=False).consumed_qty.mean()
        st.bar_chart(avg.set_index("food_type"))
    with x2:
        st.subheader("Consumption by donor")
        avg2 = HIST.groupby("donor", as_index=False).consumed_qty.mean()
        st.bar_chart(avg2.set_index("donor"))

    st.subheader("Historical records")
    st.dataframe(HIST, use_container_width=True, hide_index=True)

# ---------- About ----------
else:
    st.title("About FoodBridge MVP")
    st.markdown("""
    **FoodBridge** is a prototype of an AI-powered food waste prediction and smart redistribution platform.

    **MVP flow**
    1. Donor enters prepared food and location.
    2. Prototype predicts expected demand from preloaded historical data.
    3. Surplus is estimated as prepared food minus predicted demand.
    4. Matching ranks NGOs by food type, required quantity, and approximate road distance.
    5. Dashboards expose donor, NGO, and analytics views.

    **Implemented in this MVP**
    - Streamlit web interface
    - Preloaded historical dataset
    - XGBoost demand prediction when installed
    - Surplus estimation
    - NGO matching
    - Distance/travel-time approximation
    - Basic EDA dashboard

    **Next production steps**
    - MySQL persistence and authentication
    - Real OSRM routing API
    - Proper train/validation/test pipeline
    - More historical data and model evaluation
    - Donation status workflow and notifications
    - Production deployment
    """)

st.sidebar.caption(f"Model: {'XGBoost' if XGB_AVAILABLE else 'rolling-mean fallback'}")
