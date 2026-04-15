# ==========================================
# 1. Imports
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ========================================== 
# 2. Load Model Components
# ==========================================
model = joblib.load("automl_pro_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# ========================================== 
# 3. App Title
# ========================================== 
st.title("🐟 Fishing Catch Prediction Dashboard (PRO)")
st.write("Predict QTYTUBS and analyze fishing conditions")

# ========================================== 
# 4. File Upload
# ========================================== 
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # ========================================== 
    # 5. Preprocessing (same as training)
    # ========================================== 
    df['datetime_full'] = pd.to_datetime(df['datetime_full'], errors='coerce')

    df['hour'] = df['datetime_full'].dt.hour
    df['dayofweek'] = df['datetime_full'].dt.dayofweek
    df['month'] = df['datetime_full'].dt.month

    df['temp_x_salinity'] = df['sea_surface_temperature_K'] * df['Sea_Surface_Salinity']
    df['tide_energy'] = df['Tidal_Height'] * df['tidal_velocity_speed']

    df = df.fillna(df.median(numeric_only=True))

    # ========================================== 
    # 6. Prediction
    # ========================================== 
    if st.button("🔮 Predict QTYTUBS"):
        X = df.select_dtypes(include=[np.number])
        X = X.drop(columns=['QTYTUBS'], errors='ignore')

        # Feature selection + scaling
        X_selected = selector.transform(X)
        X_scaled = scaler.transform(X_selected)

        predictions = model.predict(X_scaled)

        df['Predicted_QTYTUBS'] = predictions

        st.subheader("📈 Predictions")
        st.write(df[['Predicted_QTYTUBS']].head())

        # ========================================== 
        # 7. Visualization: Actual vs Predicted
        # ========================================== 
        if 'QTYTUBS' in df.columns:
            plt.figure()
            plt.scatter(df['QTYTUBS'], df['Predicted_QTYTUBS'])
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            st.pyplot(plt)

        # ========================================== 
        # 8. Time Trends
# ========================================== 
        st.subheader("⏱️ Time Trends")
        if 'datetime_full' in df.columns:
            df_sorted = df.sort_values('datetime_full')

            plt.figure()
            plt.plot(df_sorted['datetime_full'], df_sorted['Predicted_QTYTUBS'])
            plt.title("Predicted Catch Over Time")
            st.pyplot(plt)

        # ========================================== 
        # 9. Geospatial Hotspots
# ========================================== 
        st.subheader("🌍 Fishing Hotspots")

        if 'coor_lat' in df.columns and 'coor_long' in df.columns:
            plt.figure()
            plt.scatter(
                df['coor_long'],
                df['coor_lat'],
                s=df['Predicted_QTYTUBS'],
            )
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Predicted Fishing Hotspots")
            st.pyplot(plt)

        # ========================================== 
        # 10. Download Results
# ========================================== 
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
