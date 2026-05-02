# =====================================
# 🏠 House Price Prediction Web App
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# =====================================
# 🎯 Page Config
# =====================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# 🎨 HEADER
# =====================================

st.markdown("""
# 🏠 House Price Prediction App
### 📊 Predict house prices using Machine Learning (Linear Regression)
---
""")

# =====================================
# 📂 LOAD DATA
# =====================================

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'SalePrice']]
    df = df.dropna()
    return df

data = load_data()

# =====================================
# 📊 DATA + HEATMAP (SIDE BY SIDE)
# =====================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

with col2:
    st.subheader("📈 Correlation Heatmap")
    fig_hm, ax_hm = plt.subplots(figsize=(5,4))
    cax = ax_hm.matshow(data.corr())
    fig_hm.colorbar(cax)
    ax_hm.set_xticks(range(len(data.columns)))
    ax_hm.set_xticklabels(data.columns, rotation=45)
    ax_hm.set_yticks(range(len(data.columns)))
    ax_hm.set_yticklabels(data.columns)

    st.pyplot(fig_hm)

# =====================================
# 🧠 MODEL TRAINING
# =====================================

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# =====================================
# 📊 METRICS
# =====================================

st.subheader("📊 Model Performance")

m1, m2 = st.columns(2)

score = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, predictions)

m1.metric("R² Score", f"{score:.2f}")
m2.metric("Mean Absolute Error", f"₹ {mae:,.0f}")

# =====================================
# 📉 SCATTER GRAPH (FIXED SIZE + CENTERED)
# =====================================

st.subheader("📉 Actual vs Predicted Prices")

colA, colB, colC = st.columns([1,2,1])

with colB:
    fig2, ax2 = plt.subplots(figsize=(5,4))  # 👈 controlled size

    ax2.scatter(y_test, predictions)

    # Perfect prediction line
    ax2.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
    )

    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Predicted Price")
    ax2.set_title("Prediction Accuracy")

    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig2)

# =====================================
# 📌 FEATURE IMPORTANCE
# =====================================

st.subheader("📌 Feature Impact")

for feature, coef in zip(X.columns, model.coef_):
    st.write(f"**{feature}** → {coef:.2f}")

# =====================================
# 🎛️ SIDEBAR INPUT
# =====================================

st.sidebar.header("⚙️ Enter House Details")

area = st.sidebar.slider("Area (sq ft)", 300, 5000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
garage = st.sidebar.slider("Garage Capacity", 0, 4, 1)

# =====================================
# 🔮 PREDICTION
# =====================================

st.subheader("🔮 Predict House Price")

if st.button("Predict Price"):

    input_data = np.array([[area, bedrooms, bathrooms, garage]])
    prediction = model.predict(input_data)

    st.markdown(f"""
    ### 💰 Estimated Price
    # ₹ {prediction[0]:,.0f}
    """)

# =====================================
# 🧾 FOOTER
# =====================================

st.markdown("""
---
Made by **Priyanka R** | ML Internship Project 🚀
""")