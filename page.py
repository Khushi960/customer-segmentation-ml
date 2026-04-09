import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Analyze customer behavior using Machine Learning")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("store_customers.csv", sep="\t")

    df.columns = df.columns.str.strip()

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
    df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')

    df = df.dropna()

    return df

df = load_data()

# KMeans
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

#  SIDEBAR 
st.sidebar.header("🧾 Customer Input")

age = st.sidebar.slider("Age", 18, 70)
income = st.sidebar.slider("Annual Income", 10, 150)
score = st.sidebar.slider("Spending Score", 1, 100)

#   (Income)
st.sidebar.markdown("### 🔍 Filter Dataset")
income_range = st.sidebar.slider("Select Income Range", 10, 150, (10, 150))

filtered_df = df[
    (df['Annual Income (k$)'] >= income_range[0]) &
    (df['Annual Income (k$)'] <= income_range[1])
]

# Labels
labels = {
    0: "💎 High Value Customer",
    1: "🛒 Budget Shopper",
    2: "🎯 Target Customer",
    3: "💰 Big Spender",
    4: "🔁 Regular Customer"
}

# Prediction
predict = st.sidebar.button("Predict Segment")

#  LAYOUT 
col1, col2 = st.columns(2)

# 📊 DATA TABLE
with col1:
    st.subheader("📋 Dataset Preview")
    rows = st.slider("Select rows", 5, len(filtered_df), 10)
    st.dataframe(filtered_df.head(rows))

# 📈 MAIN GRAPH
with col2:
    st.subheader("📈 Customer Segmentation")

    fig, ax = plt.subplots()

    ax.scatter(
        filtered_df['Annual Income (k$)'],
        filtered_df['Spending Score (1-100)'],
        c=filtered_df['Cluster'],
        alpha=0.6
    )

    if predict:
        new_customer = [[age, income, score]]
        cluster = kmeans.predict(new_customer)[0]

        ax.scatter(income, score, color='red', s=200, label="Your Input")
        st.sidebar.success(labels[cluster])
        ax.legend()

    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Segmentation")

    st.pyplot(fig)


st.markdown("## 📊 Additional Insights")

col3, col4 = st.columns(2)

# Age Distribution
with col3:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df['Age'])
    st.pyplot(fig)

# Spending Score Distribution
with col4:
    st.subheader("Spending Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df['Spending Score (1-100)'])
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("✨ Mini Project by Khushi Shirbhate 🦄")