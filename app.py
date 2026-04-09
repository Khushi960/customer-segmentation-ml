import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")

st.write("Welcome! Use the sidebar to navigate between pages.")

st.info("👉 Go to 'page' or 'analysis' from sidebar")

df = pd.read_csv("store_customers.csv", sep="\t")

print(df.head())
print(df.columns)
# Preview
print(df.head())

# Check missing values
print(df.isnull().sum())

# Drop CustomerID
df = df.drop("CustomerID", axis=1)

# Convert Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Check duplicates
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Final check
print(df.info())
print(df.head())

