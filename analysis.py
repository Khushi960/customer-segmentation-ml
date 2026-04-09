import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.title("📊 Data Analysis")

df = pd.read_csv("store_customers.csv", sep="\t")


df.columns = df.columns.str.strip()


if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)

# Fix Gender
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Convert numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')

df = df.dropna()

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df)



# Age Distribution
st.subheader("Age Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(df['Age'])
st.pyplot(fig1)

# Income Distribution
st.subheader("Income Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df['Annual Income (k$)'])
st.pyplot(fig2)

# Spending Score
st.subheader("Spending Score Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(df['Spending Score (1-100)'], kde=True, ax=ax3)
st.pyplot(fig3)

# Gender Count
st.subheader("Gender Distribution")
fig4, ax4 = plt.subplots()
sns.countplot(x='Gender', data=df, ax=ax4)
st.pyplot(fig4)

# Pie Chart 
st.subheader("Gender Pie Chart")
fig5, ax5 = plt.subplots()

gender_counts = df['Gender'].value_counts()
labels = gender_counts.index.map({0: 'Male', 1: 'Female'})

ax5.pie(gender_counts, labels=labels, autopct='%1.1f%%')
st.pyplot(fig5)

# Scatter Plot
st.subheader("Income vs Spending Score")
fig6, ax6 = plt.subplots()
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    data=df,
    ax=ax6
)
st.pyplot(fig6)

# Boxplot
st.subheader("Outlier Detection")
fig7, ax7 = plt.subplots()
sns.boxplot(data=df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], ax=ax7)
st.pyplot(fig7)