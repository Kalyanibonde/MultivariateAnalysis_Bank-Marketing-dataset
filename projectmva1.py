import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from sklearn.cluster import KMeans
import joblib
from fpdf import FPDF
import os

# Streamlit page config
st.set_page_config(layout="wide", page_title="Bank Marketing Analysis")

# Load the dataset
data = pd.read_csv(r"C:\Users\kalya\Downloads\mva\mva\bank.csv", sep=';')

# Title
st.title("📊 Bank Marketing Campaign Analysis")

# Sidebar
st.sidebar.header("🔍 User Input Filters")
if st.sidebar.checkbox("Show raw data"):
    st.write(data.head())

# Basic EDA
st.header("📌 Dataset Overview")
st.write("### Dataset Information")
st.text(data.info())
st.write("### Summary Statistics")
st.write(data.describe())

# Unique values in 'y' column
st.write("### Unique values in 'y' column")
st.write(data['y'].unique())

# Interactive Filters
st.sidebar.subheader("🎛️ Filter Data")
selected_job = st.sidebar.multiselect("Select Job Type", data['job'].unique())
selected_marital = st.sidebar.multiselect("Select Marital Status", data['marital'].unique())
selected_education = st.sidebar.multiselect("Select Education Level", data['education'].unique())
selected_housing = st.sidebar.selectbox("Has Housing Loan", ['All'] + data['housing'].unique().tolist())
selected_loan = st.sidebar.selectbox("Has Personal Loan", ['All'] + data['loan'].unique().tolist())

filtered_data = data.copy()
if selected_job:
    filtered_data = filtered_data[filtered_data['job'].isin(selected_job)]
if selected_marital:
    filtered_data = filtered_data[filtered_data['marital'].isin(selected_marital)]
if selected_education:
    filtered_data = filtered_data[filtered_data['education'].isin(selected_education)]
if selected_housing != 'All':
    filtered_data = filtered_data[filtered_data['housing'] == selected_housing]
if selected_loan != 'All':
    filtered_data = filtered_data[filtered_data['loan'] == selected_loan]

st.write("### Filtered Data")
st.write(filtered_data.head())

# Layout for visualization
col1, col2 = st.columns(2)

# Target variable distribution
with col1:
    st.subheader("📊 Subscription Status Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='y', data=filtered_data, ax=ax)
    ax.set_title('Subscription Status')
    ax.set_xlabel('Subscribed')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Job Distribution
with col2:
    st.subheader("📊 Job Type Distribution by Subscription")
    fig, ax = plt.subplots()
    sns.countplot(x='job', data=filtered_data, hue='y', ax=ax)
    ax.set_title('Job Types by Subscription')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

# Feature Correlation Analysis
st.header("📊 Feature Correlation Analysis")
labelencoder = LabelEncoder()
data_encoded = filtered_data.copy()
for col in data_encoded.select_dtypes(include=['object']).columns:
    data_encoded[col] = labelencoder.fit_transform(data_encoded[col])

corr_matrix = data_encoded.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Machine Learning Model
st.header("🤖 Subscription Prediction Model")
X = data_encoded.drop(columns=['y'])
y = data_encoded['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Model Performance")
st.write("✅ Accuracy:", accuracy_score(y_test, y_pred))
st.write("✅ Classification Report:")
st.text(classification_report(y_test, y_pred))

# Customer Segmentation
st.header("📊 Customer Segmentation")
st.subheader("Cluster Analysis (K-Means)")
kmeans = KMeans(n_clusters=3, random_state=42)
data_encoded['Cluster'] = kmeans.fit_predict(X)

st.write(data_encoded[['Cluster', 'y']].value_counts())

fig = px.scatter_3d(data_encoded, x='age', y='balance', z='duration', color='Cluster', title='Customer Segmentation')
st.plotly_chart(fig)

# Function to generate PDF report
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Bank Marketing Campaign Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, "Dataset Summary", ln=True, align='L')
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    
    total_customers = len(data)
    subscribed_customers = len(data[data['y'] == 'yes'])
    non_subscribed_customers = len(data[data['y'] == 'no'])
    
    pdf.cell(200, 10, f"Total Customers: {total_customers}", ln=True)
    pdf.cell(200, 10, f"Subscribed Customers: {subscribed_customers}", ln=True)
    pdf.cell(200, 10, f"Non-Subscribed Customers: {non_subscribed_customers}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Key Insights:", ln=True, align='L')
    pdf.set_font("Arial", size=10)
    
    pdf.cell(200, 10, "- Most subscriptions came from people with higher education.", ln=True)
    pdf.cell(200, 10, "- Job type and balance significantly impact subscription rates.", ln=True)
    pdf.cell(200, 10, "- Previous successful contacts improve chances of subscription.", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Feature Correlation (Top Influencers)", ln=True, align='L')
    pdf.set_font("Arial", size=10)
    
    corr_matrix = data_encoded.corr()
    top_corr_features = corr_matrix['y'].sort_values(ascending=False)[1:6]
    for feature, correlation in top_corr_features.items():
        pdf.cell(200, 10, f"- {feature}: {correlation:.2f}", ln=True)
    
    report_path = "bank_report.pdf"
    pdf.output(report_path)
    return report_path

# Download Options
st.sidebar.header("📥 Download Filtered Data")
st.sidebar.download_button(
    label="Download CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='filtered_bank_data.csv',
    mime='text/csv'
)

st.sidebar.header("📄 Download Report")
if st.sidebar.button("Generate PDF Report"):
    report_file = generate_pdf_report(filtered_data)
    with open(report_file, "rb") as file:
        st.sidebar.download_button(
            label="Download PDF Report",
            data=file,
            file_name="bank_marketing_report.pdf",
            mime="application/pdf"
        )
    os.remove(report_file)

# Subscribed Customers
st.header("✅ Subscribed Customers Data")
yes_customers = filtered_data[filtered_data['y'] == 'yes']
st.write(yes_customers.head())
