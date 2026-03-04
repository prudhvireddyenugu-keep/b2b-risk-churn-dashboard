import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(page_title="Client Retention Dashboard", layout="wide")

# Title
st.title("📊 B2B Client Retention Insights")

# Upload file
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.success("File uploaded successfully!")

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(df)

    # KPI Section
    st.subheader("📌 Key Metrics")

    total_clients = len(df)
    churned_clients = df[df['Churn'] == 1].shape[0]
    active_clients = df[df['Churn'] == 0].shape[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Clients", total_clients)
    col2.metric("Active Clients", active_clients)
    col3.metric("Churned Clients", churned_clients)

    # Churn Rate
    churn_rate = (churned_clients / total_clients) * 100
    st.write(f"### 🔴 Churn Rate: {churn_rate:.2f}%")

    # Filter option (NEW FEATURE)
    st.sidebar.header("🔍 Filter Data")
    industry_filter = st.sidebar.selectbox("Select Industry", df['Industry'].unique())

    filtered_df = df[df['Industry'] == industry_filter]

    st.subheader("📊 Filtered Data Preview")
    st.write(filtered_df.head())

    # Bar chart
    st.subheader("📈 Churn Distribution by Industry")
    churn_by_industry = df.groupby('Industry')['Churn'].mean()
    st.bar_chart(churn_by_industry)

    # Pie chart alternative (DIFFERENT LOOK)
    st.subheader("🥧 Client Status Breakdown")

    status_counts = df['Churn'].value_counts()
    status_df = pd.DataFrame({
        'Status': ['Active', 'Churned'],
        'Count': [status_counts[0], status_counts[1]]
    })

    st.write(status_df)

else:
    st.warning("Please upload a CSV file to proceed.")
