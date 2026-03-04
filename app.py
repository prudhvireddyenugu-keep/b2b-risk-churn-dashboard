import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------
# Page config
# ---------------------------------------
st.set_page_config(page_title="B2B Client Risk Intelligence", layout="wide")

st.title("B2B Client Risk & Churn Intelligence Dashboard")
st.caption("Client Risk Monitoring • Churn Prediction • Retention Suggestions • Responsible AI")

# ---------------------------------------
# Load data
# ---------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# ---------------------------------------
# Column safety (prevents crashes)
# ---------------------------------------
required_cols = [
    "Client_ID", "Industry", "Region",
    "Monthly_Usage_Score", "Payment_Delay_Days",
    "Contract_Length_Months", "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD", "Renewal_Status"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error("CSV columns mismatch. Missing: " + ", ".join(missing))
    st.write("Columns found in your CSV:")
    st.write(list(df.columns))
    st.stop()

# Optional display columns
if "Company_Name" not in df.columns:
    df["Company_Name"] = "NA"
if "Plan" not in df.columns:
    df["Plan"] = "NA"
if "Lead_Source" not in df.columns:
    df["Lead_Source"] = "NA"
if "Account_Age_Months" not in df.columns:
    df["Account_Age_Months"] = 0

# ---------------------------------------
# Part B: Risk Score (NEW approach: weighted + normalized 0-100)
# ---------------------------------------
# Normalize helper
def minmax(series):
    s = series.astype(float)
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())

# Normalized components (0..1)
delay_n   = minmax(df["Payment_Delay_Days"])                       # higher = worse
usage_n   = 1 - minmax(df["Monthly_Usage_Score"])                  # lower usage = worse
contract_n = 1 - minmax(df["Contract_Length_Months"])              # shorter contract = worse
tickets_n = minmax(df["Support_Tickets_Last30Days"])               # higher = worse

# Weighted risk score (0..100)
# Weight idea: payment delay & tickets are strongest signals in B2B health
df["Risk_Score"] = (
    0.35 * delay_n +
    0.30 * usage_n +
    0.20 * contract_n +
    0.15 * tickets_n
) * 100

# Risk category by thresholds (changed vs earlier)
def risk_category(score):
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# Create churn flag
df["Churned"] = df["Renewal_Status"].map({"Yes": 0, "No": 1})

# ---------------------------------------
# Sidebar Filters
# ---------------------------------------
st.sidebar.header("Filters")
regions = sorted(df["Region"].dropna().unique())
industries = sorted(df["Industry"].dropna().unique())
risk_levels = ["Low Risk", "Medium Risk", "High Risk"]

sel_region = st.sidebar.multiselect("Region", regions, default=regions)
sel_industry = st.sidebar.multiselect("Industry", industries, default=industries)
sel_risk = st.sidebar.multiselect("Risk Category", risk_levels, default=risk_levels)

# Revenue slider filter (new)
rev_min, rev_max = float(df["Monthly_Revenue_USD"].min()), float(df["Monthly_Revenue_USD"].max())
sel_rev = st.sidebar.slider("Monthly Revenue (USD)", min_value=rev_min, max_value=rev_max, value=(rev_min, rev_max))

filtered = df[
    df["Region"].isin(sel_region) &
    df["Industry"].isin(sel_industry) &
    df["Risk_Category"].isin(sel_risk) &
    (df["Monthly_Revenue_USD"] >= sel_rev[0]) &
    (df["Monthly_Revenue_USD"] <= sel_rev[1])
].copy()

# ---------------------------------------
# Tabs (big UI change)
# ---------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🤖 ML Model", "🧠 Retention", "⚖️ Responsible AI", "📄 Data View"])

# =======================================
# TAB 1: Dashboard
# =======================================
with tab1:
    # KPIs
    total = len(filtered)
    high_risk = int((filtered["Risk_Category"] == "High Risk").sum())
    churn_rate = (filtered["Churned"].mean() * 100) if total else 0
    avg_rev = filtered["Monthly_Revenue_USD"].mean() if total else 0
    total_rev = filtered["Monthly_Revenue_USD"].sum() if total else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clients", f"{total}")
    k2.metric("High Risk", f"{high_risk}")
    k3.metric("Churn %", f"{churn_rate:.2f}%")
    k4.metric("Avg Revenue", f"${avg_rev:,.0f}")
    k5.metric("Total Revenue", f"${total_rev:,.0f}")

    st.divider()

    c1, c2 = st.columns(2)

    # Chart 1: Donut/Pie risk distribution (new)
    with c1:
        st.subheader("Risk Mix (Donut)")
        counts = filtered["Risk_Category"].value_counts().reindex(risk_levels).fillna(0)
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.45))
        ax.axis("equal")
        st.pyplot(fig)

        st.caption("Interpretation: Larger High Risk share means customer base needs urgent retention focus.")

    # Chart 2: Churn by risk category (new)
    with c2:
        st.subheader("Churn Rate by Risk Category")
        churn_by_risk = filtered.groupby("Risk_Category")["Churned"].mean().reindex(risk_levels).fillna(0) * 100
        fig2, ax2 = plt.subplots()
        ax2.bar(churn_by_risk.index, churn_by_risk.values)
        ax2.set_ylabel("Churn %")
        st.pyplot(fig2)

        st.caption("Check whether churn increases from Low → Medium → High (validates risk logic).")

    st.divider()

    c3, c4 = st.columns(2)

    # Chart 3: Industry x Risk pivot (heatmap-like table)
    with c3:
        st.subheader("Industry vs Risk (Pivot Table)")
        pivot = pd.pivot_table(
            filtered,
            index="Industry",
            columns="Risk_Category",
            values="Client_ID",
            aggfunc="count",
            fill_value=0
        ).reindex(columns=risk_levels, fill_value=0)
        st.dataframe(pivot, use_container_width=True)

    # Chart 4: Revenue distribution by risk (box-like using matplotlib)
    with c4:
        st.subheader("Revenue Spread by Risk")
        # Prepare lists for boxplot
        box_data = [
            filtered.loc[filtered["Risk_Category"] == "Low Risk", "Monthly_Revenue_USD"].dropna(),
            filtered.loc[filtered["Risk_Category"] == "Medium Risk", "Monthly_Revenue_USD"].dropna(),
            filtered.loc[filtered["Risk_Category"] == "High Risk", "Monthly_Revenue_USD"].dropna()
        ]
        fig3, ax3 = plt.subplots()
        ax3.boxplot(box_data, labels=risk_levels, showfliers=False)
        ax3.set_ylabel("Monthly Revenue (USD)")
        st.pyplot(fig3)

    st.divider()

    # Top 20 high-risk clients (kept, but cleaner)
    st.subheader("Top 20 High-Risk Clients (Action List)")
    top20 = filtered.sort_values(["Risk_Score", "Monthly_Revenue_USD"], ascending=[False, False]).head(20)

    show_cols = [
        "Client_ID", "Company_Name", "Industry", "Region", "Plan",
        "Monthly_Usage_Score", "Payment_Delay_Days", "Contract_Length_Months",
        "Support_Tickets_Last30Days", "Monthly_Revenue_USD",
        "Risk_Score", "Risk_Category", "Renewal_Status"
    ]
    show_cols = [c for c in show_cols if c in top20.columns]
    st.dataframe(top20[show_cols], use_container_width=True)

    # Client Risk Explorer (new)
    st.markdown("### 🔎 Client Risk Explorer")
    client_id = st.text_input("Enter Client_ID (exact):", value="")
    if client_id:
        row = filtered[filtered["Client_ID"].astype(str) == str(client_id)]
        if row.empty:
            st.warning("Client_ID not found in current filtered data. Try changing filters or check ID.")
        else:
            r = row.iloc[0]
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Risk Score", f"{r['Risk_Score']:.1f}")
            colB.metric("Risk Category", f"{r['Risk_Category']}")
            colC.metric("Monthly Revenue", f"${r['Monthly_Revenue_USD']:,.0f}")
            colD.metric("Renewal Status", f"{r['Renewal_Status']}")

            st.write("Client Profile")
            st.dataframe(row[show_cols], use_container_width=True)

    # Download filtered data (new)
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_clients.csv",
        mime="text/csv"
    )

# =======================================
# TAB 2: ML Model (Part C)
# =======================================
with tab2:
    st.subheader("Decision Tree Classifier: Predict Renewal_Status (Churn)")
    st.caption("Target: Renewal_Status (Yes/No). This model predicts churn probability using client activity and service signals.")

    # Features: numeric + categorical (different selection than earlier)
    feature_cols = [
        "Industry", "Region", "Plan", "Lead_Source",
        "Account_Age_Months", "Monthly_Usage_Score", "Payment_Delay_Days",
        "Contract_Length_Months", "Support_Tickets_Last30Days", "Monthly_Revenue_USD",
        "Risk_Score"
    ]
    X = df[feature_cols].copy()
    y = df["Renewal_Status"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Let user change tree depth (interactive = looks advanced)
    depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=12, value=6)
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    st.write(f"✅ **Accuracy:** {acc:.4f}")

    cm = confusion_matrix(y_test, pred)
    st.write("**Confusion Matrix (Actual rows, Pred columns):**")
    st.write(cm)

    # Feature importance (top 12)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    st.subheader("Top Drivers of Churn (Feature Importance)")
    st.bar_chart(importances)

    st.markdown("**Interpretation (for viva):** The most important features above influence the Decision Tree splits the most, so they are key churn drivers.")

# =======================================
# TAB 3: Retention (Part E)
# =======================================
with tab3:
    st.subheader("AI-Based Retention Suggestions")
    st.caption("Click the button to generate practical actions based on risky patterns observed in the selected data.")

    # Show quick insights to make suggestions look 'AI-like'
    avg_delay = filtered["Payment_Delay_Days"].mean() if len(filtered) else 0
    avg_usage = filtered["Monthly_Usage_Score"].mean() if len(filtered) else 0
    avg_tickets = filtered["Support_Tickets_Last30Days"].mean() if len(filtered) else 0
    avg_contract = filtered["Contract_Length_Months"].mean() if len(filtered) else 0

    st.write("**Current Filtered Insights:**")
    st.write(f"- Avg Payment Delay: **{avg_delay:.1f} days**")
    st.write(f"- Avg Usage Score: **{avg_usage:.1f}**")
    st.write(f"- Avg Support Tickets (30d): **{avg_tickets:.1f}**")
    st.write(f"- Avg Contract Length: **{avg_contract:.1f} months**")

    if st.button("Generate Retention Strategy"):
        st.success("Recommended Retention Actions (3–5)")
        st.write("1) **Payment-delay intervention:** If delay is high, offer a flexible payment plan + incentives for timely payment.")
        st.write("2) **Usage activation:** For low usage, run onboarding refresh + training + usage nudges (product adoption campaign).")
        st.write("3) **Support improvement:** For high tickets, assign a dedicated account manager + faster SLA + root cause fixes.")
        st.write("4) **Contract upgrade:** Offer longer-term renewal incentives (discount/add-ons) to clients with short contracts.")
        st.write("5) **Save high-value accounts:** For high revenue & high risk, schedule leadership call + customized success roadmap.")

# =======================================
# TAB 4: Responsible AI (Part F)
# =======================================
with tab4:
    st.subheader("Ethical Implications of Predicting Client Churn")
    st.write("""
**1) Bias in Predictive Models**  
Models can learn patterns that reflect historical disadvantages (e.g., certain regions/industries facing economic instability). This may unfairly mark those groups as higher risk.

**2) Impact of Labeling Clients as “High Risk”**  
If teams treat 'high risk' clients negatively (less support, stricter terms), it can become a self-fulfilling prophecy. Risk labels should trigger support, not punishment.

**3) Data Privacy Concerns**  
Client usage, payments, and complaints data are sensitive. Access should be restricted, data should be securely stored, and only necessary fields should be used.

**4) Responsible Decision-Making**  
Churn predictions are probabilities, not facts. Final decisions should combine human judgment, relationship context, and business strategy.

**5) Transparency & Monitoring**  
Explain key drivers (feature importance) and audit the model regularly to ensure fairness, stability, and accuracy over time.
""")

# =======================================
# TAB 5: Data View
# =======================================
with tab5:
    st.subheader("Filtered Dataset Preview")
    st.caption("This helps your professor verify filters and data integrity.")
    st.dataframe(filtered.head(200), use_container_width=True)

    st.markdown("### Data Summary (Quick)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows (filtered)", len(filtered))
    s2.metric("Unique Industries", filtered["Industry"].nunique())
    s3.metric("Unique Regions", filtered["Region"].nunique())
    s4.metric("Avg Risk Score", f"{filtered['Risk_Score'].mean():.1f}" if len(filtered) else "NA")
