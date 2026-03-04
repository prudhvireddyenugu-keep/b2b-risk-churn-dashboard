import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="B2B Risk & Churn Dashboard", layout="wide")
st.title("B2B Client Risk & Churn Intelligence Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# ---- Risk Score (Business Logic) ----
def risk_score(r):
    s = 0
    if r["Payment_Delay_Days"] > 30: s += 3
    elif r["Payment_Delay_Days"] > 10: s += 2
    elif r["Payment_Delay_Days"] > 0: s += 1

    if r["Monthly_Usage_Score"] < 40: s += 3
    elif r["Monthly_Usage_Score"] < 60: s += 2
    elif r["Monthly_Usage_Score"] < 75: s += 1

    if r["Contract_Length_Months"] < 6: s += 3
    elif r["Contract_Length_Months"] < 12: s += 2
    elif r["Contract_Length_Months"] < 18: s += 1

    if r["Support_Tickets_Last30Days"] > 6: s += 3
    elif r["Support_Tickets_Last30Days"] > 3: s += 2
    elif r["Support_Tickets_Last30Days"] > 0: s += 1
    return s

df["Risk_Score"] = df.apply(risk_score, axis=1)

def risk_cat(x):
    if x >= 9: return "High Risk"
    if x >= 5: return "Medium Risk"
    return "Low Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_cat)

# ---- Sidebar Filters ----
st.sidebar.header("Filters")
regions = sorted(df["Region"].dropna().unique())
industries = sorted(df["Industry"].dropna().unique())

region_f = st.sidebar.multiselect("Region", regions, default=regions)
industry_f = st.sidebar.multiselect("Industry", industries, default=industries)
risk_f = st.sidebar.multiselect("Risk Category", ["Low Risk","Medium Risk","High Risk"],
                               default=["Low Risk","Medium Risk","High Risk"])

f = df[df["Region"].isin(region_f) & df["Industry"].isin(industry_f) & df["Risk_Category"].isin(risk_f)].copy()

# ---- KPI Cards ----
total = len(f)
high = (f["Risk_Category"]=="High Risk").sum()
avg_rev = f["Monthly_Revenue_USD"].mean() if total else 0
churn_pct = (f["Renewal_Status"].eq("No").mean()*100) if total else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Clients", total)
c2.metric("High Risk Clients", high)
c3.metric("Churn Rate %", f"{churn_pct:.2f}%")
c4.metric("Avg Revenue / Client", f"${avg_rev:,.2f}")

st.divider()

# ---- Charts ----
a,b = st.columns(2)

with a:
    st.subheader("Risk Category Distribution")
    counts = f["Risk_Category"].value_counts().reindex(["Low Risk","Medium Risk","High Risk"]).fillna(0)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Clients")
    st.pyplot(fig)

with b:
    st.subheader("Industry-wise Risk (table)")
    pivot = pd.pivot_table(f, index="Industry", columns="Risk_Category", values="Client_ID", aggfunc="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)

st.divider()

c,d = st.columns(2)
with c:
    st.subheader("Revenue vs Risk")
    fig2, ax2 = plt.subplots()
    ax2.scatter(f["Monthly_Revenue_USD"], f["Risk_Score"])
    ax2.set_xlabel("Monthly Revenue (USD)")
    ax2.set_ylabel("Risk Score")
    st.pyplot(fig2)

with d:
    st.subheader("Contract Length vs Churn")
    churn = f["Renewal_Status"].map({"Yes":0, "No":1})
    fig3, ax3 = plt.subplots()
    ax3.scatter(f["Contract_Length_Months"], churn)
    ax3.set_xlabel("Contract Length (Months)")
    ax3.set_ylabel("Churned (1=Yes,0=No)")
    st.pyplot(fig3)

st.divider()

# ---- ML: Decision Tree ----
st.subheader("Decision Tree: Churn Prediction")

y = df["Renewal_Status"].map({"Yes":1, "No":0})

X = df[[
    "Monthly_Usage_Score","Payment_Delay_Days","Contract_Length_Months",
    "Support_Tickets_Last30Days","Monthly_Revenue_USD","Risk_Score"
]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.write("Accuracy:", round(accuracy_score(y_test, pred), 4))
st.write("Confusion Matrix (Actual rows, Pred columns):")
st.write(confusion_matrix(y_test, pred))

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.subheader("Feature Importance")
st.bar_chart(imp)

st.divider()

# ---- Top 20 High-Risk ----
st.subheader("Top 20 High-Risk Clients")
top20 = f.sort_values(["Risk_Score","Monthly_Revenue_USD"], ascending=[False,False]).head(20)
st.dataframe(top20, use_container_width=True)

st.divider()

# ---- Retention Suggestions ----
st.subheader("AI-Based Retention Suggestions")
if st.button("Generate Retention Strategy"):
    st.write("1) Payment delay > 30: flexible payment plan + early-pay discount.")
    st.write("2) Low usage: training + onboarding refresh + usage nudges.")
    st.write("3) High tickets: dedicated account manager + priority support.")
    st.write("4) Short contract: long-term renewal incentive (discount/add-ons).")
    st.write("5) High revenue & high risk: executive call + custom success plan.")

st.divider()

# ---- Responsible AI ----
st.subheader("Responsible AI: Ethical Implications of Predicting Client Churn")
st.write("""
- **Bias:** The model may unfairly learn patterns against certain industries/regions.
- **Label harm:** ‘High Risk’ tags can cause teams to treat clients negatively and increase churn.
- **Privacy:** Usage/payment/support data must be secured and access controlled.
- **Human judgment:** Predictions are support tools, not final decisions.
- **Audit:** Regular checks for fairness + performance drift are required.
""")
