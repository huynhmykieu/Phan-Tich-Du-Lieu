import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AI Jobs Analytics Dashboard",
    layout="wide"
)

df = pd.read_csv("ai_jobs_market_2025_2026.csv")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("AI Jobs Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Job Explorer",
        "Skill Analysis",
        "Salary Prediction Model",
        "Career Path Recommendation"
    ]
)

# =========================
# PAGE 1
# =========================

if page == "Market Overview":

    st.title("AI Jobs Market Analysis (2025–2026)")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Jobs", len(df))
    c2.metric("Avg Salary (USD)", f"{df['annual_salary_usd'].mean():,.0f}")
    c3.metric("Avg Demand Score", f"{df['demand_score'].mean():.1f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        top_salary = (
            df.groupby("job_title")["annual_salary_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_salary.values, y=top_salary.index, ax=ax)
        ax.set_title("Top Roles by Salary")

        st.pyplot(fig)

    with col2:

        top_demand = (
            df.groupby("job_title")["demand_score"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_demand.values, y=top_demand.index, ax=ax)
        ax.set_title("Top Roles by Demand")

        st.pyplot(fig)

# =========================
# PAGE 2
# =========================

elif page == "Job Explorer":

    st.title("Job Explorer")

    job = st.selectbox(
        "Select Job Title",
        sorted(df["job_title"].unique())
    )

    data = df[df["job_title"] == job]

    st.subheader("Key Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Avg Salary", f"${data['annual_salary_usd'].mean():,.0f}")
    c2.metric("Demand Score", f"{data['demand_score'].mean():.1f}")
    c3.metric("Experience (yrs)", f"{data['years_of_experience'].mean():.1f}")

    st.subheader("Required Skills")

    skills = []
    for s in data["required_skills"]:
        skills.extend(str(s).split("|"))

    skill_df = pd.Series(skills).value_counts().head(10)

    st.bar_chart(skill_df)

# =========================
# PAGE 3
# =========================

elif page == "Skill Analysis":

    st.title("Skill Analysis")

    skills = []
    for s in df["required_skills"]:
        skills.extend(str(s).split("|"))

    skill_list = sorted(set(skills))

    selected = st.selectbox("Select Skill", skill_list)

    filtered = df[df["required_skills"].str.contains(selected, na=False)]

    st.metric("Number of Jobs", len(filtered))

    st.dataframe(
        filtered[
            ["job_title", "annual_salary_usd", "demand_score", "country"]
        ].sort_values("annual_salary_usd", ascending=False).head(20)
    )

# =========================
# PAGE 4
# =========================

elif page == "Salary Prediction Model":

    st.title("Salary Prediction Model")

    X = df[["years_of_experience", "demand_score", "is_llm_role"]]
    y = df["annual_salary_usd"]

    model = LinearRegression()
    model.fit(X, y)

    exp = st.slider("Years of Experience", 0, 15, 3)
    demand = st.slider("Demand Score", 0, 100, 80)
    llm = st.selectbox("LLM Role", ["No", "Yes"])

    llm_val = 1 if llm == "Yes" else 0

    if st.button("Predict"):

        pred = model.predict([[exp, demand, llm_val]])[0]

        st.success(f"Predicted Salary: ${pred:,.0f}")

# =========================
# PAGE 5
# =========================

elif page == "Career Path Recommendation":

    st.title("Career Path Recommendation")

    st.markdown("""
**Year 1**
- Python
- SQL
- Statistics

↓

**Year 2**
- Machine Learning
- Data Analysis

↓

**Year 3**
- Deep Learning
- NLP
- Cloud

↓

**Year 4**
- LLM Engineering
- MLOps
- AI Systems

---

**Target Roles**
- AI Engineer
- ML Engineer
- MLOps Engineer
- LLM Engineer
""")

    st.info("Recommendation based on market analysis 2025–2026")