import streamlit as st
import pandas as pd
import plotly.express as px
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
print("DEBUG KEY:", os.getenv("GROQ_API_KEY"))

# ================= GROQ AI SETUP =================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Water Analytics", layout="wide")

# ================= UI STYLING =================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("💧 Industrial Water Usage Analytics Dashboard")
st.markdown("---")

# ================= LOAD DATA =================
df = pd.read_csv("data/data.csv")

# ================= SIDEBAR =================
st.sidebar.header("🔍 Filters")
selected_country = st.sidebar.selectbox("🌍 Select Country", df["location"].unique())
selected_year = st.sidebar.selectbox("📅 Select Year", sorted(df["year"].unique()))

# ================= FILTER DATA =================
country_df = df[df["location"] == selected_country]
year_country_df = df[(df["location"] == selected_country) & (df["year"] == selected_year)]

# ================= KPI =================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("💧 Total Water Usage", f"{int(country_df['water_usage'].sum()):,} L")
col2.metric("📅 Year Usage", f"{int(year_country_df['water_usage'].sum()):,} L")
col3.metric("⚡ Avg Efficiency", round((country_df["water_usage"] / country_df["production_units"]).mean(), 2))

st.markdown("---")

# ================= TREND =================
st.subheader(f"🌍 Water Usage Trend in {selected_country}")
fig1 = px.line(country_df, x="year", y="water_usage", color="industry", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# ================= INDUSTRY BREAKDOWN =================
st.subheader(f"🏭 Industry Usage in {selected_country} ({selected_year})")
fig2 = px.bar(year_country_df, x="industry", y="water_usage", color="industry")
st.plotly_chart(fig2, use_container_width=True)

# ================= GLOBAL COMPARISON =================
st.subheader(f"🌎 Global Comparison ({selected_year})")
year_df = df[df["year"] == selected_year]
fig3 = px.bar(year_df, x="industry", y="water_usage", color="location", barmode="group")
st.plotly_chart(fig3, use_container_width=True)

# ================= RANKING =================
st.subheader("🥇 Industry Ranking (Overall)")
ranking_df = df.groupby("industry")["water_usage"].sum().reset_index()
ranking_df = ranking_df.sort_values(by="water_usage", ascending=False)
fig4 = px.bar(ranking_df, x="industry", y="water_usage", color="industry")
st.plotly_chart(fig4, use_container_width=True)

# ================= EFFICIENCY =================
st.subheader("⚡ Efficiency Distribution")
df["efficiency"] = df["water_usage"] / df["production_units"]
fig5 = px.box(df, x="industry", y="efficiency", color="industry")
st.plotly_chart(fig5, use_container_width=True)

# ================= WHAT-IF ANALYSIS =================
st.subheader("🔮 What-If Analysis")

reduction = st.slider("Reduce Water Usage (%)", 0, 50, 10)

what_if_df = year_country_df.copy()
what_if_df["adjusted_water"] = what_if_df["water_usage"] * (1 - reduction/100)

st.write(f"### Adjusted Water Usage after {reduction}% reduction")

fig6 = px.bar(what_if_df, x="industry", y="adjusted_water", color="industry")
st.plotly_chart(fig6, use_container_width=True)

new_eff = (what_if_df["adjusted_water"] / what_if_df["production_units"]).mean()
st.metric("New Efficiency", round(new_eff, 2))

st.markdown("---")

# ================= AI SUGGESTIONS =================
st.subheader("🤖 AI Sustainability Suggestions")

if st.button("Generate AI Suggestions"):

    summary = year_country_df.groupby("industry")["water_usage"].sum().to_dict()

    prompt = f"""
    You are a sustainability expert.

    Analyze water usage for {selected_country} in {selected_year}:
    {summary}

    Give:
    - Key problems
    - Worst industry
    - Practical solutions to reduce water usage

    Keep it short and actionable.
    """

    try:
        with st.spinner("Generating AI suggestions..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            st.success("✅ AI Suggestions Generated")
            st.write(response.choices[0].message.content)

    except Exception as e:
        st.error("❌ AI Error")
        st.code(str(e))

        # ===== FALLBACK =====
        st.warning("⚠ Using fallback suggestions...")

        avg_eff = (country_df["water_usage"] / country_df["production_units"]).mean()
        top_industry = country_df.groupby("industry")["water_usage"].sum().idxmax()

        st.write(f"🔍 Highest water consuming industry: **{top_industry}**")

        if avg_eff > 80:
            st.error("⚠ High water usage detected.")
        elif avg_eff > 50:
            st.warning("⚠ Moderate efficiency.")
        else:
            st.success("✅ Good efficiency.")

        st.write("👉 Optimize processes, reuse water, and adopt efficient technologies.")

# ================= DATA =================
st.subheader("📄 Dataset Preview")
st.dataframe(df)