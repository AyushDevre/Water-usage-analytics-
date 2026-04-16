import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

load_dotenv()

# ================= GROQ AI SETUP =================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Water Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# ================= ADVANCED UI STYLING =================
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background-color: #0F1419;
        color: #E0E0E0;
    }
    
    .main {
        background-color: #0F1419;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1C2333;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2D3748;
        border-radius: 8px;
        padding: 12px 24px;
        color: #00D4FF;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00D4FF !important;
        color: #0F1419 !important;
    }
    
    h1, h2, h3 {
        color: #00D4FF;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1C2333 0%, #2D3748 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00D4FF;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    .stMetric label {
        color: #00D4FF;
        font-weight: 600;
    }
    
    .stButton > button {
        background-color: #00D4FF;
        color: #0F1419;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00BFEB;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stSelectbox, .stSlider {
        background-color: #1C2333;
        border-radius: 8px;
    }
    
    .stDataFrame {
        background-color: #1C2333 !important;
        border-radius: 10px;
    }
    
    hr {
        border: 1px solid #2D3748;
        margin: 30px 0;
    }
    
    .card {
        background: linear-gradient(135deg, #1C2333 0%, #2D3748 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3A4556;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1C2333 0%, #2D3748 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #00D4FF;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    .anomaly-high {
        color: #FF6B6B;
        font-weight: 600;
    }
    
    .anomaly-normal {
        color: #51CF66;
        font-weight: 600;
    }
    
    .success-box {
        background-color: #1C3A3A;
        border-left: 4px solid #51CF66;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #3A3A1C;
        border-left: 4px solid #FFD700;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #3A1C1C;
        border-left: 4px solid #FF6B6B;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("data/data.csv")

df = load_data()

# ================= HELPER FUNCTIONS =================
def detect_anomalies(data, column, threshold=2):
    """Detect anomalies using z-score method"""
    mean = data[column].mean()
    std = data[column].std()
    z_scores = np.abs((data[column] - mean) / std)
    return z_scores > threshold

def get_country_comparison(country1, country2, year):
    """Get comparison data for two countries"""
    data1 = df[(df["location"] == country1) & (df["year"] == year)]
    data2 = df[(df["location"] == country2) & (df["year"] == year)]
    return data1, data2

def create_anomaly_chart(data, x_col, y_col, color_col=None):
    """Create chart with anomaly highlighting"""
    data_copy = data.copy()
    data_copy["anomaly"] = detect_anomalies(data_copy, y_col, threshold=1.5)
    
    fig = px.scatter(data_copy, x=x_col, y=y_col, color="anomaly", 
                     color_discrete_map={True: "#FF6B6B", False: "#00D4FF"},
                     title=f"Water Usage Anomalies")
    return fig

# ================= TITLE & HEADER =================
st.title("💧 Industrial Water Usage Analytics Dashboard")
st.markdown("Advanced Professional Analytics Platform | Real-time Monitoring & Forecasting")
st.markdown("---")

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔍 Filters & Options")

countries = sorted(df["location"].unique())
selected_country = st.sidebar.selectbox("🌍 Primary Country", countries)
years = sorted(df["year"].unique())
selected_year = st.sidebar.selectbox("📅 Year", years)

# Comparison feature
enable_comparison = st.sidebar.checkbox("🔄 Compare Two Countries")
if enable_comparison:
    selected_country2 = st.sidebar.selectbox("🌍 Secondary Country", 
                                             [c for c in countries if c != selected_country])
else:
    selected_country2 = None

st.sidebar.markdown("---")

# ================= DATA FILTERING =================
country_df = df[df["location"] == selected_country]
year_country_df = df[(df["location"] == selected_country) & (df["year"] == selected_year)]

if enable_comparison and selected_country2:
    country_df2 = df[df["location"] == selected_country2]
    year_country_df2 = df[(df["location"] == selected_country2) & (df["year"] == selected_year)]

# ================= TOP KPI METRICS =================
st.subheader("📊 Key Metrics Overview")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

total_water = int(df["water_usage"].sum())
highest_industry = year_country_df.groupby("industry")["water_usage"].sum().idxmax() if len(year_country_df) > 0 else "N/A"
avg_efficiency = round((country_df["water_usage"] / country_df["production_units"]).mean(), 2)
year_usage = int(year_country_df["water_usage"].sum())

kpi_col1.metric("💧 Global Water Usage", f"{total_water:,} L", 
                delta=f"Global Total", delta_color="off")
kpi_col2.metric("🏭 Top Industry", highest_industry, 
                delta=f"{selected_year}", delta_color="off")
kpi_col3.metric("⚡ Avg Efficiency", f"{avg_efficiency} L/unit", 
                delta=f"in {selected_country}", delta_color="off")
kpi_col4.metric("📅 Year Usage", f"{year_usage:,} L", 
                delta=f"{selected_year}", delta_color="off")

st.markdown("---")

# ================= TABS FOR SECTIONS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🌍 Country Analysis", "🌎 Global Comparison", "🔮 What-If Analysis", "🤖 AI Insights"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    st.subheader(f"📈 Water Usage Trend - {selected_country}")
    
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        # Line chart with multiple industries
        fig1 = px.line(country_df, x="year", y="water_usage", 
                      color="industry", markers=True,
                      title=f"Water Usage Trend by Industry",
                      labels={"water_usage": "Water Usage (L)", "year": "Year"})
        fig1.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_trend2:
        # Efficiency trend
        df_eff = country_df.copy()
        df_eff["efficiency"] = df_eff["water_usage"] / df_eff["production_units"]
        fig_eff = px.line(df_eff, x="year", y="efficiency", 
                         color="industry", markers=True,
                         title=f"Efficiency Trend by Industry",
                         labels={"efficiency": "L/Unit", "year": "Year"})
        fig_eff.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419"
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(f"🏭 Industry Breakdown - {selected_country} ({selected_year})")
    
    col_ind1, col_ind2 = st.columns(2)
    
    with col_ind1:
        fig2 = px.bar(year_country_df, x="industry", y="water_usage", 
                     color="industry",
                     title=f"Water Usage by Industry",
                     labels={"water_usage": "Water Usage (L)", "industry": "Industry"})
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_ind2:
        # Pie chart
        fig_pie = px.pie(year_country_df, values="water_usage", names="industry",
                        title="Industry Distribution",
                        color_discrete_sequence=px.colors.qualitative.Safe)
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F1419"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("⚡ Efficiency Distribution")
    df_eff_box = df.copy()
    df_eff_box["efficiency"] = df_eff_box["water_usage"] / df_eff_box["production_units"]
    
    fig5 = px.box(df_eff_box, x="industry", y="efficiency", color="industry",
                 title="Efficiency Distribution Across Industries",
                 labels={"efficiency": "L/Unit", "industry": "Industry"})
    fig5.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0F1419",
        showlegend=False
    )
    st.plotly_chart(fig5, use_container_width=True)

# ================= TAB 2: COUNTRY ANALYSIS =================
with tab2:
    st.subheader(f"🔍 Detailed Analysis - {selected_country}")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write("**Water Usage Statistics**")
        usage_stats = country_df.groupby("industry")["water_usage"].agg(['sum', 'mean', 'std']).round(2)
        st.dataframe(usage_stats)
    
    with col_info2:
        st.write("**Efficiency Statistics**")
        eff_stats = country_df.copy()
        eff_stats["efficiency"] = eff_stats["water_usage"] / eff_stats["production_units"]
        eff_by_industry = eff_stats.groupby("industry")["efficiency"].agg(['mean', 'min', 'max']).round(2)
        st.dataframe(eff_by_industry)
    
    st.markdown("---")
    
    # Anomaly detection
    st.subheader("🚨 Anomaly Detection")
    
    year_country_df_anom = year_country_df.copy()
    year_country_df_anom["anomaly"] = detect_anomalies(year_country_df_anom, "water_usage", threshold=1.5)
    
    anomaly_count = year_country_df_anom["anomaly"].sum()
    
    if anomaly_count > 0:
        st.markdown(f'<div class="error-box">⚠️ {anomaly_count} anomalies detected in water usage patterns</div>', 
                   unsafe_allow_html=True)
        anomalies_df = year_country_df_anom[year_country_df_anom["anomaly"]][["industry", "water_usage", "production_units"]]
        st.dataframe(anomalies_df)
    else:
        st.markdown('<div class="success-box">✅ No anomalies detected. Water usage is within normal range.</div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ranking
    st.subheader("🥇 Industry Ranking")
    ranking_df = year_country_df.groupby("industry")["water_usage"].sum().reset_index().sort_values(by="water_usage", ascending=False)
    
    fig_rank = px.bar(ranking_df, x="water_usage", y="industry", orientation="h", color="water_usage",
                     color_continuous_scale="Reds_r",
                     title="Industry Ranking by Water Usage",
                     labels={"water_usage": "Water Usage (L)"})
    fig_rank.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1C2333",
        paper_bgcolor="#0F1419"
    )
    st.plotly_chart(fig_rank, use_container_width=True)

# ================= TAB 3: GLOBAL COMPARISON =================
with tab3:
    st.subheader(f"🌎 Global Comparison ({selected_year})")
    
    year_df = df[df["year"] == selected_year]
    
    col_global1, col_global2 = st.columns(2)
    
    with col_global1:
        fig3 = px.bar(year_df, x="industry", y="water_usage", color="location", barmode="group",
                     title="Global Industry Comparison",
                     labels={"water_usage": "Water Usage (L)", "industry": "Industry"})
        fig3.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419",
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_global2:
        # Sunburst chart
        fig_sun = px.sunburst(year_df, path=["location", "industry"], values="water_usage",
                             color="water_usage", color_continuous_scale="Blues",
                             title="Global Water Usage Hierarchy")
        fig_sun.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F1419"
        )
        st.plotly_chart(fig_sun, use_container_width=True)
    
    st.markdown("---")
    
    # Country vs Country Comparison
    if enable_comparison and selected_country2:
        st.subheader(f"🔄 {selected_country} vs {selected_country2}")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.write(f"**{selected_country}**")
            fig_comp1 = px.bar(year_country_df, x="industry", y="water_usage", color="industry",
                              title=f"{selected_country} Water Usage",
                              labels={"water_usage": "Water Usage (L)"})
            fig_comp1.update_layout(
                template="plotly_dark",
                plot_bgcolor="#1C2333",
                paper_bgcolor="#0F1419",
                showlegend=False
            )
            st.plotly_chart(fig_comp1, use_container_width=True)
            
            country1_total = int(year_country_df["water_usage"].sum())
            st.metric("Total Usage", f"{country1_total:,} L")
        
        with comp_col2:
            st.write(f"**{selected_country2}**")
            fig_comp2 = px.bar(year_country_df2, x="industry", y="water_usage", color="industry",
                              title=f"{selected_country2} Water Usage",
                              labels={"water_usage": "Water Usage (L)"})
            fig_comp2.update_layout(
                template="plotly_dark",
                plot_bgcolor="#1C2333",
                paper_bgcolor="#0F1419",
                showlegend=False
            )
            st.plotly_chart(fig_comp2, use_container_width=True)
            
            country2_total = int(year_country_df2["water_usage"].sum())
            st.metric("Total Usage", f"{country2_total:,} L")
        
        # Trend comparison
        st.subheader("📊 Trend Comparison Over Years")
        
        trend1 = country_df.groupby("year")["water_usage"].sum().reset_index()
        trend1["location"] = selected_country
        
        trend2 = country_df2.groupby("year")["water_usage"].sum().reset_index()
        trend2["location"] = selected_country2
        
        trend_combined = pd.concat([trend1, trend2])
        
        fig_trend_comp = px.line(trend_combined, x="year", y="water_usage", color="location",
                                markers=True,
                                title="Water Usage Trend Comparison",
                                labels={"water_usage": "Water Usage (L)"})
        fig_trend_comp.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend_comp, use_container_width=True)

# ================= TAB 4: WHAT-IF ANALYSIS =================
with tab4:
    st.subheader("🔮 What-If Scenario Simulation")
    
    st.write("Adjust water reduction percentages by industry and see the impact on efficiency and costs.")
    st.markdown("---")
    
    what_if_df = year_country_df.copy()
    
    # Create sliders for each industry
    industries = what_if_df["industry"].unique()
    reductions = {}
    
    reduction_cols = st.columns(len(industries))
    for idx, industry in enumerate(industries):
        with reduction_cols[idx]:
            reductions[industry] = st.slider(
                f"Reduce {industry} (%)",
                0, 50, 10,
                key=f"slider_{industry}"
            )
    
    st.markdown("---")
    
    # Apply reductions
    what_if_df["adjusted_water"] = what_if_df.apply(
        lambda row: row["water_usage"] * (1 - reductions.get(row["industry"], 0) / 100),
        axis=1
    )
    
    what_if_df["adjusted_efficiency"] = what_if_df["adjusted_water"] / what_if_df["production_units"]
    what_if_df["potential_savings"] = what_if_df["water_usage"] - what_if_df["adjusted_water"]
    
    # Display metrics
    st.subheader("📊 Scenario Results")
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        current_total = int(year_country_df["water_usage"].sum())
        st.metric("Current Usage", f"{current_total:,} L")
    
    with result_col2:
        adjusted_total = int(what_if_df["adjusted_water"].sum())
        st.metric("Adjusted Usage", f"{adjusted_total:,} L")
    
    with result_col3:
        savings = int(what_if_df["potential_savings"].sum())
        savings_pct = (savings / current_total * 100)
        st.metric("Potential Savings", f"{savings:,} L", delta=f"{savings_pct:.1f}%")
    
    with result_col4:
        new_efficiency = round((what_if_df["adjusted_water"] / what_if_df["production_units"]).mean(), 2)
        current_efficiency = round((year_country_df["water_usage"] / year_country_df["production_units"]).mean(), 2)
        improvement = new_efficiency - current_efficiency
        st.metric("New Efficiency", f"{new_efficiency}", delta=f"{improvement:.2f} L/unit")
    
    st.markdown("---")
    
    # Comparison charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        comparison_df = what_if_df[["industry", "water_usage", "adjusted_water"]].copy()
        comparison_df = comparison_df.rename(columns={"water_usage": "Current", "adjusted_water": "Adjusted"})
        
        fig_compare = go.Figure(data=[
            go.Bar(x=comparison_df["industry"], y=comparison_df["Current"], name="Current Usage", marker_color="#FF6B6B"),
            go.Bar(x=comparison_df["industry"], y=comparison_df["Adjusted"], name="Adjusted Usage", marker_color="#51CF66")
        ])
        fig_compare.update_layout(
            title="Current vs Adjusted Water Usage",
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419",
            hovermode="x unified"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    
    with chart_col2:
        fig_savings = px.bar(what_if_df, x="industry", y="potential_savings", color="potential_savings",
                            color_continuous_scale="Greens",
                            title="Water Savings by Industry",
                            labels={"potential_savings": "Savings (L)"})
        fig_savings.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1C2333",
            paper_bgcolor="#0F1419"
        )
        st.plotly_chart(fig_savings, use_container_width=True)
    
    st.markdown("---")
    
    # Data table
    st.subheader("📋 Detailed Scenario Breakdown")
    display_df = what_if_df[["industry", "water_usage", "adjusted_water", "potential_savings", "adjusted_efficiency"]].copy()
    display_df.columns = ["Industry", "Current (L)", "Adjusted (L)", "Savings (L)", "New Efficiency (L/Unit)"]
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)

# ================= TAB 5: AI INSIGHTS =================
with tab5:
    st.subheader("🤖 AI-Powered Sustainability Insights")
    
    st.write("Leverage AI to analyze your water usage patterns and get actionable recommendations.")
    st.markdown("---")
    
    if st.button("🚀 Generate AI Insights", use_container_width=True):
        
        summary = year_country_df.groupby("industry")["water_usage"].sum().to_dict()
        avg_efficiency = (year_country_df["water_usage"] / year_country_df["production_units"]).mean()
        
        prompt = f"""
        You are a water sustainability expert analyzing industrial water consumption data.
        
        Analyze the following data for {selected_country} in {selected_year}:
        
        Water Usage by Industry:
        {summary}
        
        Average Efficiency: {avg_efficiency:.2f} L per production unit
        
        Provide a structured analysis with:
        
        1. **Key Problems** (2-3 main issues)
        2. **Worst Performing Industry** (which industry is consuming most water and why)
        3. **Actionable Solutions** (5-7 specific, practical recommendations to reduce water usage)
        4. **Priority Actions** (top 3 immediate steps)
        5. **Estimated Impact** (potential water savings percentage if recommendations are implemented)
        
        Keep the response professional, specific, and actionable. Use bullet points for clarity.
        """
        
        try:
            with st.spinner("🔄 Generating AI insights..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                st.success("✅ AI Insights Generated Successfully!")
                st.markdown("---")
                st.markdown(response.choices[0].message.content)
                
                # Download insights
                insights_text = response.choices[0].message.content
                st.markdown("---")
                
                col_download1, col_download2 = st.columns(2)
                with col_download1:
                    st.download_button(
                        label="📥 Download Insights as TXT",
                        data=insights_text,
                        file_name=f"AI_Insights_{selected_country}_{selected_year}.txt",
                        mime="text/plain"
                    )
                
                with col_download2:
                    st.info("💡 Tip: Share these insights with your team for implementation planning.")
        
        except Exception as e:
            st.error(f"❌ AI Generation Error: {str(e)}")
            st.markdown("---")
            
            st.warning("⚠️ Using Fallback Analysis...")
            
            # Fallback intelligent analysis
            avg_eff = (year_country_df["water_usage"] / year_country_df["production_units"]).mean()
            top_industry = year_country_df.groupby("industry")["water_usage"].sum().idxmax()
            top_usage = year_country_df.groupby("industry")["water_usage"].sum().max()
            total_usage = year_country_df["water_usage"].sum()
            top_pct = (top_usage / total_usage * 100)
            
            st.markdown(f"""
            ### 🔍 Key Problems
            - **{top_industry}** industry is the largest water consumer ({top_pct:.1f}% of total usage)
            - High water efficiency (>50 L/unit) indicates potential for optimization
            - Lack of water recycling infrastructure in key industries
            
            ### 🏭 Worst Performing Industry
            **{top_industry}** - Uses {top_usage:,.0f} L annually
            - Primary concern: High production volume with inefficient water practices
            - Opportunity: Implement water recirculation systems
            
            ### 💡 Actionable Solutions
            1. **Install Water Recycling Systems** - Reuse treated water in production processes
            2. **Upgrade Equipment** - Replace old machinery with water-efficient models
            3. **Implement Monitoring** - Deploy real-time water usage monitoring systems
            4. **Staff Training** - Train operators on water conservation best practices
            5. **Process Optimization** - Analyze and redesign production processes for efficiency
            6. **Leak Detection** - Conduct quarterly pipe maintenance and leak inspections
            7. **Water Harvesting** - Capture rainwater for non-critical processes
            
            ### ⚡ Priority Actions
            1. Audit current water usage in {top_industry} industry
            2. Install water meters on all major consumption points
            3. Launch awareness campaign among production staff
            
            ### 📊 Estimated Impact
            - **Potential Savings**: 20-30% reduction in water usage
            - **Timeline**: 6-12 months for full implementation
            - **ROI**: 2-3 years based on typical water costs
            """)

st.markdown("---")

# ================= DATA SECTION =================
st.subheader("📄 Data Management")

col_data1, col_data2 = st.columns(2)

with col_data1:
    st.write("**Download Options**")
    
    # Filter data for download
    download_df = year_country_df.copy()
    csv = download_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"water_data_{selected_country}_{selected_year}.csv",
        mime="text/csv"
    )

with col_data2:
    st.write(f"**Records: {len(year_country_df)} entries**")
    st.info(f"Showing data for {selected_country} in {selected_year}")

st.markdown("---")

st.subheader("📋 Dataset Preview")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #00D4FF; margin-top: 50px; padding: 20px; border-top: 1px solid #2D3748;'>
    <p><strong>💧 Industrial Water Usage Analytics Dashboard</strong></p>
    <p style='font-size: 12px; color: #888;'>Powered by Streamlit, Plotly & GROQ AI | Last Updated: 2026-04-16</p>
</div>
""", unsafe_allow_html=True)