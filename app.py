import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. INITIAL APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Behavior & Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Behavior, Wellbeing & Performance Dashboard")
st.markdown("Analyse how screentime, stress, and sleep metrics interact with academic outcomes.")

# Load the cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_student_dataset.csv")
    return df

df = load_data()

# ==========================================
# 2. SIDEBAR FILTER CONTROLS
# ==========================================
st.sidebar.header("🛠️ Dashboard Filters")

# Academic Level Filter
academic_levels = ["All"] + list(df['Academic_Level'].unique())
selected_level = st.sidebar.selectbox("Select Academic Level:", academic_levels)

# Primary Platform Filter
platforms = ["All"] + list(df['Primary_Platform'].unique())
selected_platform = st.sidebar.selectbox("Select Primary Social Platform:", platforms)

# Apply dynamic scoping filters to the data frame context
filtered_df = df.copy()
if selected_level != "All":
    filtered_df = filtered_df[filtered_df['Academic_Level'] == selected_level]
if selected_platform != "All":
    filtered_df = filtered_df[filtered_df['Primary_Platform'] == selected_platform]

# ==========================================
# 3. HIGH-LEVEL KPI METRICS ROW
# ==========================================
st.markdown("### 📊 Cohort At-A-Glance")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Sample Size", f"{len(filtered_df)} Students")
with kpi2:
    st.metric("Average Daily Screen Time", f"{filtered_df['Daily_Usage_Hours'].mean():.1f} Hrs")
with kpi3:
    st.metric("Mean Stress Score", f"{filtered_df['Perceived_Stress_Score'].mean():.1f} / 40")
with kpi4:
    st.metric("Average Cohort GPA", f"{filtered_df['Academic_Performance_GPA'].mean():.2f}")

st.markdown("---")

# ==========================================
# 4. INTERACTIVE GRAPHICS LAYOUT
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📉 Screen Time vs. Academic GPA Trend")
    fig_scatter = px.scatter(
        filtered_df,
        x="Daily_Usage_Hours",
        y="Academic_Performance_GPA",
        color="Gender",
        trendline="ols",
        labels={"Daily_Usage_Hours": "Daily Screen Hours", "Academic_Performance_GPA": "GPA Value"},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.markdown("#### 🛌 Sleep Variance Distribution Across Platforms")
    fig_box = px.box(
        filtered_df,
        x="Primary_Platform",
        y="Sleep_Duration_Hours",
        color="Primary_Platform",
        labels={"Primary_Platform": "Platform App", "Sleep_Duration_Hours": "Sleep (Hours)"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ==========================================
# 5. LIVE ML PREDICTION ENGINES
# ==========================================
st.markdown("### 🔮 Interactive AI Simulation: Predict Student GPA")
st.markdown("Adjust the behavioral markers below to immediately simulate a student's estimated GPA outcome in real time.")

# Train an inline lightweight operational regression engine model on backend load
X_train = df[['Age', 'Daily_Usage_Hours', 'Sleep_Duration_Hours', 'Perceived_Stress_Score']]
y_train = df['Academic_Performance_GPA']
sim_reg = LinearRegression().fit(X_train, y_train)

# Setup numerical input control grids for simulation
sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

with sim_col1:
    input_age = st.slider("Student Age:", int(df['Age'].min()), int(df['Age'].max()), 18)
with sim_col2:
    input_usage = st.slider("Daily Screen Time (Hours):", 0.0, 16.0, 4.5, step=0.5)
with sim_col3:
    input_sleep = st.slider("Sleep Duration (Hours):", 2.0, 12.0, 7.0, step=0.5)
with sim_col4:
    input_stress = st.slider("Perceived Stress (0-40 Scale):", 0, 40, 15)

# Calculate live array vector prediction point
user_input = np.array([[input_age, input_usage, input_sleep, input_stress]])
predicted_gpa = sim_reg.predict(user_input)[0]

# Bound the display prediction within realistic absolute mathematical scaling limits
predicted_gpa = max(0.00, min(4.00, predicted_gpa))

# Render Simulated Outcome
st.markdown(f"#### 🎯 Predicted Academic GPA Result: `{predicted_gpa:.2f}`")
if predicted_gpa >= 3.5:
    st.success("🌟 Outcome Estimate: High Academic Achievement Profile.")
elif predicted_gpa >= 2.5:
    st.info("⚖️ Outcome Estimate: Average Cohort Performance Profile.")
else:
    st.warning("⚠️ Outcome Estimate: Elevated Behavioral Risk Profile. Academic support recommended.")
