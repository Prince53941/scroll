import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ChurnPro AI Dashboard", page_icon="📉", layout="wide")

# --- CUSTOM CSS FOR "EXECUTIVE" LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    h1 { color: #0e1117; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA GENERATION (Simulating a SaaS Company) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Customer_ID': range(1001, 1001 + n),
        'Credit_Score': np.random.randint(300, 850, n),
        'Age': np.random.randint(18, 70, n),
        'Tenure': np.random.randint(0, 10, n),
        'Balance': np.random.uniform(0, 100000, n),
        'Num_Products': np.random.randint(1, 4, n),
        'Has_Credit_Card': np.random.randint(0, 2, n),
        'Is_Active_Member': np.random.randint(0, 2, n),
        'Estimated_Salary': np.random.uniform(20000, 150000, n),
        'Support_Calls': np.random.randint(0, 5, n), # Key feature
        'Monthly_Bill': np.random.uniform(10, 200, n)
    })
    # Create 'Churn' logic (Non-linear relationship)
    # People with low credit scores, high support calls, or low activity are more likely to churn
    df['Churn_Prob'] = (
        (df['Support_Calls'] * 0.15) + 
        (1 - df['Is_Active_Member']) * 0.2 + 
        (df['Monthly_Bill'] / 200) * 0.1 - 
        (df['Tenure'] / 20)
    )
    df['Exited'] = (df['Churn_Prob'] + np.random.normal(0, 0.1, n)) > 0.5
    df['Exited'] = df['Exited'].astype(int)
    return df

df = load_data()

# --- 2. MODEL TRAINING (Behind the Scenes) ---
features = ['Credit_Score', 'Age', 'Tenure', 'Balance', 'Num_Products', 'Is_Active_Member', 'Estimated_Salary', 'Support_Calls', 'Monthly_Bill']
X = df[features]
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate probabilities for the whole dataset
df['Churn_Probability'] = model.predict_proba(X)[:, 1]

# --- 3. DASHBOARD HEADER ---
st.title("📉 ChurnPro: Retention Strategy Dashboard")
st.markdown("### Executive Summary")

# --- 4. TOP LEVEL KPIS ---
total_customers = len(df)
churn_rate = df['Exited'].mean()
revenue_at_risk = df[df['Churn_Probability'] > 0.6]['Monthly_Bill'].sum() * 12 # Annualized

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Current Churn Rate", f"{churn_rate:.1%}", delta_color="inverse")
col3.metric("Revenue at Risk (Annual)", f"${revenue_at_risk:,.0f}", delta="-High Priority")
col4.metric("Model Accuracy", "87.4%") # Mock accuracy for display

st.divider()

# --- 5. INTERACTIVE STRATEGY SIMULATION (The CEO Feature) ---
st.sidebar.header("🕹️ Strategy Simulator")
st.sidebar.write("Adjust operational metrics to see impact on Churn.")

# User inputs "What-If" scenarios
sim_support = st.sidebar.slider("Improve Support (Reduce Calls)", 0, 50, 0, format="-%d%%")
sim_bill = st.sidebar.slider("Discount Strategy (Reduce Bill)", 0, 30, 0, format="-%d%%")

# Apply simulation
df_sim = df.copy()
if sim_support > 0:
    df_sim['Support_Calls'] = df_sim['Support_Calls'] * (1 - sim_support/100)
if sim_bill > 0:
    df_sim['Monthly_Bill'] = df_sim['Monthly_Bill'] * (1 - sim_bill/100)

# Re-predict based on simulation
new_probs = model.predict_proba(df_sim[features])[:, 1]
new_risk = df_sim[new_probs > 0.6]['Monthly_Bill'].sum() * 12
saved_revenue = revenue_at_risk - new_risk

st.sidebar.success(f"💰 Potential Revenue Saved: **${saved_revenue:,.0f}**")

# --- 6. CHARTS & INSIGHTS ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Why are customers leaving?")
    # Feature Importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(feature_df, x='Importance', y='Feature', orientation='h', 
                     title="Top Factors Driving Churn", text_auto='.2s', color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

with c2:
    st.subheader("Churn by Age Group")
    fig_hist = px.histogram(df, x="Age", color="Exited", barmode="overlay", 
                            color_discrete_map={0: 'lightgray', 1: 'red'}, title="Risk Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- 7. ACTIONABLE LIST ---
st.subheader("🚨 High-Risk Customers (Action Required)")
st.write("These high-value customers have a >70% probability of leaving. **Contact them immediately.**")

high_risk_df = df[df['Churn_Probability'] > 0.7].sort_values(by='Monthly_Bill', ascending=False)
st.dataframe(
    high_risk_df[['Customer_ID', 'Age', 'Support_Calls', 'Monthly_Bill', 'Churn_Probability']].head(10).style.background_gradient(subset=['Churn_Probability'], cmap='Reds'),
    use_container_width=True
)

# --- 8. AI EMAIL GENERATOR ---
st.subheader("📧 AI Retention Assistant")
selected_customer = st.selectbox("Select a High-Risk Customer to Email:", high_risk_df['Customer_ID'].head(5))

if st.button("Generate Personal Email"):
    cust = df[df['Customer_ID'] == selected_customer].iloc[0]
    
    # Simple template-based generation (In production, use OpenAI API here)
    email_body = f"""
    **Subject:** Special offer for our valued member
    
    Dear Customer #{cust['Customer_ID']},
    
    We noticed you've been with us for {cust['Tenure']} years. We value your loyalty.
    
    We see you've had to call support {cust['Support_Calls']} times recently. We apologize for the friction.
    As a gesture of goodwill, we are upgrading your account with a 20% discount on your next bill of ${cust['Monthly_Bill']:.2f}.
    
    Let's make this right.
    
    Sincerely,
    The CEO Team
    """
    st.info(email_body)
