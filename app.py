import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="Geldium Credit Risk AI", page_icon="🏦", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #10b981; color: white; font-weight: bold; width: 100%; border-radius: 8px;}
    .stButton>button:hover {background-color: #059669; color: white;}
    .risk-high {color: #dc2626; font-size: 24px; font-weight: bold;}
    .risk-medium {color: #d97706; font-size: 24px; font-weight: bold;}
    .risk-low {color: #16a34a; font-size: 24px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🏦 Geldium Proactive Collections AI")
st.markdown("Predict customer delinquency risk using an advanced XGBoost model.")
st.divider()

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('credit_risk_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load the model. The actual Python error is: {e}")
    st.stop()

# --- UI LOGIC: Input Forms ---
st.subheader("Customer Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics & Accounts**")
    age = st.slider("Age", 18, 90, 35)
    emp_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired"])
    account_tenure = st.slider("Account Tenure (Months)", 0, 120, 24)
    card_type = st.selectbox("Credit Card Type", ["Standard", "Platinum", "Student", "Business"])
    location = st.selectbox("Location", ["New York", "Los Angeles", "Chicago", "Phoenix", "Other"])

with col2:
    st.markdown("**Financial Capacity**")
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
    loan_balance = st.number_input("Current Loan Balance ($)", min_value=0, max_value=200000, value=25000, step=1000)
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01)

with col3:
    st.markdown("**Credit Behavior**")
    credit_score = st.slider("Credit Score", 300, 850, 680)
    credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.45, 0.01)
    missed_payments = st.number_input("Total Missed Payments", 0, 20, 1)

st.divider()
st.markdown("**Recent Payment History (Last 6 Months)**")
hist_cols = st.columns(6)
months = []
for i, c in enumerate(hist_cols):
    with c:
        val = st.selectbox(f"Month {i+1}", ["On-time", "Late", "Missed"], key=f"m{i}")
        months.append(val)

# --- PREDICTION LOGIC ---
st.divider()
if st.button("Calculate Delinquency Risk"):
    
    # 1. Build the dataframe
    input_dict = {
        'Age': [age],
        'Income': [income],
        'Credit_Score': [credit_score],
        'Credit_Utilization': [credit_utilization],
        'Missed_Payments': [missed_payments],
        'Loan_Balance': [loan_balance],
        'Debt_to_Income_Ratio': [dti_ratio],
        'Employment_Status': [emp_status],
        'Account_Tenure': [account_tenure],
        'Credit_Card_Type': [card_type],
        'Location': [location],
        'Month_1': [months[0]],
        'Month_2': [months[1]],
        'Month_3': [months[2]],
        'Month_4': [months[3]],
        'Month_5': [months[4]],
        'Month_6': [months[5]]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    with st.spinner("Analyzing risk profile..."):
        try:
            # 2. THE FAILSAFE: Force the web app columns to match the model's memory exactly
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
                if len(expected_cols) == len(input_df.columns):
                    input_df.columns = expected_cols # Renames columns to bypass hidden spaces
            
            # 3. Make Prediction
            probability = model.predict_proba(input_df)[0][1] * 100
            
            # 4. Display Results
            st.subheader("Risk Assessment Results")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(label="Default Probability", value=f"{probability:.1f}%")
                
                if probability < 30:
                    st.markdown("<p class='risk-low'>✅ LOW RISK</p>", unsafe_allow_html=True)
                    st.write("Account is healthy. Standard monitoring applies.")
                elif probability < 65:
                    st.markdown("<p class='risk-medium'>⚠️ MEDIUM RISK</p>", unsafe_allow_html=True)
                    st.write("Early warning signs detected. Consider sending proactive educational materials.")
                else:
                    st.markdown("<p class='risk-high'>🚨 HIGH RISK</p>", unsafe_allow_html=True)
                    st.write("Action Required: Trigger proactive restructuring offer immediately.")
                    
            with col_res2:
                st.progress(probability / 100)
                st.write("**AI Recommendation:** Based on the current risk threshold, the system flags changes in Credit Utilization and DTI as primary catalysts. To test this, try lowering the Credit Utilization slider and recalculating.")
                
        except Exception as e:
            # If it still fails, this will print the EXACT reason on your screen so we can fix it.
            st.error(f"🚨 Calculation Error: {e}")
