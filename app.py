import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    .main {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255, 255, 255, 0.02);
        padding: 10px  20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
        padding: 12px 28px;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%) !important;
        color: white !important;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .prediction-card-healthy {
        border-left: 4px solid #00ff88;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.15);
    }
    
    .prediction-card-risk {
        border-left: 4px solid #ff0055;
        box-shadow: 0 4px 20px rgba(255, 0, 85, 0.15);
    }
    
    .prediction-card-unavailable {
        border-left: 4px solid #666;
        opacity: 0.6;
    }
    
    .model-name {
        font-size: 18px;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }
    
    .prediction-result {
        font-size: 24px;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .healthy-text {
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .risk-text {
        color: #ff0055;
        text-shadow: 0 0 20px rgba(255, 0, 85, 0.5);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.5);
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 5px 7px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border: 1px solid #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #ff0055 50%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 18px;
        margin-bottom: 40px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #00d4ff;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .download-button {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.3);
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 255, 136, 0.5);
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        overflow: hidden;
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 212, 255, 0.6);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 0, 85, 0.1) 0%, rgba(255, 102, 0, 0.1) 100%);
        border-left: 4px solid #ff0055;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border-left: 4px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Constants
ALGONAMES = [
    "Decision Tree",
    "Logistic Regression",
    "Random Forest",
    "Support Vector Machine",
    "Grid Search Random Forest",
]

MODELNAMES = [
    "tree.pkl",
    "LogisticRegression.pkl",
    "RandomForest.pkl",
    "SVM.pkl",
    "gridrf.pkl",
]

EXPECTED_COLUMNS = [
    "Age","Sex","ChestPainType","RestingBP","Cholesterol",
    "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"
]

MODEL_ACCURACIES = {
    "Decision Tree": 80.97,
    "Logistic Regression": 85.86,
    "Random Forest": 84.23,
    "Support Vector Machine": 84.22,
    "Grid Search Random Forest": 89.75,
}

# Helper Functions
def get_binary_file_downloader_html(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" class="download-button">📥 Download Predictions CSV</a>'

def encode_single_patient(age, sex_label, chest_pain_label, resting_bp, cholesterol,
                          fasting_bs_label, resting_ecg_label, max_hr,
                          exercise_angina_label, oldpeak, st_slope_label):
    sex_num = 0 if sex_label == "Male" else 1
    chest_map = {
        "Atypical Angina": 0,
        "Non-Anginal Pain": 1,
        "Asymptomatic": 2,
        "Typical Angina": 3,
    }
    chest_num = chest_map[chest_pain_label]
    fasting_num = 1 if fasting_bs_label == "≥ 120 mg/dl" else 0
    resting_ecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2,
    }
    resting_ecg_num = resting_ecg_map[resting_ecg_label]
    exercise_num = 1 if exercise_angina_label == "Yes" else 0
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2,
    }
    slope_num = slope_map[st_slope_label]
    df = pd.DataFrame({
        "Age":[age],"Sex":[sex_num],"ChestPainType":[chest_num],"RestingBP":[resting_bp],
        "Cholesterol":[cholesterol],"FastingBS":[fasting_num],"RestingECG":[resting_ecg_num],
        "MaxHR":[max_hr],"ExerciseAngina":[exercise_num],"Oldpeak":[oldpeak],"ST_Slope":[slope_num]
    })
    return df[EXPECTED_COLUMNS]

def predict_all_models(single_input_df):
    preds = []
    for modelname in MODELNAMES:
        try:
            with open(modelname, "rb") as f:
                model = pickle.load(f)
            preds.append(model.predict(single_input_df)[0])
        except Exception:
            preds.append(np.nan)
    return preds

def encode_bulk_dataframe(df):
    df = df.copy()
    if df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].map({"M": 0, "Male": 0, "F": 1, "Female": 1})
    if df["ChestPainType"].dtype == object:
        df["ChestPainType"] = df["ChestPainType"].map({"ATA":0,"NAP":1,"ASY":2,"TA":3})
    if df["RestingECG"].dtype == object:
        df["RestingECG"] = df["RestingECG"].map({
            "Normal":0,"ST":1,"ST-T":1,"ST-T wave abnormality":1,"LVH":2
        })
    if df["ExerciseAngina"].dtype == object:
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"N":0,"No":0,"Y":1,"Yes":1})
    if df["ST_Slope"].dtype == object:
        df["ST_Slope"] = df["ST_Slope"].map({"Up":0,"Upsloping":0,"Flat":1,"Down":2,"Downsloping":2})
    df = df[EXPECTED_COLUMNS]
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def predict_bulk_with_logreg(encoded_df):
    with open("LogisticRegression.pkl","rb") as f:
        model = pickle.load(f)
    preds = model.predict(encoded_df.values)
    result = encoded_df.copy()
    result["Prediction LR"] = preds
    return result

# Header
st.markdown('<h1 class="hero-title">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced AI-Powered Cardiovascular Risk Assessment</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Single Prediction", "📊 Bulk Prediction", "📈 Model Information"])

# TAB 1: Single Prediction
with tab1:
    st.markdown("### Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, step=1)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["< 120 mg/dl", "≥ 120 mg/dl"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=250, value=150, step=1)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Patient Risk"):
        with st.spinner("Analyzing with multiple AI models..."):
            encoded_df = encode_single_patient(
                age, sex, chest_pain, resting_bp, cholesterol,
                fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
            )
            predictions = predict_all_models(encoded_df)
            
            st.markdown("### 🤖 Model Predictions")
            st.markdown('<div class="info-box">Analysis complete! Results from 5 different machine learning models:</div>', unsafe_allow_html=True)
            
            cols = st.columns(2)
            for idx, (algo, pred) in enumerate(zip(ALGONAMES, predictions)):
                with cols[idx % 2]:
                    if pd.isna(pred):
                        st.markdown(f'''
                        <div class="prediction-card prediction-card-unavailable">
                            <div class="model-name">{algo}</div>
                            <div class="prediction-result" style="color: #666;">⚠️ Model Unavailable</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    elif pred == 0:
                        st.markdown(f'''
                        <div class="prediction-card prediction-card-healthy">
                            <div class="model-name">{algo}</div>
                            <div class="prediction-result healthy-text">✓ Low Risk</div>
                            <div style="color: rgba(255,255,255,0.6); font-size: 14px; margin-top: 8px;">No heart disease detected</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-card prediction-card-risk">
                            <div class="model-name">{algo}</div>
                            <div class="prediction-result risk-text">⚠ High Risk</div>
                            <div style="color: rgba(255,255,255,0.6); font-size: 14px; margin-top: 8px;">Heart disease detected</div>
                        </div>
                        ''', unsafe_allow_html=True)

# TAB 2: Bulk Prediction
with tab2:
    st.markdown("### Upload Patient Data")
    st.markdown('<div class="info-box">Upload a CSV file with patient data for batch prediction using Logistic Regression model</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("#### 📋 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            
            if missing_cols:
                st.markdown(f'<div class="warning-box"><strong>⚠️ Missing Columns:</strong> {", ".join(missing_cols)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✓ All required columns found! Processing predictions...</div>', unsafe_allow_html=True)
                
                encoded_df = encode_bulk_dataframe(df)
                result_df = predict_bulk_with_logreg(encoded_df)
                
                st.markdown("#### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Total Patients</div>
                        <div class="metric-value">{len(result_df)}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    low_risk = (result_df["Prediction LR"] == 0).sum()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Low Risk</div>
                        <div class="metric-value" style="color: #00ff88;">{low_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col3:
                    high_risk = (result_df["Prediction LR"] == 1).sum()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">High Risk</div>
                        <div class="metric-value" style="color: #ff0055;">{high_risk}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(get_binary_file_downloader_html(result_df), unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f'<div class="warning-box">❌ Error processing file: {str(e)}</div>', unsafe_allow_html=True)

# TAB 3: Model Information
with tab3:
    st.markdown("### 📊 Model Performance Comparison")
    
    df_acc = pd.DataFrame(list(MODEL_ACCURACIES.items()), columns=["Model", "Accuracy"])
    
    fig = go.Figure()
    
    colors = ['#00d4ff' if acc < max(MODEL_ACCURACIES.values()) else '#00ff88' for acc in df_acc["Accuracy"]]
    
    fig.add_trace(go.Bar(
        x=df_acc["Model"],
        y=df_acc["Accuracy"],
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        text=df_acc["Accuracy"].apply(lambda x: f'{x:.2f}%'),
        textposition='outside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=False,
            title="Accuracy (%)",
            range=[0, 100]
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    best_model = max(MODEL_ACCURACIES, key=MODEL_ACCURACIES.get)
    best_acc = MODEL_ACCURACIES[best_model]
    
    st.markdown(f'''
    <div class="success-box">
        <h3 style="margin-top: 0;">🏆 Best Performing Model</h3>
        <p><strong>{best_model}</strong> achieves the highest accuracy at <strong>{best_acc}%</strong></p>
        <p style="margin-bottom: 0; opacity: 0.8;">This model is recommended for production deployment and critical predictions.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### 📚 Model Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="glass-card">
            <h4 style="color: #00d4ff; margin-top: 0;">🌳 Decision Tree</h4>
            <p>A tree-based model that makes decisions through a series of questions. Easy to interpret but may overfit.</p>
            <p><strong>Accuracy:</strong> 80.97%</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color: #00d4ff; margin-top: 0;">📈 Logistic Regression</h4>
            <p>A statistical model for binary classification. Fast and reliable for linear relationships.</p>
            <p><strong>Accuracy:</strong> 85.86%</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color: #00d4ff; margin-top: 0;">🌲 Random Forest</h4>
            <p>An ensemble of decision trees that reduces overfitting through averaging multiple predictions.</p>
            <p><strong>Accuracy:</strong> 84.23%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="glass-card">
            <h4 style="color: #00d4ff; margin-top: 0;">🎯 Support Vector Machine</h4>
            <p>Finds optimal hyperplane to separate classes. Effective in high-dimensional spaces.</p>
            <p><strong>Accuracy:</strong> 84.22%</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="glass-card">
            <h4 style="color: #00ff88; margin-top: 0;">⚡ Grid Search Random Forest</h4>
            <p>Optimized Random Forest with hyperparameter tuning. Best overall performance.</p>
            <p><strong>Accuracy:</strong> 89.75%</p>
        </div>
        ''', unsafe_allow_html=True)