import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model_utils import (
    predict_liver_disease, 
    get_feature_ranges,
    get_feature_descriptions
)

# Set page config
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .stTextInput>div>div>input {
        padding: 10px;
    }
    .stNumberInput>div>div>input {
        padding: 10px;
    }
    .stRadio>div>div {
        padding: 10px;
    }
    .stCheckbox>div>div {
        padding: 10px;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-help {
        color: #666;
        font-size: 0.8em;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for patient data
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

# Title and description
st.title("🏥 Liver Disease Prediction System")
st.markdown("""
### Welcome to the Liver Disease Prediction System
This application helps medical professionals predict liver disease stages based on patient data.
Please fill in the patient's information below to get a prediction.
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses machine learning to predict liver disease stages based on:
    
    1. **Basic Information**
       - Age and Sex
    
    2. **Laboratory Results**
       - Albumin
       - Bilirubin
       - Liver enzymes (ALT, AST, ALP)
       - Blood tests (INR, Platelets, Sodium, Creatinine)
    
    3. **Clinical Signs**
       - Presence of Ascites
       - Hepatomegaly
       - Spider Angiomas
       - Edema
    
    The system provides both a predicted stage and confidence levels for each possible stage.
    """)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "🔬 Lab Results", "🏥 Clinical Signs"])

# Get feature descriptions
feature_descriptions = get_feature_descriptions()

# Tab 1: Basic Information
with tab1:
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.patient_data['Age'] = st.number_input(
            "Age",
            min_value=18,
            max_value=90,
            value=50,
            help=feature_descriptions['Age']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Age']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.session_state.patient_data['Sex'] = st.radio(
            "Sex",
            options=["M", "F"],
            horizontal=True,
            help=feature_descriptions['Sex']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Sex']}</div>", unsafe_allow_html=True)

# Tab 2: Laboratory Results
with tab2:
    st.subheader("Laboratory Test Results")
    
    # First row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.patient_data['Albumin'] = st.number_input(
            "Albumin (g/dL)",
            min_value=2.0,
            max_value=6.0,
            value=4.0,
            step=0.1,
            help=feature_descriptions['Albumin']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Albumin']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.session_state.patient_data['Bilirubin'] = st.number_input(
            "Bilirubin (mg/dL)",
            min_value=0.3,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help=feature_descriptions['Bilirubin']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Bilirubin']}</div>", unsafe_allow_html=True)
    
    with col3:
        st.session_state.patient_data['ALT'] = st.number_input(
            "ALT (U/L)",
            min_value=7,
            max_value=2000,
            value=30,
            help=feature_descriptions['ALT']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['ALT']}</div>", unsafe_allow_html=True)

    # Second row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.patient_data['AST'] = st.number_input(
            "AST (U/L)",
            min_value=10,
            max_value=2000,
            value=30,
            help=feature_descriptions['AST']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['AST']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.session_state.patient_data['ALP'] = st.number_input(
            "ALP (U/L)",
            min_value=44,
            max_value=500,
            value=100,
            help=feature_descriptions['ALP']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['ALP']}</div>", unsafe_allow_html=True)
    
    with col3:
        st.session_state.patient_data['INR'] = st.number_input(
            "INR",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help=feature_descriptions['INR']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['INR']}</div>", unsafe_allow_html=True)

    # Third row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.patient_data['Platelets'] = st.number_input(
            "Platelets (×10⁹/L)",
            min_value=20,
            max_value=500,
            value=150,
            help=feature_descriptions['Platelets']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Platelets']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.session_state.patient_data['Sodium'] = st.number_input(
            "Sodium (mEq/L)",
            min_value=125,
            max_value=145,
            value=135,
            help=feature_descriptions['Sodium']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Sodium']}</div>", unsafe_allow_html=True)
    
    with col3:
        st.session_state.patient_data['Creatinine'] = st.number_input(
            "Creatinine (mg/dL)",
            min_value=0.5,
            max_value=4.0,
            value=1.0,
            step=0.1,
            help=feature_descriptions['Creatinine']
        )
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Creatinine']}</div>", unsafe_allow_html=True)

# Tab 3: Clinical Signs
with tab3:
    st.subheader("Clinical Signs")
    st.markdown("Please check all that apply:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.patient_data['Ascites'] = int(st.checkbox(
            "Ascites",
            help=feature_descriptions['Ascites']
        ))
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Ascites']}</div>", unsafe_allow_html=True)
        
        st.session_state.patient_data['Hepatomegaly'] = int(st.checkbox(
            "Hepatomegaly",
            help=feature_descriptions['Hepatomegaly']
        ))
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Hepatomegaly']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.session_state.patient_data['Spiders'] = int(st.checkbox(
            "Spider Angiomas",
            help=feature_descriptions['Spiders']
        ))
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Spiders']}</div>", unsafe_allow_html=True)
        
        st.session_state.patient_data['Edema'] = int(st.checkbox(
            "Edema",
            help=feature_descriptions['Edema']
        ))
        st.markdown(f"<div class='feature-help'>{feature_descriptions['Edema']}</div>", unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# Predict button
if st.button("🔍 Predict Disease Stage", use_container_width=True):
    try:
        # Get prediction
        predicted_stage, probabilities = predict_liver_disease(st.session_state.patient_data)
        
        # Display results in a nice format
        st.markdown("## 📊 Prediction Results")
        
        # Create columns for results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("### Predicted Stage")
            st.markdown(
                f"<div class='result-box'><h2 style='color: #0066cc; text-align: center;'>{predicted_stage}</h2></div>",
                unsafe_allow_html=True
            )
        
        with res_col2:
            st.markdown("### Confidence Levels")
            # Create probability chart
            fig = go.Figure(data=[go.Bar(
                x=list(probabilities.keys()),
                y=list(probabilities.values()),
                marker_color='#0066cc'
            )])
            fig.update_layout(
                title="Probability Distribution",
                xaxis_title="Disease Stage",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed probabilities
        st.markdown("### Detailed Analysis")
        prob_df = pd.DataFrame({
            'Stage': probabilities.keys(),
            'Probability': [f"{v:.1%}" for v in probabilities.values()]
        })
        st.table(prob_df)
        
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ Bishanka | Bijendra") 