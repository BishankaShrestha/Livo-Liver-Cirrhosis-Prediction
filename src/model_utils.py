import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Union, List
import streamlit as st
from pathlib import Path

# Cache the model loading to improve performance
@st.cache_resource
def load_model() -> Union[Pipeline, None]:
    """
    Load the trained model from disk with caching for better performance

    Returns:
        Union[Pipeline, None]: Loaded model or None if loading fails
    """
    try:
        # Use the absolute path relative to this script’s location
        model_path = Path(__file__).resolve().parent / 'liver_disease_staging_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def validate_input_data(patient_data: Dict) -> Tuple[bool, str]:
    """
    Validate input data against expected ranges and types

    Args:
        patient_data (Dict): Patient data to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        required_fields = {
            'Age', 'Sex', 'Albumin', 'Bilirubin', 'ALT', 'AST', 'ALP',
            'INR', 'Platelets', 'Sodium', 'Creatinine', 'Ascites',
            'Hepatomegaly', 'Spiders', 'Edema'
        }

        missing_fields = required_fields - set(patient_data.keys())
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        ranges = get_feature_ranges()
        for field, (min_val, max_val) in ranges.items():
            if field in patient_data:
                value = patient_data[field]
                if not isinstance(value, (int, float)):
                    return False, f"{field} must be a number"
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"

        if patient_data['Sex'] not in ['M', 'F']:
            return False, "Sex must be either 'M' or 'F'"

        boolean_fields = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        for field in boolean_fields:
            if patient_data[field] not in [0, 1]:
                return False, f"{field} must be 0 or 1"

        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def predict_liver_disease(patient_data: Dict) -> Tuple[str, Dict[str, float]]:
    """
    Predict liver disease stage for a patient with input validation

    Args:
        patient_data (Dict): Dictionary containing patient information

    Returns:
        Tuple[str, Dict[str, float]]: (predicted_stage, probabilities_dict)

    Raises:
        ValueError: If input validation fails
        RuntimeError: If model prediction fails
    """
    is_valid, error_message = validate_input_data(patient_data)
    if not is_valid:
        raise ValueError(error_message)

    model = load_model()
    if model is None:
        raise RuntimeError("Failed to load model")

    try:
        df = pd.DataFrame([patient_data])
        predicted_stage = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        stage_names = model.classes_
        prob_dict = {stage: float(prob) for stage, prob in zip(stage_names, probabilities)}
        return predicted_stage, prob_dict
    except Exception as e:
        raise RuntimeError(f"Prediction error: {str(e)}")

def get_feature_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Return recommended ranges for numerical features

    Returns:
        Dict[str, Tuple[float, float]]: Dictionary of feature ranges
    """
    return {
        'Age': (18, 90),
        'Albumin': (2.0, 6.0),
        'Bilirubin': (0.3, 10.0),
        'ALT': (7, 2000),
        'AST': (10, 2000),
        'ALP': (44, 500),
        'INR': (0.5, 5.0),
        'Platelets': (20, 500),
        'Sodium': (125, 145),
        'Creatinine': (0.5, 4.0)
    }

def get_feature_descriptions() -> Dict[str, str]:
    """
    Return descriptions and normal ranges for features

    Returns:
        Dict[str, str]: Dictionary of feature descriptions
    """
    return {
        'Age': "Patient's age in years",
        'Sex': "Patient's biological sex (M/F)",
        'Albumin': "Serum albumin level (Normal: 3.5-5.5 g/dL)",
        'Bilirubin': "Total bilirubin level (Normal: 0.3-1.2 mg/dL)",
        'ALT': "Alanine aminotransferase (Normal: 7-56 U/L)",
        'AST': "Aspartate aminotransferase (Normal: 10-40 U/L)",
        'ALP': "Alkaline phosphatase (Normal: 44-147 U/L)",
        'INR': "International normalized ratio (Normal: 0.8-1.1)",
        'Platelets': "Platelet count (Normal: 150-450 ×10⁹/L)",
        'Sodium': "Serum sodium level (Normal: 135-145 mEq/L)",
        'Creatinine': "Serum creatinine (Normal: 0.7-1.3 mg/dL)",
        'Ascites': "Accumulation of fluid in the peritoneal cavity",
        'Hepatomegaly': "Enlarged liver",
        'Spiders': "Spider angiomas (spider-like blood vessels)",
        'Edema': "Swelling caused by fluid retention"
    }
