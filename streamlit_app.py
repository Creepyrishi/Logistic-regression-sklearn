import streamlit as st
import pandas as pd
import joblib

# Set page title
st.title('Model Prediction App')

# Add description
st.write('This app predicts using our trained model')

# Create input form
st.header('Input Features')

# Add input fields
# Note: Update these fields based on your model's actual features
col1, col2, col3 = st.columns(3)
with col1:
    radius_mean = st.number_input('Radius Mean', value=0.0)
    texture_mean = st.number_input('Texture Mean', value=0.0)
    perimeter_mean = st.number_input('Perimeter Mean', value=0.0)
    area_mean = st.number_input('Area Mean', value=0.0)
    smoothness_mean = st.number_input('Smoothness Mean', value=0.0)
    compactness_mean = st.number_input('Compactness Mean', value=0.0)
    concavity_mean = st.number_input('Concavity Mean', value=0.0)
    concave_points_mean = st.number_input('Concave Points Mean', value=0.0)
    symmetry_mean = st.number_input('Symmetry Mean', value=0.0)
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean', value=0.0)

with col2:
    radius_se = st.number_input('Radius SE', value=0.0)
    texture_se = st.number_input('Texture SE', value=0.0)
    perimeter_se = st.number_input('Perimeter SE', value=0.0)
    area_se = st.number_input('Area SE', value=0.0)
    smoothness_se = st.number_input('Smoothness SE', value=0.0)
    compactness_se = st.number_input('Compactness SE', value=0.0)
    concavity_se = st.number_input('Concavity SE', value=0.0)
    concave_points_se = st.number_input('Concave Points SE', value=0.0)
    symmetry_se = st.number_input('Symmetry SE', value=0.0)
    fractal_dimension_se = st.number_input('Fractal Dimension SE', value=0.0)

with col3:
    radius_worst = st.number_input('Radius Worst', value=0.0)
    texture_worst = st.number_input('Texture Worst', value=0.0)
    perimeter_worst = st.number_input('Perimeter Worst', value=0.0)
    area_worst = st.number_input('Area Worst', value=0.0)
    smoothness_worst = st.number_input('Smoothness Worst', value=0.0)
    compactness_worst = st.number_input('Compactness Worst', value=0.0)
    concavity_worst = st.number_input('Concavity Worst', value=0.0)
    concave_points_worst = st.number_input('Concave Points Worst', value=0.0)
    symmetry_worst = st.number_input('Symmetry Worst', value=0.0)
    fractal_dimension_worst = st.number_input('Fractal Dimension Worst', value=0.0)

# Create a button to make predictions
if st.button('Predict'):
    try:
        # Load the model and scaler
        model = joblib.load('breast_cancer_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Create input array with all 30 features
        input_data = pd.DataFrame([[
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
            fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se, concave_points_se,
            symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
            perimeter_worst, area_worst, smoothness_worst, compactness_worst,
            concavity_worst, concave_points_worst, symmetry_worst,
            fractal_dimension_worst
        ]], columns=[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst',
            'symmetry_worst', 'fractal_dimension_worst'
        ])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Display prediction with label
        result = "Malignant" if prediction[0] == 1 else "Benign"
        st.success(f'Prediction: {result}')
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')

# Add some information about the model
st.sidebar.header('About')
st.sidebar.write('This is a machine learning model deployed using Streamlit.')
st.sidebar.write('Enter the feature values and click Predict to get predictions.')
