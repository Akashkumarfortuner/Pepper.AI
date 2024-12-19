import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load saved models and related objects
age_model_path = r"C:\Users\HP\OneDrive\Desktop\final_pepper\models\age_prediction_model.pkl"
yield_model_path = r"C:\Users\HP\OneDrive\Desktop\final_pepper\models\yield_prediction_model.pkl"
scaler_path = r"C:\Users\HP\OneDrive\Desktop\final_pepper\models\scaler.pkl"
encoder_path = r"C:\Users\HP\OneDrive\Desktop\final_pepper\models\label_encoder.pkl"

# Load models and utilities
age_model = joblib.load(age_model_path)
yield_model = joblib.load(yield_model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# Streamlit app title
st.title("Pepper Age and Yield Prediction")
st.markdown("**Predict the Age and Yield of black pepper plants based on morphological features.**")

# Sidebar inputs for the user
st.sidebar.header("Input Plant Parameters")
plant_height = st.sidebar.number_input("Plant Height (m)", min_value=0.0, max_value=10.0, step=0.1, value=2.5)
stem_width = st.sidebar.number_input("Stem Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
leaf_length = st.sidebar.number_input("Leaf Length (cm)", min_value=0.0, max_value=100.0, step=0.1, value=15.0)
leaf_color = st.sidebar.selectbox("Leaf Color", label_encoder.classes_)

# Process user input
leaf_color_encoded = label_encoder.transform([leaf_color])[0]
user_input = pd.DataFrame([{
    'Plant Height (m)': plant_height,
    'Stem Width (cm)': stem_width,
    'Leaf Length (cm)': leaf_length,
    'Leaf Color': leaf_color_encoded
}])

# Scale the input data using the pre-fitted scaler
user_input_scaled = scaler.transform(user_input)

# Predict Age and Yield
predicted_age = age_model.predict(user_input_scaled)[0]
predicted_yield = yield_model.predict(user_input_scaled)[0]

# Display predictions
st.subheader("Predicted Results")
st.write(f"**Predicted Age:** {predicted_age:.2f} years")
st.write(f"**Predicted Yield:** {predicted_yield:.2f} kg")

# Visualize predictions
st.subheader("Prediction Visualization")
fig, ax = plt.subplots()
bars = ax.bar(["Age (years)", "Yield (kg)"], [predicted_age, predicted_yield], color=['blue', 'green'])
ax.bar_label(bars, fmt='%.2f', padding=3)
plt.ylabel("Values")
plt.title("Predicted Age and Yield")
st.pyplot(fig)
