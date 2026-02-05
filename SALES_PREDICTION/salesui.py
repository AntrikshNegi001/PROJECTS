import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. Load the Saved Model ---
# We use the same 'bulletproof' path logic to find the model file
current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, 'sales_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found! Please run 'sales.py' first.")
    st.stop()

# --- 2. App Title & Description ---
st.set_page_config(page_title="Sales Predictor", page_icon="💰")
st.title("💰 Sales Prediction Dashboard")
st.markdown("""
This app predicts the **total sales** (in thousands of units) based on advertising expenditure across TV, Radio, and Newspaper.
Adjust the sliders below to see how different budgets affect sales!
""")

st.divider()

# --- 3. Input Section (Sidebar) ---
st.sidebar.header("📝 Advertising Budget ($)")

# Using sliders for interactive input
tv = st.sidebar.slider("TV Advertising", 0.0, 300.0, 150.0)
radio = st.sidebar.slider("Radio Advertising", 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider("Newspaper Advertising", 0.0, 100.0, 10.0)

# Display the inputs as a nice dataframe
input_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper]
})

# --- 4. Prediction Logic ---
if st.button("Predict Sales 🚀"):
    prediction = model.predict(input_data)
    sales_result = prediction[0]
    
    st.subheader("🎯 Prediction Result")
    st.success(f"Estimated Sales: **{sales_result:.2f} units**")
    
    # Optional: Visualization of the budget split
    st.write("---")
    st.write("### 📊 Budget Distribution")
    st.bar_chart(input_data.T) # Transpose to show categories on X-axis

# --- 5. Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Python & Streamlit")