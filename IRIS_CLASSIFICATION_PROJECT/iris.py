import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Page Configuration
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")

st.write("""
# 🌸 Iris Flower Species Predictor
This app predicts the **Iris flower species** based on the measurements you provide!
""")

# 2. Load the Data (The "Bulletproof" Way)
# This finds the directory where 'iris.py' is sitting right now
current_dir = os.path.dirname(os.path.abspath(__file__))

# This combines that directory with the filename to get the full path
file_path = os.path.join(current_dir, 'IRIS.csv')

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"⚠️ Error: Could not find file at: {file_path}")
    st.stop()

# 3. Data Preprocessing
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. Train the Model
clf = RandomForestClassifier()
clf.fit(X, y)

# 5. Sidebar for User Input
st.sidebar.header('Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        X.columns[0]: sepal_length,
        X.columns[1]: sepal_width,
        X.columns[2]: petal_length,
        X.columns[3]: petal_width
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 6. Display User Input
st.subheader('User Input parameters')
st.write(input_df)

# 7. Make Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# 8. Show Results
st.subheader('Prediction')
st.markdown("---")

species_type = prediction[0]

if "setosa" in species_type.lower():
    st.success(f"**Species: {species_type}** 🌿")
elif "versicolor" in species_type.lower():
    st.success(f"**Species: {species_type}** 🌸")
elif "virginica" in species_type.lower():
    st.success(f"**Species: {species_type}** 🌺")
else:
    st.success(f"**Species: {species_type}**")

st.subheader('Prediction Probability')
st.write(prediction_proba)