import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP ---
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="centered")

st.title(" Movie Success Predictor")
st.write("Enter Your Movie Details Below to Predict Its Rating! ")
# --- 1. DATA LOAD & TRAIN (Backend) ---
@st.cache_data
def load_and_train_model():
    
    filepath = 'MOVIE_RATING_001/data/raw/movies.csv'
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # --- DATA CLEANING  ---
    df.dropna(subset=['Rating'], inplace=True)
    
    # Removiing parentheses from Year and converting to float
    df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True).astype(float)
    
    # Removing ' min' from Duration and converting to float
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
    
    # Removing commas from Votes and converting to float
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
    
    # Missing Duration ko fill karna
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())

    # --- ENCODING ---
    genre_map = df.groupby('Genre')['Rating'].mean().to_dict()
    director_map = df.groupby('Director')['Rating'].mean().to_dict()
    actor_map = df.groupby('Actor 1')['Rating'].mean().to_dict()
    overall_mean = df['Rating'].mean()
    
    #  Encoding in Dataset 
    df['Genre_Encoded'] = df['Genre'].map(genre_map)
    df['Director_Encoded'] = df['Director'].map(director_map)
    df['Actor1_Encoded'] = df['Actor 1'].map(actor_map)
    
    # Fixing missing values after encoding
    cols_to_fix = ['Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded']
    for col in cols_to_fix:
        df[col] = df[col].fillna(overall_mean)
        
    # --- MODEL TRAINING ---
    X = df[['Year', 'Votes', 'Duration', 'Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded']]
    y = df['Rating']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, genre_map, director_map, actor_map, overall_mean

# Loading data...
with st.spinner('AI Model Is Being trained... Please Wait '):
    model, genre_map, director_map, actor_map, overall_mean = load_and_train_model()

st.success("Model Ready! ")

# --- 2. USER INPUT (Frontend) ---
st.sidebar.header("Movie Details")

# Sliders And Input Boxes
year = st.sidebar.number_input("Release Year", min_value=2000, max_value=2030, value=2026)
duration = st.sidebar.slider("Duration (Minutes)", 60, 240, 120)
votes = st.sidebar.slider("Expected Votes (Popularity)", 1000, 50000, 5000)

st.write("### Starcast & Genre")
col1, col2 = st.columns(2)

with col1:
    genre = st.text_input("Genre (e.g., Drama, Action)", "Drama")
    director = st.text_input("Director Name", "Rajkumar Hirani")

with col2:
    actor = st.text_input("Lead Actor", "Aamir Khan")

# --- 3. PREDICTION LOGIC ---
if st.button("Predict Rating "):
    # Converting inputs into numbers (Encoding)
    gen_score = genre_map.get(genre, overall_mean)
    dir_score = director_map.get(director, overall_mean)
    act_score = actor_map.get(actor, overall_mean)
    
    # Creating DataFrame for model input
    user_data = pd.DataFrame([[year, votes, duration, gen_score, dir_score, act_score]],
                             columns=['Year', 'Votes', 'Duration', 'Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded'])
    
    # Prediction
    prediction = model.predict(user_data)[0]
    
    # Result Display
    st.markdown(f"##  Predicted Rating: **{prediction:.1f}/10**")
    
    if prediction > 8.0:
        st.balloons()
        st.success("Verdict: SUPERHIT / BLOCKBUSTER! ")
    elif prediction > 6.0:
        st.warning("Verdict: Average Movie (One-time watch). ")
    else:
        st.error("Verdict: FLOP! Paisa barbad. ")