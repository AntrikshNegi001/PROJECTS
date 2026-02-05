import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#FILE PATH
filepath = 'MOVIE_RATING_001/data/raw/movies.csv'

try:
    print(f"Reading file from: {filepath} ...")
    df = pd.read_csv(filepath, encoding='latin-1')
    print("File mil gayi! Cleaning start kar rahe hain...\n")

    # --- DATA CLEANING ---
    df.dropna(subset=['Rating'], inplace=True)
    df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True).astype(float)
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
    
    print(f"Success! Data clean ho gaya.")
    print(f"Total Movies ab bachi hain: {df.shape[0]}")

    # --- GRAPH  ---
    print("\nGraph open ho raha hai... (Nayi window dekho)")
    plt.figure(figsize=(10, 6))
    
    # Histogram Of Movie Ratings
    sns.histplot(df['Rating'], bins=20, kde=True, color='dodgerblue')
    
    plt.title('Movie Ratings Distribution')
    plt.xlabel('Rating (1-10)')
    plt.ylabel('Movies ki Ginti')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # command for opening window of graph
    plt.show() 
    # Filling missing Duration values
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())
    print(f"\nDuration Fixed! Ab missing values hain: {df['Duration'].isnull().sum()}")

    top_genres = df['Genre'].value_counts().head(10).index
    genre_data = df[df['Genre'].isin(top_genres)]

    print("\nGenre Graph open ho raha hai... (Boxplot)")
    plt.figure(figsize=(12, 6))
    
    # Making Boxplot for Top 10 Genres vs Ratings
    sns.boxplot(data=genre_data, x='Genre', y='Rating', hue='Genre', legend=False, palette='Set2')
    plt.title('Top 10 Genres vs Movie Ratings')
    plt.xticks(rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.show()

    # ---  CORRELATION ---
    # Taking only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    # --- FEATURE ENGINEERING  ---
    print("\nTraining ke liye data taiyaar kar rahe hain... (Encoding)")

    # 1. Genre ko Number banana
    genre_mean_rating = df.groupby('Genre')['Rating'].transform('mean')
    df['Genre_Encoded'] = genre_mean_rating

    # 2. Director ko Number banana
    director_mean_rating = df.groupby('Director')['Rating'].transform('mean')
    df['Director_Encoded'] = director_mean_rating

    # 3. Actors ko Number banana
    actor1_mean_rating = df.groupby('Actor 1')['Rating'].transform('mean')
    df['Actor1_Encoded'] = actor1_mean_rating

    actor2_mean_rating = df.groupby('Actor 2')['Rating'].transform('mean')
    df['Actor2_Encoded'] = actor2_mean_rating

    actor3_mean_rating = df.groupby('Actor 3')['Rating'].transform('mean')
    df['Actor3_Encoded'] = actor3_mean_rating

    print("Encoding Complete! Text data ab Numbers ban gaya hai.")
    
  
    cols_to_fix = ['Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded', 'Actor2_Encoded', 'Actor3_Encoded', 'Duration', 'Votes']

    for col in cols_to_fix:
        df[col] = df[col].fillna(df['Rating'].mean())

    print(" Saare Missing Values fix ho gaye!")
    # --- STEP 8: MODEL TRAINING  ---
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    print("\nModel Train ho raha hai... (Thoda wait karo)")

    # Input Data (X) 
    X = df[['Year', 'Duration', 'Votes', 'Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded', 'Actor2_Encoded', 'Actor3_Encoded']]
    
    # Output Data (y) 
    y = df['Rating']

    # Dividing data into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Banana (Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Testing the Model
    y_pred = model.predict(X_test)

    # Checking the Accuracy
    accuracy = r2_score(y_test, y_pred)
    print(f"\n Model Training Complete!")
    print(f"Model Accuracy (R2 Score): {accuracy*100:.2f}%")

    # --- STEP 9: PREDICTION SYSTEM  ---
    print("\n--- TEST YOUR MOVIE ---")
    sample_movie = pd.DataFrame([[2024, 150, 5000, 7.5, 8.0, 7.0, 7.5, 7.0]], 
                                columns=['Year', 'Duration', 'Votes', 'Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded', 'Actor2_Encoded', 'Actor3_Encoded'])
    
    predicted_rating = model.predict(sample_movie)
    print(f"Example Prediction: Agar ek movie 2024 mein aaye, achi starcast ke saath...")
    print(f"To Model kehta hai uski rating hogi: ⭐ {predicted_rating[0]:.1f}/10")

    
    plt.show()
    # ---  INTERACTIVE PREDICTION ---
    print("\n" + "="*30)
    print(" MOVIE RATING PREDICTOR APP  ")
    print("="*30)

    # 1. Encoding Maps banana
    genre_map = df.groupby('Genre')['Rating'].mean().to_dict()
    director_map = df.groupby('Director')['Rating'].mean().to_dict()
    actor1_map = df.groupby('Actor 1')['Rating'].mean().to_dict()
    overall_mean = df['Rating'].mean()

    # 2. Taking User Input
    try:
        user_year = float(input("Year (e.g., 2025): "))
        user_dur = float(input("Duration (Minutes, e.g., 120): "))
        user_votes = float(input("Votes (Expected, e.g., 5000): "))
        
        user_genre = input("Genre (e.g., Drama, Action, Comedy): ")
        user_director = input("Director Name (e.g., Rajkumar Hirani): ")
        user_actor = input("Main Actor Name (e.g., Aamir Khan): ")

        # 3. Converting input into numbers(Encoding)
        
        gen_score = genre_map.get(user_genre, overall_mean)
        dir_score = director_map.get(user_director, overall_mean)
        act_score = actor1_map.get(user_actor, overall_mean)

        # 4. Prediction karna
        user_data = pd.DataFrame([[user_year, user_dur, user_votes, gen_score, dir_score, act_score, act_score, act_score]],
                                 columns=['Year', 'Duration', 'Votes', 'Genre_Encoded', 'Director_Encoded', 'Actor1_Encoded', 'Actor2_Encoded', 'Actor3_Encoded'])
        
        prediction = model.predict(user_data)[0]
        
        print(f"\nPREDICTION: Ye movie {prediction:.1f}/10 rating layegi!")
        
        if prediction > 8.0:
            print(" Verdict: BLOCKBUSTER! (Superhit)")
        elif prediction > 6.0:
            print(" Verdict: Average / One-time Watch")
        else:
            print(" Verdict: FLOP! Paisa barbad.")

    except ValueError:
        print("Error: Please Enter Numbers Carefully!")

    # Graph window rokne ke liye
    plt.show()
except FileNotFoundError:
    print("ERROR: File Still Not Found!")
    print(f"Check the file path {filepath}")
    