# Movie Rating Prediction System

Welcome to my Machine Learning project.

The goal of this project is to predict the success of a movie before its release. I built an AI tool that analyzes factors like Genre, Director, and Actors to estimate the movie's IMDb rating.

This repository contains two main components:
1. **Web Application:** A user-friendly interface to test predictions instantly.
2. **Jupyter Notebook:** A detailed step-by-step analysis, including data cleaning, visualization, and model training.

## Key Features
* **Predicts Ratings:** Users can enter movie details, and the system predicts a rating (e.g., 7.5/10).
* **Verdict System:** Categorizes movies as "Hit," "Average," or "Flop."
* **Exploratory Data Analysis (EDA):** The notebook contains graphs showing how directors and genres impact ratings.
* **Hybrid Approach:** Includes both a deployment-ready web app and a research-focused notebook.

## Technology Stack
* **Python:** Core programming language.
* **Streamlit:** Used for the Web App interface.
* **Jupyter Notebook:** Used for data analysis and visualization.
* **Scikit-Learn:** Machine Learning (Linear Regression).
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Graphing and plotting.

## Project Structure
```text
MOVIE_RATING_001/
│
├── data/
│   └── raw/
│       └── movies.csv          # Dataset file
├── app.py                      # Main Web Application
├── app2.ipynb  # Jupyter Notebook (Analysis & Code)
├── requirements.txt            # List of dependencies
└── README.md                   # Project Documentation

## How to Run This Project

Follow these steps to run the application on your local machine:

1. **Install Requirements**
   Open your terminal in the project folder and run the following command to install the necessary libraries:
   
   pip install -r requirements.txt

2. **Run the Application**
   Start the web interface by running this command:
   streamlit run MOVIE_RATING_001\app.py
3. **Usage**
   A browser window will open automatically. You can enter details such as the Year, Genre, Director, and Lead Actor to see the predicted rating.

## Dataset Information
The model is trained on a dataset of Indian movies containing information on release years, duration, votes, and cast details.


Created by :: Antriksh Negi