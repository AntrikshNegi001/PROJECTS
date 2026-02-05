# 🌸 Iris Flower Classification Project

### "The 'Hello World' of Machine Learning"

This is a Machine Learning web application built with **Python** and **Streamlit**. It predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) based on its physical measurements.

I built this project to understand the fundamentals of classification algorithms and how to deploy a model into a user-friendly web interface.

---

## 🧐 What does this app do?

Imagine you are a botanist finding an Iris flower in the wild. You measure its petals and sepals, but you aren't sure exactly which species it is.

This app acts as your **digital assistant**:
1.  **Input:** You adjust the sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width.
2.  **Process:** The app feeds these numbers into a **Random Forest Classifier** (a powerful machine learning algorithm).
3.  **Output:** It instantly predicts the species of the flower with a high degree of confidence.

---

## 📊 Data Analysis (Jupyter Notebook)

I have included a Jupyter Notebook (`iris_notebook.ipynb`) in this repository to demonstrate the data science workflow behind the app.

Inside the notebook, you will find:
* **Exploratory Data Analysis (EDA):** Visualizing how the three species are grouped using **Seaborn** pairplots.
* **Model Evaluation:** Splitting the data into training (80%) and testing (20%) sets to calculate the model's accuracy.
* **Feature Engineering:** Handling dataset column names dynamically to prevent errors.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Frontend:** Streamlit (for the web interface)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Dataset:** [Iris Species Dataset from Kaggle](https://www.kaggle.com/uciml/iris)

---

## 🚀 How to Run this Project

If you want to try this out on your local machine, follow these steps:

**1. Clone the Repository (or download the files)**
Ensure you have the project folder `IRIS_CLASSIFICATION_PROJECT` on your computer.

**2. Install Requirements**
Open your terminal and run:
```bash
pip install streamlit pandas scikit-learn seaborn matplotlib
pip install streamlit pandas scikit-learn 

**3. Run the App Use the following command from your main terminal**
streamlit run "IRIS_CLASSIFICATION_PROJECT\iris.py"

**Project Structure**
Here is how the files are organized:
IRIS_CLASSIFICATION_PROJECT/
├── iris.py              # The main application code (Frontend + Model)
├── iriss.ipynb  # Jupyter Notebook for Analysis & Testing
├── IRIS.csv             # The dataset used to train the model
└── iris.md              # This documentation file

Created by :: Antriksh Negi