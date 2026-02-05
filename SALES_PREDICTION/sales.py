import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_sales_model():
    print("🚀 Starting Model Training...")

    # 1. Setup Paths (The Fix)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_folder, 'advertising.csv')
    model_path = os.path.join(current_folder, 'sales_model.pkl')

    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ Error: Could not find advertising.csv")
        return

    # 3. Train
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Save the Model (Baking the bread!)
    joblib.dump(model, model_path)
    print(f"✅ Success! Model saved to: {model_path}")

if __name__ == "__main__":
    train_sales_model()