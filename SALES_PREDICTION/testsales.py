import joblib
import pandas as pd
import os

def predict_new_campaign():
    print("🚀 Starting Sales Prediction System...")

    # --- 1. Robust Path Handling (The Fix) ---
    # Get the folder where THIS script is located
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path to the saved model file
    model_path = os.path.join(current_folder, 'sales_model.pkl')

    # --- 2. Load the Model ---
    try:
        model = joblib.load(model_path)
        print("✅ Model Loaded Successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find model at {model_path}")
        print("   -> Did you run 'main.py' first to train and save the model?")
        return

    # --- 3. Get User Input ---
    print("\n--- Enter Advertising Budget ($) ---")
    try:
        tv_spend = float(input("📺 TV Budget:        $"))
        radio_spend = float(input("📻 Radio Budget:     $"))
        news_spend = float(input("📰 Newspaper Budget: $"))
    except ValueError:
        print("❌ Error: Please enter valid numbers only.")
        return

    # --- 4. Prepare Data for Prediction ---
    # The feature names must match what we used during training
    campaign_data = pd.DataFrame(
        [[tv_spend, radio_spend, news_spend]], 
        columns=['TV', 'Radio', 'Newspaper']
    )

    # --- 5. Make Prediction ---
    predicted_sales = model.predict(campaign_data)

    # --- 6. Show Result ---
    print("\n" + "="*30)
    print(f"💰 ESTIMATED SALES: {predicted_sales[0]:.2f} Units")
    print("="*30 + "\n")

if __name__ == "__main__":
    predict_new_campaign()