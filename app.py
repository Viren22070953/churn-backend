from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import logging
import numpy as np
import os
from pymongo import MongoClient
from flask_cors import CORS
from io import BytesIO
from dotenv import load_dotenv

# ------------------ Load environment variables ------------------ #
load_dotenv()

print("🚀 Flask app.py started...")

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://churn-frontend.onrender.com"], supports_credentials=True)

# ------------------ Logger ------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ MongoDB Connection ------------------ #
try:
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("⚠️ MONGO_URI environment variable not found")

    client = MongoClient(MONGO_URI)
    db = client["ChurnDB"]
    predictions_collection = db["Predictions"]
    logger.info("✅ Connected to MongoDB Atlas successfully")

except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {str(e)}")
    predictions_collection = None

# ------------------ Load Model & Scaler ------------------ #
MODEL_PATH = os.path.join('Models', 'model.pkl')
SCALER_PATH = os.path.join('Models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info("✅ Model and Scaler loaded successfully")

# ------------------ Feature Setup ------------------ #
genre_map = {"Action": 1, "Adventure": 2, "Puzzle": 3, "Strategy": 4}
difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
engagement_map = {"Low": 1, "Medium": 2, "High": 3}
contract_map = {"Monthly": 1, "Yearly": 2}

full_features = [
    "CreditScore", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited", "GameGenre", "GameDifficulty", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
    "EngagementLevel", "Subscription_Length_Months", "Monthly_Bill",
    "Contract", "MonthlyCharges", "TotalCharges", "tenure"
]

# ------------------ Prediction Route ------------------ #
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        all_results = []

        # Read CSV in chunks for efficiency
        chunks = pd.read_csv(file, chunksize=100)
        for chunk in chunks:
            df_chunk = chunk.copy()

            # ✅ Encode categorical columns
            if "GameGenre" in df_chunk.columns:
                df_chunk["GameGenre"] = df_chunk["GameGenre"].map(genre_map).fillna(0)
            if "GameDifficulty" in df_chunk.columns:
                df_chunk["GameDifficulty"] = df_chunk["GameDifficulty"].map(difficulty_map).fillna(0)
            if "EngagementLevel" in df_chunk.columns:
                df_chunk["EngagementLevel"] = df_chunk["EngagementLevel"].map(engagement_map).fillna(0)
            if "Contract" in df_chunk.columns:
                df_chunk["Contract"] = df_chunk["Contract"].map(contract_map).fillna(0)

            # ✅ Ensure all expected columns exist
            for col in full_features:
                if col not in df_chunk.columns:
                    df_chunk[col] = 0

            # ✅ Keep column order consistent
            df_chunk = df_chunk[full_features]

            # ✅ Convert everything to numeric
            df_chunk = df_chunk.apply(pd.to_numeric, errors='coerce').fillna(0)

            # ✅ Scale and predict
            X_scaled = scaler.transform(df_chunk)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            # ✅ Prepare result chunk
            chunk_results = df_chunk.copy()
            chunk_results['Prediction'] = preds
            chunk_results['Churn_Probability'] = probs
            all_results.append(chunk_results)

        # ✅ Combine all chunks
        final_results = pd.concat(all_results, ignore_index=True)

        # ✅ Save CSV output
        output_csv = "/tmp/predictions_output.csv"
        final_results.to_csv(output_csv, index=False)

        # ✅ Optional: Log predictions to MongoDB
        if predictions_collection is not None:
            logs = []
            for i, row in final_results.iterrows():
                logs.append({
                    "CustomerID": row.get('CustomerID', f"Chunk_{i}"),
                    "prediction_result": int(row['Prediction']),
                    "churn_probability": round(float(row['Churn_Probability']), 2)
                })
            if logs:
                predictions_collection.insert_many(logs)

        return send_file(output_csv, as_attachment=True)

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ------------------ Run Flask Server ------------------ #
if __name__ == "__main__":
    print("🔥 Flask backend running locally at: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
