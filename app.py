

# from flask import Flask, request, jsonify, send_file
# import joblib
# import pandas as pd
# import logging
# import numpy as np
# import os
# from pymongo import MongoClient  
# from flask_cors import CORS  
# from io import BytesIO
# from dotenv import load_dotenv

# load_dotenv()


# print("🚀 Flask app.py started...")

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:5173", "https://churnfrontend.vercel.app"], supports_credentials=True)

# # ------------------ Logger ------------------ #
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------ DB Connection ------------------ #


# try:
#     MONGO_URI = os.getenv("MONGO_URI")  # Get from environment variable

#     if not MONGO_URI:
#         raise ValueError("⚠️ MONGO_URI environment variable not found")

#     client = MongoClient(MONGO_URI)
#     db = client["ChurnDB"]
#     predictions_collection = db["Predictions"]
#     logger.info("✅ Connected to MongoDB Atlas successfully")

# except Exception as e:
#     logger.error(f"❌ MongoDB connection failed: {str(e)}")
#     predictions_collection = None


# # ------------------ Load Model & Scaler ------------------ #
# MODEL_PATH = os.path.join('Models', 'model.pkl')
# SCALER_PATH = os.path.join('Models', 'scaler.pkl')

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# logger.info("✅ Model and Scaler loaded successfully")

# # ------------------ Feature Setup ------------------ #
# genre_map = {"Action": 1, "Adventure": 2, "Puzzle": 3, "Strategy": 4}
# difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
# engagement_map = {"Low": 1, "Medium": 2, "High": 3}
# contract_map = {"Monthly": 1, "Yearly": 2}  # <-- Encode Contract

# full_features = [
#     "CreditScore", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
#     "EstimatedSalary", "Exited", "GameGenre", "GameDifficulty", "SessionsPerWeek",
#     "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
#     "EngagementLevel", "Subscription_Length_Months", "Monthly_Bill",
#     "Contract", "MonthlyCharges", "TotalCharges", "tenure"
# ]

# # ------------------ Batch Prediction Route ------------------ #
# @app.route('/predict-batch', methods=['POST'])
# def predict_batch():
#     try:
#         logger.info("🔥 Received batch CSV prediction request")

#         if 'file' not in request.files:
#             return jsonify({"error": "CSV file not provided"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "Empty file name"}), 400

#         df = pd.read_csv(file)
#         logger.info(f"📊 Uploaded CSV shape: {df.shape}")

#         # Ensure CustomerID exists
#         if 'CustomerID' not in df.columns:
#             df['CustomerID'] = range(1, len(df) + 1)

#         # Map categorical columns
#         if 'GameGenre' in df.columns:
#             df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
#         if 'GameDifficulty' in df.columns:
#             df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
#         if 'EngagementLevel' in df.columns:
#             df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)
#         if 'Contract' in df.columns:
#             df['Contract'] = df['Contract'].map(contract_map).fillna(0.0)

#         # Ensure all features exist
#         for col in full_features:
#             if col not in df.columns:
#                 df[col] = 0.0

#         df_ordered = df[full_features]

#         # Scale input
#         scaled_input = scaler.transform(df_ordered.to_numpy())

#         # Predictions
#         raw_predictions = model.predict(scaled_input)
#         churn_probabilities = model.predict_proba(scaled_input)

#         # Prepare result DataFrame
#         results = df.copy()
#         results['Prediction'] = ["Churn" if p == 1 else "Stay" for p in raw_predictions]
#         results['Churn_Probability'] = churn_probabilities[:, 1] * 100

#         # Save predictions to DB
#         if predictions_collection is not None:
#             for i, row in results.iterrows():
#                 log_entry = {
#                     "CustomerID": row['CustomerID'],
#                     "input_data": df_ordered.iloc[i].to_dict(),
#                     "prediction_result": row['Prediction'],
#                     "churn_probability": round(row['Churn_Probability'], 2)
#                 }
#                 predictions_collection.insert_one(log_entry)

#         # Save CSV locally
#         output_file_path = "output_predictions.csv"
#         if os.path.exists(output_file_path):
#             results.to_csv(output_file_path, mode='a', index=False, header=False)
#         else:
#             results.to_csv(output_file_path, index=False)

#         # Return CSV to client
#         output = BytesIO()
#         results.to_csv(output, index=False)
#         output.seek(0)

#         return send_file(
#             output,
#             mimetype='text/csv',
#             as_attachment=True,
#             download_name='churn_predictions.csv'
#         )

#     except Exception as e:
#         logger.error(f"❌ Error in batch prediction: {str(e)}", exc_info=True)
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     print("🔥 Starting Flask server on http://127.0.0.1:5000 ...")
#     app.run(host="0.0.0.0", port=5000, debug=True)


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
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

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

# ------------------ Batch Prediction Route ------------------ #
# @app.route('/predict-batch', methods=['POST'])
# def predict_batch():
#     try:
#         logger.info("🔥 Received batch CSV prediction request")

#         if 'file' not in request.files:
#             return jsonify({"error": "CSV file not provided"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "Empty file name"}), 400

#         df = pd.read_csv(file)
#         logger.info(f"📊 Uploaded CSV shape: {df.shape}")

#         # Ensure CustomerID exists
#         if 'CustomerID' not in df.columns:
#             df['CustomerID'] = range(1, len(df) + 1)

#         # Map categorical columns
#         if 'GameGenre' in df.columns:
#             df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
#         if 'GameDifficulty' in df.columns:
#             df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
#         if 'EngagementLevel' in df.columns:
#             df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)
#         if 'Contract' in df.columns:
#             df['Contract'] = df['Contract'].map(contract_map).fillna(0.0)

#         # Ensure all features exist
#         for col in full_features:
#             if col not in df.columns:
#                 df[col] = 0.0

#         df_ordered = df[full_features]

#         # Scale input
#         scaled_input = scaler.transform(df_ordered.to_numpy())

#         # Predictions
#         raw_predictions = model.predict(scaled_input)
#         churn_probabilities = model.predict_proba(scaled_input)

#         # Prepare result DataFrame
#         results = df.copy()
#         results['Prediction'] = ["Churn" if p == 1 else "Stay" for p in raw_predictions]
#         results['Churn_Probability'] = churn_probabilities[:, 1] * 100

#         # ✅ Save predictions to DB (optimized bulk insert)
#         if predictions_collection is not None:
#             logs = []
#             for i, row in results.iterrows():
#                 logs.append({
#                     "CustomerID": row['CustomerID'],
#                     "input_data": df_ordered.iloc[i].to_dict(),
#                     "prediction_result": row['Prediction'],
#                     "churn_probability": round(row['Churn_Probability'], 2)
#                 })
#             if logs:
#                 predictions_collection.insert_many(logs)

#         # Save CSV locally
#         output_file_path = "output_predictions.csv"
#         if os.path.exists(output_file_path):
#             results.to_csv(output_file_path, mode='a', index=False, header=False)
#         else:
#             results.to_csv(output_file_path, index=False)

#         # Return CSV to client
#         output = BytesIO()
#         results.to_csv(output, index=False)
#         output.seek(0)

#         return send_file(
#             output,
#             mimetype='text/csv',
#             as_attachment=True,
#             download_name='churn_predictions.csv'
#         )

#     except Exception as e:
#         logger.error(f"❌ Error in batch prediction: {str(e)}", exc_info=True)
#         return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Temporary output storage
        all_results = []

        # Process CSV in small chunks
        chunks = pd.read_csv(file, chunksize=100)
        for chunk in chunks:
            df_chunk = chunk.copy()
            
            # ✅ Ensure consistent column order for model
            df_chunk = df_chunk.reindex(columns=expected_features, fill_value=0)

            # ✅ Scale input
            X_scaled = scaler.transform(df_chunk)

            # ✅ Predict churn
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            # ✅ Build result DataFrame for current chunk
            chunk_results = df_chunk.copy()
            chunk_results['Prediction'] = preds
            chunk_results['Churn_Probability'] = probs
            all_results.append(chunk_results)

        # ✅ Combine all chunks
        final_results = pd.concat(all_results, ignore_index=True)

        # ✅ Save to DB (optional — can comment out if still memory heavy)
        if predictions_collection is not None:
            logs = []
            for i, row in final_results.iterrows():
                logs.append({
                    "CustomerID": row.get('CustomerID', f"Chunk_{i}"),
                    "input_data": df_chunk.iloc[i % 100].to_dict(),
                    "prediction_result": int(row['Prediction']),
                    "churn_probability": round(float(row['Churn_Probability']), 2)
                })
            if logs:
                predictions_collection.insert_many(logs)

        # ✅ Save final CSV
        output_csv = "/tmp/predictions_output.csv"
        final_results.to_csv(output_csv, index=False)

        return send_file(output_csv, as_attachment=True)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500



# ------------------ Run Flask Server ------------------ #
if __name__ == "__main__":
    print("🔥 Flask backend running locally at: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
