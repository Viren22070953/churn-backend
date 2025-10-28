# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import logging
# import numpy as np
# import xgboost
# import os
# import sys
# from pymongo import MongoClient  
# from flask_cors import CORS  


# app = Flask(__name__)
# CORS(app, origins=["http://localhost:5173", "https://churnfrontend.vercel.app"], supports_credentials=True)



# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)



# MONGO_URI = "mongodb+srv://adityaadlak128:churnlyPass@cluster0.wiiam1r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" 
# client = MongoClient(MONGO_URI)
# db = client["churnDB"]
# predictions_collection = db["predictions"]


# try:
#     client.admin.command('ping')
#     logger.info(" MongoDB connection successful")
# except Exception as e:
#     logger.error(f" MongoDB connection failed: {str(e)}", exc_info=True)



# full_features = [
#     'CreditScore', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#     'EstimatedSalary', 'Exited', 'GameGenre', 'GameDifficulty', 'SessionsPerWeek',
#     'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
#     'EngagementLevel', 'Subscription_Length_Months', 'Monthly_Bill',
#     'Contract', 'MonthlyCharges', 'TotalCharges', 'tenure'
# ]

# genre_map = {"Action": 1.0, "Strategy": 2.0, "Puzzle": 3.0}
# difficulty_map = {"Easy": 1.0, "Medium": 2.0, "Hard": 3.0}
# engagement_map = {"low": 1.0, "medium": 2.0, "high": 3.0}

# logger.info(f"XGBoost version: {xgboost.__version__}")
# logger.info(f"Joblib version: {joblib.__version__}")

# MODEL_PATH = os.path.join('Models', 'model.pkl')
# SCALER_PATH = os.path.join('Models', 'scaler.pkl')

# try:
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     logger.info("✅ Model and scaler loaded successfully")

#     test_input = np.zeros((1, len(full_features)))
#     test_pred = model.predict(test_input)
#     logger.info(f"🔍 Model verification test prediction: {test_pred}")

# except Exception as e:
#     logger.error(f" Error loading model or scaler: {str(e)}", exc_info=True)
#     raise RuntimeError("Failed to load model or scaler") from e


# @app.route('/predict', methods=['POST' , 'GET'])
# def predict():
#     try:
#         logger.info("🔥 Flask server received a request")

#         data = request.get_json()
#         logger.info(f"📦 Parsed JSON data: {data}")

#         sector = data.get("sector")
#         if not sector:
#             return jsonify({"error": "Sector not specified"}), 400

#         logger.info(f"🔍 Sector identified: {sector}")

#         if sector == "Banking":
#             input_data = {
#                 'CreditScore': float(data.get('CreditScore', 0)),
#                 'Balance': float(data.get('Balance', 0)),
#                 'NumOfProducts': float(data.get('NumOfProducts', 0)),
#                 'HasCrCard': float(data.get('HasCrCard', 0)),
#                 'IsActiveMember': float(data.get('IsActiveMember', 0)),
#                 'EstimatedSalary': float(data.get('EstimatedSalary', 0)),
#                 'Exited': float(data.get('Exited', 0)),
#                 'GameGenre': 0.0, 'GameDifficulty': 0.0, 'SessionsPerWeek': 0.0,
#                 'AvgSessionDurationMinutes': 0.0, 'PlayerLevel': 0.0,
#                 'AchievementsUnlocked': 0.0, 'EngagementLevel': 0.0,
#                 'Subscription_Length_Months': 0.0, 'Monthly_Bill': 0.0,
#                 'Contract': 0.0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0, 'tenure': 0.0
#             }

#         elif sector == "Gaming":
#             input_data = {
#                 'CreditScore': 0.0, 'Balance': 0.0, 'NumOfProducts': 0.0,
#                 'HasCrCard': 0.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 0.0,
#                 'Exited': 0.0,
#                 'GameGenre': genre_map.get(data.get('GameGenre', ''), 0.0),
#                 'GameDifficulty': difficulty_map.get(data.get('GameDifficulty', ''), 0.0),
#                 'SessionsPerWeek': float(data.get('SessionsPerWeek', 0)),
#                 'AvgSessionDurationMinutes': float(data.get('AvgSessionDurationMinutes', 0)),
#                 'PlayerLevel': float(data.get('PlayerLevel', 0)),
#                 'AchievementsUnlocked': float(data.get('AchievementsUnlocked', 0)),
#                 'EngagementLevel': engagement_map.get(data.get('EngagementLevel', ''), 0.0),
#                 'Subscription_Length_Months': float(data.get('Subscription_Length_Months', 0)),
#                 'Monthly_Bill': float(data.get('Monthly_Bill', 0)),
#                 'Contract': 0.0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0, 'tenure': 0.0
#             }

#         elif sector == "Telecom":
#             input_data = {
#                 'CreditScore': 0.0, 'Balance': 0.0, 'NumOfProducts': 0.0,
#                 'HasCrCard': 0.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 0.0,
#                 'Exited': float(data.get('Exited', 0)),
#                 'GameGenre': 0.0, 'GameDifficulty': 0.0, 'SessionsPerWeek': 0.0,
#                 'AvgSessionDurationMinutes': 0.0, 'PlayerLevel': 0.0,
#                 'AchievementsUnlocked': 0.0, 'EngagementLevel': 0.0,
#                 'Subscription_Length_Months': 0.0, 'Monthly_Bill': 0.0,
#                 'Contract': float(data.get('Contract', 0)),
#                 'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
#                 'TotalCharges': float(data.get('TotalCharges', 0)),
#                 'tenure': float(data.get('tenure', 0))
#             }
#         else:
#             return jsonify({"error": "Invalid sector specified"}), 400

#         input_df = pd.DataFrame([input_data])
#         logger.info(f"📊 Input DataFrame:\n{input_df}")

#         scaled_input = scaler.transform(input_df.to_numpy())
#         logger.info(f"🔍 Scaled input: {scaled_input}")

#         raw_prediction = model.predict(scaled_input)
#         churn_probabilities = model.predict_proba(scaled_input)
#         churn_percentage = float(churn_probabilities[0][1]) * 100

#         prediction_result = (
#             "⚠️ Customer is likely to churn." if raw_prediction[0] == 1
#             else "✅ Customer is likely to stay."
#         )

        
#         log_entry = {
#             "sector": sector,
#             "input_data": input_data,
#             "prediction_result": prediction_result,
#             "churn_probability": round(churn_percentage, 2)
#         }
#         predictions_collection.insert_one(log_entry)
#         logger.info(" Prediction saved to MongoDB ")

#         return jsonify({
#             "prediction": prediction_result,
#             "churn_probability": round(churn_percentage, 2)
#         })

#     except Exception as e:
#         logger.error(f" Error in prediction: {str(e)}", exc_info=True)
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)



# from flask import Flask, request, jsonify, send_file
# import joblib
# import pandas as pd
# import logging
# import numpy as np
# import xgboost
# import os
# import sys
# from pymongo import MongoClient  
# from flask_cors import CORS  
# from io import BytesIO

# print("🚀 Flask app.py started...")

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:5173", "https://churnfrontend.vercel.app"], supports_credentials=True)

# # ------------------ Logger ------------------ #
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------ DB Connection ------------------ #
# try:
#     client = MongoClient("mongodb://localhost:27017/")  # update if using Atlas
#     db = client["ChurnDB"]
#     predictions_collection = db["Predictions"]
#     logger.info("✅ Connected to MongoDB")
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

#         # Read CSV into DataFrame
#         df = pd.read_csv(file)
#         logger.info(f"📊 Uploaded CSV shape: {df.shape}")

#         # Map categorical columns if present
#         if 'GameGenre' in df.columns:
#             df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
#         if 'GameDifficulty' in df.columns:
#             df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
#         if 'EngagementLevel' in df.columns:
#             df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)

#         # Ensure all required features exist
#         for col in full_features:
#             if col not in df.columns:
#                 df[col] = 0.0   # fill missing columns with 0.0

#         df = df[full_features]  # enforce column order

#         # Scale input
#         scaled_input = scaler.transform(df.to_numpy())

#         # Predictions
#         raw_predictions = model.predict(scaled_input)
#         churn_probabilities = model.predict_proba(scaled_input)

#         # Prepare result DataFrame
#         results = df.copy()
#         results['Prediction'] = ["Churn" if p == 1 else "Stay" for p in raw_predictions]
#         results['Churn_Probability'] = churn_probabilities[:, 1] * 100

#         # Save predictions in DB (if Mongo available)
#         if predictions_collection is not None:
#             for i, row in results.iterrows():
#                 log_entry = {
#             "input_data": df.iloc[i].to_dict(),
#             "prediction_result": row['Prediction'],
#             "churn_probability": round(row['Churn_Probability'], 2)
#         }
#         predictions_collection.insert_one(log_entry)


#         # Return CSV file
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


# # ------------------ Run Server ------------------ #
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

print("🚀 Flask app.py started...")

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://churnfrontend.vercel.app"], supports_credentials=True)

# ------------------ Logger ------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ DB Connection ------------------ #
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ChurnDB"]
    predictions_collection = db["Predictions"]
    logger.info("✅ Connected to MongoDB")
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
contract_map = {"Monthly": 1, "Yearly": 2}  # <-- Encode Contract

full_features = [
    "CreditScore", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited", "GameGenre", "GameDifficulty", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
    "EngagementLevel", "Subscription_Length_Months", "Monthly_Bill",
    "Contract", "MonthlyCharges", "TotalCharges", "tenure"
]

# ------------------ Batch Prediction Route ------------------ #
@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        logger.info("🔥 Received batch CSV prediction request")

        if 'file' not in request.files:
            return jsonify({"error": "CSV file not provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        df = pd.read_csv(file)
        logger.info(f"📊 Uploaded CSV shape: {df.shape}")

        # Ensure CustomerID exists
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = range(1, len(df) + 1)

        # Map categorical columns
        if 'GameGenre' in df.columns:
            df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
        if 'GameDifficulty' in df.columns:
            df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
        if 'EngagementLevel' in df.columns:
            df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)
        if 'Contract' in df.columns:
            df['Contract'] = df['Contract'].map(contract_map).fillna(0.0)

        # Ensure all features exist
        for col in full_features:
            if col not in df.columns:
                df[col] = 0.0

        df_ordered = df[full_features]

        # Scale input
        scaled_input = scaler.transform(df_ordered.to_numpy())

        # Predictions
        raw_predictions = model.predict(scaled_input)
        churn_probabilities = model.predict_proba(scaled_input)

        # Prepare result DataFrame
        results = df.copy()
        results['Prediction'] = ["Churn" if p == 1 else "Stay" for p in raw_predictions]
        results['Churn_Probability'] = churn_probabilities[:, 1] * 100

        # Save predictions to DB
        if predictions_collection is not None:
            for i, row in results.iterrows():
                log_entry = {
                    "CustomerID": row['CustomerID'],
                    "input_data": df_ordered.iloc[i].to_dict(),
                    "prediction_result": row['Prediction'],
                    "churn_probability": round(row['Churn_Probability'], 2)
                }
                predictions_collection.insert_one(log_entry)

        # Save CSV locally
        output_file_path = "output_predictions.csv"
        if os.path.exists(output_file_path):
            results.to_csv(output_file_path, mode='a', index=False, header=False)
        else:
            results.to_csv(output_file_path, index=False)

        # Return CSV to client
        output = BytesIO()
        results.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='churn_predictions.csv'
        )

    except Exception as e:
        logger.error(f"❌ Error in batch prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔥 Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
