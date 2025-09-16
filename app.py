


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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ChurnDB"]
    predictions_collection = db["Predictions"]
    logger.info("✅ Connected to MongoDB")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {str(e)}")
    predictions_collection = None


MODEL_PATH = os.path.join('Models', 'model.pkl')
SCALER_PATH = os.path.join('Models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

logger.info("✅ Model and Scaler loaded successfully")


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

        
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = range(1, len(df) + 1)

        if 'GameGenre' in df.columns:
            df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
        if 'GameDifficulty' in df.columns:
            df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
        if 'EngagementLevel' in df.columns:
            df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)
        if 'Contract' in df.columns:
            df['Contract'] = df['Contract'].map(contract_map).fillna(0.0)

        
        for col in full_features:
            if col not in df.columns:
                df[col] = 0.0

        df_ordered = df[full_features]

   
        scaled_input = scaler.transform(df_ordered.to_numpy())

        raw_predictions = model.predict(scaled_input)
        churn_probabilities = model.predict_proba(scaled_input)

       
        results = df.copy()
        results['Prediction'] = ["Churn" if p == 1 else "Stay" for p in raw_predictions]
        results['Churn_Probability'] = churn_probabilities[:, 1] * 100

        
        if predictions_collection is not None:
            for i, row in results.iterrows():
                log_entry = {
                    "CustomerID": row['CustomerID'],
                    "input_data": df_ordered.iloc[i].to_dict(),
                    "prediction_result": row['Prediction'],
                    "churn_probability": round(row['Churn_Probability'], 2)
                }
                predictions_collection.insert_one(log_entry)

        output_file_path = "output_predictions.csv"
        if os.path.exists(output_file_path):
            results.to_csv(output_file_path, mode='a', index=False, header=False)
        else:
            results.to_csv(output_file_path, index=False)

        
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
