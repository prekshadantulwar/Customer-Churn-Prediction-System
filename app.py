
from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import logging
import numpy as np
import os
from pymongo import MongoClient
from flask_cors import CORS
from io import BytesIO
from datetime import datetime
import uuid

print("🚀 Flask app.py started...")

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://churnfrontend.vercel.app"], supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
MONGO_URI = "mongodb+srv://virenrahangdale12_db_user:8I3cTnJKXgx5Dg9n@churncluster.ltpw5dj.mongodb.net/?appName=churnCluster"
DB_NAME = "churnDB"

def get_mongo_collection(uri=MONGO_URI, db_name=DB_NAME, coll_name="Predictions"):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[db_name]
        collection = db[coll_name]
        logger.info(f"✅ Connected to MongoDB collection: {coll_name}")
        return collection
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {str(e)}")
        return None

predictions_collection = get_mongo_collection()
users_collection = get_mongo_collection(coll_name="Users")

# Load model
MODEL_PATH = os.path.join('Models', 'model.pkl')
SCALER_PATH = os.path.join('Models', 'scaler.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info("✅ Model and Scaler loaded successfully")

# Mapping
genre_map = {"Action": 1, "Adventure": 2, "Puzzle": 3, "Strategy": 4}
difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
engagement_map = {"Low": 1, "Medium": 2, "High": 3}
contract_map = {"Monthly": 1, "Yearly": 2}
sector_map = {"Finance": 1, "Telecom": 2, "Gaming": 3}

# Model features
full_features = [
    "CreditScore", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited", "GameGenre", "GameDifficulty",
    "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel",
    "AchievementsUnlocked", "EngagementLevel", "Subscription_Length_Months",
    "Monthly_Bill", "Contract", "MonthlyCharges", "TotalCharges", "tenure",
    "Sector_encoded", "Age"
]

def generate_reason(row):
    reasons = []
    if row.get('Monthly_Bill', 0) > 100:
        reasons.append("High Monthly Bill")
    if str(row.get('EngagementLevel', '')).lower() in ['low', '1']:
        reasons.append("Low Engagement Level")
    if row.get('tenure', 0) < 12:
        reasons.append("Short Tenure")
    if row.get('IsActiveMember', 1) == 0:
        reasons.append("Inactive Member")
    if str(row.get('Contract', '')).lower() in ['monthly', '1']:
        reasons.append("Short Contract")
    if row.get('CreditScore', 800) < 600:
        reasons.append("Low Credit Score")
    if not reasons:
        reasons.append("Stable Customer Behavior")
    return " & ".join(reasons[:3])

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    global predictions_collection
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

        # Mapping
        if 'GameGenre' in df.columns:
            df['GameGenre'] = df['GameGenre'].map(genre_map).fillna(0.0)
        if 'GameDifficulty' in df.columns:
            df['GameDifficulty'] = df['GameDifficulty'].map(difficulty_map).fillna(0.0)
        if 'EngagementLevel' in df.columns:
            df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map).fillna(0.0)
        if 'Contract' in df.columns:
            df['Contract'] = df['Contract'].map(contract_map).fillna(0.0)
        if 'Sector' in df.columns:
            df['Sector_encoded'] = df['Sector'].map(sector_map).fillna(0.0)
        else:
            df['Sector_encoded'] = 0.0
            df['Sector'] = "Other"
        if 'Age' not in df.columns:
            df['Age'] = 0

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
        results['Reason_For_Prediction'] = results.apply(generate_reason, axis=1)

        if predictions_collection is None:
            predictions_collection = get_mongo_collection()

        if predictions_collection is not None:
            batch_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat() + "Z"
            for i, row in results.iterrows():
                try:
                    predictions_collection.insert_one({
                        "CustomerID": int(row['CustomerID']),
                        "prediction_result": str(row['Prediction']),
                        "churn_probability": float(round(row['Churn_Probability'], 2)),
                        "reason_for_prediction": row['Reason_For_Prediction'],
                        "Sector": str(row.get('Sector', "Other")),
                        "Age": int(row.get('Age', 0)),
                        "batch_id": batch_id,
                        "predicted_at": timestamp
                    })
                except Exception as e:
                    logger.error(f"❌ Failed to insert prediction: {str(e)}")

        output = BytesIO()
        results.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='churn_predictions.csv')

    except Exception as e:
        logger.error(f"❌ Error in batch prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/signup', methods=['POST'])
def signup():
    global users_collection
    if users_collection is None:
        users_collection = get_mongo_collection(coll_name="Users")

    data = request.json
    email = data.get("email")
    fullName = data.get("fullName")
    password = data.get("password")
    contractType = data.get("contractType")

    if not all([email, password, fullName]):
        return jsonify({"error": "Missing required fields"}), 400

    if users_collection.find_one({"email": email}) is not None:
        return jsonify({"error": "Email already registered"}), 400

    users_collection.insert_one({
        "fullName": fullName,
        "email": email,
        "password": password,
        "contractType": contractType,
        "created_at": datetime.utcnow()
    })
    return jsonify({"message": "Signup successful!"}), 200


@app.route('/login', methods=['POST'])
def login():
    global users_collection
    if users_collection is None:
        users_collection = get_mongo_collection(coll_name="Users")

    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not all([email, password]):
        return jsonify({"error": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})
    if user is None or user.get("password") != password:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful!",
        "user": {
            "fullName": user.get("fullName"),
            "email": user.get("email"),
            "contractType": user.get("contractType")
        }
    }), 200


# ----------------------------
# Most Common Churn Reasons Endpoint
# ----------------------------
@app.route('/churn-reasons', methods=['GET'])
def churn_reasons():
    global predictions_collection
    if predictions_collection is None:
        predictions_collection = get_mongo_collection()

    try:
        pipeline = [
            {"$group": {"_id": "$reason_for_prediction", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        results = list(predictions_collection.aggregate(pipeline))

        if not results:
            return jsonify({"message": "No churn reasons available", "data": []}), 200

        reasons_data = [
            {"reason": r["_id"], "count": r["count"]}
            for r in results if r["_id"] is not None
        ]

        return jsonify({"data": reasons_data}), 200

    except Exception as e:
        logger.error(f"❌ Error fetching churn reasons: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch churn reasons"}), 500


if __name__ == "__main__":
    print("🔥 Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
