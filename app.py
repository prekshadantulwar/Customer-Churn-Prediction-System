from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import numpy as np
import xgboost
import os
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


full_features = [
    'CreditScore', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Exited', 'GameGenre', 'GameDifficulty', 'SessionsPerWeek',
    'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
    'EngagementLevel', 'Subscription_Length_Months', 'Monthly_Bill',
    'Contract', 'MonthlyCharges', 'TotalCharges', 'tenure'
]


genre_map = {"Action": 1.0, "Strategy": 2.0, "Puzzle": 3.0}
difficulty_map = {"Easy": 1.0, "Medium": 2.0, "Hard": 3.0}
engagement_map = {"low": 1.0, "medium": 2.0, "high": 3.0}

logger.info(f"XGBoost version: {xgboost.__version__}")
logger.info(f"Joblib version: {joblib.__version__}")


MODEL_PATH = os.path.join('Models', 'model.pkl')
SCALER_PATH = os.path.join('Models', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    logger.info("Model and scaler loaded successfully")

    test_input = np.zeros((1, len(full_features)))
    test_pred = model.predict(test_input)
    logger.info(f"Model verification test prediction: {test_pred}")

except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}", exc_info=True)
    raise RuntimeError("Failed to load model or scaler") from e


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("🔥🔥🔥 Flask server received a request 🔥🔥🔥")

        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")

        sector = data.get("sector")
        if not sector:
            return jsonify({"error": "Sector not specified"}), 400

        logger.info(f"Sector identified: {sector}")

        if sector == "Banking":
            input_data = {
                'CreditScore': float(data.get('CreditScore', 0)),
                'Balance': float(data.get('Balance', 0)),
                'NumOfProducts': float(data.get('NumOfProducts', 0)),
                'HasCrCard': float(data.get('HasCrCard', 0)),
                'IsActiveMember': float(data.get('IsActiveMember', 0)),
                'EstimatedSalary': float(data.get('EstimatedSalary', 0)),
                'Exited': float(data.get('Exited', 0)),
                'GameGenre': 0.0, 'GameDifficulty': 0.0, 'SessionsPerWeek': 0.0,
                'AvgSessionDurationMinutes': 0.0, 'PlayerLevel': 0.0,
                'AchievementsUnlocked': 0.0, 'EngagementLevel': 0.0,
                'Subscription_Length_Months': 0.0, 'Monthly_Bill': 0.0,
                'Contract': 0.0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0, 'tenure': 0.0
            }
        elif sector == "Gaming":
            input_data = {
                'CreditScore': 0.0, 'Balance': 0.0, 'NumOfProducts': 0.0,
                'HasCrCard': 0.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 0.0,
                'Exited': 0.0,
                'GameGenre': genre_map.get(data.get('GameGenre', ''), 0.0),
                'GameDifficulty': difficulty_map.get(data.get('GameDifficulty', ''), 0.0),
                'SessionsPerWeek': float(data.get('SessionsPerWeek', 0)),
                'AvgSessionDurationMinutes': float(data.get('AvgSessionDurationMinutes', 0)),
                'PlayerLevel': float(data.get('PlayerLevel', 0)),
                'AchievementsUnlocked': float(data.get('AchievementsUnlocked', 0)),
                'EngagementLevel': engagement_map.get(data.get('EngagementLevel', ''), 0.0),
                'Subscription_Length_Months': float(data.get('Subscription_Length_Months', 0)),
                'Monthly_Bill': float(data.get('Monthly_Bill', 0)),
                'Contract': 0.0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0, 'tenure': 0.0
            }
        elif sector == "Telecom":
            input_data = {
                'CreditScore': 0.0, 'Balance': 0.0, 'NumOfProducts': 0.0,
                'HasCrCard': 0.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 0.0,
                'Exited': float(data.get('Exited', 0)),
                'GameGenre': 0.0, 'GameDifficulty': 0.0, 'SessionsPerWeek': 0.0,
                'AvgSessionDurationMinutes': 0.0, 'PlayerLevel': 0.0,
                'AchievementsUnlocked': 0.0, 'EngagementLevel': 0.0,
                'Subscription_Length_Months': 0.0, 'Monthly_Bill': 0.0,
                'Contract': float(data.get('Contract', 0)),
                'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
                'TotalCharges': float(data.get('TotalCharges', 0)),
                'tenure': float(data.get('tenure', 0))
            }
        else:
            return jsonify({"error": "Invalid sector specified"}), 400

        input_df = pd.DataFrame([input_data])
        logger.info(f"Input DataFrame:\n{input_df}")

        scaled_input = scaler.transform(input_df.to_numpy())
        logger.info(f"Scaled input shape: {scaled_input.shape}")
        logger.info(f"Scaled input values: {scaled_input}")

        raw_prediction = model.predict(scaled_input)
        churn_probabilities = model.predict_proba(scaled_input)
        churn_percentage = float(churn_probabilities[0][1]) * 100

        prediction_result = "⚠️ Customer is likely to churn." if raw_prediction[0] == 1 else "✅ Customer is likely to stay."

        return jsonify({
            "prediction": prediction_result,
            "churn_probability": round(churn_percentage, 2)
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
