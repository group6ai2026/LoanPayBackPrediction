from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load("models/random_forest_loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Debug: Print expected features
if hasattr(model, 'feature_names_in_'):
    print(f"Model expects {len(model.feature_names_in_)} features:")
    print(list(model.feature_names_in_))
else:
    print("Model does not have feature_names_in_ attribute")

# More introspection for debugging
print("Model type:", type(model))
try:
    model_repr = repr(model)
    print("Model repr (truncated):", model_repr[:1000])
except Exception:
    pass
if hasattr(model, 'named_steps'):
    print("Model has named_steps (pipeline). Steps:", list(model.named_steps.keys()))
if hasattr(model, 'steps'):
    print("Model steps:", [s[0] for s in model.steps])
if hasattr(model, 'get_feature_names_out'):
    try:
        print("get_feature_names_out sample:", model.get_feature_names_out()[:20])
    except Exception as e:
        print("get_feature_names_out error:", e)

categorical_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

numerical_cols = [
    "age",
    "annual_income",
    "monthly_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "installment",
    "num_of_open_accounts",
    "total_credit_limit",
    "current_balance",
    "delinquency_history",
    "public_records",
    "num_of_delinquencies"
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        # If the saved model is a Pipeline that includes preprocessing (OneHotEncoder/Scaler), pass
        # the raw DataFrame (categoricals as strings, numericals numeric) and let the pipeline handle it.
        try:
            from sklearn.pipeline import Pipeline
        except Exception:
            Pipeline = None

        is_pipeline = False
        if Pipeline is not None and isinstance(model, Pipeline):
            is_pipeline = True
        elif hasattr(model, 'named_steps'):
            # duck-typing for pipeline-like objects
            is_pipeline = True

        if is_pipeline:
            # Ensure required columns present
            feature_order = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else (categorical_cols + numerical_cols)
            missing = [f for f in feature_order if f not in df.columns]
            if missing:
                return jsonify({"error": f"Missing input fields for model pipeline: {missing}"}), 400

            # Cast numericals
            for num in numerical_cols:
                if num in df.columns:
                    try:
                        df[num] = pd.to_numeric(df[num])
                    except Exception as e:
                        return jsonify({"error": f"Numeric conversion failed for '{num}': {e}"}), 400

            # Reorder columns and predict using pipeline (it will preprocess internally)
            df_prepared = df[feature_order]
            prediction = model.predict(df_prepared)[0]
            probability = model.predict_proba(df_prepared)[0][1]
        else:
            # Fall back to manual preprocessing using saved label_encoders and scaler
            for col in categorical_cols:
                if col not in df.columns:
                    return jsonify({"error": f"Missing categorical field: {col}"}), 400
                if col not in label_encoders:
                    return jsonify({"error": f"No encoder available for column '{col}'"}), 500
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except Exception as e:
                    allowed = getattr(label_encoders[col], 'classes_', None)
                    return jsonify({"error": f"Invalid value for '{col}': {df[col].iloc[0]}. Allowed: {list(allowed) if allowed is not None else 'unknown'}. {str(e)}"}), 400

            # Ensure numerical columns are present and numeric
            for num in numerical_cols:
                if num not in df.columns:
                    return jsonify({"error": f"Missing numerical field: {num}"}), 400
                try:
                    df[num] = pd.to_numeric(df[num])
                except Exception as e:
                    return jsonify({"error": f"Numeric conversion failed for '{num}': {e}"}), 400

            # Scale features using the same columns the scaler was fit on
            try:
                # If the scaler was fit with feature names (sklearn >= 1.0), use them directly.
                if hasattr(scaler, "feature_names_in_"):
                    scaler_features = list(scaler.feature_names_in_)
                else:
                    # Fallback: assume scaler was trained on all categorical + numerical columns
                    scaler_features = categorical_cols + numerical_cols

                # Ensure all expected scaler features are present
                missing_scaler_feats = [f for f in scaler_features if f not in df.columns]
                if missing_scaler_feats:
                    return jsonify({
                        "error": f"Error applying scaler: missing features for scaler: {missing_scaler_feats}"
                    }), 500

                df[scaler_features] = scaler.transform(df[scaler_features])
            except Exception as e:
                return jsonify({"error": f"Error applying scaler: {e}"}), 500

            # Preserve training feature order (match the model's expectation if available,
            # otherwise align to the scaler's feature order)
            if hasattr(model, 'feature_names_in_'):
                feature_order = list(model.feature_names_in_)
            else:
                feature_order = scaler_features
            missing = [f for f in feature_order if f not in df.columns]
            if missing:
                return jsonify({"error": f"Missing features after preprocessing: {missing}"}), 500

            df_prepared = df[feature_order]

            prediction = model.predict(df_prepared)[0]
            probability = model.predict_proba(df_prepared)[0][1]

        # Apply 70% threshold
        threshold = 0.70
        final_prediction = "Loan Paid Back" if probability >= threshold else "Loan Not Paid Back"

        return jsonify({
            "prediction": final_prediction,
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
