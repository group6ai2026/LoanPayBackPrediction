from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
API_URL = "http://127.0.0.1:5000/predict"

# Mapping for education level variations to expected values
EDUCATION_LEVEL_MAP = {
    "masters": "Master's",
    "Masters": "Master's",
    "MASTERS": "Master's",
    "master": "Master's",
    "Master": "Master's",
    "bachelors": "Bachelor's",
    "Bachelors": "Bachelor's",
    "BACHELORS": "Bachelor's",
    "bachelor": "Bachelor's",
    "Bachelor": "Bachelor's",
    "high school": "High School",
    "High School": "High School",
    "HIGH SCHOOL": "High School",
    "phd": "PhD",
    "PhD": "PhD",
    "PHD": "PhD",
    "other": "Other",
    "Other": "Other",
    "OTHER": "Other"
}

# Allowed values for grade_subgrade
ALLOWED_GRADE_SUBGRADES = [
    'A1', 'A2', 'A3', 'A4', 'A5',
    'B1', 'B2', 'B3', 'B4', 'B5',
    'C1', 'C2', 'C3', 'C4', 'C5',
    'D1', 'D2', 'D3', 'D4', 'D5',
    'E1', 'E2', 'E3', 'E4', 'E5',
    'F1', 'F2', 'F3', 'F4', 'F5'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Normalize education_level to match expected values
            education_level = request.form["education_level"]
            education_level_normalized = EDUCATION_LEVEL_MAP.get(education_level, education_level)
            
            # Normalize grade_subgrade (convert to uppercase and validate)
            grade_subgrade = request.form["grade_subgrade"].strip().upper()
            if grade_subgrade not in ALLOWED_GRADE_SUBGRADES:
                error = f"Invalid grade_subgrade: '{request.form['grade_subgrade']}'. Allowed values: {', '.join(ALLOWED_GRADE_SUBGRADES)}"
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"error": error}), 400
                return render_template("index.html", error=error)
            
            payload = {
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "marital_status": request.form["marital_status"],
                "education_level": education_level_normalized,
                "employment_status": request.form["employment_status"],
                "loan_purpose": request.form["loan_purpose"],
                "grade_subgrade": grade_subgrade,
                "annual_income": float(request.form["annual_income"]),
                "monthly_income": float(request.form["monthly_income"]),
                "debt_to_income_ratio": float(request.form["debt_to_income_ratio"]),
                "credit_score": int(request.form["credit_score"]),
                "loan_amount": float(request.form["loan_amount"]),
                "interest_rate": float(request.form["interest_rate"]),
                "loan_term": int(request.form["loan_term"]),
                "installment": float(request.form["installment"]),
                "num_of_open_accounts": int(request.form["num_of_open_accounts"]),
                "total_credit_limit": float(request.form["total_credit_limit"]),
                "current_balance": float(request.form["current_balance"]),
                "delinquency_history": int(request.form["delinquency_history"]),
                "public_records": int(request.form["public_records"]),
                "num_of_delinquencies": int(request.form["num_of_delinquencies"])
            }

            try:
                print("[frontend] Sending payload to API:", payload)
                response = requests.post(API_URL, json=payload, timeout=5)
                print(f"[frontend] Received response: {response.status_code} {response.text}")
            except requests.exceptions.RequestException as e:
                error = f"Failed to reach prediction API: {e}"
            else:
                if not response.ok:
                    # Try to parse error message from API for better user feedback
                    try:
                        error_data = response.json()
                        api_error = error_data.get("error", response.text)
                        # Extract allowed values if present in error message
                        if "Allowed:" in api_error:
                            error = f"Validation Error: {api_error}"
                        else:
                            error = f"Prediction API Error: {api_error}"
                    except (ValueError, KeyError):
                        # Non-2xx response with non-JSON body
                        error = f"Prediction API returned status {response.status_code}: {response.text}"
                else:
                    try:
                        result = response.json()
                        print(f"[frontend] Parsed JSON from API: {result}")
                        prediction = result.get("prediction")
                        probability = result.get("probability")
                        # If API returned nulls or missing fields, treat as error
                        if prediction is None or probability is None:
                            error = f"Prediction API returned incomplete result: {result}"
                    except ValueError:
                        error = f"Invalid JSON response from API: {response.text}"

        except Exception as e:
            error = str(e)

    # If this is an AJAX request (fetch/XHR) return JSON so the browser can display network info
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        if error:
            return jsonify({"error": error}), 500
        return jsonify({"prediction": prediction, "probability": probability})

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability,
                           error=error)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
